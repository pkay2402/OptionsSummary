import streamlit as st
import imaplib
import email
import re
import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dateutil import parser
import yfinance as yf
import time
from datetime import timedelta
from bs4 import BeautifulSoup
from functools import lru_cache
import logging
from concurrent.futures import ThreadPoolExecutor
from st_aggrid import AgGrid, GridOptionsBuilder
import calendar
from collections import Counter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlit page configuration
st.set_page_config(page_title="Thinkorswim Scan Dashboard", layout="wide")

# Initialize session state
def init_session_state():
    if 'processed_email_ids' not in st.session_state:
        st.session_state.processed_email_ids = set()
    if 'last_refresh_time' not in st.session_state:
        st.session_state.last_refresh_time = time.time()
    if 'cached_data' not in st.session_state:
        st.session_state.cached_data = {}
    if 'previous_symbols' not in st.session_state:
        st.session_state.previous_symbols = {}

# Fetch credentials from Streamlit Secrets
try:
    EMAIL_ADDRESS = st.secrets["EMAIL_ADDRESS"]
    EMAIL_PASSWORD = st.secrets["EMAIL_PASSWORD"]
except KeyError:
    st.error("Email credentials not found in Streamlit Secrets.")
    st.stop()

# Constants
POLL_INTERVAL = 600  # 10 minutes in seconds
SENDER_EMAIL = "alerts@thinkorswim.com"
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

# Keyword lists
Lower_timeframe_KEYWORDS = ["orb_bull", "orb_bear", "A+Bull_30m", "tmo_long", "tmo_Short"]
DAILY_KEYWORDS = ["rising5sma", "falling5sma","demark13_buy","demark13_sell", "HighVolumeSymbols", "Long_IT_volume", "Short_IT_volume",
                  "LSMHG_Long", "LSMHG_Short", "StockReversalLong", "StockReversalShort"]
OPTION_KEYWORDS = ["ETF_options", "UOP_Call", "call_swing", "put_swing"]

# Keyword definitions (minimal example; replace with your full definitions)
KEYWORD_DEFINITIONS = {
    "orb_bull": {"description": "10 mins 9 ema crossed above opening range high of 30mins", "risk_level": "high", "timeframe": "Intraday", "suggested_stop": "Below the ORB high"},
    "orb_bear": {"description": "10 mins 9 ema crossed below opening range low of 30mins", "risk_level": "high", "timeframe": "Intraday", "suggested_stop": "Above the ORB low"},
    # Add other definitions here
}

def create_signal_calendar_data(all_signals, days_lookback=30):
    """
    Aggregate signals by date for calendar view
    """
    if not all_signals:
        return pd.DataFrame()
    
    try:
        # Combine all signals
        signal_data = pd.concat(all_signals, ignore_index=True)
        if signal_data.empty:
            return pd.DataFrame()
        
        # Ensure Date is datetime
        signal_data['Date'] = pd.to_datetime(signal_data['Date'], errors='coerce')
        if signal_data['Date'].isna().all():
            return pd.DataFrame()
        
        # Get counts by date
        signal_data['Date'] = signal_data['Date'].dt.date
        daily_counts = signal_data.groupby('Date').size().reset_index(name='Count')
        daily_counts['Date'] = pd.to_datetime(daily_counts['Date'])
        
        # Fill in missing dates
        end_date = daily_counts['Date'].max()
        start_date = end_date - timedelta(days=days_lookback)
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        full_date_df = pd.DataFrame({'Date': date_range})
        calendar_data = full_date_df.merge(daily_counts, on='Date', how='left').fillna(0)
        
        # Add day of week
        calendar_data['DayOfWeek'] = calendar_data['Date'].dt.day_name()
        calendar_data['Week'] = calendar_data['Date'].dt.isocalendar().week
        calendar_data['Month'] = calendar_data['Date'].dt.month_name()
        
        return calendar_data
    except Exception as e:
        logger.error(f"Error creating signal calendar data: {e}")
        return pd.DataFrame()

def render_signal_calendar(calendar_data):
    """
    Render a calendar heatmap of signals
    """
    if calendar_data.empty:
        st.warning("No data available for calendar view")
        return
    
    # Create a pivot table for the calendar view
    pivot_data = calendar_data.pivot_table(
        index=['Month', 'Week'], 
        columns='DayOfWeek',
        values='Count',
        aggfunc='sum'
    ).fillna(0)
    
    # Reorder days
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    pivot_data = pivot_data.reindex(columns=days_order)
    
    # Create heatmap
    fig = px.imshow(
        pivot_data, 
        color_continuous_scale='Viridis',
        labels=dict(x="Day of Week", y="Week", color="Signal Count")
    )
    
    fig.update_layout(
        height=600,
        margin=dict(l=40, r=20, t=60, b=20),
        title="Signal Calendar Heatmap"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show daily breakdown
    st.subheader("Daily Signal Breakdown")
    last_two_weeks = calendar_data.sort_values('Date', ascending=False).head(14)
    last_two_weeks = last_two_weeks.sort_values('Date')
    
    fig_bar = px.bar(
        last_two_weeks,
        x='Date',
        y='Count',
        labels={'Count': 'Signal Count', 'Date': 'Date'},
        title="Signal Count - Last 14 Days"
    )
    
    fig_bar.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_bar, use_container_width=True)

# Add these functions for Market Breadth Indicators
def calculate_market_breadth(all_signals):
    """
    Calculate market breadth indicators from signals
    """
    if not all_signals:
        return None, None, None
    
    try:
        # Combine all signals
        signal_data = pd.concat(all_signals, ignore_index=True)
        if signal_data.empty:
            return None, None, None
        
        # Categorize signals as bullish or bearish
        bullish_keywords = ["orb_bull", "A+Bull_30m", "tmo_long", "rising5sma", "demark13_buy", 
                          "HighVolumeSymbols", "Long_IT_volume", "LSMHG_Long", "StockReversalLong"]
        bearish_keywords = ["orb_bear", "tmo_Short", "falling5sma", "demark13_sell", 
                           "Short_IT_volume", "LSMHG_Short", "StockReversalShort"]
        
        # Create signal type column
        def categorize_signal(signal):
            signal_lower = signal.lower()
            if any(keyword.lower() in signal_lower for keyword in bullish_keywords):
                return "Bullish"
            elif any(keyword.lower() in signal_lower for keyword in bearish_keywords):
                return "Bearish"
            else:
                return "Neutral"
        
        signal_data['SignalType'] = signal_data['Signal'].apply(categorize_signal)
        
        # Ensure Date is datetime and calculate daily metrics
        signal_data['Date'] = pd.to_datetime(signal_data['Date'], errors='coerce')
        if signal_data['Date'].isna().all():
            return None, None, None
        
        signal_data['Date'] = signal_data['Date'].dt.date
        
        # Daily count by signal type
        daily_sentiment = signal_data.groupby(['Date', 'SignalType']).size().unstack(fill_value=0)
        
        if 'Bullish' not in daily_sentiment.columns:
            daily_sentiment['Bullish'] = 0
        if 'Bearish' not in daily_sentiment.columns:
            daily_sentiment['Bearish'] = 0
        if 'Neutral' not in daily_sentiment.columns:
            daily_sentiment['Neutral'] = 0
            
        # Calculate breadth indicators
        daily_sentiment['Total'] = daily_sentiment.sum(axis=1)
        daily_sentiment['Bull-Bear Ratio'] = daily_sentiment['Bullish'] / (daily_sentiment['Bearish'] + 0.0001)
        daily_sentiment['Bullish %'] = (daily_sentiment['Bullish'] / daily_sentiment['Total']) * 100
        daily_sentiment['Bearish %'] = (daily_sentiment['Bearish'] / daily_sentiment['Total']) * 100
        daily_sentiment['Net Bull-Bear'] = daily_sentiment['Bullish'] - daily_sentiment['Bearish']
        
        # Calculate most active tickers
        ticker_signal_counts = signal_data.groupby('Ticker').size().reset_index(name='Signal Count')
        top_tickers = ticker_signal_counts.sort_values('Signal Count', ascending=False).head(10)
        
        # Calculate signal distribution by type
        signal_distribution = signal_data['SignalType'].value_counts().reset_index()
        signal_distribution.columns = ['Signal Type', 'Count']
        
        return daily_sentiment.reset_index(), top_tickers, signal_distribution
    except Exception as e:
        logger.error(f"Error calculating market breadth: {e}")
        return None, None, None

def render_market_breadth(daily_sentiment, top_tickers, signal_distribution):
    """
    Render market breadth indicators
    """
    if daily_sentiment is None or daily_sentiment.empty:
        st.warning("No data available for market breadth analysis")
        return
    
    # Sort by date
    daily_sentiment['Date'] = pd.to_datetime(daily_sentiment['Date'])
    daily_sentiment = daily_sentiment.sort_values('Date')
    
    # Display current breadth metrics
    latest_date = daily_sentiment['Date'].max()
    latest_metrics = daily_sentiment[daily_sentiment['Date'] == latest_date].iloc[0]
    
    # Metrics display
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        bullish_percent = latest_metrics.get('Bullish %', 0)
        st.metric("Bullish %", f"{bullish_percent:.1f}%")
    with col2:
        bearish_percent = latest_metrics.get('Bearish %', 0)
        st.metric("Bearish %", f"{bearish_percent:.1f}%")
    with col3:
        bull_bear_ratio = latest_metrics.get('Bull-Bear Ratio', 0)
        st.metric("Bull/Bear Ratio", f"{bull_bear_ratio:.2f}")
    with col4:
        net_bull_bear = latest_metrics.get('Net Bull-Bear', 0)
        st.metric("Net Bull-Bear", f"{net_bull_bear:.0f}")
    
    # Bull-Bear Ratio Over Time
    st.subheader("Bull-Bear Ratio Trend")
    fig_ratio = px.line(
        daily_sentiment,
        x='Date',
        y='Bull-Bear Ratio',
        title="Bull/Bear Ratio - Last 30 Days"
    )
    fig_ratio.add_hline(y=1.0, line_dash="dash", line_color="gray", 
                      annotation_text="Neutral Line", annotation_position="top right")
    fig_ratio.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_ratio, use_container_width=True)
    
    # Signal Type Distribution
    st.subheader("Signal Type Distribution")
    if signal_distribution is not None and not signal_distribution.empty:
        fig_pie = px.pie(
            signal_distribution, 
            values='Count', 
            names='Signal Type',
            title="Distribution of Signal Types",
            color='Signal Type',
            color_discrete_map={'Bullish': 'green', 'Bearish': 'red', 'Neutral': 'gray'}
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Top Active Tickers
    st.subheader("Most Active Tickers")
    if top_tickers is not None and not top_tickers.empty:
        fig_bar = px.bar(
            top_tickers,
            y='Ticker',
            x='Signal Count',
            orientation='h',
            title="Top 10 Most Active Tickers",
        )
        fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Bull-Bear Net Chart
    st.subheader("Bull-Bear Net Trend")
    fig_net = px.area(
        daily_sentiment,
        x='Date',
        y='Net Bull-Bear',
        title="Net Bull-Bear Signals - Last 30 Days",
        color_discrete_sequence=['green' if daily_sentiment['Net Bull-Bear'].mean() > 0 else 'red']
    )
    fig_net.add_hline(y=0, line_dash="dash", line_color="black")
    fig_net.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_net, use_container_width=True)

def connect_to_email(retries=MAX_RETRIES):
    for attempt in range(retries):
        try:
            mail = imaplib.IMAP4_SSL('imap.gmail.com')
            mail.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            return mail
        except Exception as e:
            if attempt == retries - 1:
                raise
            logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
            time.sleep(RETRY_DELAY)

def parse_email_body(msg):
    try:
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() in ["text/plain", "text/html"]:
                    body = part.get_payload(decode=True).decode()
                    if part.get_content_type() == "text/html":
                        soup = BeautifulSoup(body, "html.parser")
                        return soup.get_text(separator=' ', strip=True)
                    return body
        else:
            body = msg.get_payload(decode=True).decode()
            if msg.get_content_type() == "text/html":
                soup = BeautifulSoup(body, "html.parser")
                return soup.get_text(separator=' ', strip=True)
            return body
    except Exception as e:
        logger.error(f"Error parsing email body: {e}")
        return ""

def extract_stock_symbols_from_email(email_address, password, sender_email, keyword, days_lookback):
    if keyword in st.session_state.cached_data:
        return st.session_state.cached_data[keyword]
    
    try:
        mail = connect_to_email()
        mail.select('inbox')
        today = datetime.date.today()
        start_date = today - datetime.timedelta(days=days_lookback-1) if days_lookback > 1 else today
        date_since = start_date.strftime("%d-%b-%Y")
        search_criteria = f'(FROM "{sender_email}" SUBJECT "{keyword}" SINCE "{date_since}")'
        _, data = mail.search(None, search_criteria)
        stock_data = []
        
        for num in data[0].split():
            if num in st.session_state.processed_email_ids:
                continue
            _, data = mail.fetch(num, '(RFC822)')
            msg = email.message_from_bytes(data[0][1])
            email_datetime = parser.parse(msg['Date'])
            email_date = email_datetime.date()
            if email_date < start_date or email_datetime.weekday() >= 5:
                continue
            body = parse_email_body(msg)
            symbols = re.findall(r'New symbols:\s*([A-Z,\s]+)\s*were added to\s*(' + re.escape(keyword) + ')', body)
            if symbols:
                for symbol_group in symbols:
                    extracted_symbols = symbol_group[0].replace(" ", "").split(",")
                    signal_type = symbol_group[1]
                    for symbol in extracted_symbols:
                        if symbol.isalpha():
                            stock_data.append([symbol, email_datetime, signal_type])
            st.session_state.processed_email_ids.add(num)
        
        mail.close()
        mail.logout()
        if stock_data:
            df = pd.DataFrame(stock_data, columns=['Ticker', 'Date', 'Signal'])
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values(by=['Date', 'Ticker']).drop_duplicates(subset=['Ticker', 'Signal', 'Date'], keep='last')
            st.session_state.cached_data[keyword] = df
            return df
        empty_df = pd.DataFrame(columns=['Ticker', 'Date', 'Signal'])
        st.session_state.cached_data[keyword] = empty_df
        return empty_df
    except Exception as e:
        logger.error(f"Error in extract_stock_symbols_from_email: {e}")
        return pd.DataFrame(columns=['Ticker', 'Date', 'Signal'])

def extract_option_symbols_from_email(email_address, password, sender_email, keyword, days_lookback):
    if keyword in st.session_state.cached_data:
        return st.session_state.cached_data[keyword]
    
    try:
        mail = connect_to_email()
        mail.select('inbox')
        today = datetime.date.today()
        start_date = today - datetime.timedelta(days=days_lookback-1) if days_lookback > 1 else today
        date_since = start_date.strftime("%d-%b-%Y")
        search_criteria = f'(FROM "{sender_email}" SUBJECT "{keyword}" SINCE "{date_since}")'
        _, data = mail.search(None, search_criteria)
        option_data = []
        
        for num in data[0].split():
            if num in st.session_state.processed_email_ids:
                continue
            _, data = mail.fetch(num, '(RFC822)')
            msg = email.message_from_bytes(data[0][1])
            email_datetime = parser.parse(msg['Date'])
            email_date = email_datetime.date()
            if email_date < start_date or email_datetime.weekday() >= 5:
                continue
            body = parse_email_body(msg)
            symbols = re.findall(r'New symbols:\s*([\.\w,\s]+)\s*were added to\s*(' + re.escape(keyword) + ')', body)
            if symbols:
                for symbol_group in symbols:
                    extracted_symbols = symbol_group[0].replace(" ", "").split(",")
                    signal_type = symbol_group[1]
                    for symbol in extracted_symbols:
                        if symbol:
                            readable_symbol = parse_option_symbol(symbol)
                            option_data.append([symbol, readable_symbol, email_datetime, signal_type])
            st.session_state.processed_email_ids.add(num)
        
        mail.close()
        mail.logout()
        if option_data:
            df = pd.DataFrame(option_data, columns=['Raw_Symbol', 'Readable_Symbol', 'Date', 'Signal'])
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values(by=['Date', 'Raw_Symbol']).drop_duplicates(subset=['Raw_Symbol', 'Signal', 'Date'], keep='last')
            st.session_state.cached_data[keyword] = df
            return df
        empty_df = pd.DataFrame(columns=['Raw_Symbol', 'Readable_Symbol', 'Date', 'Signal'])
        st.session_state.cached_data[keyword] = empty_df
        return empty_df
    except Exception as e:
        logger.error(f"Error in extract_option_symbols_from_email: {e}")
        return pd.DataFrame(columns=['Raw_Symbol', 'Readable_Symbol', 'Date', 'Signal'])

def parse_option_symbol(option_symbol):
    try:
        symbol = option_symbol.lstrip('.')
        pattern = r'([A-Z]+)(\d{2})(\d{2})(\d{2})([CP])([\d_]+)'
        match = re.match(pattern, symbol)
        if match:
            Ticker, year, month, day, opt_type, strike = match.groups()
            month_name = datetime.datetime.strptime(month, '%m').strftime('%b').upper()
            strike = strike.replace('_', '.')
            option_type = 'CALL' if opt_type == 'C' else 'PUT'
            return f"{Ticker} {day} {month_name} 20{year} {strike} {option_type}"
    except Exception as e:
        logger.error(f"Error parsing option symbol {option_symbol}: {e}")
    return option_symbol

def get_new_symbols_count(keyword, current_df):
    if current_df.empty:
        return 0
    ticker_col = next((col for col in ['Ticker', 'Raw_Symbol', 'Readable_Symbol'] if col in current_df.columns), None)
    if ticker_col is None:
        return 0
    current_symbols = set(current_df[ticker_col].unique())
    previous_symbols = st.session_state.previous_symbols.get(keyword, set())
    new_symbols = current_symbols - previous_symbols
    st.session_state.previous_symbols[keyword] = current_symbols
    return len(new_symbols)

def high_conviction_stocks(dataframes, ignore_keywords=None):
    if ignore_keywords is None:
        ignore_keywords = []
    filtered_dataframes = [df[~df['Signal'].isin(ignore_keywords)] for df in dataframes if not df.empty]
    if not filtered_dataframes:
        return pd.DataFrame(columns=['Date', 'Ticker', 'Signal'])
    
    try:
        all_data = pd.concat(filtered_dataframes, ignore_index=True)
        if all_data.empty or 'Date' not in all_data.columns:
            return pd.DataFrame(columns=['Date', 'Ticker', 'Signal'])
        
        all_data['Date'] = pd.to_datetime(all_data['Date'], errors='coerce')
        if all_data['Date'].isna().all():
            logger.warning("All 'Date' values are invalid or missing.")
            return pd.DataFrame(columns=['Date', 'Ticker', 'Signal'])
        
        all_data['Date'] = all_data['Date'].dt.date
        grouped = all_data.groupby(['Date', 'Ticker'])['Signal'].agg(lambda x: ', '.join(set(x))).reset_index()
        grouped['Count'] = grouped['Signal'].apply(lambda x: len(x.split(', ')))
        return grouped[grouped['Count'] >= 2][['Date', 'Ticker', 'Signal']]
    except Exception as e:
        logger.error(f"Error in high_conviction_stocks: {e}")
        return pd.DataFrame(columns=['Date', 'Ticker', 'Signal'])

def fetch_stock_metrics(tickers):
    metrics = []
    try:
        for ticker in tickers:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="2d")
            if not hist.empty:
                price = round(hist['Close'].iloc[-1], 2)
                prev_close = hist['Close'].iloc[-2]
                change = round(((price - prev_close) / prev_close) * 100, 2)
                volume = int(hist['Volume'].iloc[-1])
                metrics.append([ticker, price, change, volume])
    except Exception as e:
        logger.error(f"Error fetching stock metrics: {e}")
    return pd.DataFrame(metrics, columns=['Ticker', 'Price', '% Change', 'Volume'])

def aggregate_all_scans(days_lookback):
    all_data = []
    
    # Collect stock scans (Lower Timeframe and Daily)
    for keyword in Lower_timeframe_KEYWORDS + DAILY_KEYWORDS:
        df = extract_stock_symbols_from_email(EMAIL_ADDRESS, EMAIL_PASSWORD, SENDER_EMAIL, keyword, days_lookback)
        if not df.empty:
            df = df[['Ticker', 'Date', 'Signal']].copy()
            df['Type'] = 'Stock'
            all_data.append(df)
    
    # Collect option scans
    for keyword in OPTION_KEYWORDS:
        df = extract_option_symbols_from_email(EMAIL_ADDRESS, EMAIL_PASSWORD, SENDER_EMAIL, keyword, days_lookback)
        if not df.empty:
            df = df[['Readable_Symbol', 'Date', 'Signal']].copy()
            df = df.rename(columns={'Readable_Symbol': 'Ticker'})
            df['Type'] = 'Option'
            all_data.append(df)
    
    if not all_data:
        return pd.DataFrame(columns=['Interval', 'Ticker', 'Type', 'Scans', 'Price', '% Change', 'Volume'])
    
    try:
        combined = pd.concat(all_data, ignore_index=True)
        combined['Date'] = pd.to_datetime(combined['Date'], errors='coerce')
        if combined['Date'].isna().all():
            logger.warning("All 'Date' values in aggregated data are invalid.")
            return pd.DataFrame(columns=['Interval', 'Ticker', 'Type', 'Scans', 'Price', '% Change', 'Volume'])
        
        # Group by Ticker and 30-minute intervals
        combined['Interval'] = combined['Date'].dt.floor('30min')
        grouped = combined.groupby(['Interval', 'Ticker', 'Type'])['Signal'].agg(lambda x: ', '.join(set(x))).reset_index()
        grouped = grouped.rename(columns={'Signal': 'Scans'})
        
        # Fetch stock metrics for stock tickers
        stock_tickers = grouped[grouped['Type'] == 'Stock']['Ticker'].unique()
        if stock_tickers.size > 0:
            metrics_df = fetch_stock_metrics(stock_tickers)
            grouped = grouped.merge(metrics_df, on='Ticker', how='left')
        
        # Fill NaN for options (no metrics)
        grouped[['Price', '% Change', 'Volume']] = grouped[['Price', '% Change', 'Volume']].fillna('N/A')
        
        # Sort by Interval (descending) and Ticker
        grouped = grouped.sort_values(by=['Interval', 'Ticker'], ascending=[False, True])
        return grouped[['Interval', 'Ticker', 'Type', 'Scans', 'Price', '% Change', 'Volume']]
    except Exception as e:
        logger.error(f"Error in aggregate_all_scans: {e}")
        return pd.DataFrame(columns=['Interval', 'Ticker', 'Type', 'Scans', 'Price', '% Change', 'Volume'])

def render_dashboard_section(keyword, days_lookback, is_option=False):
    if is_option:
        df = extract_option_symbols_from_email(EMAIL_ADDRESS, EMAIL_PASSWORD, SENDER_EMAIL, keyword, days_lookback)
        display_cols = ['Readable_Symbol', 'Date', 'Signal']
        ticker_col = 'Readable_Symbol'
    else:
        df = extract_stock_symbols_from_email(EMAIL_ADDRESS, EMAIL_PASSWORD, SENDER_EMAIL, keyword, days_lookback)
        display_cols = ['Ticker', 'Date', 'Signal']
        ticker_col = 'Ticker'
    
    new_count = get_new_symbols_count(keyword, df)
    header = f"{keyword} {'🔴 ' + str(new_count) + ' new' if new_count > 0 else ''}"
    
    st.subheader(header)
    info = KEYWORD_DEFINITIONS.get(keyword, {})
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.info(f"Desc: {info.get('description', 'N/A')}")
    with col2:
        st.info(f"Risk: {info.get('risk_level', 'N/A')}")
    with col3:
        st.info(f"Timeframe: {info.get('timeframe', 'N/A')}")
    with col4:
        st.info(f"Stop: {info.get('suggested_stop', 'N/A')}")
    
    if not df.empty:
        df['Date'] = df['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')
        if not is_option:
            tickers = df['Ticker'].unique()
            metrics_df = fetch_stock_metrics(tickers)
            df = df.merge(metrics_df, on='Ticker', how='left')
            display_cols.extend(['Price', '% Change', 'Volume'])
        
        gb = GridOptionsBuilder.from_dataframe(df[display_cols])
        gb.configure_pagination(paginationAutoPageSize=True)
        gb.configure_side_bar()
        grid_options = gb.build()
        AgGrid(df[display_cols], grid_options=grid_options, height=300, fit_columns_on_grid_load=True)
        
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=f"Download {keyword} Data",
            data=csv,
            file_name=f"{keyword}_alerts_{datetime.date.today()}.csv",
            mime="text/csv"
        )
    else:
        st.warning(f"No signals found for {keyword} in the last {days_lookback} day(s).")

def run():
    init_session_state()
    
    with st.sidebar:
        st.header("Dashboard Settings")
        days_lookback = st.slider("Days to Look Back", 1, 30, 7)
        auto_refresh = st.checkbox("Auto-refresh", value=False)
        refresh_interval = st.slider("Refresh Interval (min)", 1, 30, 10) if auto_refresh else 10
        st.markdown("---")
        st.markdown("**Disclaimer**: This tool is for informational purposes only. Trade at your own risk.")

    st.title("Thinkorswim Scan Dashboard")
    col1, col2, col3 = st.columns([2, 2, 1])
    if st.button("🔄 Refresh", use_container_width=True):
        st.session_state.cached_data.clear()
        st.session_state.processed_email_ids.clear()
        st.rerun()
    
    if auto_refresh and time.time() - st.session_state.last_refresh_time >= refresh_interval * 60:
        st.session_state.cached_data.clear()
        st.session_state.processed_email_ids.clear()
        st.session_state.last_refresh_time = time.time()
        st.rerun()
    
    # Update tabs to include new Signal Calendar and Market Breadth tabs
    tabs = st.tabs(["All Scans Summary", "Lower Timeframe", "Daily", "High Conviction", "Live Options", "Signal Calendar", "Market Breadth"])
    
    with tabs[0]:
        st.header("All Scans Summary")
        summary_df = aggregate_all_scans(days_lookback)
        if not summary_df.empty:
            # Split by 30-minute intervals
            intervals = summary_df['Interval'].unique()
            for interval in sorted(intervals, reverse=True):
                interval_df = summary_df[summary_df['Interval'] == interval]
                st.subheader(f"Interval: {interval.strftime('%Y-%m-%d %H:%M')}")
                gb = GridOptionsBuilder.from_dataframe(interval_df)
                gb.configure_pagination(paginationAutoPageSize=True)
                gb.configure_side_bar()
                grid_options = gb.build()
                AgGrid(interval_df, grid_options=grid_options, height=300, fit_columns_on_grid_load=True)
            
            # Download all summary data
            csv = summary_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download All Scans Summary",
                data=csv,
                file_name=f"all_scans_summary_{datetime.date.today()}.csv",
                mime="text/csv"
            )
        else:
            st.warning("No signals found across all scans.")
    
    with tabs[1]:
        st.header("Lower Timeframe Scans")
        for keyword in Lower_timeframe_KEYWORDS:
            render_dashboard_section(keyword, days_lookback, is_option=False)
    
    with tabs[2]:
        st.header("Daily Scans")
        for keyword in DAILY_KEYWORDS:
            render_dashboard_section(keyword, days_lookback, is_option=False)
    
    with tabs[3]:
        st.header("High Conviction Scans")
        all_signals = []
        for keyword in Lower_timeframe_KEYWORDS + DAILY_KEYWORDS:
            df = extract_stock_symbols_from_email(EMAIL_ADDRESS, EMAIL_PASSWORD, SENDER_EMAIL, keyword, days_lookback)
            if not df.empty:
                all_signals.append(df)
        
        if all_signals:
            high_conviction_df = high_conviction_stocks(all_signals, ignore_keywords=["tmo_long", "tmo_Short", "orb_bull", "orb_bear"])
            if not high_conviction_df.empty:
                metrics_df = fetch_stock_metrics(high_conviction_df['Ticker'].unique())
                high_conviction_df = high_conviction_df.merge(metrics_df, on='Ticker', how='left')
                gb = GridOptionsBuilder.from_dataframe(high_conviction_df)
                gb.configure_pagination(paginationAutoPageSize=True)
                gb.configure_side_bar()
                AgGrid(high_conviction_df, grid_options=gb.build(), height=300, fit_columns_on_grid_load=True)
                
                csv = high_conviction_df.to_csv(index=False).encode('utf-8')

                st.download_button(
                    label="Download High Conviction Data",
                    data=csv,
                    file_name=f"high_conviction_alerts_{datetime.date.today()}.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No high conviction signals found.")
        else:
            st.warning("No signals available for high conviction analysis.")
    
    with tabs[4]:
        st.header("Live Options Scans")
        for keyword in OPTION_KEYWORDS:
            render_dashboard_section(keyword, days_lookback, is_option=True)
    
    st.markdown(f"**Last Updated**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # New Signal Calendar tab
    with tabs[5]:
        st.header("Signal Calendar View")
        
        all_signals = []
        for keyword in Lower_timeframe_KEYWORDS + DAILY_KEYWORDS:
            df = extract_stock_symbols_from_email(EMAIL_ADDRESS, EMAIL_PASSWORD, SENDER_EMAIL, keyword, days_lookback)
            if not df.empty:
                all_signals.append(df)
        
        calendar_data = create_signal_calendar_data(all_signals, days_lookback)
        render_signal_calendar(calendar_data)
    
    # New Market Breadth tab
    with tabs[6]:
        st.header("Market Breadth Indicators")
        
        all_signals = []
        for keyword in Lower_timeframe_KEYWORDS + DAILY_KEYWORDS:
            df = extract_stock_symbols_from_email(EMAIL_ADDRESS, EMAIL_PASSWORD, SENDER_EMAIL, keyword, days_lookback)
            if not df.empty:
                all_signals.append(df)
        
        daily_sentiment, top_tickers, signal_distribution = calculate_market_breadth(all_signals)
        render_market_breadth(daily_sentiment, top_tickers, signal_distribution)
    
    st.markdown(f"**Last Updated**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    run()
