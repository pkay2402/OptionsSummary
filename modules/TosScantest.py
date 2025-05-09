import streamlit as st
import imaplib
import email
import re
import datetime
import pandas as pd
from dateutil import parser
import yfinance as yf
import time
from bs4 import BeautifulSoup
from functools import lru_cache
import logging
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
DAILY_KEYWORDS = ["rising5sma", "falling5sma", "demark13_buy", "demark13_sell", "HighVolumeSymbols", 
                  "Long_IT_volume", "Short_IT_volume", "LSMHG_Long", "LSMHG_Short", "StockReversalLong", 
                  "StockReversalShort"]
OPTION_KEYWORDS = ["ETF_options", "UOP_Call", "call_swing", "put_swing"]

# Keyword definitions
KEYWORD_DEFINITIONS = {
    "orb_bull": {"description": "10 mins 9 ema crossed above opening range high of 30mins", "risk_level": "high", "timeframe": "Intraday", "suggested_stop": "Below the ORB high"},
    "orb_bear": {"description": "10 mins 9 ema crossed below opening range low of 30mins", "risk_level": "high", "timeframe": "Intraday", "suggested_stop": "Above the ORB low"},
    # Add other definitions here
}

# Signal classification for bullish/bearish
BULLISH_SIGNALS = ["orb_bull", "A+Bull_30m", "tmo_long", "rising5sma", "demark13_buy", 
                   "Long_IT_volume", "LSMHG_Long", "StockReversalLong", "UOP_Call", "call_swing"]
BEARISH_SIGNALS = ["orb_bear", "tmo_Short", "falling5sma", "demark13_sell", 
                   "Short_IT_volume", "LSMHG_Short", "StockReversalShort", "put_swing"]

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
        
        # Make sure Interval is a proper datetime column
        if not pd.api.types.is_datetime64_any_dtype(grouped['Interval']):
            grouped['Interval'] = pd.to_datetime(grouped['Interval'], errors='coerce')
            
        return grouped[['Interval', 'Ticker', 'Type', 'Scans', 'Price', '% Change', 'Volume']]
    except Exception as e:
        logger.error(f"Error in aggregate_all_scans: {e}")
        return pd.DataFrame(columns=['Interval', 'Ticker', 'Type', 'Scans', 'Price', '% Change', 'Volume'])

def get_bullish_bearish_summary(summary_df):
    # Check if summary_df is empty or doesn't have the expected columns
    if summary_df.empty or 'Interval' not in summary_df.columns:
        return pd.DataFrame(), pd.DataFrame(), {}
    
    # Make sure Interval is a datetime column
    if not pd.api.types.is_datetime64_any_dtype(summary_df['Interval']):
        logger.warning("Interval column is not datetime type. Attempting to convert.")
        try:
            summary_df['Interval'] = pd.to_datetime(summary_df['Interval'], errors='coerce')
        except Exception as e:
            logger.error(f"Failed to convert Interval to datetime: {e}")
            return pd.DataFrame(), pd.DataFrame(), {}
    
    # Filter for today's data
    today = datetime.date.today()
    
    # Check for null values before accessing .dt accessor
    if summary_df['Interval'].isna().any():
        logger.warning("Some Interval values are NaT. Filtering them out.")
        summary_df = summary_df[summary_df['Interval'].notna()]
    
    # Now safely extract date component from datetime
    today_df = summary_df[summary_df['Interval'].dt.date == today]
    
    if today_df.empty:
        return pd.DataFrame(), pd.DataFrame(), {}
    
    # Classify signals as bullish or bearish
    def classify_signal(scans):
        if pd.isna(scans):
            return pd.Series({'Bullish': [], 'Bearish': []})
        signals = scans.split(', ')
        bullish = [s for s in signals if s in BULLISH_SIGNALS]
        bearish = [s for s in signals if s in BEARISH_SIGNALS]
        return pd.Series({'Bullish': bullish, 'Bearish': bearish})
    
    # Add error handling for scans column
    if 'Scans' not in today_df.columns:
        logger.error("Scans column not found in dataframe")
        return pd.DataFrame(), pd.DataFrame(), {}
    
    today_df[['Bullish', 'Bearish']] = today_df['Scans'].apply(classify_signal)
    
    # Extract bullish and bearish signals
    bullish_signals = today_df[today_df['Bullish'].apply(len) > 0][['Interval', 'Ticker', 'Type', 'Bullish']]
    bearish_signals = today_df[today_df['Bearish'].apply(len) > 0][['Interval', 'Ticker', 'Type', 'Bearish']]
    
    # Compute statistics
    stats = {
        'Total Bullish Signals': len(bullish_signals),
        'Total Bearish Signals': len(bearish_signals),
        'Unique Bullish Tickers': len(bullish_signals['Ticker'].unique()) if not bullish_signals.empty else 0,
        'Unique Bearish Tickers': len(bearish_signals['Ticker'].unique()) if not bearish_signals.empty else 0
    }
    
    # Safely compute signal type breakdowns
    if not bullish_signals.empty:
        try:
            bullish_counts = pd.Series([s for sublist in bullish_signals['Bullish'] for s in sublist]).value_counts().to_dict()
            stats['Bullish Signal Types'] = bullish_counts
        except Exception as e:
            logger.error(f"Error computing bullish signal breakdown: {e}")
            stats['Bullish Signal Types'] = {}
    else:
        stats['Bullish Signal Types'] = {}
        
    if not bearish_signals.empty:
        try:
            bearish_counts = pd.Series([s for sublist in bearish_signals['Bearish'] for s in sublist]).value_counts().to_dict()
            stats['Bearish Signal Types'] = bearish_counts
        except Exception as e:
            logger.error(f"Error computing bearish signal breakdown: {e}")
            stats['Bearish Signal Types'] = {}
    else:
        stats['Bearish Signal Types'] = {}
    
    # Format signal columns for display
    if not bullish_signals.empty:
        bullish_signals = bullish_signals.copy()
        bullish_signals['Bullish'] = bullish_signals['Bullish'].apply(lambda x: ', '.join(x))
    
    if not bearish_signals.empty:
        bearish_signals = bearish_signals.copy()
        bearish_signals['Bearish'] = bearish_signals['Bearish'].apply(lambda x: ', '.join(x))
    
    return bullish_signals, bearish_signals, stats

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
        
        # Use Streamlit's native dataframe instead of AgGrid
        st.dataframe(
            df[display_cols],
            use_container_width=True,
            height=300,
            hide_index=True,
            column_config={
                "Price": st.column_config.NumberColumn(format="$%.2f"),
                "% Change": st.column_config.NumberColumn(format="%.2f%%")
            }
        )
        
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
        days_lookback = st.slider("Days to Look Back", 1, 3, 1)
        auto_refresh = st.checkbox("Auto-refresh", value=False)
        refresh_interval = st.slider("Refresh Interval (min)", 1, 30, 10) if auto_refresh else 10
        st.markdown("---")
        st.markdown("**Disclaimer**: This tool is for informational purposes only. Trade at your own risk.")

    st.title("Thinkorswim Scan Dashboard")
    col1, col2, col3 = st.columns([2, 2, 1])
    with col3:
        if st.button("🔄 Refresh"):
            st.session_state.cached_data.clear()
            st.session_state.processed_email_ids.clear()
            st.rerun()
    
    if auto_refresh and time.time() - st.session_state.last_refresh_time >= refresh_interval * 60:
        st.session_state.cached_data.clear()
        st.session_state.processed_email_ids.clear()
        st.session_state.last_refresh_time = time.time()
        st.rerun()
    
    # Tabs for different scan types
    tabs = st.tabs(["All Scans Summary", "Lower Timeframe", "Daily", "High Conviction", "Live Options"])
    
    with tabs[0]:
        st.header("All Scans Summary")
        summary_df = aggregate_all_scans(days_lookback)
        
        # Display raw summary dataframe info for debugging if there are issues
        logger.info(f"Summary DF shape: {summary_df.shape}")
        if not summary_df.empty:
            logger.info(f"Summary DF columns: {summary_df.columns.tolist()}")
            logger.info(f"Interval column type: {summary_df['Interval'].dtype}")
        
        # Bullish and Bearish Summary
        st.subheader("Today's Bullish and Bearish Signals")
        
        try:
            bullish_signals, bearish_signals, stats = get_bullish_bearish_summary(summary_df)
            
            # Display summary statistics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Bullish Signals", stats.get('Total Bullish Signals', 0))
                st.metric("Unique Bullish Tickers", stats.get('Unique Bullish Tickers', 0))
                st.write("**Bullish Signal Breakdown**")
                for signal, count in stats.get('Bullish Signal Types', {}).items():
                    st.write(f"{signal}: {count}")
            with col2:
                st.metric("Total Bearish Signals", stats.get('Total Bearish Signals', 0))
                st.metric("Unique Bearish Tickers", stats.get('Unique Bearish Tickers', 0))
                st.write("**Bearish Signal Breakdown**")
                for signal, count in stats.get('Bearish Signal Types', {}).items():
                    st.write(f"{signal}: {count}")
            
            # Display bullish signals
            if not bullish_signals.empty:
                st.subheader("Bullish Signals")
                st.dataframe(
                    bullish_signals,
                    use_container_width=True,
                    height=300,
                    hide_index=True
                )
            
            # Display bearish signals
            if not bearish_signals.empty:
                st.subheader("Bearish Signals")
                st.dataframe(
                    bearish_signals,
                    use_container_width=True,
                    height=300,
                    hide_index=True
                )
        except Exception as e:
            st.error(f"Error analyzing bullish/bearish signals: {e}")
            logger.error(f"Error in bullish/bearish analysis: {e}", exc_info=True)
        
        # Existing interval-based summary
        if not summary_df.empty:
            st.subheader("All Signals by Interval")
            try:
                # Make sure Interval is datetime before getting unique values
                if not pd.api.types.is_datetime64_any_dtype(summary_df['Interval']):
                    summary_df['Interval'] = pd.to_datetime(summary_df['Interval'], errors='coerce')
                
                # Filter out NaT values
                summary_df = summary_df[summary_df['Interval'].notna()]
                
                if not summary_df.empty:
                    intervals = summary_df['Interval'].unique()
                    for interval in sorted(intervals, reverse=True):
                        interval_df = summary_df[summary_df['Interval'] == interval]
                        st.subheader(f"Interval: {interval.strftime('%Y-%m-%d %H:%M')}")
                        st.dataframe(
                            interval_df,
                            use_container_width=True,
                            height=300,
                            hide_index=True,
                            column_config={
                                "Price": st.column_config.NumberColumn(format="$%.2f"),
                                "% Change": st.column_config.NumberColumn(format="%.2f%%")
                            }
                        )
                
                # Download all summary data
                csv = summary_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download All Scans Summary",
                    data=csv,
                    file_name=f"all_scans_summary_{datetime.date.today()}.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"Error displaying interval data: {e}")
                logger.error(f"Error in interval summary: {e}", exc_info=True)
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
                st.dataframe(
                    high_conviction_df,
                    use_container_width=True,
                    height=300,
                    hide_index=True,
                    column_config={
                        "Price": st.column_config.NumberColumn(format="$%.2f"),
                        "% Change": st.column_config.NumberColumn(format="%.2f%%"),
                        "Volume": st.column_config.NumberColumn(format="%d")
                    }
                )
                
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

if __name__ == "__main__":
    run()
