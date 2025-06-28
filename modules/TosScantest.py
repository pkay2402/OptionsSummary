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
import requests
from io import StringIO
from typing import List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Daily Scans Functions ---

def init_session_state():
    """Initialize session state variables"""
    if 'processed_email_ids' not in st.session_state:
        st.session_state.processed_email_ids = set()
    if 'last_refresh_time' not in st.session_state:
        st.session_state.last_refresh_time = time.time()
    if 'cached_data' not in st.session_state:
        st.session_state.cached_data = {}
    if 'previous_symbols' not in st.session_state:
        st.session_state.previous_symbols = {}

# Fetch credentials from Streamlit Secrets
EMAIL_ADDRESS = st.secrets["EMAIL_ADDRESS"]
EMAIL_PASSWORD = st.secrets["EMAIL_PASSWORD"]

# Constants
POLL_INTERVAL = 600  # 10 minutes in seconds
SENDER_EMAIL = "alerts@thinkorswim.com"
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

# Define keywords for daily scans
DAILY_KEYWORDS = [
    "HighVolumeSymbols", "Long_IT_volume", "Short_IT_volume", "demark13_buy", 
    "demark13_sell", "bull_Daily_sqz", "bear_Daily_sqz", "LSMHG_Long", 
    "LSMHG_Short", "StockReversalLong", "StockReversalShort"
]

# Keyword definitions with added risk levels and descriptions
KEYWORD_DEFINITIONS = {
    "HighVolumeSymbols": {
        "description": "On Daily TF stocks consistently trading above high volume. High volumes leads to change in trends. This can be bullish or bearish",
        "risk_level": "medium",
        "timeframe": "2 weeks",
        "suggested_stop": "Below high volume node"
    },
    "Long_IT_volume": {
        "description": "On Daily TF stocks breaking out 9ema above high volume node",
        "risk_level": "medium",
        "timeframe": "2 weeks",
        "suggested_stop": "Below high volume node"
    },
    "Short_IT_volume": {
        "description": "On Daily TF stocks breaking down 9ema below low volume node",
        "risk_level": "medium",
        "timeframe": "2 weeks",
        "suggested_stop": "Above low volume node"
    },
    "demark13_buy": {
        "description": "Daily DeMark 13 buy signal indicating potential reversal",
        "risk_level": "medium",
        "timeframe": "2 weeks",
        "suggested_stop": "Below recent low"
    },
    "demark13_sell": {
        "description": "Daily DeMark 13 sell signal indicating potential reversal",
        "risk_level": "medium",
        "timeframe": "2 weeks",
        "suggested_stop": "Above recent high"
    },
    "bull_Daily_sqz": {
        "description": "On Daily TF stocks breaking out of large squeeze",
        "risk_level": "medium",
        "timeframe": "2 weeks",
        "suggested_stop": "Below low of previous day"
    },
    "bear_Daily_sqz": {
        "description": "On Daily TF stocks breaking down of large squeeze",
        "risk_level": "medium",
        "timeframe": "2 weeks",
        "suggested_stop": "Above high of previous day"
    },
    "LSMHG_Long": {
        "description": "On Daily TF stocks being bought on 1 yr low area and macd has crossed over",
        "risk_level": "medium",
        "timeframe": "1-2 months",
        "suggested_stop": "Below low of previous day"
    },
    "LSMHG_Short": {
        "description": "On Daily TF stocks being sold on 1 yr high area and macd has crossed under",
        "risk_level": "medium",
        "timeframe": "2 weeks",
        "suggested_stop": "Above high of previous day"
    },
    "StockReversalLong": {
        "description": "On Daily TF stocks now showing signs of bull reversal",
        "risk_level": "medium",
        "timeframe": "2 weeks-1 month",
        "suggested_stop": "Below low of previous day"
    },
    "StockReversalShort": {
        "description": "On Daily TF stocks now showing signs of bear reversal",
        "risk_level": "medium",
        "timeframe": "2 weeks-1 month",
        "suggested_stop": "Above high of previous day"
    }
}

@lru_cache(maxsize=2)
def get_spy_qqq_prices():
    """Fetch the latest closing prices and daily changes for SPY and QQQ with caching."""
    try:
        spy = yf.Ticker("SPY")
        qqq = yf.Ticker("QQQ")
        
        spy_hist = spy.history(period="2d")
        qqq_hist = qqq.history(period="2d")
        
        spy_price = round(spy_hist['Close'].iloc[-1], 2)
        qqq_price = round(qqq_hist['Close'].iloc[-1], 2)
        
        spy_prev_close = spy_hist['Close'].iloc[-2]
        qqq_prev_close = qqq_hist['Close'].iloc[-2]
        
        spy_change = round(((spy_price - spy_prev_close) / spy_prev_close) * 100, 2)
        qqq_change = round(((qqq_price - qqq_prev_close) / qqq_prev_close) * 100, 2)
        
        return spy_price, qqq_price, spy_change, qqq_change
    except Exception as e:
        logger.error(f"Error fetching market prices: {e}")
        return None, None, None, None

def connect_to_email(retries=MAX_RETRIES):
    """Establish email connection with retry logic."""
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
    """Parse email body with better HTML handling."""
    try:
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() in ["text/plain", "text/html"]:
                    body = part.get_payload(decode=True).decode()
                    if part.get_content_type() == "text/html":
                        soup = BeautifulSoup(body, "html.parser", from_encoding='utf-8')
                        return soup.get_text(separator=' ', strip=True)
                    return body
        else:
            body = msg.get_payload(decode=True).decode()
            if msg.get_content_type() == "text/html":
                soup = BeautifulSoup(body, "html.parser", from_encoding='utf-8')
                return soup.get_text(separator=' ', strip=True)
            return body
    except Exception as e:
        logger.error(f"Error parsing email body: {e}")
        return ""

def extract_stock_symbols_from_email(email_address, password, sender_email, keyword, days_lookback):
    """Extract stock symbols from email alerts with proper date filtering."""
    if keyword in st.session_state.cached_data:
        return st.session_state.cached_data[keyword]

    try:
        mail = connect_to_email()
        mail.select('inbox')

        today = datetime.date.today()
        start_date = today
        if days_lookback > 1:
            start_date = today - datetime.timedelta(days=days_lookback-1)
        
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
            df = df.sort_values(by=['Date', 'Ticker']).drop_duplicates(subset=['Ticker', 'Signal', 'Date'], keep='last')
            st.session_state.cached_data[keyword] = df
            return df

        empty_df = pd.DataFrame(columns=['Ticker', 'Date', 'Signal'])
        st.session_state.cached_data[keyword] = empty_df
        return empty_df

    except Exception as e:
        logger.error(f"Error in extract_stock_symbols_from_email: {e}")
        st.error(f"Error processing emails: {str(e)}")
        return pd.DataFrame(columns=['Ticker', 'Date', 'Signal'])

def get_new_symbols_count(keyword, current_df):
    """Get count of new symbols."""
    if current_df.empty:
        return 0

    possible_ticker_columns = ['Ticker', 'ticker', 'Symbol']
    ticker_col = next((col for col in possible_ticker_columns if col in current_df.columns), None)

    if ticker_col is None:
        raise KeyError(f"No valid ticker column found in DataFrame for keyword: {keyword}")

    current_symbols = set(current_df[ticker_col].unique())
    previous_symbols = st.session_state.previous_symbols.get(keyword, set())
    
    new_symbols = current_symbols - previous_symbols
    st.session_state.previous_symbols[keyword] = current_symbols
    
    return len(new_symbols)

def render_stock_section(keyword, days_lookback):
    """Helper function to render stock section content"""
    symbols_df = extract_stock_symbols_from_email(
        EMAIL_ADDRESS, EMAIL_PASSWORD, SENDER_EMAIL, keyword, days_lookback
    )
    
    new_count = get_new_symbols_count(keyword, symbols_df)
    
    header = f"📊 {keyword}"
    if new_count > 0:
        header = f"📊 {keyword} 🔴 {new_count} new"
    
    with st.expander(header, expanded=False):
        info = KEYWORD_DEFINITIONS.get(keyword, {})
        if info:
            col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
            with col1:
                st.info(f"Desc: {info.get('description', 'N/A')}")
            with col2:
                st.info(f"Risk Level: {info.get('risk_level', 'N/A')}")
            with col3:
                st.info(f"Timeframe: {info.get('timeframe', 'N/A')}")
            with col4:
                st.info(f"Suggested Stop: {info.get('suggested_stop', 'N/A')}")
        
        if not symbols_df.empty:
            display_df = symbols_df.copy()
            display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')
            st.dataframe(display_df, use_container_width=True)
            
            csv = symbols_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=f"📥 Download {keyword} Data",
                data=csv,
                file_name=f"{keyword}_alerts_{datetime.date.today()}.csv",
                mime="text/csv",
            )
        else:
            st.warning(f"No signals found for {keyword} in the last {days_lookback} day(s).")
    
    return symbols_df

# --- Option Flow Functions ---

def validate_csv_content_type(response: requests.Response) -> bool:
    """Validate if the response content type is CSV."""
    return 'text/csv' in response.headers.get('Content-Type', '')

def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Apply filters to the DataFrame."""
    df = df[df['Volume'] >= 100]
    df['Expiration'] = pd.to_datetime(df['Expiration'])
    df = df[df['Expiration'].dt.date >= datetime.now().date()]
    return df

def fetch_data_from_url(url: str) -> Optional[pd.DataFrame]:
    """Fetch and process data from a single URL."""
    try:
        response = requests.get(url)
        response.raise_for_status()

        if validate_csv_content_type(response):
            csv_data = StringIO(response.text)
            df = pd.read_csv(csv_data)
            return apply_filters(df)
        else:
            logger.warning(f"Data from {url} is not in CSV format. Skipping...")
    except Exception as e:
        logger.error(f"Error fetching or processing data from {url}: {e}")
    return None

def fetch_data_from_urls(urls: List[str]) -> pd.DataFrame:
    """Fetch and combine data from multiple CSV URLs into a single DataFrame."""
    data_frames = []
    for url in urls:
        df = fetch_data_from_url(url)
        if df is not None:
            data_frames.append(df)
    return pd.concat(data_frames, ignore_index=True) if data_frames else pd.DataFrame()

def filter_risk_reversal(df: pd.DataFrame, exclude_symbols: List[str], strike_proximity: int = 5) -> pd.DataFrame:
    """Filter for Risk Reversal trades by grouping calls and puts with similar strike prices."""
    if exclude_symbols:
        df = df[~df['Symbol'].isin(exclude_symbols)]

    calls = df[df['Call/Put'] == 'C']
    puts = df[df['Call/Put'] == 'P']

    merged = pd.merge(
        calls, puts,
        on=['Symbol', 'Expiration'],
        suffixes=('_call', '_put')
    )

    merged = merged[
        (abs(merged['Strike Price_call'] - merged['Strike Price_put']) <= strike_proximity) &
        (merged['Volume_call'] >= 3000) &
        (merged['Volume_put'] >= 3000)
    ]

    columns_to_keep = [
        'Symbol', 'Expiration',
        'Strike Price_call', 'Volume_call', 'Last Price_call',
        'Strike Price_put', 'Volume_put', 'Last Price_put'
    ]
    merged = merged[columns_to_keep]

    merged = merged.drop_duplicates(subset=[
        'Symbol', 'Expiration', 'Strike Price_call', 'Strike Price_put'
    ])

    reshaped_data = []
    for _, row in merged.iterrows():
        reshaped_data.append({
            'Symbol': row['Symbol'],
            'Type': 'Call',
            'Expiration': row['Expiration'],
            'Strike Price': row['Strike Price_call'],
            'Volume': row['Volume_call'],
            'Last Price': row['Last Price_call']
        })
        reshaped_data.append({
            'Symbol': row['Symbol'],
            'Type': 'Put',
            'Expiration': row['Expiration'],
            'Strike Price': row['Strike Price_put'],
            'Volume': row['Volume_put'],
            'Last Price': row['Last Price_put']
        })

    reshaped_df = pd.DataFrame(reshaped_data)
    reshaped_df = reshaped_df.drop_duplicates(subset=[
        'Symbol', 'Expiration', 'Strike Price', 'Type'
    ])

    return reshaped_df

def summarize_transactions(df: pd.DataFrame, whale_filter: bool = False, exclude_symbols: List[str] = None) -> pd.DataFrame:
    """Summarize transactions from the given DataFrame."""
    if exclude_symbols:
        df = df[~df['Symbol'].isin(exclude_symbols)]

    df['Transaction Value'] = df['Volume'] * df['Last Price'] * 100

    if whale_filter:
        df = df[df['Transaction Value'] > 5_000_000]

    summary = (
        df.groupby(['Symbol', 'Expiration', 'Strike Price', 'Call/Put', 'Last Price'])
        .agg({'Volume': 'sum', 'Transaction Value': 'sum'})
        .reset_index()
    )
    return summary.sort_values(by='Transaction Value', ascending=False)

def render_option_flow_section(symbols: List[str]):
    """Render the option flow summary section for the given symbols."""
    st.subheader("Option Flow Summary for Scanned Stocks")

    # Sidebar for filters
    with st.sidebar:
        st.header("Option Flow Filters")
        whale_option = st.checkbox("Show Whale Transactions Only", value=False)
        risk_reversal_option = st.checkbox("Show Risk Reversal Trades", value=False)
        
        default_excluded_symbols = ["SPX", "SPXW", "VIX", "SPY"]
        excluded_symbols = st.text_input(
            "Enter symbols to exclude (comma-separated)",
            value=", ".join(default_excluded_symbols)
        )
        excluded_symbols = [s.strip() for s in excluded_symbols.split(",") if s.strip()]

    # URLs for CBOE data
    urls = [
        "https://www.cboe.com/us/options/market_statistics/symbol_data/csv/?mkt=cone",
        "https://www.cboe.com/us/options/market_statistics/symbol_data/csv/?mkt=opt",
        "https://www.cboe.com/us/options/market_statistics/symbol_data/csv/?mkt=ctwo",
        "https://www.cboe.com/us/options/market_statistics/symbol_data/csv/?mkt=exo"
    ]

    # Fetch data with a progress spinner
    with st.spinner("Fetching option flow data..."):
        data = fetch_data_from_urls(urls)

    if not data.empty:
        # Filter data for the symbols from the scans
        data = data[data['Symbol'].isin(symbols)]
        
        if data.empty:
            st.warning("No option flow data available for the scanned stocks.")
            return

        # Use tabs for different views
        tab1, tab2, tab3 = st.tabs(["Risk Reversal Trades", "Whale Transactions", "Options Flow Analysis"])

        with tab1:
            if risk_reversal_option:
                st.subheader("Risk Reversal Trades")
                risk_reversal_data = filter_risk_reversal(data, exclude_symbols=excluded_symbols)
                if not risk_reversal_data.empty:
                    st.dataframe(risk_reversal_data, use_container_width=True)
                    csv = risk_reversal_data.to_csv(index=False)
                    st.download_button(
                        label="Download Risk Reversal Trades as CSV",
                        data=csv,
                        file_name="risk_reversal_trades.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("No Risk Reversal trades found for the selected symbols.")
            else:
                st.info("Enable 'Show Risk Reversal Trades' in the sidebar to view this section.")

        with tab2:
            if whale_option:
                st.subheader("Whale Transactions")
                summary = summarize_transactions(data, whale_filter=True, exclude_symbols=excluded_symbols)
                if not summary.empty:
                    st.dataframe(summary, use_container_width=True)
                    csv = summary.to_csv(index=False)
                    st.download_button(
                        label="Download Whale Transactions as CSV",
                        data=csv,
                        file_name="whale_transactions_summary.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("No Whale Transactions found for the selected symbols.")
            else:
                st.info("Enable 'Show Whale Transactions Only' in the sidebar to view this section.")

        with tab3:
            st.subheader("Options Flow Analysis")
            selected_symbol = st.selectbox("Select Symbol to Analyze", sorted(symbols))
            symbol_data = data[data['Symbol'] == selected_symbol]

            if not symbol_data.empty:
                strike_prices = sorted(symbol_data['Strike Price'].unique())
                selected_strike_price = st.selectbox("Select Strike Price (Optional)", [None] + strike_prices)

                call_put_options = ['C', 'P']
                selected_call_put = st.radio("Select Call/Put (Optional)", [None] + call_put_options, horizontal=True)

                if selected_strike_price:
                    symbol_data = symbol_data[symbol_data['Strike Price'] == selected_strike_price]

                if selected_call_put:
                    symbol_data = symbol_data[symbol_data['Call/Put'] == selected_call_put]

                summary = summarize_transactions(symbol_data, whale_filter=False, exclude_symbols=excluded_symbols)
                if not summary.empty:
                    st.dataframe(summary, use_container_width=True)
                    csv = summary.to_csv(index=False)
                    st.download_button(
                        label="Download Summary as CSV",
                        data=csv,
                        file_name=f"{selected_symbol}_summary.csv",
                        mime="text/csv"
                    )
                else:
                    st.info(f"No option flow data available for {selected_symbol} with the selected filters.")
            else:
                st.info(f"No option flow data available for {selected_symbol}.")
    else:
        st.warning("No option flow data fetched from CBOE.")

def run():
    """Main function to run the Streamlit application"""
    # Initialize session state
    init_session_state()
    
    # Add sidebar for settings
    with st.sidebar:
        st.header("Daily Scan Settings")
        days_lookback = st.slider(
            "Days to Look Back",
            min_value=1,
            max_value=3,
            value=1,
            help="Choose how many days of historical alerts to analyze"
        )
        
        auto_refresh = st.checkbox("Enable Auto-refresh", value=False)
        if auto_refresh:
            refresh_interval = st.slider("Refresh Interval (minutes)", 1, 30, 10)
        
        st.markdown("---")

    # Market data
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        spy_price, qqq_price, spy_change, qqq_change = get_spy_qqq_prices()
        if spy_price and spy_change is not None:
            st.metric(
                "SPY Latest", 
                f"${spy_price}",
                f"{spy_change:+.2f}%",
                delta_color="normal"
            )
    with col2:
        if qqq_price and qqq_change is not None:
            st.metric(
                "QQQ Latest", 
                f"${qqq_price}",
                f"{qqq_change:+.2f}%",
                delta_color="normal"
            )
    with col3:
        if st.button("🔄 Refresh Data"):
            st.session_state.cached_data.clear()
            st.session_state.processed_email_ids.clear()
            st.rerun()

    # Auto-refresh logic
    if auto_refresh and st.session_state.last_refresh_time:
        time_since_refresh = time.time() - st.session_state.last_refresh_time
        if time_since_refresh >= refresh_interval * 60:
            st.session_state.cached_data.clear()
            st.session_state.processed_email_ids.clear()
            st.session_state.last_refresh_time = time.time()
            st.rerun()

    # Daily scans section
    st.subheader("Daily Scans")
    all_symbols = set()
    symbol_dfs = []
    
    for keyword in DAILY_KEYWORDS:
        symbols_df = render_stock_section(keyword, days_lookback)
        if not symbols_df.empty:
            all_symbols.update(symbols_df['Ticker'].unique())
            symbol_dfs.append(symbols_df)

    # Option flow section
    if all_symbols:
        render_option_flow_section(list(all_symbols))
    else:
        st.warning("No stocks found in the daily scans to analyze for option flow.")

    # Update last refresh time
    st.session_state.last_refresh_time = time.time()
    
    # Footer
    st.markdown("---")
    st.markdown("""
        Disclaimer: This tool is for informational purposes only and does not constitute financial advice. 
        Trade at your own risk.
        
        Last updated: {}
        """.format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")), 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    run()
