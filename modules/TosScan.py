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

# Initialize session state at the very beginning
def init_session_state():
    if 'processed_email_ids' not in st.session_state:
        st.session_state['processed_email_ids'] = set()
    if 'last_refresh_time' not in st.session_state:
        st.session_state['last_refresh_time'] = time.time()
    if 'cached_data' not in st.session_state:
        st.session_state['cached_data'] = {}
    if 'previous_symbols' not in st.session_state:
        st.session_state['previous_symbols'] = {}

# Call initialization immediately
init_session_state()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fetch credentials from Streamlit Secrets
EMAIL_ADDRESS = st.secrets["EMAIL_ADDRESS"]
EMAIL_PASSWORD = st.secrets["EMAIL_PASSWORD"]

# Constants
POLL_INTERVAL = 600  # 10 minutes in seconds
SENDER_EMAIL = "alerts@thinkorswim.com"
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

# Define keywords for different scan types
Lower_timeframe_KEYWORDS = ["Long_VP", "Short_VP", "orb_bull", "orb_bear", "volume_scan", "A+Bull_30m", "tmo_long", "tmo_Short"]
DAILY_KEYWORDS = ["Long_IT_volume", "Short_IT_volume", "bull_Daily_sqz", "bear_Daily_sqz", "LSMHG_Long", "LSMHG_Short"]
OPTION_KEYWORDS = ["ETF_options", "UOP_Call"]

# Keyword definitions with added risk levels and descriptions
KEYWORD_DEFINITIONS = {
    "Long_VP": {
        "description": "Volume Profile based long signal.",
        "risk_level": "Medium",
        "timeframe": "2 weeks",
        "suggested_stop": "Below the volume node"
    },
    "Short_VP": {
        "description": "Volume Profile based short signal.",
        "risk_level": "Medium",
        "timeframe": "2 weeks",
        "suggested_stop": "Above the volume node"
    },
    "orb_bull": {
        "description": "10 mins 9 ema crossed above opening range high of 30mins",
        "risk_level": "high",
        "timeframe": "Intraday",
        "suggested_stop": "Below the ORB high"
    },
    "orb_bear": {
        "description": "10 mins 9 ema crossed below opening range low of 30mins",
        "risk_level": "high",
        "timeframe": "Intraday",
        "suggested_stop": "Above the ORB low"
    },
    "volume_scan": {
        "description": "high intrday volume and stock atleast 2% up",
        "risk_level": "high",
        "timeframe": "Intraday. Enter at vwap test/and trading above 9 ema on 10mins",
        "suggested_stop": "below vwap"
    },
    "A+Bull_30m": {
        "description": "oversold stocks entering bullish zone",
        "risk_level": "medium",
        "timeframe": "2 weeks",
        "suggested_stop": "below recent low/support"
    },
    "tmo_long": {
        "description": "oversold stocks entering bullish momentum on 60mins",
        "risk_level": "medium",
        "timeframe": "2 weeks",
        "suggested_stop": "below recent low/support"
    },
    "tmo_Short": {
        "description": "overbought stocks entering losing bullish momentum on 60mins",
        "risk_level": "medium",
        "timeframe": "2 weeks",
        "suggested_stop": "Above recent high/resistance"
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
    "ETF_options": {
        "description": "ETF options showing potential momentum setups",
        "risk_level": "High",
        "timeframe": "As per expiry date",
        "suggested_stop": "Based on risk apetite"
    },
    "UOP_Call": {
        "description": "Unusual options activity scanner for calls",
        "risk_level": "High",
        "timeframe": "As per expiry date",
        "suggested_stop": "Based on risk apetite"
    }
}
@lru_cache(maxsize=2)
def get_spy_qqq_prices():
    """Fetch the latest closing prices and daily changes for SPY and QQQ with caching."""
    try:
        spy = yf.Ticker("SPY")
        qqq = yf.Ticker("QQQ")
        
        # Get today's data for both tickers
        spy_hist = spy.history(period="2d")  # Get 2 days to calculate % change
        qqq_hist = qqq.history(period="2d")
        
        # Calculate latest prices
        spy_price = round(spy_hist['Close'].iloc[-1], 2)
        qqq_price = round(qqq_hist['Close'].iloc[-1], 2)
        
        # Calculate percentage changes
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
    # Check if this data is already in cache
    if keyword in st.session_state['cached_data']:
        return st.session_state['cached_data'][keyword]

    try:
        mail = connect_to_email()
        mail.select('inbox')

        # Calculate the start date based on days_lookback
        today = datetime.date.today()
        start_date = today
        if days_lookback > 1:
            start_date = today - datetime.timedelta(days=days_lookback-1)
        
        date_since = start_date.strftime("%d-%b-%Y")
        search_criteria = f'(FROM "{sender_email}" SUBJECT "{keyword}" SINCE "{date_since}")'
        _, data = mail.search(None, search_criteria)

        stock_data = []
        
        for num in data[0].split():
            if num in st.session_state['processed_email_ids']:
                continue

            _, data = mail.fetch(num, '(RFC822)')
            msg = email.message_from_bytes(data[0][1])
            
            # Parse the email datetime
            email_datetime = parser.parse(msg['Date'])
            email_date = email_datetime.date()
            
            # Skip if email date is before start_date
            if email_date < start_date:
                continue
                
            # Skip weekends
            if email_datetime.weekday() >= 5:
                continue

            body = parse_email_body(msg)
            symbols = re.findall(r'New symbols:\s*([A-Z,\s]+)\s*were added to\s*(' + re.escape(keyword) + ')', body)
            
            if symbols:
                for symbol_group in symbols:
                    extracted_symbols = symbol_group[0].replace(" ", "").split(",")
                    signal_type = symbol_group[1]
                    for symbol in extracted_symbols:
                        if symbol.isalpha():  # Basic symbol validation
                            stock_data.append([symbol, email_datetime, signal_type])
            
            st.session_state['processed_email_ids'].add(num)

        mail.close()
        mail.logout()

        if stock_data:
            df = pd.DataFrame(stock_data, columns=['Ticker', 'Date', 'Signal'])
            df = df.sort_values(by=['Date', 'Ticker']).drop_duplicates(subset=['Ticker', 'Signal', 'Date'], keep='last')
            
            # Cache the data for this keyword
            st.session_state['cached_data'][keyword] = df
            return df

        # Cache empty DataFrame if no data found
        st.session_state['cached_data'][keyword] = pd.DataFrame(columns=['Ticker', 'Date', 'Signal'])
        return st.session_state['cached_data'][keyword]

    except Exception as e:
        logger.error(f"Error in extract_stock_symbols_from_email: {e}")
        st.error(f"Error processing emails: {str(e)}")
        return pd.DataFrame(columns=['Ticker', 'Date', 'Signal'])

def high_conviction_stocks(dataframes, ignore_keywords=None):
    """Find stocks with high conviction - at least two unique keyword matches on the same date."""
    if ignore_keywords is None:
        ignore_keywords = []
    
    # Filter out the ignored keywords before processing
    filtered_dataframes = [df[~df['Signal'].isin(ignore_keywords)] for df in dataframes if not df.empty]
    
    all_data = pd.concat(filtered_dataframes, ignore_index=True)
    
    # Convert datetime to date for grouping in high conviction
    all_data['Date'] = all_data['Date'].dt.date
    
    # Aggregate using set to ensure unique signals
    grouped = all_data.groupby(['Date', 'Ticker'])['Signal'].agg(lambda x: ', '.join(set(x))).reset_index()
    
    # Count unique signals
    grouped['Count'] = grouped['Signal'].apply(lambda x: len(x.split(', ')))
    
    high_conviction = grouped[grouped['Count'] >= 2][['Date', 'Ticker', 'Signal']]
    
    return high_conviction

def get_new_symbols_count(keyword, current_df):
    if current_df.empty:
        print(f"⚠️ Warning: DataFrame is empty for keyword: {keyword}")
        return 0

    print(f"🔎 Available columns in current_df for {keyword}: {list(current_df.columns)}")  # Debugging line

    # Determine which column to use instead of 'Ticker'
    possible_ticker_columns = ['Ticker', 'ticker', 'Symbol', 'Raw_Symbol', 'Readable_Symbol']
    ticker_col = next((col for col in possible_ticker_columns if col in current_df.columns), None)

    if ticker_col is None:
        raise KeyError(f"❌ No valid ticker column found in DataFrame for keyword: {keyword}. Found columns: {list(current_df.columns)}")

    print(f"✅ Using '{ticker_col}' as the ticker column.")
    current_symbols = set(current_df[ticker_col].unique())
    return len(current_symbols)


        
    # Get previous symbols for this keyword
    previous_symbols = st.session_state['previous_symbols'].get(keyword, set())
    
    # Get current symbols
    current_symbols = set(current_df['Ticker'].unique())
    
    # Calculate new symbols
    new_symbols = current_symbols - previous_symbols
    
    # Update previous symbols for next comparison
    st.session_state['previous_symbols'][keyword] = current_symbols
    
    return len(new_symbols)

def parse_option_symbol(option_symbol):
    try:
        # Remove leading dot if present
        symbol = option_symbol.lstrip('.')
        
        # Extract components using regex
        pattern = r'([A-Z]+)(\d{2})(\d{2})(\d{2})([CP])([\d_]+)'
        match = re.match(pattern, symbol)
        
        if match:
            Ticker, year, month, day, opt_type, strike = match.groups()
            
            # Convert month number to month abbreviation
            month_name = datetime.datetime.strptime(month, '%m').strftime('%b').upper()
            
            # Format strike price (replace underscore with a decimal point)
            strike = strike.replace('_', '.')
            
            # Determine if it's a CALL or PUT
            option_type = 'CALL' if opt_type == 'C' else 'PUT'
            
            return f"{Ticker} {day} {month_name} 20{year} {strike} {option_type}"
    
    except Exception as e:
        logger.error(f"Error parsing option symbol {option_symbol}: {e}")

    return option_symbol
    

def extract_option_symbols_from_email(email_address, password, sender_email, keyword, days_lookback):
    """Extract option symbols from email alerts with proper date filtering."""
    # Check cache
    if keyword in st.session_state['cached_data']:
        return st.session_state['cached_data'][keyword]

    try:
        mail = connect_to_email()
        mail.select('inbox')

        # Calculate the start date based on days_lookback
        today = datetime.date.today()
        start_date = today
        if days_lookback > 1:
            start_date = today - datetime.timedelta(days=days_lookback-1)
            
        date_since = start_date.strftime("%d-%b-%Y")
        search_criteria = f'(FROM "{sender_email}" SUBJECT "{keyword}" SINCE "{date_since}")'
        _, data = mail.search(None, search_criteria)

        option_data = []
        
        for num in data[0].split():
            if num in st.session_state['processed_email_ids']:
                continue

            _, data = mail.fetch(num, '(RFC822)')
            msg = email.message_from_bytes(data[0][1])
            
            # Parse the email datetime
            email_datetime = parser.parse(msg['Date'])
            email_date = email_datetime.date()
            
            # Skip if email date is before start_date
            if email_date < start_date:
                continue
                
            # Skip weekends
            if email_datetime.weekday() >= 5:
                continue

            body = parse_email_body(msg)
            symbols = re.findall(r'New symbols:\s*([\.\w,\s]+)\s*were added to\s*(' + re.escape(keyword) + ')', body)
            
            if symbols:
                for symbol_group in symbols:
                    extracted_symbols = symbol_group[0].replace(" ", "").split(",")
                    signal_type = symbol_group[1]
                    for symbol in extracted_symbols:
                        if symbol:  # Basic validation
                            readable_symbol = parse_option_symbol(symbol)
                            option_data.append([symbol, readable_symbol, email_datetime, signal_type])
            
            st.session_state['processed_email_ids'].add(num)

        mail.close()
        mail.logout()

        if option_data:
            df = pd.DataFrame(option_data, columns=['Raw_Symbol', 'Readable_Symbol', 'Date', 'Signal'])
            df = df.sort_values(by=['Date', 'Raw_Symbol']).drop_duplicates(subset=['Raw_Symbol', 'Signal', 'Date'], keep='last')
            
            st.session_state['cached_data'][keyword] = df
            return df

        st.session_state['cached_data'][keyword] = pd.DataFrame(columns=['Raw_Symbol', 'Readable_Symbol', 'Date', 'Signal'])
        return st.session_state['cached_data'][keyword]

    except Exception as e:
        logger.error(f"Error in extract_option_symbols_from_email: {e}")
        st.error(f"Error processing emails: {str(e)}")
        return pd.DataFrame(columns=['Raw_Symbol', 'Readable_Symbol', 'Date', 'Signal'])

def render_options_section(keyword, days_lookback):
    """Helper function to render options section content"""
    symbols_df = extract_option_symbols_from_email(
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
            display_df = display_df.drop('Raw_Symbol', axis=1)
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

import streamlit as st
import datetime

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

def run():

    
    # Add sidebar for settings
    with st.sidebar:
        st.header("Settings")
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
            st.session_state['cached_data'].clear()
            st.session_state['processed_email_ids'].clear()
            st.rerun()

    # Auto-refresh logic
    if auto_refresh and st.session_state['last_refresh_time']:
        time_since_refresh = time.time() - st.session_state['last_refresh_time']
        if time_since_refresh >= refresh_interval * 60:
            st.session_state['cached_data'].clear()
            st.session_state['processed_email_ids'].clear()
            st.session_state['last_refresh_time'] = time.time()
            st.rerun()

    # Scan type selection
    section = st.radio("Select View", ["Lower_timeframe", "Daily", "High Conviction", "Live Options"], 
                      index=0, horizontal=True)
    
    if section == "Lower_timeframe":
        st.subheader("Lower Timeframe Scans")
        for keyword in Lower_timeframe_KEYWORDS:
            render_stock_section(keyword, days_lookback)
            
    elif section == "Daily":
        st.subheader("Daily Scans")
        for keyword in DAILY_KEYWORDS:
            render_stock_section(keyword, days_lookback)
            
    elif section == "Live Options":
        st.subheader("Live Options Scans")
        for keyword in OPTION_KEYWORDS:
            render_options_section(keyword, days_lookback)
            
    elif section == "High Conviction":
        st.subheader("High Conviction Scans")
        all_signals = []
        
        # Collect all signals from both Lower_timeframe and Daily scans
        for keyword in Lower_timeframe_KEYWORDS + DAILY_KEYWORDS:
            df = extract_stock_symbols_from_email(
                EMAIL_ADDRESS, EMAIL_PASSWORD, SENDER_EMAIL, keyword, days_lookback
            )
            if not df.empty:
                all_signals.append(df)
        
        if all_signals:
            # Ignore tmo_long and tmo_Short for High Conviction
            high_conviction_df = high_conviction_stocks(
                all_signals, 
                ignore_keywords=["tmo_long", "tmo_Short"]
            )
            
            if not high_conviction_df.empty:
                st.dataframe(high_conviction_df, use_container_width=True)
                logger.info(f"High Conviction data: {high_conviction_df.to_string()}")
                
                csv = high_conviction_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download High Conviction Data",
                    data=csv,
                    file_name=f"high_conviction_alerts_{datetime.date.today()}.csv",
                    mime="text/csv",
                )
            else:
                st.warning("No high conviction signals found after filtering out specified keywords.")
        else:
            st.warning("No signals found to process for high conviction view.")

    # Update last refresh time
    st.session_state['last_refresh_time'] = time.time()
    
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
