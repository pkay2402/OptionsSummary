import streamlit as st
import imaplib
import email
import re
import datetime
import pandas as pd
from dateutil import parser
import time
from bs4 import BeautifulSoup
import logging
import yfinance as yf
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
SENDER_EMAIL = "alerts@thinkorswim.com"
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

# Define keywords for different scan types
OPTION_KEYWORDS = ["ETF_options", "UOP_Call","call_swing","put_swing"]

# Keyword definitions with added risk levels and descriptions
KEYWORD_DEFINITIONS = {
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
    },
    "call_swing": {
        "description": "stock options showing potential momentum setups",
        "risk_level": "High",
        "timeframe": "As per expiry date",
        "suggested_stop": "Based on risk apetite"
    },
    "put_swing": {
        "description": "stock options showing potential bearish setups",
        "risk_level": "High",
        "timeframe": "As per expiry date",
        "suggested_stop": "Based on risk apetite"
    }
}

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

def get_new_symbols_count(keyword, current_df):
    """Get count of new symbols."""
    if current_df.empty:
        return 0

    possible_ticker_columns = ['Ticker', 'ticker', 'Symbol', 'Raw_Symbol', 'Readable_Symbol']
    ticker_col = next((col for col in possible_ticker_columns if col in current_df.columns), None)

    if ticker_col is None:
        raise KeyError(f"No valid ticker column found in DataFrame for keyword: {keyword}")

    current_symbols = set(current_df[ticker_col].unique())
    previous_symbols = st.session_state.previous_symbols.get(keyword, set())
    
    new_symbols = current_symbols - previous_symbols
    st.session_state.previous_symbols[keyword] = current_symbols
    
    return len(new_symbols)

# Compile regex pattern once for better performance
OPTION_PATTERN = re.compile(r'([A-Z]+)(\d{2})(\d{2})(\d{2})([CP])([\d_]+)')
MONTH_NAMES = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 
               'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']

def parse_option_symbol(option_symbol):
    """Parse option symbol into readable format."""
    try:
        symbol = option_symbol.lstrip('.')
        match = OPTION_PATTERN.match(symbol)
        
        if match:
            ticker, year, month, day, opt_type, strike = match.groups()
            month_name = MONTH_NAMES[int(month) - 1]
            strike = strike.replace('_', '.')
            option_type = 'CALL' if opt_type == 'C' else 'PUT'
            return f"{ticker} {day} {month_name} 20{year} {strike} {option_type}"
    
    except Exception as e:
        logger.error(f"Error parsing option symbol {option_symbol}: {e}")

    return option_symbol

@lru_cache(maxsize=100)
def get_option_data(raw_symbol):
    """Fetch option data including volume and open interest."""
    try:
        # Parse the raw symbol to get ticker and option details
        symbol = raw_symbol.lstrip('.')
        match = OPTION_PATTERN.match(symbol)
        
        if not match:
            logger.warning(f"Could not parse option symbol: {raw_symbol}")
            return None, None
        
        ticker, year, month, day, opt_type, strike_str = match.groups()
        
        # Convert strike to float
        strike_price = float(strike_str.replace('_', '.'))
        
        # Construct expiration date (YYYY-MM-DD format)
        expiry_date = f"20{year}-{month}-{day}"
        
        # Construct yfinance contract symbol format: TICKER + YYMMDD + C/P + strike*1000 padded to 8 digits
        # Example: BAC251219C00057500 for BAC $57.5 Call expiring 12/19/2025
        strike_int = int(strike_price * 1000)
        yf_contract_symbol = f"{ticker}{year}{month}{day}{opt_type}{strike_int:08d}"
        
        # Get option chain data for the specific expiration date
        stock = yf.Ticker(ticker)
        
        # Get all available expiration dates
        expirations = stock.options
        
        if expiry_date not in expirations:
            logger.warning(f"Expiration date {expiry_date} not found for {ticker}. Available: {expirations[:5] if len(expirations) > 0 else 'None'}")
            return None, None
        
        # Get option chain for the specific date
        option_chain = stock.option_chain(expiry_date)
        
        # Determine if it's a call or put
        if opt_type == 'C':
            chain = option_chain.calls
        else:
            chain = option_chain.puts
        
        # Try to match by yfinance contract symbol format
        matching_options = chain[chain['contractSymbol'] == yf_contract_symbol]
        
        # If still not found, try matching by strike price (with small tolerance for floating point)
        if matching_options.empty:
            logger.info(f"Exact match not found for {yf_contract_symbol}, trying strike price match")
            strike_tolerance = 0.01
            matching_options = chain[abs(chain['strike'] - strike_price) < strike_tolerance]
        
        if not matching_options.empty:
            # If multiple matches, take the first one (should be rare)
            volume = matching_options['volume'].values[0]
            open_interest = matching_options['openInterest'].values[0]
            contract_found = matching_options['contractSymbol'].values[0]
            
            # Handle NaN values
            volume = int(volume) if pd.notna(volume) else 0
            open_interest = int(open_interest) if pd.notna(open_interest) else 0
            
            logger.info(f"Fetched data for {raw_symbol} -> {contract_found}: Volume={volume}, OI={open_interest}")
            return volume, open_interest
        else:
            logger.warning(f"Contract {raw_symbol} -> {yf_contract_symbol} (strike={strike_price}) not found in option chain for {ticker} {expiry_date}")
            return None, None
        
    except Exception as e:
        logger.warning(f"Could not fetch option data for {raw_symbol}: {e}")
        return None, None

def extract_option_symbols_from_email(email_address, password, sender_email, keyword, days_lookback):
    """Extract option symbols from email alerts."""
    if keyword in st.session_state.cached_data:
        return st.session_state.cached_data[keyword]

    empty_df = pd.DataFrame(columns=['Raw_Symbol', 'Readable_Symbol', 'Date', 'Signal'])
    
    try:
        mail = connect_to_email()
        mail.select('inbox')

        today = datetime.date.today()
        start_date = today - datetime.timedelta(days=days_lookback-1) if days_lookback > 1 else today
            
        date_since = start_date.strftime("%d-%b-%Y")
        search_criteria = f'(FROM "{sender_email}" SUBJECT "{keyword}" SINCE "{date_since}")'
        _, data = mail.search(None, search_criteria)

        email_nums = data[0].split()
        if not email_nums:
            st.session_state.cached_data[keyword] = empty_df
            mail.close()
            mail.logout()
            return empty_df

        option_data = []
        # Compile regex pattern once
        symbol_pattern = re.compile(r'New symbols:\s*([\.\w,\s]+)\s*were added to\s*(' + re.escape(keyword) + ')')
        
        for num in email_nums:
            if num in st.session_state.processed_email_ids:
                continue

            _, data = mail.fetch(num, '(RFC822)')
            msg = email.message_from_bytes(data[0][1])
            
            email_datetime = parser.parse(msg['Date'])
            email_date = email_datetime.date()
            
            # Skip weekends and dates outside range
            if email_date < start_date or email_datetime.weekday() >= 5:
                st.session_state.processed_email_ids.add(num)
                continue

            body = parse_email_body(msg)
            symbols = symbol_pattern.findall(body)
            
            if symbols:
                for symbol_group in symbols:
                    extracted_symbols = [s for s in symbol_group[0].replace(" ", "").split(",") if s]
                    signal_type = symbol_group[1]
                    for symbol in extracted_symbols:
                        readable_symbol = parse_option_symbol(symbol)
                        option_data.append([symbol, readable_symbol, email_datetime, signal_type])
            
            st.session_state.processed_email_ids.add(num)

        mail.close()
        mail.logout()

        if option_data:
            df = pd.DataFrame(option_data, columns=['Raw_Symbol', 'Readable_Symbol', 'Date', 'Signal'])
            df = df.sort_values(by=['Date', 'Raw_Symbol']).drop_duplicates(subset=['Raw_Symbol', 'Signal', 'Date'], keep='last')
            
            # Add volume and open interest data
            volumes = []
            open_interests = []
            
            for raw_symbol in df['Raw_Symbol']:
                volume, oi = get_option_data(raw_symbol)
                volumes.append(volume if volume is not None else 'N/A')
                open_interests.append(oi if oi is not None else 'N/A')
            
            df['Volume'] = volumes
            df['Open_Interest'] = open_interests
            
            st.session_state.cached_data[keyword] = df
            return df

        st.session_state.cached_data[keyword] = empty_df
        return empty_df

    except Exception as e:
        logger.error(f"Error in extract_option_symbols_from_email: {e}")
        st.error(f"Error processing emails: {str(e)}")
        return empty_df

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
            
            # Ensure Volume and Open_Interest columns exist
            if 'Volume' not in display_df.columns:
                display_df['Volume'] = 'N/A'
            if 'Open_Interest' not in display_df.columns:
                display_df['Open_Interest'] = 'N/A'
            
            # Reorder columns to show most relevant info first
            column_order = ['Readable_Symbol', 'Date', 'Signal', 'Volume', 'Open_Interest']
            display_df = display_df[column_order]
            st.dataframe(display_df, width=True)
            
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
    """Main function to run the Streamlit application"""
    # Initialize session state first
    init_session_state()
    
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

    # Refresh button
    if st.button("🔄 Refresh Data"):
        st.session_state.cached_data.clear()
        st.session_state.processed_email_ids.clear()
        get_option_data.cache_clear()  # Clear the LRU cache for option data
        st.rerun()

    # Auto-refresh logic
    if auto_refresh:
        time_since_refresh = time.time() - st.session_state.last_refresh_time
        if time_since_refresh >= refresh_interval * 60:
            st.session_state.cached_data.clear()
            st.session_state.processed_email_ids.clear()
            st.session_state.last_refresh_time = time.time()
            st.rerun()

    # Options view
    st.subheader("Options Scans")
    for keyword in OPTION_KEYWORDS:
        render_options_section(keyword, days_lookback)

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
