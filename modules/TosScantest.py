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

# Page configuration - must be first Streamlit command
st.set_page_config(
    page_title="Options Scanner",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

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

def get_all_options_data(days_lookback):
    """Fetch all options data from all keywords and combine into one dataframe."""
    all_data = []
    
    for keyword in OPTION_KEYWORDS:
        symbols_df = extract_option_symbols_from_email(
            EMAIL_ADDRESS, EMAIL_PASSWORD, SENDER_EMAIL, keyword, days_lookback
        )
        
        if not symbols_df.empty:
            # Add category column
            df_copy = symbols_df.copy()
            df_copy['Category'] = keyword
            all_data.append(df_copy)
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        return combined_df
    else:
        return pd.DataFrame()

def run():
    """Main function to run the Streamlit application"""
    # Initialize session state first
    init_session_state()
    
    # Page header
    st.title("📈 Options Scanner")
    
    # Top bar with settings
    col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
    
    with col1:
        days_lookback = st.selectbox(
            "Lookback Period",
            options=[1, 2, 3],
            index=1,  # Default to 2 days
            help="Number of days to analyze"
        )
    
    with col2:
        view_mode = st.radio(
            "View Mode",
            options=["Combined", "By Category"],
            horizontal=True,
            help="View all options together or separated by category"
        )
    
    with col3:
        auto_refresh = st.checkbox("Auto-refresh (10 min)", value=False)
    
    with col4:
        if st.button("🔄 Refresh", use_container_width=True):
            st.session_state.cached_data.clear()
            st.session_state.processed_email_ids.clear()
            get_option_data.cache_clear()
            st.rerun()

    st.markdown("---")

    # Auto-refresh logic
    if auto_refresh:
        time_since_refresh = time.time() - st.session_state.last_refresh_time
        if time_since_refresh >= 600:  # 10 minutes
            st.session_state.cached_data.clear()
            st.session_state.processed_email_ids.clear()
            st.session_state.last_refresh_time = time.time()
            st.rerun()

    # Display options based on view mode
    if view_mode == "Combined":
        # Get all options data combined
        all_options_df = get_all_options_data(days_lookback)
        
        if not all_options_df.empty:
            # Extract ticker from readable symbol for grouping
            all_options_df['Ticker'] = all_options_df['Readable_Symbol'].str.split().str[0]
            
            # Format date
            all_options_df['Date'] = all_options_df['Date'].dt.strftime('%Y-%m-%d %H:%M')
            
            # Ensure Volume and Open_Interest columns exist
            if 'Volume' not in all_options_df.columns:
                all_options_df['Volume'] = 'N/A'
            if 'Open_Interest' not in all_options_df.columns:
                all_options_df['Open_Interest'] = 'N/A'
            
            # Sort by ticker, then by date
            all_options_df = all_options_df.sort_values(['Ticker', 'Date'], ascending=[True, False])
            
            # Reorder columns
            column_order = ['Ticker', 'Readable_Symbol', 'Category', 'Date', 'Volume', 'Open_Interest']
            display_df = all_options_df[column_order]
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Alerts", len(display_df))
            with col2:
                st.metric("Unique Tickers", display_df['Ticker'].nunique())
            with col3:
                st.metric("Categories", display_df['Category'].nunique())
            
            st.markdown("### All Options Flows")
            
            # Display with better formatting
            st.dataframe(
                display_df, 
                use_container_width=True,
                hide_index=True,
                height=600,
                column_config={
                    "Ticker": st.column_config.TextColumn("Ticker", width="small"),
                    "Readable_Symbol": st.column_config.TextColumn("Option", width="large"),
                    "Category": st.column_config.TextColumn("Category", width="medium"),
                    "Date": st.column_config.TextColumn("Alert Time", width="medium"),
                    "Volume": st.column_config.NumberColumn("Volume", format="%d"),
                    "Open_Interest": st.column_config.NumberColumn("Open Interest", format="%d"),
                }
            )
            
            # Download button
            csv = all_options_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download All Data",
                data=csv,
                file_name=f"all_options_{datetime.date.today()}.csv",
                mime="text/csv",
            )
        else:
            st.info(f"No signals found in the last {days_lookback} day(s)")
    
    else:  # By Category view
        for keyword in OPTION_KEYWORDS:
            symbols_df = extract_option_symbols_from_email(
                EMAIL_ADDRESS, EMAIL_PASSWORD, SENDER_EMAIL, keyword, days_lookback
            )
            
            count = len(symbols_df) if not symbols_df.empty else 0
            
            with st.expander(f"{keyword} ({count})", expanded=False):
                if not symbols_df.empty:
                    display_df = symbols_df.copy()
                    display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d %H:%M')
                    
                    if 'Volume' not in display_df.columns:
                        display_df['Volume'] = 'N/A'
                    if 'Open_Interest' not in display_df.columns:
                        display_df['Open_Interest'] = 'N/A'
                    
                    column_order = ['Readable_Symbol', 'Date', 'Volume', 'Open_Interest']
                    display_df = display_df[column_order]
                    
                    st.dataframe(
                        display_df, 
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Readable_Symbol": st.column_config.TextColumn("Option", width="large"),
                            "Date": st.column_config.TextColumn("Alert Time", width="medium"),
                            "Volume": st.column_config.NumberColumn("Volume", format="%d"),
                            "Open_Interest": st.column_config.NumberColumn("Open Interest", format="%d"),
                        }
                    )
                    
                    csv = symbols_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv,
                        file_name=f"{keyword}_{datetime.date.today()}.csv",
                        mime="text/csv",
                        key=f"download_{keyword}"
                    )
                else:
                    st.info(f"No signals in the last {days_lookback} day(s)")

    # Update last refresh time
    st.session_state.last_refresh_time = time.time()
    
    # Compact footer
    st.markdown("---")
    st.caption(f"⚠️ For informational purposes only. Last updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    run()
