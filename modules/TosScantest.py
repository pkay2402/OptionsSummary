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
from st_aggrid import AgGrid, GridOptionsBuilder

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
Lower_timeframe_KEYWORDS = ["Long_VP", "Short_VP", "orb_bull", "orb_bear", "volume_scan", "A+Bull_30m", "tmo_long", "tmo_Short"]
DAILY_KEYWORDS = ["rising5sma", "falling5sma", "HighVolumeSymbols", "Long_IT_volume", "Short_IT_volume", 
                  "bull_Daily_sqz", "bear_Daily_sqz", "LSMHG_Long", "LSMHG_Short", "StockReversalLong", "StockReversalShort"]
OPTION_KEYWORDS = ["ETF_options", "UOP_Call", "call_swing", "put_swing"]

# Keyword definitions (abbreviated for brevity)
KEYWORD_DEFINITIONS = {
    "Long_VP": {"description": "Volume Profile based long signal.", "risk_level": "Medium", "timeframe": "2 weeks", "suggested_stop": "Below the volume node"},
    # ... (include all other definitions from your original code)
}

@lru_cache(maxsize=2)
def get_spy_qqq_prices():
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

def extract_stock_symbols_from_email(email нарaddress, password, sender_email, keyword, days_lookback):
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
            df['Date'] = pd.to_datetime(df['Date'])  # Ensure Date is datetime
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
            df['Date'] = pd.to_datetime(df['Date'])  # Ensure Date is datetime
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
        
        # Ensure 'Date' is datetime
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
        
        # Interactive table with AgGrid
        gb = GridOptionsBuilder.from_dataframe(df[display_cols])
        gb.configure_pagination(paginationAutoPageSize=True)
        gb.configure_side_bar()
        grid_options = gb.build()
        AgGrid(df[display_cols], grid_options=grid_options, height=300, fit_columns_on_grid_load=True)
        
        # Download button
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=f"Download {keyword} Data",
            data=csv,
            file_name=f"{keyword}_alerts_{datetime.date.today()}.csv",
            mime="text/csv"
        )
    else:
        st.warning(f"No signals found for {keyword} in the last {days_lookback} day(s).")

def main():
    init_session_state()
    
    # Sidebar settings
    with st.sidebar:
        st.header("Dashboard Settings")
        days_lookback = st.slider("Days to Look Back", 1, 3, 1)
        auto_refresh = st.checkbox("Auto-refresh", value=False)
        refresh_interval = st.slider("Refresh Interval (min)", 1, 30, 10) if auto_refresh else 10
        st.markdown("---")
        st.markdown("**Disclaimer**: This tool is for informational purposes only. Trade at your own risk.")

    # Header with market overview
    st.title("Thinkorswim Scan Dashboard")
    col1, col2, col3 = st.columns([2, 2, 1])
    spy_price, qqq_price, spy_change, qqq_change = get_spy_qqq_prices()
    with col1:
        if spy_price:
            st.metric("SPY", f"${spy_price}", f"{spy_change:+.2f}%")
    with col2:
        if qqq_price:
            st.metric("QQQ", f"${qqq_price}", f"{qqq_change:+.2f}%")
    with col3:
        if st.button("🔄 Refresh"):
            st.session_state.cached_data.clear()
            st.session_state.processed_email_ids.clear()
            st.rerun()
    
    # Auto-refresh logic
    if auto_refresh and time.time() - st.session_state.last_refresh_time >= refresh_interval * 60:
        st.session_state.cached_data.clear()
        st.session_state.processed_email_ids.clear()
        st.session_state.last_refresh_time = time.time()
        st.rerun()
    
    # Tabs for different scan types
    tabs = st.tabs(["Lower Timeframe", "Daily", "High Conviction", "Live Options"])
    
    with tabs[0]:
        st.header("Lower Timeframe Scans")
        for keyword in Lower_timeframe_KEYWORDS:
            render_dashboard_section(keyword, days_lookback, is_option=False)
    
    with tabs[1]:
        st.header("Daily Scans")
        for keyword in DAILY_KEYWORDS:
            render_dashboard_section(keyword, days_lookback, is_option=False)
    
    with tabs[2]:
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
    
    with tabs[3]:
        st.header("Live Options Scans")
        for keyword in OPTION_KEYWORDS:
            render_dashboard_section(keyword, days_lookback, is_option=True)
    
    # Footer
    st.markdown(f"**Last Updated**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
