import streamlit as st
import imaplib
import email
import re
import datetime
import pandas as pd
from dateutil import parser
import time
from bs4 import BeautifulSoup
from functools import lru_cache
import logging
from concurrent.futures import ThreadPoolExecutor
import yfinance as yf
import numpy as np

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
POLL_INTERVAL = 600  # 10 minutes in seconds
SENDER_EMAIL = "alerts@thinkorswim.com"
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

# Define keywords for different scan types with their signal types
Lower_timeframe_KEYWORDS = ["tmo_long", "tmo_Short"]
DAILY_KEYWORDS = ["Long_IT_volume", "Short_IT_volume","vpb_buy","vpb_sell", "demark13_buy", "demark13_sell","bull_Daily_sqz", 
"bear_Daily_sqz", "LSMHG_Long", "LSMHG_Short","StockReversalLong","StockReversalShort"]

# Define signal types
LONG_SIGNALS = ["tmo_long", "Long_IT_volume", "vpb_buy", "demark13_buy", "bull_Daily_sqz", "LSMHG_Long", "StockReversalLong"]
SHORT_SIGNALS = ["tmo_Short", "Short_IT_volume", "vpb_sell", "demark13_sell", "bear_Daily_sqz", "LSMHG_Short", "StockReversalShort"]

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

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_stock_data(symbol):
    """Get stock data from yfinance with caching."""
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="3mo")  # 3 months of data
        info = ticker.info
        
        if hist.empty:
            return None
            
        current_price = hist['Close'].iloc[-1]
        volume = hist['Volume'].iloc[-1]
        avg_volume = hist['Volume'].rolling(20).mean().iloc[-1]
        
        # Technical indicators
        sma_20 = hist['Close'].rolling(20).mean().iloc[-1]
        sma_50 = hist['Close'].rolling(50).mean().iloc[-1]
        
        # Price momentum
        price_change_1d = ((current_price - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2]) * 100
        price_change_5d = ((current_price - hist['Close'].iloc[-6]) / hist['Close'].iloc[-6]) * 100
        price_change_20d = ((current_price - hist['Close'].iloc[-21]) / hist['Close'].iloc[-21]) * 100
        
        # Volatility
        volatility = hist['Close'].pct_change().std() * np.sqrt(252) * 100
        
        # 52-week high/low
        high_52w = hist['High'].max()
        low_52w = hist['Low'].min()
        
        return {
            'symbol': symbol,
            'current_price': current_price,
            'volume': volume,
            'avg_volume': avg_volume,
            'volume_ratio': volume / avg_volume if avg_volume > 0 else 0,
            'sma_20': sma_20,
            'sma_50': sma_50,
            'price_vs_sma20': ((current_price - sma_20) / sma_20) * 100,
            'price_vs_sma50': ((current_price - sma_50) / sma_50) * 100,
            'price_change_1d': price_change_1d,
            'price_change_5d': price_change_5d,
            'price_change_20d': price_change_20d,
            'volatility': volatility,
            'high_52w': high_52w,
            'low_52w': low_52w,
            'distance_from_52w_high': ((current_price - high_52w) / high_52w) * 100,
            'distance_from_52w_low': ((current_price - low_52w) / low_52w) * 100,
            'market_cap': info.get('marketCap', 0),
            'sector': info.get('sector', 'Unknown')
        }
    except Exception as e:
        logger.error(f"Error getting data for {symbol}: {e}")
        return None

def calculate_stock_score(stock_data, signal_type):
    """Calculate a score for stock based on technical indicators."""
    if not stock_data:
        return 0
    
    score = 0
    
    # Volume scoring (higher is better)
    if stock_data['volume_ratio'] > 2:
        score += 20
    elif stock_data['volume_ratio'] > 1.5:
        score += 15
    elif stock_data['volume_ratio'] > 1.2:
        score += 10
    
    # Price momentum scoring
    if signal_type == 'long':
        # For long signals, positive momentum is better
        if stock_data['price_change_1d'] > 2:
            score += 15
        elif stock_data['price_change_1d'] > 0:
            score += 10
        elif stock_data['price_change_1d'] < -2:
            score -= 10
            
        if stock_data['price_change_5d'] > 5:
            score += 15
        elif stock_data['price_change_5d'] > 0:
            score += 10
        elif stock_data['price_change_5d'] < -5:
            score -= 10
            
        # Price vs moving averages (above is better for long)
        if stock_data['price_vs_sma20'] > 0:
            score += 10
        if stock_data['price_vs_sma50'] > 0:
            score += 10
            
    else:  # short signals
        # For short signals, negative momentum is better
        if stock_data['price_change_1d'] < -2:
            score += 15
        elif stock_data['price_change_1d'] < 0:
            score += 10
        elif stock_data['price_change_1d'] > 2:
            score -= 10
            
        if stock_data['price_change_5d'] < -5:
            score += 15
        elif stock_data['price_change_5d'] < 0:
            score += 10
        elif stock_data['price_change_5d'] > 5:
            score -= 10
            
        # Price vs moving averages (below is better for short)
        if stock_data['price_vs_sma20'] < 0:
            score += 10
        if stock_data['price_vs_sma50'] < 0:
            score += 10
    
    # Volatility scoring (moderate volatility is preferred)
    if 20 <= stock_data['volatility'] <= 40:
        score += 10
    elif stock_data['volatility'] > 60:
        score -= 10
    
    # Distance from 52-week highs/lows
    if signal_type == 'long':
        # For long, being closer to 52w low is better (more upside potential)
        if stock_data['distance_from_52w_low'] < 20:
            score += 15
        elif stock_data['distance_from_52w_low'] < 50:
            score += 10
    else:  # short signals
        # For short, being closer to 52w high is better (more downside potential)
        if stock_data['distance_from_52w_high'] > -20:
            score += 15
        elif stock_data['distance_from_52w_high'] > -50:
            score += 10
    
    # Market cap scoring (prefer liquid stocks)
    if stock_data['market_cap'] > 10_000_000_000:  # > $10B
        score += 10
    elif stock_data['market_cap'] > 1_000_000_000:  # > $1B
        score += 5
    
    return max(0, score)  # Ensure non-negative score

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

    possible_ticker_columns = ['Ticker', 'ticker', 'Symbol', 'Raw_Symbol', 'Readable_Symbol']
    ticker_col = next((col for col in possible_ticker_columns if col in current_df.columns), None)

    if ticker_col is None:
        raise KeyError(f"No valid ticker column found in DataFrame for keyword: {keyword}")

    current_symbols = set(current_df[ticker_col].unique())
    previous_symbols = st.session_state.previous_symbols.get(keyword, set())
    
    new_symbols = current_symbols - previous_symbols
    st.session_state.previous_symbols[keyword] = current_symbols
    
    return len(new_symbols)

def high_conviction_stocks(dataframes, ignore_keywords=None):
    """Find stocks with high conviction - at least two unique keyword matches on the same date."""
    if ignore_keywords is None:
        ignore_keywords = []
    
    filtered_dataframes = [df[~df['Signal'].isin(ignore_keywords)] for df in dataframes if not df.empty]
    all_data = pd.concat(filtered_dataframes, ignore_index=True)
    all_data['Date'] = all_data['Date'].dt.date
    
    grouped = all_data.groupby(['Date', 'Ticker'])['Signal'].agg(lambda x: ', '.join(set(x))).reset_index()
    grouped['Count'] = grouped['Signal'].apply(lambda x: len(x.split(', ')))
    
    return grouped[grouped['Count'] >= 2][['Date', 'Ticker', 'Signal']]

def analyze_stocks_with_prices(df, signal_type):
    """Analyze stocks with price data and scoring."""
    if df.empty:
        return pd.DataFrame()
    
    analyzed_stocks = []
    
    # Get unique symbols
    symbols = df['Ticker'].unique()
    
    # Use progress bar for stock analysis
    progress_bar = st.progress(0)
    
    for i, symbol in enumerate(symbols):
        progress_bar.progress((i + 1) / len(symbols))
        
        stock_data = get_stock_data(symbol)
        if stock_data:
            # Calculate score
            score = calculate_stock_score(stock_data, signal_type)
            
            # Get signal info
            symbol_signals = df[df['Ticker'] == symbol]
            latest_signal = symbol_signals.iloc[-1]
            
            analyzed_stocks.append({
                'Ticker': symbol,
                'Current_Price': stock_data['current_price'],
                'Score': score,
                'Volume_Ratio': stock_data['volume_ratio'],
                'Price_Change_1D': stock_data['price_change_1d'],
                'Price_Change_5D': stock_data['price_change_5d'],
                'Price_vs_SMA20': stock_data['price_vs_sma20'],
                'Price_vs_SMA50': stock_data['price_vs_sma50'],
                'Volatility': stock_data['volatility'],
                'Distance_52W_High': stock_data['distance_from_52w_high'],
                'Distance_52W_Low': stock_data['distance_from_52w_low'],
                'Market_Cap': stock_data['market_cap'],
                'Sector': stock_data['sector'],
                'Signal': latest_signal['Signal'],
                'Signal_Date': latest_signal['Date']
            })
    
    progress_bar.empty()
    
    if analyzed_stocks:
        result_df = pd.DataFrame(analyzed_stocks)
        result_df = result_df.sort_values('Score', ascending=False)
        return result_df
    
    return pd.DataFrame()

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

def render_recommendations_section(days_lookback):
    """Render the stock recommendations section."""
    st.subheader("📈 Stock Recommendations")
    
    # Collect all long and short signals
    long_signals_df = []
    short_signals_df = []
    
    for keyword in Lower_timeframe_KEYWORDS + DAILY_KEYWORDS:
        df = extract_stock_symbols_from_email(
            EMAIL_ADDRESS, EMAIL_PASSWORD, SENDER_EMAIL, keyword, days_lookback
        )
        
        if not df.empty:
            if keyword in LONG_SIGNALS:
                long_signals_df.append(df)
            elif keyword in SHORT_SIGNALS:
                short_signals_df.append(df)
    
    # Combine dataframes
    if long_signals_df:
        all_long_signals = pd.concat(long_signals_df, ignore_index=True)
        all_long_signals = all_long_signals.drop_duplicates(subset=['Ticker'], keep='last')
    else:
        all_long_signals = pd.DataFrame()
    
    if short_signals_df:
        all_short_signals = pd.concat(short_signals_df, ignore_index=True)
        all_short_signals = all_short_signals.drop_duplicates(subset=['Ticker'], keep='last')
    else:
        all_short_signals = pd.DataFrame()
    
    # Create tabs for long and short recommendations
    long_tab, short_tab = st.tabs(["🟢 Long Recommendations", "🔴 Short Recommendations"])
    
    with long_tab:
        if not all_long_signals.empty:
            st.write("**Top Long Stock Picks Based on Scanner Signals:**")
            
            analyzed_long = analyze_stocks_with_prices(all_long_signals, 'long')
            
            if not analyzed_long.empty:
                # Display top 10 long picks
                top_long = analyzed_long.head(10)
                
                # Format the display
                display_long = top_long.copy()
                display_long['Current_Price'] = display_long['Current_Price'].apply(lambda x: f"${x:.2f}")
                display_long['Volume_Ratio'] = display_long['Volume_Ratio'].apply(lambda x: f"{x:.1f}x")
                display_long['Price_Change_1D'] = display_long['Price_Change_1D'].apply(lambda x: f"{x:+.1f}%")
                display_long['Price_Change_5D'] = display_long['Price_Change_5D'].apply(lambda x: f"{x:+.1f}%")
                display_long['Price_vs_SMA20'] = display_long['Price_vs_SMA20'].apply(lambda x: f"{x:+.1f}%")
                display_long['Volatility'] = display_long['Volatility'].apply(lambda x: f"{x:.1f}%")
                display_long['Distance_52W_Low'] = display_long['Distance_52W_Low'].apply(lambda x: f"{x:+.1f}%")
                display_long['Market_Cap'] = display_long['Market_Cap'].apply(lambda x: f"${x/1e9:.1f}B" if x > 1e9 else f"${x/1e6:.1f}M")
                
                st.dataframe(display_long[['Ticker', 'Score', 'Current_Price', 'Volume_Ratio', 'Price_Change_1D', 
                                         'Price_Change_5D', 'Price_vs_SMA20', 'Volatility', 'Distance_52W_Low', 
                                         'Market_Cap', 'Sector', 'Signal']], use_container_width=True)
                
                # Download button
                csv = analyzed_long.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Long Recommendations",
                    data=csv,
                    file_name=f"long_recommendations_{datetime.date.today()}.csv",
                    mime="text/csv",
                )
            else:
                st.warning("No long signals found with valid price data.")
        else:
            st.warning("No long signals found in the specified time period.")
    
    with short_tab:
        if not all_short_signals.empty:
            st.write("**Top Short Stock Picks Based on Scanner Signals:**")
            
            analyzed_short = analyze_stocks_with_prices(all_short_signals, 'short')
            
            if not analyzed_short.empty:
                # Display top 10 short picks
                top_short = analyzed_short.head(10)
                
                # Format the display
                display_short = top_short.copy()
                display_short['Current_Price'] = display_short['Current_Price'].apply(lambda x: f"${x:.2f}")
                display_short['Volume_Ratio'] = display_short['Volume_Ratio'].apply(lambda x: f"{x:.1f}x")
                display_short['Price_Change_1D'] = display_short['Price_Change_1D'].apply(lambda x: f"{x:+.1f}%")
                display_short['Price_Change_5D'] = display_short['Price_Change_5D'].apply(lambda x: f"{x:+.1f}%")
                display_short['Price_vs_SMA20'] = display_short['Price_vs_SMA20'].apply(lambda x: f"{x:+.1f}%")
                display_short['Volatility'] = display_short['Volatility'].apply(lambda x: f"{x:.1f}%")
                display_short['Distance_52W_High'] = display_short['Distance_52W_High'].apply(lambda x: f"{x:+.1f}%")
                display_short['Market_Cap'] = display_short['Market_Cap'].apply(lambda x: f"${x/1e9:.1f}B" if x > 1e9 else f"${x/1e6:.1f}M")
                
                st.dataframe(display_short[['Ticker', 'Score', 'Current_Price', 'Volume_Ratio', 'Price_Change_1D', 
                                          'Price_Change_5D', 'Price_vs_SMA20', 'Volatility', 'Distance_52W_High', 
                                          'Market_Cap', 'Sector', 'Signal']], use_container_width=True)
                
                # Download button
                csv = analyzed_short.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Short Recommendations",
                    data=csv,
                    file_name=f"short_recommendations_{datetime.date.today()}.csv",
                    mime="text/csv",
                )
            else:
                st.warning("No short signals found with valid price data.")
        else:
            st.warning("No short signals found in the specified time period.")

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
        
        # Scoring explanation
        st.subheader("📊 Scoring Methodology")
        st.markdown("""
        **Long Signals Score:**
        - Volume ratio > 2x: +20 points
        - Positive 1D momentum: +10-15 points
        - Above SMA20/50: +10 points each
        - Near 52W low: +10-15 points
        - Market cap > $1B: +5-10 points
        
        **Short Signals Score:**
        - Volume ratio > 2x: +20 points
        - Negative 1D momentum: +10-15 points
        - Below SMA20/50: +10 points each
        - Near 52W high: +10-15 points
        - Market cap > $1B: +5-10 points
        """)

    # Header with refresh button
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.title("📈 ThinkOrSwim Email Alerts Dashboard")
    with col2:
        if st.button("🔄 Refresh Data"):
            st.session_state.cached_data.clear()
            st.session_state.processed_email_ids.clear()
            st.cache_data.clear()
            st.rerun()

    # Auto-refresh logic
    if auto_refresh and st.session_state.last_refresh_time:
        time_since_refresh = time.time() - st.session_state.last_refresh_time
        if time_since_refresh >= refresh_interval * 60:
            st.session_state.cached_data.clear()
            st.session_state.processed_email_ids.clear()
            st.cache_data.clear()
            st.session_state.last_refresh_time = time.time()
            st.rerun()

    # Scan type selection
    section = st.radio("Select View", ["Recommendations", "Lower_timeframe", "Daily", "High Conviction"], 
                      index=0, horizontal=True)
    
    if section == "Recommendations":
        render_recommendations_section(days_lookback)
        
    elif section == "Lower_timeframe":
        st.subheader("Lower Timeframe Scans")
        for keyword in Lower_timeframe_KEYWORDS:
            render_stock_section(keyword, days_lookback)
            
    elif section == "Daily":
        st.subheader("Daily Scans")
        for keyword in DAILY_KEYWORDS:
            render_stock_section(keyword, days_lookback)
            
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
