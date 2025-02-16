import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import json
import requests
from datetime import datetime, time
import pytz
from streamlit_autorefresh import st_autorefresh

# Parameters
length = 14
calc_length = 5
smooth_length = 3

# Webhook URL for Discord
DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1332367135023956009/8HH_RiKnSP7R7l7mtFHOB8kJi7ATt0TKZRrh35D82zycKC7JrFVSMpgJUmHrnDQ4mQRw"

# File to store last signals
last_signals_file = "last_signals.json"

def run():
    """Main function to run the Streamlit application."""
    st.title("Momentum ETF Signals")
    st.write("""
    Real-time stock momentum analysis tool that:
    - Tracks price movements across multiple timeframes
    - Calculates EMAs and pivot points
    - Generates buy/sell signals
    """)
    main()

def calculate_ema(data, period):
    """Calculate Exponential Moving Average (EMA) for a given period."""
    return data.ewm(span=period, adjust=False).mean()

def calculate_pivots(data, timeframe='D'):
    """Calculate pivots (Daily, Weekly, Monthly) based on High, Low, and Close prices."""
    # Flatten MultiIndex columns if they exist
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = ['_'.join(filter(None, col)) for col in data.columns]
    
    # Extract the ticker name from the columns
    ticker = [col.split('_')[1] for col in data.columns if '_' in col][0] if '_' in data.columns[0] else ''
    
    # Select relevant columns
    high_col = f'High_{ticker}' if ticker and f'High_{ticker}' in data.columns else 'High'
    low_col = f'Low_{ticker}' if ticker and f'Low_{ticker}' in data.columns else 'Low'
    close_col = f'Close_{ticker}' if ticker and f'Close_{ticker}' in data.columns else 'Close'
    
    # Ensure columns exist
    if not all(col in data.columns for col in [high_col, low_col, close_col]):
        raise KeyError(f"Required columns missing!")
    
    # Select and rename columns
    data = data[[high_col, low_col, close_col]]
    data = data.rename(columns={high_col: 'High', low_col: 'Low', close_col: 'Close'})
    
    now = datetime.now()
    
    if timeframe == 'D':
        # Daily pivot - uses previous day's data
        data = data.iloc[-2:-1] if len(data) > 1 else data.iloc[-1:]
    elif timeframe == 'W':
        # Weekly pivot - current week's data
        data = data[data.index.isocalendar().week == now.isocalendar().week]
    else:  # Monthly
        data = data[(data.index.month == now.month) & (data.index.year == now.year)]
    
    if data.empty:
        return None
        
    high = data['High'].max()
    low = data['Low'].min()
    close = data['Close'].iloc[-1]
    pivot = (high + low + close) / 3
    
    return pivot

@st.cache_data
def fetch_stock_data(symbol, interval, period="6mo"):
    """Fetch stock data using yfinance."""
    try:
        data = yf.download(symbol, period=period, interval=interval)
        if data.empty:
            st.warning(f"No data received for {symbol} ({interval})")
            return pd.DataFrame()
        return data[['Open', 'High', 'Low', 'Close', 'Volume']]
    except Exception as e:
        st.error(f"Error fetching data for {symbol} ({interval}): {e}")
        return pd.DataFrame()

def fetch_latest_price(symbol):
    """Fetch the latest price of the stock."""
    try:
        ticker = yf.Ticker(symbol)
        price = ticker.history(period="1d")['Close'].iloc[-1]
        return price
    except Exception as e:
        st.error(f"Error fetching latest price for {symbol}: {e}")
        return None

def calculate_signals(stock_data):
    """Calculate buy/sell signals."""
    if stock_data.empty or len(stock_data) < length + smooth_length * 2:
        return pd.Series(dtype=bool), pd.Series(dtype=bool)

    o = stock_data['Open'].values
    c = stock_data['Close'].values

    data = np.array([sum(np.sign(c[i] - o[max(0, i - j)]) for j in range(length)) for i in range(len(c))]).flatten()
    data_series = pd.Series(data, index=stock_data.index)

    EMA5 = data_series.ewm(span=calc_length, adjust=False).mean()
    Main = EMA5.ewm(span=smooth_length, adjust=False).mean()
    Signal = Main.ewm(span=smooth_length, adjust=False).mean()

    buy_signals = (Main > Signal) & (Main.shift(1) <= Signal)
    sell_signals = (Main < Signal) & (Main.shift(1) >= Signal)

    return buy_signals, sell_signals

def analyze_stock(symbol, timeframes):
    """Analyze a stock across the specified timeframes."""
    results = {}
    for timeframe in timeframes:
        stock_data = fetch_stock_data(symbol, timeframe)
        if stock_data.empty:
            results[timeframe] = "No Data"
        else:
            buy_signals, sell_signals = calculate_signals(stock_data)
            if not buy_signals.empty and buy_signals.iloc[-1]:
                results[timeframe] = "B"
            elif not sell_signals.empty and sell_signals.iloc[-1]:
                results[timeframe] = "S"
            else:
                results[timeframe] = "N"
    return results

def calculate_indicators(data):
    """Calculate EMAs and Pivot values."""
    data['EMA_9'] = calculate_ema(data['Close'], 9)
    data['EMA_21'] = calculate_ema(data['Close'], 21)
    data['EMA_50'] = calculate_ema(data['Close'], 50)
    data['EMA_200'] = calculate_ema(data['Close'], 200)
    
    daily_pivot = calculate_pivots(data, 'D')
    weekly_pivot = calculate_pivots(data, 'W')
    monthly_pivot = calculate_pivots(data, 'M')
    
    return data, daily_pivot, weekly_pivot, monthly_pivot

def save_signals(data):
    """Save the current signals to a JSON file."""
    with open(last_signals_file, 'w') as f:
        json.dump(data, f)

def load_signals():
    """Load the previously saved signals from JSON file."""
    try:
        with open(last_signals_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def send_to_discord(message, table=None):
    """Send message and table to a Discord webhook."""
    payload = {
        "content": message
    }

    if table is not None:
        payload["content"] += "\n\n```" + table + "```"
    
    try:
        response = requests.post(DISCORD_WEBHOOK_URL, json=payload)
        response.raise_for_status()
        st.success(f"Message sent to Discord: {message}")
    except requests.exceptions.RequestException as e:
        st.error(f"Error sending to Discord: {e}")

def df_to_markdown(df):
    """Convert a DataFrame to a markdown table format."""
    return df.to_markdown(index=False)

def is_market_open():
    """Check if the current time is within market hours (Monday to Friday, 9:30 AM to 4:00 PM EST)."""
    tz = pytz.timezone('US/Eastern')
    now = datetime.now(tz)
    market_open = time(9, 30)
    market_close = time(16, 0)

    if now.weekday() >= 5:  # Saturday or Sunday
        return False

    if market_open <= now.time() <= market_close:
        return True

    return False

def main():
    """Main function to run the Momentum Signals app."""
    # Refresh the app every 4 hours (14400000 milliseconds)
    st_autorefresh(interval=14400000, key="data_refresh")

    # Add a toggle to bypass market hours check
    bypass_market_check = st.checkbox("Bypass Market Hours Check (View Data Even When Market is Closed)")

    # Check if the market is open or if the user has chosen to bypass the check
    if not is_market_open() and not bypass_market_check:
        st.write("Market is currently closed. The app will resume during market hours.")
        return

    st.write(f"App refreshed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    st.title("ETF Summary with 1D and 5D")

    # Symbols and timeframe
    symbols = [
        "XLC",  # Communication Services
        "XLY",  # Consumer Discretionary
        "XLP",  # Consumer Staples
        "XLE",  # Energy
        "XLF",  # Financials
        "XLV",  # Health Care
        "XLI",  # Industrials
        "XLB",  # Materials
        "XLK",  # Technology
        "SPY",  # S&P 500
        "QQQ",  # NASDAQ-100
    ]
    timeframes = ["1d", "5d"]

    # Data storage
    rows = []
    trend_changes = []
    current_signals = {}

    # Load the last saved signals
    last_signals = load_signals()

    # Analyze each symbol
    for symbol in symbols:
        stock_data = fetch_stock_data(symbol, "1d")
        if stock_data.empty:
            continue

        # Calculate indicators and pivots
        stock_data, daily_pivot, weekly_pivot, monthly_pivot = calculate_indicators(stock_data)

        # Get the latest price and signals
        latest_price = fetch_latest_price(symbol)
        analysis = analyze_stock(symbol, timeframes)

        row = {
            "Symbol": symbol,
            "Price": latest_price,
            "1D": analysis.get("1d", "Error"),
            "5D": analysis.get("5d", "Error"),
            "EMA_9": stock_data['EMA_9'].iloc[-1] if not stock_data.empty else None,
            "EMA_21": stock_data['EMA_21'].iloc[-1] if not stock_data.empty else None,
            "EMA_50": stock_data['EMA_50'].iloc[-1] if not stock_data.empty else None,
            "EMA_200": stock_data['EMA_200'].iloc[-1] if not stock_data.empty else None,
            "Daily_Pivot": daily_pivot,
            "Weekly_Pivot": weekly_pivot,
            "Monthly_Pivot": monthly_pivot
        }
        rows.append(row)

        # Store the current signal
        current_signals[symbol] = analysis.get("1d", "Error")

        # Compare current signals with last signals
        current_signal = current_signals[symbol]
        last_signal = last_signals.get(symbol, "Neutral")

        # Detect signal change
        if current_signal != last_signal:
            trend_changes.append(f"Signal change for {symbol}: {last_signal} -> {current_signal}")

    # Display the current signals in the Streamlit app
    # Display the current signals in the Streamlit app
    df = pd.DataFrame(rows)
    
    # Define the styling function
    def highlight_signals(row):
        if row['1D'] == 'B' and row['5D'] == 'B':
            return ['background-color: rgba(0, 100, 0, 0.3)'] * len(row)  # Dark green for buy signals
        elif row['1D'] == 'S' and row['5D'] == 'S':
            return ['background-color: rgba(255, 0, 0, 0.3)'] * len(row)  # Red for sell signals
        return [''] * len(row)
    
    # Apply the styling
    styled_df = df.style.apply(highlight_signals, axis=1)
    
    st.write("Current Signals and Indicators")
    st.dataframe(styled_df)

    # Add a manual button to send the table to Discord
    if st.button("Send Table to Discord"):
        table = df_to_markdown(df)
        message = "Manual Push of Signals and Indicators to Discord"
        send_to_discord(message, table)
        st.write("Table sent to Discord manually.")

    # Save the current signals for the next comparison
    save_signals(current_signals)

if __name__ == "__main__":
    run()
