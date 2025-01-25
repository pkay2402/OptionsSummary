import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import requests
from datetime import datetime, time as dt_time
import pytz
from streamlit_autorefresh import st_autorefresh

# Parameters
SYMBOLS = ["SPY", "QQQ", "NVDA", "TSLA"]
INTERVAL = "30m"
DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1332367135023956009/8HH_RiKnSP7R7l7mtFHOB8kJi7ATt0TKZRrh35D82zycKC7JrFVSMpgJUmHrnDQ4mQRw"
LENGTH = 14
CALC_LENGTH = 5
SMOOTH_LENGTH = 3

def run():
    """Main function to run the Streamlit application."""
    st.title("Intraday Signals")
    main()

def calculate_ema(data, period):
    """Calculate Exponential Moving Average (EMA) for a given period."""
    return data.ewm(span=period, adjust=False).mean()

@st.cache_data
def fetch_stock_data(symbol, interval, period="1d"):
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

def calculate_signals(stock_data):
    """Calculate buy/sell signals."""
    if stock_data.empty or len(stock_data) < LENGTH + SMOOTH_LENGTH * 2:
        return pd.Series(dtype=bool), pd.Series(dtype=bool)

    o = stock_data['Open'].values
    c = stock_data['Close'].values

    data = np.array([sum(np.sign(c[i] - o[max(0, i - j)]) for j in range(LENGTH)) for i in range(len(c))]).flatten()
    data_series = pd.Series(data, index=stock_data.index)

    EMA5 = data_series.ewm(span=CALC_LENGTH, adjust=False).mean()
    Main = EMA5.ewm(span=SMOOTH_LENGTH, adjust=False).mean()
    Signal = Main.ewm(span=SMOOTH_LENGTH, adjust=False).mean()

    buy_signals = (Main > Signal) & (Main.shift(1) <= Signal)
    sell_signals = (Main < Signal) & (Main.shift(1) >= Signal)

    return buy_signals, sell_signals

def send_to_discord(message):
    """Send message to a Discord webhook."""
    payload = {"content": message}
    try:
        response = requests.post(DISCORD_WEBHOOK_URL, json=payload)
        response.raise_for_status()
        st.success(f"Message sent to Discord: {message}")
    except requests.exceptions.RequestException as e:
        st.error(f"Error sending to Discord: {e}")

def is_market_open():
    """Check if the current time is within market hours (Monday to Friday, 9:30 AM to 4:00 PM EST)."""
    tz = pytz.timezone('US/Eastern')
    now = datetime.now(tz)
    market_open = dt_time(9, 30)
    market_close = dt_time(16, 0)

    if now.weekday() >= 5:  # Saturday or Sunday
        return False

    return market_open <= now.time() <= market_close

def main():
    """Main function to run the Intraday Signals app."""
    # Refresh the app every 15 minutes (900000 milliseconds)
    st_autorefresh(interval=900000, key="data_refresh")

    # Add a toggle to bypass market hours check
    bypass_market_check = st.checkbox("Bypass Market Hours Check (View Data Even When Market is Closed)")

    # Check if the market is open or if the user has chosen to bypass the check
    if not is_market_open() and not bypass_market_check:
        st.write("Market is currently closed. The app will resume during market hours.")
        return

    st.write(f"App refreshed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Fetch and process data
    for symbol in SYMBOLS:
        stock_data = fetch_stock_data(symbol, INTERVAL, period="1d")
        if stock_data.empty:
            continue

        buy_signals, sell_signals = calculate_signals(stock_data)

        if not buy_signals.empty and buy_signals.iloc[-1]:
            message = f"Buy signal detected for {symbol} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            st.success(message)
            send_to_discord(message)

        if not sell_signals.empty and sell_signals.iloc[-1]:
            message = f"Sell signal detected for {symbol} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            st.error(message)
            send_to_discord(message)

    st.write("This is the Intraday Signals application.")

# This part allows the script to be imported as a module or run directly
if __name__ == "__main__":
    run()
