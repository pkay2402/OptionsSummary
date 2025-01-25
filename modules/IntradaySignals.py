import pandas as pd
import numpy as np
import yfinance as yf
import time  # Standard time module
from datetime import datetime, time as dt_time
import requests
import pytz

# Parameters
SYMBOLS = ["SPY", "QQQ", "NVDA", "TSLA"]
INTERVAL = "30m"
DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1332367135023956009/8HH_RiKnSP7R7l7mtFHOB8kJi7ATt0TKZRrh35D82zycKC7JrFVSMpgJUmHrnDQ4mQRw"
LENGTH = 14
CALC_LENGTH = 5
SMOOTH_LENGTH = 3

def run():
    import streamlit as st
    st.title("Intraday Signals")
    main()  # Call the main function here to run the app

# Calculate EMA
def calculate_ema(data, period):
    return data.ewm(span=period, adjust=False).mean()

# Fetch stock data
def fetch_stock_data(symbol, interval, period="1d"):
    try:
        data = yf.download(symbol, period=period, interval=interval)
        if data.empty:
            print(f"No data received for {symbol} ({interval})")
            return pd.DataFrame()
        return data[['Open', 'High', 'Low', 'Close', 'Volume']]
    except Exception as e:
        print(f"Error fetching data for {symbol} ({interval}): {e}")
        return pd.DataFrame()

# Calculate buy/sell signals
def calculate_signals(stock_data):
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

# Send alerts to Discord
def send_to_discord(message):
    payload = {"content": message}
    try:
        response = requests.post(DISCORD_WEBHOOK_URL, json=payload)
        response.raise_for_status()
        print(f"Message sent to Discord: {message}")
    except requests.exceptions.RequestException as e:
        print(f"Error sending to Discord: {e}")

# Check market hours
def is_market_open():
    """Check if the current time is within market hours (Monday to Friday, 9:30 AM to 4:00 PM EST)."""
    tz = pytz.timezone('US/Eastern')  # Set timezone to Eastern Time
    now = datetime.now(tz)
    market_open = dt_time(9, 30)
    market_close = dt_time(16, 0)

    if now.weekday() >= 5:  # Check if it's Saturday (5) or Sunday (6)
        return False

    return market_open <= now.time() <= market_close

# Main function
def main():
    last_signals = {}
    market_status_sent = False  # Flag to track if the "Market is closed" message was sent

    while True:
        if not is_market_open():
            if not market_status_sent:
                print("Market is closed!")
                send_to_discord("Market is currently closed. Signals will resume during market hours.")
                market_status_sent = True
            time.sleep(900)  # Sleep for 15 minutes
            continue
        else:
            if market_status_sent:
                print("Market is now open!")
                send_to_discord("Market has opened. Starting signal detection!")
                market_status_sent = False

        # Fetch and process data only if the market is open
        print("Market is open! Fetching data...")
        for symbol in SYMBOLS:
            stock_data = fetch_stock_data(symbol, INTERVAL, period="1d")
            if stock_data.empty:
                continue

            buy_signals, sell_signals = calculate_signals(stock_data)

            if not buy_signals.empty and buy_signals.iloc[-1]:
                message = f"Buy signal detected for {symbol} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                send_to_discord(message)

            if not sell_signals.empty and sell_signals.iloc[-1]:
                message = f"Sell signal detected for {symbol} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                send_to_discord(message)

        print(f"Checked signals at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        time.sleep(900)  # Run every 15 minutes

# This part allows the script to be imported as a module or run directly
if __name__ == "__main__":
    run()
