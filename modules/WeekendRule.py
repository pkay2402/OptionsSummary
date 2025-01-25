import pandas as pd
import yfinance as yf
import streamlit as st

# Predefined list of top 100 US stocks by market cap (example)
TOP_100_STOCKS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "BRK-B", "JNJ", "V",
    "WMT", "XOM", "UNH", "JPM", "PG", "MA", "HD", "CVX", "LLY", "ABBV",
    "PFE", "MRK", "BAC", "KO", "PEP", "TMO", "AVGO", "COST", "DIS", "CSCO",
    "WFC", "ABT", "VZ", "ACN", "CMCSA", "ADBE", "CRM", "NFLX", "TXN", "DHR",
    "PM", "NKE", "LIN", "ORCL", "AMD", "QCOM", "INTU", "HON", "AMGN", "SBUX",
    "IBM", "T", "LOW", "MDT", "GS", "UNP", "CAT", "BLK", "AXP", "SPGI",
    "PYPL", "DE", "NOW", "PLD", "GE", "ISRG", "MMM", "BKNG", "ADI", "LMT",
    "RTX", "UPS", "SYK", "BA", "MDLZ", "SCHW", "ZTS", "CI", "GILD", "MO",
    "TMUS", "FIS", "CVS", "ANTM", "BDX", "CME", "CB", "SO", "DUK", "CL",
    "NEE", "APD", "PNC", "BSX", "CCI", "ICE", "AON", "SHW", "ITW", "ETN"
]

def fetch_weekly_data(symbol):
    """Fetch weekly historical data for a symbol using yfinance."""
    try:
        data = yf.download(symbol, period="1y", interval="1wk")  # Weekly data
        if data.empty:
            st.warning(f"No data received for {symbol}")
            return None
        return data
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {e}")
        return None

def apply_weekend_rule(data):
    """Apply Brandt's Weekend Rule to the weekly data."""
    if data is None or len(data) < 2:
        return None, None

    # Get the last two weeks' data
    last_week = data.iloc[-1]
    prev_week = data.iloc[-2]

    # Check for long and short setups
    long_setup = last_week["Close"] > prev_week["High"]
    short_setup = last_week["Close"] < prev_week["Low"]

    return long_setup, short_setup

def run():
    """Main function to run the Weekend Rule module."""
    st.title("Brandt's Weekend Rule")

    # Fetch data and apply the rule for each stock
    long_setups = []
    short_setups = []

    for symbol in TOP_100_STOCKS:
        data = fetch_weekly_data(symbol)
        if data is not None:
            long_setup, short_setup = apply_weekend_rule(data)
            if long_setup:
                long_setups.append(symbol)
            if short_setup:
                short_setups.append(symbol)

    # Display results
    if long_setups:
        st.write("### Long Setups (Friday's close > previous week's high)")
        st.write(pd.DataFrame(long_setups, columns=["Stock"]))
    else:
        st.write("No long setups found.")

    if short_setups:
        st.write("### Short Setups (Friday's close < previous week's low)")
        st.write(pd.DataFrame(short_setups, columns=["Stock"]))
    else:
        st.write("No short setups found.")

    st.write("This module implements Brandt's Weekend Rule for top 100 US stocks.")
