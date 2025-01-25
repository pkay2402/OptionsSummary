import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.express as px
from datetime import datetime, timedelta

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

# Sector ETFs for Sector Strength Scanner
SECTOR_ETFS = {
    "Technology": "XLK",
    "Financials": "XLF",
    "Healthcare": "XLV",
    "Energy": "XLE",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Utilities": "XLU",
    "Industrials": "XLI",
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Communication Services": "XLC"
}

def fetch_data(symbol, period="1mo", interval="1d"):
    """Fetch historical data for a symbol using yfinance."""
    try:
        data = yf.download(symbol, period=period, interval=interval)
        if data.empty:
            st.warning(f"No data received for {symbol}")
            return None
        return data
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {e}")
        return None

def volume_spike_scanner():
    """Identify stocks with unusual volume spikes."""
    st.write("### Volume Spike Scanner")
    volume_spikes = []

    for symbol in TOP_100_STOCKS:
        data = fetch_data(symbol)
        if data is not None:
            avg_volume = data["Volume"].rolling(window=20).mean().iloc[-1]
            last_volume = data["Volume"].iloc[-1]
            if last_volume > 2 * avg_volume:
                volume_spikes.append(symbol)

    if volume_spikes:
        st.write("Stocks with volume spikes (volume > 2x average):")
        st.write(pd.DataFrame(volume_spikes, columns=["Stock"]))
    else:
        st.write("No volume spikes found.")

def breakout_scanner():
    """Identify stocks breaking out of key support/resistance levels."""
    st.write("### Breakout Scanner")
    breakouts = []

    for symbol in TOP_100_STOCKS:
        data = fetch_data(symbol)
        if data is not None and len(data) > 20:
            resistance = data["High"].rolling(window=20).max().iloc[-2]
            support = data["Low"].rolling(window=20).min().iloc[-2]
            last_close = data["Close"].iloc[-1]

            if last_close > resistance:
                breakouts.append((symbol, "Bullish Breakout"))
            elif last_close < support:
                breakouts.append((symbol, "Bearish Breakout"))

    if breakouts:
        st.write("Stocks breaking out of key levels:")
        st.write(pd.DataFrame(breakouts, columns=["Stock", "Breakout Type"]))
    else:
        st.write("No breakouts found.")

def sector_strength_scanner():
    """Identify the strongest and weakest sectors."""
    st.write("### Sector Strength Scanner")
    sector_performance = []

    for sector, etf in SECTOR_ETFS.items():
        data = fetch_data(etf, period="1mo")
        if data is not None:
            performance = (data["Close"].iloc[-1] - data["Close"].iloc[0]) / data["Close"].iloc[0] * 100
            sector_performance.append((sector, performance))

    sector_performance_df = pd.DataFrame(sector_performance, columns=["Sector", "Performance (%)"])
    sector_performance_df = sector_performance_df.sort_values(by="Performance (%)", ascending=False)

    st.write("Sector Performance (Last 1 Month):")
    st.write(sector_performance_df)

    # Plot sector performance
    fig = px.bar(sector_performance_df, x="Sector", y="Performance (%)", title="Sector Performance")
    st.plotly_chart(fig)

def seasonality_scanner():
    """Identify stocks with strong seasonal patterns."""
    st.write("### Seasonality Scanner")
    seasonal_stocks = []

    for symbol in TOP_100_STOCKS:
        data = fetch_data(symbol, period="1y")
        if data is not None:
            # Example: Check if the stock tends to perform well in December
            december_returns = data[data.index.month == 12]["Close"].pct_change().mean()
            if december_returns > 0.05:  # 5% average return in December
                seasonal_stocks.append((symbol, "December"))

    if seasonal_stocks:
        st.write("Stocks with strong seasonal patterns:")
        st.write(pd.DataFrame(seasonal_stocks, columns=["Stock", "Season"]))
    else:
        st.write("No seasonal patterns found.")

def candlestick_pattern_scanner():
    """Identify stocks with bullish or bearish candlestick patterns."""
    st.write("### Candlestick Pattern Scanner")
    bullish_stocks = []
    bearish_stocks = []

    for symbol in TOP_100_STOCKS:
        data = fetch_data(symbol)
        if data is not None and len(data) > 1:
            # Example: Check for bullish engulfing pattern
            prev_open = data["Open"].iloc[-2]
            prev_close = data["Close"].iloc[-2]
            curr_open = data["Open"].iloc[-1]
            curr_close = data["Close"].iloc[-1]

            if prev_close < prev_open and curr_close > curr_open and curr_close > prev_open:
                bullish_stocks.append(symbol)
            # Example: Check for bearish engulfing pattern
            elif prev_close > prev_open and curr_close < curr_open and curr_close < prev_open:
                bearish_stocks.append(symbol)

    if bullish_stocks:
        st.write("Stocks with bullish candlestick patterns:")
        st.write(pd.DataFrame(bullish_stocks, columns=["Stock"]))
    else:
        st.write("No bullish candlestick patterns found.")

    if bearish_stocks:
        st.write("Stocks with bearish candlestick patterns:")
        st.write(pd.DataFrame(bearish_stocks, columns=["Stock"]))
    else:
        st.write("No bearish candlestick patterns found.")

def run():
    """Main function to run the Multi-Scanner module."""
    st.title("Multi-Scanner")

    # Radio buttons for scanner selection
    scanner_choice = st.radio(
        "Choose a scanner",
        ["Volume Spike Scanner", "Breakout Scanner", "Sector Strength Scanner", "Seasonality Scanner", "Candlestick Pattern Scanner"]
    )

    if scanner_choice == "Volume Spike Scanner":
        volume_spike_scanner()
    elif scanner_choice == "Breakout Scanner":
        breakout_scanner()
    elif scanner_choice == "Sector Strength Scanner":
        sector_strength_scanner()
    elif scanner_choice == "Seasonality Scanner":
        seasonality_scanner()
    elif scanner_choice == "Candlestick Pattern Scanner":
        candlestick_pattern_scanner()

    st.write("This module combines multiple scanners to help you find trading opportunities.")
