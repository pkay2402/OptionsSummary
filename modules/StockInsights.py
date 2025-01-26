import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime

# Function to fetch stock data
@st.cache_data
def fetch_stock_data(symbol, period="1y"):
    """Fetch historical data for a stock using yfinance."""
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period=period)
        if data.empty:
            st.error(f"No data found for {symbol}. Please check the symbol and try again.")
            return None, None
        info = stock.info
        return data, info
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {e}")
        return None, None

# Function to calculate Fibonacci retracement levels
def calculate_fibonacci_levels(high, low):
    """Calculate Fibonacci retracement levels."""
    diff = high - low
    return {
        "23.6%": high - 0.236 * diff,
        "38.2%": high - 0.382 * diff,
        "50%": high - 0.5 * diff,
        "61.8%": high - 0.618 * diff,
        "78.6%": high - 0.786 * diff,
    }

# Function to display stock overview
def display_stock_overview(info):
    """Display basic stock information."""
    st.write("### Stock Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Company", info.get("longName", "N/A"))
        st.metric("Sector", info.get("sector", "N/A"))
    with col2:
        st.metric("Industry", info.get("industry", "N/A"))
        # Handle missing market cap
        market_cap = info.get("marketCap")
        if market_cap is not None:
            st.metric("Market Cap", f"${market_cap:,}")
        else:
            st.metric("Market Cap", "N/A")
    with col3:
        # Handle missing current price
        current_price = info.get("currentPrice")
        if current_price is not None:
            st.metric("Current Price", f"${current_price:.2f}")
        else:
            st.metric("Current Price", "N/A")

        # Handle missing 52-week high and low
        fifty_two_week_high = info.get("fiftyTwoWeekHigh")
        fifty_two_week_low = info.get("fiftyTwoWeekLow")
        st.metric("52-Week High", f"${fifty_two_week_high:.2f}" if fifty_two_week_high is not None else "N/A")
        st.metric("52-Week Low", f"${fifty_two_week_low:.2f}" if fifty_two_week_low is not None else "N/A")

# Function to display performance metrics
def display_performance_metrics(data):
    """Display performance metrics (daily, weekly, monthly, yearly)."""
    st.write("### Performance Metrics")
    if data is not None and len(data) > 0:
        # Calculate performance
        daily_return = (data["Close"].iloc[-1] - data["Close"].iloc[-2]) / data["Close"].iloc[-2] * 100
        weekly_return = (data["Close"].iloc[-1] - data["Close"].iloc[-5]) / data["Close"].iloc[-5] * 100
        monthly_return = (data["Close"].iloc[-1] - data["Close"].iloc[-22]) / data["Close"].iloc[-22] * 100
        yearly_return = (data["Close"].iloc[-1] - data["Close"].iloc[0]) / data["Close"].iloc[0] * 100

        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Daily Return", f"{daily_return:.2f}%")
        col2.metric("Weekly Return", f"{weekly_return:.2f}%")
        col3.metric("Monthly Return", f"{monthly_return:.2f}%")
        col4.metric("Yearly Return", f"{yearly_return:.2f}%")

# Function to display the combined chart
def display_combined_chart(data, info):
    """Display a combined chart with price, moving averages, and Fibonacci levels."""
    if data is not None and len(data) > 0:
        # Calculate moving averages
        data["50-Day MA"] = data["Close"].rolling(window=50).mean()
        data["200-Day MA"] = data["Close"].rolling(window=200).mean()

        # Calculate Fibonacci levels
        high = info.get("fiftyTwoWeekHigh")
        low = info.get("fiftyTwoWeekLow")
        if high and low:
            fib_levels = calculate_fibonacci_levels(high, low)
        else:
            fib_levels = None

        # Plot the chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data["Close"], mode="lines", name="Close Price"))
        fig.add_trace(go.Scatter(x=data.index, y=data["50-Day MA"], mode="lines", name="50-Day MA"))
        fig.add_trace(go.Scatter(x=data.index, y=data["200-Day MA"], mode="lines", name="200-Day MA"))

        # Add Fibonacci levels
        if fib_levels:
            for level, price in fib_levels.items():
                fig.add_trace(go.Scatter(
                    x=[data.index[0], data.index[-1]],
                    y=[price, price],
                    mode="lines",
                    line=dict(dash="dash"),
                    name=f"Fib {level}"
                ))

        fig.update_layout(
            title=f"{symbol} Stock Price with Moving Averages and Fibonacci Levels",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            legend_title="Indicators",
            hovermode="x unified"
        )
        st.plotly_chart(fig)
    else:
        st.warning("No data available for the chart.")

# Function to display key statistics
def display_key_statistics(info):
    """Display key statistics about the stock."""
    st.write("### Key Statistics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("P/E Ratio", info.get("trailingPE", "N/A"))
        st.metric("EPS", info.get("trailingEps", "N/A"))
    with col2:
        # Handle cases where 'dividendYield' is None or not a number
        dividend_yield = info.get("dividendYield")
        if dividend_yield is not None and isinstance(dividend_yield, (int, float)):
            st.metric("Dividend Yield", f"{dividend_yield * 100:.2f}%")
        else:
            st.metric("Dividend Yield", "N/A")
        st.metric("Beta", info.get("beta", "N/A"))
    with col3:
        st.metric("Volume", f"{info.get('volume', 'N/A'):,}")
        st.metric("Avg. Volume", f"{info.get('averageVolume', 'N/A'):,}")

# Main function to run the Stock Insights module
def run():
    """Main function to run the Stock Insights module."""
    st.title("Stock Insights")

    # Reduce font size for a cleaner look
    st.markdown("""
        <style>
        .stMetric {
            font-size: 14px;
        }
        </style>
    """, unsafe_allow_html=True)

    # User input for stock symbol
    symbol = st.text_input("Enter a stock symbol (e.g., AAPL):", "AAPL").upper()

    # Add a date range selector
    period_options = {"1 Month": "1mo", "3 Months": "3mo", "6 Months": "6mo", "1 Year": "1y", "2 Years": "2y", "5 Years": "5y"}
    selected_period = st.selectbox("Select a time period:", list(period_options.keys()), index=3)
    period = period_options[selected_period]

    if symbol:
        # Fetch stock data with a loading spinner
        with st.spinner("Fetching stock data..."):
            data, info = fetch_stock_data(symbol, period=period)

        if data is not None and info is not None:
            # Display stock insights
            display_stock_overview(info)
            display_performance_metrics(data)
            display_combined_chart(data, info)
            display_key_statistics(info)

            # Add a download button for the data
            st.download_button(
                label="Download Data as CSV",
                data=data.to_csv().encode("utf-8"),
                file_name=f"{symbol}_stock_data.csv",
                mime="text/csv"
            )
        else:
            st.error("Invalid stock symbol or no data available.")

# Run the app
if __name__ == "__main__":
    run()
