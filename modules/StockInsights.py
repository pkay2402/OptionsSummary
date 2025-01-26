import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Function to fetch stock data
def fetch_stock_data(symbol, period="1y"):
    """Fetch historical data for a stock using yfinance."""
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period=period)
        info = stock.info
        return data, info
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {e}")
        return None, None

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
        st.metric("Market Cap", f"${info.get('marketCap', 'N/A'):,}")
    with col3:
        st.metric("Current Price", f"${info.get('currentPrice', 'N/A'):.2f}")
        
        # Handle cases where 'regularMarketChangePercent' is None or not a number
        daily_change = info.get('regularMarketChangePercent')
        if daily_change is not None and isinstance(daily_change, (int, float)):
            st.metric("Daily Change", f"{daily_change:.2f}%")
        else:
            st.metric("Daily Change", "N/A")

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

        # Plot historical price chart
        st.write("#### Historical Price Chart")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data["Close"], mode="lines", name="Close Price"))
        fig.update_layout(title="Stock Price Over Time", xaxis_title="Date", yaxis_title="Price (USD)")
        st.plotly_chart(fig)
    else:
        st.warning("No performance data available.")

# Function to display key statistics
def display_key_statistics(info):
    """Display key statistics about the stock."""
    st.write("### Key Statistics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("P/E Ratio", info.get("trailingPE", "N/A"))
        st.metric("EPS", info.get("trailingEps", "N/A"))
    with col2:
        st.metric("Dividend Yield", f"{info.get('dividendYield', 'N/A') * 100:.2f}%")
        st.metric("Beta", info.get("beta", "N/A"))
    with col3:
        st.metric("52-Week High", f"${info.get('fiftyTwoWeekHigh', 'N/A'):.2f}")
        st.metric("52-Week Low", f"${info.get('fiftyTwoWeekLow', 'N/A'):.2f}")

# Function to display technical indicators
def display_technical_indicators(data):
    """Display technical indicators (moving averages, RSI, etc.)."""
    st.write("### Technical Indicators")
    if data is not None and len(data) > 0:
        # Calculate moving averages
        data["50-Day MA"] = data["Close"].rolling(window=50).mean()
        data["200-Day MA"] = data["Close"].rolling(window=200).mean()

        # Plot moving averages
        st.write("#### Moving Averages")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data["Close"], mode="lines", name="Close Price"))
        fig.add_trace(go.Scatter(x=data.index, y=data["50-Day MA"], mode="lines", name="50-Day MA"))
        fig.add_trace(go.Scatter(x=data.index, y=data["200-Day MA"], mode="lines", name="200-Day MA"))
        fig.update_layout(title="Moving Averages", xaxis_title="Date", yaxis_title="Price (USD)")
        st.plotly_chart(fig)
    else:
        st.warning("No technical data available.")

# Function to display news and sentiment
def display_news_and_sentiment(symbol):
    """Display latest news and sentiment analysis."""
    st.write("### News and Sentiment")
    try:
        stock = yf.Ticker(symbol)
        news = stock.news
        if news:
            valid_news_count = 0
            for item in news[:5]:  # Show top 5 news items
                # Check if the news item has valid data
                title = item.get('title')
                publisher = item.get('publisher')
                link = item.get('link')
                
                if title and publisher and link:  # Only display if all fields are present
                    st.write(f"**{title}**")
                    st.write(f"*{publisher}* - [Read more]({link})")
                    valid_news_count += 1

            if valid_news_count == 0:
                st.warning("No valid news items found.")
        else:
            st.warning("No news available.")
    except Exception as e:
        st.error(f"Error fetching news: {e}")

# Main function to run the Stock Insights module
def run():
    """Main function to run the Stock Insights module."""
    st.title("Stock Insights")

    # User input for stock symbol
    symbol = st.text_input("Enter a stock symbol (e.g., AAPL):", "AAPL").upper()

    if symbol:
        # Fetch stock data
        data, info = fetch_stock_data(symbol)
        if data is not None and info is not None:
            # Display stock insights
            display_stock_overview(info)
            display_performance_metrics(data)
            display_key_statistics(info)
            display_technical_indicators(data)
            display_news_and_sentiment(symbol)
        else:
            st.error("Invalid stock symbol or no data available.")
