import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Function to convert month name to month number
def get_month_number(month_name):
    return {
        "January": 1, "February": 2, "March": 3, "April": 4,
        "May": 5, "June": 6, "July": 7, "August": 8,
        "September": 9, "October": 10, "November": 11, "December": 12
    }[month_name]

# Fetch historical stock data from Yahoo Finance
def fetch_stock_data(symbol, start_year):
    try:
        data = yf.download(symbol, start=f"{start_year}-01-01", end="2025-01-01")
        data["Day"] = data.index.day
        data["Month"] = data.index.month
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# Calculate daily average returns for a specific month
def calculate_seasonality(data, month_number):
    data["Daily Return"] = data["Close"].pct_change()
    monthly_data = data[data["Month"] == month_number]
    daily_avg_returns = monthly_data.groupby("Day")["Daily Return"].mean()
    monthly_avg_return = daily_avg_returns.mean()
    return daily_avg_returns, monthly_avg_return

# Plot the seasonality chart
def plot_seasonality_chart(stock, month_name, daily_avg_returns):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(daily_avg_returns.index, daily_avg_returns.values, color="green")
    ax.set_title(f"{stock.upper()} Seasonality Chart - {month_name}")
    ax.set_xlabel("Day of the Month")
    ax.set_ylabel("Average Return (%)")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    st.pyplot(fig)

# Streamlit App
def main():
    st.title("Stock Seasonality Analysis by Day")
    
    # Inputs for stock and month
    stock = st.text_input("Enter the stock symbol (e.g., TSLA):").strip()
    month_name = st.selectbox("Select a month:", [
        "January", "February", "March", "April", "May", "June", 
        "July", "August", "September", "October", "November", "December"
    ])
    
    if stock and month_name:
        # Get month number
        month_number = get_month_number(month_name)
        
        # Fetch stock data
        start_year = 2011
        data = fetch_stock_data(stock, start_year)
        
        if data is not None:
            # Calculate seasonality
            daily_avg_returns, monthly_avg_return = calculate_seasonality(data, month_number)
            
            # Display average return for the month
            st.header(f"Average Return for {month_name}: {monthly_avg_return:.2%}")
            
            # Plot the seasonality chart
            if not daily_avg_returns.empty:
                plot_seasonality_chart(stock, month_name, daily_avg_returns)
            else:
                st.warning(f"No data available for {month_name}.")
        else:
            st.error("Failed to fetch stock data. Please check the stock symbol and try again.")

if __name__ == "__main__":
    main()
