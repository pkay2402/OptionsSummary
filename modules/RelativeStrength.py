import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Function to fetch stock data with more detailed logging
def fetch_stock_data(symbol, start_date, end_date):
    try:
        data = yf.download(symbol, start=start_date, end=end_date)
        print(f"Data for {symbol}:")
        print(data.head())  # Shows the first few rows of the DataFrame
        print(data.columns)  # Lists all available columns
        if 'Close' not in data.columns:
            st.warning(f"'Close' column not found in data for {symbol}. Available columns are: {data.columns}")
        return data
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {e}")
        return None

# Function to calculate Relative Strength (RS)
def calculate_relative_strength(stock_data, benchmark_data):
    if stock_data is None or benchmark_data is None:
        raise ValueError("Stock or benchmark data not available.")
    if 'Close' not in stock_data.columns or 'Close' not in benchmark_data.columns:
        raise KeyError("Missing 'Close' column in stock or benchmark data.")
    
    # Calculate Relative Strength
    rs = stock_data['Close'] / benchmark_data['Close']
    return rs

# Function to plot Relative Strength
def plot_relative_strength(stock, benchmark, rs_series):
    plt.figure(figsize=(12, 6))
    plt.plot(rs_series, label=f"{stock} / {benchmark}", color="blue")
    plt.title(f"Relative Strength: {stock} vs {benchmark}")
    plt.xlabel("Date")
    plt.ylabel("Relative Strength")
    plt.legend()
    plt.grid(alpha=0.5)
    st.pyplot(plt)

# Streamlit app
st.title("Relative Strength Indicator")
st.write("This app calculates and displays the relative strength (RS) of a stock compared to a benchmark index or ETF.")

# User inputs
stock_symbol = st.text_input("Enter the stock symbol (e.g., TSLA):", "TSLA")
benchmark_symbol = st.text_input("Enter the benchmark symbol (e.g., SPY):", "SPY")
start_date = st.date_input("Start Date:", pd.to_datetime("2023-01-01"))
end_date = st.date_input("End Date:", pd.to_datetime("2025-01-01"))

if st.button("Calculate Relative Strength"):
    if stock_symbol and benchmark_symbol:
        # Fetch data
        stock_data = fetch_stock_data(stock_symbol, start_date, end_date)
        benchmark_data = fetch_stock_data(benchmark_symbol, start_date, end_date)

        if stock_data is not None and benchmark_data is not None:
            print("Stock Data shape:", stock_data.shape if stock_data is not None else "None")
            print("Benchmark Data shape:", benchmark_data.shape if benchmark_data is not None else "None")
            
            # Align data by date
            if 'Close' in stock_data.columns and 'Close' in benchmark_data.columns:
                combined_data = pd.merge(stock_data['Close'], benchmark_data['Close'], left_index=True, right_index=True, suffixes=("_stock", "_benchmark"))
                print("Combined Data shape:", combined_data.shape)

                # Calculate Relative Strength
                try:
                    # Pass the combined data with the correct column names
                    rs_series = calculate_relative_strength(combined_data[['Close_stock']], combined_data[['Close_benchmark']]) 
                    if not rs_series.dropna().empty:
                        plot_relative_strength(stock_symbol.upper(), benchmark_symbol.upper(), rs_series)
                    else:
                        st.error("No valid data points for Relative Strength calculation.")
                except (ValueError, KeyError) as e:
                    st.error(f"Error in calculating Relative Strength: {e}")
            else:
                st.error("One or both data sets are missing the 'Close' column.")
        else:
            st.error("Failed to fetch data for one or both symbols. Please try again.")
    else:
        st.error("Please enter valid stock and benchmark symbols.")
