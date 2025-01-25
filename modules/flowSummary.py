import pandas as pd
import requests
from io import StringIO
from datetime import datetime
import streamlit as st

def fetch_data_from_urls(urls):
    all_data = pd.DataFrame()  # Initialize an empty DataFrame to combine data from all URLs
    for url in urls:
        try:
            # Fetch the CSV data from the URL
            response = requests.get(url)
            response.raise_for_status()  # Raise an error if the request fails
            
            # Check if the response is a CSV file
            if 'text/csv' in response.headers.get('Content-Type', ''):
                # Read the CSV content into a pandas DataFrame
                csv_data = StringIO(response.text)
                df = pd.read_csv(csv_data)

                # Filter out records with Volume less than 100
                df = df[df['Volume'] >= 100]

                # Convert Expiration to datetime and filter out records with the current date
                df['Expiration'] = pd.to_datetime(df['Expiration'])
                current_date = datetime.now().date()
                df = df[df['Expiration'].dt.date != current_date]

                # Append the current DataFrame to the all_data DataFrame
                all_data = pd.concat([all_data, df], ignore_index=True)
            else:
                st.warning(f"Data from {url} is not in CSV format. Skipping...")

        except Exception as e:
            st.error(f"Error fetching or processing data from {url}: {e}")
    
    return all_data

def summarize_whale_transactions(df, selected_symbol=None, exclude_spx=True):
    # Exclude SPX and SPXW by default
    if exclude_spx:
        df = df[~df['Symbol'].isin(['SPX', 'SPXW'])]
    
    # Calculate Whale transactions (Volume * Last Price > 5 million)
    df['Transaction Value'] = df['Volume'] * df['Last Price'] * 100
    whale_transactions = df[df['Transaction Value'] > 5_000_000]

    # If a symbol is selected, filter by it
    if selected_symbol:
        whale_transactions = whale_transactions[whale_transactions['Symbol'] == selected_symbol]

    # Summarize whale transactions
    summary = (
        whale_transactions.groupby(['Symbol', 'Expiration', 'Strike Price', 'Call/Put', 'Last Price'])
        .agg({'Volume': 'sum', 'Transaction Value': 'sum'})
        .reset_index()
    )
    summary = summary.sort_values(by='Transaction Value', ascending=False)
    return summary

# Streamlit UI
st.title("Options Flow Analyzer")

# Input: List of URLs
urls = [
    "https://www.cboe.com/us/options/market_statistics/symbol_data/csv/?mkt=cone",
    "https://www.cboe.com/us/options/market_statistics/symbol_data/csv/?mkt=opt",
    "https://www.cboe.com/us/options/market_statistics/symbol_data/csv/?mkt=ctwo",
    "https://www.cboe.com/us/options/market_statistics/symbol_data/csv/?mkt=exo"
]

st.write("Fetching data from the following URLs:", urls)

# Fetch data from all URLs
data = fetch_data_from_urls(urls)

if not data.empty:
    # Add Whale Transaction filter
    whale_option = st.checkbox("Show Whale Transactions Only")

    if whale_option:
        st.subheader("Whale Transactions")

        # Option to exclude SPX and SPXW
        exclude_spx = st.checkbox("Exclude SPX and SPXW (default)", value=True)

        view_option = st.radio(
            "Do you want to filter by a specific stock or see all stocks?",
            options=["Specific Stock", "All Stocks"]
        )

        if view_option == "Specific Stock":
            # Show available symbols for filtering
            symbols = sorted(data['Symbol'].unique())
            selected_symbol = st.selectbox("Select Symbol to Analyze", symbols)
            summary = summarize_whale_transactions(data, selected_symbol, exclude_spx)
            st.subheader(f"Whale Transactions for {selected_symbol}")
        else:
            # Show all whale transactions across all stocks
            summary = summarize_whale_transactions(data, exclude_spx=exclude_spx)
            st.subheader("Whale Transactions Across All Stocks")

        st.dataframe(summary)

        # Option to download the summary
        csv = summary.to_csv(index=False)
        st.download_button(
            label="Download Whale Transactions as CSV",
            data=csv,
            file_name="whale_transactions_summary.csv",
            mime="text/csv"
        )
    else:
        # Regular options flow analysis
        st.subheader("Options Flow Analysis")
        symbols = sorted(data['Symbol'].unique())
        selected_symbol = st.selectbox("Select Symbol to Analyze", symbols)

        # Optional filter for Call/Put
        call_put_options = ['All', 'C', 'P']
        selected_call_put = st.selectbox("Select Call/Put", call_put_options)

        # Optional filter for Expiration
        expiration_dates = sorted(data['Expiration'].dt.date.unique())
        selected_expiration = st.selectbox("Select Expiration Date", [None] + expiration_dates)

        # Apply filters and summarize
        if selected_symbol:
            if selected_call_put == 'All':
                selected_call_put = None  # Set to None to filter out if "All" is selected
            
            summary = summarize_whale_transactions(data[data['Symbol'] == selected_symbol])
            st.subheader(f"Summary of Flows for {selected_symbol}")
            st.dataframe(summary)

            # Option to download the summary
            csv = summary.to_csv(index=False)
            st.download_button(
                label="Download Summary as CSV",
                data=csv,
                file_name=f"{selected_symbol}_summary.csv",
                mime="text/csv"
            )
