import pandas as pd
import requests
from io import StringIO
from datetime import datetime
import streamlit as st

def fetch_data_from_urls(urls):
    """
    Fetch and combine data from multiple CSV URLs into a single DataFrame.

    :param urls: List of URLs to fetch CSV data from
    :return: DataFrame containing combined data from all URLs
    """
    all_data = pd.DataFrame()
    for url in urls:
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            if 'text/csv' in response.headers.get('Content-Type', ''):
                csv_data = StringIO(response.text)
                df = pd.read_csv(csv_data)

                df = df[df['Volume'] >= 100]
                df['Expiration'] = pd.to_datetime(df['Expiration'])
                current_date = datetime.now().date()
                df = df[df['Expiration'].dt.date != current_date]

                all_data = pd.concat([all_data, df], ignore_index=True)
            else:
                st.warning(f"Data from {url} is not in CSV format. Skipping...")

        except Exception as e:
            st.error(f"Error fetching or processing data from {url}: {e}")
    
    return all_data

def summarize_transactions(df, whale_filter=False, selected_symbol=None, exclude_spx=True):
    """
    Summarize transactions from the given DataFrame.
    
    :param df: DataFrame with options data
    :param whale_filter: Boolean, whether to apply the whale transaction filter (>5M)
    :param selected_symbol: Optional, filter to a specific symbol
    :param exclude_spx: Boolean, whether to exclude SPX and SPXW options
    :return: DataFrame with summarized transactions
    """
    if exclude_spx:
        df = df[~df['Symbol'].isin(['SPX', 'SPXW'])]
    
    # Always calculate Transaction Value for consistency
    df['Transaction Value'] = df['Volume'] * df['Last Price'] * 100

    # Apply whale transaction filter if selected
    if whale_filter:
        df = df[df['Transaction Value'] > 5_000_000]

    if selected_symbol:
        df = df[df['Symbol'] == selected_symbol]

    summary = (
        df.groupby(['Symbol', 'Expiration', 'Strike Price', 'Call/Put', 'Last Price'])
        .agg({'Volume': 'sum', 'Transaction Value': 'sum'})
        .reset_index()
    )
    return summary.sort_values(by='Transaction Value', ascending=False)

def run():
    """
    Main function to run the Streamlit application for Options Flow Analysis.
    This function sets up the UI, fetches data, and processes it for display.
    """
    st.title("Flow Summary")
    
    urls = [
        "https://www.cboe.com/us/options/market_statistics/symbol_data/csv/?mkt=cone",
        "https://www.cboe.com/us/options/market_statistics/symbol_data/csv/?mkt=opt",
        "https://www.cboe.com/us/options/market_statistics/symbol_data/csv/?mkt=ctwo",
        "https://www.cboe.com/us/options/market_statistics/symbol_data/csv/?mkt=exo"
    ]
    
    st.write("Fetching data from the following URLs:", urls)
    
    data = fetch_data_from_urls(urls)

    if not data.empty:
        whale_option = st.checkbox("Show Whale Transactions Only")

        if whale_option:
            st.subheader("Whale Transactions")
            exclude_spx = st.checkbox("Exclude SPX and SPXW (default)", value=True)
            view_option = st.radio(
                "Do you want to filter by a specific stock or see all stocks?",
                options=["Specific Stock", "All Stocks"]
            )

            if view_option == "Specific Stock":
                symbols = sorted(data['Symbol'].unique())
                selected_symbol = st.selectbox("Select Symbol to Analyze", symbols)
                summary = summarize_transactions(data, whale_filter=True, selected_symbol=selected_symbol, exclude_spx=exclude_spx)
                st.subheader(f"Whale Transactions for {selected_symbol}")
            else:
                summary = summarize_transactions(data, whale_filter=True, exclude_spx=exclude_spx)
                st.subheader("Whale Transactions Across All Stocks")

            st.dataframe(summary)

            csv = summary.to_csv(index=False)
            st.download_button(
                label="Download Whale Transactions as CSV",
                data=csv,
                file_name="whale_transactions_summary.csv",
                mime="text/csv"
            )
        else:
            st.subheader("Options Flow Analysis")
            symbols = sorted(data['Symbol'].unique())
            selected_symbol = st.selectbox("Select Symbol to Analyze", symbols)
            call_put_options = ['All', 'C', 'P']
            selected_call_put = st.selectbox("Select Call/Put", call_put_options)
            expiration_dates = sorted(data['Expiration'].dt.date.unique())
            selected_expiration = st.selectbox("Select Expiration Date", [None] + expiration_dates)

            if selected_symbol:
                if selected_call_put == 'All':
                    selected_call_put = None  
                
                summary = summarize_transactions(data[data['Symbol'] == selected_symbol], whale_filter=False)
                st.subheader(f"Summary of Flows for {selected_symbol}")
                st.dataframe(summary)

                csv = summary.to_csv(index=False)
                st.download_button(
                    label="Download Summary as CSV",
                    data=csv,
                    file_name=f"{selected_symbol}_summary.csv",
                    mime="text/csv"
                )

        st.write("This is the Flow Summary application.")
