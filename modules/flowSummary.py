import pandas as pd
import requests
from io import StringIO
from datetime import datetime
import logging
from typing import List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_csv_content_type(response: requests.Response) -> bool:
    """Validate if the response content type is CSV."""
    return 'text/csv' in response.headers.get('Content-Type', '')

def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Apply filters to the DataFrame."""
    df = df[df['Volume'] >= 100]
    df['Expiration'] = pd.to_datetime(df['Expiration'])
    df = df[df['Expiration'].dt.date >= datetime.now().date()]
    return df

def fetch_data_from_url(url: str) -> Optional[pd.DataFrame]:
    """Fetch and process data from a single URL."""
    try:
        response = requests.get(url)
        response.raise_for_status()

        if validate_csv_content_type(response):
            csv_data = StringIO(response.text)
            df = pd.read_csv(csv_data)
            return apply_filters(df)
        else:
            logger.warning(f"Data from {url} is not in CSV format. Skipping...")
    except Exception as e:
        logger.error(f"Error fetching or processing data from {url}: {e}")
    return None

def fetch_data_from_urls(urls: List[str]) -> pd.DataFrame:
    """Fetch and combine data from multiple CSV URLs into a single DataFrame."""
    data_frames = []
    for url in urls:
        df = fetch_data_from_url(url)
        if df is not None:
            data_frames.append(df)
    return pd.concat(data_frames, ignore_index=True) if data_frames else pd.DataFrame()

def summarize_transactions(df: pd.DataFrame, whale_filter: bool = False, exclude_symbols: List[str] = None) -> pd.DataFrame:
    """Summarize transactions from the given DataFrame."""
    if exclude_symbols:
        df = df[~df['Symbol'].isin(exclude_symbols)]

    df['Transaction Value'] = df['Volume'] * df['Last Price'] * 100

    if whale_filter:
        df = df[df['Transaction Value'] > 5_000_000]

    summary = (
        df.groupby(['Symbol', 'Expiration', 'Strike Price', 'Call/Put', 'Last Price'])
        .agg({'Volume': 'sum', 'Transaction Value': 'sum'})
        .reset_index()
    )
    return summary.sort_values(by='Transaction Value', ascending=False)

def run():
    """Main function to run the Streamlit application."""
    import streamlit as st

    st.title("📊 Flow Summary")

    # Sidebar for filters and options
    with st.sidebar:
        st.header("Filters & Options")
        whale_option = st.checkbox("Show Whale Transactions Only")
        risk_reversal_option = st.checkbox("Show Risk Reversal Trades")
        
        # User input for excluded symbols
        default_excluded_symbols = ["SPX", "SPXW", "VIX", "SPY"]
        excluded_symbols = st.text_input(
            "Enter symbols to exclude (comma-separated)",
            value=", ".join(default_excluded_symbols)
        )
        excluded_symbols = [s.strip() for s in excluded_symbols.split(",") if s.strip()]

    # URLs for data fetching
    urls = [
        "https://www.cboe.com/us/options/market_statistics/symbol_data/csv/?mkt=cone",
        "https://www.cboe.com/us/options/market_statistics/symbol_data/csv/?mkt=opt",
        "https://www.cboe.com/us/options/market_statistics/symbol_data/csv/?mkt=ctwo",
        "https://www.cboe.com/us/options/market_statistics/symbol_data/csv/?mkt=exo"
    ]

    # Fetch data with a progress spinner
    with st.spinner("Fetching data..."):
        data = fetch_data_from_urls(urls)

    if not data.empty:
        # Use tabs for different views
        tab1, tab2, tab3 = st.tabs(["Risk Reversal Trades", "Whale Transactions", "Options Flow Analysis"])

        with tab1:
            if risk_reversal_option:
                st.subheader("Risk Reversal Trades")
                risk_reversal_data = filter_risk_reversal(data, exclude_symbols=excluded_symbols)
                st.dataframe(risk_reversal_data)

                csv = risk_reversal_data.to_csv(index=False)
                st.download_button(
                    label="Download Risk Reversal Trades as CSV",
                    data=csv,
                    file_name="risk_reversal_trades.csv",
                    mime="text/csv"
                )
            else:
                st.info("Enable 'Show Risk Reversal Trades' in the sidebar to view this section.")

        with tab2:
            if whale_option:
                st.subheader("Whale Transactions")
                summary = summarize_transactions(data, whale_filter=True, exclude_symbols=excluded_symbols)
                st.dataframe(summary)

                csv = summary.to_csv(index=False)
                st.download_button(
                    label="Download Whale Transactions as CSV",
                    data=csv,
                    file_name="whale_transactions_summary.csv",
                    mime="text/csv"
                )
            else:
                st.info("Enable 'Show Whale Transactions Only' in the sidebar to view this section.")

        with tab3:
            st.subheader("Options Flow Analysis")

            symbols = sorted(data['Symbol'].unique())
            selected_symbol = st.selectbox("Select Symbol to Analyze", symbols)
            symbol_data = data[data['Symbol'] == selected_symbol]

            strike_prices = sorted(symbol_data['Strike Price'].unique())
            selected_strike_price = st.selectbox("Select Strike Price (Optional)", [None] + strike_prices)

            call_put_options = ['C', 'P']
            selected_call_put = st.radio("Select Call/Put (Optional)", [None] + call_put_options, horizontal=True)

            if selected_strike_price:
                symbol_data = symbol_data[symbol_data['Strike Price'] == selected_strike_price]

            if selected_call_put:
                symbol_data = symbol_data[symbol_data['Call/Put'] == selected_call_put]

            summary = summarize_transactions(symbol_data, whale_filter=False, exclude_symbols=excluded_symbols)
            st.dataframe(summary)

            csv = summary.to_csv(index=False)
            st.download_button(
                label="Download Summary as CSV",
                data=csv,
                file_name=f"{selected_symbol}_summary.csv",
                mime="text/csv"
            )

    st.write("This is the Flow Summary application.")
