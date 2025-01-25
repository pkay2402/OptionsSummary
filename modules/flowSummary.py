def run():
    """Main function to run the Streamlit application."""
    import streamlit as st

    # Remove this line:
    # st.set_page_config(page_title="Flow Summary", layout="wide")

    st.title("📊 Flow Summary")

    # Rest of the code remains the same...
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
