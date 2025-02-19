# modules/BlockTrade.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# [Previous functions: fetch_stock_data, detect_volume_spikes, analyze_block_trade_reaction remain unchanged]

# Main function renamed to match your import
def Blocktrade_run():
    st.title("Stock Activity Dashboard")

    # User input for ticker symbol
    ticker = st.text_input("Enter Stock Ticker Symbol (e.g., AAPL, MSFT):", "AAPL").upper()

    # Define date range (last 6 months)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)

    # Fetch and process data
    if st.button("Analyze"):
        with st.spinner("Fetching and analyzing data..."):
            hist, info = fetch_stock_data(ticker, start_date, end_date)
            if hist is not None:
                hist = detect_volume_spikes(hist)
                block_trades = analyze_block_trade_reaction(hist)

                # [Rest of the plotting and display code remains unchanged]
                # Create the combined figure
                fig = go.Figure()
                # Add Price trace (left y-axis)
                fig.add_trace(go.Scatter(
                    x=hist.index,
                    y=hist['Close'],
                    name='Price',
                    line=dict(color='yellow'),
                    yaxis='y1'
                ))
                # Add Volume bars (right y-axis)
                fig.add_trace(go.Bar(
                    x=hist.index,
                    y=hist['Volume'],
                    name='Volume',
                    opacity=0.5,
                    marker_color='grey',
                    yaxis='y2'
                ))
                # Add Block Trades as scatter points on volume
                fig.add_trace(go.Scatter(
                    x=block_trades.index,
                    y=block_trades['Volume'],
                    mode='markers',
                    name='Block Trades',
                    marker=dict(
                        size=12,
                        color=np.where(block_trades['Trade_Type'] == 'Buy', 'green', 'red'),
                        line=dict(width=2, color='white')
                    ),
                    yaxis='y2',
                    text=[f"1D: {p1:.2f}%, 5D: {p5:.2f}%" for p1, p5 in zip(block_trades['Price_Change_1D'], block_trades['Price_Change_5D'])],
                    hoverinfo='text'
                ))
                # Update layout with dual y-axes
                fig.update_layout(
                    title=f"{ticker} Price, Volume, and Block Trade Reactions",
                    yaxis=dict(
                        title='Price',
                        side='left',
                        color='yellow'
                    ),
                    yaxis2=dict(
                        title='Volume',
                        overlaying='y',
                        side='right',
                        color='grey'
                    ),
                    legend=dict(x=0.01, y=0.99),
                    template='plotly_dark',
                    hovermode='x unified',
                    height=600
                )
                st.plotly_chart(fig, use_container_width=True)
                st.subheader("Stock Information")
                st.write(f"Company Name: {info.get('longName', 'N/A')}")
                st.write(f"Sector: {info.get('sector', 'N/A')}")
                st.write(f"Market Cap: ${info.get('marketCap', 'N/A'):,.0f}")
