# modules/BlockTrade.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

def fetch_stock_data(ticker, start_date, end_date):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(start=start_date, end=end_date)
        info = stock.info
        if hist.empty:
            raise ValueError("No data available for this ticker")
        return hist, info
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None, None

def detect_volume_spikes(df, window=20, threshold=2):
    df['Volume_MA'] = df['Volume'].rolling(window=window).mean()
    df['Volume_Std'] = df['Volume'].rolling(window=window).std()
    df['Volume_Z_Score'] = (df['Volume'] - df['Volume_MA']) / df['Volume_Std']
    df['Block_Trade'] = df['Volume_Z_Score'] > threshold
    return df

def analyze_block_trade_reaction(df, days_after=5):
    block_trades = df[df['Block_Trade']].copy()
    block_trades['Price_Change_1D'] = np.nan
    block_trades['Price_Change_5D'] = np.nan
    block_trades['Trade_Type'] = 'Unknown'

    for idx in block_trades.index:
        future_idx_1d = df.index.get_loc(idx) + 1
        future_idx_5d = df.index.get_loc(idx) + days_after

        if future_idx_1d < len(df):
            price_change_1d = (df['Close'].iloc[future_idx_1d] - df['Close'].loc[idx]) / df['Close'].loc[idx] * 100
            block_trades.loc[idx, 'Price_Change_1D'] = price_change_1d
            if price_change_1d > 0.5:
                block_trades.loc[idx, 'Trade_Type'] = 'Buy'
            elif price_change_1d < -0.5:
                block_trades.loc[idx, 'Trade_Type'] = 'Sell'

        if future_idx_5d < len(df):
            price_change_5d = (df['Close'].iloc[future_idx_5d] - df['Close'].loc[idx]) / df['Close'].loc[idx] * 100
            block_trades.loc[idx, 'Price_Change_5D'] = price_change_5d

    return block_trades

def Blocktrade_run():
    st.title("Stock Activity Dashboard")

    ticker = st.text_input("Enter Stock Ticker Symbol (e.g., AAPL, MSFT):", "AAPL").upper()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)

    if st.button("Analyze"):  # Only run this block when the button is clicked
        with st.spinner("Fetching and analyzing data..."):
            hist, info = fetch_stock_data(ticker, start_date, end_date)
            if hist is None or info is None:  # Explicitly check for None
                return  # Exit early if data fetch failed

            hist = detect_volume_spikes(hist)
            block_trades = analyze_block_trade_reaction(hist)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hist.index,
                y=hist['Close'],
                name='Price',
                line=dict(color='yellow'),
                yaxis='y1'
            ))
            fig.add_trace(go.Bar(
                x=hist.index,
                y=hist['Volume'],
                name='Volume',
                opacity=0.5,
                marker_color='grey',
                yaxis='y2'
            ))
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
            fig.update_layout(
                title=f"{ticker} Price, Volume, and Block Trade Reactions",
                yaxis=dict(title='Price', side='left', color='yellow'),
                yaxis2=dict(title='Volume', overlaying='y', side='right', color='grey'),
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
