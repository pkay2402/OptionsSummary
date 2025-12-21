# modules/BlockTrade.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Sample list of top 100 stocks (S&P 100 subset for demonstration)
with st.expander("Edit Ticker List", expanded=False):
    default_tickers = [
        'AAPL', 'ORCL','MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NVDA', 'JPM', 'WMT', 'V',
        'PG', 'JNJ', 'UNH', 'HD', 'BAC', 'MA', 'XOM', 'AVGO', 'CVX', 'ABBV',
        'PFE', 'CSCO', 'LLY', 'COST', 'MRK', 'ADBE', 'CRM', 'KO', 'PEP', 'TMO',
        'ORCL', 'NKE', 'ABT', 'MCD', 'CMCSA', 'ACN', 'DHR', 'WFC', 'VZ', 'TXN',
        'PM', 'NEE', 'NFLX', 'AMD', 'BMY', 'RTX', 'UPS', 'INTC', 'QCOM', 'T',
        'HON', 'IBM', 'GE', 'CAT', 'LIN', 'AMT', 'BA', 'SPGI', 'INTU', 'LOW',
        'UNP', 'SBUX', 'CVS', 'GILD', 'GS', 'C', 'MDLZ', 'AMGN', 'BLK', 'MMM',
        'TJX', 'DE', 'MO', 'AXP', 'MS', 'PLD', 'LMT', 'MDT', 'SCHW', 'BKNG',
        'PNC', 'DUK', 'ADP', 'TGT', 'SYK', 'ISRG', 'COP', 'SO', 'CI', 'BDX',
        'CB', 'CSX', 'CL', 'CME', 'EOG', 'NOC', 'BSX', 'USB', 'ICE', 'MMC','FCX','ZS'
    ]
    tickers_input = st.text_area("Enter tickers separated by commas", value=",".join(default_tickers))
    SP100_TICKERS = [ticker.strip() for ticker in tickers_input.split(",") if ticker.strip()]

def fetch_stock_data(ticker, start_date, end_date):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(start=start_date, end=end_date)
        if hist.empty:
            raise ValueError("No data available for this ticker")
        return hist
    except Exception as e:
        return None

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
    st.title("Stocks Block Trade Analyzer")

    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)

    if st.button("Analyze Stocks"):
        with st.spinner("Fetching and analyzing data for top 100 stocks..."):
            all_block_trades = []
            progress_bar = st.progress(0)
            
            for i, ticker in enumerate(SP100_TICKERS):
                hist = fetch_stock_data(ticker, start_date, end_date)
                if hist is not None:
                    hist = detect_volume_spikes(hist)
                    block_trades = analyze_block_trade_reaction(hist)
                    if not block_trades.empty:
                        # Get current price
                        stock = yf.Ticker(ticker)
                        current_price = stock.info['regularMarketPrice']
                        block_trades['Ticker'] = ticker
                        block_trades['Current_Price'] = current_price
                        all_block_trades.append(block_trades)
                progress_bar.progress((i + 1) / len(SP100_TICKERS))

            if all_block_trades:
                combined_block_trades = pd.concat(all_block_trades)
                
                # Sort by date in descending order
                combined_block_trades = combined_block_trades.sort_values(by='Date', ascending=False)
                
                # Display summary table with Current_Price added
                st.subheader("Detected Block Trades in Top 100 Stocks")
                st.dataframe(combined_block_trades[[
                    'Ticker', 'Volume', 'Price_Change_1D', 
                    'Price_Change_5D', 'Trade_Type', 'Current_Price'
                ]].reset_index())

                # Plot block trades summary
                fig = go.Figure()
                for ticker in combined_block_trades['Ticker'].unique():
                    ticker_trades = combined_block_trades[combined_block_trades['Ticker'] == ticker]
                    fig.add_trace(go.Scatter(
                        x=ticker_trades.index,
                        y=ticker_trades['Volume'],
                        mode='markers',
                        name=ticker,
                        marker=dict(
                            size=12,
                            color=np.where(ticker_trades['Trade_Type'] == 'Buy', 'green', 'red'),
                            line=dict(width=2, color='white')
                        ),
                        text=[f"{ticker} - 1D: {p1:.2f}%, 5D: {p5:.2f}%" 
                              for p1, p5 in zip(ticker_trades['Price_Change_1D'], ticker_trades['Price_Change_5D'])],
                        hoverinfo='text'
                    ))

                fig.update_layout(
                    title="Block Trades Across Top 100 Stocks",
                    yaxis=dict(title='Volume'),
                    legend=dict(x=0.01, y=0.99),
                    template='plotly_dark',
                    hovermode='x unified',
                    height=600
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("No significant block trades detected in the top 100 stocks.")

if __name__ == "__main__":
    Blocktrade_run()
