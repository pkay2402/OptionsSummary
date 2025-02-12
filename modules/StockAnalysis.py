import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import numpy as np
from math import ceil

# Set page configuration for responsiveness
st.set_page_config(layout="wide")

# Initialize session state for chart display
if 'chart_symbol' not in st.session_state:
    st.session_state['chart_symbol'] = None

def calculate_rsi(data, periods=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def get_rsi_status(rsi):
    if rsi > 70:
        return "Overbought"
    elif rsi > 50:
        return "Strong"
    elif rsi > 30:
        return "Weak"
    else:
        return "Oversold"

def calculate_support_resistance(hist, n=20):
    # Get recent highs and lows
    highs = hist['High'].nlargest(n)
    lows = hist['Low'].nsmallest(n)
    
    # Calculate price clusters for support and resistance
    price_clusters = pd.concat([highs, lows]).round(2)
    price_clusters = price_clusters.value_counts().sort_index()
    
    # Find significant levels (where multiple price touches occurred)
    significant_levels = price_clusters[price_clusters >= 2].index.values
    
    if len(significant_levels) >= 2:
        current_price = hist['Close'].iloc[-1]
        
        # Find nearest support (below current price)
        supports = significant_levels[significant_levels < current_price]
        support = supports[-1] if len(supports) > 0 else significant_levels[0]
        
        # Find nearest resistance (above current price)
        resistances = significant_levels[significant_levels > current_price]
        resistance = resistances[0] if len(resistances) > 0 else significant_levels[-1]
        
        return round(support, 2), round(resistance, 2)
    else:
        # Fallback to simple support/resistance based on recent price action
        return round(hist['Low'].tail(n).mean(), 2), round(hist['High'].tail(n).mean(), 2)

def calculate_relative_strength(symbol_hist, spy_hist, lookback=20):
    # Ensure we're comparing the same timeframe
    common_dates = symbol_hist.index.intersection(spy_hist.index)
    if len(common_dates) < 2:
        return 0, "N/A"
    
    # Calculate percentage changes
    symbol_change = symbol_hist['Close'].loc[common_dates].pct_change(periods=lookback).iloc[-1]
    spy_change = spy_hist['Close'].loc[common_dates].pct_change(periods=lookback).iloc[-1]
    
    # Calculate relative strength
    if spy_change != 0:
        rs = ((1 + symbol_change) / (1 + spy_change) - 1) * 100
        rs_status = "Strong" if rs > 0 else "Weak"
        return round(rs, 2), rs_status
    return 0, "N/A"

def fetch_stock_data(symbol, period="1d", interval="5m", spy_hist=None):
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period, interval=interval)
        if hist.empty:
            raise ValueError(f"No data available for {symbol}")
        
        # Calculate VWAP
        hist['Cumulative_Volume'] = hist['Volume'].cumsum()
        hist['Cumulative_PV'] = (hist['Close'] * hist['Volume']).cumsum()
        hist['VWAP'] = hist['Cumulative_PV'] / hist['Cumulative_Volume']
        
        # Calculate RSI
        hist['RSI'] = calculate_rsi(hist)
        current_rsi = round(hist['RSI'].iloc[-1], 2)
        rsi_status = get_rsi_status(current_rsi)
        
        # Calculate 21 EMA
        hist['EMA21'] = hist['Close'].ewm(span=21, adjust=False).mean()
        
        # Calculate Relative Strength vs SPY
        rs_value, rs_status = calculate_relative_strength(hist, spy_hist) if spy_hist is not None else (0, "N/A")
        
        today_data = hist.iloc[-1]
        open_price = round(today_data["Open"], 2)
        high_price = round(hist["High"].iloc[-1], 2)
        low_price = round(hist["Low"].iloc[-1], 2)
        current_price = round(today_data["Close"], 2)
        vwap = round(hist['VWAP'].iloc[-1], 2)
        ema_21 = round(hist['EMA21'].iloc[-1], 2)

        # Daily Pivot Calculation
        daily_pivot = round((high_price + low_price + current_price) / 3, 2)

        # EMAs for KeyMAs Logic
        ema_9 = round(hist["Close"].ewm(span=9, adjust=False).mean().iloc[-1], 2)
        ema_50 = round(hist["Close"].ewm(span=50, adjust=False).mean().iloc[-1], 2)

        # KeyMAs Logic
        if current_price > ema_9 and current_price > ema_21 and current_price > ema_50:
            key_mas = "Bullish"
        elif current_price < ema_9 and current_price < ema_21 and current_price < ema_50:
            key_mas = "Bearish"
        else:
            key_mas = "Mixed"

        # Determine Price_Vwap
        if current_price > vwap and current_price > open_price:
            direction = "Bullish"
        elif current_price < vwap and current_price < open_price:
            direction = "Bearish"
        else:
            direction = "Neutral"

        return pd.DataFrame({
            "Symbol": [symbol],
            "Current Price": [current_price],
            "VWAP": [vwap],
            "EMA21": [ema_21],
            "Rel Strength SPY": [rs_status],
            "Daily Pivot": [daily_pivot],
            "Price_Vwap": [direction],
            "KeyMAs": [key_mas],
            "RSI_Status": [rsi_status]
        }), hist.round(2)
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame(), pd.DataFrame()

def plot_candlestick(data, symbol):
    fig = go.Figure()
    
    # Add candlestick
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name=symbol
    ))
    
    # Add VWAP
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['VWAP'],
        name='VWAP',
        line=dict(color='purple', width=2)
    ))
    
    # Add 21 EMA
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['EMA21'],
        name='21 EMA',
        line=dict(color='orange', width=2)
    ))

    fig.update_layout(
        title=f'{symbol} Candlestick Chart with VWAP and 21 EMA',
        yaxis_title='Price',
        template='plotly_white',
        height=600,
        xaxis_rangeslider_visible=False
    )
    
    # Format y-axis to show 2 decimal places
    fig.update_layout(yaxis_tickformat='.2f')
    
    return fig

def run():

# Settings at the top
 st.header("⚙️ Settings")
with st.container():
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        stock_list = st.text_area("Enter Stock Symbols (comma separated)", "^SPX, SPY, QQQ, UVXY, AAPL, GOOGL, META, NVDA, TSLA, AMZN, COIN, PLTR").upper()
        symbols = [s.strip() for s in stock_list.split(",")]

    with col2:
        time_frames = {
            "1 Day": "1d",
            "5 Days": "5d",
            "1 Month": "1mo",
            "3 Months": "3mo",
            "6 Months": "6mo",
            "1 Year": "1y",
            "2 Years": "2y",
            "5 Years": "5y"
        }
        selected_timeframe = st.selectbox("Choose Time Frame", list(time_frames.keys()), index=0)
        period = time_frames[selected_timeframe]

    with col3:
        intervals = ["1m", "5m", "15m", "30m", "1h", "1d"]
        selected_interval = st.selectbox("Choose Interval", intervals, index=1)
        auto_refresh = st.checkbox("Auto Refresh every 5 mins")

# Main content area
st.subheader(f"📈 Stock Data for {selected_timeframe} ({selected_interval} interval)")

# Create tooltips dictionary
tooltips = {
    "Current Price": "The latest closing price for the stock",
    "VWAP": "Volume Weighted Average Price - Average price weighted by volume",
    "EMA21": "21-period Exponential Moving Average",
    "Rel Strength SPY": "Relative strength compared to SPY (Strong: outperforming, Weak: underperforming)",
    "Daily Pivot": "Calculated as (High + Low + Close) / 3",
    "Price_Vwap": "Bullish: Price > VWAP & Open, Bearish: Price < VWAP & Open, Neutral: Mixed conditions",
    "KeyMAs": "Based on 9, 21, 50 EMAs - Bullish: Price > all EMAs, Bearish: Price < all EMAs, Mixed: Other conditions",
    "RSI_Status": "RSI indicator status - Overbought(>70), Strong(50-70), Weak(30-50), Oversold(<30)"
}

with st.container():
    main_col1, main_col2 = st.columns([2, 1])

with main_col1:
    # Fetch SPY data first for relative strength calculations
    spy_data, spy_hist = fetch_stock_data("SPY", period=period, interval=selected_interval)
    stock_histories = {"SPY": spy_hist}
    all_data = pd.DataFrame()
    
    num_symbols = len(symbols)
    num_cols = 3
    num_rows = ceil(num_symbols / num_cols)
    
    for i in range(num_rows):
        cols = st.columns(num_cols)
        for j in range(num_cols):
            idx = i * num_cols + j
            if idx < num_symbols:
                symbol = symbols[idx]
                if symbol == "SPY":
                    data = spy_data
                    history = spy_hist
                else:
                    data, history = fetch_stock_data(symbol, period=period, interval=selected_interval, spy_hist=spy_hist)
                
                if not data.empty:
                    all_data = pd.concat([all_data, data], ignore_index=True)
                    stock_histories[symbol] = history
                    if cols[j].button(f'📈 {symbol}', key=f'btn_{symbol}'):
                        st.session_state['chart_symbol'] = symbol

    if "SPY" not in all_data["Symbol"].values:
        all_data = pd.concat([spy_data, all_data], ignore_index=True)
    stock_histories["SPY"] = spy_hist

    def color_columns(val):
        if isinstance(val, str):
            if val in ["Bullish", "Strong"]:
                return 'background-color: #90EE90; color: black'
            elif val in ["Bearish", "Weak"]:
                return 'background-color: #FF7F7F; color: black'
            elif val in ["Neutral", "Mixed"]:
                return 'background-color: #D3D3D3; color: black'
            elif val == "Overbought":
                return 'background-color: #FFB6C1; color: black'
            elif val == "Oversold":
                return 'background-color: #87CEEB; color: black'
        return ''

    # Add tooltips to the DataFrame
    styled_df = all_data.style.format({
        'Current Price': '{:.2f}',
        'VWAP': '{:.2f}',
        'Daily Pivot': '{:.2f}'
    }).applymap(color_columns, subset=['Price_Vwap', 'KeyMAs', 'RSI_Status', 'Rel Strength SPY'])

    # Display DataFrame with tooltips
    st.markdown("""
        <style>
        .tooltip {
            position: relative;
            display: inline-block;
        }
        .tooltip .tooltiptext {
            visibility: hidden;
            background-color: black;
            color: white;
            text-align: center;
            padding: 5px;
            border-radius: 6px;
            position: absolute;
            z-index: 1;
            bottom: 100%;
            left: 50%;
            margin-left: -60px;
            opacity: 0;
            transition: opacity 0.3s;
        }
        .tooltip:hover .tooltiptext {
            visibility: visible;
            opacity: 1;
        }
        </style>
    """, unsafe_allow_html=True)

    # Display compact tooltip information above the DataFrame
    with st.expander("ℹ️ Column Descriptions"):
        for col, desc in tooltips.items():
            st.markdown(f"**{col}**: {desc}")
    
    # Display the DataFrame with styling
    st.dataframe(styled_df, use_container_width=True)

with main_col2:
    if st.session_state['chart_symbol'] and st.session_state['chart_symbol'] in stock_histories:
        symbol = st.session_state['chart_symbol']
        st.plotly_chart(plot_candlestick(stock_histories[symbol], symbol), use_container_width=True)

# Manual Refresh Button
if st.button("🔄 Refresh Data"):
    st.rerun()

# Auto-refresh logic
if auto_refresh:
    st.info("Auto-refresh active. Data will update every 5 minutes.")
    import time
    time.sleep(300)
    st.rerun()

# Styling
st.markdown("""
<style>
.stApp {
    max-width: 100%;
}
div[data-testid="stHorizontalBlock"] {
    gap: 0.5rem;
    margin-bottom: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    run()
