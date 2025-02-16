import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from math import ceil
from datetime import datetime, timedelta

# Constants
DEFAULT_STOCK_LIST = {
    'AAPL': 'Apple Inc.',
    'MSFT': 'Microsoft',
    'GOOGL': 'Alphabet',
    'AMZN': 'Amazon',
    'NVDA': 'NVIDIA',
    'META': 'Meta Platforms',
    'TSLA': 'Tesla',
    'AMD': 'AMD',
    'NFLX': 'Netflix',
    'SPY': 'SP500',
    'QQQ': 'NQ100'
}

# Technical Analysis Functions
def calculate_rsi(data, periods=14):
    """Calculate Relative Strength Index"""
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_wilders_ma(data, periods):
    """Calculate Wilder's Moving Average"""
    return data.ewm(alpha=1/periods, adjust=False).mean()

def calculate_trend_oscillator(df, l1=20, l2=50):
    """Calculate Trend Oscillator using higher timeframe data"""
    price_change = df['Close'] - df['Close'].shift(1)
    abs_price_change = abs(price_change)
    
    a1 = calculate_wilders_ma(price_change, l1)
    a2 = calculate_wilders_ma(abs_price_change, l1)
    
    a3 = np.where(a2 != 0, a1 / a2, 0)
    trend_osc = 50 * (a3 + 1)
    
    ema = pd.Series(trend_osc).ewm(span=l2, adjust=False).mean()
    
    return pd.Series(trend_osc), pd.Series(ema)

def calculate_relative_strength(symbol_hist, spy_hist, lookback=20):
    """Calculate relative strength compared to SPY"""
    common_dates = symbol_hist.index.intersection(spy_hist.index)
    if len(common_dates) < 2:
        return 0, "N/A"
    
    symbol_change = symbol_hist['Close'].loc[common_dates].pct_change(periods=lookback).iloc[-1]
    spy_change = spy_hist['Close'].loc[common_dates].pct_change(periods=lookback).iloc[-1]
    
    if spy_change != 0:
        rs = ((1 + symbol_change) / (1 + spy_change) - 1) * 100
        rs_status = "Strong" if rs > 0 else "Weak"
        return round(rs, 2), rs_status
    return 0, "N/A"

def get_rsi_status(rsi):
    """Get RSI status based on value"""
    if rsi > 70:
        return "Overbought"
    elif rsi > 50:
        return "Strong"
    elif rsi > 30:
        return "Weak"
    else:
        return "Oversold"

# Data Fetching and Processing
def fetch_stock_data(symbol, period="1d", interval="5m", spy_hist=None):
    """Fetch and process stock data with technical indicators"""
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period, interval=interval)
        if hist.empty:
            raise ValueError(f"No data available for {symbol}")
        
        # Calculate basic indicators
        hist['Cumulative_Volume'] = hist['Volume'].cumsum()
        hist['Cumulative_PV'] = (hist['Close'] * hist['Volume']).cumsum()
        hist['VWAP'] = hist['Cumulative_PV'] / hist['Cumulative_Volume']
        hist['RSI'] = calculate_rsi(hist)
        hist['EMA21'] = hist['Close'].ewm(span=21, adjust=False).mean()
        hist['EMA9'] = hist['Close'].ewm(span=9, adjust=False).mean()
        hist['EMA50'] = hist['Close'].ewm(span=50, adjust=False).mean()
        
        # Calculate relative strength
        rs_value, rs_status = calculate_relative_strength(hist, spy_hist) if spy_hist is not None else (0, "N/A")
        
        # Calculate price levels
        today_data = hist.iloc[-1]
        current_price = round(today_data["Close"], 2)
        vwap = round(hist['VWAP'].iloc[-1], 2)
        ema_21 = round(hist['EMA21'].iloc[-1], 2)
        daily_pivot = round((hist["High"].iloc[-1] + hist["Low"].iloc[-1] + current_price) / 3, 2)
        
        # Determine market conditions
        if current_price > hist['EMA9'].iloc[-1] and current_price > ema_21 and current_price > hist['EMA50'].iloc[-1]:
            key_mas = "Bullish"
        elif current_price < hist['EMA9'].iloc[-1] and current_price < ema_21 and current_price < hist['EMA50'].iloc[-1]:
            key_mas = "Bearish"
        else:
            key_mas = "Mixed"
            
        if current_price > vwap and current_price > today_data["Open"]:
            direction = "Bullish"
        elif current_price < vwap and current_price < today_data["Open"]:
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
            "RSI_Status": [get_rsi_status(hist['RSI'].iloc[-1])]
        }), hist.round(2)
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame(), pd.DataFrame()

# Visualization Functions
def plot_candlestick(data, symbol, show_trend_oscillator=False):
    """Create candlestick chart with optional trend oscillator"""
    if show_trend_oscillator:
        fig = make_subplots(rows=2, cols=1, 
                           shared_xaxes=True,
                           vertical_spacing=0.05,
                           row_heights=[0.7, 0.3])
    else:
        fig = go.Figure()

    # Add candlestick
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name=symbol
        ),
        row=1, col=1
    )
    
    # Add VWAP and EMAs
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['VWAP'],
            name='VWAP',
            line=dict(color='purple', width=2)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['EMA21'],
            name='21 EMA',
            line=dict(color='orange', width=2)
        ),
        row=1, col=1
    )

    if show_trend_oscillator:
        # Add Trend Oscillator
        trend_osc, ema = calculate_trend_oscillator(data)
        
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=trend_osc,
                name='Trend Oscillator',
                line=dict(color='green', width=2)
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=ema,
                name='Signal Line',
                line=dict(color='red', width=2)
            ),
            row=2, col=1
        )

        # Add buy/sell signals
        buy_signals = ((trend_osc > ema) & (trend_osc.shift(1) <= ema.shift(1)))
        sell_signals = ((trend_osc < ema) & (trend_osc.shift(1) >= ema.shift(1)))
        
        fig.add_trace(
            go.Scatter(
                x=data.index[buy_signals],
                y=trend_osc[buy_signals],
                mode='markers',
                marker=dict(symbol='triangle-up', size=12, color='green'),
                name='Buy Signal'
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=data.index[sell_signals],
                y=trend_osc[sell_signals],
                mode='markers',
                marker=dict(symbol='triangle-down', size=12, color='red'),
                name='Sell Signal'
            ),
            row=2, col=1
        )

    # Update layout
    title = f'{symbol} Chart with Indicators'
    if show_trend_oscillator:
        title += ' and Trend Oscillator'
    
    fig.update_layout(
        title=title,
        yaxis_title='Price',
        template='plotly_white' if not show_trend_oscillator else 'plotly_dark',
        height=800 if show_trend_oscillator else 600,
        xaxis_rangeslider_visible=False
    )
    
    return fig

def run():
    """Main application function"""
    # Page configuration for better spacing
    #st.set_page_config(layout="wide", page_title="Stock Analysis Dashboard")
    
    # Custom CSS for better spacing and styling
    st.markdown("""
        <style>
        .stButton button {
            width: 100%;
            padding: 0.5rem;
            background-color: #2c3e50;
            color: white;
            border: none;
            border-radius: 0.5rem;
        }
        .stButton button:hover {
            background-color: #34495e;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        div[data-testid="stVerticalBlock"] > div {
            padding: 0.5rem 0;
        }
        div[data-baseweb="select"] > div {
            background-color: #2c3e50;
            border-radius: 0.5rem;
        }
        .stTextArea textarea {
            background-color: #2c3e50;
            color: white;
            border-radius: 0.5rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'chart_symbol' not in st.session_state:
        st.session_state['chart_symbol'] = None
    
    # Main title with custom styling
    st.markdown("""
        <h1 style='text-align: center; color: #3498db; padding: 1rem 0;'>
            📈 Stock Analysis Dashboard
        </h1>
    """, unsafe_allow_html=True)
    
    # Settings section with better spacing
    st.markdown("### 🛠️ Dashboard Configuration")
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Layout for settings with improved spacing
    settings_col1, settings_col2, settings_col3, settings_col4 = st.columns([3, 2, 2, 2])
    
    with settings_col1:
        st.markdown("##### Enter Stock Symbols")
        stock_list = st.text_area(
            "",  # Remove label as we're using markdown above
            "SPY, QQQ, AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA",
            height=100
        ).upper()
        symbols = [s.strip() for s in stock_list.split(",")]

    with settings_col2:
        st.markdown("##### Time Frame")
        time_frames = {
            "1 Day": "1d", "5 Days": "5d", "1 Month": "1mo",
            "3 Months": "3mo", "6 Months": "6mo", "1 Year": "1y"
        }
        selected_timeframe = st.selectbox("", list(time_frames.keys()), index=0)
        period = time_frames[selected_timeframe]

    with settings_col3:
        st.markdown("##### Chart Interval")
        intervals = ["1m", "5m", "15m", "30m", "1h", "1d"]
        selected_interval = st.selectbox("", intervals, index=1)

    with settings_col4:
        st.markdown("##### Additional Options")
        show_oscillator = st.checkbox("Show Trend Oscillator", value=True)
        st.markdown("<br>", unsafe_allow_html=True)
        auto_refresh = st.checkbox("Auto Refresh (5m)")

    # Divider
    st.markdown("<hr style='margin: 2rem 0; border-color: #3498db33;'>", unsafe_allow_html=True)

    # Data display section with improved title
    st.markdown(f"""
        <h3 style='color: #3498db; display: flex; align-items: center; gap: 0.5rem;'>
            📊 Market Data <span style='color: #7f8c8d; font-size: 1rem;'>
            ({selected_timeframe}, {selected_interval})</span>
        </h3>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)

    # Main content columns with better proportions
    main_col1, main_col2 = st.columns([1.2, 1])

    with main_col1:
        # Fetch SPY data first for relative strength calculations
        spy_data, spy_hist = fetch_stock_data("SPY", period=period, interval=selected_interval)
        stock_histories = {"SPY": spy_hist}
        all_data = pd.DataFrame()
        
        # Create an improved grid layout for stock buttons
        num_symbols = len(symbols)
        num_cols = 4  # Increased number of columns for better spacing
        num_rows = ceil(num_symbols / num_cols)
        
        st.markdown("""
            <style>
            div[data-testid="column"] > div > div > div > div > div[data-testid="stHorizontalBlock"] {
                gap: 1rem;
                margin-bottom: 1rem;
            }
            </style>
        """, unsafe_allow_html=True)
        
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
                        data, history = fetch_stock_data(
                            symbol, 
                            period=period, 
                            interval=selected_interval, 
                            spy_hist=spy_hist
                        )
                    
                    if not data.empty:
                        all_data = pd.concat([all_data, data], ignore_index=True)
                        stock_histories[symbol] = history
                        
                        # Enhanced button styling with conditional colors
                        if not data.empty:
                            price_direction = data.iloc[0]['Price_Vwap']
                            button_color = ('#2ecc71' if price_direction == 'Bullish' 
                                          else '#e74c3c' if price_direction == 'Bearish' 
                                          else '#7f8c8d')
                            
                            # Create styled button container
                            button_container = cols[j].container()
                            
                            # Add styled button with background color based on price direction
                            if button_container.button(
                                f"📈 {symbol}",
                                key=f'btn_{symbol}',
                                help=f"Click to view {symbol} chart",
                                use_container_width=True
                            ):
                                st.session_state['chart_symbol'] = symbol
                            
                            # Add custom styling for the button
                            button_container.markdown(f"""
                                <style>
                                    div[data-testid="stButton"] button {{
                                        background-color: {button_color};
                                        border: none;
                                        padding: 0.5rem;
                                        font-weight: bold;
                                        color: white;
                                        transition: all 0.3s ease;
                                    }}
                                    div[data-testid="stButton"] button:hover {{
                                        filter: brightness(1.2);
                                        transform: translateY(-2px);
                                    }}
                                </style>
                            """, unsafe_allow_html=True)

        # Display data table with styling
        if not all_data.empty:
            # Enhanced styling for the dataframe
            styled_df = all_data.style.format({
                'Current Price': '${:.2f}',
                'VWAP': '${:.2f}',
                'Daily Pivot': '${:.2f}',
                'EMA21': '${:.2f}'
            }).applymap(
                lambda val: (
                    'background-color: rgba(46, 204, 113, 0.2); color: #2ecc71; font-weight: bold;' 
                        if val in ["Bullish", "Strong"]
                    else 'background-color: rgba(231, 76, 60, 0.2); color: #e74c3c; font-weight: bold;' 
                        if val in ["Bearish", "Weak"]
                    else 'background-color: rgba(149, 165, 166, 0.2); color: #95a5a6; font-weight: bold;' 
                        if val in ["Neutral", "Mixed"]
                    else 'background-color: rgba(142, 68, 173, 0.2); color: #8e44ad; font-weight: bold;' 
                        if val == "Overbought"
                    else 'background-color: rgba(52, 152, 219, 0.2); color: #3498db; font-weight: bold;' 
                        if val == "Oversold"
                    else ''
                ),
                subset=['Price_Vwap', 'KeyMAs', 'RSI_Status', 'Rel Strength SPY']
            ).set_properties(**{
                'background-color': '#1a1a1a',
                'color': '#ffffff',
                'border-color': '#2c3e50'
            }).set_table_styles([
                {'selector': 'th', 'props': [
                    ('background-color', '#2c3e50'),
                    ('color', '#ffffff'),
                    ('font-weight', 'bold'),
                    ('padding', '12px'),
                    ('border', '1px solid #34495e')
                ]},
                {'selector': 'td', 'props': [
                    ('padding', '12px'),
                    ('border', '1px solid #34495e')
                ]}
            ])
            
            st.dataframe(styled_df, use_container_width=True)

    # Chart display
    with main_col2:
        if st.session_state['chart_symbol'] and st.session_state['chart_symbol'] in stock_histories:
            symbol = st.session_state['chart_symbol']
            fig = plot_candlestick(stock_histories[symbol], symbol, show_oscillator)
            st.plotly_chart(fig, use_container_width=True)
            
            if show_oscillator:
                # Display trend oscillator metrics
                trend_osc, ema = calculate_trend_oscillator(stock_histories[symbol])
                
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                with metric_col1:
                    st.metric("Trend Oscillator", f"{trend_osc.iloc[-1]:.2f}")
                with metric_col2:
                    st.metric("Signal Line", f"{ema.iloc[-1]:.2f}")
                with metric_col3:
                    trend = "Bullish" if trend_osc.iloc[-1] > ema.iloc[-1] else "Bearish"
                    st.metric("Current Trend", trend)

    # Tooltip information
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

    with st.expander("ℹ️ Column Descriptions"):
        for col, desc in tooltips.items():
            st.markdown(f"**{col}**: {desc}")

    # Refresh controls
    if st.button("🔄 Refresh Data"):
        st.rerun()

    if auto_refresh:
        st.info("Auto-refresh active. Data will update every 5 minutes.")
        import time
        time.sleep(300)
        st.rerun()

if __name__ == "__main__":
    run()
