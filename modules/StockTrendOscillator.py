import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# Pre-selected stocks
STOCK_LIST = {
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
    'QQQ': 'NQ100',
    'MSTR': 'Strategy'
}

def calculate_wilders_ma(data, periods):
    """Calculate Wilder's Moving Average"""
    return data.ewm(alpha=1/periods, adjust=False).mean()

def calculate_trend_oscillator(df_higher, l1=20, l2=50):
    """Calculate Trend Oscillator using higher timeframe data (1D or 5D) and return buy/sell signals"""
    # Get price changes on higher timeframe
    price_change = df_higher['Close'] - df_higher['Close'].shift(1)
    abs_price_change = abs(price_change)
    
    # Calculate Wilder's Moving Averages on higher timeframe data
    a1 = calculate_wilders_ma(price_change, l1)
    a2 = calculate_wilders_ma(abs_price_change, l1)
    
    # Calculate trend oscillator (ensure it's a pandas Series)
    a3 = np.where(a2 != 0, a1 / a2, 0)
    trend_osc_higher = pd.Series(50 * (a3 + 1), index=df_higher.index)
    
    # Calculate EMA of trend oscillator on higher timeframe data (ensure it's a pandas Series)
    ema_higher = trend_osc_higher.ewm(span=l2, adjust=False).mean()
    
    # Calculate buy/sell signals on higher timeframe data (using pandas Series)
    buy_signals_higher = ((trend_osc_higher > ema_higher) & (trend_osc_higher.shift(1) <= ema_higher.shift(1)))
    sell_signals_higher = ((trend_osc_higher < ema_higher) & (trend_osc_higher.shift(1) >= ema_higher.shift(1)))
    
    return trend_osc_higher, ema_higher, buy_signals_higher, sell_signals_higher

def create_chart(df_lower, df_higher, symbol, timeframe_lower, timeframe_higher):
    """Create interactive chart with independent lower timeframe price chart and higher timeframe trend oscillator"""
    # Calculate 21 and 50 EMAs for lower timeframe price
    df_lower['EMA21'] = df_lower['Close'].ewm(span=21, adjust=False).mean()
    df_lower['EMA50'] = df_lower['Close'].ewm(span=50, adjust=False).mean()
    
    # Calculate trend oscillator, signal line, and signals for higher timeframe data
    trend_osc_higher, ema_higher, buy_signals_higher, sell_signals_higher = calculate_trend_oscillator(df_higher)
    
    # Create figure with two independent subplots (no shared x-axes)
    fig = make_subplots(rows=2, cols=1, 
                       vertical_spacing=0.05,
                       row_heights=[0.7, 0.3])

    # Add lower timeframe candlestick chart (upper subplot)
    fig.add_trace(
        go.Candlestick(
            x=df_lower.index,
            open=df_lower['Open'],
            high=df_lower['High'],
            low=df_lower['Low'],
            close=df_lower['Close'],
            name=f'Price ({timeframe_lower})'
        ),
        row=1, col=1
    )
    
    # Add EMAs to lower timeframe price chart
    fig.add_trace(
        go.Scatter(
            x=df_lower.index,
            y=df_lower['EMA21'],
            name=f'EMA21 ({timeframe_lower})',
            line=dict(color='red', width=1),
            connectgaps=True
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df_lower.index,
            y=df_lower['EMA50'],
            name=f'EMA50 ({timeframe_lower})',
            line=dict(color='purple', width=1),
            connectgaps=True
        ),
        row=1, col=1
    )

    # Add higher timeframe Trend Oscillator and EMA to lower subplot
    fig.add_trace(
        go.Scatter(
            x=df_higher.index,
            y=trend_osc_higher,
            name=f'Trend Oscillator ({timeframe_higher})',
            line=dict(color='green', width=2),
            connectgaps=True
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df_higher.index,
            y=ema_higher,
            name=f'Signal Line ({timeframe_higher})',
            line=dict(color='red', width=2),
            connectgaps=True
        ),
        row=2, col=1
    )

    # Add horizontal lines for overbought/oversold levels on higher timeframe oscillator
    for level in [30, 50, 65]:
        fig.add_hline(
            y=level,
            line_dash="dash",
            line_color="white",
            opacity=0.5,
            row=2, col=1
        )

    # Plot buy signals on higher timeframe oscillator
    fig.add_trace(
        go.Scatter(
            x=df_higher.index[buy_signals_higher],
            y=trend_osc_higher[buy_signals_higher],
            mode='markers',
            marker=dict(
                symbol='triangle-up',
                size=12,
                color='green',
                line=dict(width=2)
            ),
            name=f'Buy Signal ({timeframe_higher})'
        ),
        row=2, col=1
    )
    
    # Plot sell signals on higher timeframe oscillator
    fig.add_trace(
        go.Scatter(
            x=df_higher.index[sell_signals_higher],
            y=trend_osc_higher[sell_signals_higher],
            mode='markers',
            marker=dict(
                symbol='triangle-down',
                size=12,
                color='red',
                line=dict(width=2)
            ),
            name=f'Sell Signal ({timeframe_higher})'
        ),
        row=2, col=1
    )

    # Filter out None values from rangebreaks
    rangebreaks = [
        dict(bounds=["sat", "mon"]) if timeframe_lower == '1H' else None,  # Hide weekends for 1H
        dict(bounds=[16, 9.5], pattern="hour") if timeframe_lower == '1H' else None  # Hide non-trading hours for 1H
    ]
    rangebreaks = [rb for rb in rangebreaks if rb is not None]

    # Update layout
    fig.update_layout(
        title=f'{symbol} - {timeframe_lower} Price Chart and {timeframe_higher} Trend Oscillator',
        yaxis_title=f'Price ({timeframe_lower})',
        yaxis2_title=f'Trend Oscillator ({timeframe_higher})',
        xaxis_rangeslider_visible=False,
        height=800,
        template='plotly_dark',
        xaxis=dict(  # Lower timeframe x-axis (1H or 1D)
            rangebreaks=rangebreaks
        ),
        xaxis2=dict(  # Higher timeframe x-axis (1D or 5D)
            title=f'Date ({timeframe_higher})'
        )
    )
    
    # Update y-axes
    fig.update_yaxes(title_text=f'Price ({timeframe_lower})', row=1, col=1)
    fig.update_yaxes(title_text=f'Trend Oscillator ({timeframe_higher})', row=2, col=1, range=[0, 100])  # Ensure y-axis range is 0-100

    return fig

def show_trend_oscillator():
    """Main function to show the trend oscillator in the Streamlit app"""
    st.header('Multi-Stock Trend Oscillator Dashboard')

    st.write("""
    A Streamlit-based interactive stock analysis dashboard that combines price action with a custom trend oscillator.
    
    Key features:
    - Pre-selected major stocks (AAPL, MSFT, GOOGL, etc.) and custom symbol input
    - Choose between 1-hour/1-day or 1-day/5-day setups
    - Interactive price charts with 21 and 50 EMAs
    - Trend oscillator with signal line and buy/sell indicators
    """)
    
    # User selection for timeframe setup
    timeframe_setup = st.radio(
        "Select Timeframe Setup:",
        ('1H/1D', '1D/5D')
    )
    
    # Create a grid of stock buttons
    cols = st.columns(5)
    selected_stock = None
    
    for idx, (symbol, name) in enumerate(STOCK_LIST.items()):
        col_idx = idx % 5
        if cols[col_idx].button(f'{symbol}\n{name}', use_container_width=True):
            selected_stock = symbol
    
    # Additional stock input
    custom_stock = st.text_input('Enter another stock symbol:', '')
    if custom_stock:
        selected_stock = custom_stock.upper()
    
    if selected_stock:
        try:
            with st.spinner('Fetching and analyzing data...'):
                # Fetch data
                end_date = datetime.now()
                start_date = end_date - timedelta(days=180)  # Fetch more data (increased to 180 days for 5D)
                
                if timeframe_setup == '1H/1D':
                    # Fetch 60-minute data (used as 1-hour) for price chart
                    stock = yf.Ticker(selected_stock)
                    df_60m = stock.history(start=start_date, end=end_date, interval="60m")
                    
                    # Ensure indices are DatetimeIndex and handle timezone
                    if not isinstance(df_60m.index, pd.DatetimeIndex):
                        df_60m.index = pd.DatetimeIndex(df_60m.index)
                    
                    if df_60m.index.tz is not None:
                        df_60m.index = df_60m.index.tz_convert('America/New_York')
                    else:
                        df_60m.index = df_60m.index.tz_localize('UTC').tz_convert('America/New_York')
                    
                    # Filter 60-minute data for market hours (9:30 AM to 4:00 PM ET) and remove weekends
                    df_60m = df_60m.between_time('09:30', '16:00')
                    df_60m = df_60m[df_60m.index.dayofweek < 5]
                    
                    # Keep only last 60 days of trading data for 60-minute
                    df_60m = df_60m.last('60D')
                    
                    # Use 60-minute data as 1-hour price chart
                    df_lower = df_60m.copy()
                    
                    # Fetch 1-day data for 1-day oscillator (directly, no resampling needed)
                    df_1d = stock.history(start=start_date, end=end_date, interval="1d")
                    
                    # Ensure indices are DatetimeIndex and handle timezone
                    if not isinstance(df_1d.index, pd.DatetimeIndex):
                        df_1d.index = pd.DatetimeIndex(df_1d.index)
                    
                    if df_1d.index.tz is not None:
                        df_1d.index = df_1d.index.tz_convert('America/New_York')
                    else:
                        df_1d.index = df_1d.index.tz_localize('UTC').tz_convert('America/New_York')
                    
                    # Remove weekends from 1-day data
                    df_1d = df_1d[df_1d.index.dayofweek < 5]
                    
                    # Keep only last 60 days of trading data for 1-day
                    df_1d = df_1d.last('60D')
                    
                    # Use 1-day data as higher timeframe for oscillator
                    df_higher = df_1d.copy()
                    
                    if df_lower.empty or df_higher.empty:
                        st.error('No data found for the specified stock (1H/1D).')
                        return
                    
                    # Create and display chart
                    fig = create_chart(df_lower, df_higher, selected_stock, '1H', '1D')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display current indicator values (from 1-day oscillator)
                    trend_osc_1d, ema_1d, _, _ = calculate_trend_oscillator(df_higher)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Current Trend Oscillator (1D)", f"{trend_osc_1d.iloc[-1]:.2f}")
                    with col2:
                        st.metric("Current Signal Line (1D)", f"{ema_1d.iloc[-1]:.2f}")
                    with col3:
                        trend = "Bullish" if trend_osc_1d.iloc[-1] > ema_1d.iloc[-1] else "Bearish"
                        st.metric("Current Trend (1D)", trend)
                
                else:  # 1D/5D setup
                    # Fetch 1-day data for price chart
                    stock = yf.Ticker(selected_stock)
                    df_1d = stock.history(start=start_date, end=end_date, interval="1d")
                    
                    # Ensure indices are DatetimeIndex and handle timezone
                    if not isinstance(df_1d.index, pd.DatetimeIndex):
                        df_1d.index = pd.DatetimeIndex(df_1d.index)
                    
                    if df_1d.index.tz is not None:
                        df_1d.index = df_1d.index.tz_convert('America/New_York')
                    else:
                        df_1d.index = df_1d.index.tz_localize('UTC').tz_convert('America/New_York')
                    
                    # Remove weekends from 1-day data
                    df_1d = df_1d[df_1d.index.dayofweek < 5]
                    
                    # Keep only last 180 days of trading data for 1-day (to ensure enough data for 5D)
                    df_1d = df_1d.last('180D')
                    
                    # Use 1-day data as 1-day price chart
                    df_lower = df_1d.copy()
                    
                    # Fetch 5-day data for 5-day oscillator (directly, as 5d is supported)
                    df_5d = stock.history(start=start_date, end=end_date, interval="5d")
                    
                    # Ensure indices are DatetimeIndex and handle timezone
                    if not isinstance(df_5d.index, pd.DatetimeIndex):
                        df_5d.index = pd.DatetimeIndex(df_5d.index)
                    
                    if df_5d.index.tz is not None:
                        df_5d.index = df_5d.index.tz_convert('America/New_York')
                    else:
                        df_5d.index = df_5d.index.tz_localize('UTC').tz_convert('America/New_York')
                    
                    # Remove weekends from 5-day data (optional, as 5d data typically only includes trading weeks)
                    df_5d = df_5d[df_5d.index.dayofweek < 5]
                    
                    # Keep only last 180 days of trading data for 5-day
                    df_5d = df_5d.last('180D')
                    
                    # Use 5-day data as higher timeframe for oscillator
                    df_higher = df_5d.copy()
                    
                    if df_lower.empty or df_higher.empty:
                        st.error('No data found for the specified stock (1D/5D).')
                        return
                    
                    # Create and display chart
                    fig = create_chart(df_lower, df_higher, selected_stock, '1D', '5D')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display current indicator values (from 5-day oscillator)
                    trend_osc_5d, ema_5d, _, _ = calculate_trend_oscillator(df_higher)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Current Trend Oscillator (5D)", f"{trend_osc_5d.iloc[-1]:.2f}")
                    with col2:
                        st.metric("Current Signal Line (5D)", f"{ema_5d.iloc[-1]:.2f}")
                    with col3:
                        trend = "Bullish" if trend_osc_5d.iloc[-1] > ema_5d.iloc[-1] else "Bearish"
                        st.metric("Current Trend (5D)", trend)
                
        except Exception as e:
            st.error(f'Error occurred: {str(e)}')

if __name__ == "__main__":
    show_trend_oscillator()
