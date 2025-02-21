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
    'COIN': 'COIN'
}

def calculate_wilders_ma(data, periods):
    """Calculate Wilder's Moving Average, matching ThinkScript's implementation"""
    # Ensure data is a pandas Series; fill NaN with forward-fill to maintain length
    if data.isna().any():
        data = data.ffill()
    return data.ewm(alpha=1/periods, adjust=False).mean()

def calculate_trend_oscillator(df_higher, l1, l2):
    """Calculate Trend Oscillator using higher timeframe data (1D or 5D) and return buy/sell signals, matching ThinkScript"""
    # Ensure df_higher has no NaN values in 'Close' column, but preserve index length
    df_higher = df_higher.copy()
    if df_higher['Close'].isna().any():
        df_higher['Close'] = df_higher['Close'].ffill().bfill()  # Forward and backward fill to maintain length
    
    # Get price changes on higher timeframe (close - close[1]), preserving index
    price_change = df_higher['Close'] - df_higher['Close'].shift(1)
    abs_price_change = abs(price_change)
    
    # Calculate Wilder's Moving Averages, ensuring no NaN and maintaining index length
    a1 = calculate_wilders_ma(price_change, l1).reindex(df_higher.index, method='ffill').fillna(0)
    a2 = calculate_wilders_ma(abs_price_change, l1).reindex(df_higher.index, method='ffill').fillna(0)
    
    # Calculate A3 and Trend Oscillator, matching ThinkScript's logic, ensuring same length as df_higher.index
    a3 = np.where(a2 != 0, a1 / a2, 0)
    trend_oscillator = pd.Series(50 * (a3 + 1), index=df_higher.index).fillna(50)  # Fill NaN with 50 (midline) as default
    
    # Calculate EMA of Trend Oscillator (signal line), matching ThinkScript's ExpAverage, ensuring same length
    ema = trend_oscillator.ewm(span=l2, adjust=False).mean().reindex(df_higher.index, method='ffill').fillna(50)  # Fill NaN with 50 (midline)
    
    # Calculate buy/sell signals (crossings of TrendOscillator and EMA), matching ThinkScript's arrows
    # Ensure signals maintain the same index length, filling with False where needed
    trend_oscillator_shifted = trend_oscillator.shift(1).reindex(df_higher.index, method='ffill').fillna(50)
    ema_shifted = ema.shift(1).reindex(df_higher.index, method='ffill').fillna(50)
    
    buy_signals = ((trend_oscillator > ema) & (trend_oscillator_shifted <= ema_shifted))
    sell_signals = ((trend_oscillator < ema) & (trend_oscillator_shifted >= ema_shifted))
    
    # Reindex signals to match df_higher.index, filling with False
    buy_signals = buy_signals.reindex(df_higher.index, method='ffill').fillna(False)
    sell_signals = sell_signals.reindex(df_higher.index, method='ffill').fillna(False)
    
    # Debug prints to check lengths (remove or comment out in production)
    print(f"df_higher.index length: {len(df_higher.index)}")
    print(f"trend_oscillator length: {len(trend_oscillator)}")
    print(f"ema length: {len(ema)}")
    print(f"buy_signals length: {len(buy_signals)}")
    print(f"sell_signals length: {len(sell_signals)}")
    
    return trend_oscillator, ema, buy_signals, sell_signals

def create_chart(df_lower, df_higher, symbol, timeframe_lower, timeframe_higher):
    """Create interactive chart with independent lower timeframe price chart and higher timeframe trend oscillator, matching ThinkScript"""
    # Determine L1 and L2 based on ThinkScript logic for the lower timeframe
    if timeframe_lower == '1H':
        l1, l2 = 20, 30  # For 1H, use 1D higher timeframe with L1=20, L2=30 (from ThinkScript for hourly)
    else:  # 1D
        l1, l2 = 20, 35  # For 1D, use 5D higher timeframe with L1=20, L2=35 (from ThinkScript for daily)

    # Calculate 21 and 50 EMAs for lower timeframe price
    df_lower['EMA21'] = df_lower['Close'].ewm(span=21, adjust=False).mean()
    df_lower['EMA50'] = df_lower['Close'].ewm(span=50, adjust=False).mean()
    
    # Calculate trend oscillator, signal line, and signals for higher timeframe data
    trend_oscillator, ema, buy_signals, sell_signals = calculate_trend_oscillator(df_higher, l1, l2)
    
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

    # Add higher timeframe Trend Oscillator and EMA to lower subplot, matching ThinkScript line weights
    fig.add_trace(
        go.Scatter(
            x=df_higher.index,
            y=trend_oscillator,
            name=f'Trend Oscillator ({timeframe_higher})',
            line=dict(color='green', width=5),  # Use single green color, matching ThinkScript
            connectgaps=True
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df_higher.index,
            y=ema,
            name=f'Signal Line ({timeframe_higher})',
            line=dict(color='red', width=5),  # Match ThinkScript line weight
            connectgaps=True
        ),
        row=2, col=1
    )

    # Add horizontal lines for overbought (65), oversold (30), and midline (50), matching ThinkScript
    for level in [30, 50, 65]:
        fig.add_hline(
            y=level,
            line_dash="dash" if level in [30, 65] else "solid",  # Dashed for overbought/oversold, solid for midline
            line_color="white",
            opacity=0.7,  # Increase opacity to match ThinkScript visibility
            row=2, col=1
        )

    # Plot buy signals (upward green triangles) and sell signals (downward red triangles) on higher timeframe, matching ThinkScript arrows
    fig.add_trace(
        go.Scatter(
            x=df_higher.index[buy_signals],
            y=trend_oscillator[buy_signals],
            mode='markers',
            marker=dict(
                symbol='triangle-up',
                size=12,
                color='green',
                line=dict(width=1)  # Match ThinkScript line weight
            ),
            name=f'Buy Signal ({timeframe_higher})'
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df_higher.index[sell_signals],
            y=trend_oscillator[sell_signals],
            mode='markers',
            marker=dict(
                symbol='triangle-down',
                size=12,
                color='red',
                line=dict(width=1)  # Match ThinkScript line weight
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

    # Update layout to match ThinkScript's dark background and styling
    fig.update_layout(
        title=f'{symbol} - {timeframe_lower} Price Chart and {timeframe_higher} Trend Oscillator',
        yaxis_title=f'Price ({timeframe_lower})',
        yaxis2_title=f'Trend Oscillator ({timeframe_higher})',
        xaxis_rangeslider_visible=False,
        height=800,
        template='plotly_dark',
        plot_bgcolor='black',  # Match ThinkScript's dark background
        paper_bgcolor='black',  # Match ThinkScript's dark background
        xaxis=dict(  # Lower timeframe x-axis (1H or 1D)
            rangebreaks=rangebreaks,
            showgrid=False,  # Remove gridlines to match ThinkScript
            zeroline=False
        ),
        xaxis2=dict(  # Higher timeframe x-axis (1D or 5D)
            title=f'Date ({timeframe_higher})',
            showgrid=False,  # Remove gridlines to match ThinkScript
            zeroline=False
        ),
        yaxis=dict(showgrid=False, zeroline=False),  # Remove gridlines for price chart
        yaxis2=dict(showgrid=False, zeroline=False)  # Remove gridlines for oscillator
    )
    
    # Update y-axes
    fig.update_yaxes(title_text=f'Price ({timeframe_lower})', row=1, col=1)
    fig.update_yaxes(title_text=f'Trend Oscillator ({timeframe_higher})', row=2, col=1, range=[0, 100])  # Ensure y-axis range is 30-65

    return fig

def show_trend_oscillator():
    """Main function to show the trend oscillator in the Streamlit app"""
    st.header('Multi-Stock Trend Oscillator Dashboard')

    st.write("""
    A Streamlit-based interactive stock analysis dashboard that combines price action with a custom trend oscillator, matching ThinkScript logic.
    
    Key features:
    - Pre-selected major stocks (AAPL, MSFT, GOOGL, etc.) and custom symbol input
    - Choose between 1-hour/1-day or 1-day/5-day setups
    - Interactive price charts with 21 and 50 EMAs
    - Trend oscillator with signal line, buy/sell signals, and overbought/oversold levels matching ThinkScript
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
                    trend_osc_1d, ema_1d, _, _ = calculate_trend_oscillator(df_higher, 20, 30)  # L1=20, L2=30 for 1H (1D oscillator)
                    
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
                    trend_osc_5d, ema_5d, _, _ = calculate_trend_oscillator(df_higher, 20, 35)  # L1=20, L2=35 for 1D (5D oscillator)
                    
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
