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
    'QQQ': 'NQ100'
}

def calculate_wilders_ma(data, periods):
    """Calculate Wilder's Moving Average"""
    return data.ewm(alpha=1/periods, adjust=False).mean()

def get_higher_timeframe_data(df, interval='4H'):
    """Resample data to higher timeframe"""
    resampled = df['Close'].resample(interval).agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last'
    })
    return resampled

def calculate_trend_oscillator(df_4h, l1=20, l2=50):
    """Calculate Trend Oscillator using 4-hour timeframe data and return buy/sell signals"""
    # Get price changes on 4-hour timeframe
    price_change = df_4h['Close'] - df_4h['Close'].shift(1)
    abs_price_change = abs(price_change)
    
    # Calculate Wilder's Moving Averages on 4-hour data
    a1 = calculate_wilders_ma(price_change, l1)
    a2 = calculate_wilders_ma(abs_price_change, l1)
    
    # Calculate trend oscillator (ensure it's a pandas Series)
    a3 = np.where(a2 != 0, a1 / a2, 0)
    trend_osc_4h = pd.Series(50 * (a3 + 1), index=df_4h.index)
    
    # Calculate EMA of trend oscillator on 4-hour data (ensure it's a pandas Series)
    ema_4h = trend_osc_4h.ewm(span=l2, adjust=False).mean()
    
    # Calculate buy/sell signals on 4-hour data (using pandas Series)
    buy_signals_4h = ((trend_osc_4h > ema_4h) & (trend_osc_4h.shift(1) <= ema_4h.shift(1)))
    sell_signals_4h = ((trend_osc_4h < ema_4h) & (trend_osc_4h.shift(1) >= ema_4h.shift(1)))
    
    return trend_osc_4h, ema_4h, buy_signals_4h, sell_signals_4h

def create_chart(df_1h, df_4h, symbol):
    """Create interactive chart with independent 1-hour price chart and 4-hour trend oscillator"""
    # Calculate 21 and 50 EMAs for 1-hour price (still on 1-hour timeframe)
    df_1h['EMA21'] = df_1h['Close'].ewm(span=21, adjust=False).mean()
    df_1h['EMA50'] = df_1h['Close'].ewm(span=50, adjust=False).mean()
    
    # Calculate trend oscillator, signal line, and signals for 4-hour data
    trend_osc_4h, ema_4h, buy_signals_4h, sell_signals_4h = calculate_trend_oscillator(df_4h)
    
    # Create figure with two independent subplots (no shared x-axes)
    fig = make_subplots(rows=2, cols=1, 
                       vertical_spacing=0.05,
                       row_heights=[0.7, 0.3])

    # Add 1-hour candlestick chart (upper subplot)
    fig.add_trace(
        go.Candlestick(
            x=df_1h.index,
            open=df_1h['Open'],
            high=df_1h['High'],
            low=df_1h['Low'],
            close=df_1h['Close'],
            name='Price (1H)'
        ),
        row=1, col=1
    )
    
    # Add EMAs to 1-hour price chart
    fig.add_trace(
        go.Scatter(
            x=df_1h.index,
            y=df_1h['EMA21'],
            name='EMA21 (1H)',
            line=dict(color='red', width=1),
            connectgaps=True
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df_1h.index,
            y=df_1h['EMA50'],
            name='EMA50 (1H)',
            line=dict(color='purple', width=1),
            connectgaps=True
        ),
        row=1, col=1
    )

    # Add 4-hour Trend Oscillator and EMA to lower subplot
    fig.add_trace(
        go.Scatter(
            x=df_4h.index,
            y=trend_osc_4h,
            name='Trend Oscillator (4H)',
            line=dict(color='green', width=2),
            connectgaps=True
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df_4h.index,
            y=ema_4h,
            name='Signal Line (4H)',
            line=dict(color='red', width=2),
            connectgaps=True
        ),
        row=2, col=1
    )

    # Add horizontal lines for overbought/oversold levels on 4-hour oscillator
    for level in [30, 50, 65]:
        fig.add_hline(
            y=level,
            line_dash="dash",
            line_color="white",
            opacity=0.5,
            row=2, col=1
        )

    # Plot buy signals on 4-hour oscillator
    fig.add_trace(
        go.Scatter(
            x=df_4h.index[buy_signals_4h],
            y=trend_osc_4h[buy_signals_4h],
            mode='markers',
            marker=dict(
                symbol='triangle-up',
                size=12,
                color='green',
                line=dict(width=2)
            ),
            name='Buy Signal (4H)'
        ),
        row=2, col=1
    )
    
    # Plot sell signals on 4-hour oscillator
    fig.add_trace(
        go.Scatter(
            x=df_4h.index[sell_signals_4h],
            y=trend_osc_4h[sell_signals_4h],
            mode='markers',
            marker=dict(
                symbol='triangle-down',
                size=12,
                color='red',
                line=dict(width=2)
            ),
            name='Sell Signal (4H)'
        ),
        row=2, col=1
    )

    # Update layout
    fig.update_layout(
        title=f'{symbol} - 1H Price Chart and 4H Trend Oscillator',
        yaxis_title='Price (1H)',
        yaxis2_title='Trend Oscillator (4H)',
        xaxis_rangeslider_visible=False,
        height=800,
        template='plotly_dark',
        xaxis=dict(  # 1-hour x-axis
            rangebreaks=[
                dict(bounds=["sat", "mon"]),  # hide weekends
                dict(bounds=[16, 9.5], pattern="hour"),  # hide non-trading hours
            ]
        ),
        xaxis2=dict(  # 4-hour x-axis (no rangebreaks since it's 4-hour data)
            title='Date (4H)'
        )
    )
    
    # Update y-axes
    fig.update_yaxes(title_text="Price (1H)", row=1, col=1)
    fig.update_yaxes(title_text="Trend Oscillator (4H)", row=2, col=1, range=[0, 100])  # Ensure y-axis range is 0-100

    return fig

def show_trend_oscillator():
    """Main function to show the trend oscillator in the Streamlit app"""
    st.header('Multi-Stock Trend Oscillator Dashboard')

    st.write("""
    A Streamlit-based interactive stock analysis dashboard that combines price action with a custom trend oscillator.
    
    Key features:
    - Pre-selected major stocks (AAPL, MSFT, GOOGL, etc.) and custom symbol input
    - Interactive 1-hour candlestick charts with 21 and 50 EMAs
    - 4-hour trend oscillator with signal line and buy/sell indicators
    """)
    
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
                start_date = end_date - timedelta(days=60)  # Fetch more data to account for non-trading days
                
                # Fetch 1-hour data for price chart
                stock = yf.Ticker(selected_stock)
                df_1h = stock.history(start=start_date, end=end_date, interval='1h')
                
                # Fetch 4-hour data for trend oscillator
                df_4h = stock.history(start=start_date, end=end_date, interval='4h')
                
                # Filter 1-hour data for market hours (9:30 AM to 4:00 PM ET)
                df_1h.index = df_1h.index.tz_localize(None)
                df_1h = df_1h.between_time('09:30', '16:00')
                
                # Remove weekends from 1-hour data
                df_1h = df_1h[df_1h.index.dayofweek < 5]
                
                # Keep only last 30 days of trading data for 1-hour
                df_1h = df_1h.last('30D')
                
                # Filter 4-hour data (remove weekends if needed, but typically 4-hour data already accounts for trading hours)
                df_4h.index = df_4h.index.tz_localize(None)
                df_4h = df_4h[df_4h.index.dayofweek < 5]
                df_4h = df_4h.last('30D')
                
                if df_1h.empty or df_4h.empty:
                    st.error('No data found for the specified stock.')
                    return
                
                # Create and display chart
                fig = create_chart(df_1h, df_4h, selected_stock)
                st.plotly_chart(fig, use_container_width=True)
                
                # Display current indicator values (from 4-hour oscillator)
                trend_osc_4h, ema_4h, _, _ = calculate_trend_oscillator(df_4h)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Current Trend Oscillator (4H)", f"{trend_osc_4h.iloc[-1]:.2f}")
                with col2:
                    st.metric("Current Signal Line (4H)", f"{ema_4h.iloc[-1]:.2f}")
                with col3:
                    trend = "Bullish" if trend_osc_4h.iloc[-1] > ema_4h.iloc[-1] else "Bearish"
                    st.metric("Current Trend (4H)", trend)
                
        except Exception as e:
            st.error(f'Error occurred: {str(e)}')

if __name__ == "__main__":
    show_trend_oscillator()
