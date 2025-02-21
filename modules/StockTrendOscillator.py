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

def calculate_trend_oscillator(df_1h, l1=20, l2=50):
    """Calculate Trend Oscillator using 4-hour timeframe data"""
    # Resample 1-hour data to 4-hour data
    df_4h = get_higher_timeframe_data(df_1h, interval='4H')
    
    # Get price changes on 4-hour timeframe
    price_change = df_4h['Close'] - df_4h['Close'].shift(1)
    abs_price_change = abs(price_change)
    
    # Calculate Wilder's Moving Averages on 4-hour data
    a1 = calculate_wilders_ma(price_change, l1)
    a2 = calculate_wilders_ma(abs_price_change, l1)
    
    # Calculate trend oscillator
    a3 = np.where(a2 != 0, a1 / a2, 0)
    trend_osc_4h = 50 * (a3 + 1)
    
    # Calculate EMA of trend oscillator on 4-hour data
    ema_4h = pd.Series(trend_osc_4h).ewm(span=l2, adjust=False).mean()
    
    # Reindex to match 1-hour timeframe (forward-fill to align)
    trend_osc = pd.Series(trend_osc_4h, index=df_4h.index).reindex(df_1h.index, method='ffill')
    ema = pd.Series(ema_4h, index=df_4h.index).reindex(df_1h.index, method='ffill')
    
    return trend_osc, ema

def create_chart(df, symbol):
    """Create interactive chart with trend oscillator"""
    # Calculate indicators (pass the 1-hour DataFrame)
    trend_osc, ema = calculate_trend_oscillator(df)
    
    # Calculate 21 and 50 EMAs for price (still on 1-hour timeframe)
    df['EMA21'] = df['Close'].ewm(span=21, adjust=False).mean()
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
    
    # Create figure with secondary y-axis
    fig = make_subplots(rows=2, cols=1, 
                       shared_xaxes=True,
                       vertical_spacing=0.05,
                       row_heights=[0.7, 0.3])

    # Add candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price'
        ),
        row=1, col=1
    )
    
    # Add EMAs to price chart
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['EMA21'],
            name='EMA21',
            line=dict(color='red', width=1),
            connectgaps=True
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['EMA50'],
            name='EMA50',
            line=dict(color='purple', width=1),
            connectgaps=True
        ),
        row=1, col=1
    )

    # Add Trend Oscillator and EMA to subplot
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=trend_osc,
            name='Trend Oscillator',
            line=dict(color='green', width=2),
            connectgaps=True
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=ema,
            name='Signal Line',
            line=dict(color='red', width=2),
            connectgaps=True
        ),
        row=2, col=1
    )

    # Add horizontal lines for overbought/oversold levels
    for level in [30, 50, 65]:
        fig.add_hline(
            y=level,
            line_dash="dash",
            line_color="white",
            opacity=0.5,
            row=2, col=1
        )

    # Add buy/sell signals
    buy_signals = ((trend_osc > ema) & (trend_osc.shift(1) <= ema.shift(1)))
    sell_signals = ((trend_osc < ema) & (trend_osc.shift(1) >= ema.shift(1)))
    
    # Plot buy signals
    fig.add_trace(
        go.Scatter(
            x=df.index[buy_signals],
            y=trend_osc[buy_signals],
            mode='markers',
            marker=dict(
                symbol='triangle-up',
                size=12,
                color='green',
                line=dict(width=2)
            ),
            name='Buy Signal'
        ),
        row=2, col=1
    )
    
    # Plot sell signals
    fig.add_trace(
        go.Scatter(
            x=df.index[sell_signals],
            y=trend_osc[sell_signals],
            mode='markers',
            marker=dict(
                symbol='triangle-down',
                size=12,
                color='red',
                line=dict(width=2)
            ),
            name='Sell Signal'
        ),
        row=2, col=1
    )

    # Update layout
    fig.update_layout(
        title=f'{symbol} - 1H Chart with 4H Trend Oscillator',
        yaxis_title='Price',
        yaxis2_title='Trend Oscillator',
        xaxis_rangeslider_visible=False,
        height=800,
        template='plotly_dark',
        xaxis={
            'rangebreaks': [
                dict(bounds=["sat", "mon"]),  # hide weekends
                dict(bounds=[16, 9.5], pattern="hour"),  # hide non-trading hours
            ]
        }
    )
    
    # Update y-axes
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Trend Oscillator", row=2, col=1)

    return fig

def show_trend_oscillator():
    """Main function to show the trend oscillator in the Streamlit app"""
    st.header('Multi-Stock Trend Oscillator Dashboard')

    st.write("""
    A Streamlit-based interactive stock analysis dashboard that combines price action with a custom trend oscillator.
    
    Key features:
    - Pre-selected major stocks (AAPL, MSFT, GOOGL, etc.) and custom symbol input
    - Interactive candlestick charts with 21 and 50 EMAs
    - Custom trend oscillator with signal line and buy/sell indicators
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
                
                # Fetch 1-hour data
                stock = yf.Ticker(selected_stock)
                df = stock.history(start=start_date, end=end_date, interval='1h')
                
                # Filter for market hours (9:30 AM to 4:00 PM ET)
                df.index = df.index.tz_localize(None)
                df = df.between_time('09:30', '16:00')
                
                # Remove weekends
                df = df[df.index.dayofweek < 5]
                
                # Keep only last 30 days of trading data
                df = df.last('30D')
                
                if df.empty:
                    st.error('No data found for the specified stock.')
                    return
                
                # Create and display chart
                fig = create_chart(df, selected_stock)
                st.plotly_chart(fig, use_container_width=True)
                
                # Display current indicator values
                trend_osc, ema = calculate_trend_oscillator(df)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Current Trend Oscillator", f"{trend_osc.iloc[-1]:.2f}")
                with col2:
                    st.metric("Current Signal Line", f"{ema.iloc[-1]:.2f}")
                with col3:
                    trend = "Bullish" if trend_osc.iloc[-1] > ema.iloc[-1] else "Bearish"
                    st.metric("Current Trend", trend)
                
        except Exception as e:
            st.error(f'Error occurred: {str(e)}')
