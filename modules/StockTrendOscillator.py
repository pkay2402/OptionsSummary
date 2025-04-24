import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import streamlit.components.v1 as components
from PIL import Image
import logging
logging.basicConfig(level=logging.INFO)

# Page configuration
#st.set_page_config(
    #page_title="Advanced Stock Trend Oscillator",
    #page_icon="📈",
    #layout="wide"
#)

# Custom CSS for better styling
st.markdown("""
<style>
    .stock-card {
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #f0f2f6;
        margin-bottom: 1rem;
        transition: transform 0.3s;
    }
    .stock-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .buy-signal {
        background-color: rgba(0, 204, 150, 0.1);
        border-left: 5px solid #00CC96;
    }
    .sell-signal {
        background-color: rgba(239, 83, 80, 0.1);
        border-left: 5px solid #EF5350;
    }
    .no-signal {
        background-color: rgba(176, 190, 197, 0.1);
        border-left: 5px solid #B0BEC5;
    }
    .card-header {
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #616161;
    }
    .metric-value {
        font-size: 1.1rem;
        font-weight: bold;
    }
    .tab-header {
        font-size: 1.8rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #616161;
        margin-bottom: 1.5rem;
    }
    .insights-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .insight-title {
        font-weight: bold;
        color: #1976D2;
    }
    .stButton>button {
        width: 100%;
    }
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Pre-selected stocks with sectors
STOCK_LIST = {
    'AAPL': {'name': 'Apple Inc.', 'sector': 'Technology'},
    'MSFT': {'name': 'Microsoft', 'sector': 'Technology'},
    'GOOGL': {'name': 'Alphabet', 'sector': 'Technology'},
    'AMZN': {'name': 'Amazon', 'sector': 'Consumer Cyclical'},
    'NVDA': {'name': 'NVIDIA', 'sector': 'Technology'},
    'META': {'name': 'Meta Platforms', 'sector': 'Technology'},
    'TSLA': {'name': 'Tesla', 'sector': 'Automotive'},
    'AMD': {'name': 'AMD', 'sector': 'Technology'},
    'NFLX': {'name': 'Netflix', 'sector': 'Communication Services'},
    'SPY': {'name': 'S&P 500 ETF', 'sector': 'ETF'},
    'QQQ': {'name': 'Nasdaq 100 ETF', 'sector': 'ETF'},
    'COIN': {'name': 'Coinbase', 'sector': 'Financial Services'},
    'TSM': {'name': 'Taiwan Semiconductor', 'sector': 'Technology'},
    'HOOD': {'name': 'Hood', 'sector': 'Financial Services'},
    'MSTR': {'name': 'Mastercard', 'sector': 'Financial Services'},
    'JPM': {'name': 'JP Morgan', 'sector': 'Financial Services'},
    'V': {'name': 'Visa', 'sector': 'Financial Services'},
    'XOM': {'name': 'Exxon Mobil', 'sector': 'Energy'},
    'UVXY': {'name': 'VIX', 'sector': 'Volatility'}
}

# Initialize session state variables
if 'last_signals' not in st.session_state:
    st.session_state.last_signals = {symbol: 'None' for symbol in STOCK_LIST.keys()}

if 'current_chart_symbol' not in st.session_state:
    st.session_state.current_chart_symbol = None

if 'stock_data' not in st.session_state:
    st.session_state.stock_data = []

if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = datetime.now() - timedelta(hours=1)

if 'view_mode' not in st.session_state:
    st.session_state.view_mode = 'grid'  # 'grid' or 'table'

if 'filter_signal' not in st.session_state:
    st.session_state.filter_signal = 'All'

if 'filter_sector' not in st.session_state:
    st.session_state.filter_sector = 'All'

def calculate_wilders_average(data, period):
    """Calculate Wilder's Moving Average."""
    result = data.copy()
    for i in range(period, len(data)):
        result[i] = ((period - 1) * result[i-1] + data[i]) / period
    return result

def calculate_rsi(data, period=14):
    """Calculate Relative Strength Index (RSI)."""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range (ATR)."""
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    return atr

def fetch_stock_data(symbol):
    """Fetch 60-minute data for a given stock."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=60)  # Fetch 60 days for 1H
    
    try:
        stock = yf.Ticker(symbol)
        df_60m = stock.history(start=start_date, end=end_date, interval="60m")
        
        # Ensure indices are DatetimeIndex and handle timezone
        if not isinstance(df_60m.index, pd.DatetimeIndex):
            df_60m.index = pd.DatetimeIndex(df_60m.index)

        if df_60m.index.tz is not None:
            df_60m.index = df_60m.index.tz_convert('America/New_York')
        else:
            df_60m.index = df_60m.index.tz_localize('UTC').tz_convert('America/New_York')

        # Filter for market hours (9:30 AM to 4:00 PM ET) and remove weekends
        df_60m = df_60m.between_time('09:30', '16:00')
        df_60m = df_60m[df_60m.index.dayofweek < 5]
        
        return df_60m
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        return pd.DataFrame()

def calculate_indicators(df):
    """Calculate Trend Oscillator and related indicators for 60-minute data."""
    if df.empty:
        return df, 0
        
    # Parameters
    L1 = 30
    L2 = 40
    trend_period = 20
    vol_period = 10
    vol_multiplier = 0.8

    # Calculate EMAs for price chart
    df['EMA21'] = df['Close'].ewm(span=21, adjust=False).mean()
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()

    # Calculate Trend Oscillator
    price_diff = df['Close'].diff()
    a1 = calculate_wilders_average(price_diff, L1)
    a2 = calculate_wilders_average(abs(price_diff), L1)
    df['TrendOscillator'] = 50 * ((a1 / a2).fillna(0) + 1)
    df['EMA'] = df['TrendOscillator'].ewm(span=L2, adjust=False).mean()

    # Calculate supporting indicators
    df['SMA_Trend'] = df['Close'].rolling(window=trend_period).mean()
    df['Vol_SMA'] = df['Volume'].rolling(window=vol_period).mean()
    df['RSI'] = calculate_rsi(df['Close'], period=14)
    df['ATR'] = calculate_atr(df['High'], df['Low'], df['Close'], period=14)

    # Trend and volume filters
    df['AboveTrend'] = df['Close'] > df['SMA_Trend']
    df['BelowTrend'] = df['Close'] < df['SMA_Trend']
    df['VolFilter'] = df['Volume'] > vol_multiplier * df['Vol_SMA']

    # Buy signal conditions
    buy_conditions = (
        (df['TrendOscillator'] > df['EMA']) &
        (df['TrendOscillator'].shift(1) <= df['EMA'].shift(1)) &
        df['AboveTrend'] &
        (df['Volume'] > 1.2 * df['Vol_SMA']) &
        (df['TrendOscillator'] < 45) &
        (df['RSI'] < 60) &
        (df['ATR'] < df['ATR'].rolling(window=20).mean() * 1.5)
    )

    # Sell signal conditions
    sell_conditions = (
        (df['TrendOscillator'] < df['EMA']) &
        (df['TrendOscillator'].shift(1) >= df['EMA'].shift(1)) &
        df['BelowTrend'] &
        df['VolFilter'] &
        (df['TrendOscillator'] > 48) &
        (df['RSI'] > 40)
    )

    # Assign signals
    df['BuySignal'] = np.where(buy_conditions, df['Close'], np.nan)
    df['SellSignal'] = np.where(sell_conditions, df['Close'], np.nan)
    
    # Calculate overbought/oversold zones
    df['Overbought'] = df['TrendOscillator'] > 65
    df['Oversold'] = df['TrendOscillator'] < 30
    
    # Calculate crossovers for signals
    df['CrossAbove'] = (df['TrendOscillator'] > df['EMA']) & (df['TrendOscillator'].shift(1) <= df['EMA'].shift(1))
    df['CrossBelow'] = (df['TrendOscillator'] < df['EMA']) & (df['TrendOscillator'].shift(1) >= df['EMA'].shift(1))

    return df, vol_multiplier

def create_chart(df, symbol, company_name):
    """Create an improved interactive chart for 60-minute data with price and Trend Oscillator."""
    if df.empty:
        return None
        
    # Create figure with two subplots: price and oscillator
    fig = make_subplots(
        rows=2, cols=1,
        vertical_spacing=0.05,
        row_heights=[0.6, 0.4],
        subplot_titles=[
            f"{symbol} - {company_name} (60-Minute Chart)",
            "Trend Oscillator (0-100)"
        ]
    )

    # Add candlestick chart with softer colors
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price',
            increasing_line_color='#00CC96',
            decreasing_line_color='#EF5350',
            increasing_fillcolor='#00CC96',
            decreasing_fillcolor='#EF5350',
            opacity=0.8
        ),
        row=1, col=1
    )

    # Add EMAs with distinct colors
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['EMA21'],
            name='EMA21',
            line=dict(color='#FF9800', width=1.5),  # Orange
            connectgaps=True
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['EMA50'],
            name='EMA50',
            line=dict(color='#AB47BC', width=1.5),  # Purple
            connectgaps=True
        ),
        row=1, col=1
    )

    # Add buy/sell signals on price chart
    buy_signals = df[df['BuySignal'].notna()]
    sell_signals = df[df['SellSignal'].notna()]
    
    # Buy signals
    fig.add_trace(
        go.Scatter(
            x=buy_signals.index,
            y=buy_signals['BuySignal'],
            mode='markers',
            marker=dict(
                symbol='triangle-up',
                size=12,
                color='#00CC96',
                line=dict(color='#008060', width=1)
            ),
            name='Buy Signal',
            hoverinfo='text',
            hovertext=[f"Buy Signal<br>Price: ${price:.2f}<br>Date: {idx.strftime('%Y-%m-%d %H:%M')}" 
                       for idx, price in zip(buy_signals.index, buy_signals['BuySignal'])]
        ),
        row=1, col=1
    )
    
    # Sell signals
    fig.add_trace(
        go.Scatter(
            x=sell_signals.index,
            y=sell_signals['SellSignal'],
            mode='markers',
            marker=dict(
                symbol='triangle-down',
                size=12,
                color='#EF5350',
                line=dict(color='#B71C1C', width=1)
            ),
            name='Sell Signal',
            hoverinfo='text',
            hovertext=[f"Sell Signal<br>Price: ${price:.2f}<br>Date: {idx.strftime('%Y-%m-%d %H:%M')}" 
                       for idx, price in zip(sell_signals.index, sell_signals['SellSignal'])]
        ),
        row=1, col=1
    )

    # Add Trend Oscillator and EMA with smoother lines
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['TrendOscillator'],
            name='Trend Oscillator',
            line=dict(color='#4CAF50', width=2),  # Green
            connectgaps=True
        ),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['EMA'],
            name='Signal Line',
            line=dict(color='#F44336', width=2),  # Red
            connectgaps=True
        ),
        row=2, col=1
    )
    
    # Color zones for oscillator
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=[65] * len(df),
            fill=None,
            mode='lines',
            line=dict(color='rgba(0,0,0,0)'),
            showlegend=False,
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=[100] * len(df),
            fill='tonexty',
            mode='lines',
            line=dict(color='rgba(0,0,0,0)'),
            fillcolor='rgba(244, 67, 54, 0.1)',  # Light red
            showlegend=False,
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=[30] * len(df),
            fill=None,
            mode='lines',
            line=dict(color='rgba(0,0,0,0)'),
            showlegend=False,
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=[0] * len(df),
            fill='tonexty',
            mode='lines',
            line=dict(color='rgba(0,0,0,0)'),
            fillcolor='rgba(76, 175, 80, 0.1)',  # Light green
            showlegend=False,
        ),
        row=2, col=1
    )

    # Add horizontal lines for oscillator with labels
    for level, label, color in [(30, 'Oversold', '#4CAF50'), (50, 'Neutral', '#B0BEC5'), (65, 'Overbought', '#F44336')]:
        fig.add_hline(
            y=level,
            line_dash="dash",
            line_color=color,
            opacity=0.5,
            row=2, col=1
        )
        fig.add_annotation(
            x=df.index[0],
            y=level,
            text=label,
            font=dict(size=10, color=color),
            showarrow=False,
            xshift=-50,
            row=2, col=1
        )

    # Update layout with a clean, modern theme
    fig.update_layout(
        yaxis_title='Price (USD)',
        yaxis2_title='Trend Oscillator',
        xaxis_rangeslider_visible=False,
        height=700,
        template='plotly_white',
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=60, r=20, t=80, b=20),
        xaxis=dict(
            rangebreaks=[
                dict(bounds=["sat", "mon"]),  # Hide weekends
                dict(bounds=[16, 9.5], pattern="hour")  # Hide non-trading hours
            ],
            showgrid=True,
            gridcolor='rgba(200, 200, 200, 0.3)',
            zeroline=False,
            tickformat='%b %d %H:%M',
            title='Date'
        ),
        xaxis2=dict(
            showgrid=True,
            gridcolor='rgba(200, 200, 200, 0.3)',
            zeroline=False,
            tickformat='%b %d %H:%M',
            title='Date'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(200, 200, 200, 0.3)',
            zeroline=False,
            tickformat='$,.2f'
        ),
        yaxis2=dict(
            showgrid=True,
            gridcolor='rgba(200, 200, 200, 0.3)',
            zeroline=False,
            range=[0, 100]
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        hovermode="x unified"
    )

    return fig

def analyze_stock(symbol):
    """Analyze a stock and return its current signal status and metrics."""
    try:
        # Fetch data
        df = fetch_stock_data(symbol)
        if df.empty:
            return None

        # Calculate indicators
        df, vol_multiplier = calculate_indicators(df)

        # Get latest data
        latest = df.iloc[-1]
        latest_price = latest['Close']
        trend_oscillator = latest['TrendOscillator']
        signal_line = latest['EMA']
        rsi = latest['RSI']
        atr = latest['ATR']
        
        # Get price change
        if len(df) > 1:
            price_change = (latest_price - df.iloc[-2]['Close']) / df.iloc[-2]['Close'] * 100
        else:
            price_change = 0
            
        # Get EMA21 and EMA50 status
        above_ema21 = latest_price > latest['EMA21']
        above_ema50 = latest_price > latest['EMA50']
        
        # Check for crossover patterns in the last 3 periods
        recent_df = df[-10:]
        latest_buy = recent_df['BuySignal'].notna().any()
        latest_sell = recent_df['SellSignal'].notna().any()
        
        # Update the last signal if a new one is generated
        if latest_buy:
            st.session_state.last_signals[symbol] = 'Buy'
        elif latest_sell:
            st.session_state.last_signals[symbol] = 'Sell'

        # Use the last signal if no new signal is generated
        signal = st.session_state.last_signals[symbol]
        
        # Calculate oscillator status
        if trend_oscillator > 65:
            osc_status = "Overbought"
        elif trend_oscillator < 30:
            osc_status = "Oversold"
        else:
            osc_status = "Neutral"
            
        # Calculate signal strength (0-100)
        if signal == 'Buy':
            # Higher score for: lower oscillator, higher volume, above trend, lower RSI
            signal_strength = min(100, max(0, 60 + 
                                         (30 - trend_oscillator) * 0.5 + 
                                         (above_ema21 * 15) + 
                                         (above_ema50 * 15) +
                                         (50 - rsi) * 0.2))
        elif signal == 'Sell':
            # Higher score for: higher oscillator, higher volume, below trend, higher RSI
            signal_strength = min(100, max(0, 60 + 
                                         (trend_oscillator - 50) * 0.5 + 
                                         (not above_ema21 * 15) + 
                                         (not above_ema50 * 15) +
                                         (rsi - 50) * 0.2))
        else:
            signal_strength = 0
            
        # Get volume comparison
        vol_ratio = latest['Volume'] / latest['Vol_SMA'] if latest['Vol_SMA'] > 0 else 1
        
        # Create insights
        insights = []
        
        if above_ema21 and above_ema50:
            insights.append("Price above both EMA21 and EMA50 - Bullish")
        elif not above_ema21 and not above_ema50:
            insights.append("Price below both EMA21 and EMA50 - Bearish")
        elif above_ema21 and not above_ema50:
            insights.append("Price above EMA21 but below EMA50 - Short-term bullish")
            
        if trend_oscillator > signal_line and trend_oscillator < 45:
            insights.append("Oscillator rising from low levels - Possible bullish setup")
        elif trend_oscillator < signal_line and trend_oscillator > 55:
            insights.append("Oscillator falling from high levels - Possible bearish setup")
            
        if vol_ratio > 1.5:
            insights.append(f"High volume activity (×{vol_ratio:.1f} normal) - Strong interest")
            
        if rsi < 30:
            insights.append(f"RSI low at {rsi:.1f} - Possible oversold condition")
        elif rsi > 70:
            insights.append(f"RSI high at {rsi:.1f} - Possible overbought condition")

        return {
            'Symbol': symbol,
            'Company Name': STOCK_LIST[symbol]['name'],
            'Sector': STOCK_LIST[symbol]['sector'],
            'Latest Signal': signal,
            'Signal Strength': signal_strength,
            'Current Price': latest_price,
            'Price Change': price_change,
            'Trend Oscillator': trend_oscillator,
            'Signal Line': signal_line,
            'Oscillator Status': osc_status,
            'RSI': rsi,
            'ATR': atr,
            'Volume Ratio': vol_ratio,
            'Above EMA21': above_ema21,
            'Above EMA50': above_ema50,
            'Insights': insights,
            'df': df
        }
    except Exception as e:
        st.warning(f"Error processing {symbol}: {str(e)}")
        return None

def display_metric(label, value, format_str="{:.2f}", prefix="", suffix="", condition=None, positive_is_good=True):
    """Display a metric with conditional formatting."""
    #logging.info(f"Displaying metric: {label}, Value: {value}, Type: {type(value)}")
    
    # Handle non-numeric values
    try:
        formatted_value = f"{prefix}{format_str.format(float(value))}{suffix}"
    except (ValueError, TypeError):
        formatted_value = f"{prefix}{value}{suffix}"  # Fallback to string representation
    
    # Determine color based on condition
    if condition is not None:
        if (condition and positive_is_good) or (not condition and not positive_is_good):
            color = "#00CC96"  # Green - good
        else:
            color = "#EF5350"  # Red - bad
    else:
        color = "#212121"  # Default text color
    
    # Render the metric
    st.markdown(f"""
    <div>
        <div class="metric-label">{label}</div>
        <div class="metric-value" style="color: {color}">{formatted_value}</div>
    </div>
    """, unsafe_allow_html=True)

def render_stock_card(stock_info):
    """Render a card for a stock with its key metrics."""
    signal = stock_info['Latest Signal']
    signal_class = "buy-signal" if signal == "Buy" else "sell-signal" if signal == "Sell" else "no-signal"
    
    st.markdown(f"""
    <div class="stock-card {signal_class}">
        <div class="card-header">{stock_info['Symbol']} - {stock_info['Company Name']}</div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        display_metric("Current Price", stock_info['Current Price'], prefix="$")
        display_metric("Change", stock_info['Price Change'], suffix="%", 
                      condition=stock_info['Price Change'] > 0)
                      
    with col2:
        display_metric("Signal", signal)
        if signal != "None":
            display_metric("Signal Strength", stock_info['Signal Strength'], suffix="%")
        
    with col3:
        display_metric("Trend Oscillator", stock_info['Trend Oscillator'], 
                      condition=stock_info['Trend Oscillator'] > stock_info['Signal Line'])
        display_metric("Signal Line", stock_info['Signal Line'])
        
    with col4:
        display_metric("RSI", stock_info['RSI'], 
                      condition=stock_info['RSI'] < 50, positive_is_good=False)
        display_metric("Volume", stock_info['Volume Ratio'], suffix="× avg", 
                      condition=stock_info['Volume Ratio'] > 1.2)
    
    if stock_info['Insights']:
        st.markdown('<div class="insights-box">', unsafe_allow_html=True)
        for insight in stock_info['Insights'][:2]:  # Show only top 2 insights
            st.markdown(f'<div class="insight-title">→ {insight}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Button to view detailed chart
    if st.button("View Chart", key=f"chart_{stock_info['Symbol']}"):
        st.session_state.current_chart_symbol = stock_info['Symbol']
    
    st.markdown("</div>", unsafe_allow_html=True)

def refresh_data():
    """Refresh all stock data."""
    st.session_state.stock_data = []
    st.session_state.last_refresh = datetime.now()
    with st.spinner('Analyzing stocks...'):
        for symbol in STOCK_LIST.keys():
            result = analyze_stock(symbol)
            if result:
                st.session_state.stock_data.append(result)

def main():
    """Main function to show the Trend Oscillator dashboard."""
    # Header with app info and last refresh
    col1, col2 = st.columns([2, 1])
    with col1:
        st.title("Advanced Stock Trend Oscillator Dashboard")
        st.markdown('<div class="subtitle">Monitor stocks using the Trend Oscillator strategy on 60-minute charts</div>', unsafe_allow_html=True)
    
    with col2:
        last_refresh_time = st.session_state.last_refresh.strftime("%H:%M:%S")
        st.markdown(f"<div style='text-align: right; padding-top: 30px;'>Last refresh: {last_refresh_time}</div>", unsafe_allow_html=True)
        if st.button("🔄 Refresh Data", key="refresh_all"):
            refresh_data()
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Dashboard", "Chart View", "Settings"])
    
    with tab1:
        # Dashboard tab
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            view_options = ["Grid View", "Table View"]
            selected_view = st.radio("Display Style:", view_options, horizontal=True)
            st.session_state.view_mode = "grid" if selected_view == "Grid View" else "table"
        
        with col2:
            signal_filter = st.selectbox(
                "Filter by Signal:",
                ["All", "Buy", "Sell", "None"],
                index=0
            )
            st.session_state.filter_signal = signal_filter
            
        with col3:
            sectors = ["All"] + sorted(list(set(stock['sector'] for stock in STOCK_LIST.values())))
            sector_filter = st.selectbox(
                "Filter by Sector:",
                sectors,
                index=0
            )
            st.session_state.filter_sector = sector_filter
        
        # If no data loaded yet or refresh requested, analyze all stocks
        # If no data loaded yet or refresh requested, analyze all stocks
        if not st.session_state.stock_data or (datetime.now() - st.session_state.last_refresh).seconds > 3600:
            refresh_data()
            
        # Filter stocks based on selections
        filtered_stocks = st.session_state.stock_data.copy()
        
        if st.session_state.filter_signal != "All":
            filtered_stocks = [stock for stock in filtered_stocks if stock['Latest Signal'] == st.session_state.filter_signal]
            
        if st.session_state.filter_sector != "All":
            filtered_stocks = [stock for stock in filtered_stocks if stock['Sector'] == st.session_state.filter_sector]
            
        # Sort stocks: Buy signals first, then Sell signals, then None
        # Within each group, sort by signal strength
        filtered_stocks.sort(key=lambda x: (
            0 if x['Latest Signal'] == 'Buy' else 1 if x['Latest Signal'] == 'Sell' else 2,
            -x['Signal Strength'] if 'Signal Strength' in x else 0
        ))
        
        # Display stocks based on view mode
        if st.session_state.view_mode == "grid":
            # Grid view: display cards in a 3-column layout
            if not filtered_stocks:
                st.info("No stocks match your filter criteria.")
            else:
                cols = st.columns(3)
                for i, stock in enumerate(filtered_stocks):
                    with cols[i % 3]:
                        render_stock_card(stock)
        else:
            # Table view: display a sortable table
            if not filtered_stocks:
                st.info("No stocks match your filter criteria.")
            else:
                # Create DataFrame for the table
                table_data = [{
                    'Symbol': stock['Symbol'],
                    'Company': stock['Company Name'],
                    'Price': f"${stock['Current Price']:.2f}",
                    'Change': f"{stock['Price Change']:.2f}%",
                    'Signal': stock['Latest Signal'],
                    'Trend Osc.': f"{stock['Trend Oscillator']:.1f}",
                    'Signal Line': f"{stock['Signal Line']:.1f}",
                    'RSI': f"{stock['RSI']:.1f}",
                    'Volume': f"{stock['Volume Ratio']:.1f}x"
                } for stock in filtered_stocks]
                
                df_table = pd.DataFrame(table_data)
                
                # Display with custom formatting
                st.dataframe(
                    df_table, 
                    column_config={
                        "Symbol": st.column_config.TextColumn("Symbol", width="small"),
                        "Company": st.column_config.TextColumn("Company", width="medium"),
                        "Price": st.column_config.TextColumn("Price", width="small"),
                        "Change": st.column_config.TextColumn("Change", width="small"),
                        "Signal": st.column_config.TextColumn("Signal", width="small"),
                        "Trend Osc.": st.column_config.TextColumn("Trend Osc.", width="small"),
                        "Signal Line": st.column_config.TextColumn("Signal Line", width="small"),
                        "RSI": st.column_config.TextColumn("RSI", width="small"),
                        "Volume": st.column_config.TextColumn("Volume", width="small")
                    },
                    use_container_width=True,
                    hide_index=True
                )
                
                # Add buttons for each stock
                selected_symbol = st.selectbox(
                    "Select a stock to view its chart:",
                    [f"{stock['Symbol']} - {stock['Company Name']}" for stock in filtered_stocks]
                )
                
                if st.button("View Selected Chart"):
                    selected_symbol = selected_symbol.split(" - ")[0]
                    st.session_state.current_chart_symbol = selected_symbol
    
    with tab2:
        # Chart view tab
        st.markdown('<div class="tab-header">Detailed Stock Analysis</div>', unsafe_allow_html=True)
        
        if st.session_state.current_chart_symbol:
            symbol = st.session_state.current_chart_symbol
            stock_info = next((item for item in st.session_state.stock_data if item['Symbol'] == symbol), None)
            
            if stock_info:
                # Display stock info header
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
                    <div>
                        <h2>{stock_info['Symbol']} - {stock_info['Company Name']}</h2>
                        <p>Sector: {stock_info['Sector']} | Latest Signal: <span style="font-weight: bold; color: {'#00CC96' if stock_info['Latest Signal'] == 'Buy' else '#EF5350' if stock_info['Latest Signal'] == 'Sell' else '#B0BEC5'}">{stock_info['Latest Signal']}</span></p>
                    </div>
                    <div style="text-align: right;">
                        <h3>${stock_info['Current Price']:.2f} <span style="color: {'#00CC96' if stock_info['Price Change'] > 0 else '#EF5350'}">({stock_info['Price Change']:.2f}%)</span></h3>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Display chart
                with st.spinner(f"Generating chart for {symbol}..."):
                    df_stock = stock_info['df']
                    fig = create_chart(df_stock, symbol, stock_info['Company Name'])
                    st.plotly_chart(fig, use_container_width=True)
                
                # Stock metrics in columns
                st.markdown("### Key Metrics")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Trend Oscillator", f"{stock_info['Trend Oscillator']:.2f}", 
                             f"{stock_info['Trend Oscillator'] - stock_info['Signal Line']:.2f} vs Signal")
                    st.metric("Volume", f"{stock_info['Volume Ratio']:.2f}x average")
                
                with col2:
                    st.metric("Signal Line", f"{stock_info['Signal Line']:.2f}")
                    st.metric("RSI", f"{stock_info['RSI']:.2f}")
                
                with col3:
                    ema_status = "Above" if stock_info['Above EMA21'] else "Below"
                    st.metric("EMA21", f"{ema_status}")
                    st.metric("ATR", f"{stock_info['ATR']:.4f}")
                
                with col4:
                    ema50_status = "Above" if stock_info['Above EMA50'] else "Below"
                    st.metric("EMA50", f"{ema50_status}")
                    st.metric("Oscillator Status", f"{stock_info['Oscillator Status']}")
                
                # Analysis and insights
                st.markdown("### Analysis and Insights")
                st.markdown('<div class="insights-box" style="padding: 20px;">', unsafe_allow_html=True)
                
                # Display all insights
                for insight in stock_info['Insights']:
                    st.markdown(f'<div class="insight-title" style="margin-bottom: 10px;">→ {insight}</div>', unsafe_allow_html=True)
                
                # Generate additional analysis based on current conditions
                if stock_info['Latest Signal'] == 'Buy':
                    st.markdown("""
                    <h4 style="margin-top: 20px;">Buy Signal Analysis:</h4>
                    <p>The stock is showing a buy signal based on the Trend Oscillator strategy. 
                    This indicates potential upward momentum in the price action, especially when confirmed by the price being above key moving averages.</p>
                    <p>Key factors to watch:</p>
                    <ul>
                        <li>Continued price action above EMA21 and EMA50</li>
                        <li>Sustained volume above average</li>
                        <li>Trend Oscillator remaining above its signal line</li>
                    </ul>
                    """, unsafe_allow_html=True)
                elif stock_info['Latest Signal'] == 'Sell':
                    st.markdown("""
                    <h4 style="margin-top: 20px;">Sell Signal Analysis:</h4>
                    <p>The stock is displaying a sell signal according to the Trend Oscillator strategy.
                    This suggests potential downward momentum, particularly when the price is below key moving averages.</p>
                    <p>Key factors to watch:</p>
                    <ul>
                        <li>Continued price action below EMA21 and EMA50</li>
                        <li>Volume confirming the downward movement</li>
                        <li>Trend Oscillator remaining below its signal line</li>
                    </ul>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <h4 style="margin-top: 20px;">No Active Signal:</h4>
                    <p>The stock currently shows no active buy or sell signal based on the Trend Oscillator strategy.
                    Consider monitoring for potential crossovers of the Trend Oscillator and its signal line
                    along with price action relative to key moving averages.</p>
                    """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Add button to close chart
                if st.button("← Back to Dashboard"):
                    st.session_state.current_chart_symbol = None
                    st.experimental_rerun()
            else:
                st.error(f"No data available for {symbol}.")
        else:
            st.info("Select a stock from the dashboard to view its detailed chart and analysis.")

    with tab3:
        # Settings tab
        st.markdown('<div class="tab-header">Dashboard Settings</div>', unsafe_allow_html=True)
        
        st.subheader("Stock Watchlist")
        st.write("Manage the list of stocks you want to track in the dashboard.")
        
        # Display current stocks with option to remove
        st.markdown("### Current Watchlist")
        
        # Convert to DataFrame for better display
        stocks_df = pd.DataFrame([
            {
                'Symbol': symbol,
                'Company': info['name'],
                'Sector': info['sector']
            } for symbol, info in STOCK_LIST.items()
        ])
        
        st.dataframe(stocks_df, use_container_width=True, hide_index=True)
        
        # Add new stock
        st.markdown("### Add New Stock")
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            new_symbol = st.text_input("Symbol:", "").upper()
        
        with col2:
            new_name = st.text_input("Company Name:", "")
        
        with col3:
            sectors = sorted(list(set(stock['sector'] for stock in STOCK_LIST.values())))
            new_sector = st.selectbox("Sector:", sectors + ["Other"])
        
        if st.button("Add Stock") and new_symbol and new_name:
            if new_symbol in STOCK_LIST:
                st.warning(f"{new_symbol} is already in your watchlist!")
            else:
                STOCK_LIST[new_symbol] = {'name': new_name, 'sector': new_sector}
                st.session_state.last_signals[new_symbol] = 'None'
                st.success(f"{new_symbol} added to your watchlist!")
                st.experimental_rerun()
        
        # App settings
        st.markdown("### Application Settings")
        
        # Theme settings
        st.radio("Theme:", ["Light", "Dark"], horizontal=True, 
                 help="Choose the appearance theme for the dashboard")
        
        # Data refresh rate
        st.slider("Auto-refresh interval (minutes):", 5, 60, 15, 
                  help="How often the dashboard should automatically refresh data")
        
        # Disclaimer
        st.markdown("""
        <div style="margin-top: 50px; padding: 15px; background-color: #f8f9fa; border-radius: 5px; font-size: 0.8rem;">
        <strong>Disclaimer:</strong> This tool is for informational purposes only. It is not financial advice, and should not be used as the basis for any investment decisions.
        Stock market investing involves risk, and past performance is not indicative of future results.
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
