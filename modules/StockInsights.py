import streamlit as st
import pandas as pd
import plotly.express as px
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from math import ceil
from datetime import datetime, timedelta, date
from scipy import stats
import requests
import io
from io import StringIO
import pytz
import logging
from typing import List, Optional
import holidays

last_fetch_time = datetime.now()

if st.sidebar.button("Refresh Data", help="Clear cache and reload all market data", use_container_width=True):
    st.cache_data.clear()
    last_fetch_time = datetime.now()
    st.rerun()

st.sidebar.markdown(f"Last data fetch: {last_fetch_time.strftime('%H:%M:%S')}")

FLOW_URLS = [
    "https://www.cboe.com/us/options/market_statistics/symbol_data/csv/?mkt=cone",
    "https://www.cboe.com/us/options/market_statistics/symbol_data/csv/?mkt=opt",
    "https://www.cboe.com/us/options/market_statistics/symbol_data/csv/?mkt=ctwo",
    "https://www.cboe.com/us/options/market_statistics/symbol_data/csv/?mkt=exo"
]

STOCK_LISTS = {
    "Index": ["SPY", "QQQ", "IWM", "DIA", "SMH", 'IBIT',"UVXY"],
    "Sector ETF": ["XLF", "XLE", "XLV", "XLY", "XLC", "XLI", "XLB", "XLRE", "XLU", "XLP", "XBI", "XOP", "XME", "XRT", "XHB"],
    "Tech Giants": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSM", "AVGO", "ADBE", 
               "CRM", "ORCL", "CSCO", "AMD", "INTC", "IBM", "TXN", "QCOM", "AMAT", "MU", "NOW"],
    "Dow Components": ["AAPL", "AMGN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS", "DOW", "GS", "HD", "HON", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "MMM", "MRK", "NKE", "PG", "TRV", "UNH", "V", "VZ", "WBA", "WMT"],
    "Financial Sector": ["JPM", "BAC", "WFC", "C", "GS", "MS", "BLK", "AXP", "USB", "PNC"],
    "Healthcare Leaders": ["JNJ", "PFE", "MRK", "ABBV", "BMY", "LLY", "AMGN", "GILD", "REGN", "BIIB"],
    "Energy Stocks": ["XOM", "CVX", "COP", "EOG", "SLB", "PXD", "OXY", "DVN", "MPC", "PSX"],
    "Retail Giants": ["WMT", "TGT", "COST", "HD", "LOW", "AMZN", "EBAY", "ETSY", "BBY", "DG"]
}

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_csv_content_type(response: requests.Response) -> bool:
    """Validate if the response content type is CSV."""
    return 'text/csv' in response.headers.get('Content-Type', '')

def fetch_data_from_url(url: str) -> Optional[pd.DataFrame]:
    """Fetch and process data from a single URL."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        if validate_csv_content_type(response):
            csv_data = StringIO(response.text)
            df = pd.read_csv(csv_data)
            df['Expiration'] = pd.to_datetime(df['Expiration'])
            df = df[df['Expiration'].dt.date >= datetime.now().date()]
            return df
        else:
            logger.warning(f"Data from {url} is not in CSV format. Skipping...")
    except Exception as e:
        logger.error(f"Error fetching data from {url}: {e}")
    return None

def fetch_options_flow_data(urls: List[str], ticker: str) -> pd.DataFrame:
    """Fetch and combine options flow data for a specific ticker."""
    data_frames = []
    for url in urls:
        df = fetch_data_from_url(url)
        if df is not None and not df.empty:
            df = df[df['Symbol'] == ticker]
            if not df.empty:
                data_frames.append(df)
    return pd.concat(data_frames, ignore_index=True) if data_frames else pd.DataFrame()

def get_current_price(ticker: str) -> float:
    """Fetch the current price of the ticker using yfinance."""
    try:
        stock = yf.Ticker(ticker)
        return stock.history(period="1d")['Close'].iloc[-1]
    except Exception as e:
        logger.error(f"Error fetching current price for {ticker}: {e}")
        return None

def get_top_otm_flows(ticker: str, urls: List[str], top_n: int = 10) -> pd.DataFrame:
    """Get the top N OTM options flows ordered by transaction value."""
    try:
        # Fetch options flow data
        df = fetch_options_flow_data(urls, ticker)
        if df.empty:
            return pd.DataFrame()

        # Fetch current price
        current_price = get_current_price(ticker)
        if current_price is None:
            logger.error(f"Could not determine current price for {ticker}")
            return pd.DataFrame()

        # Filter for OTM options
        df['Transaction Value'] = df['Volume'] * df['Last Price'] * 100
        otm_df = df[
            ((df['Call/Put'] == 'C') & (df['Strike Price'] > current_price)) |  # OTM Calls
            ((df['Call/Put'] == 'P') & (df['Strike Price'] < current_price))    # OTM Puts
        ]

        # Summarize and sort by transaction value
        summary = (
            otm_df.groupby(['Symbol', 'Expiration', 'Strike Price', 'Call/Put', 'Last Price'])
            .agg({'Volume': 'sum', 'Transaction Value': 'sum'})
            .reset_index()
        )
        summary = summary.sort_values(by='Transaction Value', ascending=False)

        # Return top N
        return summary.head(top_n)

    except Exception as e:
        logger.error(f"Error in get_top_otm_flows for {ticker}: {e}")
        return pd.DataFrame()

# StockAnalysis Functions
def calculate_rsi(data, periods=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_wilders_ma(data, periods):
    return data.ewm(alpha=1/periods, adjust=False).mean()

def calculate_relative_strength(symbol_hist, spy_hist, lookback=20):
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
    if rsi > 70:
        return "Overbought"
    elif rsi > 50:
        return "Strong"
    elif rsi > 30:
        return "Weak"
    else:
        return "Oversold"

def fetch_stock_data(symbol, period="1y", interval="1d", spy_hist=None):
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period, interval=interval)
        if hist.empty:
            raise ValueError(f"No data available for {symbol}")
        
        hist['Cumulative_Volume'] = hist['Volume'].cumsum()
        hist['Cumulative_PV'] = (hist['Close'] * hist['Volume']).cumsum()
        hist['VWAP'] = hist['Cumulative_PV'] / hist['Cumulative_Volume']
        hist['RSI'] = calculate_rsi(hist)
        hist['EMA21'] = hist['Close'].ewm(span=21, adjust=False).mean()
        hist['EMA9'] = hist['Close'].ewm(span=9, adjust=False).mean()
        hist['EMA50'] = hist['Close'].ewm(span=50, adjust=False).mean()
        
        # For longer timeframes, add more relevant MAs
        if period in ["1y", "6mo"]:
            hist['SMA200'] = hist['Close'].rolling(window=200).mean()
            
        rs_value, rs_status = calculate_relative_strength(hist, spy_hist) if spy_hist is not None else (0, "N/A")
        
        today_data = hist.iloc[-1]
        current_price = round(today_data["Close"], 2)
        vwap = round(hist['VWAP'].iloc[-1], 2)
        ema_21 = round(hist['EMA21'].iloc[-1], 2)
        daily_pivot = round((hist["High"].iloc[-1] + hist["Low"].iloc[-1] + current_price) / 3, 2)
        
        # Adjust MA logic based on timeframe
        if period in ["1y", "6mo"]:
            sma_200 = round(hist['SMA200'].iloc[-1], 2) if not pd.isna(hist['SMA200'].iloc[-1]) else None
            if current_price > hist['EMA9'].iloc[-1] and current_price > ema_21 and current_price > hist['EMA50'].iloc[-1] and (sma_200 is None or current_price > sma_200):
                key_mas = "Bullish"
            elif current_price < hist['EMA9'].iloc[-1] and current_price < ema_21 and current_price < hist['EMA50'].iloc[-1] and (sma_200 is None or current_price < sma_200):
                key_mas = "Bearish"
            else:
                key_mas = "Mixed"
        else:
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
            "RSI_Status": [get_rsi_status(hist['RSI'].iloc[-1])],
            "Timeframe": [f"{period} {interval}"]  # Add timeframe info for reference
        }), hist.round(2)
    except Exception as e:
        st.error(f"Error in fetch_stock_data for {symbol}: {e}")
        return pd.DataFrame(), pd.DataFrame()

def get_us_market_holidays(start_year, end_year):
    us_holidays = holidays.US(years=range(start_year, end_year + 1))
    return [date for date in us_holidays.keys()]

def calculate_swing_levels(data, look_forward=10, look_backward=10):
    data['fLow'] = data['Low'].rolling(window=look_forward).min().shift(-look_forward)
    data['bLow'] = data['Low'].rolling(window=look_backward).min().shift(1)
    data['sLow'] = (data['Low'] < data['fLow']) & (data['Low'] <= data['bLow'])

    data['fHigh'] = data['High'].rolling(window=look_forward).max().shift(-look_forward)
    data['bHigh'] = data['High'].rolling(window=look_backward).max().shift(1)
    data['sHigh'] = (data['High'] > data['fHigh']) & (data['High'] >= data['bHigh'])

    data['priorSL'] = data['Low'].where(data['sLow']).ffill()
    data['priorSH'] = data['High'].where(data['sHigh']).ffill()

    return data

def plot_candlestick(data, symbol, price_chart_timeframe):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name=symbol))
    fig.add_trace(go.Scatter(x=data.index, y=data['VWAP'], name='VWAP', line=dict(color='purple', width=2)))
    fig.add_trace(go.Scatter(x=data.index, y=data['EMA21'], name='21 EMA', line=dict(color='orange', width=2)))
    fig.add_trace(go.Scatter(x=data.index, y=data['EMA9'], name='9 EMA', line=dict(color='blue', width=1.5)))
    
    # Add more indicators for daily timeframes
    if 'SMA200' in data.columns:
        fig.add_trace(go.Scatter(x=data.index, y=data['SMA200'], name='200 SMA', line=dict(color='red', width=1.5)))
    
    # Calculate swing levels
    data = calculate_swing_levels(data)
    
    # Plot swing highs and lows as horizontal lines
    fig.add_trace(go.Scatter(x=data.index, y=data['priorSH'], mode='lines', name='Swing High', line=dict(color='green', width=2, dash='dash')))
    fig.add_trace(go.Scatter(x=data.index, y=data['priorSL'], mode='lines', name='Swing Low', line=dict(color='red', width=2, dash='dash')))
    
    # Apply different xaxis configurations based on the time interval
    if len(data) > 0:
        # Check if we're dealing with daily data by examining the time difference
        time_diff = None
        if len(data) > 1:
            time_diff = data.index[1] - data.index[0]
        
        # Configure axis for different timeframes
        if time_diff is not None and time_diff.days >= 1:
            # For daily or longer timeframes, remove weekend gaps and holidays
            start_year = data.index.min().year
            end_year = data.index.max().year
            us_market_holidays = get_us_market_holidays(start_year, end_year)
            fig.update_xaxes(
                rangebreaks=[
                    dict(bounds=["sat", "mon"]),  # Hide weekends
                    dict(values=us_market_holidays)  # Hide US market holidays
                ]
            )
        elif time_diff is not None and time_diff.seconds >= 3600:  # For hourly data
            # Filter out None values from rangebreaks
            rangebreaks = [
                dict(bounds=["sat", "mon"]) if price_chart_timeframe == '1D 5min' else None,  # Hide weekends for 1H
                dict(bounds=[16, 9.5], pattern="hour") if price_chart_timeframe == '1D 5min' else None  # Hide non-trading hours for 1H
            ]
            rangebreaks = [rb for rb in rangebreaks if rb is not None]
            
            fig.update_xaxes(
                rangebreaks=rangebreaks
            )
    
    fig.update_layout(
        title=f'{symbol} Chart with Indicators',
        yaxis_title='Price',
        template='plotly_white',
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=600,
        xaxis_rangeslider_visible=False,
        xaxis=dict(  # Lower timeframe x-axis (1H or 1D)
            showgrid=False,  # Remove gridlines to match ThinkScript
            zeroline=False
        )
    )
    
    return fig

# StockTrendOscillator Functions
def calculate_trend_oscillator(df_higher, l1, l2):
    try:
        if df_higher.empty or 'Close' not in df_higher:
            return pd.Series(), pd.Series()
        df_higher = df_higher.copy()
        if df_higher['Close'].isna().any():
            df_higher['Close'] = df_higher['Close'].ffill().bfill()
        price_change = df_higher['Close'] - df_higher['Close'].shift(1)
        abs_price_change = abs(price_change)
        a1 = calculate_wilders_ma(price_change, l1).reindex(df_higher.index, method='ffill').fillna(0)
        a2 = calculate_wilders_ma(abs_price_change, l1).reindex(df_higher.index, method='ffill').fillna(0)
        a3 = np.where(a2 != 0, a1 / a2, 0)
        trend_oscillator = pd.Series(50 * (a3 + 1), index=df_higher.index).fillna(50)
        ema = trend_oscillator.ewm(span=l2, adjust=False).mean().reindex(df_higher.index, method='ffill').fillna(50)
        return trend_oscillator, ema
    except Exception as e:
        st.error(f"Error in calculate_trend_oscillator: {e}")
        return pd.Series(), pd.Series()

def create_chart(df_lower, df_higher, symbol, timeframe_lower, timeframe_higher):
    try:
        if df_lower.empty or df_higher.empty or 'Close' not in df_lower or 'Close' not in df_higher:
            return None
        if timeframe_lower == '1H':
            l1, l2 = 20, 30
        else:
            l1, l2 = 20, 35
        df_lower['EMA21'] = df_lower['Close'].ewm(span=21, adjust=False).mean()
        df_lower['EMA50'] = df_lower['Close'].ewm(span=50, adjust=False).mean()
        trend_oscillator, ema = calculate_trend_oscillator(df_higher, l1, l2)
        if trend_oscillator.empty or ema.empty or len(trend_oscillator) == 0 or len(ema) == 0:
            return None
        fig = make_subplots(rows=2, cols=1, vertical_spacing=0.05, row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(x=df_lower.index, open=df_lower['Open'], high=df_lower['High'], low=df_lower['Low'], close=df_lower['Close'], name=f'Price ({timeframe_lower})'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_lower.index, y=df_lower['EMA21'], name=f'EMA21 ({timeframe_lower})', line=dict(color='red', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_lower.index, y=df_lower['EMA50'], name=f'EMA50 ({timeframe_lower})', line=dict(color='purple', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_higher.index, y=trend_oscillator, name=f'Trend Oscillator ({timeframe_higher})', line=dict(color='green', width=2)), row=2, col=1)
        fig.add_trace(go.Scatter(x=df_higher.index, y=ema, name=f'Signal Line ({timeframe_higher})', line=dict(color='red', width=2)), row=2, col=1)
        for level in [30, 50, 65]:
            fig.add_hline(y=level, line_dash="dash" if level in [30, 65] else "solid", line_color="gray", opacity=0.7, row=2, col=1)
        
        fig.update_layout(
            title=f'{symbol} - {timeframe_lower} Price Chart and {timeframe_higher} Trend Oscillator',
            yaxis_title=f'Price ({timeframe_higher})',
            yaxis2_title=f'Trend Oscillator ({timeframe_higher})',
            template='plotly_white',
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis_rangeslider_visible=False,
            height=800,
            xaxis=dict(showgrid=False, zeroline=False),
            xaxis2=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
            yaxis2=dict(showgrid=False, zeroline=False, range=[0, 100])
        )
        return fig
    except Exception as e:
        st.error(f"Error in create_chart: {e}")
        return None

@st.cache_data(ttl=300)
def get_oscillator(ticker, timeframe_setup='1H/1D'):
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)
        stock = yf.Ticker(ticker)
        
        if timeframe_setup == '1H/1D':
            df_lower = stock.history(start=start_date, end=end_date, interval="60m").between_time('09:30', '16:00').last('60D')
            df_higher = stock.history(start=start_date, end=end_date, interval="1d").last('60D')
            timeframe_lower, timeframe_higher = '1H', '1D'
        else:
            df_lower = stock.history(start=start_date, end=end_date, interval="1d").last('180D')
            df_higher = stock.history(start=start_date, end=end_date, interval="5d").last('180D')
            timeframe_lower, timeframe_higher = '1D', '5D'
        
        if df_lower.empty or df_higher.empty or 'Close' not in df_lower or 'Close' not in df_higher:
            return {"error": "No data available", "chart": None, "trend_oscillator": None, "signal_line": None, "trend": "N/A"}
        
        for df in [df_lower, df_higher]:
            if df.index.tz is not None:
                df.index = df.index.tz_convert('America/New_York')
            else:
                df.index = df.index.tz_localize('UTC').tz_convert('America/New_York')
            df = df[df.index.dayofweek < 5]
        
        chart = create_chart(df_lower, df_higher, ticker, timeframe_lower, timeframe_higher)
        if chart is None:
            return {"error": "Chart creation failed", "chart": None, "trend_oscillator": None, "signal_line": None, "trend": "N/A"}
        
        trend_osc, ema = calculate_trend_oscillator(df_higher, 20, 30 if timeframe_lower == '1H' else 35)
        if trend_osc.empty or ema.empty or len(trend_osc) == 0 or len(ema) == 0:
            return {"error": "No oscillator data available", "chart": chart, "trend_oscillator": None, "signal_line": None, "trend": "N/A"}
        
        trend_osc_last = trend_osc.iloc[-1] if pd.notna(trend_osc.iloc[-1]) and not pd.isna(trend_osc.iloc[-1]) else None
        ema_last = ema.iloc[-1] if pd.notna(ema.iloc[-1]) and not pd.isna(ema.iloc[-1]) else None
        
        trend = "N/A"
        if trend_osc_last is not None and ema_last is not None:
            if isinstance(trend_osc_last, (pd.Series, pd.DataFrame)):
                trend_osc_last = trend_osc_last.iloc[0] if not trend_osc_last.empty else None
            if isinstance(ema_last, (pd.Series, pd.DataFrame)):
                ema_last = ema_last.iloc[0] if not ema_last.empty else None
                
            if trend_osc_last is not None and ema_last is not None:
                trend = "Bullish" if trend_osc_last > ema_last else "Bearish"
        
        return {
            "trend_oscillator": trend_osc_last,
            "signal_line": ema_last,
            "trend": trend,
            "chart": chart
        }
    except Exception as e:
        st.error(f"Error in get_oscillator for {ticker}: {e}")
        return {"error": str(e), "chart": None, "trend_oscillator": None, "signal_line": None, "trend": "N/A"}

# MomentumSignals Functions
length = 14
calc_length = 5
smooth_length = 3

def calculate_ema(data, period):
    return data.ewm(span=period, adjust=False).mean()

def calculate_pivots(data, timeframe='D'):
    try:
        now = datetime.now()
        if timeframe == 'D':
            # Use the last complete day if available, otherwise the latest
            data = data.iloc[-2:-1] if len(data) > 1 else data.iloc[-1:]
        elif timeframe == 'W':
            data = data[data.index.isocalendar().week == now.isocalendar().week]
        else:
            data = data[(data.index.month == now.month) & (data.index.year == now.year)]
        if data.empty:
            return None
        high = float(data['High'].max())  # Ensure scalar
        low = float(data['Low'].min())    # Ensure scalar
        close = float(data['Close'].iloc[-1])  # Ensure scalar
        return (high + low + close) / 3
    except Exception as e:
        st.error(f"Error in calculate_pivots: {e}")
        return None

@st.cache_data(ttl=300)
def fetch_momentum_data(symbol, interval, period="6mo"):
    try:
        data = yf.download(symbol, period=period, interval=interval)
        if data.empty:
            return pd.DataFrame()
        return data[['Open', 'High', 'Low', 'Close', 'Volume']]
    except Exception as e:
        st.error(f"Error in fetch_momentum_data for {symbol}: {e}")
        return pd.DataFrame()

def fetch_latest_price(symbol):
    try:
        ticker = yf.Ticker(symbol)
        return ticker.history(period="1d")['Close'].iloc[-1]
    except Exception as e:
        st.error(f"Error in fetch_latest_price for {symbol}: {e}")
        return None

def calculate_signals(stock_data):
    try:
        if stock_data.empty or len(stock_data) < length + smooth_length * 2:
            return pd.Series(False, index=stock_data.index if not stock_data.empty else pd.RangeIndex(1)), pd.Series(False, index=stock_data.index if not stock_data.empty else pd.RangeIndex(1))
        
        o = stock_data['Open'].values
        c = stock_data['Close'].values
        
        data_values = []
        for i in range(len(c)):
            sum_val = 0
            for j in range(length):
                idx = max(0, i - j)
                sum_val += 1 if c[i] > o[idx] else -1 if c[i] < o[idx] else 0
            data_values.append(sum_val)
        
        data_series = pd.Series(data_values, index=stock_data.index)
        
        EMA5 = data_series.ewm(span=calc_length, adjust=False).mean()
        Main = EMA5.ewm(span=smooth_length, adjust=False).mean()
        Signal = Main.ewm(span=smooth_length, adjust=False).mean()
        
        buy_signals = np.zeros(len(stock_data), dtype=bool)
        sell_signals = np.zeros(len(stock_data), dtype=bool)
        
        main_arr = Main.values
        signal_arr = Signal.values
        
        main_arr_shifted = np.zeros_like(main_arr)
        main_arr_shifted[1:] = main_arr[:-1]
        
        signal_arr_shifted = np.zeros_like(signal_arr)
        signal_arr_shifted[1:] = signal_arr[:-1]
        
        for i in range(1, len(main_arr)):
            buy_signals[i] = (main_arr[i] > signal_arr[i]) and (main_arr_shifted[i] <= signal_arr_shifted[i])
            sell_signals[i] = (main_arr[i] < signal_arr[i]) and (main_arr_shifted[i] >= signal_arr_shifted[i])
        
        buy_series = pd.Series(buy_signals, index=stock_data.index)
        sell_series = pd.Series(sell_signals, index=stock_data.index)
        
        return buy_series, sell_series
    except Exception as e:
        st.error(f"Error in calculate_signals: {e}")
        return pd.Series(False, index=pd.RangeIndex(1)), pd.Series(False, index=pd.RangeIndex(1))

def calculate_indicators(data):
    try:
        data['EMA_9'] = calculate_ema(data['Close'], 9)
        data['EMA_21'] = calculate_ema(data['Close'], 21)
        data['EMA_50'] = calculate_ema(data['Close'], 50)
        data['EMA_200'] = calculate_ema(data['Close'], 200)
        daily_pivot = calculate_pivots(data, 'D')
        weekly_pivot = calculate_pivots(data, 'W')
        monthly_pivot = calculate_pivots(data, 'M')
        return data, daily_pivot, weekly_pivot, monthly_pivot
    except Exception as e:
        st.error(f"Error in calculate_indicators: {e}")
        return data, None, None, None

@st.cache_data(ttl=300)
def get_momentum(ticker):
    try:
        timeframes = ["1d", "5d"]
        stock_data = fetch_momentum_data(ticker, "1d")
        if stock_data.empty:
            return {"error": "No data available", "1D_signal": "No Data", "5D_signal": "No Data"}
        
        stock_data, daily_pivot, weekly_pivot, monthly_pivot = calculate_indicators(stock_data)
        latest_price = fetch_latest_price(ticker)
        analysis = {}
        
        for timeframe in timeframes:
            data = fetch_momentum_data(ticker, timeframe)
            if not data.empty:
                buy_signals, sell_signals = calculate_signals(data)
                
                is_buy = False
                is_sell = False
                
                if not buy_signals.empty:
                    try:
                        last_buy = buy_signals.iloc[-1]
                        if isinstance(last_buy, pd.Series):
                            is_buy = last_buy.any()
                        else:
                            is_buy = bool(last_buy) if pd.notna(last_buy) else False
                    except:
                        pass
                
                if not sell_signals.empty:
                    try:
                        last_sell = sell_signals.iloc[-1]
                        if isinstance(last_sell, pd.Series):
                            is_sell = last_sell.any()
                        else:
                            is_sell = bool(last_sell) if pd.notna(last_sell) else False
                    except:
                        pass
                
                if is_buy:
                    analysis[timeframe] = "Buy"
                elif is_sell:
                    analysis[timeframe] = "Sell"
                else:
                    analysis[timeframe] = "Neutral"
            else:
                analysis[timeframe] = "No Data"
        
        result = {
            "1D_signal": analysis.get("1d", "No Data"),
            "5D_signal": analysis.get("5d", "No Data")
        }
        
        if latest_price is not None:
            result["price"] = round(latest_price, 2)
        else:
            result["price"] = None
            
        for ema in ["EMA_9", "EMA_21", "EMA_50", "EMA_200"]:
            if not stock_data.empty and ema in stock_data.columns and not stock_data[ema].empty and pd.notna(stock_data[ema].iloc[-1]):
                result[ema] = round(stock_data[ema].iloc[-1], 2)
            else:
                result[ema] = None
        
        # Ensure pivots are floats and rounded
        result["daily_pivot"] = round(daily_pivot, 2) if daily_pivot is not None else None
        result["weekly_pivot"] = round(weekly_pivot, 2) if weekly_pivot is not None else None
        result["monthly_pivot"] = round(monthly_pivot, 2) if monthly_pivot is not None else None
        
        return result
    except Exception as e:
        st.error(f"Error in get_momentum for {ticker}: {e}")
        return {"error": str(e), "1D_signal": "No Data", "5D_signal": "No Data"}

# Seasonality Functions
MONTHS = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12
}

def get_month_number(month_name):
    return MONTHS[month_name]

def fetch_seasonality_data(symbol, start_date, end_date):
    try:
        data = yf.download(symbol, start=start_date, end=end_date)
        data["Day"] = data.index.day
        data["Month"] = data.index.month
        data["Year"] = data.index.year
        data["Daily Return"] = data["Close"].pct_change()
        return data
    except Exception as e:
        st.error(f"Error in fetch_seasonality_data for {symbol}: {e}")
        return None

def calculate_seasonality(data, month_number, current_year):
    try:
        historical_data = data[data.index.year < current_year].copy()
        current_data = data[data.index.year == current_year].copy()
        
        historical_monthly = historical_data[historical_data["Month"] == month_number].copy()
        daily_avg_returns = historical_monthly.groupby("Day")["Daily Return"].agg(["mean", "std", "count"])
        
        confidence_level = 1.96
        daily_avg_returns["ci_lower"] = daily_avg_returns["mean"] - confidence_level * (daily_avg_returns["std"] / np.sqrt(daily_avg_returns["count"]))
        daily_avg_returns["ci_upper"] = daily_avg_returns["mean"] + confidence_level * (daily_avg_returns["std"] / np.sqrt(daily_avg_returns["count"]))
        
        current_monthly = current_data[current_data["Month"] == month_number].copy()
        current_returns = current_monthly.groupby("Day")["Daily Return"].mean()
        
        monthly_returns = historical_monthly.groupby("Year")["Daily Return"].mean()
        t_stat, p_value = stats.ttest_1samp(monthly_returns.dropna(), 0)
        
        return {
            "historical_daily_returns": daily_avg_returns,
            "current_returns": current_returns,
            "monthly_avg_return": monthly_returns.mean(),
            "current_month_return": current_returns.mean() if not current_returns.empty else None,
            "t_statistic": t_stat,
            "p_value": p_value
        }
    except Exception as e:
        st.error(f"Error in calculate_seasonality: {e}")
        return {"error": str(e)}

def plot_seasonality_with_current(seasonality_results, stock, month_name):
    try:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=seasonality_results["historical_daily_returns"].index, y=seasonality_results["historical_daily_returns"]["mean"], mode="lines", name="Historical Average", line=dict(color="blue")))
        fig.add_trace(go.Scatter(x=seasonality_results["historical_daily_returns"].index, y=seasonality_results["historical_daily_returns"]["ci_upper"], mode="lines", line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=seasonality_results["historical_daily_returns"].index, y=seasonality_results["historical_daily_returns"]["ci_lower"], mode="lines", line=dict(width=0), fillcolor="rgba(68, 68, 68, 0.3)", fill="tonexty", name="95% Confidence Interval"))
        if not seasonality_results["current_returns"].empty:
            fig.add_trace(go.Scatter(x=seasonality_results["current_returns"].index, y=seasonality_results["current_returns"], mode="lines+markers", name=f"Current Year ({datetime.now().year})", line=dict(color="red"), marker=dict(size=8)))
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.update_layout(
            title=f"{stock.upper()} Seasonality - {month_name}",
            xaxis_title="Day of Month",
            yaxis_title="Return (%)",
            hovermode="x unified",
            xaxis=dict(tickmode="linear", tick0=1, dtick=1),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            template='plotly_white',
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        return fig
    except Exception as e:
        st.error(f"Error in plot_seasonality_with_current: {e}")
        return None

@st.cache_data(ttl=300)
def get_seasonality(ticker, month_name=None):
    try:
        if month_name is None:
            month_name = list(MONTHS.keys())[datetime.now().month - 1]
        month_number = get_month_number(month_name)
        start_date = date(2011, 1, 1)
        end_date = date.today()
        data = fetch_seasonality_data(ticker, start_date, end_date)
        if data is None or data.empty:
            return {"error": "No data available", "chart": None}
        
        current_year = datetime.now().year
        seasonality_results = calculate_seasonality(data, month_number, current_year)
        if "error" in seasonality_results:
            return {"error": seasonality_results["error"], "chart": None}
        chart = plot_seasonality_with_current(seasonality_results, ticker, month_name)
        return {
            "historical_avg_return": seasonality_results["monthly_avg_return"],
            "current_year_return": seasonality_results["current_month_return"],
            "p_value": seasonality_results["p_value"],
            "chart": chart
        }
    except Exception as e:
        st.error(f"Error in get_seasonality for {ticker}: {e}")
        return {"error": str(e), "chart": None}

# Block Trades Functions
@st.cache_data(ttl=300)
def fetch_block_trade_data(ticker, start_date, end_date):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(start=start_date, end=end_date)
        if hist.empty:
            raise ValueError(f"No data available for {ticker}")
        return hist
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return None

def detect_volume_spikes(df, window=20, threshold=2):
    try:
        df['Volume_MA'] = df['Volume'].rolling(window=window).mean()
        df['Volume_Std'] = df['Volume'].rolling(window=window).std()
        df['Volume_Z_Score'] = (df['Volume'] - df['Volume_MA']) / df['Volume_Std']
        df['Block_Trade'] = df['Volume_Z_Score'] > threshold
        return df
    except Exception as e:
        st.error(f"Error detecting volume spikes: {e}")
        return df

def analyze_block_trade_reaction(df, days_after=5):
    try:
        block_trades = df[df['Block_Trade']].copy()
        if block_trades.empty:
            return block_trades
        
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
    except Exception as e:
        st.error(f"Error analyzing block trade reaction: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def get_block_trades(ticker):
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        hist = fetch_block_trade_data(ticker, start_date, end_date)
        if hist is None or hist.empty:
            return {"error": "No data available", "block_trades": None, "chart": None}
        
        hist = detect_volume_spikes(hist)
        block_trades = analyze_block_trade_reaction(hist)
        
        if block_trades.empty:
            return {"error": "No significant block trades detected", "block_trades": None, "chart": None}
        
        stock = yf.Ticker(ticker)
        current_price = stock.info.get('regularMarketPrice', None)
        block_trades['Ticker'] = ticker
        block_trades['Current_Price'] = current_price
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=block_trades.index,
            y=block_trades['Volume'],
            mode='markers',
            name=ticker,
            marker=dict(
                size=12,
                color=np.where(block_trades['Trade_Type'] == 'Buy', 'green', 'red'),
                line=dict(width=2, color='white')
            ),
            text=[f"{ticker} - 1D: {p1:.2f}%, 5D: {p5:.2f}%" 
                  for p1, p5 in zip(block_trades['Price_Change_1D'], block_trades['Price_Change_5D'])],
            hoverinfo='text'
        ))
        
        fig.update_layout(
            title=f"Block Trades for {ticker}",
            yaxis_title="Volume",
            xaxis_title="Date",
            template='plotly_white',
            plot_bgcolor='white',
            paper_bgcolor='white',
            hovermode='x unified',
            height=600
        )
        
        return {
            "block_trades": block_trades,
            "current_price": current_price,
            "chart": fig
        }
    except Exception as e:
        st.error(f"Error in get_block_trades for {ticker}: {e}")
        return {"error": str(e), "block_trades": None, "chart": None}

# GEX Analysis Functions
def calculate_gamma(S, K, T, r, sigma, option_type='call'):
    if T <= 0.001:
        T = 0.001
    
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    gamma = np.exp(-d1**2/2) / (S*sigma*np.sqrt(2*np.pi*T))
    return gamma

def get_all_expirations(ticker_symbol, max_days=365):
    try:
        ticker = yf.Ticker(ticker_symbol)
        expirations = ticker.options
        
        eastern = pytz.timezone('US/Eastern')
        now = datetime.now(eastern)
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        
        expiration_data = []
        for exp in expirations:
            try:
                exp_date = datetime.strptime(exp, '%Y-%m-%d')
                exp_date = eastern.localize(exp_date)
                days_to_exp = (exp_date.date() - today_start.date()).days
                
                if days_to_exp > max_days or days_to_exp < 0:
                    continue
                    
                opt = ticker.option_chain(exp)
                calls_oi = opt.calls['openInterest'].sum() if not opt.calls.empty else 0
                puts_oi = opt.puts['openInterest'].sum() if not opt.puts.empty else 0
                total_oi = calls_oi + puts_oi
                
                if total_oi > 0:
                    expiration_data.append({
                        'date': exp,
                        'days': days_to_exp,
                        'oi': total_oi,
                        'calls_oi': calls_oi,
                        'puts_oi': puts_oi
                    })
            except Exception as e:
                st.warning(f"Skipping expiration {exp}: {str(e)}")
                continue
                
        if not expiration_data:
            return pd.DataFrame()
            
        df = pd.DataFrame(expiration_data)
        df = df.sort_values('days')
        return df
        
    except Exception as e:
        st.error(f"Error fetching expirations for {ticker_symbol}: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def get_gex(ticker_symbol, expiration=None, price_range_pct=15, threshold=5.0, 
            strike_spacing_override=None, risk_free_rate=0.05, bar_width=2.0, show_labels=True):
    try:
        exp_data = get_all_expirations(ticker_symbol)
        if exp_data.empty:
            return {"error": "No valid expirations found", "exp_data": None, "gex_data": None, "chart": None}
        
        if expiration is None:
            expiration = exp_data['date'].iloc[0]
            
        ticker = yf.Ticker(ticker_symbol)
        hist = ticker.history(period='5d')
        if hist.empty:
            raise ValueError(f"No price data available for {ticker_symbol}")
        
        current_price = hist['Close'].iloc[-1]
        
        exp_date = datetime.strptime(expiration, '%Y-%m-%d')
        today = datetime.now()
        T = max((exp_date - today).days / 365, 0.001)
        
        opt = ticker.option_chain(expiration)
        
        if strike_spacing_override:
            strike_spacing = strike_spacing_override
        else:
            is_index = ticker_symbol in ['SPY', 'QQQ', 'IWM']
            if is_index:
                strike_spacing = 1.0 if current_price > 200 else 0.5
            else:
                if current_price > 500:
                    strike_spacing = 5.0
                elif current_price > 100:
                    strike_spacing = 2.5
                else:
                    strike_spacing = 1.0
                    
        calls = opt.calls[['strike', 'openInterest', 'impliedVolatility']].copy()
        calls['type'] = 'call'
        
        puts = opt.puts[['strike', 'openInterest', 'impliedVolatility']].copy()
        puts['type'] = 'put'
        
        price_range = current_price * (price_range_pct / 100)
        calls = calls[
            (calls['strike'] >= current_price - price_range) & 
            (calls['strike'] <= current_price + price_range)
        ]
        puts = puts[
            (puts['strike'] >= current_price - price_range) & 
            (puts['strike'] <= current_price + price_range)
        ]
        
        calls = calls[calls['openInterest'] > 0].dropna()
        puts = puts[puts['openInterest'] > 0].dropna()
        
        options_data = pd.concat([calls, puts])
        
        if options_data.empty:
            return {
                "error": "No options with open interest",
                "exp_data": exp_data,
                "gex_data": None,
                "chart": None,
                "current_price": current_price
            }
        
        gex_data = []
        for _, row in options_data.iterrows():
            K = row['strike']
            sigma = row['impliedVolatility']
            oi = row['openInterest']
            
            gamma = calculate_gamma(current_price, K, T, risk_free_rate, sigma)
            gex = gamma * oi * current_price / 10000
            if row['type'] == 'put':
                gex = -gex
            
            gex_data.append({
                'strike': K,
                'gex': gex,
                'oi': oi,
                'type': row['type']
            })
        
        df = pd.DataFrame(gex_data)
        if df.empty:
            return {
                "error": "No GEX data calculated",
                "exp_data": exp_data,
                "gex_data": None,
                "chart": None,
                "current_price": current_price
            }
        
        dynamic_threshold = max(np.percentile(df['gex'].abs(), 25), 0.01)
        final_threshold = min(threshold, dynamic_threshold)
        df['abs_gex'] = abs(df['gex'])
        filtered_df = df[df['abs_gex'] > final_threshold].drop('abs_gex', axis=1)
        
        if not filtered_df.empty:
            filtered_df = filtered_df.sort_values('strike')
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=filtered_df['strike'],
                y=filtered_df['gex'],
                marker_color=['green' if x >= 0 else 'red' for x in filtered_df['gex']],
                opacity=0.6,
                width=bar_width,
                text=[f'{x:.1f}' if abs(x) >= filtered_df['gex'].abs().max() * 0.1 and show_labels else '' 
                      for x in filtered_df['gex']],
                textposition='auto'
            ))
            
            fig.add_vline(
                x=current_price,
                line_color='blue',
                line_dash='dash',
                annotation_text=f'Current Price: {current_price:.2f}',
                annotation_position='top right'
            )
            
            stats_text = (f"Total GEX: {filtered_df['gex'].sum():.1f}<br>"
                         f"Max +GEX: {filtered_df['gex'].max():.1f}<br>"
                         f"Max -GEX: {filtered_df['gex'].min():.1f}")
            
            fig.update_layout(
                title=f'Gamma Exposure (GEX) for {ticker_symbol}',
                xaxis_title='Strike Price',
                yaxis_title='GEX',
                showlegend=False,
                template='plotly_white',
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Segoe UI, sans-serif", size=12, color="#2c3e50"),
                height=400,
                annotations=[dict(
                    x=0.02,
                    y=0.98,
                    xref="paper",
                    yref="paper",
                    text=stats_text,
                    showarrow=False,
                    bgcolor="white",
                    opacity=0.8
                )],
                xaxis=dict(showgrid=True, gridcolor='rgba(230,230,230,0.5)'),
                yaxis=dict(showgrid=True, gridcolor='rgba(230,230,230,0.5)')
            )
        else:
            fig = None
            
        return {
            "exp_data": exp_data,
            "gex_data": filtered_df,
            "chart": fig,
            "current_price": current_price,
            "total_gex": filtered_df['gex'].sum() if not filtered_df.empty else 0,
            "max_positive_gex": filtered_df['gex'].max() if not filtered_df.empty else 0,
            "max_negative_gex": filtered_df['gex'].min() if not filtered_df.empty else 0,
            "strongest_gex_strike": filtered_df.loc[filtered_df['gex'].abs().idxmax(), 'strike'] 
                if not filtered_df.empty else None
        }
        
    except Exception as e:
        st.error(f"Error in GEX calculation for {ticker_symbol}: {e}")
        return {"error": str(e), "exp_data": None, "gex_data": None, "chart": None}

        # Whale Positioning Functions
def get_weekly_expirations(ticker_symbol, num_weeks=8):
    try:
        ticker = yf.Ticker(ticker_symbol)
        expirations = ticker.options
        eastern = pytz.timezone('US/Eastern')
        now = datetime.now(eastern)
        today = now.date()
        
        weekly_exps = []
        for exp in expirations:
            exp_date = datetime.strptime(exp, '%Y-%m-%d').date()
            days_to_exp = (exp_date - today).days
            if 0 <= days_to_exp <= (num_weeks * 7 + 7):
                weekly_exps.append({'date': exp, 'days': days_to_exp})
        
        return sorted(weekly_exps, key=lambda x: x['days'])[:num_weeks]
    except Exception as e:
        st.error(f"Error fetching weekly expirations: {str(e)}")
        return []

def fetch_whale_positions(ticker_symbol, expirations, price_range_pct=10):
    try:
        ticker = yf.Ticker(ticker_symbol)
        hist = ticker.history(period='1d', interval='1m')
        current_price = hist['Close'].iloc[-1] if not hist.empty else None
        
        if not current_price:
            raise ValueError("Unable to fetch current price")
            
        last_hour = hist.tail(60)
        price_trend = (last_hour['Close'].iloc[-1] - last_hour['Close'].iloc[0]) / last_hour['Close'].iloc[0] * 100
        
        whale_data = []
        price_min = current_price * (1 - price_range_pct/100)
        price_max = current_price * (1 + price_range_pct/100)
        
        for exp in expirations:
            opt = ticker.option_chain(exp['date'])
            calls = opt.calls[
                (opt.calls['strike'] >= price_min) & 
                (opt.calls['strike'] <= price_max)
            ][['strike', 'openInterest', 'volume']].copy()
            calls['type'] = 'call'
            calls['days_to_exp'] = exp['days']
            calls['expiry_date'] = exp['date']
            
            puts = opt.puts[
                (opt.puts['strike'] >= price_min) & 
                (opt.puts['strike'] <= price_max)
            ][['strike', 'openInterest', 'volume']].copy()
            puts['type'] = 'put'
            puts['days_to_exp'] = exp['days']
            puts['expiry_date'] = exp['date']
            
            whale_data.append(calls)
            whale_data.append(puts)
        
        df = pd.concat(whale_data)
        
        df['price_weight'] = 1 - (abs(df['strike'] - current_price) / (current_price * price_range_pct/100))
        df['time_weight'] = 1 / (df['days_to_exp'] + 1)
        df['weighted_volume'] = df['volume'] * df['price_weight'] * df['time_weight']
        
        agg_df = df.groupby(['strike', 'type', 'expiry_date']).agg({
            'openInterest': 'sum',
            'volume': 'sum',
            'weighted_volume': 'sum'
        }).reset_index()
        
        pivot_df = agg_df.pivot(index=['strike', 'expiry_date'], columns='type', 
                              values=['openInterest', 'volume', 'weighted_volume']).fillna(0)
        pivot_df.columns = ['call_oi', 'put_oi', 'call_vol', 'put_vol', 'call_wv', 'put_wv']
        
        pivot_df['total_oi'] = pivot_df['call_oi'] + pivot_df['put_oi']
        pivot_df['net_vol'] = pivot_df['call_vol'] - pivot_df['put_vol']
        pivot_df['net_wv'] = pivot_df['call_wv'] - pivot_df['put_wv']
        
        strike_df = pivot_df.groupby(level='strike').sum()
        
        return pivot_df, strike_df, current_price, price_trend
    except Exception as e:
        st.error(f"Error fetching whale positions: {str(e)}")
        return pd.DataFrame(), pd.DataFrame(), None, None

def predict_price_direction(current_price, whale_df, price_trend):
    if whale_df.empty:
        return "Insufficient data", None, None, "gray"
    
    call_wv_score = whale_df['call_wv'].sum()
    put_wv_score = whale_df['put_wv'].sum()
    net_wv_score = call_wv_score - put_wv_score
    
    call_oi_concentration = whale_df[whale_df.index.get_level_values('strike') > current_price]['call_oi'].sum()
    put_oi_concentration = whale_df[whale_df.index.get_level_values('strike') < current_price]['put_oi'].sum()
    
    direction_score = (
        (net_wv_score / max(abs(net_wv_score), 1)) * 0.4 +
        (price_trend / max(abs(price_trend), 0.1)) * 0.3 +
        ((call_oi_concentration - put_oi_concentration) / 
         max(call_oi_concentration + put_oi_concentration, 1)) * 0.3
    )
    
    if direction_score > 0:
        target_df = whale_df[whale_df.index.get_level_values('strike') > current_price]
        if not target_df.empty:
            target_row = target_df.loc[target_df['call_wv'].idxmax()]
            target_strike = target_row.name[0]
            target_expiry = target_row.name[1]
        else:
            target_strike = current_price
            target_expiry = whale_df.index.get_level_values('expiry_date')[0]
    else:
        target_df = whale_df[whale_df.index.get_level_values('strike') < current_price]
        if not target_df.empty:
            target_row = target_df.loc[target_df['put_wv'].idxmax()]
            target_strike = target_row.name[0]
            target_expiry = target_row.name[1]
        else:
            target_strike = current_price
            target_expiry = whale_df.index.get_level_values('expiry_date')[0]
    
    if direction_score > 0.3:
        direction = "Bullish"
        color = "green"
    elif direction_score < -0.3:
        direction = "Bearish"
        color = "red"
    else:
        direction = "Neutral"
        color = "gray"
    
    confidence = min(abs(direction_score) * 100, 100)
    
    return (f"{direction} toward ${target_strike:.2f} for {target_expiry} "
            f"(Confidence: {confidence:.0f}%)"), target_strike, target_expiry, color

# Placeholder Functions
@st.cache_data(ttl=300)
def get_whale_positions(ticker_symbol):
    try:
        weekly_exps = get_weekly_expirations(ticker_symbol)
        if not weekly_exps:
            return {"error": "No weekly expirations found", "chart": None, "details": None}
        
        whale_df, strike_df, current_price, price_trend = fetch_whale_positions(ticker_symbol, weekly_exps)
        if whale_df.empty or current_price is None:
            return {"error": "No data within price range", "chart": None, "details": None}
        
        direction, target_strike, target_expiry, color = predict_price_direction(current_price, whale_df, price_trend)
        
        # Convert to Plotly
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=strike_df.index - 0.2,
            y=strike_df['call_wv'],
            width=0.4,
            name='Call Weighted Volume',
            marker_color='green',
            opacity=0.7
        ))
        fig.add_trace(go.Bar(
            x=strike_df.index + 0.2,
            y=strike_df['put_wv'],
            width=0.4,
            name='Put Weighted Volume',
            marker_color='red',
            opacity=0.7
        ))
        fig.add_vline(
            x=current_price,
            line_color='blue',
            line_dash='dash',
            annotation_text=f'Current Price: ${current_price:.2f}',
            annotation_position='top right'
        )
        stats_text = f"Call WV: {strike_df['call_wv'].sum():,.0f}<br>Put WV: {strike_df['put_wv'].sum():,.0f}"
        fig.update_layout(
            title=f"Weighted Volume Trend Analysis for {ticker_symbol} (±10% Price Range)",
            xaxis_title="Strike Price",
            yaxis_title="Weighted Volume",
            template='plotly_white',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Segoe UI, sans-serif", size=12, color="#2c3e50"),
            height=400,
            annotations=[dict(
                x=0.02,
                y=0.98,
                xref="paper",
                yref="paper",
                text=stats_text,
                showarrow=False,
                bgcolor="white",
                opacity=0.8
            )],
            xaxis=dict(showgrid=True, gridcolor='rgba(230,230,230,0.5)'),
            yaxis=dict(showgrid=True, gridcolor='rgba(230,230,230,0.5)')
        )
        
        return {
            "whale_df": whale_df,
            "strike_df": strike_df,
            "current_price": current_price,
            "price_trend": price_trend,
            "direction": direction,
            "target_strike": target_strike,
            "target_expiry": target_expiry,
            "color": color,
            "chart": fig
        }
    except Exception as e:
        st.error(f"Error in whale positions for {ticker_symbol}: {str(e)}")
        return {"error": str(e), "chart": None, "details": None}

# FINRA Short Sale Functions for Individual Symbol Analysis
def download_finra_short_sale_data(date):
    url = f"https://cdn.finra.org/equity/regsho/daily/CNMSshvol{date}.txt"
    response = requests.get(url)
    return response.text if response.status_code == 200 else None

def process_finra_short_sale_data(data):
    if not data:
        return pd.DataFrame()
    df = pd.read_csv(io.StringIO(data), delimiter="|")
    return df[df["Symbol"].str.len() <= 4]

def calculate_metrics(row, total_volume):
    short_volume = row.get('ShortVolume', 0)
    short_exempt_volume = row.get('ShortExemptVolume', 0)
    
    bought_volume = short_volume + short_exempt_volume
    sold_volume = total_volume - bought_volume
    
    buy_to_sell_ratio = bought_volume / sold_volume if sold_volume > 0 else float('inf')
    short_volume_ratio = bought_volume / total_volume if total_volume > 0 else 0
    
    return {
        'total_volume': total_volume,
        'bought_volume': bought_volume,
        'sold_volume': sold_volume,
        'buy_to_sell_ratio': round(buy_to_sell_ratio, 2),
        'short_volume_ratio': round(short_volume_ratio, 4)
    }
def analyze_stock_list(stock_list, timeframe_setup='1H/1D'):
    """
    Analyzes a list of stocks and suggests long/short trades based on app signals.
    
    Parameters:
    - stock_list: List of stock ticker symbols (e.g., ["AAPL", "MSFT", ...])
    - timeframe_setup: Oscillator timeframe (e.g., '1H/1D' or '1D/5D')
    
    Returns:
    - DataFrame with trade recommendations and key metrics
    """
    
    # Initialize result storage
    results = []
    
    # Fetch SPY data once for relative strength calculations
    _, spy_hist = fetch_stock_data("SPY", period="1d", interval="5m")
    
    # Process each stock
    for ticker in stock_list:
        try:
            # Fetch key data from existing functions
            stock_summary, _ = fetch_stock_data(ticker, period="1y", interval="1d", spy_hist=spy_hist)
            momentum_data = get_momentum(ticker)
            oscillator_data = get_oscillator(ticker, timeframe_setup)
            
            # Skip if critical data is missing
            if stock_summary.empty or "error" in momentum_data or "error" in oscillator_data:
                continue
            
            # Extract key signals
            current_price = stock_summary['Current Price'].iloc[0]
            rsi_status = stock_summary['RSI_Status'].iloc[0]
            key_mas = stock_summary['KeyMAs'].iloc[0]
            price_vwap = stock_summary['Price_Vwap'].iloc[0]
            rel_strength = stock_summary['Rel Strength SPY'].iloc[0]
            momentum_1d = momentum_data.get('1D_signal', 'No Data')
            momentum_5d = momentum_data.get('5D_signal', 'No Data')
            trend = oscillator_data.get('trend', 'N/A')
            
            # Define trade recommendation logic
            long_score = 0
            short_score = 0
            
            # Momentum signals
            if momentum_1d == "Buy":
                long_score += 2
            elif momentum_1d == "Sell":
                short_score += 2
            if momentum_5d == "Buy":
                long_score += 1
            elif momentum_5d == "Sell":
                short_score += 1
            
            # Oscillator trend
            if trend == "Bullish":
                long_score += 2
            elif trend == "Bearish":
                short_score += 2
            
            # RSI status
            if rsi_status == "Overbought":
                short_score += 1
            elif rsi_status == "Oversold":
                long_score += 1
            elif rsi_status == "Strong":
                long_score += 1
            elif rsi_status == "Weak":
                short_score += 1
            
            # Key moving averages
            if key_mas == "Bullish":
                long_score += 2
            elif key_mas == "Bearish":
                short_score += 2
            
            # Price vs VWAP
            if price_vwap == "Bullish":
                long_score += 1
            elif price_vwap == "Bearish":
                short_score += 1
            
            # Relative strength vs SPY
            if rel_strength == "Strong":
                long_score += 1
            elif rel_strength == "Weak":
                short_score += 1
            
            # Determine trade recommendation
            if long_score >= 5 and long_score > short_score + 2:
                recommendation = "Long"
                confidence = min((long_score / 10) * 100, 95)  # Cap confidence at 95%
            elif short_score >= 5 and short_score > long_score + 2:
                recommendation = "Short"
                confidence = min((short_score / 10) * 100, 95)
            else:
                recommendation = "Hold"
                confidence = 50
            
            # Compile results
            results.append({
                "Ticker": ticker,
                "Current Price": current_price,
                "Recommendation": recommendation,
                "Long Score": long_score,
                "Short Score": short_score,
                "Confidence (%)": round(confidence, 1),
                "RSI Status": rsi_status,
                "Key MAs": key_mas,
                "Price vs VWAP": price_vwap,
                "Rel Strength SPY": rel_strength,
                "1D Momentum": momentum_1d,
                "5D Momentum": momentum_5d,
                "Oscillator Trend": trend
            })
            
        except Exception as e:
            st.warning(f"Error processing {ticker}: {str(e)}")
            continue
    
    # Convert to DataFrame
    result_df = pd.DataFrame(results)
    
    # Sort by confidence and recommendation strength
    if not result_df.empty:
        result_df = result_df.sort_values(
            by=["Recommendation", "Confidence (%)"],
            ascending=[False, False]
        )
    
    return result_df

@st.cache_data(ttl=300)
def analyze_symbol_finra(symbol, lookback_days=20, threshold=1.5):
    results = []
    significant_days = 0
    
    for i in range(lookback_days * 2):  # Double to account for non-trading days
        date = (datetime.now() - timedelta(days=i)).strftime("%Y%m%d")
        data = download_finra_short_sale_data(date)
        
        if data:
            df = process_finra_short_sale_data(data)
            symbol_data = df[df['Symbol'] == symbol]
            
            if not symbol_data.empty:
                row = symbol_data.iloc[0]
                total_volume = row.get('TotalVolume', 0)
                metrics = calculate_metrics(row, total_volume)
                
                if metrics['buy_to_sell_ratio'] > threshold:
                    significant_days += 1
                
                metrics['date'] = date
                results.append(metrics)
    
    df_results = pd.DataFrame(results)
    if not df_results.empty:
        df_results['date'] = pd.to_datetime(df_results['date'], format='%Y%m%d')
        df_results = df_results.sort_values('date', ascending=False)
    
    return df_results, significant_days

# Main App 
def run():
    # Apply custom CSS for professional styling
    chart_title_prefix = "Intraday"
    st.markdown("""
        <style>
        /* Base styling */
        .main {
            background-color: #f9f9fb;
            padding: 1.5rem;
        }
        
        /* Typography */
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            font-weight: 600;
        }
        
        /* Main title styling */
        .main-title {
            font-size: 2.2rem;
            font-weight: 700;
            color: #2c3e50;
            margin-bottom: 1rem;
            padding-bottom: 0.8rem;
            border-bottom: 2px solid #3498db;
        }
        
        /* Section headers */
        .section-header {
            font-size: 1.5rem;
            color: #2c3e50;
            margin-top: 1.5rem;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid #e0e0e0;
        }
        
        /* Card styling for metric displays */
        .metric-card {
            background-color: white;
            border-radius: 8px;
            padding: 1.2rem;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            height: 100%;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        .metric-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        
        /* Metric styling */
        .metric-label {
            font-size: 0.85rem;
            color: #7f8c8d;
            font-weight: 500;
            margin-bottom: 0.3rem;
        }
        
        .metric-value {
            font-size: 1.2rem;
            font-weight: 600;
            color: #2c3e50;
        }
        
        .metric-bullish {
            color: #27ae60;
            font-weight: 600;
        }
        
        .metric-bearish {
            color: #e74c3c;
            font-weight: 600;
        }
        
        .metric-neutral {
            color: #f39c12;
            font-weight: 600;
        }
        
        /* Chart containers */
        .chart-container {
            background-color: white;
            border-radius: 8px;
            padding: 1rem;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            margin-top: 1rem;
            margin-bottom: 1.5rem;
        }
        
        /* Input styling */
        .stTextInput > div > div > input {
            background-color: white;
            color: #2c3e50;
            border-radius: 4px;
            border: 1px solid #e0e0e0;
            padding: 0.5rem 1rem;
            font-size: 1rem;
        }
        
        .stTextInput > div > div > input:focus {
            border-color: #3498db;
            box-shadow: 0 0 0 2px rgba(52,152,219,0.2);
        }
        
        /* Sidebar styling */
        .sidebar .sidebar-content {
            background-color: white;
            padding: 1.5rem;
        }
        
        /* Button styling */
        .stButton > button {
            background-color: #3498db;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 0.5rem 1.2rem;
            font-size: 0.9rem;
            font-weight: 500;
            transition: background-color 0.2s;
        }
        
        .stButton > button:hover {
            background-color: #2980b9;
        }
        
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 1rem;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 3rem;
            white-space: pre-wrap;
            background-color: white;
            border-radius: 4px 4px 0 0;
            border: 1px solid #e0e0e0;
            border-bottom: none;
            padding: 0.5rem 1rem;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: white;
            border-top: 3px solid #3498db;
        }
        
        /* Alert and info boxes */
        .info-box {
            background-color: #d1ecf1;
            border-left: 4px solid #17a2b8;
            padding: 1rem;
            border-radius: 4px;
            margin-bottom: 1rem;
        }
        
        .warning-box {
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 1rem;
            border-radius: 4px;
            margin-bottom: 1rem;
        }
        
        /* Status Indicators */
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        
        .status-bullish {
            background-color: #27ae60;
        }
        
        .status-bearish {
            background-color: #e74c3c;
        }
        
        .status-neutral {
            background-color: #f39c12;
        }
        
        /* Tables */
        .dataframe {
            border-collapse: collapse;
            width: 100%;
            margin-bottom: 1rem;
            font-size: 0.9rem;
        }
        
        .dataframe th {
            background-color: #f8f9fa;
            color: #2c3e50;
            font-weight: 500;
            text-align: left;
            padding: 0.75rem;
            border-bottom: 2px solid #e0e0e0;
        }
        
        .dataframe td {
            padding: 0.75rem;
            border-bottom: 1px solid #e0e0e0;
        }
        
        .dataframe tr:hover {
            background-color: #f8f9fa;
        }
        </style>
    """, unsafe_allow_html=True)

    # App header with logo and title
    col_logo, col_title, col_empty = st.columns([1, 8, 1])
    with col_logo:
        st.markdown("📈")
    with col_title:
        st.markdown('<h1 class="main-title">Stock Insights Hub</h1>', unsafe_allow_html=True)
        st.markdown('<p>Stock analysis tools for informed trading decisions</p>', unsafe_allow_html=True)

    # Main layout with input section
    with st.container() as input_container:
        st.write("### Stock Selection")
        
        # Add tabs for selection methods
        selection_method = st.radio(
            "Choose analysis method:",
            ["Single Stock", "Pre-defined Lists", "Custom List"],
            horizontal=True
        )
        
        if selection_method == "Single Stock":
            ticker = st.text_input("Enter Stock Ticker", "").upper()
            stock_list_input = ""
            selected_list = None
        
        elif selection_method == "Pre-defined Lists":
            ticker = ""
            stock_list_input = ""
            selected_list = st.selectbox(
                "Select a pre-defined list of stocks",
                options=list(STOCK_LISTS.keys())
            )
            
            if selected_list:
                stocks = STOCK_LISTS[selected_list]
                st.write(f"Selected {len(stocks)} stocks: {', '.join(stocks)}")
        
        else:  # Custom List
            ticker = ""
            selected_list = None
            stock_list_input = st.text_area(
                "Enter Custom Stock List (comma-separated)",
                "",
                height=100,
                help="Example: AAPL, MSFT, GOOGL, AMZN"
            )
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            timeframe_setup = st.selectbox("Oscillator Timeframe", ['1H/1D', '1D/5D'], index=0)
        with col2:
            # Add new price chart timeframe selector
            price_chart_timeframe = st.selectbox(
                "Price Chart Timeframe",
                ['1Y Daily','1D 5min'],
                index=0  # Default to 1D 5min
            )
        with col3:
            analyze_button = st.button("Analyze", use_container_width=True)

    st.markdown('<hr style="margin: 1rem 0; border: 0; border-top: 1px solid #e0e0e0;">', unsafe_allow_html=True)

    # Process inputs and run analysis
    if analyze_button:
        # Handle pre-defined list selection
        if selected_list:
            stock_list = STOCK_LISTS[selected_list]
        # Handle custom list input
        elif stock_list_input:
            stock_list = [s.strip().upper() for s in stock_list_input.split(",") if s.strip()]
        # Handle single ticker
        elif ticker:
            stock_list = None
        else:
            st.warning("Please enter a ticker symbol or select a list of stocks to analyze.")
            st.stop()
            
        # Process the stock list (either pre-defined or custom)
        if stock_list:
            if len(stock_list) > 0:
                with st.spinner(f"Analyzing {len(stock_list)} stocks..."):
                    trade_recommendations = analyze_stock_list(stock_list, timeframe_setup)
                
                if not trade_recommendations.empty:
                    st.markdown('<h2 class="section-header">Trade Recommendations</h2>', unsafe_allow_html=True)
                    
                    # Add filter controls
                    filter_cols = st.columns(3)
                    with filter_cols[0]:
                        show_only = st.multiselect(
                            "Filter by Recommendation",
                            options=["Long", "Short", "Hold"],
                            default=["Long", "Short", "Hold"]
                        )
                    with filter_cols[1]:
                        min_confidence = st.slider("Minimum Confidence (%)", 0, 100, 0)
                    with filter_cols[2]:
                        sort_by = st.selectbox(
                            "Sort by",
                            options=["Confidence (%)", "Ticker"],
                            index=0
                        )
                    
                    # Apply filters
                    filtered_df = trade_recommendations[
                        (trade_recommendations["Recommendation"].isin(show_only)) &
                        (trade_recommendations["Confidence (%)"] >= min_confidence)
                    ]
                    
                    # Sort the data
                    if sort_by == "Confidence (%)":
                        filtered_df = filtered_df.sort_values("Confidence (%)", ascending=False)
                    else:
                        filtered_df = filtered_df.sort_values("Ticker")
                    
                    # Style the DataFrame for better readability
                    def highlight_recommendation(row):
                        color = '#90ee90' if row['Recommendation'] == 'Long' else '#ffcccb' if row['Recommendation'] == 'Short' else '#f0f0f0'
                        return [f'background-color: {color}' if col == 'Recommendation' else '' for col in row.index]
                    
                    styled_df = filtered_df.style.apply(highlight_recommendation, axis=1).format({
                        'Current Price': '${:.2f}',
                        'Confidence (%)': '{:.1f}%'
                    })
                    
                    st.dataframe(styled_df, use_container_width=True, height=600)
                    
                    # Summary stats
                    st.markdown('<h3 class="section-header">Summary</h3>', unsafe_allow_html=True)
                    summary_cols = st.columns(3)
                    with summary_cols[0]:
                        st.markdown(f'<div class="metric-card"><div class="metric-label">Long Recommendations</div><div class="metric-value metric-bullish">{len(filtered_df[filtered_df["Recommendation"] == "Long"])}</div></div>', unsafe_allow_html=True)
                    with summary_cols[1]:
                        st.markdown(f'<div class="metric-card"><div class="metric-label">Short Recommendations</div><div class="metric-value metric-bearish">{len(filtered_df[filtered_df["Recommendation"] == "Short"])}</div></div>', unsafe_allow_html=True)
                    with summary_cols[2]:
                        st.markdown(f'<div class="metric-card"><div class="metric-label">Hold Recommendations</div><div class="metric-value metric-neutral">{len(filtered_df[filtered_df["Recommendation"] == "Hold"])}</div></div>', unsafe_allow_html=True)
                    
                    # Option to download results
                    csv = filtered_df.to_csv(index=False)
                    st.download_button(
                        label="Download Recommendations as CSV",
                        data=csv,
                        file_name=f"trade_recommendations_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv",
                    )
                else:
                    st.warning("No valid trade recommendations could be generated from the provided stock list.")
            else:
                st.error("Please enter a valid list of stock tickers separated by commas.")
        
        
        
        elif ticker:
            # Initialize data containers
            stock_summary = pd.DataFrame()
            stock_hist = pd.DataFrame()
            oscillator_data = {}
            momentum_data = {}
            seasonality_data = {}
            block_trade_data = {}
            gex_data = {"error": "Data not yet loaded"}
            whale_data = {"error": "Data not yet loaded"}
            
            # Convert selection to yfinance parameters
            if price_chart_timeframe == '1Y Daily':
                period = "1y"
                interval = "1d"
                chart_title_prefix = "1-Year Daily"
            elif price_chart_timeframe == '6M Daily':
                period = "6mo"
                interval = "1d"
                chart_title_prefix = "6-Month Daily"
            elif price_chart_timeframe == '10D 30min':
                period = "10d"
                interval = "30m"
                chart_title_prefix = "10-Day 30min"
            else:  # '1D 5min'
                period = "1d"
                interval = "5m"
                chart_title_prefix = "Intraday 5min"
            
            with st.spinner("Analyzing market data..."):
                try:
                    # Fetch all required data
                    _, spy_hist = fetch_stock_data("SPY", period=period, interval=interval)
                    stock_summary, stock_hist = fetch_stock_data(ticker, period=period, interval=interval, spy_hist=spy_hist)
                    oscillator_data = get_oscillator(ticker, timeframe_setup)
                    momentum_data = get_momentum(ticker)
                    seasonality_data = get_seasonality(ticker)
                    block_trade_data = get_block_trades(ticker)
                    gex_data = get_gex(ticker)
                    whale_data = get_whale_positions(ticker)
                    
                    # Add flow summary data fetch here
                    flow_summary = get_top_otm_flows(ticker, FLOW_URLS)
                    
                except Exception as e:
                    st.error(f"Error retrieving data for {ticker}: {str(e)}")
                    st.stop()

            # Quick overview - Summary card at the top
            st.markdown(f'<h2 class="section-header">{ticker} Momentum Signals</h2>', unsafe_allow_html=True)
            overview_cols = st.columns(4)
        
            # Only show if we have momentum data
            if isinstance(momentum_data, dict) and "error" not in momentum_data:
                with overview_cols[0]:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.markdown(f'<div class="metric-label">Current Price</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="metric-value">${momentum_data.get("price", "N/A")}</div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Other overview columns...
            
            # ADD THE FLOW SUMMARY SECTION HERE ON THE MAIN PAGE
            st.markdown('<h3 class="section-header">Top 10 OTM Options Flows</h3>', unsafe_allow_html=True)
            
            if 'flow_summary' in locals() and not flow_summary.empty:
                flow_cols = st.columns([1, 2])

                with flow_cols[0]:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    current_price = get_current_price(ticker)
                    total_value = flow_summary['Transaction Value'].sum()
                    call_count = len(flow_summary[flow_summary['Call/Put'] == 'C'])
                    put_count = len(flow_summary[flow_summary['Call/Put'] == 'P'])
                    sentiment = "Bullish" if call_count > put_count else "Bearish" if put_count > call_count else "Neutral"
                    sentiment_class = "metric-bullish" if sentiment == "Bullish" else "metric-bearish" if sentiment == "Bearish" else "metric-neutral"
                    
                    # Fix for f-string formatting issue
                    price_display = f"${current_price:.2f}" if current_price is not None else "N/A"
                    
                    st.markdown(f'''
                        <p style="font-weight:600; margin-bottom:10px;">Flow Overview</p>
                        <div class="metric-label">Current Price</div>
                        <div class="metric-value">{price_display}</div>
                        <div class="metric-label" style="margin-top:10px;">Total Transaction Value</div>
                        <div class="metric-value">${total_value:,.2f}</div>
                        <div class="metric-label" style="margin-top:10px;">Call/Put Split</div>
                        <div class="metric-value">{call_count} Calls / {put_count} Puts</div>
                        <div class="metric-label" style="margin-top:10px;">Sentiment</div>
                        <div class="metric-value {sentiment_class}">{sentiment}</div>
                    ''', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                with flow_cols[1]:
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)

                    # Format the DataFrame for display
                    display_df = flow_summary.copy()
                    display_df['Expiration'] = display_df['Expiration'].dt.strftime('%Y-%m-%d')
                    display_df['Transaction Value'] = display_df['Transaction Value'].apply(lambda x: f"${x:,.2f}")
                    display_df['Last Price'] = display_df['Last Price'].apply(lambda x: f"${x:.2f}")
                    display_df = display_df.rename(columns={
                        'Call/Put': 'Type',
                        'Strike Price': 'Strike',
                        'Last Price': 'Price',
                        'Transaction Value': 'Value'
                    })

                    # Style the DataFrame
                    def highlight_type(row):
                        color = '#90ee90' if row['Type'] == 'C' else '#ffcccb' if row['Type'] == 'P' else ''
                        return [f'background-color: {color}' if col == 'Type' else '' for col in row.index]

                    styled_df = display_df.style.apply(highlight_type, axis=1)
                    st.dataframe(styled_df, use_container_width=True, height=400)
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                st.markdown(f'No OTM options flow data available for {ticker}. This may be due to unavailable data from CBOE or no qualifying OTM transactions.', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Add a horizontal separator before the tabs section
            st.markdown('<hr style="margin: 2rem 0; border: 0; border-top: 1px solid #e0e0e0;">', unsafe_allow_html=True)
            
        
        # Main analysis section
        main_tabs = st.tabs(["Technical Analysis", "Trend Oscillator", "Insights:Seasonality,Block Trades,GEX Analysis,Whale Positions,FINRA Short Sales,Flow Summary"])
        
        # Technical Analysis Tab
        with main_tabs[0]:
            if 'chart_title_prefix' in locals() and ticker:
               st.markdown(f'<h3 class="section-header">{chart_title_prefix} Price Action & Indicators</h3>', unsafe_allow_html=True)
            else:
               st.markdown('<h3 class="section-header">Price Action & Indicators</h3>', unsafe_allow_html=True)
    
            # Only proceed with technical analysis if we're analyzing a single ticker
            # and stock_summary has been defined and isn't empty
            if 'stock_summary' in locals() and ticker and not stock_summary.empty:
                # Metrics in top row
                metric_cols = st.columns(3)
        
                with metric_cols[0]:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.markdown('<p style="font-weight:600; margin-bottom:10px;">Price Data</p>', unsafe_allow_html=True)
            
                    price_status = stock_summary['Price_Vwap'].iloc[0]
                    price_class = "metric-bullish" if price_status == "Bullish" else "metric-bearish" if price_status == "Bearish" else "metric-neutral"
            
                    st.markdown(f'''
                        <div class="metric-label">Current Price</div>
                        <div class="metric-value">${stock_summary['Current Price'].iloc[0]:.2f}</div>
                        <div class="metric-label" style="margin-top:10px;">VWAP</div>
                        <div class="metric-value">${stock_summary['VWAP'].iloc[0]:.2f}</div>
                        <div class="metric-label" style="margin-top:10px;">Price vs VWAP</div>
                        <div class="metric-value {price_class}">{price_status}</div>
                    ''', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
        
                with metric_cols[1]:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.markdown('<p style="font-weight:600; margin-bottom:10px;">Key Indicators</p>', unsafe_allow_html=True)
            
                    mas_status = stock_summary['KeyMAs'].iloc[0]
                    mas_class = "metric-bullish" if mas_status == "Bullish" else "metric-bearish" if mas_status == "Bearish" else "metric-neutral"
            
                    rsi_status = stock_summary['RSI_Status'].iloc[0]
                    rsi_class = "metric-bullish" if rsi_status in ["Strong", "Overbought"] else "metric-bearish" if rsi_status in ["Weak", "Oversold"] else "metric-neutral"
            
                    st.markdown(f'''
                        <div class="metric-label">EMA21</div>
                        <div class="metric-value">${stock_summary['EMA21'].iloc[0]:.2f}</div>
                        <div class="metric-label" style="margin-top:10px;">Key Moving Averages</div>
                        <div class="metric-value {mas_class}">{mas_status}</div>
                        <div class="metric-label" style="margin-top:10px;">RSI Status</div>
                        <div class="metric-value {rsi_class}">{rsi_status}</div>
                    ''', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
        
                with metric_cols[2]:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.markdown('<p style="font-weight:600; margin-bottom:10px;">Market Context</p>', unsafe_allow_html=True)
            
                    rs_status = stock_summary['Rel Strength SPY'].iloc[0]
                    rs_class = "metric-bullish" if rs_status == "Strong" else "metric-bearish" if rs_status == "Weak" else "metric-neutral"
            
                    st.markdown(f'''
                        <div class="metric-label">Daily Pivot</div>
                        <div class="metric-value">${stock_summary['Daily Pivot'].iloc[0]:.2f}</div>
                        <div class="metric-label" style="margin-top:10px;">Relative Strength (vs SPY)</div>
                        <div class="metric-value {rs_class}">{rs_status}</div>
                    ''', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
        
                # Chart in a nice container
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                fig = plot_candlestick(stock_hist, ticker, price_chart_timeframe)
                # Enhance the chart styling and update title
                fig.update_layout(
                    title=f'{ticker} {chart_title_prefix} Price Action',
                    template='plotly_white',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(
                        showgrid=True,
                        gridcolor='rgba(230,230,230,0.5)'
                    ),
                    yaxis=dict(
                        showgrid=True,
                        gridcolor='rgba(230,230,230,0.5)',
                        title='Price ($)'
                    ),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ),
                    font=dict(
                        family="Segoe UI, sans-serif",
                        size=12,
                        color="#2c3e50"
                    ),
                    height=500,
                    margin=dict(l=40, r=40, t=80, b=40)
                )
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
        
                # Moving averages table in clean format
                st.markdown('<h3 class="section-header">Moving Averages on Daily Timeframe</h3>', unsafe_allow_html=True)
                ma_cols = st.columns(3)
        
                if isinstance(momentum_data, dict) and "error" not in momentum_data:
                    with ma_cols[0]:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.markdown('<p style="font-weight:600; margin-bottom:10px;">Short-Term</p>', unsafe_allow_html=True)
                        st.markdown(f'''
                            <div class="metric-label">EMA 9</div>
                            <div class="metric-value">${momentum_data.get('EMA_9', 'N/A')}</div>
                            <div class="metric-label" style="margin-top:10px;">EMA 21</div>
                            <div class="metric-value">${momentum_data.get('EMA_21', 'N/A')}</div>
                        ''', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
            
                    with ma_cols[1]:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.markdown('<p style="font-weight:600; margin-bottom:10px;">Long-Term</p>', unsafe_allow_html=True)
                        st.markdown(f'''
                            <div class="metric-label">EMA 50</div>
                            <div class="metric-value">${momentum_data.get('EMA_50', 'N/A')}</div>
                            <div class="metric-label" style="margin-top:10px;">EMA 200</div>
                            <div class="metric-value">${momentum_data.get('EMA_200', 'N/A')}</div>
                        ''', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
            
                    with ma_cols[2]:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.markdown('<p style="font-weight:600; margin-bottom:10px;">Key Levels</p>', unsafe_allow_html=True)
                        st.markdown(f'''
                            <div class="metric-label">Daily Pivot</div>
                            <div class="metric-value">${momentum_data.get('daily_pivot', 'N/A')}</div>
                            <div class="metric-label" style="margin-top:10px;">Weekly Pivot</div>
                            <div class="metric-value">${momentum_data.get('weekly_pivot', 'N/A')}</div>
                            <div class="metric-label" style="margin-top:10px;">Monthly Pivot</div>
                            <div class="metric-value">${momentum_data.get('monthly_pivot', 'N/A')}</div>
                        ''', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
            elif selection_method == "Pre-defined Lists" or selection_method == "Custom List":
                st.markdown('<div class="info-box">', unsafe_allow_html=True)
                st.markdown('Technical analysis is only available for single stock mode. Please switch to "Single Stock" mode to see detailed technical analysis.', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                st.markdown('No technical data available. Please enter a valid ticker symbol.', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Trend Oscillator Tab
        with main_tabs[1]:
            st.markdown(f'<h3 class="section-header">Trend Oscillator ({timeframe_setup})</h3>', unsafe_allow_html=True)
            
            if 'oscillator_data' in locals() and ticker and isinstance(oscillator_data, dict) and "error" not in oscillator_data:
                oscillator_cols = st.columns([1, 2])
                
                with oscillator_cols[0]:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    
                    trend = oscillator_data.get('trend', 'N/A')
                    trend_class = "metric-bullish" if trend == "Bullish" else "metric-bearish" if trend == "Bearish" else "metric-neutral"
                    
                    st.markdown(f'''
                        <p style="font-weight:600; margin-bottom:15px;">Oscillator Signals</p>
                        <div class="metric-label">Trend Oscillator</div>
                        <div class="metric-value">{oscillator_data.get('trend_oscillator', 'N/A'):.2f}</div>
                        <div class="metric-label" style="margin-top:15px;">Signal Line</div>
                        <div class="metric-value">{oscillator_data.get('signal_line', 'N/A'):.2f}</div>
                        <div class="metric-label" style="margin-top:15px;">Current Trend</div>
                        <div class="metric-value {trend_class}">
                            <span class="status-indicator status-{trend_class.split('-')[1]}"></span>
                            {trend}
                        </div>
                        <div style="margin-top:20px;">
                            <p style="margin-bottom:5px; font-size:0.85rem; color:#7f8c8d;">Interpretation Guide:</p>
                            <ul style="font-size:0.8rem; color:#7f8c8d; padding-left:15px;">
                                <li>Above 65: Strong bullish trend</li>
                                <li>Between 50-65: Moderate bullish trend</li>
                                <li>Between 30-50: Moderate bearish trend</li>
                                <li>Below 30: Strong bearish trend</li>
                            </ul>
                        </div>
                    ''', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with oscillator_cols[1]:
                    if oscillator_data["chart"]:
                        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                        # Enhance chart styling for oscillator chart
                        oscillator_data["chart"].update_layout(
                            template='plotly_white',
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(
                                family="Segoe UI, sans-serif",
                                size=12,
                                color="#2c3e50"
                            ),
                            height=600,
                            margin=dict(l=40, r=40, t=60, b=40)
                        )
                        st.plotly_chart(oscillator_data["chart"], use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)
            elif selection_method == "Pre-defined Lists" or selection_method == "Custom List":
                 st.markdown('<div class="info-box">', unsafe_allow_html=True)
                 st.markdown('Trend Oscillator analysis is only available for single stock mode. Please switch to "Single Stock" mode to see detailed oscillator analysis.', unsafe_allow_html=True)
                 st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                st.markdown('No oscillator data available. Please enter a valid ticker symbol.', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Insights Tab (Seasonality & Block Trades)
        with main_tabs[2]:
            insight_tabs = st.tabs(["Seasonality", "Block Trades", "FINRA Analysis", "Gamma Exposure", "Whale Positioning", "Flow Summary"])
            
            # Seasonality subtab
            with insight_tabs[0]:
                st.markdown('<h3 class="section-header">Seasonality Analysis</h3>', unsafe_allow_html=True)
                
                if 'seasonality_data' in locals() and ticker and isinstance(seasonality_data, dict) and "error" not in seasonality_data:
                    seasonality_cols = st.columns([1, 3])
                    
                    with seasonality_cols[0]:
                        month_name = list(MONTHS.keys())[datetime.now().month - 1]
                        historical_return = seasonality_data.get('historical_avg_return')
                        current_return = seasonality_data.get('current_year_return')
                        p_value = seasonality_data.get('p_value')
                        
                        historical_class = "metric-bullish" if historical_return and historical_return > 0 else "metric-bearish" if historical_return and historical_return < 0 else "metric-neutral"
                        current_class = "metric-bullish" if current_return and current_return > 0 else "metric-bearish" if current_return and current_return < 0 else "metric-neutral"
                        
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.markdown(f'''
                            <p style="font-weight:600; margin-bottom:10px;">{month_name} Seasonality</p>
                            <div class="metric-label">Historical Average Return</div>
                            <div class="metric-value {historical_class}">
                                {f"{historical_return:.2%}" if historical_return is not None else "N/A"}
                            </div>
                            
                            <div class="metric-label" style="margin-top:15px;">Current Year Return</div>
                            <div class="metric-value {current_class}">
                                {f"{current_return:.2%}" if current_return is not None else "N/A"}
                            </div>
                            
                            <div class="metric-label" style="margin-top:15px;">Statistical Significance</div>
                            <div class="metric-value">
                                {f"p-value: {p_value:.3f}" if p_value is not None else "N/A"}
                            </div>
                            
                            <div style="margin-top:20px;">
                                <p style="margin-bottom:5px; font-size:0.85rem; color:#7f8c8d;">Interpretation:</p>
                                <p style="font-size:0.8rem; color:#7f8c8d;">
                                    {f"The historical pattern for {ticker} in {month_name} is {'statistically significant' if p_value and p_value < 0.05 else 'not statistically significant'}." if p_value is not None else ""}
                                </p>
                            </div>
                        ''', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with seasonality_cols[1]:
                        if seasonality_data["chart"]:
                            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                            # Enhance seasonality chart
                            seasonality_data["chart"].update_layout(
                                template='plotly_white',
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                font=dict(
                                    family="Segoe UI, sans-serif",
                                    size=12,
                                    color="#2c3e50"
                                ),
                                height=400,
                                margin=dict(l=40, r=40, t=60, b=40)
                            )
                            st.plotly_chart(seasonality_data["chart"], use_container_width=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                elif selection_method == "Pre-defined Lists" or selection_method == "Custom List":
                        st.markdown('<div class="info-box">', unsafe_allow_html=True)
                        st.markdown('Seasonality analysis is only available for single stock mode. Please switch to "Single Stock" mode to see detailed seasonality patterns.', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                else:
                        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                        st.markdown('No seasonality data available. Please enter a valid ticker symbol.', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
            
            # Block Trades subtab
            with insight_tabs[1]:
                st.markdown('<h3 class="section-header">Institutional Block Trades</h3>', unsafe_allow_html=True)
                
                if 'block_trade_data' in locals() and ticker and isinstance(block_trade_data, dict) and "error" not in block_trade_data:
                    block_cols = st.columns([1, 2])
                    
                    with block_cols[0]:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.markdown(f'''
                            <p style="font-weight:600; margin-bottom:10px;">Block Trade Analysis</p>
                            <div class="metric-label">Current Price</div>
                            <div class="metric-value">${f"{block_trade_data['current_price']:.2f}" if block_trade_data['current_price'] is not None else "N/A"}</div>
                            
                            <div style="margin-top:20px;">
                                <p style="margin-bottom:5px; font-size:0.85rem; color:#7f8c8d;">What are block trades?</p>
                                <p style="font-size:0.8rem; color:#7f8c8d;">
                                    Block trades are large transactions typically executed by institutional investors.
                                    They often signal significant market interest and can precede major price moves.
                                </p>
                            </div>
                        ''', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with block_cols[1]:
                        if block_trade_data["block_trades"] is not None:
                            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                            
                            # Display the block trades data in a nice table format first
                            if not block_trade_data["block_trades"].empty:
                                cols_to_show = ['Volume', 'Price_Change_1D', 'Price_Change_5D', 'Trade_Type']
                                
                                # Format dataframe for display
                                display_df = block_trade_data["block_trades"][cols_to_show].copy()
                                display_df = display_df.reset_index()
                                display_df.columns = ['Date', 'Volume', '1-Day Change (%)', '5-Day Change (%)', 'Type']
                                
                                # Format the date column
                                display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
                                
                                # Format percentage columns
                                display_df['1-Day Change (%)'] = display_df['1-Day Change (%)'].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "N/A")
                                display_df['5-Day Change (%)'] = display_df['5-Day Change (%)'].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "N/A")
                                
                                # Apply custom styling with HTML
                                st.markdown('<p style="font-weight:600; margin-bottom:10px;">Recent Block Trades</p>', unsafe_allow_html=True)
                                st.dataframe(display_df, use_container_width=True, height=200)
                            
                            # Then display the chart
                            if block_trade_data["chart"]:
                                # Enhance block trades chart
                                block_trade_data["chart"].update_layout(
                                    template='plotly_white',
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    font=dict(
                                        family="Segoe UI, sans-serif",
                                        size=12,
                                        color="#2c3e50"
                                    ),
                                    height=400,
                                    margin=dict(l=40, r=40, t=60, b=40)
                                )
                                st.plotly_chart(block_trade_data["chart"], use_container_width=True)
                            
                            st.markdown('</div>', unsafe_allow_html=True)
                elif selection_method == "Pre-defined Lists" or selection_method == "Custom List":
                    st.markdown('<div class="info-box">', unsafe_allow_html=True)
                    st.markdown('Block trade analysis is only available for single stock mode. Please switch to "Single Stock" mode to see institutional block trades.', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                    st.markdown(f'No block trade data available: {block_trade_data["error"]}', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
            
            # FINRA Analysis subtab
            with insight_tabs[2]:
                        st.markdown('<h3 class="section-header">FINRA Short Sale Analysis</h3>', unsafe_allow_html=True)
                        col1, col2 = st.columns(2)
                        with col1:
                            # Use the same ticker as the main app
                            finra_lookback_days = st.slider("Lookback Days", 1, 30, 20, key="finra_lookback")
                        with col2:
                            finra_threshold = st.number_input("Buy/Sell Ratio Threshold", min_value=1.0, max_value=5.0, value=1.5, step=0.1)

                        if ticker:  # Only run if a ticker is entered
                            with st.spinner("Analyzing FINRA short sale data..."):
                                results_df, significant_days = analyze_symbol_finra(ticker, finra_lookback_days, finra_threshold)

                                if not results_df.empty:
                                    finra_cols = st.columns(3)
                                    with finra_cols[0]:
                                        avg_ratio = results_df['buy_to_sell_ratio'].mean()
                                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                                        st.markdown('<p style="font-weight:600; margin-bottom:10px;">Buy/Sell Ratio</p>', unsafe_allow_html=True)
                                        ratio_class = "metric-bullish" if avg_ratio > 1.0 else "metric-bearish"
                                        st.markdown(f'''
                                            <div class="metric-label">Average Ratio</div>
                                            <div class="metric-value {ratio_class}">{avg_ratio:.2f}</div>
                                        ''', unsafe_allow_html=True)
                                        st.markdown('</div>', unsafe_allow_html=True)

                                    with finra_cols[1]:
                                        max_ratio = results_df['buy_to_sell_ratio'].max()
                                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                                        st.markdown('<p style="font-weight:600; margin-bottom:10px;">Maximum Ratio</p>', unsafe_allow_html=True)
                                        st.markdown(f'''
                                            <div class="metric-label">Highest Recorded</div>
                                            <div class="metric-value">{max_ratio:.2f}</div>
                                        ''', unsafe_allow_html=True)
                                        st.markdown('</div>', unsafe_allow_html=True)

                                    with finra_cols[2]:
                                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                                        st.markdown('<p style="font-weight:600; margin-bottom:10px;">Buying Pressure</p>', unsafe_allow_html=True)
                                        pressure_class = "metric-bullish" if significant_days > finra_lookback_days/4 else "metric-neutral"
                                        st.markdown(f'''
                                            <div class="metric-label">Days Above Threshold</div>
                                            <div class="metric-value {pressure_class}">{significant_days} of {min(len(results_df), finra_lookback_days)}</div>
                                        ''', unsafe_allow_html=True)
                                        st.markdown('</div>', unsafe_allow_html=True)

                                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)

                                    # Create a buy/sell ratio chart
                                    fig = go.Figure()
                                    fig.add_trace(go.Bar(
                                        x=results_df['date'],
                                        y=results_df['buy_to_sell_ratio'],
                                        marker_color=['green' if x > finra_threshold else 'gray' for x in results_df['buy_to_sell_ratio']],
                                        name='Buy/Sell Ratio'
                                    ))
                                    fig.add_hline(y=finra_threshold, line_dash="dash", line_color="red", annotation_text=f"Threshold: {finra_threshold}")
                                    fig.add_hline(y=1.0, line_dash="dash", line_color="gray", annotation_text="Neutral: 1.0")

                                    fig.update_layout(
                                        title=f'FINRA Short Sale Buy/Sell Ratio for {ticker}',
                                        xaxis_title='Date',
                                        yaxis_title='Buy/Sell Ratio',
                                        template='plotly_white',
                                        plot_bgcolor='rgba(0,0,0,0)',
                                        paper_bgcolor='rgba(0,0,0,0)',
                                        height=400,
                                        margin=dict(l=40, r=40, t=60, b=40)
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                    st.markdown('</div>', unsafe_allow_html=True)

                                    st.subheader("Daily FINRA Short Sale Data")
                                    st.markdown('''
                                        <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;">
                                            <p style="font-size: 0.9rem; color: #666;">
                                                <strong>Understanding the data:</strong> The Buy/Sell Ratio represents the relationship between short volume (buying pressure) 
                                                and regular volume (selling pressure). A ratio > 1.0 indicates net buying, while < 1.0 indicates net selling. 
                                                Days with ratios above the threshold are highlighted in green.
                                            </p>
                                        </div>
                                    ''', unsafe_allow_html=True)

                                    def highlight_significant(row):
                                        return ['background-color: rgba(144, 238, 144, 0.3)' if row['buy_to_sell_ratio'] > finra_threshold else ''] * len(row)

                                    display_df = results_df.copy()
                                    for col in ['total_volume', 'bought_volume', 'sold_volume']:
                                        display_df[col] = display_df[col].astype(int)

                                    styled_df = display_df.style.apply(highlight_significant, axis=1)
                                    st.dataframe(styled_df, use_container_width=True)
                                else:
                                    st.warning(f"No FINRA short sale data available for {ticker}. This could be because the symbol is not found in FINRA's database.")
                        else:
                            st.info("Enter a ticker symbol to view FINRA short sale analysis.")
            
            # Gamma Exposure subtab
            with insight_tabs[3]:
                st.markdown('<h3 class="section-header">Gamma Exposure Analysis</h3>', unsafe_allow_html=True)
                
                if 'gex_data' in locals() and ticker and isinstance(gex_data, dict) and "error" not in gex_data:
                    seasonality_cols = st.columns([1, 3])    
                    st.markdown("#### Analysis Parameters", unsafe_allow_html=True)
                    param_cols = st.columns([2, 1, 1])
                    with param_cols[0]:
                        if gex_data["exp_data"] is not None and not gex_data["exp_data"].empty:
                            selected_exp = st.selectbox(
                                "Select Expiration Date",
                                gex_data["exp_data"]["date"].tolist(),
                                key="gex_exp_select"
                            )
                        else:
                            selected_exp = None
                            st.warning("No expiration dates available")
                    with param_cols[1]:
                        price_range = st.slider("Price Range (%)", 1, 50, 15, key="gex_price_range")
                    with param_cols[2]:
                        gex_threshold = st.slider("GEX Threshold", 0.1, 50.0, 5.0, key="gex_threshold")
                    analyze_gex = st.button("Update GEX Analysis", key="gex_analyze")
                    if analyze_gex and selected_exp:
                        with st.spinner("Calculating GEX..."):
                            gex_data = get_gex(
                                ticker,
                                expiration=selected_exp,
                                price_range_pct=price_range,
                                threshold=gex_threshold
                            )
                    if gex_data["gex_data"] is not None and not gex_data["gex_data"].empty:
                        gex_cols = st.columns([1, 2])
                        with gex_cols[0]:
                            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                            total_gex = gex_data["total_gex"]
                            market_bias = "Positive" if total_gex > 0 else "Negative"
                            gex_strength = "Strong" if abs(total_gex) > 20 else "Moderate" if abs(total_gex) > 10 else "Weak"
                            st.markdown(f'''
                                <p style="font-weight:600; margin-bottom:10px;">GEX Summary</p>
                                <div class="metric-label">Current Price</div>
                                <div class="metric-value">${gex_data["current_price"]:.2f}</div>
                                <div class="metric-label" style="margin-top:10px;">Total GEX</div>
                                <div class="metric-value">{total_gex:.2f}</div>
                                <div class="metric-label" style="margin-top:10px;">Market Bias</div>
                                <div class="metric-value {'metric-bullish' if total_gex > 0 else 'metric-bearish'}">{market_bias}</div>
                                <div class="metric-label" style="margin-top:10px;">GEX Strength</div>
                                <div class="metric-value">{gex_strength}</div>
                            ''', unsafe_allow_html=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                        with gex_cols[1]:
                            if gex_data["chart"]:
                                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                                st.plotly_chart(gex_data["chart"], use_container_width=True)
                                st.markdown('</div>', unsafe_allow_html=True)
                        st.markdown('<h4 style="margin-top:20px;">Key Takeaways</h4>', unsafe_allow_html=True)
                        strongest_strike = gex_data["strongest_gex_strike"]
                        st.markdown(f'''
                            <div style="background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem;">
                                <strong>Market Structure:</strong>
                                - Overall GEX Bias: {market_bias} with {gex_strength.lower()} strength<br>
                                - Strongest GEX Level: ${strongest_strike:.2f}<br>
                                - Largest Positive GEX: {gex_data["max_positive_gex"]:.2f}<br>
                                - Largest Negative GEX: {gex_data["max_negative_gex"]:.2f}<br><br>
                                <strong>What This Means:</strong><br>
                                1. <strong>Price Magnetism:</strong> ${strongest_strike:.2f} is the strongest magnetic level<br>
                                2. <strong>Market Movement:</strong> Current bias suggests {'resistance to downward moves' if total_gex > 0 else 'resistance to upward moves'}<br>
                                3. <strong>Volatility:</strong> {'High GEX levels typically suppress volatility' if abs(total_gex) > 20 else 'Moderate GEX levels suggest normal volatility' if abs(total_gex) > 10 else 'Low GEX levels may allow larger price movements'}
                            </div>
                        ''', unsafe_allow_html=True)
                        with st.expander("View Raw GEX Data"):
                            st.dataframe(gex_data["gex_data"])
                    else:
                        st.warning("No significant GEX data found for the selected parameters")

                elif selection_method == "Pre-defined Lists" or selection_method == "Custom List":
                    st.markdown('<div class="info-box">', unsafe_allow_html=True)
                    st.markdown('Gex analysis is only available for single stock mode. Please switch to "Single Stock" mode to see gex data.', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                    st.markdown(f'No GEX data available: {gex_data["error"]}', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
            
            # Whale Positioning subtab
            with insight_tabs[4]:
                st.markdown('<h3 class="section-header">Whale Positioning Analysis</h3>', unsafe_allow_html=True)
                
                if 'whale_data' in locals() and ticker and isinstance(whale_data, dict) and "error" not in whale_data:
                    whale_cols = st.columns([1, 2])
                    
                    with whale_cols[0]:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.markdown(f'''
                            <p style="font-weight:600; margin-bottom:10px;">Whale Activity Summary</p>
                            <div class="metric-label">Current Price</div>
                            <div class="metric-value">${whale_data["current_price"]:.2f}</div>
                            <div class="metric-label" style="margin-top:10px;">Last Hour Trend</div>
                            <div class="metric-value">{whale_data["price_trend"]:+.2f}%</div>
                            <div class="metric-label" style="margin-top:10px;">Predicted Direction</div>
                            <div class="metric-value" style="background-color: {whale_data['color']}; color: white; padding: 5px; border-radius: 3px;">
                                {whale_data["direction"]}
                            </div>
                        ''', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with whale_cols[1]:
                        if whale_data["chart"]:
                            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                            st.plotly_chart(whale_data["chart"], use_container_width=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.markdown('<h4 style="margin-top:20px;">Key Insights</h4>', unsafe_allow_html=True)
                    total_call_oi = whale_data["strike_df"]['call_oi'].sum()
                    total_put_oi = whale_data["strike_df"]['put_oi'].sum()
                    total_call_vol = whale_data["strike_df"]['call_vol'].sum()
                    total_put_vol = whale_data["strike_df"]['put_vol'].sum()
                    total_call_wv = whale_data["strike_df"]['call_wv'].sum()
                    total_put_wv = whale_data["strike_df"]['put_wv'].sum()
                    st.markdown(f'''
                        <div style="background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem;">
                            - <strong>Analysis Range</strong>: Strikes between ${whale_data["current_price"]*0.9:.2f} and ${whale_data["current_price"]*1.1:.2f}<br>
                            - <strong>Open Interest</strong>: Calls: {total_call_oi:,.0f}, Puts: {total_put_oi:,.0f}, Ratio: {total_call_oi/total_put_oi if total_put_oi > 0 else 'N/A':.2f}<br>
                            - <strong>Today\'s Volume</strong>: Calls: {total_call_vol:,.0f}, Puts: {total_put_vol:,.0f}<br>
                            - <strong>Weighted Volume</strong>: Calls: {total_call_wv:,.0f}, Puts: {total_put_wv:,.0f}<br>
                            - <strong>Momentum</strong>: Price Trend: {whale_data["price_trend"]:+.2f}%, Volume Momentum: {total_call_wv - total_put_wv:+,.0f}<br>
                            - <strong>Target</strong>: ${whale_data["target_strike"]:.2f} by {whale_data["target_expiry"]}
                        </div>
                    ''', unsafe_allow_html=True)
                    
                    with st.expander("View Detailed Whale Data"):
                        display_df = whale_data["strike_df"][['call_oi', 'put_oi', 'total_oi', 'call_vol', 'put_vol', 'net_vol']]
                        st.dataframe(
                            display_df.style.format({
                                'call_oi': '{:,.0f}',
                                'put_oi': '{:,.0f}',
                                'total_oi': '{:,.0f}',
                                'call_vol': '{:,.0f}',
                                'put_vol': '{:,.0f}',
                                'net_vol': '{:+,.0f}'
                            }),
                            use_container_width=True
                        )
                elif selection_method == "Pre-defined Lists" or selection_method == "Custom List":
                    st.markdown('<div class="info-box">', unsafe_allow_html=True)
                    st.markdown('Whale positioning analysis is only available for single stock mode. Please switch to "Single Stock" mode to see whale positioning.', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                    st.markdown(f'No whale positioning data available: {whale_data["error"]}', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
    
                #else:
                    # Show welcome message when no ticker is entered
                    st.markdown('''
                <div style="background-color: white; border-radius: 8px; padding: 2rem; box-shadow: 0 2px 5px rgba(0,0,0,0.05); text-align: center; margin: 2rem 0;">
                    <h2 style="color: #2c3e50; margin-bottom: 1.5rem;">Welcome to Stock Insights Hub</h2>
                    <p style="color: #7f8c8d; font-size: 1.1rem; margin-bottom: 2rem;">
                        Enter a stock ticker symbol above to access comprehensive market analysis tools including:
                    </p>
                    <div style="display: flex; justify-content: space-around; flex-wrap: wrap; gap: 1rem; margin-bottom: 2rem;">
                        <div style="background-color: #f8f9fa; border-radius: 8px; padding: 1rem; width: 200px;">
                            <h3 style="color: #3498db; font-size: 1.1rem; margin-bottom: 0.5rem;">Technical Analysis</h3>
                            <p style="color: #7f8c8d; font-size: 0.9rem;">Price action, key indicators, and moving averages</p>
                        </div>
                        <div style="background-color: #f8f9fa; border-radius: 8px; padding: 1rem; width: 200px;">
                            <h3 style="color: #3498db; font-size: 1.1rem; margin-bottom: 0.5rem;">Trend Oscillator</h3>
                            <p style="color: #7f8c8d; font-size: 0.9rem;">Advanced trend analysis with custom timeframes</p>
                        </div>
                        <div style="background-color: #f8f9fa; border-radius: 8px; padding: 1rem; width: 200px;">
                            <h3 style="color: #3498db; font-size: 1.1rem; margin-bottom: 0.5rem;">Market Insights</h3>
                            <p style="color: #7f8c8d; font-size: 0.9rem;">Seasonality patterns and institutional block trades</p>
                        </div>
                    </div>
                    <p style="color: #7f8c8d; font-size: 0.9rem;">
                        Example tickers: AAPL, MSFT, AMZN, TSLA, NVDA, GOOGL
                    </p>
                </div>
            ''', unsafe_allow_html=True)

            # Flow Summary subtab
            with insight_tabs[5]:
                st.markdown('<h3 class="section-header">Flow Summary - Top 10 OTM Options Flows</h3>', unsafe_allow_html=True)

                if ticker:  # Only run if a ticker is entered
                    with st.spinner("Fetching options flow data..."):
                        flow_summary = get_top_otm_flows(ticker, FLOW_URLS)

                    if not flow_summary.empty:
                        flow_cols = st.columns([1, 2])

                        with flow_cols[0]:
                            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                            current_price = get_current_price(ticker)
                            total_value = flow_summary['Transaction Value'].sum()
                            call_count = len(flow_summary[flow_summary['Call/Put'] == 'C'])
                            put_count = len(flow_summary[flow_summary['Call/Put'] == 'P'])
                            sentiment = "Bullish" if call_count > put_count else "Bearish" if put_count > call_count else "Neutral"
                            sentiment_class = "metric-bullish" if sentiment == "Bullish" else "metric-bearish" if sentiment == "Bearish" else "metric-neutral"
                            price_display = f"${current_price:.2f}" if current_price is not None else "N/A"
                            st.markdown(f'''
                                <p style="font-weight:600; margin-bottom:10px;">Flow Overview</p>
                                <div class="metric-label">Current Price</div>
                                <div class="metric-value">{price_display}</div>
                                <div class="metric-label" style="margin-top:10px;">Total Transaction Value</div>
                                <div class="metric-value">${total_value:,.2f}</div>
                                <div class="metric-label" style="margin-top:10px;">Call/Put Split</div>
                                <div class="metric-value">{call_count} Calls / {put_count} Puts</div>
                                <div class="metric-label" style="margin-top:10px;">Sentiment</div>
                                <div class="metric-value {sentiment_class}">{sentiment}</div>
                            ''', unsafe_allow_html=True)
                            st.markdown('</div>', unsafe_allow_html=True)

                        with flow_cols[1]:
                            st.markdown('<div class="chart-container">', unsafe_allow_html=True)

                            # Format the DataFrame for display
                            display_df = flow_summary.copy()
                            display_df['Expiration'] = display_df['Expiration'].dt.strftime('%Y-%m-%d')
                            display_df['Transaction Value'] = display_df['Transaction Value'].apply(lambda x: f"${x:,.2f}")
                            display_df['Last Price'] = display_df['Last Price'].apply(lambda x: f"${x:.2f}")
                            display_df = display_df.rename(columns={
                                'Call/Put': 'Type',
                                'Strike Price': 'Strike',
                                'Last Price': 'Price',
                                'Transaction Value': 'Value'
                            })

                            # Style the DataFrame
                            def highlight_type(row):
                                color = '#90ee90' if row['Type'] == 'C' else '#ffcccb' if row['Type'] == 'P' else ''
                                return [f'background-color: {color}' if col == 'Type' else '' for col in row.index]

                            styled_df = display_df.style.apply(highlight_type, axis=1)
                            st.dataframe(styled_df, use_container_width=True, height=400)

                            # Download button
                            csv = display_df.to_csv(index=False)
                            st.download_button(
                                label="Download Top 10 OTM Flows as CSV",
                                data=csv,
                                file_name=f"{ticker}_top_10_otm_flows_{datetime.now().strftime('%Y%m%d')}.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                            st.markdown('</div>', unsafe_allow_html=True)

                        st.markdown('''
                            <div style="background-color: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin-top: 1rem;">
                                <strong>Flow Summary Notes:</strong><br>
                                - Showing top 10 out-of-the-money (OTM) options flows by transaction value.<br>
                                - OTM Calls: Strike > Current Price | OTM Puts: Strike < Current Price.<br>
                                - Data sourced from CBOE options market statistics.<br>
                                - Transaction Value = Volume × Last Price × 100.
                            </div>
                        ''', unsafe_allow_html=True)

                    elif selection_method == "Pre-defined Lists" or selection_method == "Custom List":
                        st.markdown('<div class="info-box">', unsafe_allow_html=True)
                        st.markdown('Flow Summary is only available for single stock mode. Please switch to "Single Stock" mode to see the top 10 OTM options flows.', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                        st.markdown(f'No OTM options flow data available for {ticker}. This may be due to unavailable data from CBOE or no qualifying OTM transactions.', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.info("Enter a ticker symbol to view the top 10 OTM options flows.")   
    # Enhanced sidebar
    with st.sidebar:
        st.markdown('''
            <div style="padding: 1rem 0; margin-bottom: 1rem; border-bottom: 1px solid #e0e0e0;">
                <h3 style="color: #2c3e50; font-size: 1.3rem; margin-bottom: 0.5rem;">Options</h3>
            </div>
        ''', unsafe_allow_html=True)
        
        #st.button("Refresh Data", help="Clear cache and reload all market data", use_container_width=True, key="refresh_data_button")
        
        st.markdown('''
            <div style="padding: 1rem 0; margin-top: 2rem; border-top: 1px solid #e0e0e0;">
                <h3 style="color: #2c3e50; font-size: 1.3rem; margin-bottom: 1rem;">About</h3>
                <p style="color: #7f8c8d; font-size: 0.9rem; margin-bottom: 0.5rem;">
                    Stock Insights Hub combines multiple technical analysis methods to provide traders with comprehensive market insights.
                </p>
            </div>
        ''', unsafe_allow_html=True)
        
        st.markdown(f'''
            <div style="padding: 1rem 0; margin-top: 2rem; font-size: 0.8rem; color: #95a5a6; border-top: 1px solid #e0e0e0;">
                <p style="margin-bottom: 0.5rem;">Developed by Learn2Trade</p>
                <p>Last updated: {datetime.now().strftime('%Y-%m-%d')}</p>
                <p>Version 2.0</p>
            </div>
        ''', unsafe_allow_html=True)

if __name__ == "__main__":
    run()
