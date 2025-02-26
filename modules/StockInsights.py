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

def fetch_stock_data(symbol, period="1d", interval="5m", spy_hist=None):
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
        
        rs_value, rs_status = calculate_relative_strength(hist, spy_hist) if spy_hist is not None else (0, "N/A")
        
        today_data = hist.iloc[-1]
        current_price = round(today_data["Close"], 2)
        vwap = round(hist['VWAP'].iloc[-1], 2)
        ema_21 = round(hist['EMA21'].iloc[-1], 2)
        daily_pivot = round((hist["High"].iloc[-1] + hist["Low"].iloc[-1] + current_price) / 3, 2)
        
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
        st.error(f"Error in fetch_stock_data for {symbol}: {e}")
        return pd.DataFrame(), pd.DataFrame()

def plot_candlestick(data, symbol):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'], name=symbol))
    fig.add_trace(go.Scatter(x=data.index, y=data['VWAP'], name='VWAP', line=dict(color='purple', width=2)))
    fig.add_trace(go.Scatter(x=data.index, y=data['EMA21'], name='21 EMA', line=dict(color='orange', width=2)))
    fig.update_layout(
        title=f'{symbol} Chart with Indicators',
        yaxis_title='Price',
        template='plotly_white',
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=600,
        xaxis_rangeslider_visible=False
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

@st.cache_data
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

@st.cache_data
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

@st.cache_data
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

@st.cache_data
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
@st.cache_data
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

@st.cache_data
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

# Placeholder Functions (Hidden in UI)
@st.cache_data
def get_gex(ticker):
    return {"gamma_exposure": "TBD"}

@st.cache_data
def get_whale_positions(ticker):
    return {"net_buy": "TBD"}

@st.cache_data
def get_finra_data(ticker):
    return {"short_volume": "TBD"}

# Main App 
def run():
    # Apply custom CSS for professional styling
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
    input_container = st.container()
    with input_container:
        col1, col2, col3 = st.columns([3, 2, 3])
        with col1:
            ticker = st.text_input("Enter Stock Ticker", "").upper()
        with col2:
            timeframe_setup = st.selectbox("Oscillator Timeframe", ['1H/1D', '1D/5D'], index=0)
        with col3:
            analyze_button = st.button("Analyze", use_container_width=True)

    # Display a clean divider
    st.markdown('<hr style="margin: 1rem 0; border: 0; border-top: 1px solid #e0e0e0;">', unsafe_allow_html=True)
    
    # Process if ticker is provided
    if ticker:
        # Initialize data containers
        stock_summary = pd.DataFrame()
        stock_hist = pd.DataFrame()
        oscillator_data = {}
        momentum_data = {}
        seasonality_data = {}
        block_trade_data = {}
        
        with st.spinner("Analyzing market data..."):
            try:
                # Fetch all required data (same as original)
                _, spy_hist = fetch_stock_data("SPY", period="1d", interval="5m")
                stock_summary, stock_hist = fetch_stock_data(ticker, period="1d", interval="5m", spy_hist=spy_hist)
                oscillator_data = get_oscillator(ticker, timeframe_setup)
                momentum_data = get_momentum(ticker)
                seasonality_data = get_seasonality(ticker)
                block_trade_data = get_block_trades(ticker)
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
            
            with overview_cols[1]:
                signal_1d = momentum_data.get('1D_signal', 'No Data')
                signal_class = "metric-bullish" if signal_1d == "Buy" else "metric-bearish" if signal_1d == "Sell" else "metric-neutral"
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-label">1D Signal</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value {signal_class}">{signal_1d}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with overview_cols[2]:
                signal_5d = momentum_data.get('5D_signal', 'No Data')
                signal_class = "metric-bullish" if signal_5d == "Buy" else "metric-bearish" if signal_5d == "Sell" else "metric-neutral"
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-label">5D Signal</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value {signal_class}">{signal_5d}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with overview_cols[3]:
                trend = "N/A"
                if isinstance(oscillator_data, dict) and "trend" in oscillator_data:
                    trend = oscillator_data["trend"]
                trend_class = "metric-bullish" if trend == "Bullish" else "metric-bearish" if trend == "Bearish" else "metric-neutral"
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-label">Trend</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value {trend_class}">{trend}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

        # Main analysis section
        main_tabs = st.tabs(["Technical Analysis", "Trend Oscillator", "Insights"])
        
        # Technical Analysis Tab
        with main_tabs[0]:
            st.markdown('<h3 class="section-header">Intraday Price Action & Indicators</h3>', unsafe_allow_html=True)
            
            if not stock_summary.empty:
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
                fig = plot_candlestick(stock_hist, ticker)
                # Enhance the chart styling
                fig.update_layout(
                    title=f'{ticker} Intraday Price Action',
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
            else:
                st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                st.markdown('No technical data available for this ticker. Please try a different symbol.', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Trend Oscillator Tab
        with main_tabs[1]:
            st.markdown(f'<h3 class="section-header">Trend Oscillator ({timeframe_setup})</h3>', unsafe_allow_html=True)
            
            if "error" not in oscillator_data:
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
            else:
                st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                st.markdown(f'No oscillator data available: {oscillator_data["error"]}', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Insights Tab (Seasonality & Block Trades)
        with main_tabs[2]:
            insight_tabs = st.tabs(["Seasonality", "Block Trades"])
            
            # Seasonality subtab
            with insight_tabs[0]:
                st.markdown('<h3 class="section-header">Seasonality Analysis</h3>', unsafe_allow_html=True)
                
                if "error" not in seasonality_data:
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
                else:
                    st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                    st.markdown(f'No seasonality data available: {seasonality_data["error"]}', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
            
            # Block Trades subtab
            with insight_tabs[1]:
                st.markdown('<h3 class="section-header">Institutional Block Trades</h3>', unsafe_allow_html=True)
                
                if "error" not in block_trade_data:
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
                else:
                    st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                    st.markdown(f'No block trade data available: {block_trade_data["error"]}', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
        
        # Footer section with summary and disclaimer
        st.markdown('<hr style="margin: 2rem 0; border: 0; border-top: 1px solid #e0e0e0;">', unsafe_allow_html=True)
        st.markdown(f'''
            <div style="background-color: white; border-radius: 8px; padding: 1.2rem; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
                <h3 style="color: #2c3e50; font-size: 1.3rem; margin-bottom: 1rem;">Summary for {ticker}</h3>
                <p style="color: #2c3e50; margin-bottom: 1rem;">
                    This analysis combines technical indicators, trend oscillators, and historical patterns to provide a comprehensive view of {ticker}'s market position.
                </p>
                <p style="font-size: 0.8rem; color: #7f8c8d; margin-top: 1rem;">
                    <em>Disclaimer: This tool provides analysis based on historical data and technical indicators. It is not financial advice. 
                    Always conduct your own research and consider consulting with a financial advisor before making investment decisions.</em>
                </p>
            </div>
        ''', unsafe_allow_html=True)
    
    else:
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

    # Enhanced sidebar
    with st.sidebar:
        st.markdown('''
            <div style="padding: 1rem 0; margin-bottom: 1rem; border-bottom: 1px solid #e0e0e0;">
                <h3 style="color: #2c3e50; font-size: 1.3rem; margin-bottom: 0.5rem;">Options</h3>
            </div>
        ''', unsafe_allow_html=True)
        
        st.button("Refresh Data", help="Clear cache and reload all market data", use_container_width=True)
        
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
