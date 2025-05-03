import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
import time

# --- Streamlit Configuration ---
st.set_page_config(
    page_title="Breadth Analysis Dashboard (Nasdaq 100 & S&P 500)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 36px;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 20px;
    }
    .subheader {
        font-size: 24px;
        color: #1E3A8A;
        margin-top: 20px;
    }
    .metric-box {
        background-color: #F0F2F6;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        margin: 10px 0;
    }
    .trend-box {
        background-color: #E6F3FA;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #1E3A8A;
        margin: 20px 0;
        font-size: 16px;
    }
    .stButton>button {
        background-color: #1E3A8A;
        color: white;
        border-radius: 5px;
    }
    .stDataFrame {
        border: 1px solid #E0E0E0;
        border-radius: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Define ticker lists
nasdaq_100_tickers = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "AVGO", "PEP", "COST",
    "CSCO", "ADBE", "TXN", "CMCSA", "NFLX", "QCOM", "AMD", "INTC", "HON", "AMAT",
    "INTU", "SBUX", "ISRG", "MDLZ", "BKNG", "ADI", "PYPL", "GILD", "REGN", "VRTX",
    "LRCX", "MRNA", "CSX", "ADP", "MU", "PANW", "KDP", "MAR", "CTAS", "KLAC",
    "ORLY", "MCHP", "SNPS", "FTNT", "CDNS", "AEP", "XEL", "IDXX"
]

sp_500_top_50_tickers = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "UNH",
    "V", "MA", "HD", "PG", "DIS", "ADBE", "NFLX", "CRM", "KO", "PEP",
    "WMT", "MCD", "CSCO", "INTC", "AMD", "QCOM", "ORCL", "T", "VZ", "NKE",
    "IBM", "BA", "UPS", "CAT", "GS", "MMM", "HON", "CVX", "XOM", "JNJ",
    "PFE", "MRK", "ABBV", "LLY", "BMY", "AMGN", "GILD", "MDT", "SYK", "ISRG"
]

# --- Sidebar for User Inputs ---
with st.sidebar:
    st.header("📊 Dashboard Settings")
    st.markdown("Configure the analysis parameters below:")
    
    # Polygon.io API Key Input
    try:
        POLYGON_API_KEY = st.secrets["polygon"]["api_key"]
    except (KeyError, FileNotFoundError):
        POLYGON_API_KEY = st.text_input("Enter Polygon.io API Key", type="password")
        if not POLYGON_API_KEY:
            st.warning("Please provide a Polygon.io API key to fetch data.")
    
    # Index selection
    index_choice = st.selectbox("Select Index", ["Nasdaq 100 (QQQ)", "S&P 500 (SPY)"])
    index_ticker = "QQQ" if index_choice == "Nasdaq 100 (QQQ)" else "SPY"
    index_name = "Nasdaq 100" if index_choice == "Nasdaq 100 (QQQ)" else "S&P 500"
    
    today = datetime.now().date()
    three_months_ago = today - timedelta(days=90)
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", three_months_ago)
    with col2:
        end_date = st.date_input("End Date", today)
    
    start_date = start_date.strftime("%Y-%m-%d")
    end_date = end_date.strftime("%Y-%m-%d")
    
    ma_windows = st.multiselect(
        "Moving Average Periods",
        options=[5, 20, 50],
        default=[5, 20, 50]
    )
    if not ma_windows:
        st.warning("Please select at least one moving average period.")
        st.stop()
    
    # Display stocks used for analysis
    st.markdown("---")
    st.markdown(f"### Stocks Used for {index_name}")
    
    if index_choice == "Nasdaq 100 (QQQ)":
        st.write(f"Using {len(nasdaq_100_tickers)} stocks from Nasdaq 100 index")
        if st.checkbox("Show all stocks"):
            st.code(", ".join(nasdaq_100_tickers))
    else:
        st.write(f"Using {len(sp_500_top_50_tickers)} stocks from S&P 500 index")
        if st.checkbox("Show all stocks"):
            st.code(", ".join(sp_500_top_50_tickers))

# --- Main Dashboard ---
st.markdown(f'<div class="main-title">{index_name} Breadth Analysis Dashboard</div>', unsafe_allow_html=True)

# --- Data Acquisition ---
st.markdown('<div class="subheader">1. Data Acquisition</div>', unsafe_allow_html=True)

# Select tickers based on index choice
tickers = nasdaq_100_tickers if index_ticker == "QQQ" else sp_500_top_50_tickers

@st.cache_data
def fetch_polygon_data(ticker, start, end, api_key):
    """
    Fetch stock data from Polygon.io for a given ticker and date range.
    Returns a pandas DataFrame or None if data is unavailable.
    """
    try:
        url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}"
        params = {
            "adjusted": "true",
            "sort": "asc",
            "apiKey": api_key
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if data.get("resultsCount", 0) == 0 or "results" not in data:
            return None
        
        # Convert Polygon.io data to DataFrame
        df = pd.DataFrame(data["results"])
        df["date"] = pd.to_datetime(df["t"], unit="ms")
        df.set_index("date", inplace=True)
        df = df.rename(columns={
            "o": "Open",
            "h": "High",
            "l": "Low",
            "c": "Close",
            "v": "Volume"
        })
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        return df
    except Exception as e:
        st.warning(f"Polygon.io error for {ticker}: {e}")
        return None

@st.cache_data
def fetch_yfinance_data(ticker, start, end):
    """
    Fetch stock data from yfinance as a fallback.
    Returns a pandas DataFrame or None if data is unavailable.
    """
    try:
        data = yf.download(ticker, start=start, end=end, progress=False)
        if data.empty:
            st.warning(f"No yfinance data returned for {ticker}")
            return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        if 'Close' not in data.columns:
            st.warning(f"Missing 'Close' column for {ticker}. Columns: {data.columns}")
            return None
        return data
    except Exception as e:
        st.warning(f"yfinance error for {ticker}: {e}")
        return None

@st.cache_data
def fetch_stock_data(ticker, start, end, api_key):
    """
    Fetch stock data, prioritizing Polygon.io and falling back to yfinance.
    """
    if api_key:
        data = fetch_polygon_data(ticker, start, end, api_key)
        if data is not None and not data.empty:
            return data
        st.warning(f"Falling back to yfinance for {ticker} due to Polygon.io data issue.")
    
    return fetch_yfinance_data(ticker, start, end)

# Fetch index data (QQQ or SPY)
with st.spinner(f"Fetching {index_ticker} data..."):
    index_data = fetch_stock_data(index_ticker, start_date, end_date, POLYGON_API_KEY)
    if index_data is None or index_data.empty:
        st.warning(f"Invalid or missing {index_ticker} data.")
        st.stop()

# Fetch stock data with progress bar
st.markdown(f"**Fetching {index_name} stock data...**")
progress_bar = st.progress(0)
status_text = st.empty()
stock_data = {}
for i, ticker in enumerate(tickers):
    status_text.text(f"Fetching data for {ticker}...")
    data = fetch_stock_data(ticker, start_date, end_date, POLYGON_API_KEY)
    if data is not None and not data.empty:
        stock_data[ticker] = data
    progress_bar.progress((i + 1) / len(tickers))
    # Respect Polygon.io rate limits (5 calls/min for free tier)
    if POLYGON_API_KEY and i % 5 == 0:
        time.sleep(12)  # Wait 12 seconds after every 5 calls
progress_bar.empty()
status_text.empty()

if not stock_data:
    st.warning(f"No stock data fetched for the {index_name}.")
    st.stop()

# --- Summary Metrics ---
st.markdown('<div class="subheader">Summary Metrics</div>', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
col1.markdown(
    f'<div class="metric-box"><b>Stocks Analyzed</b><br>{len(stock_data)}</div>',
    unsafe_allow_html=True
)
col2.markdown(
    f'<div class="metric-box"><b>Date Range</b><br>{start_date} to {end_date}</div>',
    unsafe_allow_html=True
)
col3.markdown(
    f'<div class="metric-box"><b>Moving Averages</b><br>{", ".join(map(str, ma_windows))}</div>',
    unsafe_allow_html=True
)

# --- Technical Indicators ---
st.markdown('<div class="subheader">2. Technical Indicator Calculation</div>', unsafe_allow_html=True)

@st.cache_data
def calculate_moving_averages(df, windows):
    df = df.copy()
    if 'Close' not in df.columns or not isinstance(df['Close'], pd.Series):
        st.error(f"Invalid 'Close' column in DataFrame: {df.columns}")
        return df
    for window in windows:
        if len(df) >= window:
            try:
                close_series = df['Close'].squeeze()
                df[f'MA_{window}'] = ta.trend.sma_indicator(close_series, window=window, fillna=False)
            except Exception as e:
                st.warning(f"Error calculating MA_{window} for DataFrame: {e}")
                df[f'MA_{window}'] = float('nan')
        else:
            df[f'MA_{window}'] = float('nan')
    return df

with st.spinner("Calculating moving averages..."):
    processed_stock_data = {}
    for ticker, df in stock_data.items():
        processed_stock_data[ticker] = calculate_moving_averages(df, ma_windows)
    index_data = calculate_moving_averages(index_data, ma_windows)

# --- Breadth Analysis ---
st.markdown('<div class="subheader">3. Breadth Analysis</div>', unsafe_allow_html=True)

@st.cache_data
def calculate_breadth(data_dict, ma_windows):
    valid_dfs = [df for df in data_dict.values() if df is not None and not df.empty]
    if not valid_dfs:
        return pd.DataFrame()

    common_dates = sorted(set.intersection(*[set(df.index) for df in valid_dfs]))
    if not common_dates:
        return pd.DataFrame()

    breadth_df = pd.DataFrame(index=common_dates)
    for ma in ma_windows:
        above_ma = pd.Series(0, index=common_dates, dtype=int)
        for ticker, df in data_dict.items():
            if df is not None and not df.empty:
                df_filtered = df[df.index.isin(common_dates) &
                                df['Close'].notna() &
                                df[f'MA_{ma}'].notna()]
                if not df_filtered.empty:
                    valid = df_filtered['Close'] > df_filtered[f'MA_{ma}']
                    above_ma += valid.groupby(df_filtered.index).sum().reindex(common_dates, fill_value=0).astype(int)
        breadth_df[f'Above_MA_{ma}'] = above_ma
        breadth_df[f'Below_MA_{ma}'] = len(data_dict) - above_ma
    return breadth_df

with st.spinner("Performing breadth analysis..."):
    breadth_df = calculate_breadth(processed_stock_data, ma_windows)

# --- Trend Analysis ---
st.markdown(f'<div class="subheader">Current Trend Analysis for {index_ticker}</div>', unsafe_allow_html=True)

if not breadth_df.empty:
    trend_insights = []
    recent_breadth = breadth_df.tail(5)
    for ma in ma_windows:
        avg_above_ma = recent_breadth[f'Above_MA_{ma}'].mean() / len(stock_data) * 100
        if avg_above_ma > 70:
            trend_insights.append(f"**{ma}-day MA Breadth**: Bullish ({avg_above_ma:.1f}% of stocks above MA)")
        elif avg_above_ma < 30:
            trend_insights.append(f"**{ma}-day MA Breadth**: Bearish ({avg_above_ma:.1f}% of stocks above MA)")
        else:
            trend_insights.append(f"**{ma}-day MA Breadth**: Neutral ({avg_above_ma:.1f}% of stocks above MA)")
    
    index_aligned = index_data.join(breadth_df, how='inner')
    latest_data = index_aligned.tail(1)
    for ma in ma_windows:
        if latest_data['Close'].iloc[0] > latest_data[f'MA_{ma}'].iloc[0]:
            trend_insights.append(f"**Price vs. {ma}-day MA**: Bullish (Price above MA)")
        else:
            trend_insights.append(f"**Price vs. {ma}-day MA**: Bearish (Price below MA)")
    
    recent_prices = index_aligned['Close'].tail(5)
    if len(recent_prices) >= 5:
        price_change = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0] * 100
        if price_change > 2:
            trend_insights.append(f"**Price Momentum**: Bullish ({price_change:.1f}% increase over 5 days)")
        elif price_change < -2:
            trend_insights.append(f"**Price Momentum**: Bearish ({price_change:.1f}% decrease over 5 days)")
        else:
            trend_insights.append(f"**Price Momentum**: Neutral ({price_change:.1f}% change over 5 days)")
    
    bullish_signals = sum(1 for insight in trend_insights if "Bullish" in insight)
    bearish_signals = sum(1 for insight in trend_insights if "Bearish" in insight)
    
    if bullish_signals >= bearish_signals + 2:
        overall_trend = "Bullish"
        trend_color = "#2ECC71"
    elif bearish_signals >= bullish_signals + 2:
        overall_trend = "Bearish"
        trend_color = "#EF4444"
    else:
        overall_trend = "Neutral"
        trend_color = "#F39C12"
    
    trend_summary = f"<div class='trend-box' style='border-color: {trend_color};'>"
    trend_summary += f"<b>Overall Trend for {index_ticker}: {overall_trend}</b><br><br>"
    trend_summary += "<br>".join(trend_insights)
    trend_summary += "</div>"
    
    st.markdown(trend_summary, unsafe_allow_html=True)
else:
    st.warning("Cannot perform trend analysis without breadth data.")

if not breadth_df.empty:
    st.markdown(f"**Number of {index_name} Stocks Above/Below Moving Averages**")
    st.dataframe(
        breadth_df.style.format(precision=0).background_gradient(cmap="Blues", axis=0),
        height=300
    )
    
    csv = breadth_df.to_csv(index=True)
    st.download_button(
        label=f"Download {index_name} Breadth Data as CSV",
        data=csv,
        file_name=f"{index_name.lower().replace(' ', '_')}_breadth_analysis.csv",
        mime="text/csv"
    )

    # --- Breadth Charts with Index Performance ---
    st.markdown(f'<div class="subheader">4. Breadth Charts with {index_ticker} Performance</div>', unsafe_allow_html=True)
    
    index_aligned = index_data.join(breadth_df, how='inner')
    
    for ma in ma_windows:
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(
                x=breadth_df.index,
                y=breadth_df[f'Above_MA_{ma}'],
                name=f'Above MA {ma}',
                line=dict(color='#1E3A8A', width=2),
                hovertemplate='Date: %{x}<br>Stocks Above: %{y}<extra></extra>'
            )
        )
        fig.add_trace(
            go.Scatter(
                x=breadth_df.index,
                y=breadth_df[f'Below_MA_{ma}'],
                name=f'Below MA {ma}',
                line=dict(color='#EF4444', width=2),
                hovertemplate='Date: %{x}<br>Stocks Below: %{y}<extra></extra>'
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=index_aligned.index,
                y=index_aligned['Close'],
                name=f'{index_ticker} Close',
                line=dict(color='#2ECC71', width=2, dash='solid'),
                yaxis="y2",
                hovertemplate='Date: %{x}<br>' + index_ticker + ' Close: %{y:.2f}<extra></extra>'
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=index_aligned.index,
                y=index_aligned[f'MA_{ma}'],
                name=f'{index_ticker} MA {ma}',
                line=dict(color='#F39C12', width=2, dash='dash'),
                yaxis="y2",
                hovertemplate='Date: %{x}<br>' + f'{index_ticker} MA {ma}' + ': %{y:.2f}<extra></extra>'
            )
        )
        
        fig.update_layout(
            title=f"{index_name} Breadth Analysis vs. {index_ticker} Performance ({ma}-day MA)",
            xaxis_title="Date",
            yaxis_title="Number of Stocks",
            yaxis2=dict(
                title=f"{index_ticker} Price",
                overlaying="y",
                side="right",
                tickformat=".2f"
            ),
            template="plotly_white",
            font=dict(family="Arial", size=12, color="#333"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=50, r=50, t=80, b=50),
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)

else:
    st.warning(f"Breadth data is empty, cannot perform {index_ticker} performance analysis.")

# --- Footer ---
st.markdown("---")
st.markdown(
    '<div style="text-align: center; color: #666;">'
    'Breadth Analysis Dashboard | Powered by Streamlit, Polygon.io & yfinance | Data as of May 2025'
    '</div>',
    unsafe_allow_html=True
)
