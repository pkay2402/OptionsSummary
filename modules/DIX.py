import pandas as pd
import requests
import io
from datetime import datetime, timedelta
import plotly.graph_objects as go
import logging
import yfinance as yf
import streamlit as st
from typing import Optional, List, Dict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Sample list of top 100 stocks (subset of S&P 100 tickers for demonstration)
# In practice, replace with a full list or fetch dynamically
TOP_100_STOCKS = [
    'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'CRWD', 'JPM', 'JNJ', 'V',
    'PG', 'NVDA', 'HD', 'DIS', 'MA', 'PYPL', 'UNH', 'VZ', 'ADBE', 'NFLX',
    'CSCO', 'XOM', 'KO', 'NKE', 'MRK', 'PEP', 'QQQ', 'INTC', 'WMT', 'BA',
    'HOOD', 'COST', 'ABBV', 'CRM', 'AVGO', 'MCD', 'QCOM', 'TXN', 'NEE', 'HON',
    'PM', 'AMGN', 'ORCL', 'IBM', 'CVX', 'MDT', 'ACN', 'LLY', 'DHR', 'BMY',
    'UNP', 'LIN', 'LOW', 'UPS', 'MS', 'GS', 'RTX', 'CAT', 'BLK', 'SCHW',
    'SPGI', 'PLD', 'CB', 'TMO', 'AXP', 'NOW', 'BKNG', 'DE', 'ISRG', 'ADI',
    'ZTS', 'GILD', 'REGN', 'ADP', 'LMT', 'SYK', 'MO', 'EL', 'MMC', 'CL',
    'MSTR', 'CCI', 'CSX', 'NSC', 'SO', 'DUK', 'BDX', 'ITW', 'APD', 'PGR',
    'FISV', 'HUM', 'PLTR', 'WM', 'COIN', 'AON', 'ZS', 'SPG', 'NET', 'FDX','SPY','DIA','VXX',
    'IWM', 'XLF', 'XLI', 'XLB', 'XLC', 'XLY', 'XLC', 'XLI', 'XLB', 'XLY'
]

# FINRA data processing functions
def download_finra_short_sale_data(date: str) -> Optional[str]:
    url = f"https://cdn.finra.org/equity/regsho/daily/CNMSshvol{date}.txt"
    response = requests.get(url)
    return response.text if response.status_code == 200 else None

def process_finra_short_sale_data(data: Optional[str]) -> pd.DataFrame:
    if not data:
        return pd.DataFrame()
    df = pd.read_csv(io.StringIO(data), delimiter="|")
    return df[df["Symbol"].str.len() <= 4]

def calculate_metrics(row: pd.Series, total_volume: float) -> dict:
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
        'short_volume_ratio': round(short_volume_ratio, 4),  # This is the DIX
        'symbol': row['Symbol']
    }

# Single stock analysis with DIX calculation
def analyze_symbol(symbol: str, lookback_days: int = 20) -> pd.DataFrame:
    results = []
    for i in range(lookback_days):
        date = (datetime.now() - timedelta(days=i)).strftime("%Y%m%d")
        data = download_finra_short_sale_data(date)
        if data:
            df = process_finra_short_sale_data(data)
            symbol_data = df[df['Symbol'] == symbol]
            if not symbol_data.empty:
                row = symbol_data.iloc[0]
                total_volume = row.get('TotalVolume', 0)
                metrics = calculate_metrics(row, total_volume)
                metrics['date'] = date
                results.append(metrics)
    df_results = pd.DataFrame(results)
    if not df_results.empty:
        df_results['date'] = pd.to_datetime(df_results['date'], format='%Y%m%d')
        df_results = df_results.sort_values('date', ascending=True)
    return df_results

# New function to scan top 100 stocks for latest day and previous day
def scan_top_stocks(stocks: List[str], days: int = 2) -> Dict[str, pd.DataFrame]:
    """
    Scan the provided stocks for DIX data over the specified number of days.
    Returns a dictionary with 'latest' and 'previous' DataFrames.
    """
    results = []
    for i in range(days):
        date = (datetime.now() - timedelta(days=i)).strftime("%Y%m%d")
        data = download_finra_short_sale_data(date)
        if data:
            df = process_finra_short_sale_data(data)
            for symbol in stocks:
                symbol_data = df[df['Symbol'] == symbol]
                if not symbol_data.empty:
                    row = symbol_data.iloc[0]
                    total_volume = row.get('TotalVolume', 0)
                    metrics = calculate_metrics(row, total_volume)
                    metrics['date'] = date
                    results.append(metrics)
    df_results = pd.DataFrame(results)
    if df_results.empty:
        return {'latest': pd.DataFrame(), 'previous': pd.DataFrame()}
    
    df_results['date'] = pd.to_datetime(df_results['date'], format='%Y%m%d')
    # Split into latest and previous day
    latest_date = df_results['date'].max()
    prev_date = latest_date - timedelta(days=1)
    
    latest_df = df_results[df_results['date'] == latest_date][['symbol', 'short_volume_ratio', 'date']]
    prev_df = df_results[df_results['date'] == prev_date][['symbol', 'short_volume_ratio', 'date']]
    
    return {'latest': latest_df, 'previous': prev_df}

# Fetch historical stock price data
def fetch_stock_price(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(start=start_date, end=end_date)
        if hist.empty:
            logger.warning(f"No price data available for {symbol}")
            return pd.DataFrame()
        hist.reset_index(inplace=True)
        hist['Date'] = pd.to_datetime(hist['Date']).dt.tz_localize(None)
        return hist[['Date', 'Close']].rename(columns={'Date': 'date', 'Close': 'price'})
    except Exception as e:
        logger.error(f"Error fetching price data for {symbol}: {e}")
        return pd.DataFrame()

# Plot stock price and DIX
def plot_stock_and_dix(symbol: str, dix_df: pd.DataFrame, price_df: pd.DataFrame) -> Optional[go.Figure]:
    if dix_df.empty or price_df.empty:
        st.warning(f"No data to plot for {symbol}. Check symbol or data availability.")
        logger.warning(f"No data to plot for {symbol}")
        return None

    logger.info(f"dix_df['date'] dtype: {dix_df['date'].dtype}")
    logger.info(f"price_df['date'] dtype: {price_df['date'].dtype}")

    merged_df = pd.merge(dix_df[['date', 'short_volume_ratio']], 
                         price_df[['date', 'price']], 
                         on='date', 
                         how='inner')
    
    if merged_df.empty:
        st.warning(f"No overlapping data to plot for {symbol}. Try increasing lookback days.")
        logger.warning(f"No overlapping data to plot for {symbol}")
        return None

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=merged_df['date'],
            y=merged_df['price'],
            name=f"{symbol} Price",
            line=dict(color='blue'),
            yaxis='y1'
        )
    )
    fig.add_trace(
        go.Scatter(
            x=merged_df['date'],
            y=merged_df['short_volume_ratio'],
            name='DIX (Short Volume Ratio)',
            line=dict(color='orange'),
            yaxis='y2'
        )
    )
    fig.add_hline(
        y=0.45, 
        line_dash="dash", 
        line_color="green", 
        annotation_text="Bullish (DIX ≥ 0.45)", 
        annotation_position="top right",
        yref='y2'
    )
    fig.add_hline(
        y=0.40, 
        line_dash="dash", 
        line_color="red", 
        annotation_text="Bearish (DIX < 0.40)", 
        annotation_position="bottom right",
        yref='y2'
    )
    fig.update_layout(
        title=f"{symbol} Stock Price and DIX (Dark Pool Short Volume Ratio)",
        xaxis=dict(title='Date'),
        yaxis=dict(
            title=f"{symbol} Price (USD)",
            titlefont=dict(color='blue'),
            tickfont=dict(color='blue')
        ),
        yaxis2=dict(
            title='DIX (Short Volume / Total Volume)',
            titlefont=dict(color='orange'),
            tickfont=dict(color='orange'),
            overlaying='y',
            side='right',
            range=[0, 1]
        ),
        legend=dict(x=0.01, y=0.99),
        hovermode='x unified'
    )
    return fig

# Streamlit run function
def run():
    st.markdown("""
        <style>
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 5px;
            padding: 8px 16px;
        }
        .stTextInput, .stSlider {
            max-width: 300px;
        }
        </style>
    """, unsafe_allow_html=True)

    # New Section: Top 100 Stocks Analysis
    st.subheader("Top 100 Stocks DIX Analysis")
    st.write("Summary of DIX for top 100 stocks for the latest available day.")
    
    if st.button("Scan Top 100 Stocks"):
        with st.spinner("Scanning top 100 stocks..."):
            # Scan top stocks for latest and previous day
            scan_results = scan_top_stocks(TOP_100_STOCKS, days=2)
            latest_df = scan_results['latest']
            prev_df = scan_results['previous']
            
            if latest_df.empty:
                st.warning("No data available for the latest day.")
            else:
                # Summarize DIX > 0.45 and ≤ 0.45
                above_045 = latest_df[latest_df['short_volume_ratio'] > 0.45]
                below_or_equal_045 = latest_df[latest_df['short_volume_ratio'] <= 0.45]
                
                st.write(f"**Summary for {latest_df['date'].iloc[0].strftime('%Y-%m-%d')}:**")
                st.write(f"- Stocks with DIX > 0.45: **{len(above_045)}**")
                st.write(f"- Stocks with DIX ≤ 0.45: **{len(below_or_equal_045)}**")
                
                # Detect crossovers
                merged_df = pd.merge(
                    latest_df[['symbol', 'short_volume_ratio']],
                    prev_df[['symbol', 'short_volume_ratio']],
                    on='symbol',
                    how='inner',
                    suffixes=('_latest', '_prev')
                )
                
                # Stocks that crossed above 0.45
                crossed_above = merged_df[
                    (merged_df['short_volume_ratio_prev'] <= 0.45) &
                    (merged_df['short_volume_ratio_latest'] > 0.45)
                ]['symbol'].tolist()
                
                # Stocks that crossed below 0.45
                crossed_below = merged_df[
                    (merged_df['short_volume_ratio_prev'] > 0.45) &
                    (merged_df['short_volume_ratio_latest'] <= 0.45)
                ]['symbol'].tolist()
                
                # Display crossover results
                st.write("**Stocks that crossed DIX thresholds today:**")
                if crossed_above:
                    st.write(f"- Crossed above 0.45: {', '.join(crossed_above)}")
                else:
                    st.write("- Crossed above 0.45: None")
                if crossed_below:
                    st.write(f"- Crossed below 0.45: {', '.join(crossed_below)}")
                else:
                    st.write("- Crossed below 0.45: None")
                
                # Optionally display full latest data
                st.subheader("Latest DIX Data for Top Stocks")
                display_df = latest_df[['symbol', 'short_volume_ratio', 'date']].copy()
                display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
                st.dataframe(
                    display_df.style.format({
                        'short_volume_ratio': '{:.4f}'
                    }),
                    use_container_width=True
                )

    # Existing Single Stock Analysis Section
    st.subheader("Single Stock DIX Analysis")
    st.write("Analyze the Dark Pool Short Volume Ratio (DIX) alongside stock price for a given symbol.")

    col1, col2 = st.columns(2)
    with col1:
        symbol = st.text_input("Enter Stock Symbol", "SPY").strip().upper()
    with col2:
        lookback_days = st.slider("Lookback Days", min_value=5, max_value=60, value=20, step=5)

    if st.button("Analyze"):
        with st.spinner(f"Fetching data for {symbol}..."):
            logger.info(f"Analyzing {symbol} for DIX...")
            dix_df = analyze_symbol(symbol, lookback_days)
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
            logger.info(f"Fetching price data for {symbol} from {start_date} to {end_date}...")
            price_df = fetch_stock_price(symbol, start_date, end_date)
            logger.info(f"Generating plot for {symbol}...")
            fig = plot_stock_and_dix(symbol, dix_df, price_df)
            
            if fig:
                st.plotly_chart(fig, use_container_width=True)
                if not dix_df.empty:
                    st.subheader("DIX Data")
                    display_df = dix_df[['date', 'short_volume_ratio', 'total_volume', 'bought_volume', 'sold_volume']].copy()
                    display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
                    for col in ['total_volume', 'bought_volume', 'sold_volume']:
                        display_df[col] = display_df[col].astype(int)
                    st.dataframe(display_df.style.format({
                        'short_volume_ratio': '{:.4f}',
                        'total_volume': '{:,.0f}',
                        'bought_volume': '{:,.0f}',
                        'sold_volume': '{:,.0f}'
                    }), use_container_width=True)

if __name__ == "__main__":
    run()
