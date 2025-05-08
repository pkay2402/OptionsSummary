import pandas as pd
import requests
import io
from datetime import datetime, timedelta
import plotly.graph_objects as go
import logging
import yfinance as yf
import streamlit as st
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
        'short_volume_ratio': round(short_volume_ratio, 4)  # This is the DIX
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
        df_results = df_results.sort_values('date', ascending=True)  # Ascending for plotting
    return df_results

# Fetch historical stock price data
def fetch_stock_price(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(start=start_date, end=end_date)
        if hist.empty:
            logger.warning(f"No price data available for {symbol}")
            return pd.DataFrame()
        hist.reset_index(inplace=True)
        # Convert Date to timezone-naive datetime to match FINRA data
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

    # Log dtypes for debugging
    logger.info(f"dix_df['date'] dtype: {dix_df['date'].dtype}")
    logger.info(f"price_df['date'] dtype: {price_df['date'].dtype}")

    # Merge DIX and price data on date
    merged_df = pd.merge(dix_df[['date', 'short_volume_ratio']], 
                         price_df[['date', 'price']], 
                         on='date', 
                         how='inner')
    
    if merged_df.empty:
        st.warning(f"No overlapping data to plot for {symbol}. Try increasing lookback days.")
        logger.warning(f"No overlapping data to plot for {symbol}")
        return None

    # Create dual-axis plot
    fig = go.Figure()

    # Add stock price (left y-axis)
    fig.add_trace(
        go.Scatter(
            x=merged_df['date'],
            y=merged_df['price'],
            name=f"{symbol} Price",
            line=dict(color='blue'),
            yaxis='y1'
        )
    )

    # Add DIX (right y-axis)
    fig.add_trace(
        go.Scatter(
            x=merged_df['date'],
            y=merged_df['short_volume_ratio'],
            name='DIX (Short Volume Ratio)',
            line=dict(color='orange'),
            yaxis='y2'
        )
    )

    # Add bullish/bearish thresholds
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

    # Update layout for dual axes
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

    st.subheader("Dark Index (DIX) Analysis")
    st.write("Analyze the Dark Pool Short Volume Ratio (DIX) alongside stock price for a given symbol.")

    # Input widgets
    col1, col2 = st.columns(2)
    with col1:
        symbol = st.text_input("Enter Stock Symbol", "SPY").strip().upper()
    with col2:
        lookback_days = st.slider("Lookback Days", min_value=5, max_value=60, value=20, step=5)

    # Analyze button
    if st.button("Analyze"):
        with st.spinner(f"Fetching data for {symbol}..."):
            # Analyze symbol for DIX
            logger.info(f"Analyzing {symbol} for DIX...")
            dix_df = analyze_symbol(symbol, lookback_days)
            
            # Fetch stock price data
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
            logger.info(f"Fetching price data for {symbol} from {start_date} to {end_date}...")
            price_df = fetch_stock_price(symbol, start_date, end_date)
            
            # Plot results
            logger.info(f"Generating plot for {symbol}...")
            fig = plot_stock_and_dix(symbol, dix_df, price_df)
            
            if fig:
                st.plotly_chart(fig, use_container_width=True)
                
                # Display data table
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
