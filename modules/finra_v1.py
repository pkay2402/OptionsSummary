import pandas as pd
import streamlit as st
import requests
import io
from datetime import datetime, timedelta
import plotly.express as px
import logging
import sqlite3
import yfinance as yf
from typing import Optional, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Output to console
    ]
)
logger = logging.getLogger(__name__)

# Theme mapping for tabs
theme_mapping = {
    "Indexes": [
        "SPY", "QQQ", "IWM", "DIA", "SMH"
    ],
    "Bull Leverage ETF": [
        "SPXL", "UPRO", "TQQQ", "SOXL", "UDOW", "FAS", "SPUU", "TNA"
    ],
    "Bear Leverage ETF": [
        "SQQQ", "SPXS", "SOXS", "SDOW", "FAZ","SPDN", "TZA", "SPXU"
    ],
    "Volatility": [
        "VXX", "VIXY", "UVXY"
    ],
    "Bonds": [
        "TLT", "IEF", "SHY", "LQD", "HYG", "AGG"
    ],
    "Commodities": [
        "SPY", "GLD", "SLV", "USO", "UNG", "DBA", "DBB", "DBC"
    ],
    "Nuclear Power": [
        "CEG", "NNE", "GEV", "OKLO", "UUUU", "ASPI", "CCJ"
    ],
    "Crypto": [
        "IBIT", "FBTC", "MSTR", "COIN", "HOOD", "ETHU"
    ],
    "Metals": [
        "GLD", "SLV", "GDX", "GDXJ", "IAU", "SIVR"
    ],
    "Real Estate": [
        "VNQ", "IYR", "XHB", "XLF", "SPG", "PLD", "AMT", "DLR"
    ],
    "Consumer Discretionary": [
        "AMZN", "TSLA", "HD", "NKE", "MCD", "DIS", "LOW", "TGT", "LULU"
    ],
    "Consumer Staples": [
        "PG", "KO", "PEP", "WMT", "COST", "CL", "KMB", "MDLZ", "GIS"
    ],
    "Utilities": [
        "XLU", "DUK", "SO", "D", "NEE", "EXC", "AEP", "SRE", "ED"
    ],
    "Telecommunications": [
        "XLC", "T", "VZ", "TMUS", "S", "LUMN", "VOD"
    ],
    "Materials": [
        "XLB", "XME", "XLI", "FCX", "NUE", "DD", "APD", "LIN", "IFF"
    ],
    "Transportation": [
        "UPS", "FDX", "DAL", "UAL", "LUV", "CSX", "NSC", "KSU", "WAB"
    ],
    "Aerospace & Defense": [
        "LMT", "BA", "NOC", "RTX", "GD", "HII", "LHX", "COL", "TXT"
    ],
    "Retail": [
        "AMZN", "WMT", "TGT", "COST", "HD", "LOW", "TJX", "M", "KSS"
    ],
    "Automotive": [
        "TSLA", "F", "GM", "RIVN", "LCID", "NIO", "XPEV", "BYDDF", "FCAU"
    ],
    "Pharmaceuticals": [
        "PFE", "MRK", "JNJ", "ABBV", "BMY", "GILD", "AMGN", "LLY", "VRTX"
    ],
    "Biotechnology": [
        "AMGN", "REGN", "ILMN", "VRTX", "CRSP", "MRNA", "BMRN", "ALNY",
        "SRPT", "EDIT", "NTLA", "BEAM", "BLUE", "FATE", "SANA"
    ],
    "Insurance": [
        "AIG", "PRU", "MET", "UNM", "LNC", "TRV", "CINF", "PGR", "ALL"
    ],
    "Technology": [
        "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD",
        "ORCL", "CRM", "ADBE", "INTC", "CSCO", "QCOM", "TXN", "IBM",
        "NOW", "AVGO", "INTU", "PANW", "SNOW"
    ],
    "Financials": [
        "JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "C", "AXP", "SCHW",
        "COF", "MET", "AIG", "BK", "BLK", "TFC", "USB", "PNC", "CME", "SPGI"
    ],
    "Healthcare": [
        "LLY", "UNH", "JNJ", "PFE", "MRK", "ABBV", "TMO", "AMGN", "GILD",
        "CVS", "MDT", "BMY", "ABT", "DHR", "ISRG", "SYK", "REGN", "VRTX",
        "CI", "ZTS"
    ],
    "Consumer": [
        "WMT", "PG", "KO", "PEP", "COST", "MCD", "DIS", "NKE", "SBUX",
        "LOW", "TGT", "HD", "CL", "MO", "KHC", "PM", "TJX", "DG", "DLTR", "YUM"
    ],
    "Energy": [
        "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "OXY", "VLO",
        "XLE", "HES", "WMB", "KMI", "OKE", "HAL", "BKR", "FANG", "DVN",
        "TRGP", "APA"
    ],
    "Industrials": [
        "CAT", "DE", "UPS", "FDX", "BA", "HON", "UNP", "MMM", "GE", "LMT",
        "RTX", "GD", "CSX", "NSC", "WM", "ETN", "ITW", "EMR", "PH", "ROK"
    ],
    "Semiconductors": [
        "NVDA", "AMD", "QCOM", "TXN", "INTC", "AVGO", "ASML", "KLAC",
        "LRCX", "AMAT", "ADI", "MCHP", "ON", "STM", "MPWR", "TER", "ENTG",
        "SWKS", "QRVO", "LSCC"
    ],
    "Cybersecurity": [
        "CRWD", "PANW", "ZS", "FTNT", "S", "OKTA", "CYBR", "RPD", "NET",
        "QLYS", "TENB", "VRNS", "SPLK", "CHKP", "FEYE", "DDOG", "ESTC",
        "FSLY", "MIME", "KNBE"
    ],
    "Quantum Computing": [
        "IBM", "GOOGL", "MSFT", "RGTI", "IONQ", "QUBT", "HON", "QCOM",
        "INTC", "AMAT", "MKSI", "NTNX", "XERI", "QTUM", "FORM",
        "LMT", "BA", "NOC", "ACN"
    ],
    "Clean Energy": [
        "TSLA", "ENPH", "FSLR", "NEE", "PLUG", "SEDG", "RUN", "SHLS",
        "ARRY", "NOVA", "BE", "BLDP", "FCEL", "CWEN", "DTE", "AES",
        "EIX", "SRE"
    ],
    "Artificial Intelligence": [
        "NVDA", "GOOGL", "MSFT", "AMD", "PLTR", "SNOW", "AI", "CRM", "IBM",
        "AAPL", "ADBE", "MSCI", "DELL", "BIDU", "UPST", "AI", "PATH",
        "SOUN", "VRNT", "ANSS"
    ],
    "Biotechnology": [
        "MRNA", "CRSP", "VRTX", "REGN", "ILMN", "AMGN", "NBIX", "BIIB",
        "INCY", "GILD", "BMRN", "ALNY", "SRPT", "BEAM", "NTLA", "EDIT",
        "BLUE", "SANA", "VKTX", "KRYS"
    ]
}

# Database functions
def setup_stock_database() -> None:
    conn = sqlite3.connect('stock_data.db')
    cursor = conn.cursor()
    cursor.execute('DROP TABLE IF EXISTS stocks')
    logger.info("Dropped existing `stocks` table (if it existed).")
    cursor.execute('''
        CREATE TABLE stocks (
            symbol TEXT PRIMARY KEY,
            price REAL,
            market_cap REAL,
            last_updated TEXT
        )
    ''')
    logger.info("Created `stocks` table with correct schema.")
    conn.commit()
    conn.close()

def check_and_setup_database() -> None:
    try:
        conn = sqlite3.connect('stock_data.db')
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='stocks'")
        if not cursor.fetchone():
            conn.close()
            setup_stock_database()
            return
        cursor.execute("PRAGMA table_info(stocks)")
        columns = [info[1] for info in cursor.fetchall()]
        if not all(col in columns for col in ['symbol', 'price', 'market_cap', 'last_updated']):
            conn.close()
            setup_stock_database()
            return
        conn.close()
    except Exception as e:
        logger.error(f"Error checking database: {e}")
        setup_stock_database()

check_and_setup_database()

def update_stock_database(symbols: List[str]) -> None:
    conn = sqlite3.connect('stock_data.db')
    cursor = conn.cursor()
    try:
        tickers = yf.Tickers(' '.join(symbols))
        for symbol in symbols:
            try:
                ticker = tickers.tickers[symbol]
                info = ticker.info
                price = info.get('regularMarketPrice', 0)
                market_cap = info.get('marketCap', 0)
                cursor.execute('''
                    INSERT OR REPLACE INTO stocks (symbol, price, market_cap, last_updated)
                    VALUES (?, ?, ?, ?)
                ''', (symbol, price, market_cap, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {e}")
                cursor.execute('''
                    INSERT OR REPLACE INTO stocks (symbol, price, market_cap, last_updated)
                    VALUES (?, ?, ?, ?)
                ''', (symbol, 0, 0, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    finally:
        conn.commit()
        conn.close()

def get_stock_info_from_db(symbol: str) -> dict:
    conn = sqlite3.connect('stock_data.db')
    cursor = conn.cursor()
    cursor.execute('SELECT price, market_cap FROM stocks WHERE symbol = ?', (symbol,))
    result = cursor.fetchone()
    conn.close()
    return {'price': result[0], 'market_cap': result[1]} if result else {'price': 0, 'market_cap': 0}

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
        'short_volume_ratio': round(short_volume_ratio, 4)
    }

# Single stock analysis
@st.cache_data(ttl=11520)
def analyze_symbol(symbol: str, lookback_days: int = 20, threshold: float = 1.5) -> tuple[pd.DataFrame, int]:
    results = []
    significant_days = 0
    cumulative_bought = 0
    cumulative_sold = 0
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
                cumulative_bought += metrics['bought_volume']
                cumulative_sold += metrics['sold_volume']
                if metrics['buy_to_sell_ratio'] > threshold:
                    significant_days += 1
                metrics['date'] = date
                metrics['cumulative_bought'] = cumulative_bought
                metrics['cumulative_sold'] = cumulative_sold
                results.append(metrics)
    df_results = pd.DataFrame(results)
    if not df_results.empty:
        df_results['date'] = pd.to_datetime(df_results['date'], format='%Y%m%d')
        df_results = df_results.sort_values('date', ascending=False)
    return df_results, significant_days

# Latest data fetch
def get_latest_data(symbols: List[str] = None) -> tuple[pd.DataFrame, Optional[str]]:
    for i in range(7):  # Check the last 7 days
        date = (datetime.now() - timedelta(days=i)).strftime("%Y%m%d")
        data = download_finra_short_sale_data(date)
        if data:
            df = process_finra_short_sale_data(data)
            if not df.empty:
                if symbols:
                    df = df[df['Symbol'].isin(symbols)]
                return df, date
    return pd.DataFrame(), None

# Stock summary generation
def generate_stock_summary() -> tuple[pd.DataFrame, pd.DataFrame, Optional[str]]:
    symbol_themes = {}
    for theme, symbols in theme_mapping.items():
        for symbol in symbols:
            if symbol not in symbol_themes:
                symbol_themes[symbol] = theme
    all_symbols = list(symbol_themes.keys())
    latest_df, latest_date = get_latest_data(all_symbols)
    if latest_df.empty or latest_date is None:
        return pd.DataFrame(), pd.DataFrame(), None
    metrics_list = []
    for _, row in latest_df.iterrows():
        total_volume = row.get('TotalVolume', 0)
        metrics = calculate_metrics(row, total_volume)
        metrics['Symbol'] = row['Symbol']
        metrics_list.append(metrics)
    df = pd.DataFrame(metrics_list)
    df = df.drop_duplicates(subset=['Symbol'], keep='first')
    df['Theme'] = df['Symbol'].map(symbol_themes)
    high_buy_df = df[df['bought_volume'] > 2 * df['sold_volume']].copy()
    high_sell_df = df[df['sold_volume'] > 2 * df['bought_volume']].copy()
    columns = ['Symbol', 'Theme', 'buy_to_sell_ratio', 'bought_volume', 'sold_volume', 'total_volume']
    if not high_buy_df.empty:
        high_buy_df = high_buy_df[columns].sort_values(by='buy_to_sell_ratio', ascending=False)
        for col in ['bought_volume', 'sold_volume', 'total_volume']:
            high_buy_df[col] = high_buy_df[col].astype(int)
        high_buy_df['buy_to_sell_ratio'] = high_buy_df['buy_to_sell_ratio'].round(2)
    if not high_sell_df.empty:
        high_sell_df = high_sell_df[columns].sort_values(by='buy_to_sell_ratio', ascending=True)
        for col in ['bought_volume', 'sold_volume', 'total_volume']:
            high_sell_df[col] = high_sell_df[col].astype(int)
        high_sell_df['buy_to_sell_ratio'] = high_sell_df['buy_to_sell_ratio'].round(2)
    return high_buy_df, high_sell_df, latest_date

def get_signal(ratio):
    if ratio > 1.5:
        return 'Buy'
    elif ratio > 1.0:
        return 'Add'
    else:
        return 'Trim'

def style_signal(val):
    color = 'green' if val in ['Buy', 'Add'] else 'red'
    return f'color: {color}; font-weight: bold'

def run():
    st.markdown("""
        <style>
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 5px;
            padding: 8px 16px;
        }
        .stTabs [data-baseweb="tab"] {
            font-size: 16px;
            padding: 10px;
        }
        .stSelectbox {
            max-width: 300px;
        }
        .stDataFrame {
            font-size: 14px;
        }
        </style>
        """, unsafe_allow_html=True)
    
    st.title("FINRA Short Sale Analysis")
    
    # Create tabs
    tabs = st.tabs(["Single Stock", "Stock Summary", "Watchlist Summary"])
    
    # Single Stock Tab
    with tabs[0]:
        st.subheader("Single Stock Analysis")
        col1, col2 = st.columns(2)
        with col1:
            symbol = st.text_input("Enter Symbol", "NVDA").strip().upper()
            lookback_days = st.slider("Lookback Days", 1, 30, 20)
        with col2:
            threshold = st.number_input("Buy/Sell Ratio Threshold", min_value=1.0, max_value=5.0, value=1.5, step=0.1)
        if st.button("Analyze Stock"):
            with st.spinner(f"Analyzing {symbol}..."):
                results_df, significant_days = analyze_symbol(symbol, lookback_days, threshold)
                if not results_df.empty:
                    avg_buy = results_df['bought_volume'].mean()
                    avg_sell = results_df['sold_volume'].mean()
                    avg_total = results_df['total_volume'].mean()
                    total_buy = results_df['bought_volume'].sum()
                    total_sell = results_df['sold_volume'].sum()
                    total_volume_sum = results_df['total_volume'].sum()
                    aggregate_ratio = total_buy / total_sell if total_sell > 0 else float('inf')
                    
                    st.subheader("Summary Metrics")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Total Bought", f"{total_buy:,.0f}")
                    col2.metric("Total Sold", f"{total_sell:,.0f}")
                    col3.metric("Avg Buy Volume", f"{avg_buy:,.0f}")
                    col4.metric("Avg Sell Volume", f"{avg_sell:,.0f}")
                    col1.metric("Total Volume", f"{total_volume_sum:,.0f}")
                    col2.metric("Aggregate Buy Ratio", f"{aggregate_ratio:.2f}")
                    
                    display_df = results_df.copy()
                    display_df['Short %'] = (display_df['bought_volume'] / display_df['total_volume'] * 100).round(0).astype(int).apply(lambda x: f"{x}%")
                    display_df['BOT %'] = display_df['Short %']
                    display_df['Signal'] = display_df['buy_to_sell_ratio'].apply(get_signal)
                    for col in ['bought_volume', 'sold_volume', 'total_volume']:
                        display_df[col] = display_df[col].astype(int).apply(lambda x: f"{x:,.0f}")
                    display_df['buy_to_sell_ratio'] = display_df['buy_to_sell_ratio'].round(2)
                    display_df['date'] = display_df['date'].dt.strftime('%Y%m%d')
                    columns = ['date', 'bought_volume', 'sold_volume', 'Short %', 'buy_to_sell_ratio', 'BOT %', 'total_volume', 'Signal']
                    st.dataframe(display_df[columns].style.applymap(style_signal, subset=['Signal']))
                else:
                    st.write(f"No data available for {symbol}.")
    
    # Stock Summary Tab
    with tabs[1]:
        st.subheader("Stock Summary")
        if st.button("Generate Stock Summary"):
            with st.spinner("Analyzing stock volume data..."):
                high_buy_df, high_sell_df, latest_date = generate_stock_summary()
                if latest_date:
                    st.write(f"Data analyzed for: {latest_date}")
                st.write("### High Buy Stocks (Bought > 2x Sold)")
                if not high_buy_df.empty:
                    high_buy_df['Short %'] = (high_buy_df['bought_volume'] / high_buy_df['total_volume'] * 100).round(0).astype(int).apply(lambda x: f"{x}%")
                    high_buy_df['BOT %'] = high_buy_df['Short %']
                    high_buy_df['Signal'] = high_buy_df['buy_to_sell_ratio'].apply(get_signal)
                    for col in ['bought_volume', 'sold_volume', 'total_volume']:
                        high_buy_df[col] = high_buy_df[col].apply(lambda x: f"{x:,.0f}")
                    columns = ['Symbol', 'bought_volume', 'sold_volume', 'Short %', 'buy_to_sell_ratio', 'BOT %', 'total_volume', 'Signal']
                    st.dataframe(high_buy_df[columns].style.applymap(style_signal, subset=['Signal']))
                else:
                    st.write("No high buy stocks found.")
                st.write("### High Sell Stocks (Sold > 2x Bought)")
                if not high_sell_df.empty:
                    high_sell_df['Short %'] = (high_sell_df['bought_volume'] / high_sell_df['total_volume'] * 100).round(0).astype(int).apply(lambda x: f"{x}%")
                    high_sell_df['BOT %'] = high_sell_df['Short %']
                    high_sell_df['Signal'] = high_sell_df['buy_to_sell_ratio'].apply(get_signal)
                    for col in ['bought_volume', 'sold_volume', 'total_volume']:
                        high_sell_df[col] = high_sell_df[col].apply(lambda x: f"{x:,.0f}")
                    columns = ['Symbol', 'bought_volume', 'sold_volume', 'Short %', 'buy_to_sell_ratio', 'BOT %', 'total_volume', 'Signal']
                    st.dataframe(high_sell_df[columns].style.applymap(style_signal, subset=['Signal']))
                else:
                    st.write("No high sell stocks found.")
    
    # Watchlist Summary Tab
    with tabs[2]:
        st.subheader("Watchlist Summary")
        selected_theme = st.selectbox("Select Watchlist (Theme)", list(theme_mapping.keys()), index=0)
        if st.button("Generate Watchlist Summary"):
            with st.spinner(f"Analyzing {selected_theme}..."):
                symbols = theme_mapping[selected_theme]
                latest_df, latest_date = get_latest_data(symbols)
                if not latest_df.empty:
                    st.write(f"Data for: {latest_date}")
                    metrics_list = []
                    for _, row in latest_df.iterrows():
                        total_volume = row.get('TotalVolume', 0)
                        metrics = calculate_metrics(row, total_volume)
                        metrics['Symbol'] = row['Symbol']
                        metrics_list.append(metrics)
                    theme_df = pd.DataFrame(metrics_list)
                    theme_df = theme_df.sort_values(by=['buy_to_sell_ratio'], ascending=False)
                    total_buy = theme_df['bought_volume'].sum()
                    total_sell = theme_df['sold_volume'].sum()
                    total_volume_sum = theme_df['total_volume'].sum()
                    aggregate_ratio = total_buy / total_sell if total_sell > 0 else float('inf')
                    
                    st.subheader("Summary Metrics")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Bought", f"{total_buy:,.0f}")
                    col2.metric("Total Sold", f"{total_sell:,.0f}")
                    col3.metric("Aggregate Buy Ratio", f"{aggregate_ratio:.2f}")
                    
                    display_df = theme_df.copy()
                    display_df['Short %'] = (display_df['bought_volume'] / display_df['total_volume'] * 100).round(0).astype(int).apply(lambda x: f"{x}%")
                    display_df['BOT %'] = display_df['Short %']
                    display_df['Signal'] = display_df['buy_to_sell_ratio'].apply(get_signal)
                    for col in ['bought_volume', 'sold_volume', 'total_volume']:
                        display_df[col] = display_df[col].apply(lambda x: f"{x:,.0f}")
                    display_df['buy_to_sell_ratio'] = display_df['buy_to_sell_ratio'].round(2)
                    columns = ['Symbol', 'bought_volume', 'sold_volume', 'Short %', 'buy_to_sell_ratio', 'BOT %', 'total_volume', 'Signal']
                    st.dataframe(display_df[columns].style.applymap(style_signal, subset=['Signal']))
                else:
                    st.write(f"No data available for {selected_theme}.")

if __name__ == "__main__":
    run()
