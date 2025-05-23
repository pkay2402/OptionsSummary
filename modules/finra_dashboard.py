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

# Accumulation/Distribution functions
def validate_pattern(symbol: str, dates: List[str], pattern_type: str) -> bool:
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(start=min(dates), end=max(dates))
        if hist.empty:
            return False
        latest_price = hist['Close'].iloc[-1]
        avg_price = hist['Close'].mean()
        return latest_price > avg_price if pattern_type == "accumulation" else latest_price < avg_price
    except Exception:
        return False

def find_patterns(lookback_days=10, min_volume=1000000, pattern_type="accumulation",
                  use_price_validation=False, min_price=5.0, min_market_cap=500_000_000):
    pattern_data = {}
    for i in range(lookback_days):
        date = (datetime.now() - timedelta(days=i)).strftime("%Y%m%d")
        data = download_finra_short_sale_data(date)
        if data:
            df = process_finra_short_sale_data(data)
            df = df[df['TotalVolume'] > min_volume]
            for _, row in df.iterrows():
                symbol = row['Symbol']
                total_volume = row.get('TotalVolume', 0)
                metrics = calculate_metrics(row, total_volume)
                if symbol not in pattern_data:
                    pattern_data[symbol] = {'dates': [], 'volumes': [], 'ratios': [], 'total_volume': 0, 'days_pattern': 0}
                pattern_data[symbol]['dates'].append(date)
                pattern_data[symbol]['volumes'].append(total_volume)
                pattern_data[symbol]['ratios'].append(metrics['buy_to_sell_ratio'])
                pattern_data[symbol]['total_volume'] += total_volume
                if pattern_type == "accumulation" and metrics['buy_to_sell_ratio'] > 1.5:
                    pattern_data[symbol]['days_pattern'] += 1
                elif pattern_type == "distribution" and metrics['buy_to_sell_ratio'] < 0.7:
                    pattern_data[symbol]['days_pattern'] += 1
    results = []
    for symbol, data in pattern_data.items():
        if len(data['dates']) >= 3 and data['days_pattern'] >= 2:
            stock_info = get_stock_info_from_db(symbol)
            price = stock_info['price']
            market_cap = stock_info['market_cap']
            if price < min_price or market_cap < min_market_cap:
                continue
            if use_price_validation and not validate_pattern(symbol, data['dates'], pattern_type):
                continue
            avg_ratio = sum(data['ratios']) / len(data['ratios'])
            avg_volume = data['total_volume'] / len(data['dates'])
            results.append({
                'Symbol': symbol,
                'Price': round(price, 2),
                'Market Cap (M)': round(market_cap / 1_000_000, 2),
                'Avg Daily Volume': int(avg_volume),
                'Avg Buy/Sell Ratio': round(avg_ratio, 2),
                'Volume Trend': 'Increasing' if data['volumes'][0] > avg_volume else 'Decreasing',
                'Days Showing Pattern': data['days_pattern'],
                'Total Volume': int(data['total_volume']),
                'Latest Ratio': data['ratios'][0],
                'Pattern Type': pattern_type.capitalize()
            })
    if pattern_type == "accumulation":
        results = sorted(results, key=lambda x: (x['Days Showing Pattern'], x['Avg Buy/Sell Ratio'], x['Total Volume']), reverse=True)
    else:
        results = sorted(results, key=lambda x: (x['Days Showing Pattern'], -x['Avg Buy/Sell Ratio'], x['Total Volume']), reverse=True)
    return pd.DataFrame(results[:40])

# Recommendation generation
def generate_recommendations() -> tuple[pd.DataFrame, pd.DataFrame, Optional[str]]:
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
        metrics['TotalVolume'] = total_volume
        metrics_list.append(metrics)
    df = pd.DataFrame(metrics_list)
    df = df.drop_duplicates(subset=['Symbol'], keep='first')
    theme_factors = {}
    for theme, symbols in theme_mapping.items():
        theme_data = df[df['Symbol'].isin(symbols)]
        if not theme_data.empty:
            total_bought = theme_data['bought_volume'].sum()
            total_sold = theme_data['sold_volume'].sum()
            bullish_factor = min(total_bought / (total_sold + 1), 2.0)
            bearish_factor = min(total_sold / (total_bought + 1), 2.0)
            theme_factors[theme] = {'bullish': bullish_factor, 'bearish': bearish_factor}
        else:
            theme_factors[theme] = {'bullish': 1.0, 'bearish': 1.0}
    df['Theme'] = df['Symbol'].map(symbol_themes)
    df['buy_to_sell_ratio'] = df['buy_to_sell_ratio'].replace(float('inf'), 5.0)
    df['norm_ratio'] = df['buy_to_sell_ratio'].clip(upper=5.0)
    df['buy_volume_ratio'] = df['bought_volume'] / df['total_volume']
    df['sell_volume_ratio'] = df['sold_volume'] / df['total_volume']
    df['theme_bullish_factor'] = df['Theme'].map(lambda x: theme_factors.get(x, {'bullish': 1.0})['bullish'])
    df['theme_bearish_factor'] = df['Theme'].map(lambda x: theme_factors.get(x, {'bearish': 1.0})['bearish'])
    df['long_score'] = (
        (df['norm_ratio'] * 0.5) +
        (df['buy_volume_ratio'] * 0.3) +
        (df['theme_bullish_factor'] * 0.2)
    ) * (df['total_volume'] / df['total_volume'].max())
    df['short_score'] = (
        ((1 / df['norm_ratio'].replace(0, 0.1)) * 0.5) +
        (df['sell_volume_ratio'] * 0.3) +
        (df['theme_bearish_factor'] * 0.2)
    ) * (df['total_volume'] / df['total_volume'].max())
    df['preference_score'] = df['long_score'] - df['short_score']
    def select_diverse_candidates(df, score_column, top_n=5, max_per_theme=2, exclude_symbols=None):
        if exclude_symbols is None:
            exclude_symbols = set()
        df_filtered = df[~df['Symbol'].isin(exclude_symbols)]
        df_sorted = df_filtered.sort_values(by=score_column, ascending=False)
        selected = []
        theme_count = {}
        for _, row in df_sorted.iterrows():
            theme = row['Theme']
            if theme not in theme_count:
                theme_count[theme] = 0
            if theme_count[theme] < max_per_theme and len(selected) < top_n:
                selected.append(row)
                theme_count[theme] += 1
        return pd.DataFrame(selected)
    long_candidates = select_diverse_candidates(df, 'long_score', top_n=5, max_per_theme=2)
    long_symbols = set(long_candidates['Symbol'].values) if not long_candidates.empty else set()
    short_candidates = select_diverse_candidates(df, 'short_score', top_n=5, max_per_theme=2, exclude_symbols=long_symbols)
    columns = ['Symbol', 'Theme', 'buy_to_sell_ratio', 'bought_volume', 'sold_volume', 'total_volume', 'long_score']
    long_df = long_candidates[columns].copy() if not long_candidates.empty else pd.DataFrame(columns=columns)
    columns = ['Symbol', 'Theme', 'buy_to_sell_ratio', 'bought_volume', 'sold_volume', 'total_volume', 'short_score']
    short_df = short_candidates[columns].copy() if not short_candidates.empty else pd.DataFrame(columns=columns)
    if not long_df.empty:
        long_df['long_score'] = long_df['long_score'].round(4)
        for col in ['bought_volume', 'sold_volume', 'total_volume']:
            long_df[col] = long_df[col].astype(int)
    if not short_df.empty:
        short_df['short_score'] = short_df['short_score'].round(4)
        for col in ['bought_volume', 'sold_volume', 'total_volume']:
            short_df[col] = short_df[col].astype(int)
    return long_df, short_df, latest_date

# Theme summary generation
def generate_theme_summary() -> tuple[pd.DataFrame, pd.DataFrame, Optional[str]]:
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
        metrics['TotalVolume'] = total_volume
        metrics_list.append(metrics)
    df = pd.DataFrame(metrics_list)
    df = df.drop_duplicates(subset=['Symbol'], keep='first')
    df['Theme'] = df['Symbol'].map(symbol_themes)
    theme_summary = []
    for theme in theme_mapping.keys():
        theme_data = df[df['Theme'] == theme]
        if not theme_data.empty:
            total_bought = theme_data['bought_volume'].sum()
            total_sold = theme_data['sold_volume'].sum()
            total_volume = theme_data['total_volume'].sum()
            avg_buy_sell_ratio = theme_data['buy_to_sell_ratio'].mean()
            num_stocks = len(theme_data)
            bought_to_sold_ratio = total_bought / (total_sold + 1)
            sold_to_bought_ratio = total_sold / (total_bought + 1)
            theme_summary.append({
                'Theme': theme,
                'Total Bought Volume': int(total_bought),
                'Total Sold Volume': int(total_sold),
                'Bought-to-Sold Ratio': round(bought_to_sold_ratio, 2),
                'Sold-to-Bought Ratio': round(sold_to_bought_ratio, 2),
                'Average Buy/Sell Ratio': round(avg_buy_sell_ratio, 2),
                'Total Volume': int(total_volume),
                'Number of Stocks': num_stocks
            })
    summary_df = pd.DataFrame(theme_summary)
    bullish_df = summary_df[summary_df['Bought-to-Sold Ratio'] > 1].sort_values(by='Bought-to-Sold Ratio', ascending=False)
    bearish_df = summary_df[summary_df['Bought-to-Sold Ratio'] <= 1].sort_values(by='Sold-to-Bought Ratio', ascending=False)
    return bullish_df, bearish_df, latest_date

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
        metrics['TotalVolume'] = total_volume
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

# Alert check
def check_alerts(df_results: pd.DataFrame, symbol: str, threshold: float = 2.0) -> None:
    if not df_results.empty and df_results['buy_to_sell_ratio'].max() > threshold:
        st.warning(f"Alert: {symbol} has a Buy/Sell Ratio above {threshold} on {df_results['date'].iloc[0].strftime('%Y-%m-%d')}!")

# Natural language query processing
def process_natural_language_query(query: str, theme_mapping: dict) -> dict:
    query = query.lower().strip()
    result = {'intent': None, 'params': {}, 'error': None, 'output': None}
    def extract_days(text):
        import re
        match = re.search(r'(\d+)\s*(days?|weeks?)', text)
        if match:
            num = int(match.group(1))
            unit = match.group(2)
            return num * 7 if 'week' in unit else num
        return 20
    def find_theme(text):
        for theme in theme_mapping.keys():
            if theme.lower() in text:
                return theme
        return None
    def extract_symbol(text):
        import re
        words = text.split()
        for word in words:
            word_clean = word.replace('$', '').upper()
            if re.match(r'^[A-Z]{1,4}$', word_clean):
                return word_clean
        return None
    if any(keyword in query for keyword in ['bullish', 'bearish', 'top', 'recommend', 'long', 'short']):
        result['intent'] = 'recommendations'
        theme = find_theme(query)
        if theme:
            result['params']['theme'] = theme
        else:
            result['params']['theme'] = None
    elif any(keyword in query for keyword in ['analyze', 'stock', 'data for']):
        symbol = extract_symbol(query)
        if symbol:
            result['intent'] = 'single_stock'
            result['params']['symbol'] = symbol
            result['params']['lookback_days'] = extract_days(query)
            result['params']['threshold'] = 1.5
        else:
            result['error'] = f"Could not identify a valid stock symbol in query: {query}"
    elif any(keyword in query for keyword in ['theme', 'sentiment', 'summarize']):
        theme = find_theme(query)
        if theme:
            result['intent'] = 'theme_summary'
            result['params']['theme'] = theme
        else:
            result['error'] = f"Could not identify a valid theme in query: {query}"
    elif any(keyword in query for keyword in ['show data', 'theme data', 'stocks in']):
        theme = find_theme(query)
        if theme:
            result['intent'] = 'theme_analysis'
            result['params']['theme'] = theme
        else:
            result['error'] = f"Could not identify a valid theme in query: {query}"
    elif any(keyword in query for keyword in ['accumulation', 'distribution', 'patterns']):
        result['intent'] = 'accumulation_distribution'
        pattern_type = 'accumulation' if 'accumulation' in query else 'distribution'
        result['params']['pattern_type'] = pattern_type
        result['params']['lookback_days'] = extract_days(query)
        result['params']['min_volume'] = 1000000
        result['params']['min_price'] = 5.0
        result['params']['min_market_cap'] = 500
        result['params']['use_price_validation'] = True
    else:
        result['error'] = f"Unrecognized query: {query}. Try phrases like 'Show top bullish stocks in Technology', 'Analyze AAPL for 10 days', 'Show accumulation patterns', or 'Summarize sentiment for Financials'."
    return result

def run():
    #st.set_page_config(page_title="FINRA Short Sale Analysis")
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
    
    # Initialize session state
    if 'analysis_results' not in st.session_state:
        st.session_state['analysis_results'] = {}
    if 'recommendations' not in st.session_state:
        st.session_state['recommendations'] = None
    if 'theme_summary' not in st.session_state:
        st.session_state['theme_summary'] = None
    if 'stock_summary' not in st.session_state:
        st.session_state['stock_summary'] = None
    
    
    # Natural Language Query Interface
    st.subheader("Ask a Question")
    query = st.text_input("Enter your query (e.g., 'Show top bullish stocks in Technology', 'Analyze AAPL for 10 days', 'Show accumulation patterns')", "")
    if query:
        with st.spinner("Processing your query..."):
            query_result = process_natural_language_query(query, theme_mapping)
            if query_result['error']:
                st.error(query_result['error'])
            else:
                intent = query_result['intent']
                params = query_result['params']
                
                if intent == 'recommendations':
                    long_df, short_df, latest_date = generate_recommendations()
                    if params['theme']:
                        long_df = long_df[long_df['Theme'] == params['theme']]
                        short_df = short_df[short_df['Theme'] == params['theme']]
                    st.write(f"### Top Recommendations for {params['theme'] or 'All Themes'} (Data: {latest_date})")
                    if not long_df.empty:
                        st.write("#### Long Candidates")
                        st.dataframe(long_df.style.format({
                            'buy_to_sell_ratio': '{:.2f}',
                            'bought_volume': '{:,.0f}',
                            'sold_volume': '{:,.0f}',
                            'total_volume': '{:,.0f}',
                            'long_score': '{:.4f}'
                        }))
                        fig = px.bar(long_df, x='Symbol', y='long_score', title="Long Candidates",
                                     color_discrete_sequence=['#4CAF50'])
                        st.plotly_chart(fig)
                    if not short_df.empty:
                        st.write("#### Short Candidates")
                        st.dataframe(short_df.style.format({
                            'buy_to_sell_ratio': '{:.2f}',
                            'bought_volume': '{:,.0f}',
                            'sold_volume': '{:,.0f}',
                            'total_volume': '{:,.0f}',
                            'short_score': '{:.4f}'
                        }))
                        fig = px.bar(short_df, x='Symbol', y='short_score', title="Short Candidates",
                                     color_discrete_sequence=['#F44336'])
                        st.plotly_chart(fig)
                    if long_df.empty and short_df.empty:
                        st.write("No candidates found for the specified theme.")
                
                elif intent == 'single_stock':
                    symbol = params['symbol']
                    lookback_days = params['lookback_days']
                    threshold = params['threshold']
                    results_df, significant_days = analyze_symbol(symbol, lookback_days, threshold)
                    st.session_state['analysis_results'][symbol] = {
                        'df': results_df,
                        'significant_days': significant_days
                    }
                    if not results_df.empty:
                        st.write(f"### Analysis for {symbol} (Last {lookback_days} Days)")
                        avg_ratio = results_df['buy_to_sell_ratio'].mean()
                        max_ratio = results_df['buy_to_sell_ratio'].max()
                        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                        with metrics_col1:
                            st.metric("Average Buy/Sell Ratio", f"{avg_ratio:.2f}")
                        with metrics_col2:
                            st.metric("Max Buy/Sell Ratio", f"{max_ratio:.2f}")
                        with metrics_col3:
                            st.metric(f"Days Above {threshold}", significant_days)
                        check_alerts(results_df, symbol)
                        fig = px.line(results_df, x='date shaky ', y='buy_to_sell_ratio',
                                      title=f"{symbol} Buy/Sell Ratio Over Time",
                                      hover_data=['total_volume', 'short_volume_ratio'])
                        fig.add_hline(y=threshold, line_dash="dash", line_color="red",
                                      annotation_text=f"Threshold: {threshold}")
                        st.plotly_chart(fig)
                        st.write("#### Daily Analysis")
                        display_df = results_df.copy()
                        for col in ['total_volume', 'bought_volume', 'sold_volume']:
                            display_df[col] = display_df[col].astype(int)
                        st.dataframe(display_df.style.format({
                            'buy_to_sell_ratio': '{:.2f}',
                            'short_volume_ratio': '{:.4f}',
                            'total_volume': '{:,.0f}',
                            'bought_volume': '{:,.0f}',
                            'sold_volume': '{:,.0f}'
                        }))
                    else:
                        st.write(f"No data available for {symbol}.")
                
                elif intent == 'theme_summary':
                    bullish_df, bearish_df, latest_date = generate_theme_summary()
                    theme = params['theme']
                    st.write(f"### Sentiment Summary for {theme} (Data: {latest_date})")
                    theme_bullish = bullish_df[bullish_df['Theme'] == theme]
                    theme_bearish = bearish_df[bearish_df['Theme'] == theme]
                    if not theme_bullish.empty:
                        st.write("#### Bullish Sentiment")
                        st.dataframe(theme_bullish.style.format({
                            'Bought-to-Sold Ratio': '{:.2f}',
                            'Average Buy/Sell Ratio': '{:.2f}',
                            'Total Bought Volume': '{:,.0f}',
                            'Total Sold Volume': '{:,.0f}',
                            'Total Volume': '{:,.0f}',
                            'Number of Stocks': '{:d}'
                        }))
                    elif not theme_bearish.empty:
                        st.write("#### Bearish Sentiment")
                        st.dataframe(theme_bearish.style.format({
                            'Bought-to-Sold Ratio': '{:.2f}',
                            'Average Buy/Sell Ratio': '{:.2f}',
                            'Total Bought Volume': '{:,.0f}',
                            'Total Sold Volume': '{:,.0f}',
                            'Total Volume': '{:,.0f}',
                            'Number of Stocks': '{:d}'
                        }))
                    else:
                        st.write(f"No sentiment data available for {theme}.")
                    if not theme_bullish.empty or not theme_bearish.empty:
                        plot_data = []
                        if not theme_bullish.empty:
                            plot_data.append(theme_bullish[['Theme', 'Bought-to-Sold Ratio']].assign(Sentiment='Bullish'))
                        if not theme_bearish.empty:
                            plot_data.append(theme_bearish[['Theme', 'Bought-to-Sold Ratio']].assign(Sentiment='Bearish'))
                        plot_df = pd.concat(plot_data, ignore_index=True)
                        fig = px.bar(plot_df, x='Theme', y='Bought-to-Sold Ratio', color='Sentiment',
                                     title=f"{theme} Sentiment", color_discrete_map={'Bullish': '#4CAF50', 'Bearish': '#F44336'})
                        st.plotly_chart(fig)
                
                elif intent == 'theme_analysis':
                    theme = params['theme']
                    symbols = theme_mapping[theme]
                    latest_df, latest_date = get_latest_data(symbols)
                    if not latest_df.empty:
                        st.write(f"### {theme} Analysis (Data: {latest_date})")
                        metrics_list = []
                        for _, row in latest_df.iterrows():
                            total_volume = row.get('TotalVolume', 0)
                            metrics = calculate_metrics(row, total_volume)
                            metrics['Symbol'] = row['Symbol']
                            metrics['TotalVolume'] = total_volume
                            metrics_list.append(metrics)
                        theme_df = pd.DataFrame(metrics_list)
                        theme_df = theme_df.sort_values(by=['buy_to_sell_ratio', 'total_volume'], ascending=[False, False])
                        display_df = theme_df.copy()
                        for col in ['total_volume', 'bought_volume', 'sold_volume']:
                            display_df[col] = display_df[col].astype(int)
                        st.dataframe(display_df.style.format({
                            'buy_to_sell_ratio': '{:.2f}',
                            'short_volume_ratio': '{:.4f}',
                            'total_volume': '{:,.0f}',
                            'bought_volume': '{:,.0f}',
                            'sold_volume': '{:,.0f}'
                        }))
                        fig = px.bar(theme_df, x='Symbol', y='buy_to_sell_ratio',
                                     title=f"{theme} Buy/Sell Ratios",
                                     hover_data=['total_volume', 'bought_volume', 'sold_volume'])
                        st.plotly_chart(fig)
                    else:
                        st.write(f"No data available for {theme} stocks.")
                
                elif intent == 'accumulation_distribution':
                    pattern_type = params['pattern_type']
                    pattern_df = find_patterns(
                        lookback_days=params['lookback_days'],
                        min_volume=params['min_volume'],
                        pattern_type=pattern_type,
                        use_price_validation=params['use_price_validation'],
                        min_price=params['min_price'],
                        min_market_cap=params['min_market_cap'] * 1_000_000
                    )
                    st.session_state['analysis_results']['patterns'] = pattern_df
                    st.write(f"### Top 40 Stocks Showing {pattern_type.capitalize()} (Last {params['lookback_days']} Days)")
                    if not pattern_df.empty:
                        st.dataframe(pattern_df.style.background_gradient(
                            subset=['Avg Buy/Sell Ratio'],
                            cmap='Greens' if pattern_type == 'accumulation' else 'Reds'
                        ), use_container_width=True)
                        fig = px.bar(
                            pattern_df.head(10),
                            x='Symbol',
                            y='Avg Buy/Sell Ratio',
                            color='Volume Trend',
                            title=f"Top 10 {pattern_type.capitalize()} Stocks",
                            hover_data=['Price', 'Market Cap (M)', 'Avg Daily Volume', 'Days Showing Pattern']
                        )
                        fig.update_layout(xaxis_tickangle=-45)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.write(f"No {pattern_type} patterns detected with current filters.")
    
    # Create tabs
    tab_names = ["Recommendations", "Theme Summary", "Stock Summary", "Single Stock", "Theme Analysis", "Accumulation/Distribution"]
    tabs = st.tabs(tab_names)
    
    # Recommendations Tab
    with tabs[0]:
        st.subheader("Top Long and Short Recommendations")
        st.write("Based on a holistic analysis of FINRA short sale data across all themes.")
        if st.button("Generate Recommendations"):
            with st.spinner("Analyzing data for recommendations..."):
                long_df, short_df, latest_date = generate_recommendations()
                st.session_state['recommendations'] = {'long_df': long_df, 'short_df': short_df, 'date': latest_date}
        if st.session_state['recommendations']:
            recommendations = st.session_state['recommendations']
            long_df = recommendations['long_df']
            short_df = recommendations['short_df']
            latest_date = recommendations['date']
            if latest_date:
                st.write(f"Data analyzed for: {latest_date}")
            st.write("### Top 5 Long Candidates")
            if not long_df.empty:
                def highlight_long(row):
                    return ['background-color: rgba(144, 238, 144, 0.3)'] * len(row)
                st.dataframe(long_df.style.apply(highlight_long, axis=1).format({
                    'buy_to_sell_ratio': '{:.2f}',
                    'bought_volume': '{:,.0f}',
                    'sold_volume': '{:,.0f}',
                    'total_volume': '{:,.0f}',
                    'long_score': '{:.4f}'
                }))
            else:
                st.write("No long candidates found with current data.")
            st.write("### Top 5 Short Candidates")
            if not short_df.empty:
                def highlight_short(row):
                    return ['background-color: rgba(255, 182, 193, 0.3)'] * len(row)
                st.dataframe(short_df.style.apply(highlight_short, axis=1).format({
                    'buy_to_sell_ratio': '{:.2f}',
                    'bought_volume': '{:,.0f}',
                    'sold_volume': '{:,.0f}',
                    'total_volume': '{:,.0f}',
                    'short_score': '{:.4f}'
                }))
            else:
                st.write("No short candidates found with current data.")
            if not long_df.empty or not short_df.empty:
                plot_data = []
                if not long_df.empty:
                    long_plot = long_df[['Symbol', 'long_score']].copy()
                    long_plot['Type'] = 'Long'
                    long_plot['Score'] = long_plot['long_score']
                    plot_data.append(long_plot[['Symbol', 'Score', 'Type']])
                if not short_df.empty:
                    short_plot = short_df[['Symbol', 'short_score']].copy()
                    short_plot['Type'] = 'Short'
                    short_plot['Score'] = short_plot['short_score']
                    plot_data.append(short_plot[['Symbol', 'Score', 'Type']])
                if plot_data:
                    plot_df = pd.concat(plot_data, ignore_index=True)
                    fig = px.bar(plot_df, x='Symbol', y='Score', color='Type',
                                title="Top Long and Short Candidates by Score",
                                color_discrete_map={'Long': '#4CAF50', 'Short': '#F44336'},
                                hover_data=['Symbol', 'Score'])
                    fig.update_layout(barmode='group', xaxis_tickangle=-45)
                    st.plotly_chart(fig)
    
    # Theme Summary Tab
    with tabs[1]:
        st.subheader("Theme Sentiment Summary")
        st.write("Analysis of overall bullish and bearish themes based on FINRA short sale data.")
        if st.button("Generate Theme Summary"):
            with st.spinner("Analyzing theme sentiment..."):
                bullish_df, bearish_df, latest_date = generate_theme_summary()
                st.session_state['theme_summary'] = {'bullish_df': bullish_df, 'bearish_df': bearish_df, 'date': latest_date}
        if st.session_state['theme_summary']:
            theme_summary = st.session_state['theme_summary']
            bullish_df = theme_summary['bullish_df']
            bearish_df = theme_summary['bearish_df']
            latest_date = theme_summary['date']
            if latest_date:
                st.write(f"Data analyzed for: {latest_date}")
            st.write("### Bullish Themes (Bought Volume > Sold Volume)")
            if not bullish_df.empty:
                def highlight_bullish(row):
                    return ['background-color: rgba(144, 238, 144, 0.3)'] * len(row)
                st.dataframe(bullish_df.style.apply(highlight_bullish, axis=1).format({
                    'Bought-to-Sold Ratio': '{:.2f}',
                    'Average Buy/Sell Ratio': '{:.2f}',
                    'Total Bought Volume': '{:,.0f}',
                    'Total Sold Volume': '{:,.0f}',
                    'Total Volume': '{:,.0f}',
                    'Number of Stocks': '{:d}'
                }))
            else:
                st.write("No bullish themes found with current data.")
            st.write("### Bearish Themes (Sold Volume >= Bought Volume)")
            if not bearish_df.empty:
                def highlight_bearish(row):
                    return ['background-color: rgba(255, 182, 193, 0.3)'] * len(row)
                st.dataframe(bearish_df.style.apply(highlight_bearish, axis=1).format({
                    'Bought-to-Sold Ratio': '{:.2f}',
                    'Average Buy/Sell Ratio': '{:.2f}',
                    'Total Bought Volume': '{:,.0f}',
                    'Total Sold Volume': '{:,.0f}',
                    'Total Volume': '{:,.0f}',
                    'Number of Stocks': '{:d}'
                }))
            else:
                st.write("No bearish themes found with current data.")
            if not bullish_df.empty or not bearish_df.empty:
                plot_data = []
                if not bullish_df.empty:
                    bullish_plot = bullish_df[['Theme', 'Bought-to-Sold Ratio']].copy()
                    bullish_plot['Sentiment'] = 'Bullish'
                    bullish_plot['Ratio'] = bullish_plot['Bought-to-Sold Ratio']
                    plot_data.append(bullish_plot[['Theme', 'Ratio', 'Sentiment']])
                if not bearish_df.empty:
                    bearish_plot = bearish_df[['Theme', 'Bought-to-Sold Ratio']].copy()
                    bearish_plot['Sentiment'] = 'Bearish'
                    bearish_plot['Ratio'] = bearish_plot['Bought-to-Sold Ratio']
                    plot_data.append(bearish_plot[['Theme', 'Ratio', 'Sentiment']])
                if plot_data:
                    plot_df = pd.concat(plot_data, ignore_index=True)
                    fig = px.bar(plot_df, x='Theme', y='Ratio', color='Sentiment',
                                title="Theme Sentiment by Bought-to-Sold Volume Ratio",
                                color_discrete_map={'Bullish': '#4CAF50', 'Bearish': '#F44336'},
                                hover_data=['Theme', 'Ratio'])
                    fig.update_layout(barmode='group', xaxis_tickangle=-45)
                    fig.add_hline(y=1.0, line_dash="dash", line_color="gray", annotation_text="Neutral Line")
                    st.plotly_chart(fig)
    
    # Stock Summary Tab
    with tabs[2]:
        st.subheader("Stock Volume Summary")
        st.write("Stocks with significant buying (Bought Volume > 2x Sold Volume) or selling (Sold Volume > 2x Bought Volume) based on latest FINRA short sale data.")
        if st.button("Generate Stock Summary"):
            with st.spinner("Analyzing stock volume data..."):
                high_buy_df, high_sell_df, latest_date = generate_stock_summary()
                st.session_state['stock_summary'] = {'high_buy_df': high_buy_df, 'high_sell_df': high_sell_df, 'date': latest_date}
        if st.session_state['stock_summary']:
            stock_summary = st.session_state['stock_summary']
            high_buy_df = stock_summary['high_buy_df']
            high_sell_df = stock_summary['high_sell_df']
            latest_date = stock_summary['date']
            if latest_date:
                st.write(f"Data analyzed for: {latest_date}")
            st.write("### Stocks with High Buying (Bought Volume > 2x Sold Volume)")
            if not high_buy_df.empty:
                def highlight_high_buy(row):
                    return ['background-color: rgba(144, 238, 144, 0.3)'] * len(row)
                st.dataframe(high_buy_df.style.apply(highlight_high_buy, axis=1).format({
                    'buy_to_sell_ratio': '{:.2f}',
                    'bought_volume': '{:,.0f}',
                    'sold_volume': '{:,.0f}',
                    'total_volume': '{:,.0f}'
                }))
            else:
                st.write("No stocks found with Bought Volume > 2x Sold Volume.")
            st.write("### Stocks with High Selling (Sold Volume > 2x Bought Volume)")
            if not high_sell_df.empty:
                def highlight_high_sell(row):
                    return ['background-color: rgba(255, 182, 193, 0.3)'] * len(row)
                st.dataframe(high_sell_df.style.apply(highlight_high_sell, axis=1).format({
                    'buy_to_sell_ratio': '{:.2f}',
                    'bought_volume': '{:,.0f}',
                    'sold_volume': '{:,.0f}',
                    'total_volume': '{:,.0f}'
                }))
            else:
                st.write("No stocks found with Sold Volume > 2x Bought Volume.")
            if not high_buy_df.empty or not high_sell_df.empty:
                plot_data = []
                if not high_buy_df.empty:
                    buy_plot = high_buy_df[['Symbol', 'buy_to_sell_ratio']].copy()
                    buy_plot['Type'] = 'High Buy'
                    buy_plot['Ratio'] = buy_plot['buy_to_sell_ratio']
                    plot_data.append(buy_plot[['Symbol', 'Ratio', 'Type']])
                if not high_sell_df.empty:
                    sell_plot = high_sell_df[['Symbol', 'buy_to_sell_ratio']].copy()
                    sell_plot['Type'] = 'High Sell'
                    sell_plot['Ratio'] = sell_plot['buy_to_sell_ratio']
                    plot_data.append(sell_plot[['Symbol', 'Ratio', 'Type']])
                if plot_data:
                    plot_df = pd.concat(plot_data, ignore_index=True)
                    fig = px.bar(plot_df, x='Symbol', y='Ratio', color='Type',
                                title="Stocks with Significant Buy or Sell Volume",
                                color_discrete_map={'High Buy': '#4CAF50', 'High Sell': '#F44336'},
                                hover_data=['Symbol', 'Ratio'])
                    fig.update_layout(barmode='group', xaxis_tickangle=-45)
                    fig.add_hline(y=1.0, line_dash="dash", line_color="gray", annotation_text="Neutral Line")
                    st.plotly_chart(fig)
    
    # Single Stock Tab
    with tabs[3]:
        col1, col2 = st.columns(2)
        with col1:
            symbol = st.text_input("Enter Symbol", "SPY").strip().upper()
            lookback_days = st.slider("Lookback Days", 1, 30, 20)
        with col2:
            threshold = st.number_input("Buy/Sell Ratio Threshold", min_value=1.0, max_value=5.0, value=1.5, step=0.1)
        if st.button("Analyze Stock"):
            with st.spinner(f"Analyzing {symbol}..."):
                results_df, significant_days = analyze_symbol(symbol, lookback_days, threshold)
                st.session_state['analysis_results'][symbol] = {'df': results_df, 'significant_days': significant_days}
        if symbol in st.session_state['analysis_results']:
            results_df = st.session_state['analysis_results'][symbol]['df']
            significant_days = st.session_state['analysis_results'][symbol]['significant_days']
            if not results_df.empty:
                st.subheader("Summary")
                avg_ratio = results_df['buy_to_sell_ratio'].mean()
                max_ratio = results_df['buy_to_sell_ratio'].max()
                metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                with metrics_col1:
                    st.metric("Average Buy/Sell Ratio", f"{avg_ratio:.2f}")
                with metrics_col2:
                    st.metric("Max Buy/Sell Ratio", f"{max_ratio:.2f}")
                with metrics_col3:
                    st.metric(f"Days Above {threshold}", significant_days)
                check_alerts(results_df, symbol)
                fig = px.line(results_df, x='date', y='buy_to_sell_ratio', title=f"{symbol} Buy/Sell Ratio Over Time",
                              hover_data=['total_volume', 'short_volume_ratio', 'cumulative_bought', 'cumulative_sold'])
                fig.add_hline(y=threshold, line_dash="dash", line_color="red", annotation_text=f"Threshold: {threshold}")
                st.plotly_chart(fig)
                st.subheader("Daily Analysis")
                display_df = results_df.copy()
                for col in ['total_volume', 'bought_volume', 'sold_volume', 'cumulative_bought', 'cumulative_sold']:
                    display_df[col] = display_df[col].astype(int)
                def highlight_row(row):
                    color = 'background-color: rgba(144, 238, 144, 0.3)' if row['cumulative_bought'] > row['cumulative_sold'] else 'background-color: rgba(255, 182, 193, 0.3)'
                    return [color] * len(row)
                st.dataframe(display_df.style.apply(highlight_row, axis=1))
            else:
                st.write(f"No data available for {symbol}.")
    
    # Theme Analysis Tab
    with tabs[4]:
        st.subheader("Theme Analysis")
        st.write("Select a theme to view the latest FINRA short sale data for its stocks.")
        selected_theme = st.selectbox("Select Theme", list(theme_mapping.keys()), index=0)
        symbols = theme_mapping[selected_theme]
        latest_df, latest_date = get_latest_data(symbols)
        if latest_df.empty or latest_date is None:
            st.write(f"No data available for {selected_theme} stocks on the latest day.")
        else:
            st.write(f"Showing data for: {latest_date}")
            metrics_list = []
            for _, row in latest_df.iterrows():
                total_volume = row.get('TotalVolume', 0)
                metrics = calculate_metrics(row, total_volume)
                metrics['Symbol'] = row['Symbol']
                metrics['TotalVolume'] = total_volume
                metrics_list.append(metrics)
            theme_df = pd.DataFrame(metrics_list)
            theme_df = theme_df.sort_values(by=['buy_to_sell_ratio', 'total_volume'], ascending=[False, False])
            if not theme_df.empty:
                display_df = theme_df.copy()
                for col in ['total_volume', 'bought_volume', 'sold_volume']:
                    display_df[col] = display_df[col].astype(int)
                display_df['cumulative_bought'] = display_df['bought_volume'].cumsum()
                display_df['cumulative_sold'] = display_df['sold_volume'].cumsum()
                def highlight_row(row):
                    color = 'background-color: rgba(144, 238, 144, 0.3)' if row['bought_volume'] > row['sold_volume'] else 'background-color: rgba(255, 182, 193, 0.3)'
                    return [color] * len(row)
                column_order = ['Symbol', 'buy_to_sell_ratio', 'total_volume', 'bought_volume', 'sold_volume', 'short_volume_ratio', 'cumulative_bought', 'cumulative_sold']
                st.dataframe(display_df[column_order].style.apply(highlight_row, axis=1).format({
                    'buy_to_sell_ratio': '{:.2f}',
                    'short_volume_ratio': '{:.4f}',
                    'total_volume': '{:,.0f}',
                    'bought_volume': '{:,.0f}',
                    'sold_volume': '{:,.0f}',
                    'cumulative_bought': '{:,.0f}',
                    'cumulative_sold': '{:,.0f}'
                }))
                total_bought_volume = theme_df['bought_volume'].sum()
                total_sold_volume = theme_df['sold_volume'].sum()
                st.write("### Summary")
                st.write(f"Total Bought Volume: {total_bought_volume:,.0f}")
                st.write(f"Total Sold Volume: {total_sold_volume:,.0f}")
                st.write(f"Dark Pools: {'Bullish' if total_bought_volume > total_sold_volume else 'Bearish'}")
                fig = px.bar(theme_df, x='Symbol', y='buy_to_sell_ratio',
                            title=f"{selected_theme} Buy/Sell Ratios for {latest_date}",
                            hover_data=['total_volume', 'bought_volume', 'sold_volume'])
                fig.update_layout(barmode='group', xaxis_tickangle=-45)
                st.plotly_chart(fig)
            else:
                st.write(f"No records found for {selected_theme} stocks on {latest_date}.")
    
    # Accumulation/Distribution Tab
    with tabs[5]:
        st.subheader("Accumulation/Distribution Analysis")
        st.write("Identify stocks showing accumulation (buying pressure) or distribution (selling pressure) patterns over the last 10 days.")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            min_volume = st.number_input("Minimum Daily Volume", value=500000, step=100000, format="%d")
        with col2:
            min_price = st.number_input("Minimum Price ($)", value=5.0, min_value=0.5, max_value=50.0, step=0.5)
        with col3:
            min_market_cap = st.number_input("Minimum Market Cap ($M)", value=500, min_value=50, max_value=5000, step=50)
        with col4:
            use_validation = st.checkbox("Use Price Validation", value=False, help="Ensure price trends align with pattern")
        with col5:
            pattern_type = st.selectbox("Pattern Type", ["Accumulation", "Distribution"], index=0)
        if st.button("Find Patterns"):
            with st.spinner(f"Updating stock data and finding {pattern_type.lower()} patterns..."):
                # Automatically update database with all symbols from theme_mapping
                all_symbols = set()
                for symbols in theme_mapping.values():
                    all_symbols.update(symbols)
                update_stock_database(list(all_symbols))
                logger.info(f"Updated database with {len(all_symbols)} symbols")
                pattern_df = find_patterns(
                    lookback_days=10,
                    min_volume=min_volume,
                    pattern_type=pattern_type.lower(),
                    use_price_validation=use_validation,
                    min_price=min_price,
                    min_market_cap=min_market_cap * 1_000_000
                )
                st.session_state['analysis_results']['patterns'] = pattern_df
        if 'patterns' in st.session_state['analysis_results']:
            pattern_df = st.session_state['analysis_results']['patterns']
            if not pattern_df.empty:
                st.dataframe(pattern_df.style.background_gradient(
                    subset=['Avg Buy/Sell Ratio'],
                    cmap='Greens' if pattern_type == 'accumulation' else 'Reds'
                ), use_container_width=True)
                fig = px.bar(
                    pattern_df.head(10),
                    x='Symbol',
                    y='Avg Buy/Sell Ratio',
                    color='Volume Trend',
                    title=f"Top 10 {pattern_type} Stocks",
                    hover_data=['Price', 'Market Cap (M)', 'Avg Daily Volume', 'Days Showing Pattern']
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write(f"No {pattern_type.lower()} patterns detected with current filters. Try relaxing volume, price, or market cap filters, or disabling price validation.")

if __name__ == "__main__":
    run()
