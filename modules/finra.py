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
import numpy as np

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

all_symbols = list(set([symbol for symbols in theme_mapping.values() for symbol in symbols]))

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

def get_price_data(symbols: List[str]) -> dict:
    """Get current price and 1-day change for symbols"""
    price_data = {}
    try:
        tickers = yf.download(symbols, period="5d", group_by='ticker', progress=False)
        for symbol in symbols:
            try:
                if len(symbols) == 1:
                    hist = tickers
                else:
                    if symbol in tickers.columns.levels[0]:
                        hist = tickers[symbol]
                    else:
                        continue
                
                if not hist.empty and len(hist) >= 2:
                    current_price = hist['Close'].iloc[-1]
                    prev_price = hist['Close'].iloc[-2]
                    change_1d = ((current_price / prev_price) - 1) * 100
                    price_data[symbol] = {
                        'current_price': round(current_price, 2),
                        'change_1d': round(change_1d, 2)
                    }
            except Exception as e:
                continue
    except Exception as e:
        logger.warning(f"Error fetching price data: {e}")
    
    return price_data

def calculate_volume_strength(current_volume: float, historical_volumes: List[float]) -> float:
    """Calculate volume strength vs historical average"""
    if not historical_volumes or len(historical_volumes) == 0:
        return 1.0
    avg_volume = np.mean(historical_volumes)
    return current_volume / avg_volume if avg_volume > 0 else 1.0

def get_trend_direction(historical_ratios: List[float]) -> str:
    """Get 3-day trend direction for buy/sell ratio"""
    if len(historical_ratios) < 3:
        return "-"
    
    recent = historical_ratios[-3:]
    if recent[-1] > recent[-2] > recent[-3]:
        return "↗️ Up"
    elif recent[-1] < recent[-2] < recent[-3]:
        return "↘️ Down"
    else:
        return "→ Flat"

def calculate_momentum_score(historical_data: List[dict]) -> float:
    """Calculate momentum score based on recent ratio trends"""
    if len(historical_data) < 3:
        return 0.0
    
    recent_ratios = [entry['buy_to_sell_ratio'] for entry in historical_data[-3:]]
    if len(recent_ratios) >= 3:
        # Simple trend calculation: +1 for each day ratio increases
        momentum = 0
        for i in range(1, len(recent_ratios)):
            if recent_ratios[i] > recent_ratios[i-1]:
                momentum += 1
        return momentum / (len(recent_ratios) - 1)  # Normalize to 0-1
    return 0.0

def get_volume_percentile(current_volume: float, historical_volumes: List[float]) -> int:
    """Calculate what percentile current volume represents vs historical"""
    if not historical_volumes:
        return 50
    
    sorted_volumes = sorted(historical_volumes)
    if current_volume <= sorted_volumes[0]:
        return 0
    if current_volume >= sorted_volumes[-1]:
        return 100
    
    position = sum(1 for v in sorted_volumes if v < current_volume)
    return int((position / len(sorted_volumes)) * 100)

def calculate_consistency_score(historical_ratios: List[float], threshold: float = 1.2) -> tuple[int, int]:
    """Calculate how many days in last N had bullish/bearish ratios"""
    if not historical_ratios:
        return 0, 0
    
    bullish_days = sum(1 for ratio in historical_ratios if ratio > threshold)
    bearish_days = sum(1 for ratio in historical_ratios if ratio < (1/threshold))
    return bullish_days, bearish_days

def get_market_cap_category(market_cap: float) -> str:
    """Categorize stocks by market cap"""
    if market_cap >= 200_000_000_000:  # 200B+
        return "Mega Cap"
    elif market_cap >= 10_000_000_000:  # 10B+
        return "Large Cap"
    elif market_cap >= 2_000_000_000:   # 2B+
        return "Mid Cap"
    elif market_cap >= 300_000_000:     # 300M+
        return "Small Cap"
    else:
        return "Micro Cap"

def calculate_risk_reward_ratio(current_price: float, support_level: float, resistance_level: float) -> str:
    """Calculate basic risk/reward based on price levels"""
    if support_level <= 0 or resistance_level <= 0:
        return "N/A"
    
    risk = abs(current_price - support_level)
    reward = abs(resistance_level - current_price)
    
    if risk > 0:
        ratio = reward / risk
        return f"{ratio:.1f}:1"
    return "N/A"

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

# Historical metrics function
@st.cache_data(ttl=3600)
def get_historical_metrics(symbols: List[str], max_days: int = 30) -> dict:
    date_to_df = {}
    for i in range(max_days):
        date_str = (datetime.now() - timedelta(days=i)).strftime("%Y%m%d")
        data = download_finra_short_sale_data(date_str)
        if data:
            df = process_finra_short_sale_data(data)
            date_to_df[date_str] = df
    historical = {symbol: [] for symbol in symbols}
    for date_str, df in date_to_df.items():
        for symbol in symbols:
            symbol_data = df[df['Symbol'] == symbol]
            if not symbol_data.empty:
                row = symbol_data.iloc[0]
                total_volume = row.get('TotalVolume', 0)
                metrics = calculate_metrics(row, total_volume)
                metrics['date'] = pd.to_datetime(date_str, format='%Y%m%d')
                historical[symbol].append(metrics)
    for symbol in historical:
        historical[symbol] = sorted(historical[symbol], key=lambda x: x['date'])
    return historical

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
        df_results = df_results.sort_values('date', ascending=True)
        # Compute deviation using rolling averages (only 5-day now)
        df_results['rolling_avg_b_5'] = df_results['bought_volume'].rolling(5, min_periods=5).mean().shift(1)
        df_results['rolling_avg_s_5'] = df_results['sold_volume'].rolling(5, min_periods=5).mean().shift(1)
        
        df_results['dev_b_5'] = np.where(
            (df_results['rolling_avg_b_5'] > 0) & pd.notnull(df_results['rolling_avg_b_5']),
            ((df_results['bought_volume'] - df_results['rolling_avg_b_5']) / df_results['rolling_avg_b_5'] * 100).round(0),
            np.nan
        )
        df_results['dev_s_5'] = np.where(
            (df_results['rolling_avg_s_5'] > 0) & pd.notnull(df_results['rolling_avg_s_5']),
            ((df_results['sold_volume'] - df_results['rolling_avg_s_5']) / df_results['rolling_avg_s_5'] * 100).round(0),
            np.nan
        )
        
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

# Enhanced stock summary generation
def generate_stock_summary() -> tuple[pd.DataFrame, pd.DataFrame, Optional[str]]:
    symbol_themes = {}
    for theme, symbols in theme_mapping.items():
        for symbol in symbols:
            if symbol not in symbol_themes:
                symbol_themes[symbol] = theme
    
    # Get price data
    price_data = get_price_data(all_symbols)
    
    historical = get_historical_metrics(all_symbols)
    latest_date = None
    for hist in historical.values():
        if hist:
            latest_date = max(latest_date, hist[-1]['date'].strftime('%Y%m%d')) if latest_date else hist[-1]['date'].strftime('%Y%m%d')
    if not latest_date:
        return pd.DataFrame(), pd.DataFrame(), None
    
    metrics_list = []
    for symbol in all_symbols:
        hist = historical[symbol]
        if hist and hist[-1]['date'].strftime('%Y%m%d') == latest_date:
            metrics = hist[-1].copy()
            metrics['Symbol'] = symbol
            past = hist[:-1]
            
            # Calculate 5-day deviations only
            dev_b_5 = np.nan
            dev_s_5 = np.nan
            if len(past) >= 5:
                avg_b_5 = np.mean([p['bought_volume'] for p in past[-5:]])
                avg_s_5 = np.mean([p['sold_volume'] for p in past[-5:]])
                if avg_b_5 > 0:
                    dev_b_5 = round(((metrics['bought_volume'] - avg_b_5) / avg_b_5 * 100), 0)
                if avg_s_5 > 0:
                    dev_s_5 = round(((metrics['sold_volume'] - avg_s_5) / avg_s_5 * 100), 0)
                
                # Calculate additional metrics
                historical_volumes = [p['total_volume'] for p in past[-5:]]
                historical_ratios = [p['buy_to_sell_ratio'] for p in past[-3:]]
                
                metrics['volume_strength'] = calculate_volume_strength(metrics['total_volume'], historical_volumes)
                metrics['trend_3d'] = get_trend_direction(historical_ratios)
                
                # Calculate momentum score (0-1 scale) using past data
                metrics['momentum_score'] = calculate_momentum_score(past[-3:]) if len(past) >= 3 else 0.0
                
                # Calculate volume percentile over 10 days
                ten_day_volumes = [p['total_volume'] for p in past[-10:] if p.get('total_volume')]
                metrics['volume_percentile'] = get_volume_percentile(metrics['total_volume'], ten_day_volumes)
                
                # Calculate consistency score
                past_10_days = [p for p in past[-10:]]
                past_10_ratios = [p['buy_to_sell_ratio'] for p in past_10_days if 'buy_to_sell_ratio' in p]
                bullish_days, bearish_days = calculate_consistency_score(past_10_ratios)
                metrics['bullish_days_10'] = bullish_days
                metrics['bearish_days_10'] = bearish_days
            else:
                metrics['volume_strength'] = 1.0
                metrics['trend_3d'] = "-"
                metrics['momentum_score'] = 0.0
                metrics['volume_percentile'] = 50.0
                metrics['bullish_days_10'] = 0
                metrics['bearish_days_10'] = 0
            
            # Add price data
            if symbol in price_data:
                metrics['current_price'] = price_data[symbol]['current_price']
                metrics['price_change_1d'] = price_data[symbol]['change_1d']
            else:
                metrics['current_price'] = 0.0
                metrics['price_change_1d'] = 0.0
            
            metrics['dev_b_5'] = dev_b_5
            metrics['dev_s_5'] = dev_s_5
            metrics_list.append(metrics)
    
    df = pd.DataFrame(metrics_list)
    df = df.drop_duplicates(subset=['Symbol'], keep='first')
    df['Theme'] = df['Symbol'].map(symbol_themes)
    high_buy_df = df[df['bought_volume'] > 2 * df['sold_volume']].copy()
    high_sell_df = df[df['sold_volume'] > 2 * df['bought_volume']].copy()
    
    # Add deviation columns to the filtered dataframes
    for filtered_df in [high_buy_df, high_sell_df]:
        if not filtered_df.empty:
            for col in ['bought_volume', 'sold_volume', 'total_volume']:
                filtered_df[col] = filtered_df[col].astype(int)
            filtered_df['buy_to_sell_ratio'] = filtered_df['buy_to_sell_ratio'].round(2)
    
    return high_buy_df, high_sell_df, latest_date

def generate_themes_summary(period_days: int = 1):
    historical = get_historical_metrics(all_symbols)
    
    # Get all unique dates
    all_dates = set()
    for hist in historical.values():
        for entry in hist:
            all_dates.add(entry['date'])
    sorted_dates = sorted(list(all_dates), reverse=True)
    
    if len(sorted_dates) < period_days:
        period_dates = sorted_dates
    else:
        period_dates = sorted_dates[:period_days]
    
    # Now, for each theme, aggregate
    theme_aggregates = {}
    for theme, symbols in theme_mapping.items():
        total_b = 0
        total_s = 0
        stock_aggregates = {sym: {'b': 0, 's': 0} for sym in symbols}
        for sym in symbols:
            hist = historical.get(sym, [])
            for entry in hist:
                if entry['date'] in period_dates:
                    stock_aggregates[sym]['b'] += entry['bought_volume']
                    stock_aggregates[sym]['s'] += entry['sold_volume']
                    total_b += entry['bought_volume']
                    total_s += entry['sold_volume']
        
        total_v = total_b + total_s
        if total_v == 0:
            continue
        
        if total_s > 0:
            theme_ratio = total_b / total_s
        else:
            theme_ratio = float('inf')
        
        stock_ratios = {}
        for sym, ag in stock_aggregates.items():
            if ag['b'] + ag['s'] == 0:
                continue
            if ag['s'] > 0:
                stock_ratios[sym] = ag['b'] / ag['s']
            else:
                stock_ratios[sym] = float('inf')
        
        if not stock_ratios:
            continue
        
        theme_aggregates[theme] = {
            'ratio': theme_ratio,
            'stock_ratios': stock_ratios
        }
    
    # Sort themes by ratio desc
    sorted_themes = sorted(theme_aggregates.items(), key=lambda x: x[1]['ratio'], reverse=True)
    
    return sorted_themes, period_dates

def get_signal(ratio):
    if ratio > 1.2:
        return 'Buy'
    elif ratio > 1.0:
        return 'Add'
    elif 0.5 < ratio <= 1.0:
        return 'Trim'
    else:
        return 'Sell'

def style_signal_dark(val):
    if val == 'Buy':
        return 'background-color: #22c55e; color: #ffffff; font-weight: bold'
    elif val == 'Add':
        return 'background-color: #4ade80; color: #ffffff; font-weight: bold'
    elif val == 'Trim':
        return 'background-color: #ef4444; color: #ffffff; font-weight: bold'
    elif val == 'Sell':
        return 'background-color: #b91c1c; color: #ffffff; font-weight: bold'
    return ''

def style_dev_dark(val):
    if val == '-':
        return 'background-color: #2d2d2d; color: #888888'
    try:
        num = float(val.rstrip('%'))
        if num > 50:
            return 'background-color: #22c55e; color: #ffffff; font-weight: bold'
        elif num > 20:
            return 'background-color: #4ade80; color: #ffffff'
        elif num < -50:
            return 'background-color: #ef4444; color: #ffffff; font-weight: bold'
        elif num < -20:
            return 'background-color: #fca5a5; color: #ffffff'
    except:
        pass
    return 'background-color: #2d2d2d; color: #ffffff'

def style_price_change_dark(val):
    try:
        num = float(val.rstrip('%'))
        if num > 2:
            return 'background-color: #22c55e; color: #ffffff; font-weight: bold'
        elif num > 0:
            return 'background-color: #4ade80; color: #ffffff'
        elif num < -2:
            return 'background-color: #ef4444; color: #ffffff; font-weight: bold'
        elif num < 0:
            return 'background-color: #fca5a5; color: #ffffff'
    except:
        pass
    return 'background-color: #2d2d2d; color: #ffffff'

def style_anomaly_dark(val):
    try:
        num = float(val)
        if num > 2.0:
            return 'background-color: #ef4444; color: #ffffff; font-weight: bold'
        elif num > 1.0:
            return 'background-color: #f97316; color: #ffffff; font-weight: bold'
        elif num > 0.5:
            return 'background-color: #fde047; color: #000000'
    except:
        pass
    return 'background-color: #2d2d2d; color: #ffffff'

def style_bot_percentage(val):
    try:
        num = float(val.rstrip('%'))
        if num >= 60:
            return 'background-color: #22c55e; color: #ffffff; font-weight: bold'
        elif num >= 50:
            return 'background-color: #4ade80; color: #ffffff'
        elif num >= 40:
            return 'background-color: #fde047; color: #000000'
        elif num >= 30:
            return 'background-color: #f97316; color: #ffffff'
        else:
            return 'background-color: #fca5a5; color: #ffffff'
    except:
        pass
    return 'background-color: #2d2d2d; color: #ffffff'

def style_ratio_dark(val):
    if val == '∞':
        return 'background-color: #22c55e; color: #ffffff; font-weight: bold'
    try:
        num = float(val)
        if num > 1.5:
            return 'background-color: #22c55e; color: #ffffff; font-weight: bold'
        elif num > 1:
            return 'background-color: #4ade80; color: #ffffff'
        elif num < 0.5:
            return 'background-color: #ef4444; color: #ffffff; font-weight: bold'
        elif num < 1:
            return 'background-color: #fca5a5; color: #ffffff'
    except:
        pass
    return 'background-color: #2d2d2d; color: #ffffff'

def format_enhanced_dataframe(df, focus_type="bought"):
    """Format dataframe with enhanced trading insights"""
    display_df = df.copy()
    
    # Basic formatting
    display_df['Current Price'] = display_df['current_price'].apply(lambda x: f"${x:.2f}")
    display_df['Price Change'] = display_df['price_change_1d'].apply(lambda x: f"{x:+.1f}%")
    display_df['BOT %'] = (display_df['bought_volume'] / display_df['total_volume'] * 100).round(0).astype(int).apply(lambda x: f"{x}%")
    display_df['Signal'] = display_df['buy_to_sell_ratio'].apply(get_signal)
    display_df['Volume Strength'] = display_df['volume_strength'].apply(lambda x: f"{x:.1f}x")
    
    # Enhanced metrics
    display_df['Volume Rank'] = display_df['volume_percentile'].apply(lambda x: f"{x}th %ile")
    display_df['Momentum'] = display_df['momentum_score'].apply(lambda x: 
        "🚀 Strong" if x >= 0.8 else "📈 Good" if x >= 0.5 else "🔄 Mixed" if x >= 0.2 else "📉 Weak")
    
    # Consistency score
    if focus_type == "bought":
        display_df['Consistency'] = display_df['bullish_days_10'].apply(lambda x:
            f"🟢 {x}/10" if x >= 7 else f"🟡 {x}/10" if x >= 4 else f"🔴 {x}/10")
    else:
        display_df['Consistency'] = display_df['bearish_days_10'].apply(lambda x:
            f"🔴 {x}/10" if x >= 7 else f"🟡 {x}/10" if x >= 4 else f"🟢 {x}/10")
    
    # Volume formatting
    for col in ['bought_volume', 'sold_volume', 'total_volume']:
        display_df[col] = display_df[col].apply(lambda x: f"{x:,.0f}")
    
    # Deviation formatting
    display_df['Bought Dev 5d'] = display_df['dev_b_5'].apply(lambda x: f"{x:+.0f}%" if pd.notnull(x) else "-")
    display_df['Sold Dev 5d'] = display_df['dev_s_5'].apply(lambda x: f"{x:+.0f}%" if pd.notnull(x) else "-")
    
    # Trend formatting
    display_df['Trend 3D'] = display_df['trend_3d']
    
    return display_df

def style_momentum(val):
    """Style momentum indicators"""
    if "Strong" in val:
        return 'background-color: #22c55e; color: #ffffff; font-weight: bold'
    elif "Good" in val:
        return 'background-color: #4ade80; color: #ffffff'
    elif "Mixed" in val:
        return 'background-color: #fbbf24; color: #000000'
    elif "Weak" in val:
        return 'background-color: #ef4444; color: #ffffff'
    return 'background-color: #2d2d2d; color: #ffffff'

def style_consistency(val):
    """Style consistency indicators"""
    if val.startswith("🟢"):
        return 'background-color: #22c55e; color: #ffffff; font-weight: bold'
    elif val.startswith("🟡"):
        return 'background-color: #fbbf24; color: #000000; font-weight: bold'
    elif val.startswith("🔴"):
        return 'background-color: #ef4444; color: #ffffff; font-weight: bold'
    return 'background-color: #2d2d2d; color: #ffffff'

def style_volume_rank(val):
    """Style volume percentile ranking"""
    try:
        percentile = int(val.split('th')[0])
        if percentile >= 90:
            return 'background-color: #22c55e; color: #ffffff; font-weight: bold'
        elif percentile >= 75:
            return 'background-color: #4ade80; color: #ffffff'
        elif percentile >= 50:
            return 'background-color: #fbbf24; color: #000000'
        else:
            return 'background-color: #6b7280; color: #ffffff'
    except:
        pass
    return 'background-color: #2d2d2d; color: #ffffff'

def create_enhanced_styled_dataframe(display_df, columns, focus_type="bought"):
    """Create enhanced styled dataframe with new insights"""
    # Filter columns to only those that exist in display_df
    available_columns = [col for col in columns if col in display_df.columns]
    
    styled_df = display_df[available_columns].style
    
    # Apply styling only to columns that exist
    if 'Signal' in available_columns:
        styled_df = styled_df.applymap(style_signal_dark, subset=['Signal'])
    if 'Bought Dev 5d' in available_columns and 'Sold Dev 5d' in available_columns:
        styled_df = styled_df.applymap(style_dev_dark, subset=['Bought Dev 5d', 'Sold Dev 5d'])
    if 'Price Change' in available_columns:
        styled_df = styled_df.applymap(style_price_change_dark, subset=['Price Change'])
    if 'Momentum' in available_columns:
        styled_df = styled_df.applymap(style_momentum, subset=['Momentum'])
    if 'Consistency' in available_columns:
        styled_df = styled_df.applymap(style_consistency, subset=['Consistency'])
    if 'Volume Rank' in available_columns:
        styled_df = styled_df.applymap(style_volume_rank, subset=['Volume Rank'])
    if 'BOT %' in available_columns:
        styled_df = styled_df.applymap(style_bot_percentage, subset=['BOT %'])
    
    return styled_df.set_table_styles([
        {'selector': 'th', 'props': [
            ('background-color', '#2d2d2d'), 
            ('color', '#ffffff'), 
            ('font-weight', 'bold'),
            ('text-align', 'center'),
            ('border', '1px solid #4d4d4d')
        ]},
        {'selector': 'td', 'props': [
            ('background-color', '#1e1e1e'), 
            ('color', '#ffffff'), 
            ('border', '1px solid #3d3d3d'),
            ('text-align', 'center')
        ]},
        {'selector': 'table', 'props': [
            ('border-collapse', 'collapse'),
            ('width', '100%')
        ]},
        {'selector': 'tr:hover td', 'props': [
            ('background-color', '#3f3f3f')
        ]}
    ])

def create_theme_dataframe(symbols, historical, price_data, latest_date):
    """Create dataframe for theme analysis"""
    metrics_list = []
    for symbol in symbols:
        hist = historical[symbol]
        if hist and hist[-1]['date'].strftime('%Y%m%d') == latest_date:
            metrics = hist[-1].copy()
            metrics['Symbol'] = symbol
            past = hist[:-1]
            dev_b_5 = np.nan
            dev_s_5 = np.nan
            if len(past) >= 5:
                avg_b_5 = np.mean([p['bought_volume'] for p in past[-5:]])
                avg_s_5 = np.mean([p['sold_volume'] for p in past[-5:]])
                if avg_b_5 > 0:
                    dev_b_5 = round(((metrics['bought_volume'] - avg_b_5) / avg_b_5 * 100), 0)
                if avg_s_5 > 0:
                    dev_s_5 = round(((metrics['sold_volume'] - avg_s_5) / avg_s_5 * 100), 0)
                
                # Calculate additional metrics
                historical_volumes = [p['total_volume'] for p in past[-5:]]
                historical_ratios = [p['buy_to_sell_ratio'] for p in past[-3:]]
                
                metrics['volume_strength'] = calculate_volume_strength(metrics['total_volume'], historical_volumes)
                metrics['trend_3d'] = get_trend_direction(historical_ratios)
                
                # Calculate momentum score (0-1 scale) using past data
                metrics['momentum_score'] = calculate_momentum_score(past[-3:]) if len(past) >= 3 else 0.0
                
                # Calculate volume percentile over 10 days
                ten_day_volumes = [p['total_volume'] for p in past[-10:] if p.get('total_volume')]
                metrics['volume_percentile'] = get_volume_percentile(metrics['total_volume'], ten_day_volumes)
                
                # Calculate consistency score
                past_10_days = [p for p in past[-10:]]
                past_10_ratios = [p['buy_to_sell_ratio'] for p in past_10_days if 'buy_to_sell_ratio' in p]
                bullish_days, bearish_days = calculate_consistency_score(past_10_ratios)
                metrics['bullish_days_10'] = bullish_days
                metrics['bearish_days_10'] = bearish_days
            else:
                metrics['volume_strength'] = 1.0
                metrics['trend_3d'] = "-"
                metrics['momentum_score'] = 0.0
                metrics['volume_percentile'] = 50.0
                metrics['bullish_days_10'] = 0
                metrics['bearish_days_10'] = 0
            
            # Add price data
            if symbol in price_data:
                metrics['current_price'] = price_data[symbol]['current_price']
                metrics['price_change_1d'] = price_data[symbol]['change_1d']
            else:
                metrics['current_price'] = 0.0
                metrics['price_change_1d'] = 0.0
            
            metrics['dev_b_5'] = dev_b_5
            metrics['dev_s_5'] = dev_s_5
            metrics_list.append(metrics)
    
    theme_df = pd.DataFrame(metrics_list)
    theme_df = theme_df.sort_values(by=['buy_to_sell_ratio'], ascending=False)
    return theme_df

def run():
    st.markdown("""
        <style>
        /* Dark theme styling */
        .stApp {
            background-color: #1e1e1e;
            color: #ffffff;
        }
        
        /* Button styling */
        .stButton>button {
            background-color: #22c55e;
            color: white;
            border-radius: 5px;
            padding: 8px 16px;
            border: none;
        }
        
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            background-color: #2d2d2d;
        }
        
        .stTabs [data-baseweb="tab"] {
            background-color: #2d2d2d;
            color: #ffffff;
            font-size: 16px;
            padding: 10px;
            border-radius: 5px 5px 0 0;
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            background-color: #3d3d3d;
        }
        
        /* Input styling */
        .stSelectbox > div > div {
            background-color: #2d2d2d;
            color: #ffffff;
        }
        
        .stTextInput > div > div > input {
            background-color: #2d2d2d;
            color: #ffffff;
            border: 1px solid #4d4d4d;
        }
        
        /* Dataframe styling */
        .stDataFrame {
            font-size: 13px;
            background-color: #1e1e1e;
        }
        
        /* Metric styling */
        .metric-container {
            background-color: #2d2d2d;
            padding: 10px;
            border-radius: 5px;
            margin: 5px;
        }
        
        /* Header styling */
        h1, h2, h3 {
            color: #ffffff !important;
        }
        
        /* Sidebar styling */
        .css-1d391kg {
            background-color: #2d2d2d;
        }
        </style>
        """, unsafe_allow_html=True)
    
    #st.set_page_config(layout="wide", page_title="Dark Pool Analysis")
    #st.title("📊 Dark Pool Analysis")
    
    # Create tabs
    tabs = st.tabs(["Single Stock", "High Bought Stocks", "High Sold Stocks", "Watchlist Summary"])
    
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
                    display_df['BOT %'] = (display_df['bought_volume'] / display_df['total_volume'] * 100).round(0).astype(int).apply(lambda x: f"{x}%")
                    display_df['Signal'] = display_df['buy_to_sell_ratio'].apply(get_signal)
                    display_df['Bought Dev 5d'] = display_df['dev_b_5'].apply(lambda x: f"{x:+.0f}%" if pd.notnull(x) else "-")
                    display_df['Sold Dev 5d'] = display_df['dev_s_5'].apply(lambda x: f"{x:+.0f}%" if pd.notnull(x) else "-")
                    for col in ['bought_volume', 'sold_volume', 'total_volume']:
                        display_df[col] = display_df[col].astype(int).apply(lambda x: f"{x:,.0f}")
                    display_df['buy_to_sell_ratio'] = display_df['buy_to_sell_ratio'].round(2)
                    display_df['date'] = display_df['date'].dt.strftime('%Y%m%d')
                    columns = ['date', 'bought_volume', 'sold_volume', 'BOT %', 'buy_to_sell_ratio', 'Signal', 'total_volume', 'Bought Dev 5d', 'Sold Dev 5d']
                    
                    styled_df = display_df[columns].style.applymap(
                        style_signal_dark, subset=['Signal']
                    ).applymap(
                        style_dev_dark, subset=['Bought Dev 5d', 'Sold Dev 5d']
                    ).applymap(
                        style_bot_percentage, subset=['BOT %']
                    ).set_table_styles([
                        {'selector': 'th', 'props': [('background-color', '#2d2d2d'), 
                                                    ('color', '#ffffff'), 
                                                    ('font-weight', 'bold')]},
                        {'selector': 'td', 'props': [('background-color', '#1e1e1e'), 
                                                    ('color', '#ffffff'), 
                                                    ('border', '1px solid #3d3d3d')]},
                        {'selector': 'table', 'props': [('border-collapse', 'collapse')]}
                    ])
                    
                    st.dataframe(styled_df, use_container_width=True)
                else:
                    st.write(f"No data available for {symbol}.")
    
    # High Bought Stocks Tab
    with tabs[1]:
        st.subheader("🟢 High Bought Stocks (Bought > 2x Sold)")
        if st.button("Generate High Bought Analysis", key="high_bought"):
            with st.spinner("Analyzing high bought stocks..."):
                high_buy_df, high_sell_df, latest_date = generate_stock_summary()
                if latest_date:
                    st.markdown(f"**📅 Data analyzed for:** `{latest_date}`")
                
                if not high_buy_df.empty:
                    # Enhanced summary metrics for high bought stocks
                    st.subheader("📈 High Bought Overview")
                    col1, col2, col3, col4, col5 = st.columns(5)
                    col1.metric("High Bought Stocks", len(high_buy_df))
                    col2.metric("Avg Buy/Sell Ratio", f"{high_buy_df['buy_to_sell_ratio'].mean():.2f}")
                    col3.metric("Avg Price Change", f"{high_buy_df['price_change_1d'].mean():.1f}%")
                    col4.metric("High Volume (>75th %)", len(high_buy_df[high_buy_df['volume_percentile'] > 75]))
                    col5.metric("Strong Momentum", len(high_buy_df[high_buy_df['momentum_score'] >= 0.5]))
                    
                    # Add filters
                    st.subheader("🔍 Filters")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        min_ratio = st.slider("Min Buy/Sell Ratio", 0.5, 5.0, 2.0, key="bought_ratio")
                    with col2:
                        min_volume = st.number_input("Min Volume (M)", 0, 100, 1, key="bought_volume") * 1_000_000
                    with col3:
                        min_consistency = st.slider("Min Bullish Days (10d)", 0, 10, 3, key="bought_consistency")
                    with col4:
                        momentum_filter = st.selectbox("Momentum Filter", ["All", "Strong Only", "Good+"], key="bought_momentum")
                    
                    # Apply filters
                    filtered_df = high_buy_df.copy()
                    filtered_df = filtered_df[filtered_df['buy_to_sell_ratio'] >= min_ratio]
                    filtered_df = filtered_df[filtered_df['total_volume'] >= min_volume]
                    filtered_df = filtered_df[filtered_df['bullish_days_10'] >= min_consistency]
                    
                    if momentum_filter == "Strong Only":
                        filtered_df = filtered_df[filtered_df['momentum_score'] >= 0.8]
                    elif momentum_filter == "Good+":
                        filtered_df = filtered_df[filtered_df['momentum_score'] >= 0.5]
                    
                    st.subheader(f"📊 Results ({len(filtered_df)} stocks)")
                    if not filtered_df.empty:
                        display_df = format_enhanced_dataframe(filtered_df, focus_type="bought")
                        columns = ['Symbol', 'Theme', 'Current Price', 'Price Change', 'bought_volume', 'sold_volume', 'BOT %', 'buy_to_sell_ratio', 'Signal', 'Volume Strength', 'Bought Dev 5d', 'Sold Dev 5d', 'Volume Rank', 'Momentum', 'Consistency', 'Trend 3D']
                        
                        styled_df = create_enhanced_styled_dataframe(display_df, columns, "bought")
                        st.dataframe(styled_df, use_container_width=True)
                        
                        # Enhanced insights
                        st.subheader("💡 Key Insights")
                        
                        # Top performers in different categories
                        col1, col2, col3, col4 = st.columns(4)
                        
                        top_ratio = filtered_df.nlargest(1, 'buy_to_sell_ratio')
                        if not top_ratio.empty:
                            col1.info(f"🏆 **Highest Ratio:** {top_ratio.iloc[0]['Symbol']} ({top_ratio.iloc[0]['buy_to_sell_ratio']:.2f})")
                        
                        top_momentum = filtered_df.nlargest(1, 'momentum_score')
                        if not top_momentum.empty:
                            col2.info(f"🚀 **Best Momentum:** {top_momentum.iloc[0]['Symbol']} ({top_momentum.iloc[0]['momentum_score']:.2f})")
                            
                        top_consistency = filtered_df.nlargest(1, 'bullish_days_10')
                        if not top_consistency.empty:
                            col3.info(f"� **Most Consistent:** {top_consistency.iloc[0]['Symbol']} ({top_consistency.iloc[0]['bullish_days_10']}/10)")
                            
                        top_volume_rank = filtered_df.nlargest(1, 'volume_percentile')
                        if not top_volume_rank.empty:
                            col4.info(f"📊 **Highest Volume Rank:** {top_volume_rank.iloc[0]['Symbol']} ({top_volume_rank.iloc[0]['volume_percentile']}th %ile)")
                        
                        # Additional trading insights
                        st.subheader("🎯 Trading Insights")
                        
                        # Strong setups (high ratio + good momentum + consistency)
                        strong_setups = filtered_df[
                            (filtered_df['buy_to_sell_ratio'] > 3.0) & 
                            (filtered_df['momentum_score'] >= 0.5) & 
                            (filtered_df['bullish_days_10'] >= 5)
                        ]
                        
                        # Momentum breakouts (recent momentum + high volume rank)
                        momentum_breakouts = filtered_df[
                            (filtered_df['momentum_score'] >= 0.8) & 
                            (filtered_df['volume_percentile'] >= 80)
                        ]
                        
                        # Consistent accumulation (high consistency but moderate ratio)
                        steady_accumulation = filtered_df[
                            (filtered_df['bullish_days_10'] >= 7) & 
                            (filtered_df['buy_to_sell_ratio'].between(2.0, 4.0))
                        ]
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown("**� Strong Setups**")
                            if not strong_setups.empty:
                                for _, row in strong_setups.head(3).iterrows():
                                    st.success(f"{row['Symbol']}: {row['buy_to_sell_ratio']:.1f}x ratio, {row['momentum_score']:.1f} momentum")
                            else:
                                st.info("No strong setups found")
                        
                        with col2:
                            st.markdown("**🚀 Momentum Breakouts**")
                            if not momentum_breakouts.empty:
                                for _, row in momentum_breakouts.head(3).iterrows():
                                    st.success(f"{row['Symbol']}: {row['volume_percentile']}th %ile volume, {row['momentum_score']:.1f} momentum")
                            else:
                                st.info("No momentum breakouts found")
                        
                        with col3:
                            st.markdown("**📈 Steady Accumulation**")
                            if not steady_accumulation.empty:
                                for _, row in steady_accumulation.head(3).iterrows():
                                    st.success(f"{row['Symbol']}: {row['bullish_days_10']}/10 days, {row['buy_to_sell_ratio']:.1f}x ratio")
                            else:
                                st.info("No steady accumulation found")
                    else:
                        st.info("No stocks match the current filters.")
                else:
                    st.info("No high bought stocks found for this date.")

    # High Sold Stocks Tab
    with tabs[2]:
        st.subheader("🔴 High Sold Stocks (Sold > 2x Bought)")
        if st.button("Generate High Sold Analysis", key="high_sold"):
            with st.spinner("Analyzing high sold stocks..."):
                high_buy_df, high_sell_df, latest_date = generate_stock_summary()
                if latest_date:
                    st.markdown(f"**📅 Data analyzed for:** `{latest_date}`")
                
                if not high_sell_df.empty:
                    # Enhanced summary metrics for high sold stocks
                    st.subheader("� High Sold Overview")
                    col1, col2, col3, col4, col5 = st.columns(5)
                    col1.metric("High Sold Stocks", len(high_sell_df))
                    col2.metric("Avg Buy/Sell Ratio", f"{high_sell_df['buy_to_sell_ratio'].mean():.2f}")
                    col3.metric("Avg Price Change", f"{high_sell_df['price_change_1d'].mean():.1f}%")
                    col4.metric("High Volume Count", len(high_sell_df[high_sell_df['volume_strength'] > 1.5]))
                    col5.metric("High Momentum", len(high_sell_df[high_sell_df['momentum_score'] > 0.6]))
                    
                    # Add filters
                    st.subheader("🔍 Filters")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        max_ratio = st.slider("Max Buy/Sell Ratio", 0.1, 1.0, 0.5, key="sold_ratio")
                    with col2:
                        min_volume_sold = st.number_input("Min Volume (M)", 0, 100, 1, key="sold_volume") * 1_000_000
                    with col3:
                        show_momentum_only_sold = st.checkbox("Show High Momentum Only", key="sold_momentum")
                    
                    # Apply filters
                    filtered_df = high_sell_df.copy()
                    filtered_df = filtered_df[filtered_df['buy_to_sell_ratio'] <= max_ratio]
                    filtered_df = filtered_df[filtered_df['total_volume'] >= min_volume_sold]
                    if show_momentum_only_sold:
                        filtered_df = filtered_df[filtered_df['momentum_score'] > 0.6]
                    
                    st.subheader(f"📊 Results ({len(filtered_df)} stocks)")
                    if not filtered_df.empty:
                        display_df = format_enhanced_dataframe(filtered_df, focus_type="sold")
                        columns = ['Symbol', 'Theme', 'Current Price', 'Price Change', 'bought_volume', 'sold_volume', 'BOT %', 'buy_to_sell_ratio', 'Signal', 'total_volume', 'Volume Strength', 'Bought Dev 5d', 'Sold Dev 5d', 'Trend 3D', 'Momentum', 'Volume Rank', 'Consistency']
                        
                        styled_df = create_enhanced_styled_dataframe(display_df, columns, focus_type="sold")
                        st.dataframe(styled_df, use_container_width=True)
                        
                        # Additional insights
                        st.subheader("💡 Key Insights")
                        lowest_ratio = filtered_df.nsmallest(1, 'buy_to_sell_ratio')
                        top_volume_sold = filtered_df.nlargest(1, 'total_volume')
                        top_momentum_sold = filtered_df.nlargest(1, 'momentum_score')
                        
                        col1, col2, col3 = st.columns(3)
                        if not lowest_ratio.empty:
                            col1.info(f"📉 **Lowest Ratio:** {lowest_ratio.iloc[0]['Symbol']} ({lowest_ratio.iloc[0]['buy_to_sell_ratio']:.2f})")
                        if not top_volume_sold.empty:
                            col2.info(f"📊 **Highest Volume:** {top_volume_sold.iloc[0]['Symbol']} ({top_volume_sold.iloc[0]['total_volume']:,.0f})")
                        if not top_momentum_sold.empty:
                            col3.info(f"� **Highest Momentum:** {top_momentum_sold.iloc[0]['Symbol']} ({top_momentum_sold.iloc[0]['momentum_score']:.2f})")
                    else:
                        st.info("No stocks match the current filters.")
                else:
                    st.info("No high sold stocks found for this date.")
    
    # Watchlist Summary Tab
    with tabs[3]:
        st.subheader("Watchlist Summary")
        selected_theme = st.selectbox("Select Watchlist (Theme)", list(theme_mapping.keys()), index=0)
        if st.button("Generate Watchlist Summary"):
            with st.spinner(f"Analyzing {selected_theme}..."):
                symbols = theme_mapping[selected_theme]
                price_data = get_price_data(symbols)
                historical = get_historical_metrics(symbols)
                latest_date = None
                for hist in historical.values():
                    if hist:
                        latest_date = max(latest_date, hist[-1]['date'].strftime('%Y%m%d')) if latest_date else hist[-1]['date'].strftime('%Y%m%d')
                if latest_date:
                    st.markdown(f"**📅 Data for:** `{latest_date}`")
                    theme_df = create_theme_dataframe(symbols, historical, price_data, latest_date)
                    
                    if not theme_df.empty:
                        total_buy = theme_df['bought_volume'].sum()
                        total_sell = theme_df['sold_volume'].sum()
                        aggregate_ratio = total_buy / total_sell if total_sell > 0 else float('inf')
                        
                        st.subheader("📊 Summary Metrics")
                        col1, col2, col3, col4, col5 = st.columns(5)
                        col1.metric("Total Bought", f"{total_buy:,.0f}")
                        col2.metric("Total Sold", f"{total_sell:,.0f}")
                        col3.metric("Aggregate Ratio", f"{aggregate_ratio:.2f}")
                        col4.metric("Avg Price Change", f"{theme_df['price_change_1d'].mean():.1f}%")
                        col5.metric("High Momentum", len(theme_df[theme_df['momentum_score'] > 0.6]))
                        
                        display_df = format_enhanced_dataframe(theme_df, focus_type="mixed")
                        columns = ['Symbol', 'Current Price', 'Price Change', 'bought_volume', 'sold_volume', 'BOT %', 'buy_to_sell_ratio', 'Signal', 'total_volume', 'Volume Strength', 'Bought Dev 5d', 'Sold Dev 5d', 'Trend 3D', 'Momentum', 'Volume Rank', 'Consistency']
                        
                        styled_df = create_enhanced_styled_dataframe(display_df, columns, focus_type="mixed")
                        st.dataframe(styled_df, use_container_width=True)
                else:
                    st.warning(f"No data available for {selected_theme}.")

# When imported as a module the `run()` function can be invoked from the host app.
# Do not auto-execute UI on import to allow embedding in a larger Streamlit app.
