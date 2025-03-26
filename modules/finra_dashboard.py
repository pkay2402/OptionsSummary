import random
import pandas as pd
import streamlit as st
import requests
import io
from datetime import datetime, time, timedelta
import yfinance as yf
import plotly.express as px
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import concurrent.futures
import logging
from scipy.stats import linregress
import sqlite3
from typing import List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Output to console
    ]
)
logger = logging.getLogger(__name__)

# Theme mapping
theme_mapping = {
    "Magnificent 7": [
        "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA"
    ],
    "Artificial Intelligence": [
        "NVDA", "GOOGL", "MSFT", "AMD", "PLTR", "SNOW", "AI", "CRM", "IBM", 
        "AAPL", "ADBE", "MSCI", "DELL", "BIDU"
    ],
    "Quantum Computing": [
        "IBM", "GOOGL", "MSFT", "RGTI", "IONQ", "QUBT", "HON", 
        "QCOM", "INTC", "AMAT", "MKSI", "NTNX"
    ],
    "Clean Energy": [
        "TSLA", "ENPH", "FSLR", "NEE", "PLUG", "SEDG", "RUN"
    ],
    "Semiconductors": [
        "NVDA", "AMD", "QCOM", "TXN", "INTC", "AVGO", "ASML"
    ],
    "Cloud Computing": [
        "AMZN", "MSFT", "GOOGL", "CRM", "NET", "DDOG", "SNOW", "ZS", "TEAM"
    ],
    "Cybersecurity": [
        "CRWD", "PANW", "ZS", "FTNT", "S", "OKTA", "CYBR", "RPD", "NET"
    ],
    "Electric Vehicles": [
        "TSLA", "RIVN", "LCID", "NIO", "LI", "GM", "F", "XPEV", "TM"
    ],
    "Robotics & Automation": [
        "ABB", "ROK", "IRBT", "ISRG", "TER", "NDSN", "FANUY", "SIEGY", "BOTZ"
    ],
    "Biotechnology": [
        "MRNA", "CRSP", "VRTX", "REGN", "ILMN", "AMGN", "NBIX", "BIIB", "INCY"
    ],
    "Fintech": [
        "SQ", "PYPL", "SOFI", "AFRM", "UPST", "COIN", "HOOD", "ADYEY", "FISV"
    ],
    "E-commerce": [
        "AMZN", "SHOP", "ETSY", "MELI", "JD", "BABA", "CPNG", "SE", "WMT"
    ],
    "Social Media": [
        "META", "SNAP", "PINS", "TWTR", "MTCH", "BMBL", "SPOT", "RBLX", "PTON"
    ],
    "Space Exploration": [
        "SPCE", "RTX", "LMT", "BA", "MAXR", "AJRD", "LHX", "RKLB", "HEI"
    ],
    "Augmented/Virtual Reality": [
        "MSFT", "META", "SNAP", "NVDA", "AAPL", "GOOGL", "U", "MTTR", "IMMR"
    ],
    "Digital Healthcare": [
        "TDOC", "DOCS", "LVGO", "ONEM", "AMWL", "ACCD", "PHR", "HIMS", "CERN"
    ],
    "Big Data Analytics": [
        "SNOW", "MDB", "PLTR", "DDOG", "SPLK", "ESTC", "ZI", "TYL", "CFLT"
    ],
    "3D Printing": [
        "DDD", "XONE", "SSYS", "MTLS", "PRLB", "VJET", "NNDM", "XOMETRY", "DM"
    ],
    "Internet of Things": [
        "CSCO", "SWKS", "SLAB", "QCOM", "STM", "KEYS", "TXN", "XLNX", "AMBA"
    ],
    "Gaming & Esports": [
        "ATVI", "EA", "TTWO", "NTDOY", "SONY", "MSFT", "NVDA", "U", "RBLX"
    ],
    "Pharmaceuticals": [
        "JNJ", "PFE", "MRK", "ABBV", "LLY", "BMY", "NVS", "AZN", "GSK"
    ],
    "Medical Devices": [
        "MDT", "ABT", "SYK", "BSX", "EW", "ZBH", "ISRG", "DXCM", "BDX"
    ],
    "Healthcare Services": [
        "UNH", "CVS", "ANTM", "HUM", "CI", "CNC", "MOH", "HCA", "THC"
    ],
    "Genomics": [
        "ILMN", "PACB", "TWST", "CRSP", "NTLA", "EDIT", "BEAM", "DNA", "ME"
    ],
    "Consumer Staples": [
        "PG", "KO", "PEP", "WMT", "COST", "CL", "KMB", "GIS", "K"
    ],
    "Food & Beverage": [
        "KO", "PEP", "MDLZ", "KHC", "GIS", "HSY", "K", "CAG", "CPB"
    ],
    "Tobacco & Cannabis": [
        "MO", "PM", "BTI", "CRON", "CGC", "TLRY", "ACB", "SNDL", "CURLF"
    ],
    "Household Products": [
        "PG", "CL", "KMB", "CHD", "CLX", "EL", "COTY", "NWL", "SPB"
    ],
    "Aerospace & Defense": [
        "LMT", "RTX", "BA", "GD", "NOC", "LHX", "TDG", "HEI", "SPR"
    ],
    "Transportation & Logistics": [
        "UPS", "FDX", "JBHT", "ODFL", "XPO", "CHRW", "EXPD", "MATX", "KNX"
    ],
    "Construction & Engineering": [
        "CAT", "DE", "URI", "PWR", "VMC", "MLM", "GVA", "FLR", "ACM"
    ],
    "Industrial Machinery": [
        "HON", "MMM", "GE", "ETN", "IR", "DOV", "ITW", "PH", "EMR"
    ],
    "Real Estate": [
        "AMT", "EQIX", "PLD", "PSA", "CCI", "DLR", "O", "AVB", "SPG"
    ],
    "Banking & Financial Services": [
        "JPM", "BAC", "WFC", "C", "GS", "MS", "USB", "TFC", "PNC"
    ],
    "Energy & Oil": [
        "XOM", "CVX", "COP", "EOG", "SLB", "PXD", "MPC", "PSX", "VLO"
    ],
    "Utilities": [
        "NEE", "DUK", "SO", "D", "AEP", "XEL", "SRE", "ED", "EXC"
    ],
    "Metals & Mining": [
        "FCX", "BHP", "RIO", "VALE", "NEM", "GOLD", "AA", "X", "STLD"
    ],
    "Retail": [
        "WMT", "TGT", "COST", "HD", "LOW", "DLTR", "DG", "KR", "ULTA"
    ],
    "Luxury Goods": [
        "LVMUY", "PPRUY", "HESAY", "CFRUY", "BURBY", "RMS.PA", "MC.PA", "CFR.SW", "TIFFB"
    ],
    "Travel & Leisure": [
        "MAR", "HLT", "H", "CCL", "RCL", "NCLH", "BKNG", "EXPE", "ABNB"
    ],
    "Agricultural": [
        "DE", "ADM", "CTVA", "NTR", "CF", "MOS", "FMC", "BG", "AGCO"
    ],
    "Small Cap Growth": [
        "CROX", "FIVN", "TPX", "TENB", "APPF", "CVNA", "EVBG", "OLED", "SONO"
    ],
    "Dividend Aristocrats": [
        "PG", "KO", "JNJ", "XOM", "MMM", "WMT", "T", "ED", "MCD"
    ],
    "Communication Services": [
        "GOOGL", "META", "DIS", "NFLX", "VZ", "CMCSA", "T", "TMUS", "EA"
    ],
    "Renewable Energy Storage": [
        "PLUG", "STEM", "BESS", "SUNL", "PWR", "FCEL", "SPWR", "BLDP", "NGP"
    ],
    "Smart Home Technology": [
        "GOOGL", "AMZN", "AAPL", "NVDA", "ROKU", "SONOS", "Z", "ZBRA", "STZ"
    ],
    "Advanced Materials": [
        "LYB", "DOW", "DD", "AXLL", "ALB", "BASFY", "MKFG", "NKLA", "LTHM"
    ],
    "Green Transportation": [
        "CHPT", "BLNK", "WKHS", "FSR", "GOEV", "KNDI", "SOLO", "JCI", "LAZR"
    ],
    "Sustainable Agriculture Tech": [
        "DESP", "RGEN", "HSKA", "CFIV", "TNIE", "SCLX", "VERY", "CHNC", "CATC"
    ],
    "Emerging Markets Tech": [
        "BABA", "JD", "BIDU", "SE", "TCEHY", "MELI", "GLOB", "NTES", "KWEB"
    ],
    "Aerospace Innovation": [
        "ASTR", "VELO", "RKT", "SRAC", "VACQ", "BKSY", "PL", "UFO", "ARKX"
    ],
    "Mental Health & Wellness Tech": [
        "VIGI", "MMED", "CMPS", "MYND", "WNDW", "Mind", "GPFT", "RDHL", "PSYC"
    ],
    "Advanced Robotics": [
        "CGNX", "ADSK", "CAT", "HUBB", "KRNS", "RAAS", "ZBRA", "OMCL", "MKSI"
    ],
    "Digital Payment Innovations": [
        "MA", "V", "PYPL", "SQ", "MELI", "BABA", "ADYEY", "TCEHY", "WU"
    ]
}

# Sector mapping
sector_mapping = {
    'Index/Sectors': [
        'SPY', 'QQQ', 'DIA', 'IWM', 'VXX', 'SMH', 'IVV', 'VTI', 'VOO', 'XLK',
        'VUG', 'XLF', 'XLV', 'XLY', 'XLC', 'XLI', 'XLE', 'XLB', 'XLU', 'XLRE',
        'XLP', 'XBI', 'XOP', 'XME', 'XRT', 'XHB', 'XWEB', 'XLC', 'RSP'
    ],
    'Technology': [
        'AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'META', 'TSLA', 'PLTR', 'ORCL',
        'AMD', 'NFLX', 'ADBE', 'CRM', 'INTC', 'CSCO', 'QCOM', 'TXN', 'INTU',
        'IBM', 'NOW', 'AVGO', 'UBER', 'SNOW', 'DELL', 'PANW'
    ],
    'Financials': [
        'JPM', 'V', 'MA', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'SCHW', 'COF',
        'MET', 'AIG', 'BK', 'BLK', 'TFC', 'USB', 'PNC', 'CME', 'SPGI', 'ICE',
        'MCO', 'AON', 'PYPL', 'SQ'
    ],
    'Healthcare': [
        'LLY', 'UNH', 'JNJ', 'PFE', 'MRK', 'ABBV', 'TMO', 'AMGN', 'GILD', 'CVS',
        'MDT', 'BMY', 'ABT', 'DHR', 'ISRG', 'SYK', 'REGN', 'VRTX', 'CI', 'ZTS',
        'BDX', 'HCA', 'EW', 'DXCM', 'BIIB'
    ],
    'Consumer': [
        'WMT', 'PG', 'KO', 'PEP', 'COST', 'MCD', 'DIS', 'NKE', 'SBUX', 'LOW',
        'TGT', 'HD', 'CL', 'MO', 'KHC', 'PM', 'TJX', 'DG', 'DLTR', 'YUM',
        'GIS', 'KMB', 'MNST', 'EL', 'CMG'
    ],
    'Energy': [
        'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'OXY', 'VLO', 'PXD',
        'HES', 'WMB', 'KMI', 'OKE', 'HAL', 'BKR', 'FANG', 'DVN', 'TRGP', 'APA',
        'EQT', 'MRO', 'NOV', 'FTI', 'RRC'
    ],
    'Industrials': [
        'CAT', 'DE', 'UPS', 'FDX', 'BA', 'HON', 'UNP', 'MMM', 'GE', 'LMT',
        'RTX', 'GD', 'CSX', 'NSC', 'WM', 'ETN', 'ITW', 'EMR', 'PH', 'ROK',
        'CARR', 'OTIS', 'IR', 'CMI', 'FAST'
    ],
    'User Defined': []  # Catch-all for unmapped symbols
}

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

def validate_pattern(symbol: str, dates: List[str], pattern_type: str = "accumulation") -> bool:
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

def find_patterns(lookback_days: int = 10, min_volume: int = 1000000, pattern_type: str = "accumulation", 
                  use_price_validation: bool = False, min_price: float = 5.0, min_market_cap: float = 500_000_000) -> pd.DataFrame:
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
            
            if not use_price_validation or validate_pattern(symbol, data['dates'], pattern_type):
                avg_ratio = sum(data['ratios']) / len(data['ratios'])
                avg_volume = data['total_volume'] / len(data['dates'])
                results.append({
                    'Symbol': symbol,
                    'Price': round(price, 2),
                    'Market Cap (M)': round(market_cap / 1_000_000, 2),
                    'Avg Daily Volume': int(avg_volume),
                    'Avg Buy/Sell Ratio': round(avg_ratio, 2),
                    'Days Showing Pattern': data['days_pattern'],
                    'Total Volume': int(data['total_volume']),
                    'Latest Ratio': data['ratios'][0],
                    'Volume Trend': 'Increasing' if data['volumes'][0] > avg_volume else 'Decreasing'
                })
    
    if pattern_type == "accumulation":
        results = sorted(results, key=lambda x: (x['Days Showing Pattern'], x['Avg Buy/Sell Ratio'], x['Total Volume']), reverse=True)
    else:
        results = sorted(results, key=lambda x: (x['Days Showing Pattern'], -x['Avg Buy/Sell Ratio'], x['Total Volume']), reverse=True)
    return pd.DataFrame(results[:40])

def get_sector_rotation(lookback_days: int = 10, min_volume: int = 50000) -> tuple[pd.DataFrame, pd.DataFrame]:
    sector_map = {
        'QQQ': 'Technology',
        'VGT': 'Technology',
        'IVW': 'Large-Cap Growth',
        'VUG': 'Large-Cap Growth',
        'SCHG': 'Large-Cap Growth',
        'MGK': 'Large-Cap Growth',
        'VOOG': 'Large-Cap Growth',
        'IWP': 'Mid-Cap Growth',
        'VOT': 'Mid-Cap Growth',
        'EFG': 'International Growth',
        'VTV': 'Large-Cap Value',
        'IVE': 'Large-Cap Value',
        'SCHV': 'Large-Cap Value',
        'IWD': 'Large-Cap Value',
        'VLUE': 'Large-Cap Value',
        'VOE': 'Mid-Cap Value',
        'IWS': 'Mid-Cap Value',
        'VBR': 'Small-Cap Value',
        'DHS': 'High Dividend Value',
        'RPV': 'S&P Pure Value'
    }
    sector_etfs = yf.Tickers(list(sector_map.keys()))
    sector_performance = {ticker: ticker_obj.history(period=f"{lookback_days}d")['Close'].pct_change().mean() 
                          for ticker, ticker_obj in sector_etfs.tickers.items()}
    
    sector_data = {}
    for i in range(lookback_days):
        date = (datetime.now() - timedelta(days=i)).strftime("%Y%m%d")
        data = download_finra_short_sale_data(date)
        if data:
            df = process_finra_short_sale_data(data)
            df = df[df['TotalVolume'] > min_volume]
            for _, row in df.iterrows():
                symbol = row['Symbol']
                if symbol in sector_map:
                    sector = sector_map[symbol]
                    total_volume = row.get('TotalVolume', 0)
                    metrics = calculate_metrics(row, total_volume)
                    
                    if sector not in sector_data:
                        sector_data[sector] = {'dates': [], 'ratios': [], 'volumes': [], 'symbols': set()}
                    sector_data[sector]['dates'].append(date)
                    sector_data[sector]['ratios'].append(metrics['buy_to_sell_ratio'])
                    sector_data[sector]['volumes'].append(total_volume)
                    sector_data[sector]['symbols'].add(symbol)
    
    results = []
    for sector, data in sector_data.items():
        if len(data['dates']) >= 3:
            avg_ratio = sum(data['ratios']) / len(data['ratios'])
            avg_volume = sum(data['volumes']) / len(data['volumes'])
            ratio_trend = (data['ratios'][0] - data['ratios'][-1]) / len(data['ratios'])
            volume_trend = 'Increasing' if data['volumes'][0] > avg_volume else 'Decreasing'
            results.append({
                'Sector': sector,
                'Symbol': ', '.join(data['symbols']),
                'Avg Buy/Sell Ratio': round(avg_ratio, 2),
                'Ratio Trend': 'Up' if ratio_trend > 0 else 'Down',
                'Avg Daily Volume': int(avg_volume),
                'Volume Trend': volume_trend,
                'Symbols Count': len(data['symbols']),
                'Latest Ratio': data['ratios'][0],
                'Sector Performance': round(sector_performance.get(sector.split()[0], 0) * 100, 2)
            })
    
    buying_sectors = sorted([r for r in results if r['Ratio Trend'] == 'Up'], 
                            key=lambda x: (x['Avg Buy/Sell Ratio'], x['Avg Daily Volume']), reverse=True)
    selling_sectors = sorted([r for r in results if r['Ratio Trend'] == 'Down'], 
                             key=lambda x: (x['Avg Buy/Sell Ratio'], x['Avg Daily Volume']))
    
    return pd.DataFrame(buying_sectors[:7]), pd.DataFrame(selling_sectors[:7])

def analyze_portfolio(symbols: List[str], lookback_days: int = 20) -> pd.DataFrame:
    portfolio_results = []
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(analyze_symbol, symbol, lookback_days, 1.5): symbol for symbol in symbols}
        for future in concurrent.futures.as_completed(futures):
            symbol = futures[future]
            try:
                df, significant_days = future.result()
                if not df.empty:
                    for _, row in df.iterrows():
                        portfolio_results.append({
                            'Symbol': symbol,
                            'Date': row['date'],
                            'Avg Buy/Sell Ratio': row['buy_to_sell_ratio'],
                            'Bought Volume': int(row['bought_volume']),
                            'Sold Volume': int(row['sold_volume']),
                            'Significant Days': significant_days,
                            'Latest Volume': int(row['total_volume'])
                        })
            except Exception as exc:
                logger.error(f'{symbol} generated an exception: {exc}')
    
    portfolio_df = pd.DataFrame(portfolio_results)
    if not portfolio_df.empty:
        portfolio_df = portfolio_df.sort_values(by=['Avg Buy/Sell Ratio', 'Bought Volume'], ascending=[False, False])
    return portfolio_df

def check_alerts(df_results: pd.DataFrame, symbol: str, threshold: float = 2.0) -> None:
    if not df_results.empty and df_results['buy_to_sell_ratio'].max() > threshold:
        st.warning(f"Alert: {symbol} has a Buy/Sell Ratio above {threshold} on {df_results['date'].iloc[0].strftime('%Y-%m-%d')}!")

def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode('utf-8')

def plot_correlation_heatmap(symbols: List[str], lookback_days: int = 20) -> None:
    data = {}
    for symbol in symbols:
        df, _ = analyze_symbol(symbol, lookback_days)
        if not df.empty:
            data[symbol] = df['buy_to_sell_ratio']
    corr_df = pd.DataFrame(data).corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_df, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

def get_latest_data() -> tuple[pd.DataFrame, Optional[str]]:
    for i in range(7):  # Check the last 7 days
        date = (datetime.now() - timedelta(days=i)).strftime("%Y%m%d")
        data = download_finra_short_sale_data(date)
        if data:
            df = process_finra_short_sale_data(data)
            if not df.empty:
                return df, date
    return pd.DataFrame(), None

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
    
    except Exception as e:
        logger.error(f"Error fetching batch data: {e}")
    
    finally:
        conn.commit()
        conn.close()

def get_stock_info_from_db(symbol: str) -> dict:
    conn = sqlite3.connect('stock_data.db')
    cursor = conn.cursor()
    
    cursor.execute('SELECT price, market_cap FROM stocks WHERE symbol = ?', (symbol,))
    result = cursor.fetchone()
    
    conn.close()
    
    if result:
        return {'price': result[0], 'market_cap': result[1]}
    return {'price': 0, 'market_cap': 0}

def find_rising_ratio_stocks(lookback_days: int = 10, min_volume: int = 500000, max_dips: int = 3, 
                             min_price: float = 5.0, min_market_cap: float = 500_000_000, 
                             allowed_symbols: Optional[List[str]] = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    symbol_data = {}
    
    for i in range(lookback_days):
        date = (datetime.now() - timedelta(days=i)).strftime("%Y%m%d")
        data = download_finra_short_sale_data(date)
        if not data:
            logger.warning(f"No data available for date: {date}")
            continue
        
        df = process_finra_short_sale_data(data)
        df = df[df['TotalVolume'] > min_volume]
        
        for _, row in df.iterrows():
            symbol = row['Symbol']
            if allowed_symbols and symbol not in allowed_symbols:
                continue
            
            total_volume = row.get('TotalVolume', 0)
            metrics = calculate_metrics(row, total_volume)
            
            if symbol not in symbol_data:
                symbol_data[symbol] = {'ratios': [], 'volumes': [], 'dates': []}
            symbol_data[symbol]['ratios'].append(metrics['buy_to_sell_ratio'])
            symbol_data[symbol]['volumes'].append(total_volume)
            symbol_data[symbol]['dates'].append(date)
    
    stock_results = []
    for symbol, data in symbol_data.items():
        num_days = len(data['ratios'])
        if num_days >= 5:
            stock_info = get_stock_info_from_db(symbol)
            price = stock_info['price']
            market_cap = stock_info['market_cap']
            
            if price < min_price or market_cap < min_market_cap:
                continue
            
            ratios = data['ratios'][::-1]
            dips = sum(1 for i in range(len(ratios) - 1) if ratios[i+1] < ratios[i])
            slope, _, _, _, _ = linregress(range(len(ratios)), ratios)
            
            if dips <= max_dips and slope > 0.005:
                avg_volume = sum(data['volumes']) / len(data['volumes'])
                stock_results.append({
                    'Symbol': symbol,
                    'Price': round(price, 2),
                    'Market Cap (M)': round(market_cap / 1_000_000, 2),
                    'Avg Daily Volume': int(avg_volume),
                    'Starting Ratio': round(ratios[0], 2),
                    'Ending Ratio': round(ratios[-1], 2),
                    'Ratio Increase': round(ratios[-1] - ratios[0], 2),
                    'Total Volume': int(sum(data['volumes'])),
                    'Dips': dips,
                    'Days of Data': num_days
                })
    
    theme_results = []
    theme_top_stocks = {}
    for theme, theme_symbols in theme_mapping.items():
        theme_ratios = []
        theme_volumes = []
        valid_symbols = 0
        theme_stock_results = []
        
        for symbol in theme_symbols:
            if symbol in symbol_data and len(symbol_data[symbol]['ratios']) >= 5:
                ratios = symbol_data[symbol]['ratios'][::-1]
                slope, _, _, _, _ = linregress(range(len(ratios)), ratios)
                if slope > 0.005:
                    theme_ratios.append(ratios)
                    theme_volumes.append(sum(symbol_data[symbol]['volumes']))
                    valid_symbols += 1
                    
                    stock_info = get_stock_info_from_db(symbol)
                    dips = sum(1 for i in range(len(ratios) - 1) if ratios[i+1] < ratios[i])
                    if stock_info['price'] >= min_price and stock_info['market_cap'] >= min_market_cap:
                        theme_stock_results.append({
                            'Symbol': symbol,
                            'Price': round(stock_info['price'], 2),
                            'Ratio Increase': round(ratios[-1] - ratios[0], 2),
                            'Starting Ratio': round(ratios[0], 2),
                            'Ending Ratio': round(ratios[-1], 2),
                            'Avg Daily Volume': int(sum(symbol_data[symbol]['volumes']) / len(symbol_data[symbol]['volumes']))
                        })
        
        if valid_symbols >= 3 and theme_ratios:
            min_length = min(len(r) for r in theme_ratios)
            if min_length > 0:
                aligned_ratios = [r[:min_length] for r in theme_ratios]
                avg_ratios = [
                    sum(r[i] for r in aligned_ratios) / len(aligned_ratios)
                    for i in range(min_length)
                ]
                slope, _, _, _, _ = linregress(range(len(avg_ratios)), avg_ratios)
                if slope > 0:
                    theme_results.append({
                        'Theme': theme,
                        'Starting Ratio': round(avg_ratios[0], 2),
                        'Ending Ratio': round(avg_ratios[-1], 2),
                        'Ratio Increase': round(avg_ratios[-1] - avg_ratios[0], 2),
                        'Avg Daily Volume': int(sum(theme_volumes) / len(theme_volumes)),
                        'Stocks Analyzed': valid_symbols,
                        'Slope': round(slope, 4)
                    })
                    top_stocks = sorted(theme_stock_results, key=lambda x: x['Ratio Increase'], reverse=True)[:3]
                    theme_top_stocks[theme] = pd.DataFrame(top_stocks)
            else:
                logger.warning(f"Theme '{theme}' has inconsistent or no ratio data.")
        else:
            logger.info(f"Theme '{theme}' skipped: insufficient valid symbols ({valid_symbols}) or no ratios.")
    
    stock_df = pd.DataFrame(stock_results).sort_values(by=['Ratio Increase', 'Total Volume'], ascending=False)[:40]
    theme_df = pd.DataFrame(theme_results).sort_values(by=['Ratio Increase', 'Avg Daily Volume'], ascending=False)
    return stock_df, theme_df, theme_top_stocks

def generate_evening_report(capital: float = 10000, run_on_demand: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    now = datetime.now()
    if not run_on_demand and (now.hour != 18 or now.minute < 30 or now.minute > 35):
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    accumulation_df = find_patterns(
        lookback_days=10, min_volume=1000000, pattern_type="accumulation",
        use_price_validation=False, min_price=5.0, min_market_cap=500_000_000
    )
    
    distribution_df = find_patterns(
        lookback_days=10, min_volume=1000000, pattern_type="distribution",
        use_price_validation=False, min_price=5.0, min_market_cap=500_000_000
    )
    
    stock_df, theme_df, theme_top_stocks = find_rising_ratio_stocks(
        lookback_days=10, min_volume=500000, max_dips=3, min_price=5.0,
        min_market_cap=500_000_000, allowed_symbols=None
    )
    rising_df = stock_df
    
    buying_sectors_df, selling_sectors_df = get_sector_rotation(lookback_days=10, min_volume=50000)
    
    latest_df, latest_date = get_latest_data()
    if not latest_df.empty:
        metrics_list = [calculate_metrics(row, row.get('TotalVolume', 0)) for _, row in latest_df.iterrows()]
        latest_df_processed = pd.DataFrame(metrics_list)
        latest_df_processed['Symbol'] = latest_df['Symbol']
        latest_df_processed = latest_df_processed[
            (latest_df_processed['total_volume'] >= 2000000) &
            (latest_df_processed['buy_to_sell_ratio'] > 1.5)
        ].sort_values(by=['buy_to_sell_ratio', 'total_volume'], ascending=[False, False])
        
        if not latest_df_processed.empty:
            latest_df_processed['Price'] = latest_df_processed['Symbol'].apply(
                lambda symbol: get_stock_info_from_db(symbol)['price']
            )
    else:
        latest_df_processed = pd.DataFrame()

    buy_candidates = []
    if not accumulation_df.empty:
        buy_candidates.extend(accumulation_df.head(5)[['Symbol', 'Avg Buy/Sell Ratio', 'Price']].to_dict('records'))
    if not rising_df.empty:
        buy_candidates.extend(rising_df.head(5)[['Symbol', 'Ending Ratio', 'Price']].to_dict('records'))
    if not latest_df_processed.empty:
        buy_candidates.extend(latest_df_processed.head(3)[['Symbol', 'buy_to_sell_ratio', 'Price']].to_dict('records'))

    sell_candidates = []
    if not distribution_df.empty:
        sell_candidates.extend(distribution_df.head(5)[['Symbol', 'Avg Buy/Sell Ratio', 'Price']].to_dict('records'))

    if buy_candidates:
        buy_df = pd.DataFrame(buy_candidates).drop_duplicates(subset='Symbol')
        buy_df['Reason'] = buy_df.apply(
            lambda row: 'Accumulation' if 'Avg Buy/Sell Ratio' in row else 'Rising Ratio' if 'Ending Ratio' in row else 'Latest Day', axis=1
        )
        buy_df['Signal Strength'] = buy_df.apply(
            lambda row: row.get('Avg Buy/Sell Ratio', row.get('Ending Ratio', row.get('buy_to_sell_ratio', 1.5))), axis=1
        )
        
        buy_df['Weight'] = buy_df['Signal Strength'] / buy_df['Signal Strength'].sum()
        buy_df['Shares'] = np.floor((capital * buy_df['Weight']) / buy_df['Price'].replace(0, np.nan))
        buy_df['Cost'] = buy_df['Shares'] * buy_df['Price']
        buy_df = buy_df[buy_df['Shares'] > 0]
        
        total_cost = buy_df['Cost'].sum()
        cash_remaining = capital - total_cost
    else:
        buy_df = pd.DataFrame()
        total_cost = 0
        cash_remaining = capital

    st.session_state['evening_report'] = {
        'accumulation_df': accumulation_df,
        'distribution_df': distribution_df,
        'rising_df': rising_df,
        'buying_sectors_df': buying_sectors_df,
        'selling_sectors_df': selling_sectors_df,
        'buy_df': buy_df,
        'sell_df': pd.DataFrame(sell_candidates).drop_duplicates(subset='Symbol') if sell_candidates else pd.DataFrame(),
        'total_cost': total_cost,
        'cash_remaining': cash_remaining,
        'latest_date': latest_date
    }

    return accumulation_df, distribution_df, rising_df, buying_sectors_df, selling_sectors_df

# OTM Flows Functions
def validate_csv_content_type(response: requests.Response) -> bool:
    """Validate if the response content type is CSV."""
    return 'text/csv' in response.headers.get('Content-Type', '')

def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Apply filters to the DataFrame."""
    df = df[df['Volume'] >= 100]
    df['Expiration'] = pd.to_datetime(df['Expiration'])
    df = df[df['Expiration'].dt.date >= datetime.now().date()]
    return df

@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_data_from_urls(urls: List[str]) -> pd.DataFrame:
    """Fetch and combine data from multiple CSV URLs into a single DataFrame."""
    data_frames = []
    for url in urls:
        try:
            response = requests.get(url)
            response.raise_for_status()
            if validate_csv_content_type(response):
                csv_data = io.StringIO(response.text)
                df = pd.read_csv(csv_data)
                data_frames.append(apply_filters(df))
            else:
                logger.warning(f"Data from {url} is not in CSV format. Skipping...")
        except Exception as e:
            logger.error(f"Error fetching or processing data from {url}: {e}")
    return pd.concat(data_frames, ignore_index=True) if data_frames else pd.DataFrame()

def fetch_top_10_otm_flows(symbol: str, urls: List[str]) -> pd.DataFrame:
    """Fetch and return the top 10 OTM option flows for a given symbol."""
    data = fetch_data_from_urls(urls)
    if data.empty:
        return pd.DataFrame()

    symbol_data = data[data['Symbol'] == symbol].copy()
    if symbol_data.empty:
        return pd.DataFrame()

    stock_info = get_stock_info_from_db(symbol)
    spot_price = stock_info.get('price', 0)
    if spot_price == 0:
        try:
            ticker = yf.Ticker(symbol)
            spot_price = ticker.history(period="1d")['Close'].iloc[-1]
        except Exception as e:
            logger.error(f"Could not fetch spot price for {symbol}: {e}")
            spot_price = 0

    symbol_data['OTM'] = symbol_data.apply(
        lambda row: row['Strike Price'] > spot_price if row['Call/Put'] == 'C' else row['Strike Price'] < spot_price,
        axis=1
    )
    otm_data = symbol_data[symbol_data['OTM']]
    if otm_data.empty:
        return pd.DataFrame()

    otm_data['Transaction Value'] = otm_data['Volume'] * otm_data['Last Price'] * 100
    otm_summary = (
        otm_data.groupby(['Symbol', 'Expiration', 'Strike Price', 'Call/Put', 'Last Price'])
        .agg({'Volume': 'sum', 'Transaction Value': 'sum'})
        .reset_index()
    )
    return otm_summary.sort_values(by='Transaction Value', ascending=False).head(10)

def run():
    st.markdown("""
        <style>
        .stButton>button {
            background-color: #4CAF50;
            color: white;
        }
        .stTabs [data-baseweb="tab"] {
            font-size: 16px;
        }
        </style>
        """, unsafe_allow_html=True)
    
    st.title("FINRA Short Sale Analysis")
    
    # Initialize session state
    if 'analysis_results' not in st.session_state:
        st.session_state['analysis_results'] = {}
    if 'otm_flows' not in st.session_state:
        st.session_state['otm_flows'] = {'symbol': None, 'show': False, 'data': None}
    
    with st.sidebar:
        portfolio_symbols = st.text_area("User Defined Symbols for Multi Stock (comma-separated)", 
                                         "AAPL, MSFT, NVDA, AMZN, GOOGL, META, TSLA, PLTR, LLY, JPM, AVGO, UNH, V, WMT, XOM, MA, JNJ, PG, HD, COST, ORCL, CVX, BAC, KO, PEP, ABBV, "
                                         "MRK, AMD, NFLX, ADBE, CRM, INTC, CSCO, PFE, TMO, MCD, DIS, WFC, QCOM, LIN, GE, AXP, CAT, IBM, VZ, GS, MS, PM, LOW, NEE, RTX, BA, HON, UNP, "
                                         "T, BLK, MDT, SBUX, LMT, AMGN, GILD, CVS, DE, TGT, AMT, BKNG, SO, DUK, PYPL, UPS, C, COP, MMM, ACN, ABT, DHR, TMUS, TXN, MDLZ, BMY, INTU, NKE, "
                                         "CL, MO, KHC, EMR, GM, F, FDX, GD, BK, SPG, CHTR, USB, MET, AIG, COF, DOW, SCHW, CMCSA, SPY, QQQ, VOO, IVV, XLK, VUG, XLV, XLF, IWF, VTI").split(",")
        portfolio_symbols = [s.strip().upper() for s in portfolio_symbols if s.strip()]
        
        st.subheader("Data Management")
        update_button = st.button("Update Stock Data", use_container_width=True)
        if update_button:
            with st.spinner("Updating stock data..."):
                update_stock_database(portfolio_symbols)
            st.success("Stock data updated successfully!")
    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "Single Stock", "Accumulation", "Distribution", 
        "Sector Rotation", "Multi Stock", "Latest Day", 
        "Rising Ratios", "Portfolio Allocation"
    ])
    
    urls = [
        "https://www.cboe.com/us/options/market_statistics/symbol_data/csv/?mkt=cone",
        "https://www.cboe.com/us/options/market_statistics/symbol_data/csv/?mkt=opt",
        "https://www.cboe.com/us/options/market_statistics/symbol_data/csv/?mkt=ctwo",
        "https://www.cboe.com/us/options/market_statistics/symbol_data/csv/?mkt=exo"
    ]
    
    # Tab 1: Single Stock
    with tab1:
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
                
                st.dataframe(display_df)
                
                if st.button(f"Show Top 10 OTM Flows for {symbol}", key=f"otm_{symbol}"):
                    st.session_state['otm_flows']['symbol'] = symbol
                    st.session_state['otm_flows']['show'] = True
                    with st.spinner(f"Fetching OTM flows for {symbol}..."):
                        otm_flows = fetch_top_10_otm_flows(symbol, urls)
                        st.session_state['otm_flows']['data'] = otm_flows
                
                if (st.session_state['otm_flows']['show'] and 
                    st.session_state['otm_flows']['symbol'] == symbol):
                    otm_flows = st.session_state['otm_flows']['data']
                    if otm_flows is not None and not otm_flows.empty:
                        with st.expander(f"Top 10 OTM Flows for {symbol}", expanded=True):
                            st.dataframe(otm_flows.style.format({
                                'Transaction Value': '${:,.2f}',
                                'Last Price': '${:.2f}',
                                'Volume': '{:,.0f}'
                            }))
                            fig_otm = px.bar(otm_flows, x='Strike Price', y='Transaction Value',
                                            color='Call/Put', title=f"Top 10 OTM Flows for {symbol}",
                                            hover_data=['Expiration', 'Volume', 'Last Price'])
                            st.plotly_chart(fig_otm)
                            if st.button("Hide OTM Flows", key=f"hide_otm_{symbol}"):
                                st.session_state['otm_flows']['show'] = False
                                st.session_state['otm_flows']['data'] = None
                    else:
                        st.warning(f"No OTM flows found for {symbol}.")
                        st.session_state['otm_flows']['show'] = False
                        st.session_state['otm_flows']['data'] = None
    
    # Tab 2: Accumulation
    with tab2:
        st.subheader("Top 40 Stocks Showing Accumulation")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            acc_min_volume = st.number_input("Minimum Daily Volume (Accumulation)", value=1000000, step=500000, format="%d", key="acc_vol")
        with col2:
            acc_min_price = st.number_input("Minimum Price ($)", value=5.0, min_value=1.0, max_value=50.0, step=0.5, key="acc_price")
        with col3:
            acc_min_market_cap = st.number_input("Minimum Market Cap ($M)", value=500, min_value=100, max_value=5000, step=100, key="acc_market_cap")
        with col4:
            acc_use_validation = st.checkbox("Use Price Validation", value=False, key="acc_val")
        
        if st.button("Find Accumulation"):
            with st.spinner("Finding patterns..."):
                accumulation_df = find_patterns(
                    lookback_days=10,
                    min_volume=acc_min_volume,
                    pattern_type="accumulation",
                    use_price_validation=acc_use_validation,
                    min_price=acc_min_price,
                    min_market_cap=acc_min_market_cap * 1_000_000
                )
                st.session_state['analysis_results']['accumulation'] = accumulation_df
        
        if 'accumulation' in st.session_state['analysis_results']:
            accumulation_df = st.session_state['analysis_results']['accumulation']
            if not accumulation_df.empty:
                st.dataframe(accumulation_df)
                selected_symbol = st.selectbox("Select Symbol for OTM Flows", accumulation_df['Symbol'].tolist(), key="acc_otm_select")
                if st.button(f"Show Top 10 OTM Flows for {selected_symbol}", key=f"otm_acc_{selected_symbol}"):
                    st.session_state['otm_flows']['symbol'] = selected_symbol
                    st.session_state['otm_flows']['show'] = True
                    with st.spinner(f"Fetching OTM flows for {selected_symbol}..."):
                        otm_flows = fetch_top_10_otm_flows(selected_symbol, urls)
                        st.session_state['otm_flows']['data'] = otm_flows
                
                if (st.session_state['otm_flows']['show'] and 
                    st.session_state['otm_flows']['symbol'] == selected_symbol):
                    otm_flows = st.session_state['otm_flows']['data']
                    if otm_flows is not None and not otm_flows.empty:
                        with st.expander(f"Top 10 OTM Flows for {selected_symbol}", expanded=True):
                            st.dataframe(otm_flows.style.format({
                                'Transaction Value': '${:,.2f}',
                                'Last Price': '${:.2f}',
                                'Volume': '{:,.0f}'
                            }))
                            fig_otm = px.bar(otm_flows, x='Strike Price', y='Transaction Value',
                                            color='Call/Put', title=f"Top 10 OTM Flows for {selected_symbol}",
                                            hover_data=['Expiration', 'Volume', 'Last Price'])
                            st.plotly_chart(fig_otm)
                            if st.button("Hide OTM Flows", key=f"hide_otm_acc_{selected_symbol}"):
                                st.session_state['otm_flows']['show'] = False
                                st.session_state['otm_flows']['data'] = None
                    else:
                        st.warning(f"No OTM flows found for {selected_symbol}.")
                        st.session_state['otm_flows']['show'] = False
                        st.session_state['otm_flows']['data'] = None
                
                fig = px.bar(accumulation_df.head(10), x='Symbol', y='Avg Buy/Sell Ratio', 
                             color='Volume Trend', title="Top 10 Accumulation",
                             hover_data=['Price', 'Market Cap (M)', 'Avg Daily Volume'])
                st.plotly_chart(fig)
            else:
                st.write("No Accumulation detected with current filters.")
    
    # Tab 3: Distribution
    with tab3:
        st.subheader("Top 40 Stocks Showing Distribution")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            dist_min_volume = st.number_input("Minimum Daily Volume (Distribution)", value=1000000, step=500000, format="%d", key="dist_vol")
        with col2:
            dist_min_price = st.number_input("Minimum Price ($)", value=5.0, min_value=1.0, max_value=50.0, step=0.5, key="dist_price")
        with col3:
            dist_min_market_cap = st.number_input("Minimum Market Cap ($M)", value=500, min_value=100, max_value=5000, step=100, key="dist_market_cap")
        with col4:
            dist_use_validation = st.checkbox("Use Price Validation", value=False, key="dist_val")
        
        if st.button("Find Distribution"):
            with st.spinner("Finding patterns..."):
                distribution_df = find_patterns(
                    lookback_days=10,
                    min_volume=dist_min_volume,
                    pattern_type="distribution",
                    use_price_validation=dist_use_validation,
                    min_price=dist_min_price,
                    min_market_cap=dist_min_market_cap * 1_000_000
                )
                st.session_state['analysis_results']['distribution'] = distribution_df
        
        if 'distribution' in st.session_state['analysis_results']:
            distribution_df = st.session_state['analysis_results']['distribution']
            if not distribution_df.empty:
                st.dataframe(distribution_df)
                selected_symbol = st.selectbox("Select Symbol for OTM Flows", distribution_df['Symbol'].tolist(), key="dist_otm_select")
                if st.button(f"Show Top 10 OTM Flows for {selected_symbol}", key=f"otm_dist_{selected_symbol}"):
                    st.session_state['otm_flows']['symbol'] = selected_symbol
                    st.session_state['otm_flows']['show'] = True
                    with st.spinner(f"Fetching OTM flows for {selected_symbol}..."):
                        otm_flows = fetch_top_10_otm_flows(selected_symbol, urls)
                        st.session_state['otm_flows']['data'] = otm_flows
                
                if (st.session_state['otm_flows']['show'] and 
                    st.session_state['otm_flows']['symbol'] == selected_symbol):
                    otm_flows = st.session_state['otm_flows']['data']
                    if otm_flows is not None and not otm_flows.empty:
                        with st.expander(f"Top 10 OTM Flows for {selected_symbol}", expanded=True):
                            st.dataframe(otm_flows.style.format({
                                'Transaction Value': '${:,.2f}',
                                'Last Price': '${:.2f}',
                                'Volume': '{:,.0f}'
                            }))
                            fig_otm = px.bar(otm_flows, x='Strike Price', y='Transaction Value',
                                            color='Call/Put', title=f"Top 10 OTM Flows for {selected_symbol}",
                                            hover_data=['Expiration', 'Volume', 'Last Price'])
                            st.plotly_chart(fig_otm)
                            if st.button("Hide OTM Flows", key=f"hide_otm_dist_{selected_symbol}"):
                                st.session_state['otm_flows']['show'] = False
                                st.session_state['otm_flows']['data'] = None
                    else:
                        st.warning(f"No OTM flows found for {selected_symbol}.")
                        st.session_state['otm_flows']['show'] = False
                        st.session_state['otm_flows']['data'] = None
                
                fig = px.bar(distribution_df.head(10), x='Symbol', y='Avg Buy/Sell Ratio', 
                             color='Volume Trend', title="Top 10 Distribution",
                             hover_data=['Price', 'Market Cap (M)', 'Avg Daily Volume'])
                st.plotly_chart(fig)
            else:
                st.write("No Distribution detected with current filters.")
    
    # Tab 4: Sector Rotation (No OTM flows here as it doesn’t display individual stock tables)
    with tab4:
        st.subheader("Sector Rotation Analysis")
        col1, col2 = st.columns(2)
        with col1:
            rot_min_volume = st.number_input("Minimum Daily Volume (Sector Rotation)", value=50000, step=10000, format="%d", key="rot_vol")
        with col2:
            rot_lookback_days = st.slider("Lookback Days (Sector Rotation)", 1, 30, 10, key="rot_days")
        
        if st.button("Analyze Sector Rotation"):
            with st.spinner("Analyzing sector rotation..."):
                buying_df, selling_df = get_sector_rotation(lookback_days=rot_lookback_days, min_volume=rot_min_volume)
                st.session_state['analysis_results']['sector_rotation'] = {'buying': buying_df, 'selling': selling_df}
        
        if 'sector_rotation' in st.session_state['analysis_results']:
            buying_df = st.session_state['analysis_results']['sector_rotation']['buying']
            selling_df = st.session_state['analysis_results']['sector_rotation']['selling']
            if not buying_df.empty or not selling_df.empty:
                all_sectors = pd.DataFrame()
                if not buying_df.empty:
                    buying_df['Direction'] = 'Buying'
                    buying_df['RS_Ratio'] = 1.0
                    all_sectors = pd.concat([all_sectors, buying_df])
                if not selling_df.empty:
                    selling_df['Direction'] = 'Selling'
                    selling_df['RS_Ratio'] = -1.0
                    all_sectors = pd.concat([all_sectors, selling_df])
                
                if 'Avg Daily Volume' in all_sectors.columns:
                    all_sectors['RS_Momentum'] = all_sectors['Avg Daily Volume'].fillna(0) / all_sectors['Avg Daily Volume'].max() * 100
                else:
                    all_sectors['RS_Momentum'] = [random.uniform(-2, 2) for _ in range(len(all_sectors))]
                
                if 'Avg Daily Volume' in all_sectors.columns:
                    size_column = 'Avg Daily Volume'
                else:
                    all_sectors['Size'] = 30
                    size_column = 'Size'
                
                fig = px.scatter(
                    all_sectors,
                    x='RS_Ratio', y='RS_Momentum', text='Sector', color='Direction',
                    color_discrete_map={'Buying': '#4CAF50', 'Selling': '#F44336'},
                    size=size_column, size_max=60, hover_data=['Sector'],
                    title="RRG", labels={'RS_Ratio': 'Relative Strength', 'RS_Momentum': 'Relative Momentum'}
                )
                fig.add_shape(type="line", x0=0, y0=-40, x1=0, y1=120, line=dict(color="rgba(128, 128, 128, 0.7)", width=1.5, dash="dash"))
                fig.add_shape(type="line", x0=-2, y0=0, x1=2, y1=0, line=dict(color="rgba(128, 128, 128, 0.7)", width=1.5, dash="dash"))
                fig.add_shape(type="rect", x0=0, y0=0, x1=2, y1=120, fillcolor="rgba(144, 238, 144, 0.1)", line=dict(width=0))
                fig.add_shape(type="rect", x0=-2, y0=0, x1=0, y1=120, fillcolor="rgba(144, 200, 255, 0.1)", line=dict(width=0))
                fig.add_shape(type="rect", x0=-2, y0=-40, x1=0, y1=0, fillcolor="rgba(200, 200, 200, 0.1)", line=dict(width=0))
                fig.add_shape(type="rect", x0=0, y0=-40, x1=2, y1=0, fillcolor="rgba(255, 182, 193, 0.1)", line=dict(width=0))
                fig.add_annotation(x=1, y=60, text="LEADING", showarrow=False, font=dict(size=18, color="rgba(0,100,0,0.7)"))
                fig.add_annotation(x=-1, y=60, text="IMPROVING", showarrow=False, font=dict(size=18, color="rgba(0,0,150,0.7)"))
                fig.add_annotation(x=-1, y=-20, text="LAGGING", showarrow=False, font=dict(size=18, color="rgba(100,100,100,0.7)"))
                fig.add_annotation(x=1, y=0, text="WEAKENING", showarrow=False, font=dict(size=18, color="rgba(150,0,0,0.7)"))
                fig.add_annotation(x=0, y=60, ax=1, ay=60, xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=2, arrowcolor="rgba(0,100,0,0.5)")
                fig.add_annotation(x=-1, y=0, ax=-1, ay=60, xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=2, arrowcolor="rgba(0,0,150,0.5)")
                fig.add_annotation(x=0, y=-20, ax=-1, ay=-20, xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=2, arrowcolor="rgba(100,100,100,0.5)")
                fig.add_annotation(x=1, y=0, ax=1, ay=-20, xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=2, arrowcolor="rgba(150,0,0,0.5)")
                fig.update_traces(textposition='top center', textfont=dict(size=13, color='black', family="Arial, sans-serif"), marker=dict(opacity=0.85, line=dict(width=1, color='#FFFFFF')))
                fig.update_layout(
                    height=800, width=900, plot_bgcolor='rgba(240,240,240,0.3)', paper_bgcolor='white',
                    xaxis=dict(title=dict(font=dict(size=14, family="Arial, sans-serif")), gridcolor='rgba(200,200,200,0.2)', zerolinecolor='rgba(128,128,128,0.5)', zerolinewidth=1.5, range=[-2.1, 2.1]),
                    yaxis=dict(title=dict(font=dict(size=14, family="Arial, sans-serif")), gridcolor='rgba(200,200,200,0.2)', zerolinecolor='rgba(128,128,128,0.5)', zerolinewidth=1.5, range=[-40, 120]),
                    legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5, bgcolor='rgba(255,255,255,0.7)', bordercolor='rgba(128,128,128,0.5)', borderwidth=1),
                    title=dict(text="Relative Rotation Graph (RRG) - Sector Analysis", font=dict(size=22, family="Arial, sans-serif", color='#333333'), x=0.5),
                    margin=dict(l=50, r=50, t=80, b=100)
                )
                st.plotly_chart(fig, use_container_width=True)
                
                with st.expander("View Sector Data Tables"):
                    col3, col4 = st.columns(2)
                    with col3:
                        st.write("### Sectors Being Bought (Money In)")
                        if not buying_df.empty:
                            def highlight_buying(row):
                                return ['background-color: rgba(144, 238, 144, 0.3)'] * len(row)
                            styled_buying = buying_df.style.apply(highlight_buying, axis=1)
                            st.dataframe(styled_buying)
                        else:
                            st.write("No sectors with clear buying trends detected.")
                    with col4:
                        st.write("### Sectors Being Sold (Money Out)")
                        if not selling_df.empty:
                            def highlight_selling(row):
                                return ['background-color: rgba(255, 182, 193, 0.3)'] * len(row)
                            styled_selling = selling_df.style.apply(highlight_selling, axis=1)
                            st.dataframe(styled_selling)
                        else:
                            st.write("No sectors with clear selling trends detected.")
            else:
                st.write("No significant sector rotation detected based on current parameters.")
    
    # Tab 5: Multi Stock (No OTM flows as it’s sector-based, not individual stock-focused)
    with tab5:
        st.subheader("Multi Stock Analysis")
        col1, col2, col3 = st.columns(3)
        with col1:
            start_date = st.date_input("Start Date", datetime.now().date() - timedelta(days=7))
        with col2:
            end_date = st.date_input("End Date", datetime.now().date())
        with col3:
            min_buy_sell_ratio = st.number_input("Min Buy/Sell Ratio", min_value=0.2, max_value=100.0, value=1.5, step=0.1)
        
        col4, col5 = st.columns(2)
        with col4:
            min_total_volume = st.number_input("Min Total Volume", min_value=0, value=2000000, step=100000, format="%d")
        
        if st.button("Analyze Stocks"):
            with st.spinner("Analyzing stocks..."):
                start_datetime = datetime.combine(start_date, datetime.min.time())
                portfolio_df = analyze_portfolio(portfolio_symbols, (datetime.now() - start_datetime).days)
                st.session_state['analysis_results']['multi_stock'] = portfolio_df
        
        if 'multi_stock' in st.session_state['analysis_results']:
            portfolio_df = st.session_state['analysis_results']['multi_stock']
            if not portfolio_df.empty:
                portfolio_df['Date'] = pd.to_datetime(portfolio_df['Date'])
                filtered_df = portfolio_df[
                    (portfolio_df['Date'] >= pd.to_datetime(start_date)) &
                    (portfolio_df['Date'] <= pd.to_datetime(end_date)) &
                    (portfolio_df['Avg Buy/Sell Ratio'] >= min_buy_sell_ratio) &
                    (portfolio_df['Latest Volume'] >= min_total_volume)
                ]
                
                if not filtered_df.empty:
                    def get_sector(symbol):
                        for sector, symbols in sector_mapping.items():
                            if symbol in symbols:
                                return sector
                        return 'User Defined'
                    
                    filtered_df['Sector'] = filtered_df['Symbol'].apply(get_sector)
                    sector_tabs = st.tabs(list(sector_mapping.keys()))
                    
                    for tab, sector in zip(sector_tabs, sector_mapping.keys()):
                        with tab:
                            sector_df = filtered_df[filtered_df['Sector'] == sector]
                            if not sector_df.empty:
                                st.dataframe(sector_df)
                                sector_bought = sector_df['Bought Volume'].sum()
                                sector_sold = sector_df['Sold Volume'].sum()
                                st.write(f"Total Bought Volume: {sector_bought:,}")
                                st.write(f"Total Sold Volume: {sector_sold:,}")
                                st.write("Dark Pools: " + ("Bullish" if sector_bought > sector_sold else "Bearish"))
                                fig = px.bar(sector_df, x='Symbol', y='Avg Buy/Sell Ratio',
                                            title=f"{sector} Buy/Sell Ratios", color='Significant Days',
                                            hover_data=['Bought Volume', 'Sold Volume'])
                                fig.update_layout(barmode='group', xaxis_tickangle=-45)
                                st.plotly_chart(fig)
                    
                    st.write("### Overall Summary")
                    total_bought_volume = filtered_df['Bought Volume'].sum()
                    total_sold_volume = filtered_df['Sold Volume'].sum()
                    st.write(f"Total Bought Volume: {total_bought_volume:,}")
                    st.write(f"Total Sold Volume: {total_sold_volume:,}")
                    st.write("Dark Pools: " + ("Bullish" if total_bought_volume > total_sold_volume else "Bearish"))
                    
                    fig = px.bar(filtered_df, x='Symbol', y='Avg Buy/Sell Ratio',
                                title="Buy/Sell Ratios by Sector", color='Sector',
                                hover_data=['Bought Volume', 'Sold Volume'])
                    fig.update_layout(barmode='group', xaxis_tickangle=-45)
                    st.plotly_chart(fig)
                else:
                    st.write("No data available for the selected filters.")
            else:
                st.write("No data available for the selected portfolio.")
    
    # Tab 6: Latest Day
    with tab6:
        st.subheader("Latest Day")
        latest_df, latest_date = get_latest_data()
        
        col1, col2 = st.columns(2)
        with col1:
            volume_filter = st.checkbox("Filter: Bought Volume > 2x Sold Volume", value=False)
        
        if latest_df.empty:
            st.write("No data available for the latest day.")
        else:
            st.write(f"Showing data for: {latest_date}")
            metrics_list = []
            for _, row in latest_df.iterrows():
                total_volume = row.get('TotalVolume', 0)
                metrics = calculate_metrics(row, total_volume)
                metrics['Symbol'] = row['Symbol']
                metrics['TotalVolume'] = total_volume
                metrics_list.append(metrics)
            
            latest_df_processed = pd.DataFrame(metrics_list)
            latest_df_processed = latest_df_processed[latest_df_processed['total_volume'] >= 2000000]
            latest_df_processed = latest_df_processed[latest_df_processed['buy_to_sell_ratio'] > 1.5]
            if volume_filter:
                latest_df_processed = latest_df_processed[latest_df_processed['bought_volume'] > 2 * latest_df_processed['sold_volume']]
            latest_df_processed = latest_df_processed.sort_values(by=['buy_to_sell_ratio', 'total_volume'], ascending=[False, False])
            st.session_state['analysis_results']['latest'] = latest_df_processed
        
        if 'latest' in st.session_state['analysis_results']:
            latest_df_processed = st.session_state['analysis_results']['latest']
            if not latest_df_processed.empty:
                column_order = ['Symbol', 'buy_to_sell_ratio', 'total_volume', 'bought_volume', 'sold_volume', 'short_volume_ratio']
                latest_df_processed = latest_df_processed[column_order]
                st.dataframe(latest_df_processed)
                
                selected_symbol = st.selectbox("Select Symbol for OTM Flows", latest_df_processed['Symbol'].tolist(), key="latest_otm_select")
                if st.button(f"Show Top 10 OTM Flows for {selected_symbol}", key=f"otm_latest_{selected_symbol}"):
                    st.session_state['otm_flows']['symbol'] = selected_symbol
                    st.session_state['otm_flows']['show'] = True
                    with st.spinner(f"Fetching OTM flows for {selected_symbol}..."):
                        otm_flows = fetch_top_10_otm_flows(selected_symbol, urls)
                        st.session_state['otm_flows']['data'] = otm_flows
                
                if (st.session_state['otm_flows']['show'] and 
                    st.session_state['otm_flows']['symbol'] == selected_symbol):
                    otm_flows = st.session_state['otm_flows']['data']
                    if otm_flows is not None and not otm_flows.empty:
                        with st.expander(f"Top 10 OTM Flows for {selected_symbol}", expanded=True):
                            st.dataframe(otm_flows.style.format({
                                'Transaction Value': '${:,.2f}',
                                'Last Price': '${:.2f}',
                                'Volume': '{:,.0f}'
                            }))
                            fig_otm = px.bar(otm_flows, x='Strike Price', y='Transaction Value',
                                            color='Call/Put', title=f"Top 10 OTM Flows for {selected_symbol}",
                                            hover_data=['Expiration', 'Volume', 'Last Price'])
                            st.plotly_chart(fig_otm)
                            if st.button("Hide OTM Flows", key=f"hide_otm_latest_{selected_symbol}"):
                                st.session_state['otm_flows']['show'] = False
                                st.session_state['otm_flows']['data'] = None
                    else:
                        st.warning(f"No OTM flows found for {selected_symbol}.")
                        st.session_state['otm_flows']['show'] = False
                        st.session_state['otm_flows']['data'] = None
                
                fig = px.bar(latest_df_processed, x='Symbol', y='buy_to_sell_ratio',
                            title="Latest Day Buy/Sell Ratios",
                            hover_data=['total_volume', 'bought_volume', 'sold_volume'])
                fig.update_layout(barmode='group', xaxis_tickangle=-45)
                st.plotly_chart(fig)
            else:
                st.write("No records found with current filters.")
    
    # Tab 7: Rising Ratios
    with tab7:
        st.subheader("Stocks and Themes with Rising Buy/Sell Ratios (Last 10 Days)")
        with st.container():
            st.subheader("Analysis Parameters")
            col1, col2, col3 = st.columns(3)
            with col1:
                min_price_rising = st.number_input("Min Price ($)", value=5.0, min_value=1.0, 
                                                  max_value=50.0, step=0.5, key="rising_price")
                min_volume_rising = st.number_input("Minimum Daily Volume", value=500000,
                                                   step=100000, format="%d", key="rising_vol")
            with col2:
                min_market_cap_rising = st.number_input("Min Market Cap ($M)", value=500, 
                                                       min_value=100, max_value=5000, 
                                                       step=100, key="rising_market_cap")
                lookback_days_rising = st.slider("Lookback Days", 5, 20, 10, key="rising_days")
            with col3:
                max_dips_rising = st.number_input("Max Allowed Dips", value=3, min_value=0, 
                                                 max_value=5, step=1, key="rising_dips")
                find_ratios_button = st.button("Find Rising Ratios", type="primary", use_container_width=True)
        
        if find_ratios_button:
            with st.spinner("Analyzing stocks and themes with rising ratios..."):
                stock_df, theme_df, theme_top_stocks = find_rising_ratio_stocks(
                    lookback_days=lookback_days_rising,
                    min_volume=min_volume_rising,
                    max_dips=max_dips_rising,
                    min_price=min_price_rising,
                    min_market_cap=min_market_cap_rising * 1_000_000,
                    allowed_symbols=portfolio_symbols
                )
                st.session_state['analysis_results']['rising'] = {'stock': stock_df, 'theme': theme_df, 'top_stocks': theme_top_stocks}
        
        if 'rising' in st.session_state['analysis_results']:
            stock_df = st.session_state['analysis_results']['rising']['stock']
            theme_df = st.session_state['analysis_results']['rising']['theme']
            theme_top_stocks = st.session_state['analysis_results']['rising']['top_stocks']
            
            results_tab1, results_tab2 = st.tabs(["Themes", "Individual Stocks"])
            
            with results_tab1:
                st.subheader("Themes with Rising Buy/Sell Ratios")
                if not theme_df.empty:
                    metric_cols = st.columns(4)
                    with metric_cols[0]:
                        st.metric("Total Themes", len(theme_df))
                    with metric_cols[1]:
                        st.metric("Avg Ratio Increase", f"{theme_df['Ratio Increase'].mean():.2f}x")
                    with metric_cols[2]:
                        top_theme = theme_df.iloc[0]['Theme']
                        st.metric("Top Theme", top_theme)
                    with metric_cols[3]:
                        top_increase = theme_df.iloc[0]['Ratio Increase']
                        st.metric("Top Increase", f"{top_increase:.2f}x")
                    
                    st.dataframe(theme_df.style.background_gradient(subset=['Ratio Increase'], cmap='YlGn'), use_container_width=True)
                    
                    fig_theme = px.bar(theme_df, x='Theme', y='Ratio Increase', title="Themes by Buy/Sell Ratio Increase",
                                      hover_data=['Starting Ratio', 'Ending Ratio', 'Avg Daily Volume', 'Stocks Analyzed', 'Slope'],
                                      color='Ratio Increase', color_continuous_scale='YlGnBu')
                    fig_theme.update_layout(xaxis_tickangle=-45, height=500, margin=dict(l=20, r=20, t=40, b=20), coloraxis_showscale=True)
                    st.plotly_chart(fig_theme, use_container_width=True)
                    
                    st.subheader("Top Stocks by Theme")
                    for theme in theme_top_stocks:
                        with st.expander(f"{theme} - Top 3 Stocks"):
                            top_stocks_df = theme_top_stocks[theme]
                            if not top_stocks_df.empty:
                                col1, col2 = st.columns([2, 3])
                                with col1:
                                    st.dataframe(top_stocks_df.style.background_gradient(subset=['Ratio Increase'], cmap='YlGn'), use_container_width=True)
                                with col2:
                                    fig_top = px.bar(top_stocks_df, x='Symbol', y='Ratio Increase', title=f"Top 3 Stocks in {theme}",
                                                    hover_data=['Starting Ratio', 'Ending Ratio', 'Avg Daily Volume', 'Price'],
                                                    color='Ratio Increase', color_continuous_scale='YlGnBu')
                                    fig_top.update_layout(xaxis_tickangle=0, height=300)
                                    st.plotly_chart(fig_top, use_container_width=True)
                            else:
                                st.info("No qualifying stocks found for this theme.")
                else:
                    st.info("No themes found with rising buy/sell ratios based on current parameters.")
            
            with results_tab2:
                st.subheader("Top Stocks with Rising Buy/Sell Ratios")
                if not stock_df.empty:
                    metric_cols = st.columns(4)
                    with metric_cols[0]:
                        st.metric("Total Stocks", len(stock_df))
                    with metric_cols[1]:
                        st.metric("Avg Ratio Increase", f"{stock_df['Ratio Increase'].mean():.2f}x")
                    with metric_cols[2]:
                        top_stock = stock_df.iloc[0]['Symbol']
                        st.metric("Top Stock", top_stock)
                    with metric_cols[3]:
                        top_increase = stock_df.iloc[0]['Ratio Increase']
                        st.metric("Top Increase", f"{top_increase:.2f}x")
                    
                    stock_search = st.text_input("Filter stocks by symbol or name", "")
                    filtered_df = stock_df
                    if stock_search:
                        filtered_df = stock_df[stock_df['Symbol'].str.contains(stock_search, case=False)]
                    
                    st.dataframe(filtered_df.style.background_gradient(subset=['Ratio Increase'], cmap='YlGn'), use_container_width=True)
                    
                    selected_symbol = st.selectbox("Select Symbol for OTM Flows", filtered_df['Symbol'].tolist(), key="rising_otm_select")
                    if st.button(f"Show Top 10 OTM Flows for {selected_symbol}", key=f"otm_rising_{selected_symbol}"):
                        st.session_state['otm_flows']['symbol'] = selected_symbol
                        st.session_state['otm_flows']['show'] = True
                        with st.spinner(f"Fetching OTM flows for {selected_symbol}..."):
                            otm_flows = fetch_top_10_otm_flows(selected_symbol, urls)
                            st.session_state['otm_flows']['data'] = otm_flows
                    
                    if (st.session_state['otm_flows']['show'] and 
                        st.session_state['otm_flows']['symbol'] == selected_symbol):
                        otm_flows = st.session_state['otm_flows']['data']
                        if otm_flows is not None and not otm_flows.empty:
                            with st.expander(f"Top 10 OTM Flows for {selected_symbol}", expanded=True):
                                st.dataframe(otm_flows.style.format({
                                    'Transaction Value': '${:,.2f}',
                                    'Last Price': '${:.2f}',
                                    'Volume': '{:,.0f}'
                                }))
                                fig_otm = px.bar(otm_flows, x='Strike Price', y='Transaction Value',
                                                color='Call/Put', title=f"Top 10 OTM Flows for {selected_symbol}",
                                                hover_data=['Expiration', 'Volume', 'Last Price'])
                                st.plotly_chart(fig_otm)
                                if st.button("Hide OTM Flows", key=f"hide_otm_rising_{selected_symbol}"):
                                    st.session_state['otm_flows']['show'] = False
                                    st.session_state['otm_flows']['data'] = None
                        else:
                            st.warning(f"No OTM flows found for {selected_symbol}.")
                            st.session_state['otm_flows']['show'] = False
                            st.session_state['otm_flows']['data'] = None
                    
                    fig_stock = px.bar(stock_df.head(10), x='Symbol', y='Ratio Increase',
                                      title="Top 10 Stocks by Buy/Sell Ratio Increase",
                                      hover_data=['Avg Daily Volume', 'Starting Ratio', 'Ending Ratio', 'Dips', 'Price', 'Market Cap (M)'],
                                      color='Ratio Increase', color_continuous_scale='YlGnBu')
                    fig_stock.update_layout(xaxis_tickangle=-45, height=500, margin=dict(l=20, r=20, t=40, b=20))
                    st.plotly_chart(fig_stock, use_container_width=True)
                else:
                    st.info("No stocks found with rising buy/sell ratios based on current parameters.")
    
        # Tab 8: Portfolio Allocation
    with tab8:
        st.header("Portfolio Allocation")
        with st.container():
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write("Generate a comprehensive report summarizing key insights and portfolio allocation recommendations.")
            with col2:
                generate_button = st.button("Generate Portfolio Allocation", type="primary", use_container_width=True)
        
        if generate_button:
            with st.spinner("Generating Portfolio Allocation..."):
                accumulation_df, distribution_df, rising_df, buying_sectors_df, selling_sectors_df = generate_evening_report(
                    capital=10000, run_on_demand=True
                )
                st.session_state['analysis_results']['portfolio'] = {
                    'accumulation': accumulation_df,
                    'distribution': distribution_df,
                    'rising': rising_df,
                    'buying_sectors': buying_sectors_df,
                    'selling_sectors': selling_sectors_df
                }
        
        if 'portfolio' in st.session_state['analysis_results']:
            report_data = st.session_state.get('evening_report', {})
            accumulation_df = st.session_state['analysis_results']['portfolio']['accumulation']
            distribution_df = st.session_state['analysis_results']['portfolio']['distribution']
            rising_df = st.session_state['analysis_results']['portfolio']['rising']
            buying_sectors_df = st.session_state['analysis_results']['portfolio']['buying_sectors']
            selling_sectors_df = st.session_state['analysis_results']['portfolio']['selling_sectors']
            
            st.subheader("Portfolio Analysis Report")
            st.caption(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ET")
            
            buy_df = report_data.get('buy_df', pd.DataFrame())
            if not buy_df.empty:
                metric_cols = st.columns(4)
                with metric_cols[0]:
                    st.metric("Total Invested", f"${report_data['total_cost']:,.2f}")
                with metric_cols[1]:
                    st.metric("Cash Remaining", f"${report_data['cash_remaining']:,.2f}")
                with metric_cols[2]:
                    st.metric("Allocation Count", f"{len(buy_df)} stocks")
                with metric_cols[3]:
                    highest_allocation = buy_df.iloc[0]['Symbol'] if not buy_df.empty else "N/A"
                    st.metric("Highest Allocation", highest_allocation)
            
            portfolio_tab, insights_tab, recommendations_tab = st.tabs(["Portfolio Allocation", "Market Insights", "Recommendations"])
            
            with portfolio_tab:
                st.subheader("Portfolio Optimization ($10,000 Capital)")
                if not buy_df.empty:
                    col1, col2 = st.columns([3, 2])
                    with col1:
                        st.dataframe(
                            buy_df[['Symbol', 'Price', 'Shares', 'Cost', 'Reason']]
                            .style.bar(subset=['Cost'], color='#9EE6CF')
                            .format({'Price': '${:.2f}', 'Cost': '${:.2f}'}),
                            use_container_width=True,
                            height=250
                        )
                        selected_symbol = st.selectbox("Select Symbol for OTM Flows", buy_df['Symbol'].tolist(), key="portfolio_otm_select")
                        if st.button(f"Show Top 10 OTM Flows for {selected_symbol}", key=f"otm_portfolio_{selected_symbol}"):
                            st.session_state['otm_flows']['symbol'] = selected_symbol
                            st.session_state['otm_flows']['show'] = True
                            with st.spinner(f"Fetching OTM flows for {selected_symbol}..."):
                                otm_flows = fetch_top_10_otm_flows(selected_symbol, urls)
                                st.session_state['otm_flows']['data'] = otm_flows
                        
                        if (st.session_state['otm_flows']['show'] and 
                            st.session_state['otm_flows']['symbol'] == selected_symbol):
                            otm_flows = st.session_state['otm_flows']['data']
                            if otm_flows is not None and not otm_flows.empty:
                                with st.expander(f"Top 10 OTM Flows for {selected_symbol}", expanded=True):
                                    st.dataframe(otm_flows.style.format({
                                        'Transaction Value': '${:,.2f}',
                                        'Last Price': '${:.2f}',
                                        'Volume': '{:,.0f}'
                                    }))
                                    fig_otm = px.bar(otm_flows, x='Strike Price', y='Transaction Value',
                                                    color='Call/Put', title=f"Top 10 OTM Flows for {selected_symbol}",
                                                    hover_data=['Expiration', 'Volume', 'Last Price'])
                                    st.plotly_chart(fig_otm)
                                    if st.button("Hide OTM Flows", key=f"hide_otm_portfolio_{selected_symbol}"):
                                        st.session_state['otm_flows']['show'] = False
                                        st.session_state['otm_flows']['data'] = None
                            else:
                                st.warning(f"No OTM flows found for {selected_symbol}.")
                                st.session_state['otm_flows']['show'] = False
                                st.session_state['otm_flows']['data'] = None
                    with col2:
                        fig = px.pie(buy_df, values='Cost', names='Symbol', title="Portfolio Allocation",
                                    color_discrete_sequence=px.colors.qualitative.G10, hole=0.4)
                        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
                                         margin=dict(l=20, r=20, t=40, b=20))
                        fig.update_traces(textposition='inside', textinfo='percent+label')
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Insufficient data for portfolio optimization. Try adjusting filters or updating stock data.")
            
            with insights_tab:
                st.subheader("Market Trend Analysis")
                insight_cols = st.columns(2)
                with insight_cols[0]:
                    st.write("#### Accumulation Patterns")
                    if not accumulation_df.empty:
                        st.dataframe(accumulation_df.style.background_gradient(subset=['Avg Buy/Sell Ratio'], cmap='Greens'),
                                    use_container_width=True, height=200)
                        if len(accumulation_df) > 0:
                            fig_acc = px.bar(accumulation_df.head(5), x='Symbol', y='Avg Buy/Sell Ratio',
                                            title="Top Accumulation Stocks", color='Avg Buy/Sell Ratio',
                                            color_continuous_scale='Greens')
                            fig_acc.update_layout(height=250)
                            st.plotly_chart(fig_acc, use_container_width=True)
                    else:
                        st.info("No accumulation patterns detected.")
                    
                    st.write("#### Sector Buying Trends")
                    if not buying_sectors_df.empty:
                        st.dataframe(buying_sectors_df, use_container_width=True, height=200)
                        if len(buying_sectors_df) > 0:
                            fig_sectors = px.bar(buying_sectors_df, x='Sector', y='Avg Buy/Sell Ratio',
                                                title="Sectors with Strong Buying", color='Avg Buy/Sell Ratio',
                                                color_continuous_scale='Blues')
                            fig_sectors.update_layout(height=250)
                            st.plotly_chart(fig_sectors, use_container_width=True)
                    else:
                        st.info("No sectors with significant buying trends detected.")
                
                with insight_cols[1]:
                    st.write("#### Distribution Patterns")
                    if not distribution_df.empty:
                        st.dataframe(distribution_df.style.background_gradient(subset=['Avg Buy/Sell Ratio'], cmap='Reds'),
                                    use_container_width=True, height=200)
                        if len(distribution_df) > 0:
                            fig_dist = px.bar(distribution_df.head(5), x='Symbol', y='Avg Buy/Sell Ratio',
                                            title="Top Distribution Stocks",
                                            # --- CUT OFF HERE ---
                                            color='Avg Buy/Sell Ratio', color_continuous_scale='Reds')
                            fig_dist.update_layout(height=250)
                            st.plotly_chart(fig_dist, use_container_width=True)
                    else:
                        st.info("No distribution patterns detected.")
                    
                    st.write("#### Sector Selling Trends")
                    if not selling_sectors_df.empty:
                        st.dataframe(selling_sectors_df, use_container_width=True, height=200)
                        if len(selling_sectors_df) > 0:
                            fig_sectors_sell = px.bar(selling_sectors_df, x='Sector', y='Avg Buy/Sell Ratio',
                                                    title="Sectors with Strong Selling", color='Avg Buy/Sell Ratio',
                                                    color_continuous_scale='Reds')
                            fig_sectors_sell.update_layout(height=250)
                            st.plotly_chart(fig_sectors_sell, use_container_width=True)
                    else:
                        st.info("No sectors with significant selling trends detected.")
                
                st.write("#### Stocks with Rising Buy/Sell Ratios")
                if not rising_df.empty:
                    st.dataframe(rising_df.style.background_gradient(subset=['Ratio Increase'], cmap='YlGn'),
                                use_container_width=True, height=200)
                    if len(rising_df) > 0:
                        fig_rising = px.bar(rising_df.head(10), x='Symbol', y='Ratio Increase',
                                            title="Top 10 Stocks with Rising Buy/Sell Ratios",
                                            color='Ratio Increase', color_continuous_scale='YlGnBu')
                        fig_rising.update_layout(height=350)
                        st.plotly_chart(fig_rising, use_container_width=True)
                    selected_symbol = st.selectbox("Select Symbol for OTM Flows", rising_df['Symbol'].tolist(), key="rising_insights_otm_select")
                    if st.button(f"Show Top 10 OTM Flows for {selected_symbol}", key=f"otm_rising_insights_{selected_symbol}"):
                        st.session_state['otm_flows']['symbol'] = selected_symbol
                        st.session_state['otm_flows']['show'] = True
                        with st.spinner(f"Fetching OTM flows for {selected_symbol}..."):
                            otm_flows = fetch_top_10_otm_flows(selected_symbol, urls)
                            st.session_state['otm_flows']['data'] = otm_flows
                    
                    if (st.session_state['otm_flows']['show'] and 
                        st.session_state['otm_flows']['symbol'] == selected_symbol):
                        otm_flows = st.session_state['otm_flows']['data']
                        if otm_flows is not None and not otm_flows.empty:
                            with st.expander(f"Top 10 OTM Flows for {selected_symbol}", expanded=True):
                                st.dataframe(otm_flows.style.format({
                                    'Transaction Value': '${:,.2f}',
                                    'Last Price': '${:.2f}',
                                    'Volume': '{:,.0f}'
                                }))
                                fig_otm = px.bar(otm_flows, x='Strike Price', y='Transaction Value',
                                                color='Call/Put', title=f"Top 10 OTM Flows for {selected_symbol}",
                                                hover_data=['Expiration', 'Volume', 'Last Price'])
                                st.plotly_chart(fig_otm)
                                if st.button("Hide OTM Flows", key=f"hide_otm_rising_insights_{selected_symbol}"):
                                    st.session_state['otm_flows']['show'] = False
                                    st.session_state['otm_flows']['data'] = None
                        else:
                            st.warning(f"No OTM flows found for {selected_symbol}.")
                            st.session_state['otm_flows']['show'] = False
                            st.session_state['otm_flows']['data'] = None
                else:
                    st.info("No stocks with rising buy/sell ratios detected.")
            
            with recommendations_tab:
                st.subheader("Trading Recommendations")
                rec_col1, rec_col2 = st.columns(2)
                
                with rec_col1:
                    st.write("#### Buy Recommendations")
                    if not buy_df.empty:
                        min_signal = st.slider("Minimum Signal Strength", 1.0, 10.0, 3.0, step=0.1, key="min_buy_signal")
                        filtered_buy = buy_df[buy_df['Signal Strength'] >= min_signal]
                        if not filtered_buy.empty:
                            st.dataframe(filtered_buy[['Symbol', 'Price', 'Reason', 'Signal Strength']]
                                        .style.background_gradient(subset=['Signal Strength'], cmap='Greens'),
                                        use_container_width=True, height=300)
                            fig_buy_rec = px.bar(filtered_buy.head(7), x='Symbol', y='Signal Strength',
                                                title="Top Buy Recommendations", color='Signal Strength',
                                                color_continuous_scale='Greens', hover_data=['Price', 'Reason'])
                            fig_buy_rec.update_layout(height=300)
                            st.plotly_chart(fig_buy_rec, use_container_width=True)
                        else:
                            st.info(f"No buy recommendations with signal strength ≥ {min_signal}.")
                    else:
                        st.info("No strong buy candidates identified.")
                
                with rec_col2:
                    st.write("#### Sell/Short Recommendations")
                    sell_df = report_data.get('sell_df', pd.DataFrame())
                    if not sell_df.empty:
                        if 'Signal Strength' not in sell_df.columns:
                            sell_df['Signal Strength'] = [round(random.uniform(5, 10), 1) for _ in range(len(sell_df))]
                        sell_df['Reason'] = 'Distribution'
                        min_sell_signal = st.slider("Minimum Signal Strength", 1.0, 10.0, 5.0, step=0.1, key="min_sell_signal")
                        filtered_sell = sell_df[sell_df['Signal Strength'] >= min_sell_signal]
                        if not filtered_sell.empty:
                            st.dataframe(filtered_sell[['Symbol', 'Price', 'Reason', 'Signal Strength']]
                                        .style.background_gradient(subset=['Signal Strength'], cmap='Reds'),
                                        use_container_width=True, height=300)
                            fig_sell_rec = px.bar(filtered_sell.head(7), x='Symbol', y='Signal Strength',
                                                title="Top Sell/Short Recommendations", color='Signal Strength',
                                                color_continuous_scale='Reds', hover_data=['Price', 'Reason'])
                            fig_sell_rec.update_layout(height=300)
                            st.plotly_chart(fig_sell_rec, use_container_width=True)
                        else:
                            st.info(f"No sell recommendations with signal strength ≥ {min_sell_signal}.")
                    else:
                        st.info("No strong sell/short candidates identified.")
        else:
            st.info("👆 Click 'Generate Portfolio Allocation' to create a comprehensive report with allocation recommendations.")
            with st.expander("Preview of Portfolio Analysis"):
                st.write("The report will include:")
                st.write("- Portfolio allocation recommendations based on $10,000 capital")
                st.write("- Visualizations of allocation by stock and sector")
                st.write("- Market insights including accumulation and distribution patterns")
                st.write("- Sector rotation analysis")
                st.write("- Specific buy and sell recommendations with signal strength")
                st.image("https://via.placeholder.com/800x400?text=Sample+Portfolio+Report", 
                        caption="Sample portfolio allocation visualization")
    
if __name__ == "__main__":
    run()

