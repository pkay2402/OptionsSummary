import random
import pandas as pd
import streamlit as st
import requests
import io
from datetime import datetime, timedelta
import yfinance as yf
import plotly.express as px
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import concurrent.futures
import logging

# Configure logging at the top of your file (add this after your imports)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Output to console
    ]
)
logger = logging.getLogger(__name__)

# Sector mapping
sector_mapping = {
    'Index/Sectors': [
        'SPY',  # SPY
        'QQQ',  # QQQ
        'DIA',  # DIA
        'IWM',  # IWM
        'VXX',  # VXX
        'SMH',  # SMH
        'IVV',  # IVV
        'VTI',  # VTI
        'VOO',  # VOO
        'XLK',  # XLK
        'VUG',  # VUG
        'XLF',  # XLF
        'XLV',  # XLV
        'XLY',  # XLY
        'XLC',  # XLC
        'XLI',  # XLI
        'XLE',  # XLE
        'XLB',  # XLB
        'XLU',  # XLU
        'XLRE', # XLRE
        'XLP',  # XLP
        'XBI',  # XBI
        'XOP',  # XOP
        'XME',  # XME
        'XRT',  # XRT
        'XHB',  # XHB
        'XWEB', # XWEB
        'XLC',  # XLC
        'RSP',  # RSP
    ],
    'Technology': [
        'AAPL',  # Apple
        'MSFT',  # Microsoft
        'NVDA',  # NVIDIA
        'AMZN',  # Amazon
        'GOOGL', # Alphabet Class A
        'META',  # Meta Platforms
        'TSLA',  # Tesla
        'PLTR',  # Palantir
        'ORCL',  # Oracle
        'AMD',   # Advanced Micro Devices
        'NFLX',  # Netflix
        'ADBE',  # Adobe
        'CRM',   # Salesforce
        'INTC',  # Intel
        'CSCO',  # Cisco Systems
        'QCOM',  # Qualcomm
        'TXN',   # Texas Instruments
        'INTU',  # Intuit
        'IBM',   # IBM
        'NOW',   # ServiceNow
        'AVGO',  # Broadcom
        'UBER',  # Uber Technologies
        'SNOW',  # Snowflake
        'DELL',  # Dell Technologies
        'PANW'   # Palo Alto Networks
    ],
    'Financials': [
        'JPM',   # JPMorgan Chase
        'V',     # Visa
        'MA',    # Mastercard
        'BAC',   # Bank of America
        'WFC',   # Wells Fargo
        'GS',    # Goldman Sachs
        'MS',    # Morgan Stanley
        'C',     # Citigroup
        'AXP',   # American Express
        'SCHW',  # Charles Schwab
        'COF',   # Capital One
        'MET',   # MetLife
        'AIG',   # American International Group
        'BK',    # Bank of New York Mellon
        'BLK',   # BlackRock
        'TFC',   # Truist Financial
        'USB',   # U.S. Bancorp
        'PNC',   # PNC Financial Services
        'CME',   # CME Group
        'SPGI',  # S&P Global
        'ICE',   # Intercontinental Exchange
        'MCO',   # Moody's
        'AON',   # Aon
        'PYPL',  # PayPal
        'SQ'     # Square (Block)
    ],
    'Healthcare': [
        'LLY',   # Eli Lilly
        'UNH',   # UnitedHealth Group
        'JNJ',   # Johnson & Johnson
        'PFE',   # Pfizer
        'MRK',   # Merck
        'ABBV',  # AbbVie
        'TMO',   # Thermo Fisher Scientific
        'AMGN',  # Amgen
        'GILD',  # Gilead Sciences
        'CVS',   # CVS Health
        'MDT',   # Medtronic
        'BMY',   # Bristol Myers Squibb
        'ABT',   # Abbott Laboratories
        'DHR',   # Danaher
        'ISRG',  # Intuitive Surgical
        'SYK',   # Stryker
        'REGN',  # Regeneron Pharmaceuticals
        'VRTX',  # Vertex Pharmaceuticals
        'CI',    # Cigna
        'ZTS',   # Zoetis
        'BDX',   # Becton Dickinson
        'HCA',   # HCA Healthcare
        'EW',    # Edwards Lifesciences
        'DXCM',  # DexCom
        'BIIB'   # Biogen
    ],
    'Consumer': [
        'WMT',   # Walmart
        'PG',    # Procter & Gamble
        'KO',    # Coca-Cola
        'PEP',   # PepsiCo
        'COST',  # Costco
        'MCD',   # McDonald's
        'DIS',   # Walt Disney
        'NKE',   # Nike
        'SBUX',  # Starbucks
        'LOW',   # Lowe's
        'TGT',   # Target
        'HD',    # Home Depot
        'CL',    # Colgate-Palmolive
        'MO',    # Altria Group
        'KHC',   # Kraft Heinz
        'PM',    # Philip Morris
        'TJX',   # TJX Companies
        'DG',    # Dollar General
        'DLTR',  # Dollar Tree
        'YUM',   # Yum! Brands
        'GIS',   # General Mills
        'KMB',   # Kimberly-Clark
        'MNST',  # Monster Beverage
        'EL',    # Estee Lauder
        'CMG'    # Chipotle Mexican Grill
    ],
    'Energy': [
        'XOM',   # Exxon Mobil
        'CVX',   # Chevron
        'COP',   # ConocoPhillips
        'SLB',   # Schlumberger
        'EOG',   # EOG Resources
        'MPC',   # Marathon Petroleum
        'PSX',   # Phillips 66
        'OXY',   # Occidental Petroleum
        'VLO',   # Valero Energy
        'PXD',   # Pioneer Natural Resources
        'HES',   # Hess
        'WMB',   # Williams Companies
        'KMI',   # Kinder Morgan
        'OKE',   # ONEOK
        'HAL',   # Halliburton
        'BKR',   # Baker Hughes
        'FANG',  # Diamondback Energy
        'DVN',   # Devon Energy
        'TRGP',  # Targa Resources
        'APA',   # APA Corporation
        'EQT',   # EQT Corporation
        'MRO',   # Marathon Oil
        'NOV',   # NOV Inc.
        'FTI',   # TechnipFMC
        'RRC'    # Range Resources
    ],
    'Industrials': [
        'CAT',   # Caterpillar
        'DE',    # Deere & Company
        'UPS',   # United Parcel Service
        'FDX',   # FedEx
        'BA',    # Boeing
        'HON',   # Honeywell
        'UNP',   # Union Pacific
        'MMM',   # 3M
        'GE',    # General Electric
        'LMT',   # Lockheed Martin
        'RTX',   # RTX Corporation
        'GD',    # General Dynamics
        'CSX',   # CSX Corporation
        'NSC',   # Norfolk Southern
        'WM',    # Waste Management
        'ETN',   # Eaton
        'ITW',   # Illinois Tool Works
        'EMR',   # Emerson Electric
        'PH',    # Parker-Hannifin
        'ROK',   # Rockwell Automation
        'CARR',  # Carrier Global
        'OTIS',  # Otis Worldwide
        'IR',    # Ingersoll Rand
        'CMI',   # Cummins
        'FAST'   # Fastenal
    ],
    'User Defined': []  # Catch-all for unmapped symbols
}

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

@st.cache_data(ttl=11520)
def analyze_symbol(symbol, lookback_days=20, threshold=1.5):
    results = []
    significant_days = 0
    
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
                
                if metrics['buy_to_sell_ratio'] > threshold:
                    significant_days += 1
                
                metrics['date'] = date
                results.append(metrics)
    
    df_results = pd.DataFrame(results)
    if not df_results.empty:
        df_results['date'] = pd.to_datetime(df_results['date'], format='%Y%m%d')
        df_results = df_results.sort_values('date', ascending=False)
    
    return df_results, significant_days

def validate_pattern(symbol, dates, pattern_type="accumulation"):
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

def find_patterns(lookback_days=10, min_volume=1000000, pattern_type="accumulation", use_price_validation=False, min_price=5.0, min_market_cap=500_000_000):
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
            # Fetch stock data from the database
            stock_info = get_stock_info_from_db(symbol)
            price = stock_info['price']
            market_cap = stock_info['market_cap']
            
            # Skip stocks that don't meet the price and market cap criteria
            if price < min_price or market_cap < min_market_cap:
                continue
            
            if not use_price_validation or validate_pattern(symbol, data['dates'], pattern_type):
                avg_ratio = sum(data['ratios']) / len(data['ratios'])
                avg_volume = data['total_volume'] / len(data['dates'])
                results.append({
                    'Symbol': symbol,
                    'Price': round(price, 2),
                    'Market Cap (M)': round(market_cap / 1_000_000, 2),  # Convert to millions
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

def get_sector_rotation(lookback_days=10, min_volume=50000):
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

def analyze_portfolio(symbols, lookback_days=20):
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
                print(f'{symbol} generated an exception: {exc}')
    
    portfolio_df = pd.DataFrame(portfolio_results)
    if not portfolio_df.empty:
        portfolio_df = portfolio_df.sort_values(by=['Avg Buy/Sell Ratio', 'Bought Volume'], ascending=[False, False])
    return portfolio_df

def check_alerts(df_results, symbol, threshold=2.0):
    if not df_results.empty and df_results['buy_to_sell_ratio'].max() > threshold:
        st.warning(f"Alert: {symbol} has a Buy/Sell Ratio above {threshold} on {df_results['date'].iloc[0].strftime('%Y-%m-%d')}!")

def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

def plot_correlation_heatmap(symbols, lookback_days=20):
    data = {}
    for symbol in symbols:
        df, _ = analyze_symbol(symbol, lookback_days)
        if not df.empty:
            data[symbol] = df['buy_to_sell_ratio']
    corr_df = pd.DataFrame(data).corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_df, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

def get_latest_data():
    for i in range(7):  # Check the last 7 days
        date = (datetime.now() - timedelta(days=i)).strftime("%Y%m%d")
        data = download_finra_short_sale_data(date)
        if data:
            df = process_finra_short_sale_data(data)
            if not df.empty:
                return df, date
    return pd.DataFrame(), None
import sqlite3

# Function to set up the SQLite database and table
def setup_stock_database():
    conn = sqlite3.connect('stock_data.db')
    cursor = conn.cursor()
    
    # Drop the existing table
    cursor.execute('DROP TABLE IF EXISTS stocks')
    print("Dropped existing `stocks` table (if it existed).")
    
    # Create the table
    cursor.execute('''
        CREATE TABLE stocks (
            symbol TEXT PRIMARY KEY,
            price REAL,
            market_cap REAL,
            last_updated TEXT
        )
    ''')
    print("Created `stocks` table with correct schema.")
    
    conn.commit()
    conn.close()

# Check if the table exists with the correct schema
def check_and_setup_database():
    try:
        conn = sqlite3.connect('stock_data.db')
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='stocks'")
        if not cursor.fetchone():
            conn.close()
            setup_stock_database()
            return
        
        # Check if the schema is correct
        cursor.execute("PRAGMA table_info(stocks)")
        columns = [info[1] for info in cursor.fetchall()]
        if not all(col in columns for col in ['symbol', 'price', 'market_cap', 'last_updated']):
            conn.close()
            setup_stock_database()
            return
        
        conn.close()
    except Exception as e:
        print(f"Error checking database: {e}")
        setup_stock_database()

# Call this at the start of your program
check_and_setup_database()

def inspect_database():
    conn = sqlite3.connect('stock_data.db')
    cursor = conn.cursor()
    
    # Check all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    print(f"Tables in database: {tables}")
    
    # Check columns in stocks table if it exists
    cursor.execute("PRAGMA table_info(stocks)")
    columns = cursor.fetchall()
    print(f"Columns in stocks table: {columns}")
    
    conn.close()

# Function to fetch stock data from yfinance and store in the database
def update_stock_database(symbols):
    conn = sqlite3.connect('stock_data.db')
    cursor = conn.cursor()
    
    # Batch-fetch stock data using yfinance
    try:
        tickers = yf.Tickers(' '.join(symbols))
        
        for symbol in symbols:
            try:
                ticker = tickers.tickers[symbol]
                info = ticker.info
                
                # Get current price (regularMarketPrice is more reliable)
                price = info.get('regularMarketPrice', 0)
                
                # Get market cap
                market_cap = info.get('marketCap', 0)
                
                # Insert or update the stock data
                cursor.execute('''
                    INSERT OR REPLACE INTO stocks (symbol, price, market_cap, last_updated)
                    VALUES (?, ?, ?, ?)
                ''', (symbol, price, market_cap, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
                
            except Exception as e:
                print(f"Error fetching data for {symbol}: {e}")
                # Insert a record with zeros when there's an error
                cursor.execute('''
                    INSERT OR REPLACE INTO stocks (symbol, price, market_cap, last_updated)
                    VALUES (?, ?, ?, ?)
                ''', (symbol, 0, 0, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
    
    except Exception as e:
        print(f"Error fetching batch data: {e}")
    
    finally:
        conn.commit()
        conn.close()

# Function to query stock data from the database
def get_stock_info_from_db(symbol):
    conn = sqlite3.connect('stock_data.db')
    cursor = conn.cursor()
    
    cursor.execute('SELECT price, market_cap FROM stocks WHERE symbol = ?', (symbol,))
    result = cursor.fetchone()
    
    conn.close()
    
    if result:
        return {'price': result[0], 'market_cap': result[1]}
    return {'price': 0, 'market_cap': 0}
# Add this new function after the existing functions
from scipy.stats import linregress

def find_rising_ratio_stocks(lookback_days=10, min_volume=500000, max_dips=3, min_price=5.0, min_market_cap=500_000_000, allowed_symbols=None):
    symbol_data = {}
    
    # Collect data for the last `lookback_days` (e.g., March 11-20, 2025)
    for i in range(lookback_days):
        date = (datetime.now() - timedelta(days=i)).strftime("%Y%m%d")
        data = download_finra_short_sale_data(date)
        if not data:
            continue
        
        df = process_finra_short_sale_data(data)
        df = df[df['TotalVolume'] > min_volume]
        
        for _, row in df.iterrows():
            symbol = row['Symbol']
            # Pre-filter: Only include symbols in allowed_symbols (if provided)
            if allowed_symbols and symbol not in allowed_symbols:
                continue
            
            total_volume = row.get('TotalVolume', 0)
            metrics = calculate_metrics(row, total_volume)
            
            if symbol not in symbol_data:
                symbol_data[symbol] = {'ratios': [], 'volumes': [], 'dates': []}
            symbol_data[symbol]['ratios'].append(metrics['buy_to_sell_ratio'])
            symbol_data[symbol]['volumes'].append(total_volume)
            symbol_data[symbol]['dates'].append(date)
    
    # Analyze for rising ratios with price and market cap filtering from the database
    results = []
    for symbol, data in symbol_data.items():
        num_days = len(data['ratios'])
        if num_days >= 5:  # Require at least 5 days of data
            # Get price and market cap from the database
            stock_info = get_stock_info_from_db(symbol)
            price = stock_info['price']
            market_cap = stock_info['market_cap']
            
            # Skip stocks with price < $5 or market cap < $500M
            if price < min_price or market_cap < min_market_cap:
                continue
            
            ratios = data['ratios'][::-1]  # Reverse to check oldest to newest
            
            # Allow for occasional dips
            dips = sum(1 for i in range(len(ratios) - 1) if ratios[i+1] < ratios[i])
            slope, _, _, _, _ = linregress(range(len(ratios)), ratios)
            
            if dips <= max_dips and slope > 0.005:  # Relaxed slope condition
                avg_volume = sum(data['volumes']) / len(data['volumes'])
                results.append({
                    'Symbol': symbol,
                    'Price': round(price, 2),
                    'Market Cap (M)': round(market_cap / 1_000_000, 2),  # Convert to millions
                    'Avg Daily Volume': int(avg_volume),
                    'Starting Ratio': round(ratios[0], 2),
                    'Ending Ratio': round(ratios[-1], 2),
                    'Ratio Increase': round(ratios[-1] - ratios[0], 2),
                    'Total Volume': int(sum(data['volumes'])),
                    'Dips': dips,
                    'Days of Data': num_days
                })
    # Sort by ratio increase and total volume
    results = sorted(results, key=lambda x: (x['Ratio Increase'], x['Total Volume']), reverse=True)
    return pd.DataFrame(results[:40])

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
    
    with st.sidebar:
        portfolio_symbols = st.text_area("User Defined Symbols for Multi Stock (comma-separated)", 
                                         "AAPL, MSFT, NVDA, AMZN, GOOGL, META, TSLA, PLTR, LLY, JPM, AVGO, UNH, V, WMT, XOM, MA, JNJ, PG, HD, COST, ORCL, CVX, BAC, KO, PEP, ABBV, "
                                         "MRK, AMD, NFLX, ADBE, CRM, INTC, CSCO, PFE, TMO, MCD, DIS, WFC, QCOM, LIN, GE, AXP, CAT, IBM, VZ, GS, MS, PM, LOW, NEE, RTX, BA, HON, UNP, "
                                         "T, BLK, MDT, SBUX, LMT, AMGN, GILD, CVS, DE, TGT, AMT, BKNG, SO, DUK, PYPL, UPS, C, COP, MMM, ACN, ABT, DHR, TMUS, TXN, MDLZ, BMY, INTU, NKE, "
                                         "CL, MO, KHC, EMR, GM, F, FDX, "
                                         "GD, BK, SPG, CHTR, USB, MET, AIG, COF, DOW, SCHW, CMCSA, SPY, QQQ, VOO, IVV, XLK, VUG, XLV, XLF, IWF, VTI").split(",")
        portfolio_symbols = [s.strip().upper() for s in portfolio_symbols if s.strip()]
    
    tab1, tab2, tab3, tab4, tab5, tab6,tab7 = st.tabs(["Single Stock", "Accumulation", "Distribution", 
                                                        "Sector Rotation", "Multi Stock", "Latest Day", 
                                                        "Rising Ratios"])
    
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
                              hover_data=['total_volume', 'short_volume_ratio'])
                fig.add_hline(y=threshold, line_dash="dash", line_color="red", annotation_text=f"Threshold: {threshold}")
                st.plotly_chart(fig)
                
                st.subheader("Daily Analysis")
                def highlight_significant(row):
                    return ['background-color: rgba(144, 238, 144, 0.3)' if row['buy_to_sell_ratio'] > threshold else ''] * len(row)
                display_df = results_df.copy()
                for col in ['total_volume', 'bought_volume', 'sold_volume']:
                    display_df[col] = display_df[col].astype(int)
                styled_df = display_df.style.apply(highlight_significant, axis=1)
                st.dataframe(styled_df)

    
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
            if not accumulation_df.empty:
                st.dataframe(accumulation_df)
                
                # Visualization
                fig = px.bar(accumulation_df.head(10), x='Symbol', y='Avg Buy/Sell Ratio', 
                             color='Volume Trend', title="Top 10 Accumulation",
                             hover_data=['Price', 'Market Cap (M)', 'Avg Daily Volume'])
                st.plotly_chart(fig)
            else:
                st.write("No Accumulation detected with current filters.")

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
            if not distribution_df.empty:
                st.dataframe(distribution_df)
                
                # Visualization
                fig = px.bar(distribution_df.head(10), x='Symbol', y='Avg Buy/Sell Ratio', 
                             color='Volume Trend', title="Top 10 Distribution",
                             hover_data=['Price', 'Market Cap (M)', 'Avg Daily Volume'])
                st.plotly_chart(fig)
            else:
                st.write("No Distribution detected with current filters.")
    
    with tab4:
        st.subheader("Sector Rotation Analysis")
        col1, col2 = st.columns(2)
    
        with col1:
            rot_min_volume = st.number_input("Minimum Daily Volume (Sector Rotation)", value=50000, step=10000, format="%d", key="rot_vol")
    
        with col2:
            rot_lookback_days = st.slider("Lookback Days (Sector Rotation)", 1, 30, 10, key="rot_days")
    
            if st.button("Analyze Sector Rotation"):
                with st.spinner("Analyzing sector rotation..."):
                    # Get sector data
                    buying_df, selling_df = get_sector_rotation(lookback_days=rot_lookback_days, min_volume=rot_min_volume)
            
            # Combine dataframes for RRG
                if not buying_df.empty or not selling_df.empty:
                # Create a DataFrame for all sectors
                        all_sectors = pd.DataFrame()
                
                # First, let's print the column names to debug
                if not buying_df.empty:
                    #st.write("Buying DataFrame columns:", buying_df.columns.tolist())
                    buying_df['Direction'] = 'Buying'
                    # Use a default value for RS_Ratio if we're not sure what column to use
                    buying_df['RS_Ratio'] = 1.0  # We'll adjust this later
                    all_sectors = pd.concat([all_sectors, buying_df])
                    
                if not selling_df.empty:
                    #st.write("Selling DataFrame columns:", selling_df.columns.tolist())
                    selling_df['Direction'] = 'Selling'
                    # Use a default value for RS_Ratio if we're not sure what column to use
                    selling_df['RS_Ratio'] = -1.0  # We'll adjust this later
                    all_sectors = pd.concat([all_sectors, selling_df])
                
                # Now check what columns we have in the combined dataframe
                #st.write("Combined DataFrame columns:", all_sectors.columns.tolist())
                
                # For momentum, we'll use a default column if we're not sure what to use
                if 'Avg Daily Volume' in all_sectors.columns:
                    all_sectors['RS_Momentum'] = all_sectors['Avg Daily Volume'].fillna(0) / all_sectors['Avg Daily Volume'].max() * 100
                else:
                    # If we don't have volume, create a random value for now
                    all_sectors['RS_Momentum'] = [random.uniform(-2, 2) for _ in range(len(all_sectors))]
                
                # Create a size column for the bubbles
                if 'Avg Daily Volume' in all_sectors.columns:
                    size_column = 'Avg Daily Volume'
                else:
                    # If we don't have volume, use a constant size
                    all_sectors['Size'] = 30
                    size_column = 'Size'
                
                # Create the Relative Rotation Graph
                fig = px.scatter(
                    all_sectors,
                    x='RS_Ratio', 
                    y='RS_Momentum',
                    text='Sector',
                    color='Direction',
                    color_discrete_map={'Buying': '#4CAF50', 'Selling': '#F44336'},
                    size=size_column,
                    size_max=60,
                    hover_data=['Sector'],
                    title="RRG",
                    labels={
                        'RS_Ratio': 'Relative Strength',
                        'RS_Momentum': 'Relative Momentum'
                    }
                )
                
                # Add quadrant lines
                fig.add_shape(type="line", x0=0, y0=-40, x1=0, y1=120,
                line=dict(color="rgba(128, 128, 128, 0.7)", width=1.5, dash="dash"))
                fig.add_shape(type="line", x0=-2, y0=0, x1=2, y1=0,
                line=dict(color="rgba(128, 128, 128, 0.7)", width=1.5, dash="dash"))

                # Add subtle quadrant backgrounds
                fig.add_shape(type="rect", x0=0, y0=0, x1=2, y1=120, 
                fillcolor="rgba(144, 238, 144, 0.1)", line=dict(width=0))  # Leading
                fig.add_shape(type="rect", x0=-2, y0=0, x1=0, y1=120, 
                fillcolor="rgba(144, 200, 255, 0.1)", line=dict(width=0))  # Improving
                fig.add_shape(type="rect", x0=-2, y0=-40, x1=0, y1=0, 
                fillcolor="rgba(200, 200, 200, 0.1)", line=dict(width=0))  # Lagging
                fig.add_shape(type="rect", x0=0, y0=-40, x1=2, y1=0, 
                fillcolor="rgba(255, 182, 193, 0.1)", line=dict(width=0))  # Weakening
                
                # Annotate the quadrants
                fig.add_annotation(x=1, y=60, text="LEADING", showarrow=False, 
                                   font=dict(size=18, color="rgba(0,100,0,0.7)"))
                fig.add_annotation(x=-1, y=60, text="IMPROVING", showarrow=False, 
                                   font=dict(size=18, color="rgba(0,0,150,0.7)"))
                fig.add_annotation(x=-1, y=-20, text="LAGGING", showarrow=False, 
                                   font=dict(size=18, color="rgba(100,100,100,0.7)"))
                fig.add_annotation(x=1, y=-20, text="WEAKENING", showarrow=False, 
                                   font=dict(size=18, color="rgba(150,0,0,0.7)"))

                # Add arrows to indicate rotation direction
                fig.add_annotation(x=0, y=60, ax=1, ay=60, xref="x", yref="y",
                                   axref="x", ayref="y", showarrow=True, arrowhead=2, arrowsize=1.5,
                                   arrowwidth=2, arrowcolor="rgba(0,100,0,0.5)")
                fig.add_annotation(x=-1, y=0, ax=-1, ay=60, xref="x", yref="y",
                                   axref="x", ayref="y", showarrow=True, arrowhead=2, arrowsize=1.5,
                                   arrowwidth=2, arrowcolor="rgba(0,0,150,0.5)")
                fig.add_annotation(x=0, y=-20, ax=-1, ay=-20, xref="x", yref="y",
                                   axref="x", ayref="y", showarrow=True, arrowhead=2, arrowsize=1.5,
                                   arrowwidth=2, arrowcolor="rgba(100,100,100,0.5)")
                fig.add_annotation(x=1, y=0, ax=1, ay=-20, xref="x", yref="y",
                                   axref="x", ayref="y", showarrow=True, arrowhead=2, arrowsize=1.5,
                                   arrowwidth=2, arrowcolor="rgba(150,0,0,0.5)")
                
                # Update layout
                fig.update_traces(
                    textposition='top center',
                    textfont=dict(size=13, color='black', family="Arial, sans-serif"),
                    marker=dict(opacity=0.85, line=dict(width=1, color='#FFFFFF'))
                )

                # Update overall layout
                fig.update_layout(
                    height=800,  # Larger size for better visibility
                    width=900,
                    plot_bgcolor='rgba(240,240,240,0.3)',  # Light background
                    paper_bgcolor='white',
                    xaxis=dict(
                        title=dict(font=dict(size=14, family="Arial, sans-serif")),
                        gridcolor='rgba(200,200,200,0.2)',
                        zerolinecolor='rgba(128,128,128,0.5)',
                        zerolinewidth=1.5,
                        range=[-2.1, 2.1]  # Fixed range for better visualization
                    ),
                    yaxis=dict(
                        title=dict(font=dict(size=14, family="Arial, sans-serif")),
                        gridcolor='rgba(200,200,200,0.2)',
                        zerolinecolor='rgba(128,128,128,0.5)',
                        zerolinewidth=1.5,
                        range=[-40, 120]  # Fixed range for better visualization
                    ),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=-0.2,
                        xanchor="center",
                        x=0.5,
                        bgcolor='rgba(255,255,255,0.7)',
                        bordercolor='rgba(128,128,128,0.5)',
                        borderwidth=1
                    ),
                    title=dict(
                        text="Relative Rotation Graph (RRG) - Sector Analysis",
                        font=dict(size=22, family="Arial, sans-serif", color='#333333'),
                        x=0.5
                    ),
                    margin=dict(l=50, r=50, t=80, b=100)
                )
                
                # Display the RRG
                st.plotly_chart(fig, use_container_width=True)
                
                # Include the tables in an expander for reference
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
            
            if not portfolio_df.empty:
                # Filter the DataFrame based on the selected criteria
                portfolio_df['Date'] = pd.to_datetime(portfolio_df['Date'])
                filtered_df = portfolio_df[
                    (portfolio_df['Date'] >= pd.to_datetime(start_date)) &
                    (portfolio_df['Date'] <= pd.to_datetime(end_date)) &
                    (portfolio_df['Avg Buy/Sell Ratio'] >= min_buy_sell_ratio) &
                    (portfolio_df['Latest Volume'] >= min_total_volume)
                ]
                
                if not filtered_df.empty:
                    # Assign sectors to each symbol
                    def get_sector(symbol):
                        for sector, symbols in sector_mapping.items():
                            if symbol in symbols:
                                return sector
                        return 'User Defined'
                    
                    filtered_df['Sector'] = filtered_df['Symbol'].apply(get_sector)
                    
                    # Create tabs for each sector
                    sector_tabs = st.tabs(list(sector_mapping.keys()))
                    
                    for tab, sector in zip(sector_tabs, sector_mapping.keys()):
                        with tab:
                            sector_df = filtered_df[filtered_df['Sector'] == sector]
                            if not sector_df.empty:
                                st.dataframe(sector_df)
                                
                                # Sector-specific summary
                                sector_bought = sector_df['Bought Volume'].sum()
                                sector_sold = sector_df['Sold Volume'].sum()
                                st.write(f"Total Bought Volume: {sector_bought:,}")
                                st.write(f"Total Sold Volume: {sector_sold:,}")
                                st.write("Dark Pools: " + 
                                       ("Bullish" if sector_bought > sector_sold else "Bearish"))
                                
                                # Sector visualization
                                fig = px.bar(sector_df, x='Symbol', y='Avg Buy/Sell Ratio',
                                           title=f"{sector} Buy/Sell Ratios",
                                           color='Significant Days',
                                           hover_data=['Bought Volume', 'Sold Volume'])
                                fig.update_layout(barmode='group', xaxis_tickangle=-45)
                                st.plotly_chart(fig)
                                
                    
                    # Overall portfolio summary
                    st.write("### Overall Summary")
                    total_bought_volume = filtered_df['Bought Volume'].sum()
                    total_sold_volume = filtered_df['Sold Volume'].sum()
                    st.write(f"Total Bought Volume: {total_bought_volume:,}")
                    st.write(f"Total Sold Volume: {total_sold_volume:,}")
                    st.write("Dark Pools: " + 
                           ("Bullish" if total_bought_volume > total_sold_volume else "Bearish"))
                    
                    # Overall visualization
                    fig = px.bar(filtered_df, x='Symbol', y='Avg Buy/Sell Ratio',
                                title="Buy/Sell Ratios by Sector",
                                color='Sector',
                                hover_data=['Bought Volume', 'Sold Volume'])
                    fig.update_layout(barmode='group', xaxis_tickangle=-45)
                    st.plotly_chart(fig)
                    
                else:
                    st.write("No data available for the selected filters.")
            else:
                st.write("No data available for the selected portfolio.")
    
    with tab6:
        st.subheader("Latest Day")
        latest_df, latest_date = get_latest_data()
    
        # Add checkbox for the optional filter
        col1, col2 = st.columns(2)
        with col1:
          volume_filter = st.checkbox("Filter: Bought Volume > 2x Sold Volume", value=False)
    
        if latest_df.empty:
            st.write("No data available for the latest day.")
        else:
            st.write(f"Showing data for: {latest_date}")
        
        # Calculate metrics for each row and add them to the DataFrame
        metrics_list = []
        for _, row in latest_df.iterrows():
            total_volume = row.get('TotalVolume', 0)
            metrics = calculate_metrics(row, total_volume)
            metrics['Symbol'] = row['Symbol']
            metrics['TotalVolume'] = total_volume
            metrics_list.append(metrics)
        
        # Convert the list of metrics to a DataFrame
        latest_df_processed = pd.DataFrame(metrics_list)
        
        # Apply base filters
        latest_df_processed = latest_df_processed[latest_df_processed['total_volume'] >= 2000000]
        latest_df_processed = latest_df_processed[latest_df_processed['buy_to_sell_ratio'] > 1.5]
        
        # Apply optional filter if checkbox is selected
        if volume_filter:
            latest_df_processed = latest_df_processed[latest_df_processed['bought_volume'] > 2 * latest_df_processed['sold_volume']]
        
        # Sort by buy_to_sell_ratio and TotalVolume
        latest_df_processed = latest_df_processed.sort_values(by=['buy_to_sell_ratio', 'total_volume'], ascending=[False, False])
        
        if not latest_df_processed.empty:
            # Reorder columns with 'Symbol' as the first column
            column_order = ['Symbol', 'buy_to_sell_ratio', 'total_volume', 'bought_volume', 'sold_volume', 'short_volume_ratio']
            latest_df_processed = latest_df_processed[column_order]
            
            # Display the DataFrame
            st.dataframe(latest_df_processed)
            
            # Visualization
            fig = px.bar(latest_df_processed, x='Symbol', y='buy_to_sell_ratio',
                        title="Latest Day Buy/Sell Ratios",
                        hover_data=['total_volume', 'bought_volume', 'sold_volume'])
            fig.update_layout(barmode='group', xaxis_tickangle=-45)
            st.plotly_chart(fig)
            
        else:
            st.write("No records found with current filters.")

    with tab7:
        st.subheader("Stocks with Rising Buy/Sell Ratios (Last 10 Days)")
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        with col1:
            min_volume_rising = st.number_input("Minimum Daily Volume", value=500000, 
                                                step=100000, format="%d", key="rising_vol")
        with col2:
            lookback_days_rising = st.slider("Lookback Days", 5, 20, 10, key="rising_days")
        with col3:
            max_dips_rising = st.number_input("Max Allowed Dips", value=3, min_value=0, max_value=5, step=1, key="rising_dips")
        with col4:
            min_price_rising = st.number_input("Min Price ($)", value=5.0, min_value=1.0, max_value=50.0, step=0.5, key="rising_price")
        with col5:
            min_market_cap_rising = st.number_input("Min Market Cap ($M)", value=500, min_value=100, max_value=5000, step=100, key="rising_market_cap")
        with col6:
            if st.button("Update Stock Data"):
                with st.spinner("Updating stock data..."):
                    update_stock_database(portfolio_symbols)
                st.success("Stock data updated successfully!")
        
        if st.button("Find Rising Ratios"):
            with st.spinner("Analyzing stocks with rising ratios..."):
                rising_df = find_rising_ratio_stocks(lookback_days=lookback_days_rising, 
                                                    min_volume=min_volume_rising, 
                                                    max_dips=max_dips_rising,
                                                    min_price=min_price_rising,
                                                    min_market_cap=min_market_cap_rising * 1_000_000,
                                                    allowed_symbols=portfolio_symbols)
            
            if not rising_df.empty:
                st.dataframe(rising_df)
                
                # Visualization
                fig = px.bar(rising_df.head(10), x='Symbol', y='Ratio Increase',
                            title="Top 10 Stocks by Buy/Sell Ratio Increase",
                            hover_data=['Avg Daily Volume', 'Starting Ratio', 'Ending Ratio', 'Dips', 'Price', 'Market Cap (M)'])
                fig.update_layout(barmode='group', xaxis_tickangle=-45)
                st.plotly_chart(fig)
                
                # Line chart showing ratio progression for top 10
                # top_10_symbols = rising_df['Symbol'].head(10).tolist()
                # detailed_data = {}
                # for symbol in top_10_symbols:
                #     df, _ = analyze_symbol(symbol, lookback_days=lookback_days_rising)
                #     if not df.empty:
                #         detailed_data[symbol] = df[['date', 'buy_to_sell_ratio']]
                
                # if detailed_data:
                #     plot_df = pd.concat([df.set_index('date')['buy_to_sell_ratio'].rename(symbol) 
                #                         for symbol, df in detailed_data.items()], axis=1)
                #     fig2 = px.line(plot_df, title="Buy/Sell Ratio Progression (Top 10 Stocks)")
                #     fig2.update_layout(xaxis_title="Date", yaxis_title="Buy/Sell Ratio")
                #     st.plotly_chart(fig2)
            else:
                st.write("No stocks found with rising buy/sell ratios over the specified period.")

if __name__ == "__main__":
    run()
