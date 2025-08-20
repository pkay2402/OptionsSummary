import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import sqlite3
import logging
import time
import hashlib
from io import StringIO
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor
import pytz

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
DATABASE_FILE = "streamlit_flow.db"

# Market hours configuration
US_EASTERN = pytz.timezone('US/Eastern')
MARKET_OPEN_TIME = "09:30"
MARKET_CLOSE_TIME = "16:00"
MARKET_DAYS = [0, 1, 2, 3, 4]  # Monday to Friday

# 2025 US Stock Market Holidays
US_HOLIDAYS_2025 = [
    '2025-01-01', '2025-01-20', '2025-02-17', '2025-04-18', '2025-05-26',
    '2025-06-19', '2025-07-04', '2025-09-01', '2025-11-27', '2025-12-25'
]

# Early close days
EARLY_CLOSE_DAYS_2025 = {
    '2025-07-03': '13:00',
    '2025-11-28': '13:00', 
    '2025-12-24': '13:00'
}

# Exclude index symbols and problematic symbols
INDEX_SYMBOLS = ['SPX', 'SPXW', 'IWM', 'DIA', 'VIX', 'VIXW', 'XSP', 'RUTW']
EXCLUDED_SYMBOLS = INDEX_SYMBOLS + [
    'BRKB', 'RUT', '4SPY', 'RUTW', 'DJX', 'BFB'
] + [s for s in [] if s.startswith('$')]

# High-profile stocks for special treatment
HIGH_PROFILE_STOCKS = ['TSLA', 'AAPL', 'NVDA', 'MSFT', 'GOOGL', 'AMZN', 'META', 'SPY', 'QQQ', 'NFLX', 'AMD']

# ETFs for main dashboard
MAIN_ETFS = ['SPY', 'QQQ', 'IWM', 'DIA']

# 🎯 NEW ENHANCEMENT: Sector mapping for sector flow analysis
SECTOR_MAP = {
    # Technology
    'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology', 'GOOG': 'Technology',
    'META': 'Technology', 'NVDA': 'Technology', 'AMD': 'Technology', 'NFLX': 'Technology',
    'CRM': 'Technology', 'ORCL': 'Technology', 'ADBE': 'Technology', 'INTC': 'Technology',
    'CSCO': 'Technology', 'TXN': 'Technology', 'QCOM': 'Technology', 'AVGO': 'Technology',
    'MU': 'Technology', 'AMAT': 'Technology', 'SNPS': 'Technology', 'PANW': 'Technology',

    # Automotive
    'TSLA': 'Automotive', 'F': 'Automotive', 'GM': 'Automotive', 'RIVN': 'Automotive',
    'LCID': 'Automotive', 'NIO': 'Automotive', 'XPEV': 'Automotive', 'TM': 'Automotive',
    'HMC': 'Automotive', 'STLA': 'Automotive',

    # Finance
    'JPM': 'Finance', 'BAC': 'Finance', 'GS': 'Finance', 'MS': 'Finance',
    'WFC': 'Finance', 'C': 'Finance', 'BRK.B': 'Finance', 'SCHW': 'Finance',
    'BLK': 'Finance', 'AXP': 'Finance', 'USB': 'Finance', 'PNC': 'Finance',

    # Healthcare
    'JNJ': 'Healthcare', 'PFE': 'Healthcare', 'UNH': 'Healthcare', 'ABBV': 'Healthcare',
    'MRK': 'Healthcare', 'LLY': 'Healthcare', 'TMO': 'Healthcare', 'BMY': 'Healthcare',
    'AMGN': 'Healthcare', 'CVS': 'Healthcare', 'GILD': 'Healthcare', 'ISRG': 'Healthcare',

    # Energy
    'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy', 'EOG': 'Energy',
    'SLB': 'Energy', 'HAL': 'Energy', 'PSX': 'Energy', 'MPC': 'Energy',
    'VLO': 'Energy', 'OXY': 'Energy',

    # Consumer Discretionary
    'AMZN': 'Consumer Discretionary', 'WMT': 'Consumer Staples', 'HD': 'Consumer Discretionary',
    'TGT': 'Consumer Discretionary', 'COST': 'Consumer Staples', 'NKE': 'Consumer Discretionary',
    'SBUX': 'Consumer Discretionary', 'LOW': 'Consumer Discretionary', 'MCD': 'Consumer Discretionary',
    'BKNG': 'Consumer Discretionary', 'DPZ': 'Consumer Discretionary',

    # Consumer Staples
    'PG': 'Consumer Staples', 'KO': 'Consumer Staples', 'PEP': 'Consumer Staples',
    'MDLZ': 'Consumer Staples', 'CL': 'Consumer Staples', 'KMB': 'Consumer Staples',

    # Industrials
    'HON': 'Industrials', 'UNP': 'Industrials', 'UPS': 'Industrials', 'BA': 'Industrials',
    'CAT': 'Industrials', 'DE': 'Industrials', 'LMT': 'Industrials', 'GE': 'Industrials',

    # Utilities
    'NEE': 'Utilities', 'DUK': 'Utilities', 'SO': 'Utilities', 'D': 'Utilities',
    'EXC': 'Utilities', 'AEP': 'Utilities',

    # Materials
    'LIN': 'Materials', 'APD': 'Materials', 'ECL': 'Materials', 'SHW': 'Materials',
    'NEM': 'Materials', 'FCX': 'Materials',

    # Real Estate
    'PLD': 'Real Estate', 'AMT': 'Real Estate', 'CCI': 'Real Estate', 'SPG': 'Real Estate',
    'EQIX': 'Real Estate',

    # Communication Services
    'DIS': 'Communication Services', 'VZ': 'Communication Services', 'T': 'Communication Services',
    'CMCSA': 'Communication Services', 'CHTR': 'Communication Services',

    # Crypto/Blockchain
    'COIN': 'Crypto', 'MARA': 'Crypto', 'RIOT': 'Crypto', 'HUT': 'Crypto',
    'BTBT': 'Crypto', 'BITF': 'Crypto', 'GBTC': 'Crypto',

    # Nuclear/Clean Energy
    'CCJ': 'Nuclear', 'UEC': 'Nuclear', 'SMR': 'Nuclear', 'BWXT': 'Nuclear',
    'PLUG': 'Clean Energy', 'BLDP': 'Clean Energy', 'FCEL': 'Clean Energy',
    'ENPH': 'Clean Energy', 'SEDG': 'Clean Energy', 'RUN': 'Clean Energy',

    # Aerospace/Defense
    'RTX': 'Aerospace & Defense', 'LMT': 'Aerospace & Defense', 'NOC': 'Aerospace & Defense',
    'GD': 'Aerospace & Defense', 'BA': 'Aerospace & Defense',

    # ETFs
    'SPY': 'ETF', 'QQQ': 'ETF', 'IWM': 'ETF', 'DIA': 'ETF', 'XLF': 'ETF',
    'XLK': 'ETF', 'XLE': 'ETF', 'XLV': 'ETF', 'XLY': 'ETF', 'XLP': 'ETF',
    'XLI': 'ETF', 'XLU': 'ETF', 'XLB': 'ETF', 'XLRE': 'ETF', 'XLC': 'ETF',
    'ARKK': 'ETF', 'ARKW': 'ETF', 'ARKG': 'ETF', 'VTI': 'ETF', 'VOO': 'ETF',

    # Other
    'FDX': 'Logistics', 'Z': 'Real Estate', 'SQ': 'Fintech', 'PYPL': 'Fintech',
    'SHOP': 'E-Commerce', 'ROKU': 'Technology', 'TWLO': 'Technology', 'SNOW': 'Technology',
    'DOCU': 'Technology', 'ZM': 'Technology', 'CRWD': 'Technology', 'DDOG': 'Technology'
}

def is_market_open():
    """Check if the US market is currently open."""
    try:
        et_now = datetime.now(US_EASTERN)
        today_str = et_now.strftime('%Y-%m-%d')

        if today_str in US_HOLIDAYS_2025:
            return False

        if et_now.weekday() not in MARKET_DAYS:
            return False

        market_close_time = EARLY_CLOSE_DAYS_2025.get(today_str, MARKET_CLOSE_TIME)
        market_open = datetime.strptime(MARKET_OPEN_TIME, "%H:%M").time()
        market_close = datetime.strptime(market_close_time, "%H:%M").time()
        current_time = et_now.time()

        return market_open <= current_time <= market_close

    except Exception as e:
        logger.error(f"Error checking market hours: {e}")
        return False

@st.cache_data(ttl=600)
def fetch_data_from_url(url: str) -> Optional[pd.DataFrame]:
    """Fetch and process data from a single URL."""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        csv_data = StringIO(response.text)
        df = pd.read_csv(csv_data)

        required_columns = ['Symbol', 'Call/Put', 'Expiration', 'Strike Price', 'Volume', 'Last Price']
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            logger.error(f"Missing columns: {missing}")
            return None

        df = df.dropna(subset=['Symbol', 'Expiration', 'Strike Price', 'Call/Put'])
        df = df[df['Volume'] >= 50].copy()

        df['Expiration'] = pd.to_datetime(df['Expiration'], errors='coerce')
        df = df.dropna(subset=['Expiration'])
        df = df[df['Expiration'].dt.date >= datetime.now().date()]

        return df
    except Exception as e:
        logger.error(f"Error fetching {url}: {e}")
        return None

@st.cache_data(ttl=600)
def fetch_all_options_data() -> pd.DataFrame:
    """Fetch and combine data from multiple CBOE URLs."""
    urls = [
        "https://www.cboe.com/us/options/market_statistics/symbol_data/csv/?mkt=cone",
        "https://www.cboe.com/us/options/market_statistics/symbol_data/csv/?mkt=opt", 
        "https://www.cboe.com/us/options/market_statistics/symbol_data/csv/?mkt=ctwo",
        "https://www.cboe.com/us/options/market_statistics/symbol_data/csv/?mkt=exo"
    ]
    
    data_frames = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(fetch_data_from_url, url) for url in urls]
        for future in futures:
            df = future.result()
            if df is not None and not df.empty:
                data_frames.append(df)
                
    return pd.concat(data_frames, ignore_index=True) if data_frames else pd.DataFrame()

@st.cache_data(ttl=300)
def get_stock_price(symbol: str) -> Optional[float]:
    """Get current stock price with optimizations."""
    try:
        if symbol.startswith('$') or len(symbol) > 5:
            return None
            
        ticker = yf.Ticker(symbol)
        
        try:
            info = ticker.fast_info
            price = info.get('last_price')
            if price and price > 0:
                return float(price)
        except:
            pass
        
        try:
            info = ticker.info
            price = (info.get('currentPrice') or 
                    info.get('regularMarketPrice') or 
                    info.get('previousClose'))
            
            if price and price > 0:
                return float(price)
        except:
            pass
            
        return None
    except Exception:
        return None

# 🎯 NEW ENHANCEMENT: Flow alerts detection
def detect_flow_alerts(flows_data: List[Dict]) -> List[Dict]:
    """Detect unusual flow patterns that warrant alerts."""
    alerts = []
    
    for flow in flows_data:
        symbol = flow['Symbol']
        details = flow['Details']
        
        # Massive flow alert (>$5M)
        if details['total_premium'] > 5000000:
            sentiment = details.get('sentiment', 'MIXED')
            bias_strength = details.get('bias_strength', 0)
            direction_quality = details.get('direction_quality', 'LOW')
            call_premium = details.get('call_premium', 0)
            put_premium = details.get('put_premium', 0)
            
            # Quality indicators for CBOE interpretation limitations
            quality_emoji = "🎯" if direction_quality == "HIGH" else "⚠️" if direction_quality == "MEDIUM" else "❓"
            
            # Determine direction with more detail
            if sentiment == "BULLISH":
                direction_text = f" ({quality_emoji}📈 {bias_strength:.0f}% BULLISH)"
            elif sentiment == "BEARISH":
                direction_text = f" ({quality_emoji}📉 {bias_strength:.0f}% BEARISH)"
            else:
                # For mixed sentiment, still show directional bias if significant
                if call_premium > 0 and put_premium > 0:
                    ratio = call_premium / put_premium
                    call_percentage = call_premium / (call_premium + put_premium) * 100
                    put_percentage = put_premium / (call_premium + put_premium) * 100
                    
                    if call_percentage >= 60:  # 60%+ calls = lean bullish
                        direction_text = f" (❓📈 {call_percentage:.0f}% LEAN BULLISH)"
                    elif put_percentage >= 60:  # 60%+ puts = lean bearish
                        direction_text = f" (❓📉 {put_percentage:.0f}% LEAN BEARISH)"
                    elif ratio > 1.1:
                        direction_text = f" (⚡ {ratio:.1f}:1 C/P)"
                    elif ratio < 0.9:
                        direction_text = f" (⚡ 1:{1/ratio:.1f} P/C)"
                    else:
                        direction_text = f" (⚡ MIXED)"
                else:
                    direction_text = " (⚡ MIXED)"
            
            alerts.append({
                'type': 'MASSIVE_FLOW',
                'symbol': symbol,
                'message': f"🚨 MASSIVE FLOW: ${details['total_premium']/1000000:.1f}M in {symbol}{direction_text}",
                'priority': 'HIGH',
                'value': details['total_premium']
            })
        
        # Unusual call/put imbalance
        if details['call_premium'] > 0 and details['put_premium'] > 0:
            ratio = details['call_premium'] / details['put_premium']
            if ratio > 10:
                alerts.append({
                    'type': 'EXTREME_BULLISH',
                    'symbol': symbol,
                    'message': f"📈 EXTREME BULLISH: {symbol} - {ratio:.1f}:1 call/put ratio",
                    'priority': 'MEDIUM',
                    'value': ratio
                })
            elif ratio < 0.1:
                put_call_ratio = 1 / ratio
                alerts.append({
                    'type': 'EXTREME_BEARISH', 
                    'symbol': symbol,
                    'message': f"📉 EXTREME BEARISH: {symbol} - {put_call_ratio:.1f}:1 put/call ratio",
                    'priority': 'MEDIUM',
                    'value': ratio
                })
        
        # High conviction flows (bias > 85%)
        if details['bias_strength'] > 85 and details['total_premium'] > 1000000:
            sentiment = details['sentiment']
            premium_text = f"${details['total_premium']/1000000:.1f}M"
            alerts.append({
                'type': 'HIGH_CONVICTION',
                'symbol': symbol,
                'message': f"💪 HIGH CONVICTION: {symbol} - {sentiment} ({details['bias_strength']:.0f}%) - {premium_text}",
                'priority': 'MEDIUM',
                'value': details['bias_strength']
            })
    
    # Sort by priority and value
    priority_order = {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}
    alerts.sort(key=lambda x: (priority_order.get(x['priority'], 0), x['value']), reverse=True)
    
    return alerts

# 🎯 NEW ENHANCEMENT: Sector flow analysis
def analyze_sector_flows(flows_data: List[Dict]) -> Dict:
    """Analyze flows by sector to identify sector rotation."""
    sector_flows = {}
    
    for flow in flows_data:
        symbol = flow['Symbol']
        sector = SECTOR_MAP.get(symbol, 'Other')
        
        if sector not in sector_flows:
            sector_flows[sector] = {
                'total_premium': 0,
                'symbols': [],
                'call_premium': 0,
                'put_premium': 0,
                'avg_score': 0
            }
        
        details = flow['Details']
        sector_flows[sector]['total_premium'] += details['total_premium']
        sector_flows[sector]['call_premium'] += details['call_premium']
        sector_flows[sector]['put_premium'] += details['put_premium']
        sector_flows[sector]['symbols'].append(symbol)
        sector_flows[sector]['avg_score'] += flow['Flow_Score']
    
    # Calculate averages and sentiment
    for sector in sector_flows:
        count = len(sector_flows[sector]['symbols'])
        sector_flows[sector]['avg_score'] /= count
        
        call_prem = sector_flows[sector]['call_premium']
        put_prem = sector_flows[sector]['put_premium']
        
        if call_prem > put_prem * 1.5:
            sector_flows[sector]['sentiment'] = 'BULLISH'
        elif put_prem > call_prem * 1.5:
            sector_flows[sector]['sentiment'] = 'BEARISH'
        else:
            sector_flows[sector]['sentiment'] = 'MIXED'
    
    return sector_flows

# 🎯 NEW ENHANCEMENT: Block trade detection
def detect_block_trades(symbol_flows: pd.DataFrame, symbol: str) -> List[Dict]:
    """Detect potential block trades and unusual activity."""
    blocks = []
    
    if symbol_flows.empty:
        return blocks
    
    # Group by strike, expiry, and type to find concentrated activity
    grouped = symbol_flows.groupby(['Strike Price', 'Expiration', 'Call/Put']).agg({
        'Volume': 'sum',
        'Premium': 'sum'
    }).reset_index()
    
    # Look for unusually large single strike activity
    for _, row in grouped.iterrows():
        if row['Volume'] > 1000 and row['Premium'] > 1000000:  # 1000+ contracts, $1M+ premium
            avg_price = row['Premium'] / (row['Volume'] * 100)
            blocks.append({
                'symbol': symbol,
                'strike': row['Strike Price'],
                'expiry': row['Expiration'],
                'type': 'CALL' if row['Call/Put'] == 'C' else 'PUT',
                'volume': row['Volume'],
                'premium': row['Premium'],
                'avg_price': avg_price
            })
    
    return sorted(blocks, key=lambda x: x['premium'], reverse=True)[:3]

# 🎯 NEW ENHANCEMENT: Price catalyst detection
def check_price_catalysts(symbol: str, current_price: float) -> List[str]:
    """Check if stock is near key technical levels."""
    catalysts = []
    
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="3mo")
        
        if len(hist) > 60:
            # Check if near 52-week high/low
            high_52w = hist['High'].max()
            low_52w = hist['Low'].min()
            
            if current_price > high_52w * 0.98:
                catalysts.append("📈 Near 52W High")
            elif current_price < low_52w * 1.02:
                catalysts.append("📉 Near 52W Low")
            
            # Check for breakout patterns
            sma_20 = hist['Close'].tail(20).mean()
            sma_50 = hist['Close'].tail(50).mean()
            
            if current_price > sma_20 * 1.05:
                catalysts.append("🚀 Above 20-day SMA")
            
            if sma_20 > sma_50 and current_price > sma_50:
                catalysts.append("⬆️ Golden Cross Setup")
            
            # Volume analysis
            avg_volume = hist['Volume'].tail(20).mean()
            recent_volume = hist['Volume'].tail(5).mean()
            
            if recent_volume > avg_volume * 2:
                catalysts.append("📊 High Volume")
                
    except Exception as e:
        logger.warning(f"Error checking catalysts for {symbol}: {e}")
    
    return catalysts

def calculate_flow_score(symbol_flows: pd.DataFrame, current_price: float) -> Dict:
    """Enhanced flow scoring with buy/sell direction inference using multiple heuristics."""
    if symbol_flows.empty or current_price is None:
        return {'score': 0, 'details': {}}
    
    total_premium = symbol_flows['Premium'].sum()
    
    symbol = symbol_flows['Symbol'].iloc[0] if 'Symbol' in symbol_flows.columns else 'UNKNOWN'
    high_profile = symbol in HIGH_PROFILE_STOCKS
    
    if high_profile and total_premium < 100000:
        return {'score': 0, 'details': {'reason': 'Below premium threshold'}}
    elif not high_profile and total_premium < 500000:
        return {'score': 0, 'details': {'reason': 'Below premium threshold'}}
    
    score = total_premium / 1000000 * 10
    
    calls = symbol_flows[symbol_flows['Call/Put'] == 'C']
    puts = symbol_flows[symbol_flows['Call/Put'] == 'P']
    
    call_premium = calls['Premium'].sum()
    put_premium = puts['Premium'].sum()
    total_volume = symbol_flows['Volume'].sum()
    
    # 🎯 ENHANCED: Multi-factor direction inference due to CBOE data limitations
    direction_signals = []
    confidence_factors = []
    
    # Factor 1: Premium weighting (basic)
    premium_ratio = call_premium / put_premium if put_premium > 0 else float('inf')
    if premium_ratio > 1.5:
        direction_signals.append("BULLISH")
        confidence_factors.append(("Premium Ratio", min(premium_ratio, 5) / 5 * 30))
    elif premium_ratio < 0.67:
        direction_signals.append("BEARISH") 
        confidence_factors.append(("Premium Ratio", min(1/premium_ratio, 5) / 5 * 30))
    
    # Factor 2: Strike price analysis (OTM vs ITM activity)
    otm_calls = calls[calls['Strike Price'] > current_price * 1.02]  # >2% OTM
    itm_calls = calls[calls['Strike Price'] <= current_price * 0.98]  # >2% ITM
    otm_puts = puts[puts['Strike Price'] < current_price * 0.98]  # >2% OTM  
    itm_puts = puts[puts['Strike Price'] >= current_price * 1.02]  # >2% ITM
    
    otm_call_premium = otm_calls['Premium'].sum()
    itm_call_premium = itm_calls['Premium'].sum()
    otm_put_premium = otm_puts['Premium'].sum()
    itm_put_premium = itm_puts['Premium'].sum()
    
    # OTM calls usually bought (bullish), ITM puts usually bought (bearish)
    speculative_bullish = otm_call_premium
    speculative_bearish = otm_put_premium + itm_put_premium
    
    if speculative_bullish > speculative_bearish * 1.3:
        direction_signals.append("BULLISH")
        confidence_factors.append(("OTM Activity", 25))
    elif speculative_bearish > speculative_bullish * 1.3:
        direction_signals.append("BEARISH")
        confidence_factors.append(("OTM Activity", 25))
    
    # Factor 3: Unusual volume patterns (big trades likely institutional)
    unusual_threshold = symbol_flows['Volume'].quantile(0.8)
    unusual_flows = symbol_flows[symbol_flows['Volume'] > unusual_threshold]
    
    if not unusual_flows.empty:
        unusual_calls = unusual_flows[unusual_flows['Call/Put'] == 'C']['Premium'].sum()
        unusual_puts = unusual_flows[unusual_flows['Call/Put'] == 'P']['Premium'].sum()
        
        if unusual_calls > unusual_puts * 1.2:
            direction_signals.append("BULLISH")
            confidence_factors.append(("Large Trades", 20))
        elif unusual_puts > unusual_calls * 1.2:
            direction_signals.append("BEARISH")
            confidence_factors.append(("Large Trades", 20))
    
    # Factor 4: Time to expiration patterns
    near_term = symbol_flows[symbol_flows['Days_to_Expiry'] <= 30]
    if not near_term.empty:
        nt_call_premium = near_term[near_term['Call/Put'] == 'C']['Premium'].sum()
        nt_put_premium = near_term[near_term['Call/Put'] == 'P']['Premium'].sum()
        
        # Near-term OTM calls often speculative buying
        nt_otm_calls = near_term[
            (near_term['Call/Put'] == 'C') & 
            (near_term['Strike Price'] > current_price * 1.05)
        ]['Premium'].sum()
        
        if nt_otm_calls > nt_put_premium * 0.8:
            direction_signals.append("BULLISH")
            confidence_factors.append(("Near-term OTM", 15))
    
    # Aggregate signals and determine confidence
    bullish_count = direction_signals.count("BULLISH")
    bearish_count = direction_signals.count("BEARISH")
    total_confidence = sum([factor[1] for factor in confidence_factors])
    
    if bullish_count > bearish_count:
        sentiment = "BULLISH"
        bias_strength = min(95, 55 + total_confidence)
        direction_quality = "HIGH" if total_confidence > 50 else "MEDIUM" if total_confidence > 25 else "LOW"
    elif bearish_count > bullish_count:
        sentiment = "BEARISH"
        bias_strength = min(95, 55 + total_confidence)
        direction_quality = "HIGH" if total_confidence > 50 else "MEDIUM" if total_confidence > 25 else "LOW"
    else:
        sentiment = "MIXED"
        bias_strength = 60
        direction_quality = "LOW"
    
    strike_analysis = symbol_flows.groupby(['Strike Price', 'Call/Put', 'Expiration']).agg({
        'Premium': 'sum',
        'Volume': 'sum'
    }).reset_index()
    
    top_strikes = strike_analysis.nlargest(5, 'Premium')
    
    # 🎯 NEW: Add block trades detection
    block_trades = detect_block_trades(symbol_flows, symbol)
    
    details = {
        'total_premium': total_premium,
        'total_volume': total_volume,
        'call_premium': call_premium,
        'put_premium': put_premium,
        'sentiment': sentiment,
        'bias_strength': bias_strength,
        'direction_quality': direction_quality,  # NEW: Confidence level
        'confidence_factors': confidence_factors,  # NEW: What drove the decision
        'interpretation_note': "⚠️ CBOE data lacks buy/sell direction - analysis based on heuristics",  # NEW
        'otm_call_premium': otm_call_premium,  # NEW: More granular data
        'otm_put_premium': otm_put_premium,  # NEW
        'speculative_bias': "BULLISH" if speculative_bullish > speculative_bearish else "BEARISH",  # NEW
        'top_strikes': top_strikes,
        'block_trades': block_trades
    }
    
    return {'score': score, 'details': details}

def analyze_options_flows(df: pd.DataFrame) -> Dict[str, List[Dict]]:   
    """Analyze options flows without separating Mag7 from others."""
    if df.empty:
        return {'all_flows': []}
    
    df = df[~df['Symbol'].isin(EXCLUDED_SYMBOLS)].copy()
    df = df[df['Symbol'].str.len() <= 5]
    df = df[~df['Symbol'].str.contains(r'[\$\d]', regex=True)]
    
    df['Days_to_Expiry'] = (df['Expiration'] - datetime.now()).dt.days
    df = df[(df['Days_to_Expiry'] <= 90) & (df['Days_to_Expiry'] >= 0)]
    df['Premium'] = df['Volume'] * df['Last Price'] * 100
    
    df = df[df['Premium'] >= 200000]
    
    all_premiums = df.groupby('Symbol')['Premium'].sum().sort_values(ascending=False)
    top_symbols = all_premiums.head(500).index.tolist()
    
    price_cache = {}
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_symbol = {executor.submit(get_stock_price, symbol): symbol for symbol in top_symbols}
        for future in future_to_symbol:
            symbol = future_to_symbol[future]
            try:
                price = future.result(timeout=5)
                if price:
                    price_cache[symbol] = price
            except Exception:
                continue
    
    symbol_scores = []
    for symbol in top_symbols:
        if symbol not in price_cache:
            continue
            
        current_price = price_cache[symbol]
        symbol_flows = df[df['Symbol'] == symbol].copy()
        
        if symbol_flows.empty:
            continue
            
        if symbol in HIGH_PROFILE_STOCKS:
            analyzed_flows = symbol_flows.copy()
        else:
            otm_calls = symbol_flows[
                (symbol_flows['Call/Put'] == 'C') & 
                (symbol_flows['Strike Price'] > current_price)
            ]
            otm_puts = symbol_flows[
                (symbol_flows['Call/Put'] == 'P') & 
                (symbol_flows['Strike Price'] < current_price)
            ]
            analyzed_flows = pd.concat([otm_calls, otm_puts], ignore_index=True)
        
        if analyzed_flows.empty:
            continue
            
        flow_analysis = calculate_flow_score(analyzed_flows, current_price)
        
        threshold = 15 if symbol in HIGH_PROFILE_STOCKS else 25
        
        if flow_analysis['score'] > threshold:
            # 🎯 NEW: Add price catalysts
            catalysts = check_price_catalysts(symbol, current_price)
            
            symbol_scores.append({
                'Symbol': symbol,
                'Current_Price': current_price,
                'Flow_Score': flow_analysis['score'],
                'Details': flow_analysis['details'],
                'Flows': analyzed_flows,
                'Catalysts': catalysts  # NEW
            })
    
    symbol_scores.sort(key=lambda x: x['Flow_Score'], reverse=True)
    return {'all_flows': symbol_scores}

def get_market_insights(flows_data: List[Dict]) -> Dict:
    """Generate real-time market insights from flows data."""
    if not flows_data:
        return {}
    
    insights = {}
    
    largest_flow = max(flows_data, key=lambda x: x['Details']['total_premium'])
    insights['largest_flow'] = {
        'symbol': largest_flow['Symbol'],
        'premium': largest_flow['Details']['total_premium'],
        'sentiment': largest_flow['Details']['sentiment']
    }
    
    bullish_flows = [f for f in flows_data if f['Details']['sentiment'] == 'BULLISH']
    if bullish_flows:
        most_bullish = max(bullish_flows, key=lambda x: x['Details']['bias_strength'])
        insights['most_bullish'] = {
            'symbol': most_bullish['Symbol'],
            'conviction': most_bullish['Details']['bias_strength']
        }
    
    bearish_flows = [f for f in flows_data if f['Details']['sentiment'] == 'BEARISH']
    if bearish_flows:
        most_bearish = max(bearish_flows, key=lambda x: x['Details']['bias_strength'])
        insights['most_bearish'] = {
            'symbol': most_bearish['Symbol'],
            'conviction': most_bearish['Details']['bias_strength']
        }
    
    total_premium = sum(f['Details']['total_premium'] for f in flows_data)
    insights['total_premium'] = total_premium
    
    total_call_premium = sum(f['Details']['call_premium'] for f in flows_data)
    total_put_premium = sum(f['Details']['put_premium'] for f in flows_data)
    
    if total_put_premium > 0:
        call_put_ratio = total_call_premium / total_put_premium
    else:
        call_put_ratio = float('inf') if total_call_premium > 0 else 0
    
    insights['call_put_ratio'] = call_put_ratio
    insights['market_sentiment'] = 'BULLISH' if call_put_ratio > 1.5 else 'BEARISH' if call_put_ratio < 0.67 else 'MIXED'
    
    unusual_activity = [f for f in flows_data if f['Flow_Score'] > 50]
    insights['unusual_count'] = len(unusual_activity)
    
    return insights

# 🎯 NEW ENHANCEMENT: Enhanced insights display with alerts and sectors
def display_market_insights(insights: Dict, flows_data: List[Dict]):
    """Display enhanced market insights with alerts and sector analysis."""
    if not insights:
        return
        
    #st.markdown("## 📊 Live Market Flow Insights")
    
    # 🚨 NEW: Flow Alerts Section
    alerts = detect_flow_alerts(flows_data)
    if alerts:
        st.markdown("### 🚨 Flow Alerts")
        alert_cols = st.columns(min(3, len(alerts)))
        
        for i, alert in enumerate(alerts[:3]):
            with alert_cols[i]:
                if alert['priority'] == 'HIGH':
                    st.error(alert['message'])
                elif alert['priority'] == 'MEDIUM':
                    st.warning(alert['message'])
                else:
                    st.info(alert['message'])
        
        # Add interpretation disclaimer
        st.info("💡 **Confidence Indicators**: 🎯 = High confidence | ⚠️ = Medium confidence | ❓ = Low confidence (CBOE data lacks buy/sell direction)")
    
    st.divider()
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if 'largest_flow' in insights:
            st.metric(
                "🎯 Largest Flow",
                f"{insights['largest_flow']['symbol']}",
                f"${insights['largest_flow']['premium']/1000000:.1f}M"
            )
    
    with col2:
        if 'total_premium' in insights:
            st.metric(
                "💰 Total Flow",
                f"${insights['total_premium']/1000000:.1f}M",
                f"{len(flows_data)} symbols"
            )
    
    with col3:
        if 'call_put_ratio' in insights:
            ratio = insights['call_put_ratio']
            if ratio == float('inf'):
                ratio_text = "ALL CALLS"
            elif ratio == 0:
                ratio_text = "ALL PUTS"
            else:
                ratio_text = f"{ratio:.2f}"
            
            st.metric(
                "📈 Call/Put Ratio",
                ratio_text,
                f"{insights.get('market_sentiment', 'MIXED')}"
            )
    
    with col4:
        if 'unusual_count' in insights:
            st.metric(
                "🚨 Unusual Activity",
                f"{insights['unusual_count']} stocks",
                "High conviction flows"
            )
    
    # 🏭 NEW: Sector Analysis
    sector_data = analyze_sector_flows(flows_data)
    if len(sector_data) > 1:
        st.markdown("### 🏭 Sector Flow Analysis")
        
        # Sort sectors by total premium
        sorted_sectors = sorted(sector_data.items(), key=lambda x: x[1]['total_premium'], reverse=True)
        
        sector_cols = st.columns(min(4, len(sorted_sectors)))
        for i, (sector, data) in enumerate(sorted_sectors[:4]):
            with sector_cols[i]:
                sentiment_emoji = "🟢" if data['sentiment'] == 'BULLISH' else "🔴" if data['sentiment'] == 'BEARISH' else "🟡"
                st.metric(
                    f"{sentiment_emoji} {sector}",
                    f"${data['total_premium']/1000000:.1f}M",
                    f"{len(data['symbols'])} symbols"
                )
    
    # Directional insights
    col1, col2 = st.columns(2)
    
    with col1:
        if 'most_bullish' in insights:
            st.success(f"📈 **Most Bullish**: {insights['most_bullish']['symbol']} ({insights['most_bullish']['conviction']:.0f}% conviction)")
    
    with col2:
        if 'most_bearish' in insights:
            st.error(f"📉 **Most Bearish**: {insights['most_bearish']['symbol']} ({insights['most_bearish']['conviction']:.0f}% conviction)")
    
    st.divider()

def calculate_daily_change(symbol: str, current_price: float) -> float:
    """Calculate daily price change percentage for a symbol."""
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="2d")
        if len(hist) >= 2:
            yesterday_close = hist['Close'].iloc[-2]
            today_price = current_price
            return ((today_price - yesterday_close) / yesterday_close) * 100
        return 0.0
    except:
        return 0.0

def display_flow_table_header():
    """Display the table header with proper alignment."""
    st.markdown("""
    <div style="
        display: grid; 
        grid-template-columns: 2fr 1fr 1fr 1fr 1fr 1fr; 
        gap: 10px; 
        padding: 15px; 
        background-color: #f8f9fa; 
        border-radius: 8px; 
        margin-bottom: 5px;
        font-weight: bold;
        color: #495057;
        border-bottom: 2px solid #dee2e6;
    ">
        <div>Name</div>
        <div style="text-align: center;">Chg.</div>
        <div style="text-align: center;">Score</div>
        <div style="text-align: center;">Momentum</div>
        <div style="text-align: center;">Daily</div>
        <div style="text-align: center;">Large Deal</div>
    </div>
    """, unsafe_allow_html=True)

def display_flow_row(stock_data: Dict, rank: int, daily_changes: Dict):
    """Display enhanced flow row with catalysts and block trades."""
    symbol = stock_data.get('Symbol', 'UNKNOWN')
    price = stock_data.get('Current_Price', 0.0)
    score = stock_data.get('Flow_Score', 0.0)
    details = stock_data.get('Details', {})
    catalysts = stock_data.get('Catalysts', [])  # NEW
    
    if not symbol or symbol == 'UNKNOWN':
        st.error(f"Missing symbol data for row {rank}")
        return
    
    sentiment = details.get('sentiment', 'MIXED')
    total_premium = details.get('total_premium', 0)
    bias_strength = details.get('bias_strength', 0)
    top_strikes = details.get('top_strikes', pd.DataFrame())
    block_trades = details.get('block_trades', [])  # NEW
    
    daily_change = daily_changes.get(symbol, 0.0)
    
    if sentiment == "BULLISH" and bias_strength > 70:
        momentum_text = f"📈 +{bias_strength:.0f}"
        momentum_color = "#28a745"
    elif sentiment == "BEARISH" and bias_strength > 70:
        momentum_text = f"📉 -{bias_strength:.0f}"
        momentum_color = "#dc3545"
    else:
        momentum_text = f"⚡ {bias_strength:.0f}"
        momentum_color = "#ffc107"
    
    change_color = "#28a745" if daily_change >= 0 else "#dc3545"
    change_icon = "🟢" if daily_change >= 0 else "🔴"
    large_deal = f"{int(total_premium/1000):,}K"
    
    # Add catalyst indicators to symbol display
    catalyst_indicators = ""
    if catalysts:
        catalyst_indicators = f" {''.join(catalysts[:2])}"
    
    st.markdown(f"""
    <div style="
        display: grid; 
        grid-template-columns: 2fr 1fr 1fr 1fr 1fr 1fr; 
        gap: 10px; 
        padding: 15px; 
        background-color: white; 
        border: 1px solid #dee2e6;
        border-radius: 8px; 
        margin-bottom: 2px;
        align-items: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    ">
        <div>
            <strong style="font-size: 1.1em; color: #212529;">{symbol}</strong>{catalyst_indicators}<br>
            <small style="color: #6c757d;">${price:.2f}</small>
        </div>
        <div style="text-align: center;">
            <span style="color: {change_color}; font-weight: bold;">
                {change_icon} {daily_change:+.2f}%
            </span>
        </div>
        <div style="text-align: center; font-weight: bold; color: #495057;">
            {score:.1f}
        </div>
        <div style="text-align: center;">
            <span style="color: {momentum_color}; font-weight: bold;">
                {momentum_text}
            </span>
        </div>
        <div style="text-align: center;">
            <span style="color: {change_color}; font-weight: bold;">
                {daily_change:+.1f}%
            </span>
        </div>
        <div style="text-align: center; font-weight: bold; color: #495057;">
            {large_deal}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 🎯 ENHANCED: Strike details with catalysts and block trades
    with st.expander("🎯 View Strike Details", expanded=False):
        
        # 🎯 NEW: Price Catalysts
        if catalysts:
            st.markdown("#### 🎯 Price Catalysts")
            catalyst_text = " | ".join(catalysts)
            st.info(f"**{symbol}** - {catalyst_text}")
            st.markdown("---")
        
        # 🎯 NEW: Block Trades
        if block_trades:
            st.markdown("#### 🏢 Block Trades Detected")
            for block in block_trades[:2]:
                st.markdown(f"""
                **{block['type']} ${block['strike']:.0f}** - {block['volume']:,} contracts
                - Premium: ${block['premium']/1000:.0f}K | Avg Price: ${block['avg_price']:.2f}
                """)
            st.markdown("---")
        
        if not top_strikes.empty:
            st.markdown("#### 🎯 Top Strike Activity")
            
            for i, (_, strike_row) in enumerate(top_strikes.head(3).iterrows()):
                strike_price = strike_row['Strike Price']
                call_put = "CALL" if strike_row['Call/Put'] == 'C' else "PUT"
                expiry = strike_row['Expiration'].strftime('%m/%d/%y')
                premium = strike_row['Premium']
                volume = strike_row['Volume']
                move_pct = ((strike_price - price) / price) * 100
                
                strike_col1, strike_col2, strike_col3 = st.columns([2, 2, 1])
                
                with strike_col1:
                    emoji = '📈' if call_put == 'CALL' else '📉'
                    st.markdown(f"**{emoji} ${strike_price:.0f} {call_put}**")
                    st.caption(f"Expires: {expiry}")
                
                with strike_col2:
                    st.markdown(f"**${premium/1000:.0f}K**")
                    st.caption(f"{volume:,} contracts")
                
                with strike_col3:
                    color = "🔴" if move_pct < 0 else "🟢" if move_pct > 0 else "⚫"
                    st.markdown(f"**{color} {move_pct:+.1f}%**")
                    st.caption("Move needed")
                
                if i < min(2, len(top_strikes) - 1):
                    st.markdown("---")
            
            st.markdown("---")
            st.markdown("#### 📊 Flow Summary")
            
            summary_col1, summary_col2, summary_col3 = st.columns(3)
            
            with summary_col1:
                st.markdown(f"**${total_premium/1000000:.1f}M**")
                st.caption("Total Premium")
            
            with summary_col2:
                sentiment_emoji = "🟢" if sentiment == "BULLISH" else "🔴" if sentiment == "BEARISH" else "🟡"
                st.markdown(f"**{sentiment_emoji} {sentiment}**")
                st.caption("Sentiment")
            
            with summary_col3:
                st.markdown(f"**{bias_strength:.0f}%**")
                st.caption("Conviction")

@st.cache_data(ttl=300)
def get_technical_analysis(symbol: str) -> Dict:
    """Get comprehensive technical analysis for a symbol."""
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="6mo")
        
        if len(hist) < 20:
            return {}
        
        current_price = hist['Close'].iloc[-1]
        
        # Moving averages
        sma_20 = hist['Close'].rolling(20).mean().iloc[-1]
        sma_50 = hist['Close'].rolling(50).mean().iloc[-1]
        sma_200 = hist['Close'].rolling(200).mean().iloc[-1] if len(hist) >= 200 else None
        
        # RSI calculation
        delta = hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]
        
        # Volume analysis
        avg_volume = hist['Volume'].rolling(20).mean().iloc[-1]
        current_volume = hist['Volume'].iloc[-1]
        volume_ratio = current_volume / avg_volume
        
        # Price levels
        high_52w = hist['High'].max()
        low_52w = hist['Low'].min()
        high_20d = hist['High'].tail(20).max()
        low_20d = hist['Low'].tail(20).min()
        
        # Volatility
        returns = hist['Close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252) * 100  # Annualized volatility
        
        # Support and resistance levels
        recent_highs = hist['High'].tail(60).nlargest(5).mean()
        recent_lows = hist['Low'].tail(60).nsmallest(5).mean()
        
        analysis = {
            'current_price': current_price,
            'sma_20': sma_20,
            'sma_50': sma_50,
            'sma_200': sma_200,
            'rsi': current_rsi,
            'volume_ratio': volume_ratio,
            'high_52w': high_52w,
            'low_52w': low_52w,
            'high_20d': high_20d,
            'low_20d': low_20d,
            'volatility': volatility,
            'support_level': recent_lows,
            'resistance_level': recent_highs,
            'price_data': hist
        }
        
        # Technical signals
        signals = []
        if current_price > sma_20 > sma_50:
            signals.append("📈 Above key moving averages")
        elif current_price < sma_20 < sma_50:
            signals.append("📉 Below key moving averages")
        
        if current_rsi > 70:
            signals.append("⚠️ Overbought (RSI > 70)")
        elif current_rsi < 30:
            signals.append("🔄 Oversold (RSI < 30)")
        
        if volume_ratio > 2:
            signals.append("📊 High volume surge")
        elif volume_ratio < 0.5:
            signals.append("📊 Low volume")
        
        if current_price > high_52w * 0.98:
            signals.append("🚀 Near 52-week high")
        elif current_price < low_52w * 1.02:
            signals.append("💥 Near 52-week low")
        
        analysis['signals'] = signals
        return analysis
        
    except Exception as e:
        logger.error(f"Error getting technical analysis for {symbol}: {e}")
        return {}

@st.cache_data(ttl=300)
def get_individual_symbol_flows(df: pd.DataFrame, symbol: str) -> Dict:
    """Get comprehensive options flow analysis for individual symbol."""
    symbol_flows = df[df['Symbol'] == symbol].copy()
    
    if symbol_flows.empty:
        return {}
    
    current_price = get_stock_price(symbol)
    if not current_price:
        return {}
    
    symbol_flows['Premium'] = symbol_flows['Volume'] * symbol_flows['Last Price'] * 100
    symbol_flows['Days_to_Expiry'] = (symbol_flows['Expiration'] - datetime.now()).dt.days
    
    # Categorize options
    calls = symbol_flows[symbol_flows['Call/Put'] == 'C'].copy()
    puts = symbol_flows[symbol_flows['Call/Put'] == 'P'].copy()
    
    # Moneyness categorization
    calls['Moneyness'] = calls['Strike Price'] / current_price
    puts['Moneyness'] = current_price / puts['Strike Price']
    
    calls['Category'] = calls['Moneyness'].apply(
        lambda x: 'ITM' if x < 1 else 'ATM' if 0.95 <= x <= 1.05 else 'OTM'
    )
    puts['Category'] = puts['Moneyness'].apply(
        lambda x: 'ITM' if x < 1 else 'ATM' if 0.95 <= x <= 1.05 else 'OTM'
    )
    
    # Time to expiration buckets
    def dte_bucket(days):
        if days <= 7:
            return 'Weekly'
        elif days <= 30:
            return 'Monthly'
        elif days <= 90:
            return 'Quarterly'
        else:
            return 'Long-term'
    
    symbol_flows['DTE_Bucket'] = symbol_flows['Days_to_Expiry'].apply(dte_bucket)
    
    # Flow metrics
    total_call_volume = calls['Volume'].sum()
    total_put_volume = puts['Volume'].sum()
    total_call_premium = calls['Premium'].sum()
    total_put_premium = puts['Premium'].sum()
    
    call_put_ratio = total_call_volume / max(total_put_volume, 1)
    premium_ratio = total_call_premium / max(total_put_premium, 1)
    
    # Top strikes by premium
    top_call_strikes = calls.groupby(['Strike Price', 'Expiration']).agg({
        'Volume': 'sum',
        'Premium': 'sum'
    }).reset_index().nlargest(5, 'Premium')
    
    top_put_strikes = puts.groupby(['Strike Price', 'Expiration']).agg({
        'Volume': 'sum',
        'Premium': 'sum'
    }).reset_index().nlargest(5, 'Premium')
    
    # Gamma exposure levels
    gamma_levels = {}
    for strike in symbol_flows['Strike Price'].unique():
        strike_volume = symbol_flows[symbol_flows['Strike Price'] == strike]['Volume'].sum()
        if strike_volume > 100:  # Significant volume threshold
            gamma_levels[strike] = strike_volume
    
    analysis = {
        'symbol': symbol,
        'current_price': current_price,
        'total_volume': symbol_flows['Volume'].sum(),
        'total_premium': symbol_flows['Premium'].sum(),
        'call_volume': total_call_volume,
        'put_volume': total_put_volume,
        'call_premium': total_call_premium,
        'put_premium': total_put_premium,
        'call_put_ratio': call_put_ratio,
        'premium_ratio': premium_ratio,
        'top_call_strikes': top_call_strikes,
        'top_put_strikes': top_put_strikes,
        'flows_by_dte': symbol_flows.groupby('DTE_Bucket')['Premium'].sum().to_dict(),
        'flows_by_moneyness': {
            'calls': calls.groupby('Category')['Premium'].sum().to_dict(),
            'puts': puts.groupby('Category')['Premium'].sum().to_dict()
        },
        'gamma_levels': gamma_levels,
        'raw_flows': symbol_flows
    }
    
    return analysis

def display_individual_symbol_analysis():
    """Display individual symbol analysis tab."""
    st.header("🔍 Individual Symbol Analysis")
    
    # Symbol selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_symbol = st.text_input(
            "Enter Symbol", 
            value="AAPL",
            help="Enter a stock symbol for detailed analysis"
        ).upper().strip()
    
    with col2:
        analyze_button = st.button("🔍 Analyze", type="primary")
    
    if not selected_symbol or (not analyze_button and 'analyzed_symbol' not in st.session_state):
        st.info("👆 Enter a symbol and click Analyze to get detailed options flow and technical analysis")
        return
    
    if analyze_button:
        st.session_state.analyzed_symbol = selected_symbol
    
    symbol = st.session_state.get('analyzed_symbol', selected_symbol)
    
    with st.spinner(f"🔍 Analyzing {symbol}..."):
        # Get data
        df = fetch_all_options_data()
        technical_data = get_technical_analysis(symbol)
        flows_data = get_individual_symbol_flows(df, symbol)
        
        if not technical_data and not flows_data:
            st.error(f"No data available for {symbol}")
            return
        
        # Display results
        st.subheader(f"📊 {symbol} Analysis")
        
        # Key metrics row
        if technical_data:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                current_price = technical_data.get('current_price', 0)
                st.metric(
                    "Current Price",
                    f"${current_price:.2f}",
                    help="Current stock price"
                )
            
            with col2:
                rsi = technical_data.get('rsi', 0)
                rsi_color = "🔴" if rsi > 70 else "🟢" if rsi < 30 else "🟡"
                st.metric(
                    "RSI",
                    f"{rsi:.1f} {rsi_color}",
                    help="Relative Strength Index"
                )
            
            with col3:
                volume_ratio = technical_data.get('volume_ratio', 1)
                volume_text = f"{volume_ratio:.1f}x avg"
                st.metric(
                    "Volume",
                    volume_text,
                    help="Current volume vs 20-day average"
                )
            
            with col4:
                volatility = technical_data.get('volatility', 0)
                st.metric(
                    "Volatility",
                    f"{volatility:.1f}%",
                    help="Annualized volatility"
                )
    
    # Create tabs for detailed analysis
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Technical", "🎯 Options Flow", "📊 Strike Analysis", "🎪 Sentiment"])
    
    with tab1:
        if technical_data:
            display_technical_analysis(symbol, technical_data)
        else:
            st.warning("Technical data not available")
    
    with tab2:
        if flows_data:
            display_options_flow_analysis(flows_data)
        else:
            st.warning("Options flow data not available")
    
    with tab3:
        if flows_data:
            display_strike_analysis(flows_data)
        else:
            st.warning("Strike analysis data not available")
    
    with tab4:
        display_sentiment_analysis(symbol, technical_data, flows_data)

def display_technical_analysis(symbol: str, data: Dict):
    """Display technical analysis section."""
    st.markdown("### 📈 Technical Analysis")
    
    # Price levels
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎯 Key Levels")
        current_price = data.get('current_price', 0)
        support = data.get('support_level', 0)
        resistance = data.get('resistance_level', 0)
        high_52w = data.get('high_52w', 0)
        low_52w = data.get('low_52w', 0)
        
        st.markdown(f"""
        - **Current**: ${current_price:.2f}
        - **Support**: ${support:.2f}
        - **Resistance**: ${resistance:.2f}
        - **52W High**: ${high_52w:.2f}
        - **52W Low**: ${low_52w:.2f}
        """)
    
    with col2:
        st.markdown("#### 📊 Moving Averages")
        sma_20 = data.get('sma_20', 0)
        sma_50 = data.get('sma_50', 0)
        sma_200 = data.get('sma_200')
        
        # Fix None formatting issue
        sma_200_text = f"${sma_200:.2f}" if sma_200 is not None else "N/A"
        
        st.markdown(f"""
        - **20-day SMA**: ${sma_20:.2f}
        - **50-day SMA**: ${sma_50:.2f}
        - **200-day SMA**: {sma_200_text}
        """)
    
    # Technical signals
    signals = data.get('signals', [])
    if signals:
        st.markdown("#### 🚨 Technical Signals")
        for signal in signals:
            st.markdown(f"- {signal}")

def display_options_flow_analysis(data: Dict):
    """Display options flow analysis section."""
    st.markdown("### 🎯 Options Flow Analysis")
    
    symbol = data.get('symbol', '')
    total_premium = data.get('total_premium', 0)
    call_put_ratio = data.get('call_put_ratio', 0)
    premium_ratio = data.get('premium_ratio', 0)
    
    # Flow metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Total Premium",
            f"${total_premium/1000000:.1f}M",
            help="Total options premium traded"
        )
    
    with col2:
        st.metric(
            "Call/Put Ratio",
            f"{call_put_ratio:.2f}",
            help="Volume ratio of calls to puts"
        )
    
    with col3:
        sentiment = "BULLISH" if premium_ratio > 1.5 else "BEARISH" if premium_ratio < 0.67 else "MIXED"
        sentiment_emoji = "📈" if sentiment == "BULLISH" else "📉" if sentiment == "BEARISH" else "⚡"
        st.metric(
            "Flow Sentiment",
            f"{sentiment_emoji} {sentiment}",
            help="Based on premium ratio"
        )
    
    # Flow breakdown by time
    flows_by_dte = data.get('flows_by_dte', {})
    if flows_by_dte:
        st.markdown("#### ⏰ Flow by Time to Expiration")
        
        dte_cols = st.columns(len(flows_by_dte))
        for i, (bucket, premium) in enumerate(flows_by_dte.items()):
            with dte_cols[i]:
                st.metric(
                    bucket,
                    f"${premium/1000:.0f}K",
                    help=f"Premium in {bucket.lower()} options"
                )

def display_strike_analysis(data: Dict):
    """Display strike analysis section."""
    st.markdown("### 📊 Strike Analysis")
    
    current_price = data.get('current_price', 0)
    top_call_strikes = data.get('top_call_strikes', pd.DataFrame())
    top_put_strikes = data.get('top_put_strikes', pd.DataFrame())
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📈 Top Call Strikes")
        if not top_call_strikes.empty:
            for _, row in top_call_strikes.head(5).iterrows():
                strike = row['Strike Price']
                volume = row['Volume']
                premium = row['Premium']
                expiry = row['Expiration'].strftime('%m/%d')
                move_needed = ((strike - current_price) / current_price) * 100
                
                st.markdown(f"""
                **${strike:.0f}** ({expiry}) - {move_needed:+.1f}%
                - Volume: {volume:,} | Premium: ${premium/1000:.0f}K
                """)
        else:
            st.info("No significant call activity")
    
    with col2:
        st.markdown("#### 📉 Top Put Strikes")
        if not top_put_strikes.empty:
            for _, row in top_put_strikes.head(5).iterrows():
                strike = row['Strike Price']
                volume = row['Volume']
                premium = row['Premium']
                expiry = row['Expiration'].strftime('%m/%d')
                move_needed = ((current_price - strike) / current_price) * 100
                
                st.markdown(f"""
                **${strike:.0f}** ({expiry}) - {move_needed:+.1f}%
                - Volume: {volume:,} | Premium: ${premium/1000:.0f}K
                """)
        else:
            st.info("No significant put activity")
    
    # Gamma levels
    gamma_levels = data.get('gamma_levels', {})
    if gamma_levels:
        st.markdown("#### ⚡ Key Gamma Levels")
        sorted_gamma = sorted(gamma_levels.items(), key=lambda x: x[1], reverse=True)
        
        gamma_text = []
        for strike, volume in sorted_gamma[:8]:
            distance = abs(strike - current_price) / current_price * 100
            gamma_text.append(f"${strike:.0f} ({volume:,} vol, {distance:.1f}% away)")
        
        st.markdown(" | ".join(gamma_text))

def display_sentiment_analysis(symbol: str, technical_data: Dict, flows_data: Dict):
    """Display sentiment analysis section."""
    st.markdown("### 🎪 Sentiment Analysis")
    
    # Overall sentiment score
    sentiment_factors = []
    
    # Technical sentiment
    if technical_data:
        signals = technical_data.get('signals', [])
        bullish_signals = sum(1 for s in signals if '📈' in s or '🚀' in s)
        bearish_signals = sum(1 for s in signals if '📉' in s or '💥' in s)
        
        if bullish_signals > bearish_signals:
            sentiment_factors.append(("Technical", "BULLISH", bullish_signals - bearish_signals))
        elif bearish_signals > bullish_signals:
            sentiment_factors.append(("Technical", "BEARISH", bearish_signals - bullish_signals))
        else:
            sentiment_factors.append(("Technical", "NEUTRAL", 0))
    
    # Options sentiment
    if flows_data:
        premium_ratio = flows_data.get('premium_ratio', 1)
        if premium_ratio > 2:
            sentiment_factors.append(("Options", "VERY BULLISH", 3))
        elif premium_ratio > 1.5:
            sentiment_factors.append(("Options", "BULLISH", 2))
        elif premium_ratio < 0.5:
            sentiment_factors.append(("Options", "VERY BEARISH", -3))
        elif premium_ratio < 0.67:
            sentiment_factors.append(("Options", "BEARISH", -2))
        else:
            sentiment_factors.append(("Options", "MIXED", 0))
    
    # Display sentiment factors
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Sentiment Breakdown")
        for factor, sentiment, strength in sentiment_factors:
            if sentiment in ["VERY BULLISH", "BULLISH"]:
                emoji = "🟢"
            elif sentiment in ["VERY BEARISH", "BEARISH"]:
                emoji = "🔴"
            else:
                emoji = "🟡"
            
            st.markdown(f"**{factor}**: {emoji} {sentiment}")
    
    with col2:
        st.markdown("#### 🎯 Key Catalysts")
        if technical_data:
            catalysts = check_price_catalysts(symbol, technical_data.get('current_price', 0))
            if catalysts:
                for catalyst in catalysts:
                    st.markdown(f"- {catalyst}")
            else:
                st.info("No immediate catalysts detected")
    
    # Overall sentiment score
    total_score = sum(strength for _, _, strength in sentiment_factors)
    
    if total_score >= 3:
        overall_sentiment = "🟢 VERY BULLISH"
    elif total_score >= 1:
        overall_sentiment = "🟢 BULLISH"
    elif total_score <= -3:
        overall_sentiment = "🔴 VERY BEARISH"
    elif total_score <= -1:
        overall_sentiment = "🔴 BEARISH"
    else:
        overall_sentiment = "🟡 MIXED"
    
    st.markdown(f"### 🎯 Overall Sentiment: {overall_sentiment}")

def filter_flows_by_category(all_flows: List[Dict], category: str) -> List[Dict]:
    """Filter flows by category (main_etfs, high_profile, others)."""
    if category == "main_etfs":
        return [f for f in all_flows if f['Symbol'] in MAIN_ETFS]
    elif category == "high_profile":
        return [f for f in all_flows if f['Symbol'] in HIGH_PROFILE_STOCKS and f['Symbol'] not in MAIN_ETFS]
    elif category == "others":
        return [f for f in all_flows if f['Symbol'] not in HIGH_PROFILE_STOCKS]
    else:
        return all_flows

def display_flows_tab(flows_data: List[Dict], daily_changes: Dict, tab_title: str, show_insights: bool = True):
    """Display flows for a specific tab with optional insights."""
    if not flows_data:
        st.info(f"No significant flows detected for {tab_title}")
        return
    
    if show_insights:
        insights = get_market_insights(flows_data)
        display_market_insights(insights, flows_data)
    
    # Main flows table
    st.header(f"🎯 {tab_title} Options Flows")
    
    display_flow_table_header()
    
    # Show flows (limit based on category)
    max_flows = 10 if tab_title.startswith("📊") else 20
    for i, stock_data in enumerate(flows_data[:max_flows], 1):
        display_flow_row(stock_data, i, daily_changes)

def main():
    st.set_page_config(
        page_title="Enhanced Options Flow",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Enhanced CSS
    st.markdown("""
    <style>
    .main > div {
        padding: 1rem;
    }
    
    @media (max-width: 768px) {
        .main > div {
            padding: 0.5rem;
        }
        .stColumns > div {
            min-width: 100% !important;
        }
    }
    
    .stExpander {
        border: 1px solid #e6e6e6 !important;
        border-radius: 4px !important;
        margin: 4px 0 !important;
        background-color: #ffffff !important;
    }
    
    .stExpander:hover {
        background-color: #f8f9fa !important;
        border-color: #5470c6 !important;
    }
    
    * {
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header with enhanced branding
    st.title("🚀 Enhanced Options Flow Scanner")
    st.markdown("*Real-time flow analysis with alerts, sector insights, and catalyst detection*")
    
    # Market status row
    market_open = is_market_open()
    market_status = "🟢 MARKET OPEN" if market_open else "🔴 MARKET CLOSED"
    et_now = datetime.now(US_EASTERN)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.metric("Status", market_status)
    with col2:
        st.metric("Time (ET)", et_now.strftime('%H:%M:%S'))
    with col3:
        if st.button("🔄 Refresh Data", type="primary"):
            st.cache_data.clear()
            st.rerun()
    
    # Auto-refresh logic
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = time.time()
    
    current_time = time.time()
    if current_time - st.session_state.last_refresh > 600:
        st.session_state.last_refresh = current_time
        st.cache_data.clear()
        st.rerun()
    
    # Load and analyze data once
    with st.spinner("🔍 Analyzing market flows..."):
        try:
            df = fetch_all_options_data()
            
            if df.empty:
                st.error("No options data available")
                return
                
            results = analyze_options_flows(df)
            all_flows = results['all_flows']
            
            if not all_flows:
                st.warning("No significant flows detected")
                return
            
            # Calculate daily changes
            daily_changes = {}
            with st.spinner("📊 Calculating market metrics..."):
                with ThreadPoolExecutor(max_workers=10) as executor:
                    future_to_symbol = {
                        executor.submit(calculate_daily_change, stock['Symbol'], stock['Current_Price']): stock['Symbol'] 
                        for stock in all_flows
                    }
                    for future in future_to_symbol:
                        symbol = future_to_symbol[future]
                        try:
                            change = future.result(timeout=3)
                            daily_changes[symbol] = change
                        except:
                            daily_changes[symbol] = 0.0
            
        except Exception as e:
            st.error(f"Error loading data: {e}")
            logger.error(f"Data loading error: {e}")
            return
    
    # Create the new three-tab structure
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Main ETFs (SPY,QQQ,IWM,DIA)", 
        "⭐ High Profile Stocks", 
        "🔍 All Other Stocks",
        "🔎 Individual Analysis"
    ])
    
    with tab1:
        main_etf_flows = filter_flows_by_category(all_flows, "main_etfs")
        display_flows_tab(main_etf_flows, daily_changes, "📊 Main ETFs", show_insights=True)
    
    with tab2:
        high_profile_flows = filter_flows_by_category(all_flows, "high_profile")
        display_flows_tab(high_profile_flows, daily_changes, "⭐ High Profile Stocks", show_insights=True)
    
    with tab3:
        other_flows = filter_flows_by_category(all_flows, "others")
        display_flows_tab(other_flows, daily_changes, "🔍 Other Stocks", show_insights=True)
    
    with tab4:
        display_individual_symbol_analysis()
    
    # Footer with enhanced data limitations disclaimer
    st.markdown("---")
    
    # Data limitations disclaimer
    with st.expander("⚠️ Important: Data Interpretation Limitations", expanded=False):
        st.markdown("""
        **CBOE Data Limitations & Our Analysis Methods:**
        
        📊 **What CBOE Provides:**
        - Volume and premium data for options contracts
        - Strike prices, expirations, and open interest
        - Call vs Put identification
        
        ❌ **What's Missing (Critical!):**
        - **Buy vs Sell direction** - We don't know if options were bought or sold
        - **Market maker vs retail** - Can't distinguish order flow type
        - **Opening vs closing positions** - Unknown if creating or closing trades
        
        🎯 **Our Interpretation Methods:**
        - **OTM Call Activity**: Likely speculative buying (bullish)
        - **Large Premium Flows**: May indicate institutional activity
        - **Strike Clustering**: Suggests directional expectations
        - **Time to Expiry**: Near-term OTM often speculative
        
        📈 **Confidence Levels:**
        - 🎯 **High**: Multiple signals align (50+ confidence points)
        - ⚠️ **Medium**: Some signals present (25-50 points)  
        - ❓ **Low**: Unclear signals or conflicting data
        
        ⚠️ **Use Caution**: All directional interpretations are probabilistic estimates based on market patterns, not definitive buy/sell data.
        """)
    
    st.caption(f"🕒 Last updated: {et_now.strftime('%Y-%m-%d %H:%M:%S ET')} | Auto-refresh: 10 min")
    st.caption("📊 Data: CBOE | Enhanced analysis with multi-factor direction inference")

if __name__ == "__main__":
    main()
