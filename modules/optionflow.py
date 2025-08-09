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
INDEX_SYMBOLS = ['SPX', 'SPXW', 'IWM', 'DIA', 'VIX', 'VIXW', 'XSP', 'RTUW']
EXCLUDED_SYMBOLS = INDEX_SYMBOLS + [
    'BRKB', 'RUT', '4SPY', 'RUTW', 'DJX', 'BFB'
] + [s for s in [] if s.startswith('$')]

# High-profile stocks for special treatment
HIGH_PROFILE_STOCKS = ['TSLA', 'AAPL', 'NVDA', 'MSFT', 'GOOGL', 'AMZN', 'META', 'SPY', 'QQQ', 'NFLX', 'AMD']

# 🎯 NEW ENHANCEMENT: Sector mapping for sector flow analysis
SECTOR_MAP = {
    # Technology
    'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology', 'GOOG': 'Technology',
    'META': 'Technology', 'NVDA': 'Technology', 'AMD': 'Technology', 'NFLX': 'Technology',
    'CRM': 'Technology', 'ORCL': 'Technology', 'ADBE': 'Technology', 'INTC': 'Technology',
    
    # Automotive
    'TSLA': 'Automotive', 'F': 'Automotive', 'GM': 'Automotive', 'RIVN': 'Automotive',
    'LCID': 'Automotive', 'NIO': 'Automotive', 'XPEV': 'Automotive',
    
    # Finance
    'JPM': 'Finance', 'BAC': 'Finance', 'GS': 'Finance', 'MS': 'Finance',
    'WFC': 'Finance', 'C': 'Finance', 'BRK.B': 'Finance',
    
    # Healthcare
    'JNJ': 'Healthcare', 'PFE': 'Healthcare', 'UNH': 'Healthcare', 'ABBV': 'Healthcare',
    'MRK': 'Healthcare', 'LLY': 'Healthcare', 'TMO': 'Healthcare',
    
    # Energy
    'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy', 'EOG': 'Energy',
    
    # Consumer
    'AMZN': 'Consumer', 'WMT': 'Consumer', 'HD': 'Consumer', 'TGT': 'Consumer',
    'COST': 'Consumer', 'NKE': 'Consumer', 'SBUX': 'Consumer',
    
    # ETFs
    'SPY': 'ETF', 'QQQ': 'ETF', 'IWM': 'ETF', 'DIA': 'ETF', 'XLF': 'ETF'
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
            alerts.append({
                'type': 'MASSIVE_FLOW',
                'symbol': symbol,
                'message': f"🚨 MASSIVE FLOW: ${details['total_premium']/1000000:.1f}M in {symbol}",
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
                alerts.append({
                    'type': 'EXTREME_BEARISH', 
                    'symbol': symbol,
                    'message': f"📉 EXTREME BEARISH: {symbol} - Heavy put buying",
                    'priority': 'MEDIUM',
                    'value': ratio
                })
        
        # High conviction flows (bias > 85%)
        if details['bias_strength'] > 85 and details['total_premium'] > 1000000:
            sentiment = details['sentiment']
            alerts.append({
                'type': 'HIGH_CONVICTION',
                'symbol': symbol,
                'message': f"💪 HIGH CONVICTION: {symbol} - {sentiment} ({details['bias_strength']:.0f}%)",
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
    """Ultra-simplified flow scoring - just rank by total premium spent."""
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
    
    if call_premium > put_premium * 1.3:
        sentiment = "BULLISH"
        bias_strength = call_premium / (call_premium + put_premium) * 100
    elif put_premium > call_premium * 1.3:
        sentiment = "BEARISH" 
        bias_strength = put_premium / (call_premium + put_premium) * 100
    else:
        sentiment = "MIXED"
        bias_strength = 60
    
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
        'top_strikes': top_strikes,
        'block_trades': block_trades  # NEW
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
        
    st.markdown("## 📊 Live Market Flow Insights")
    
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
    
    # Market status
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
    
    # Load and analyze data
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
            
            insights = get_market_insights(all_flows)
            
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
    
    # 🎯 ENHANCED: Display insights with alerts and sectors
    display_market_insights(insights, all_flows)
    
    # Main flows table
    st.header("🎯 Top Options Flows")
    st.markdown("*Enhanced with price catalysts, block trade detection, and sector analysis*")
    
    display_flow_table_header()
    
    # Show top 20 flows
    for i, stock_data in enumerate(all_flows[:20], 1):
        display_flow_row(stock_data, i, daily_changes)
    
    # 🎯 NEW: Additional insights sidebar
    with st.sidebar:
        st.markdown("## 📈 Quick Stats")
        
        if all_flows:
            total_symbols = len(all_flows)
            avg_score = sum(f['Flow_Score'] for f in all_flows) / total_symbols
            high_conviction = len([f for f in all_flows if f['Details']['bias_strength'] > 80])
            
            st.metric("Total Symbols", total_symbols)
            st.metric("Avg Flow Score", f"{avg_score:.1f}")
            st.metric("High Conviction", f"{high_conviction}")
            
            # Top sectors
            sectors = analyze_sector_flows(all_flows)
            if sectors:
                st.markdown("### 🏭 Top Sectors")
                sorted_sectors = sorted(sectors.items(), key=lambda x: x[1]['total_premium'], reverse=True)
                for sector, data in sorted_sectors[:5]:
                    st.markdown(f"**{sector}**: ${data['total_premium']/1000000:.1f}M")
    
    # Footer
    st.markdown("---")
    st.caption(f"🕒 Last updated: {et_now.strftime('%Y-%m-%d %H:%M:%S ET')} | Auto-refresh: 10 min")
    st.caption("📊 Data: CBOE | Enhanced with alerts, sectors, and catalysts")

if __name__ == "__main__":
    main()
