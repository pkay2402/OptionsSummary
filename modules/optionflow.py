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
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

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

# Exclude index symbols and problematic symbols - focus only on individual stocks
INDEX_SYMBOLS = ['SPX', 'SPXW', 'IWM', 'DIA', 'VIX', 'VIXW', 'XSP', 'RTUW']
EXCLUDED_SYMBOLS = INDEX_SYMBOLS + [
    'BRKB', 'RUT', '4SPY', 'RUTW', 'DJX', 'BFB'
] + [s for s in [] if s.startswith('$')]

# Magnificent 7 - separate these out for dedicated analysis
MAG7_SYMBOLS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'SPY', 'QQQ']

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

@st.cache_data(ttl=600)  # Cache for 10 minutes
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

        # Clean and filter data
        df = df.dropna(subset=['Symbol', 'Expiration', 'Strike Price', 'Call/Put'])
        df = df[df['Volume'] >= 50].copy()

        # Parse expiration dates
        df['Expiration'] = pd.to_datetime(df['Expiration'], errors='coerce')
        df = df.dropna(subset=['Expiration'])
        df = df[df['Expiration'].dt.date >= datetime.now().date()]

        return df
    except Exception as e:
        logger.error(f"Error fetching {url}: {e}")
        return None

@st.cache_data(ttl=600)  # Cache for 10 minutes
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

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_stock_price(symbol: str) -> Optional[float]:
    """Get current stock price with optimizations."""
    try:
        if symbol.startswith('$') or len(symbol) > 5:
            return None
            
        ticker = yf.Ticker(symbol)
        
        # Try fast method first
        try:
            info = ticker.fast_info
            price = info.get('last_price')
            if price and price > 0:
                return float(price)
        except:
            pass
        
        # Fallback to regular info
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

def calculate_flow_score(symbol_flows: pd.DataFrame, current_price: float) -> Dict:
    """Ultra-simplified flow scoring - just rank by total premium spent."""
    if symbol_flows.empty or current_price is None:
        return {'score': 0, 'details': {}}
    
    total_premium = symbol_flows['Premium'].sum()
    
    # For high-profile stocks, lower the threshold significantly
    symbol = symbol_flows['Symbol'].iloc[0] if 'Symbol' in symbol_flows.columns else 'UNKNOWN'
    high_profile = symbol in ['TSLA', 'AAPL', 'NVDA', 'AMZN', 'GOOGL', 'MSFT', 'META', 'SPY', 'QQQ']
    
    if high_profile and total_premium < 100000:  # $100K threshold for big names
        return {'score': 0, 'details': {'reason': 'Below premium threshold'}}
    elif not high_profile and total_premium < 500000:  # $500K for others  
        return {'score': 0, 'details': {'reason': 'Below premium threshold'}}
    
    # Score is simply premium in millions (scaled)
    score = total_premium / 1000000 * 10  # Each $1M = 10 points
    
    # Separate calls and puts
    calls = symbol_flows[symbol_flows['Call/Put'] == 'C']
    puts = symbol_flows[symbol_flows['Call/Put'] == 'P']
    
    call_premium = calls['Premium'].sum()
    put_premium = puts['Premium'].sum()
    total_volume = symbol_flows['Volume'].sum()
    
    # Determine sentiment
    if call_premium > put_premium * 1.3:
        sentiment = "BULLISH"
        bias_strength = call_premium / (call_premium + put_premium) * 100
    elif put_premium > call_premium * 1.3:
        sentiment = "BEARISH" 
        bias_strength = put_premium / (call_premium + put_premium) * 100
    else:
        sentiment = "MIXED"
        bias_strength = 60
    
    # Find top strikes
    strike_analysis = symbol_flows.groupby(['Strike Price', 'Call/Put', 'Expiration']).agg({
        'Premium': 'sum',
        'Volume': 'sum'
    }).reset_index()
    
    top_strikes = strike_analysis.nlargest(5, 'Premium')
    
    details = {
        'total_premium': total_premium,
        'total_volume': total_volume,
        'call_premium': call_premium,
        'put_premium': put_premium,
        'sentiment': sentiment,
        'bias_strength': bias_strength,
        'top_strikes': top_strikes,
    }
    
    return {'score': score, 'details': details}

def analyze_options_flows(df: pd.DataFrame) -> Dict[str, List[Dict]]:   
    """Analyze options flows without separating Mag7 from others."""
    if df.empty:
        return {'all_flows': []}
    
    # Exclude problematic symbols early
    df = df[~df['Symbol'].isin(EXCLUDED_SYMBOLS)].copy()
    
    # Additional filters
    df = df[df['Symbol'].str.len() <= 5]
    df = df[~df['Symbol'].str.contains(r'[\$\d]', regex=True)]
    
    # Calculate days to expiry and premium
    df['Days_to_Expiry'] = (df['Expiration'] - datetime.now()).dt.days
    df = df[(df['Days_to_Expiry'] <= 90) & (df['Days_to_Expiry'] >= 0)]
    df['Premium'] = df['Volume'] * df['Last Price'] * 100
    
    # Pre-filter by premium
    df = df[df['Premium'] >= 200000]
    
    # Get top symbols by premium (increased to 500 for better coverage)
    all_premiums = df.groupby('Symbol')['Premium'].sum().sort_values(ascending=False)
    top_symbols = all_premiums.head(500).index.tolist()
    
    # Get stock prices in batch
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
    
    # Process all symbols
    symbol_scores = []
    for symbol in top_symbols:
        if symbol not in price_cache:
            continue
            
        current_price = price_cache[symbol]
        symbol_flows = df[df['Symbol'] == symbol].copy()
        
        if symbol_flows.empty:
            continue
            
        # For high-profile stocks, include all flows (ITM + OTM)
        # For others, stick to OTM only
        high_profile_stocks = ['TSLA', 'AAPL', 'NVDA', 'MSFT', 'GOOGL', 'AMZN', 'META', 'SPY', 'QQQ', 'NFLX', 'AMD']
        
        if symbol in high_profile_stocks:
            # Include all significant flows for high-profile stocks
            analyzed_flows = symbol_flows.copy()
        else:
            # Filter for OTM only for other stocks
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
        
        # Special handling for high-profile stocks (TSLA, AAPL, etc.) - lower threshold
        high_profile_stocks = ['TSLA', 'AAPL', 'NVDA', 'MSFT', 'GOOGL', 'AMZN', 'META', 'SPY', 'QQQ']
        threshold = 15 if symbol in high_profile_stocks else 25
        
        if flow_analysis['score'] > threshold:
            symbol_scores.append({
                'Symbol': symbol,
                'Current_Price': current_price,
                'Flow_Score': flow_analysis['score'],
                'Details': flow_analysis['details'],
                'Flows': analyzed_flows
            })
    
    symbol_scores.sort(key=lambda x: x['Flow_Score'], reverse=True)
    return {'all_flows': symbol_scores}

def get_market_insights(flows_data: List[Dict]) -> Dict:
    """Generate real-time market insights from flows data."""
    if not flows_data:
        return {}
    
    insights = {}
    
    # 1. Largest single flow today
    largest_flow = max(flows_data, key=lambda x: x['Details']['total_premium'])
    insights['largest_flow'] = {
        'symbol': largest_flow['Symbol'],
        'premium': largest_flow['Details']['total_premium'],
        'sentiment': largest_flow['Details']['sentiment']
    }
    
    # 2. Most bullish flows (highest call dominance)
    bullish_flows = [f for f in flows_data if f['Details']['sentiment'] == 'BULLISH']
    if bullish_flows:
        most_bullish = max(bullish_flows, key=lambda x: x['Details']['bias_strength'])
        insights['most_bullish'] = {
            'symbol': most_bullish['Symbol'],
            'conviction': most_bullish['Details']['bias_strength']
        }
    
    # 3. Most bearish flows
    bearish_flows = [f for f in flows_data if f['Details']['sentiment'] == 'BEARISH']
    if bearish_flows:
        most_bearish = max(bearish_flows, key=lambda x: x['Details']['bias_strength'])
        insights['most_bearish'] = {
            'symbol': most_bearish['Symbol'],
            'conviction': most_bearish['Details']['bias_strength']
        }
    
    # 4. Total market premium flow
    total_premium = sum(f['Details']['total_premium'] for f in flows_data)
    insights['total_premium'] = total_premium
    
    # 5. Call vs Put ratio across all flows
    total_call_premium = sum(f['Details']['call_premium'] for f in flows_data)
    total_put_premium = sum(f['Details']['put_premium'] for f in flows_data)
    
    if total_put_premium > 0:
        call_put_ratio = total_call_premium / total_put_premium
    else:
        call_put_ratio = float('inf') if total_call_premium > 0 else 0
    
    insights['call_put_ratio'] = call_put_ratio
    insights['market_sentiment'] = 'BULLISH' if call_put_ratio > 1.5 else 'BEARISH' if call_put_ratio < 0.67 else 'MIXED'
    
    # 6. Unusual activity (flows with score > 50)
    unusual_activity = [f for f in flows_data if f['Flow_Score'] > 50]
    insights['unusual_count'] = len(unusual_activity)
    
    return insights

def display_market_insights(insights: Dict):
    """Display real-time market insights at the top of the page."""
    if not insights:
        return
        
    st.markdown("## 📊 Live Market Flow Insights")
    
    # Top row - Key metrics
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
                f"{len([i for i in insights if 'flows' in str(i)])} symbols"
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
    
    # Second row - Directional insights
    col1, col2 = st.columns(2)
    
    with col1:
        if 'most_bullish' in insights:
            st.info(f"📈 **Most Bullish**: {insights['most_bullish']['symbol']} ({insights['most_bullish']['conviction']:.0f}% conviction)")
    
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
    """Display a single flow row with proper alignment using CSS Grid."""
    symbol = stock_data.get('Symbol', 'UNKNOWN')
    price = stock_data.get('Current_Price', 0.0)
    score = stock_data.get('Flow_Score', 0.0)
    details = stock_data.get('Details', {})
    
    if not symbol or symbol == 'UNKNOWN':
        st.error(f"Missing symbol data for row {rank}")
        return
    
    sentiment = details.get('sentiment', 'MIXED')
    total_premium = details.get('total_premium', 0)
    bias_strength = details.get('bias_strength', 0)
    top_strikes = details.get('top_strikes', pd.DataFrame())
    
    # Get daily change
    daily_change = daily_changes.get(symbol, 0.0)
    
    # Calculate momentum based on sentiment and conviction
    if sentiment == "BULLISH" and bias_strength > 70:
        momentum_text = f"📈 +{bias_strength:.0f}"
        momentum_color = "#28a745"
    elif sentiment == "BEARISH" and bias_strength > 70:
        momentum_text = f"📉 -{bias_strength:.0f}"
        momentum_color = "#dc3545"
    else:
        momentum_text = f"⚡ {bias_strength:.0f}"
        momentum_color = "#ffc107"
    
    # Format values
    change_color = "#28a745" if daily_change >= 0 else "#dc3545"
    change_icon = "🟢" if daily_change >= 0 else "🔴"
    large_deal = f"{int(total_premium/1000):,}K"
    
    # Main row with proper grid alignment
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
            <strong style="font-size: 1.1em; color: #212529;">{symbol}</strong><br>
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
    
    # Strike details with reduced spacing
    with st.expander("🎯 View Strike Details", expanded=False):
        if not top_strikes.empty:
            st.markdown("#### 🎯 Top Strike Activity")
            
            # Display strikes with compact layout
            for i, (_, strike_row) in enumerate(top_strikes.head(3).iterrows()):
                strike_price = strike_row['Strike Price']
                call_put = "CALL" if strike_row['Call/Put'] == 'C' else "PUT"
                expiry = strike_row['Expiration'].strftime('%m/%d/%y')
                premium = strike_row['Premium']
                volume = strike_row['Volume']
                move_pct = ((strike_price - price) / price) * 100
                
                # Compact strike display
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
                
                # Only add separator if not the last item
                if i < min(2, len(top_strikes) - 1):
                    st.markdown("---")
            
            # Compact flow summary
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

def get_technical_analysis(symbol: str) -> Dict:
    """Get comprehensive technical analysis for a symbol."""
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="6mo")
        info = ticker.info
        
        if hist.empty:
            return {'error': 'No historical data available'}
        
        current_price = hist['Close'].iloc[-1]
        
        # Calculate technical indicators
        # Moving averages
        sma_20 = hist['Close'].rolling(20).mean().iloc[-1]
        sma_50 = hist['Close'].rolling(50).mean().iloc[-1]
        sma_200 = hist['Close'].rolling(200).mean() if len(hist) >= 200 else None
        sma_200_val = sma_200.iloc[-1] if sma_200 is not None else None
        
        # RSI calculation
        delta = hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]
        
        # Support and resistance levels
        high_52w = hist['High'].max()
        low_52w = hist['Low'].min()
        
        # Volume analysis
        avg_volume = hist['Volume'].rolling(20).mean().iloc[-1]
        current_volume = hist['Volume'].iloc[-1]
        volume_ratio = current_volume / avg_volume
        
        # Price levels
        daily_change = ((current_price - hist['Close'].iloc[-2]) / hist['Close'].iloc[-2]) * 100
        
        # Determine trend
        trend = "BULLISH" if current_price > sma_20 > sma_50 else "BEARISH" if current_price < sma_20 < sma_50 else "SIDEWAYS"
        
        return {
            'current_price': current_price,
            'daily_change': daily_change,
            'sma_20': sma_20,
            'sma_50': sma_50,
            'sma_200': sma_200_val,
            'rsi': current_rsi,
            'high_52w': high_52w,
            'low_52w': low_52w,
            'volume_ratio': volume_ratio,
            'trend': trend,
            'market_cap': info.get('marketCap'),
            'pe_ratio': info.get('trailingPE'),
            'sector': info.get('sector'),
            'industry': info.get('industry'),
            'chart_data': hist
        }
    except Exception as e:
        logger.error(f"Error getting technical analysis for {symbol}: {e}")
        return {'error': str(e)}

def get_symbol_options_data(symbol: str, df: pd.DataFrame) -> Dict:
    """Get options flow data for a specific symbol."""
    try:
        if df.empty:
            return {'error': 'No options data available'}
        
        # Filter for the symbol
        symbol_data = df[df['Symbol'] == symbol.upper()].copy()
        
        if symbol_data.empty:
            return {'error': f'No options data found for {symbol}'}
        
        # Calculate premium
        symbol_data['Premium'] = symbol_data['Volume'] * symbol_data['Last Price'] * 100
        
        # Separate calls and puts
        calls = symbol_data[symbol_data['Call/Put'] == 'C']
        puts = symbol_data[symbol_data['Call/Put'] == 'P']
        
        call_volume = calls['Volume'].sum()
        put_volume = puts['Volume'].sum()
        call_premium = calls['Premium'].sum()
        put_premium = puts['Premium'].sum()
        
        total_volume = call_volume + put_volume
        total_premium = call_premium + put_premium
        
        # Calculate sentiment
        if call_premium > put_premium * 1.3:
            sentiment = "BULLISH"
            conviction = (call_premium / total_premium) * 100
        elif put_premium > call_premium * 1.3:
            sentiment = "BEARISH"
            conviction = (put_premium / total_premium) * 100
        else:
            sentiment = "NEUTRAL"
            conviction = 60
        
        # Put/Call ratio
        pc_ratio = put_premium / call_premium if call_premium > 0 else float('inf')
        
        # Top strikes by premium
        strike_analysis = symbol_data.groupby(['Strike Price', 'Call/Put', 'Expiration']).agg({
            'Premium': 'sum',
            'Volume': 'sum'
        }).reset_index()
        
        top_strikes = strike_analysis.nlargest(10, 'Premium')
        
        # Unusual activity (high premium relative to volume)
        symbol_data['Price_per_Contract'] = symbol_data['Premium'] / symbol_data['Volume']
        expensive_flows = symbol_data[symbol_data['Price_per_Contract'] > symbol_data['Price_per_Contract'].quantile(0.8)]
        
        return {
            'total_volume': total_volume,
            'total_premium': total_premium,
            'call_volume': call_volume,
            'put_volume': put_volume,
            'call_premium': call_premium,
            'put_premium': put_premium,
            'sentiment': sentiment,
            'conviction': conviction,
            'pc_ratio': pc_ratio,
            'top_strikes': top_strikes,
            'unusual_flows': expensive_flows,
            'raw_data': symbol_data
        }
        
    except Exception as e:
        logger.error(f"Error analyzing options for {symbol}: {e}")
        return {'error': str(e)}

def display_symbol_analysis():
    """Display the symbol analysis tab."""
    st.header("🔍 Symbol Analysis")
    st.markdown("*Enter a symbol to get comprehensive options flow and technical analysis*")
    
    # Symbol input
    col1, col2 = st.columns([3, 1])
    with col1:
        symbol_input = st.text_input("Enter Symbol (e.g., AAPL, TSLA, SPY)", value="", placeholder="AAPL").upper().strip()
    
    with col2:
        analyze_button = st.button("🔍 Analyze", type="primary")
    
    if symbol_input and (analyze_button or st.session_state.get('last_analyzed_symbol') == symbol_input):
        st.session_state['last_analyzed_symbol'] = symbol_input
        
        with st.spinner(f"Analyzing {symbol_input}..."):
            # Get fresh options data
            df = fetch_all_options_data()
            
            # Get technical analysis
            technical = get_technical_analysis(symbol_input)
            
            # Get options analysis
            options_analysis = get_symbol_options_data(symbol_input, df)
            
        if 'error' in technical:
            st.error(f"Technical Analysis Error: {technical['error']}")
            return
            
        # Display results in tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Technical", "🎯 Options Flow", "📋 Strike Details"])
        
        with tab1:
            # Overview section
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Current Price", 
                    f"${technical['current_price']:.2f}",
                    f"{technical['daily_change']:+.2f}%"
                )
            
            with col2:
                if 'error' not in options_analysis:
                    st.metric(
                        "Options Sentiment",
                        options_analysis['sentiment'],
                        f"{options_analysis['conviction']:.0f}% conviction"
                    )
                else:
                    st.metric("Options Sentiment", "No Data", "")
            
            with col3:
                st.metric(
                    "Trend",
                    technical['trend'],
                    f"RSI: {technical['rsi']:.1f}"
                )
            
            with col4:
                if 'error' not in options_analysis:
                    st.metric(
                        "Total Premium",
                        f"${options_analysis['total_premium']/1000000:.1f}M",
                        f"{options_analysis['total_volume']:,} contracts"
                    )
                else:
                    st.metric("Total Premium", "No Data", "")
            
            # Key insights
            st.markdown("---")
            st.subheader("🎯 Key Insights")
            
            insights_col1, insights_col2 = st.columns(2)
            
            with insights_col1:
                st.markdown("**📈 Technical Summary**")
                
                # Price vs moving averages
                if technical['current_price'] > technical['sma_20']:
                    st.success(f"✅ Above 20-day SMA (${technical['sma_20']:.2f})")
                else:
                    st.error(f"❌ Below 20-day SMA (${technical['sma_20']:.2f})")
                
                if technical['sma_200'] and technical['current_price'] > technical['sma_200']:
                    st.success(f"✅ Above 200-day SMA (${technical['sma_200']:.2f})")
                elif technical['sma_200']:
                    st.error(f"❌ Below 200-day SMA (${technical['sma_200']:.2f})")
                
                # RSI analysis
                if technical['rsi'] > 70:
                    st.warning(f"⚠️ Overbought (RSI: {technical['rsi']:.1f})")
                elif technical['rsi'] < 30:
                    st.warning(f"⚠️ Oversold (RSI: {technical['rsi']:.1f})")
                else:
                    st.info(f"ℹ️ Neutral RSI: {technical['rsi']:.1f}")
                
                # Volume analysis
                if technical['volume_ratio'] > 2:
                    st.success(f"🔥 High Volume ({technical['volume_ratio']:.1f}x avg)")
                elif technical['volume_ratio'] > 1.5:
                    st.info(f"📊 Above Average Volume ({technical['volume_ratio']:.1f}x)")
                
            with insights_col2:
                if 'error' not in options_analysis:
                    st.markdown("**🎯 Options Summary**")
                    
                    # Sentiment analysis
                    if options_analysis['sentiment'] == 'BULLISH':
                        st.success(f"🟢 Bullish Flow ({options_analysis['conviction']:.0f}% calls)")
                    elif options_analysis['sentiment'] == 'BEARISH':
                        st.error(f"🔴 Bearish Flow ({options_analysis['conviction']:.0f}% puts)")
                    else:
                        st.info("🟡 Neutral Flow")
                    
                    # Put/Call ratio
                    if options_analysis['pc_ratio'] < 0.5:
                        st.success(f"📈 Low P/C Ratio: {options_analysis['pc_ratio']:.2f} (Bullish)")
                    elif options_analysis['pc_ratio'] > 2:
                        st.error(f"📉 High P/C Ratio: {options_analysis['pc_ratio']:.2f} (Bearish)")
                    else:
                        st.info(f"📊 P/C Ratio: {options_analysis['pc_ratio']:.2f}")
                    
                    # Volume analysis
                    total_contracts = options_analysis['total_volume']
                    if total_contracts > 10000:
                        st.success(f"🔥 High Activity: {total_contracts:,} contracts")
                    elif total_contracts > 5000:
                        st.info(f"📊 Moderate Activity: {total_contracts:,} contracts")
                
        with tab2:
            # Technical analysis tab
            st.subheader(f"📈 Technical Analysis - {symbol_input}")
            
            # Price chart
            if 'chart_data' in technical:
                fig = make_subplots(
                    rows=2, cols=1,
                    row_heights=[0.7, 0.3],
                    subplot_titles=[f'{symbol_input} Price Chart', 'Volume'],
                    vertical_spacing=0.1
                )
                
                # Candlestick chart
                fig.add_trace(
                    go.Candlestick(
                        x=technical['chart_data'].index,
                        open=technical['chart_data']['Open'],
                        high=technical['chart_data']['High'],
                        low=technical['chart_data']['Low'],
                        close=technical['chart_data']['Close'],
                        name="Price"
                    ),
                    row=1, col=1
                )
                
                # Add moving averages
                fig.add_trace(
                    go.Scatter(
                        x=technical['chart_data'].index,
                        y=technical['chart_data']['Close'].rolling(20).mean(),
                        name="20-day SMA",
                        line=dict(color='orange', width=2)
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=technical['chart_data'].index,
                        y=technical['chart_data']['Close'].rolling(50).mean(),
                        name="50-day SMA",
                        line=dict(color='red', width=2)
                    ),
                    row=1, col=1
                )
                
                # Volume chart
                colors = ['green' if close >= open else 'red' 
                         for close, open in zip(technical['chart_data']['Close'], technical['chart_data']['Open'])]
                
                fig.add_trace(
                    go.Bar(
                        x=technical['chart_data'].index,
                        y=technical['chart_data']['Volume'],
                        name="Volume",
                        marker_color=colors
                    ),
                    row=2, col=1
                )
                
                fig.update_layout(
                    title=f"{symbol_input} - 6 Month Chart",
                    xaxis_rangeslider_visible=False,
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Technical metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**📊 Price Levels**")
                st.write(f"Current: ${technical['current_price']:.2f}")
                st.write(f"52W High: ${technical['high_52w']:.2f}")
                st.write(f"52W Low: ${technical['low_52w']:.2f}")
                
                # Distance from highs/lows
                pct_from_high = ((technical['current_price'] - technical['high_52w']) / technical['high_52w']) * 100
                pct_from_low = ((technical['current_price'] - technical['low_52w']) / technical['low_52w']) * 100
                
                st.write(f"From High: {pct_from_high:+.1f}%")
                st.write(f"From Low: {pct_from_low:+.1f}%")
            
            with col2:
                st.markdown("**📈 Moving Averages**")
                st.write(f"20-day SMA: ${technical['sma_20']:.2f}")
                st.write(f"50-day SMA: ${technical['sma_50']:.2f}")
                if technical['sma_200']:
                    st.write(f"200-day SMA: ${technical['sma_200']:.2f}")
                
                st.write(f"Trend: {technical['trend']}")
            
            with col3:
                st.markdown("**🎯 Indicators**")
                st.write(f"RSI (14): {technical['rsi']:.1f}")
                st.write(f"Volume Ratio: {technical['volume_ratio']:.1f}x")
                
                if technical.get('market_cap'):
                    market_cap_b = technical['market_cap'] / 1e9
                    st.write(f"Market Cap: ${market_cap_b:.1f}B")
                
                if technical.get('pe_ratio'):
                    st.write(f"P/E Ratio: {technical['pe_ratio']:.1f}")
                
        with tab3:
            # Options flow analysis
            if 'error' in options_analysis:
                st.warning(f"No options data available for {symbol_input}")
                st.info("This could mean:")
                st.write("• No options trading activity today")
                st.write("• Symbol not optionable")
                st.write("• Data not available in CBOE feed")
                return
                
            st.subheader(f"🎯 Options Flow Analysis - {symbol_input}")
            
            # Flow metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Call Volume", f"{options_analysis['call_volume']:,}")
            
            with col2:
                st.metric("Put Volume", f"{options_analysis['put_volume']:,}")
            
            with col3:
                st.metric("Call Premium", f"${options_analysis['call_premium']/1000:.0f}K")
            
            with col4:
                st.metric("Put Premium", f"${options_analysis['put_premium']/1000:.0f}K")
            
            # Flow visualization
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                # Volume pie chart
                fig_vol = px.pie(
                    values=[options_analysis['call_volume'], options_analysis['put_volume']],
                    names=['Calls', 'Puts'],
                    title="Volume Distribution",
                    color_discrete_sequence=['#00cc96', '#ff6b6b']
                )
                st.plotly_chart(fig_vol, use_container_width=True)
            
            with col2:
                # Premium pie chart
                fig_prem = px.pie(
                    values=[options_analysis['call_premium'], options_analysis['put_premium']],
                    names=['Call Premium', 'Put Premium'],
                    title="Premium Distribution",
                    color_discrete_sequence=['#00cc96', '#ff6b6b']
                )
                st.plotly_chart(fig_prem, use_container_width=True)
            
            # Flow sentiment analysis
            st.markdown("---")
            st.subheader("📊 Flow Sentiment Analysis")
            
            sentiment_col1, sentiment_col2 = st.columns(2)
            
            with sentiment_col1:
                if options_analysis['sentiment'] == 'BULLISH':
                    st.success(f"🟢 **{options_analysis['sentiment']}** ({options_analysis['conviction']:.0f}% conviction)")
                    st.write("Call premiums dominate, suggesting upward price expectation")
                elif options_analysis['sentiment'] == 'BEARISH':
                    st.error(f"🔴 **{options_analysis['sentiment']}** ({options_analysis['conviction']:.0f}% conviction)")
                    st.write("Put premiums dominate, suggesting downward price expectation")
                else:
                    st.info(f"🟡 **{options_analysis['sentiment']}** ({options_analysis['conviction']:.0f}% conviction)")
                    st.write("Balanced call/put activity, no clear directional bias")
            
            with sentiment_col2:
                st.write(f"**Put/Call Ratio:** {options_analysis['pc_ratio']:.2f}")
                
                if options_analysis['pc_ratio'] < 0.7:
                    st.success("📈 Bullish (Low P/C ratio)")
                elif options_analysis['pc_ratio'] > 1.3:
                    st.error("📉 Bearish (High P/C ratio)")
                else:
                    st.info("📊 Neutral P/C ratio")
        
        with tab4:
            # Strike details
            if 'error' not in options_analysis and not options_analysis['top_strikes'].empty:
                st.subheader(f"📋 Strike Analysis - {symbol_input}")
                
                # Current price reference
                st.info(f"**Current Price:** ${technical['current_price']:.2f}")
                
                # Top strikes table
                strikes_df = options_analysis['top_strikes'].copy()
                strikes_df['Call/Put'] = strikes_df['Call/Put'].map({'C': 'CALL', 'P': 'PUT'})
                strikes_df['Premium ($K)'] = strikes_df['Premium'] / 1000
                strikes_df['Expiration'] = pd.to_datetime(strikes_df['Expiration']).dt.strftime('%m/%d/%y')
                
                # Calculate move needed
                strikes_df['Move Needed (%)'] = ((strikes_df['Strike Price'] - technical['current_price']) / technical['current_price']) * 100
                
                # Display table
                display_columns = ['Strike Price', 'Call/Put', 'Expiration', 'Volume', 'Premium ($K)', 'Move Needed (%)']
                st.dataframe(
                    strikes_df[display_columns].round(2),
                    use_container_width=True,
                    height=400
                )
                
                # Unusual activity
                if not options_analysis['unusual_flows'].empty:
                    st.markdown("---")
                    st.subheader("🚨 Unusual Activity")
                    st.write("*Options with unusually high premium per contract*")
                    
                    unusual_df = options_analysis['unusual_flows'].copy()
                    unusual_df = unusual_df.nlargest(5, 'Price_per_Contract')
                    unusual_df['Premium ($K)'] = unusual_df['Premium'] / 1000
                    unusual_df['Expiration'] = pd.to_datetime(unusual_df['Expiration']).dt.strftime('%m/%d/%y')
                    unusual_df['Call/Put'] = unusual_df['Call/Put'].map({'C': 'CALL', 'P': 'PUT'})
                    
                    unusual_columns = ['Strike Price', 'Call/Put', 'Expiration', 'Volume', 'Premium ($K)', 'Last Price']
                    st.dataframe(
                        unusual_df[unusual_columns].round(2),
                        use_container_width=True
                    )
            else:
                st.info(f"No detailed strike data available for {symbol_input}")

# Update the main function to include the new tab:

def main():
    st.set_page_config(
        page_title="Enhanced Options Flow",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Enhanced CSS (keep your existing CSS)
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
    
    * {
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main navigation tabs
    tab1, tab2 = st.tabs(["🏠 Market Overview", "🔍 Symbol Analysis"])
    
    with tab1:
        # Your existing main content goes here
        st.title("🎯 Top Options Flow")
        
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
        
        # Load and analyze data (your existing code)
        with st.spinner("Loading options flow data..."):
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
                
                daily_changes = {}
                with st.spinner("Calculating daily changes..."):
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
                return
        
        # Display your existing content
        display_market_insights(insights)
        
        st.markdown("---")
        st.header("🎯 Top Options Flows")
        st.markdown("*Ranked by flow score - showing the most significant options activity*")
        
        display_flow_table_header()
        
        for i, stock_data in enumerate(all_flows[:20], 1):
            display_flow_row(stock_data, i, daily_changes)
        
        # Footer
        st.markdown("---")
        st.caption(f"Data refreshed: {et_now.strftime('%Y-%m-%d %H:%M:%S ET')} | Auto-refresh every 10 minutes")
        st.caption("Data source: CBOE Options Market Statistics")
    
    with tab2:
        # New symbol analysis tab
        display_symbol_analysis()

if __name__ == "__main__":
    main()
