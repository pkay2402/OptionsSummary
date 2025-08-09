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
    """Display the table header matching TradePulse format."""
    st.markdown("""
    <div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
    <div style="display: flex; font-weight: bold; color: #666;">
        <div style="flex: 3;">Name</div>
        <div style="flex: 1; text-align: right;">Chg.</div>
        <div style="flex: 1; text-align: right;">Score</div>
        <div style="flex: 1; text-align: right;">Momentum</div>
        <div style="flex: 1; text-align: right;">Daily</div>
        <div style="flex: 1; text-align: right;">Large Deal</div>
    </div>
    </div>
    """, unsafe_allow_html=True)

def display_flow_row(stock_data: Dict, rank: int, daily_changes: Dict):
    """Display a single flow row in TradePulse table format."""
    symbol = stock_data.get('Symbol', 'UNKNOWN')
    price = stock_data.get('Current_Price', 0.0)
    score = stock_data.get('Flow_Score', 0.0)
    details = stock_data.get('Details', {})
    
    # Debug check
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
        momentum = f"+{bias_strength:.0f}"
    elif sentiment == "BEARISH" and bias_strength > 70:
        momentum = f"-{bias_strength:.0f}"
    else:
        momentum = f"{bias_strength:.0f}"
    
    # Format large deal (total premium in thousands)
    large_deal = f"{int(total_premium/1000):,}"
    
    # Use Streamlit columns instead of HTML for better reliability
    col1, col2, col3, col4, col5, col6 = st.columns([3, 1, 1, 1, 1, 1])
    
    with col1:
        st.markdown(f"### {symbol}")
        st.caption(f"${price:.2f}")
    
    with col2:
        color = "🟢" if daily_change >= 0 else "🔴"
        st.markdown(f"**{color} {daily_change:+.2f}%**")
    
    with col3:
        st.markdown(f"**{score:.1f}**")
    
    with col4:
        emoji = '📈' if 'BULLISH' in sentiment else '📉' if 'BEARISH' in sentiment else '⚡'
        st.markdown(f"**{emoji} {momentum}**")
    
    with col5:
        color = "🟢" if daily_change >= 0 else "🔴"
        st.markdown(f"**{color} {daily_change:+.1f}%**")
    
    with col6:
        st.markdown(f"**{large_deal}**")
    
    # Expandable section for detailed strike information
    with st.expander("🎯 View Strike Details", expanded=False):
        if not top_strikes.empty:
            st.markdown("**🎯 Top Strike Activity**")
            
            for i, (_, strike_row) in enumerate(top_strikes.head(3).iterrows()):
                strike_price = strike_row['Strike Price']
                call_put = "CALL" if strike_row['Call/Put'] == 'C' else "PUT"
                expiry = strike_row['Expiration'].strftime('%m/%d/%y')
                premium = strike_row['Premium']
                volume = strike_row['Volume']
                
                move_pct = ((strike_price - price) / price) * 100
                
                # Create a clean strike row
                strike_col1, strike_col2, strike_col3 = st.columns([2, 2, 1])
                
                with strike_col1:
                    direction = "�" if call_put == "CALL" else "📉"
                    st.markdown(f"**{direction} ${strike_price:.0f} {call_put}**")
                    st.caption(f"Expires: {expiry}")
                
                with strike_col2:
                    st.metric("Premium", f"${premium/1000:.0f}K", f"{volume:,} contracts")
                
                with strike_col3:
                    move_color = "normal" if abs(move_pct) < 10 else "inverse" if move_pct < 0 else "normal"
                    st.metric("Move Needed", f"{move_pct:+.1f}%", delta_color=move_color)
                
                if i < len(top_strikes.head(3)) - 1:
                    st.divider()
            
            # Simplified metrics
            st.markdown("---")
            st.markdown("**📊 Flow Summary**")
            
            met1, met2, met3 = st.columns(3)
            with met1:
                st.metric("Total Premium", f"${total_premium/1000000:.1f}M")
            with met2:
                sentiment_color = "🟢" if sentiment == "BULLISH" else "🔴" if sentiment == "BEARISH" else "🟡"
                st.metric("Sentiment", f"{sentiment_color} {sentiment}")
            with met3:
                st.metric("Conviction", f"{bias_strength:.0f}%")
                
            # Simple score breakdown
            st.markdown("**Score Components:**")
            score_info = f"Premium: {details.get('premium_score', 0):.0f} | Conviction: {details.get('conviction_score', 0):.0f} | Volume: {details.get('volume_score', 0):.0f}"
            st.caption(score_info)

def main():
    st.set_page_config(
        page_title="Top Options Flow",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Custom CSS to match TradePulse styling
    st.markdown("""
    <style>
    .main > div {
        padding: 1rem;
    }
    .stContainer > div {
        border: 1px solid #e0e0e0;
        border-radius: 5px;
        padding: 0.5rem;
        margin: 0.2rem 0;
        background-color: #ffffff;
    }
    .stExpander > div > div {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 5px;
    }
    .stExpander [data-testid="stExpanderToggleIcon"] {
        visibility: hidden;
    }
    .element-container {
        margin: 0 !important;
        padding: 0 !important;
    }
    /* Table row styling */
    div[data-testid="stExpander"] {
        border: 1px solid #e6e6e6;
        border-radius: 4px;
        margin: 2px 0;
        background-color: #ffffff;
    }
    div[data-testid="stExpander"]:hover {
        background-color: #f8f9fa;
        border-color: #5470c6;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.title("🎯 Top Flows")
    
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
    
    # Auto-refresh every 10 minutes
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = time.time()
    
    current_time = time.time()
    if current_time - st.session_state.last_refresh > 600:  # 10 minutes
        st.session_state.last_refresh = current_time
        st.cache_data.clear()
        st.rerun()
    
    # Load and analyze data
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
            
            # Generate market insights
            insights = get_market_insights(all_flows)
            
            # Calculate daily changes for all symbols
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
    
    # Display market insights
    display_market_insights(insights)
    
    # Display results
    st.markdown("---")
    
    # Main flows table
    st.header("🎯 Top Options Flows")
    st.markdown("*Ranked by flow score - showing the most significant options activity*")
    
    display_flow_table_header()
    
    # Show top 20 flows regardless of category
    for i, stock_data in enumerate(all_flows[:20], 1):
        display_flow_row(stock_data, i, daily_changes)
    
    # Footer
    st.markdown("---")
    st.caption(f"Data refreshed: {et_now.strftime('%Y-%m-%d %H:%M:%S ET')} | Auto-refresh every 10 minutes")
    st.caption("Data source: CBOE Options Market Statistics")

if __name__ == "__main__":
    main()
