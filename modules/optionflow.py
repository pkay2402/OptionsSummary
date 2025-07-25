import pandas as pd
import requests
from io import StringIO
from datetime import datetime
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor
import streamlit as st
import yfinance as yf
import time
from streamlit_autorefresh import st_autorefresh
import plotly.express as px

REFRESH_INTERVAL = 600  # 10 minutes in seconds

@st.cache_data(ttl=600)
def fetch_data_from_url(url: str) -> Optional[pd.DataFrame]:
    """Fetch and process data from a single URL."""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        if 'text/csv' not in response.headers.get('Content-Type', ''):
            return None

        csv_data = StringIO(response.text)
        df = pd.read_csv(csv_data)

        # Apply basic filters
        if 'Volume' not in df.columns:
            return None
        df = df[df['Volume'] >= 100].copy()

        if 'Expiration' in df.columns:
            df['Expiration'] = pd.to_datetime(df['Expiration'], errors='coerce')
            df = df.dropna(subset=['Expiration'])
            df = df[df['Expiration'].dt.date >= datetime.now().date()]

        return df
    except (requests.RequestException, pd.errors.ParserError):
        return None

@st.cache_data(ttl=600)
def fetch_data_from_urls(urls: List[str]) -> pd.DataFrame:
    """Fetch and combine data from multiple CSV URLs in parallel."""
    data_frames = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(fetch_data_from_url, url) for url in urls]
        for future in futures:
            df = future.result()
            if df is not None and not df.empty:
                data_frames.append(df)
    return pd.concat(data_frames, ignore_index=True) if data_frames else pd.DataFrame()

def filter_options_flow(df: pd.DataFrame, exclude_symbols: List[str], days_limit: int = 45, min_premium: int = 300000) -> pd.DataFrame:
    """Filter and group options flow data."""
    if df.empty:
        return pd.DataFrame()

    required_columns = ['Symbol', 'Strike Price', 'Call/Put', 'Expiration', 'Volume', 'Last Price']
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        return pd.DataFrame()

    df = df.copy()
    if exclude_symbols:
        df = df[~df['Symbol'].isin(exclude_symbols)]

    df['Days_to_Expiry'] = (df['Expiration'] - datetime.now()).dt.days
    df = df[(df['Days_to_Expiry'] <= days_limit) & (df['Days_to_Expiry'] >= 0)]

    df['Premium'] = df['Volume'] * df['Last Price'] * 100
    df = df[df['Premium'] >= min_premium]

    if df.empty:
        return pd.DataFrame()

    grouped = df.groupby(['Symbol', 'Strike Price', 'Call/Put']).agg({
        'Expiration': lambda x: list(x.unique()),
        'Volume': 'sum',
        'Premium': 'sum',
        'Last Price': 'mean',
        'Days_to_Expiry': lambda x: list(x.unique())
    }).reset_index()

    grouped['Expiry_Count'] = grouped['Expiration'].apply(len)
    grouped['Expiry_Dates'] = grouped['Expiration'].apply(
        lambda x: ', '.join(sorted(pd.to_datetime(d).strftime('%Y-%m-%d') for d in x if not pd.isna(d)))
    )
    grouped['Days_to_Expiry_List'] = grouped['Days_to_Expiry'].apply(lambda x: ', '.join(map(str, sorted(set(x)))))

    grouped = grouped.sort_values('Premium', ascending=False)

    result_df = pd.DataFrame({
        'Symbol': grouped['Symbol'],
        'Strike Price': grouped['Strike Price'],
        'Call/Put': grouped['Call/Put'],
        'Total Volume': grouped['Volume'],
        'Total Premium ($)': grouped['Premium'],
        'Avg Price': grouped['Last Price'].round(2),
        'Expiry Count': grouped['Expiry_Count'],
        'Expiry Dates': grouped['Expiry_Dates'],
        'Days to Expiry': grouped['Days_to_Expiry_List'],
        'Pattern Type': grouped['Expiry_Count'].apply(lambda x: 'Same Strike Multiple Expiry' if x > 1 else 'Single Expiry'),
        'Expiration': grouped['Expiration'].apply(lambda x: x[0] if x else pd.NaT)
    })

    return result_df

def filter_otm_flows(df: pd.DataFrame, prices_df: pd.DataFrame) -> pd.DataFrame:
    """Filter for only OTM (Out of The Money) flows."""
    if df.empty or prices_df.empty:
        return df
    
    # Merge flow data with current prices
    merged = df.merge(prices_df[['Symbol', 'Close']], on='Symbol', how='left')
    
    # Filter for OTM flows
    otm_mask = (
        ((merged['Call/Put'] == 'C') & (merged['Strike Price'] > merged['Close'])) |  # OTM Calls
        ((merged['Call/Put'] == 'P') & (merged['Strike Price'] < merged['Close']))    # OTM Puts
    )
    
    return merged[otm_mask].drop('Close', axis=1)

def classify_money_type(df: pd.DataFrame) -> pd.DataFrame:
    """Classify flows as Smart Money, Retail, or Institutional."""
    
    def get_money_type(row):
        premium = row['Total Premium ($)']
        volume = row['Total Volume']
        dte_str = str(row['Days to Expiry'])
        
        # Extract first number from days to expiry string
        try:
            dte = int(dte_str.split(',')[0]) if ',' in dte_str else int(dte_str)
        except:
            dte = 0
        
        # Smart Money patterns
        if premium >= 1000000 and dte >= 30:
            return "🧠 Smart Money"
        elif volume >= 3000 and premium >= 500000:
            return "🏦 Institutional"
        elif premium >= 2000000:
            return "🐋 Whale"
        elif premium < 100000 and dte <= 7:
            return "🎰 Retail Gamble"
        elif premium < 500000:
            return "📱 Retail"
        else:
            return "⚖️ Mixed"
    
    df['Money_Type'] = df.apply(get_money_type, axis=1)
    return df

def display_smart_money_flows(tech_flow):
    """Display Smart Money focused view with improved colors."""
    st.markdown("### 🧠 Smart Money Tracker")
    
    # Apply money type classification
    classified_flow = classify_money_type(tech_flow)
    
    smart_flows = classified_flow[
        classified_flow['Money_Type'].isin(['🧠 Smart Money', '🐋 Whale', '🏦 Institutional'])
    ]
    
    if smart_flows.empty:
        st.info("No smart money flows found with current filters.")
        return
    
    # Group by money type with better formatting
    money_summary = smart_flows.groupby('Money_Type').agg({
        'Total Premium ($)': 'sum',
        'Symbol': 'count'
    }).rename(columns={'Symbol': 'Flow Count'})
    
    # Display summary with better styling
    st.markdown("#### 📊 Smart Money Summary")
    col1, col2, col3 = st.columns(3)
    
    total_smart_premium = money_summary['Total Premium ($)'].sum()
    total_smart_flows = money_summary['Flow Count'].sum()
    
    with col1:
        if '🧠 Smart Money' in money_summary.index:
            smart_premium = money_summary.loc['🧠 Smart Money', 'Total Premium ($)']
            st.metric("🧠 Smart Money", f"${smart_premium/1000000:.1f}M")
        else:
            st.metric("🧠 Smart Money", "$0M")
    
    with col2:
        if '🐋 Whale' in money_summary.index:
            whale_premium = money_summary.loc['🐋 Whale', 'Total Premium ($)']
            st.metric("🐋 Whale Trades", f"${whale_premium/1000000:.1f}M")
        else:
            st.metric("🐋 Whale Trades", "$0M")
    
    with col3:
        if '🏦 Institutional' in money_summary.index:
            inst_premium = money_summary.loc['🏦 Institutional', 'Total Premium ($)']
            st.metric("🏦 Institutional", f"${inst_premium/1000000:.1f}M")
        else:
            st.metric("🏦 Institutional", "$0M")
    
    st.markdown("---")
    
    # Display top smart money flows with improved colors
    st.markdown("#### 🔍 Top Smart Money Flows")
    
    # Define better color schemes for each money type
    color_schemes = {
        '🧠 Smart Money': {
            'bg_color': 'rgba(138, 43, 226, 0.1)',  # Purple
            'border_color': '#8A2BE2',
            'text_color': '#4B0082'
        },
        '🐋 Whale': {
            'bg_color': 'rgba(0, 100, 148, 0.1)',   # Deep Blue
            'border_color': '#006494',
            'text_color': '#003D5C'
        },
        '🏦 Institutional': {
            'bg_color': 'rgba(0, 123, 85, 0.1)',    # Teal
            'border_color': '#007B55',
            'text_color': '#004D33'
        }
    }
    
    for _, row in smart_flows.head(15).iterrows():
        money_type = row['Money_Type']
        colors = color_schemes.get(money_type, {
            'bg_color': 'rgba(128, 128, 128, 0.1)',
            'border_color': '#808080',
            'text_color': '#404040'
        })
        
        # Get call/put indicator
        direction = "📈" if row['Call/Put'] == 'C' else "📉"
        
        st.markdown(f"""
        <div style="background: {colors['bg_color']}; padding: 12px; margin: 4px; 
                    border-radius: 8px; border-left: 4px solid {colors['border_color']}; 
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <div style="display: flex; align-items: center; margin-bottom: 6px;">
                <span style="background: {colors['border_color']}; color: white; padding: 2px 8px; 
                            border-radius: 12px; font-size: 0.8em; margin-right: 8px;">
                    {money_type.split()[0]}
                </span>
                <strong style="font-size: 1.1em; color: {colors['text_color']};">{row['Symbol']}</strong>
                <span style="margin-left: 8px;">{direction}</span>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="color: {colors['text_color']}; font-weight: bold;">
                    ${row['Strike Price']:.0f} {row['Call/Put']}
                </span>
                <span style="color: {colors['border_color']}; font-weight: bold; font-size: 1.1em;">
                    ${row['Total Premium ($)']/1000000:.1f}M
                </span>
            </div>
            <div style="font-size: 0.9em; color: #666; margin-top: 4px;">
                📊 Volume: {row['Total Volume']:,} • 📅 Days: {row['Days to Expiry']}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Add a breakdown chart if there are multiple types
    if len(money_summary) > 1:
        st.markdown("#### 📈 Smart Money Breakdown")
        
        # Create a simple bar chart
        chart_data = money_summary.reset_index()
        chart_data['Premium (M)'] = chart_data['Total Premium ($)'] / 1000000
        
        fig = px.bar(
            chart_data, 
            x='Money_Type', 
            y='Premium (M)',
            title="Smart Money Distribution",
            color='Money_Type',
            color_discrete_map={
                '🧠 Smart Money': '#8A2BE2',
                '🐋 Whale': '#006494', 
                '🏦 Institutional': '#007B55'
            }
        )
        fig.update_layout(
            showlegend=False,
            height=400,
            xaxis_title="Money Type",
            yaxis_title="Premium ($M)"
        )
        st.plotly_chart(fig, use_container_width=True)

def validate_symbols(symbols: List[str]) -> List[str]:
    """Validate stock symbols using yfinance."""
    valid_symbols = []
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol.upper())
            hist = ticker.history(period="1d")
            if not hist.empty:
                valid_symbols.append(symbol.upper())
        except:
            continue
    return valid_symbols

def display_custom_flows(custom_symbols: List[str], min_premium: int, days_limit: int, show_otm_only: bool):
    """Display flows for custom symbols."""
    if not custom_symbols:
        st.info("Please enter stock symbols to search for flows.")
        return
    
    st.markdown(f"### 🔍 Custom Search Results for: {', '.join(custom_symbols)}")
    
    # Validate symbols
    with st.spinner("Validating symbols..."):
        valid_symbols = validate_symbols(custom_symbols)
        
        if not valid_symbols:
            st.error("No valid symbols found. Please check your input.")
            return
        
        if len(valid_symbols) < len(custom_symbols):
            invalid_symbols = [s for s in custom_symbols if s.upper() not in valid_symbols]
            st.warning(f"Invalid symbols skipped: {', '.join(invalid_symbols)}")
    
    # Fetch stock prices for valid symbols
    with st.spinner("Fetching stock prices..."):
        price_data = []
        for symbol in valid_symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="1d")
                if not hist.empty:
                    open_p = hist['Open'].iloc[0]
                    high = hist['High'].iloc[0]
                    close = hist['Close'].iloc[0]
                    change_pct = (close - open_p) / open_p * 100 if open_p != 0 else 0
                    price_data.append({
                        'Symbol': symbol,
                        'Open': round(open_p, 2),
                        'High': round(high, 2),
                        'Close': round(close, 2),
                        'Change %': round(change_pct, 2)
                    })
            except Exception:
                pass
        
        prices_df = pd.DataFrame(price_data)
    
    # Fetch options flow data
    with st.spinner("Fetching options flow data..."):
        urls = [
            "https://www.cboe.com/us/options/market_statistics/symbol_data/csv/?mkt=cone",
            "https://www.cboe.com/us/options/market_statistics/symbol_data/csv/?mkt=opt",
            "https://www.cboe.com/us/options/market_statistics/symbol_data/csv/?mkt=ctwo",
            "https://www.cboe.com/us/options/market_statistics/symbol_data/csv/?mkt=exo"
        ]
        
        data = fetch_data_from_urls(urls)
        if data.empty:
            st.warning("No options flow data fetched from CBOE.")
            return
        
        # Filter for custom symbols
        flow_data = filter_options_flow(data, [], days_limit=days_limit, min_premium=min_premium)
        
        # Filter for OTM if requested
        if show_otm_only:
            flow_data = filter_otm_flows(flow_data, prices_df)
        
        # Filter to valid symbols
        custom_flow = flow_data[flow_data['Symbol'].isin(valid_symbols)]
    
    if custom_flow.empty:
        st.warning(f"No options flow data found for {', '.join(valid_symbols)} with current filters.")
        return
    
    # Display results
    st.success(f"Found {len(custom_flow)} flows for {len(valid_symbols)} symbols")
    
    # Compute sentiment analysis
    symbol_group = custom_flow.groupby(['Symbol', 'Call/Put'])['Total Premium ($)'].sum().unstack(fill_value=0)
    if 'C' in symbol_group.columns and 'P' in symbol_group.columns:
        symbol_group.columns = ['Call Premium', 'Put Premium']
    elif 'C' in symbol_group.columns:
        symbol_group.columns = ['Call Premium']
        symbol_group['Put Premium'] = 0
    elif 'P' in symbol_group.columns:
        symbol_group.columns = ['Put Premium']
        symbol_group['Call Premium'] = 0
    else:
        st.warning("No call or put data found.")
        return
    
    symbol_group['Total Premium'] = symbol_group['Call Premium'] + symbol_group['Put Premium']
    symbol_group['Net Sentiment'] = (symbol_group['Call Premium'] - symbol_group['Put Premium']) / symbol_group['Total Premium'].replace(0, 1)
    
    # Calculate flow counts
    flow_counts = custom_flow.groupby('Symbol').size().rename('Flow Count')
    
    sentiment_df = symbol_group.reset_index()
    sentiment_df = sentiment_df.set_index('Symbol').join(flow_counts).reset_index().fillna(0)
    
    # Merge with prices
    merged_df = prices_df.merge(sentiment_df, on='Symbol', how='left').fillna(0)
    merged_df = merged_df[merged_df['Total Premium'] > 0]
    
    # Separate bullish and bearish flows
    bullish_df = merged_df[merged_df['Net Sentiment'] > 0].sort_values('Call Premium', ascending=False)
    bearish_df = merged_df[merged_df['Net Sentiment'] < 0].sort_values('Put Premium', ascending=False)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_bullish_premium = bullish_df['Call Premium'].sum()
        st.metric("💚 Total Bullish Premium", f"${total_bullish_premium/1000000:.1f}M")
    
    with col2:
        total_bearish_premium = bearish_df['Put Premium'].sum()
        st.metric("❤️ Total Bearish Premium", f"${total_bearish_premium/1000000:.1f}M")
    
    with col3:
        total_flows = len(bullish_df) + len(bearish_df)
        st.metric("📊 Total Active Symbols", total_flows)
    
    with col4:
        if (total_bullish_premium + total_bearish_premium) > 0:
            net_sentiment = (total_bullish_premium - total_bearish_premium) / (total_bullish_premium + total_bearish_premium) * 100
        else:
            net_sentiment = 0
        st.metric("🎯 Net Sentiment", f"{net_sentiment:.1f}%")
    
    st.markdown("---")
    
    # Stock price table
    if not prices_df.empty:
        st.markdown("#### 📈 Stock Prices")
        st.dataframe(prices_df.style.format({
            'Open': '${:.2f}',
            'High': '${:.2f}',
            'Close': '${:.2f}',
            'Change %': '{:.2f}%'
        }), use_container_width=True)
    
    # Bulls & Bears display
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🟢 Bullish Flows")
        
        if not bullish_df.empty:
            for _, row in bullish_df.iterrows():
                # Get detailed flow for this symbol
                symbol_flows = custom_flow[
                    (custom_flow['Symbol'] == row['Symbol']) & 
                    (custom_flow['Call/Put'] == 'C')
                ].sort_values('Total Premium ($)', ascending=False)
                
                if not symbol_flows.empty:
                    top_flow = symbol_flows.iloc[0]
                    strike = top_flow['Strike Price']
                    expiry = top_flow['Expiry Dates']
                    current_price = row['Close']
                    otm_distance = ((strike - current_price) / current_price * 100) if current_price > 0 else 0
                    
                    st.markdown(f"""
                    <div style="background: rgba(76, 175, 80, 0.1); padding: 8px; margin: 2px; border-radius: 5px; border-left: 4px solid #4CAF50;">
                        <strong>{row['Symbol']}</strong>    
                        <span style="color: #666;">Flow#: {int(row['Flow Count'])}</span>   
                        <span style="color: #4CAF50; font-weight: bold;">${row['Call Premium']/1000000:.1f}M</span><br>
                        <span style="color: #888; font-size: 0.9em;">${strike:.0f} CALL • {expiry}</span><br>
                        <span style="color: #999; font-size: 0.8em;">Stock: ${current_price:.2f} | OTM: +{otm_distance:.1f}%</span>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No bullish flows found")
    
    with col2:
        st.markdown("### 🔴 Bearish Flows")
        
        if not bearish_df.empty:
            for _, row in bearish_df.iterrows():
                # Get detailed flow for this symbol
                symbol_flows = custom_flow[
                    (custom_flow['Symbol'] == row['Symbol']) & 
                    (custom_flow['Call/Put'] == 'P')
                ].sort_values('Total Premium ($)', ascending=False)
                
                if not symbol_flows.empty:
                    top_flow = symbol_flows.iloc[0]
                    strike = top_flow['Strike Price']
                    expiry = top_flow['Expiry Dates']
                    current_price = row['Close']
                    otm_distance = ((current_price - strike) / current_price * 100) if current_price > 0 else 0
                    
                    st.markdown(f"""
                    <div style="background: rgba(244, 67, 54, 0.1); padding: 8px; margin: 2px; border-radius: 5px; border-left: 4px solid #f44336;">
                        <strong>{row['Symbol']}</strong>    
                        <span style="color: #666;">Flow#: {int(row['Flow Count'])}</span>   
                        <span style="color: #f44336; font-weight: bold;">-${row['Put Premium']/1000000:.1f}M</span><br>
                        <span style="color: #888; font-size: 0.9em;">${strike:.0f} PUT • {expiry}</span><br>
                        <span style="color: #999; font-size: 0.8em;">Stock: ${current_price:.2f} | OTM: +{otm_distance:.1f}%</span>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No bearish flows found")
    
    # Detailed flow data
    with st.expander("📊 Detailed Flow Data"):
        display_df = custom_flow[['Symbol', 'Strike Price', 'Call/Put', 'Total Premium ($)', 
                                  'Total Volume', 'Expiry Dates', 'Days to Expiry']].copy()
        display_df['Total Premium ($)'] = display_df['Total Premium ($)'].apply(lambda x: f"${x:,.0f}")
        display_df['Total Volume'] = display_df['Total Volume'].apply(lambda x: f"{x:,}")
        st.dataframe(display_df, use_container_width=True)

def filter_high_volume_flows(df: pd.DataFrame) -> pd.DataFrame:
    """Filter for high volume options and group by symbol with summary."""
    if df.empty:
        return pd.DataFrame()

    required_cols = ['Symbol', 'Call/Put', 'Volume', 'Last Price', 'Expiration']
    if not all(col in df.columns for col in required_cols):
        return pd.DataFrame()

    df = df.copy()
    df['Expiration'] = pd.to_datetime(df['Expiration'], errors='coerce')
    df = df.dropna(subset=['Expiration'])
    df = df[df['Expiration'] >= datetime.now()]

    df = df[df['Volume'] > 1000]
    df = df[df['Last Price'] > 1]

    df['Premium'] = df['Volume'] * df['Last Price'] * 100

    grouped = df.groupby('Symbol').agg(
        total_premium=('Premium', 'sum'),
        total_volume=('Volume', 'sum'),
        option_count=('Premium', 'count')
    )

    call_prem = df[df['Call/Put'] == 'C'].groupby('Symbol')['Premium'].sum().rename('call_premium')
    put_prem = df[df['Call/Put'] == 'P'].groupby('Symbol')['Premium'].sum().rename('put_premium')

    grouped = grouped.join(call_prem, how='left').join(put_prem, how='left').fillna(0)

    grouped['net_sentiment'] = (grouped['call_premium'] - grouped['put_premium']) / grouped['total_premium'].where(grouped['total_premium'] > 0, 0)

    grouped = grouped.sort_values('total_premium', ascending=False)

    return grouped.reset_index()

# List of tech stocks (same as before)
tech_stocks = [
    "MSFT", "AAPL", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "TSM", "AVGO", "ASML", "SNOW", "MSTR", "COIN",
    "ORCL", "ADBE", "CRM", "INTC", "CSCO", "AMD", "QCOM", "TXN", "AMAT", "LRCX", "HOOD", "PLTR",
    "JNJ", "PFE", "MRK", "ABBV", "UNH",
    "NFLX", "DIS", "HD", "LOW", "NKE",
    "JPM", "GS", "MS", "BAC", "WFC",
    "CAT", "DE", "HON", "GE", "MMM",
    "PEP", "KO", "COST", "WMT", "TGT",
    "QQQ", "SPY", "TQQQ", "SQQQ", "SMH", "UVXY", "IBIT"
]

def main():
    st.set_page_config(
        page_title="Advanced Options Flow", 
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Initialize session state
    if 'auto_refresh' not in st.session_state:
        st.session_state.auto_refresh = True

    # Sidebar for filters
    with st.sidebar:
        st.markdown("### ⚙️ Filters")
        st.session_state.auto_refresh = st.checkbox("Enable Auto-Refresh (every 10 min)", value=st.session_state.auto_refresh)
        min_premium = st.slider("Min Premium ($)", 100000, 10000000, 300000, step=100000)
        days_limit = st.slider("Max Days to Expiry", 7, 90, 45)
        selected_symbols = st.multiselect("Select Symbols", options=tech_stocks, default=tech_stocks)
        show_otm_only = st.checkbox("Show OTM Flows Only", value=True)

    st.title("🚀 Options Flow: 15mins delayed")

    # Create tabs - Added High Volume Summary tab
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🧠 Smart Money", "🔍 Custom Search", "📈 High Volume Summary"])
    
    with tab1:
        # Existing overview tab logic
        # Fetch stock prices
        with st.spinner("Fetching stock prices..."):
            price_data = []
            for symbol in selected_symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="1d")
                    if not hist.empty:
                        open_p = hist['Open'].iloc[0]
                        high = hist['High'].iloc[0]
                        close = hist['Close'].iloc[0]
                        change_pct = (close - open_p) / open_p * 100 if open_p != 0 else 0
                        price_data.append({
                            'Symbol': symbol,
                            'Open': round(open_p, 2),
                            'High': round(high, 2),
                            'Close': round(close, 2),
                            'Change %': round(change_pct, 2)
                        })
                except Exception:
                    pass

            prices_df = pd.DataFrame(price_data)

        # Fetch options flow data
        with st.spinner("Fetching options flow data..."):
            urls = [
                "https://www.cboe.com/us/options/market_statistics/symbol_data/csv/?mkt=cone",
                "https://www.cboe.com/us/options/market_statistics/symbol_data/csv/?mkt=opt",
                "https://www.cboe.com/us/options/market_statistics/symbol_data/csv/?mkt=ctwo",
                "https://www.cboe.com/us/options/market_statistics/symbol_data/csv/?mkt=exo"
            ]

            data = fetch_data_from_urls(urls)
            if data.empty:
                st.warning("No options flow data fetched.")
                return

            exclude_symbols = ["TLRY"]
            flow_data = filter_options_flow(data, exclude_symbols, days_limit=days_limit, min_premium=min_premium)
            
            # Optionally filter for OTM
            if show_otm_only:
                flow_data = filter_otm_flows(flow_data, prices_df)
            
            # Filter to selected symbols
            tech_flow = flow_data[flow_data['Symbol'].isin(selected_symbols)]

        if tech_flow.empty:
            st.warning("No options flow data found for selected symbols.")
            return

        # Display last updated in sidebar
        with st.sidebar:
            st.markdown(f"**Last updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Compute sentiment analysis
        symbol_group = tech_flow.groupby(['Symbol', 'Call/Put'])['Total Premium ($)'].sum().unstack(fill_value=0)
        if 'C' in symbol_group.columns and 'P' in symbol_group.columns:
            symbol_group.columns = ['Call Premium', 'Put Premium']
        elif 'C' in symbol_group.columns:
            symbol_group.columns = ['Call Premium']
            symbol_group['Put Premium'] = 0
        elif 'P' in symbol_group.columns:
            symbol_group.columns = ['Put Premium']
            symbol_group['Call Premium'] = 0
        else:
            st.warning("No call or put data found.")
            return

        symbol_group['Total Premium'] = symbol_group['Call Premium'] + symbol_group['Put Premium']
        symbol_group['Net Sentiment'] = (symbol_group['Call Premium'] - symbol_group['Put Premium']) / symbol_group['Total Premium'].replace(0, 1)
        
        # Calculate flow counts
        flow_counts = tech_flow.groupby('Symbol').size().rename('Flow Count')
        
        sentiment_df = symbol_group.reset_index()
        sentiment_df = sentiment_df.set_index('Symbol').join(flow_counts).reset_index().fillna(0)
        
        # Merge with prices
        merged_df = prices_df.merge(sentiment_df, on='Symbol', how='left').fillna(0)
        merged_df = merged_df[merged_df['Total Premium'] > 0]
        
        # Separate bullish and bearish flows
        bullish_df = merged_df[merged_df['Net Sentiment'] > 0].sort_values('Call Premium', ascending=False)
        bearish_df = merged_df[merged_df['Net Sentiment'] < 0].sort_values('Put Premium', ascending=False)
        
        # Summary metrics
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_bullish_premium = bullish_df['Call Premium'].sum()
            st.metric("💚 Total Bullish Premium", f"${total_bullish_premium/1000000:.1f}M")
        
        with col2:
            total_bearish_premium = bearish_df['Put Premium'].sum()
            st.metric("❤️ Total Bearish Premium", f"${total_bearish_premium/1000000:.1f}M")
        
        with col3:
            total_flows = len(bullish_df) + len(bearish_df)
            st.metric("📊 Total Active Symbols", total_flows)
        
        with col4:
            if (total_bullish_premium + total_bearish_premium) > 0:
                net_sentiment = (total_bullish_premium - total_bearish_premium) / (total_bullish_premium + total_bearish_premium) * 100
            else:
                net_sentiment = 0
            st.metric("🎯 Net Sentiment", f"{net_sentiment:.1f}%")
        
        st.markdown("---")
        st.info(f"📊 Showing {len(tech_flow)} flows after filtering")

        # Bulls & Bears display
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🟢 Bullish: " + str(len(bullish_df)))
            
            if not bullish_df.empty:
                for _, row in bullish_df.head(15).iterrows():
                    # Get detailed flow for this symbol
                    symbol_flows = tech_flow[
                        (tech_flow['Symbol'] == row['Symbol']) & 
                        (tech_flow['Call/Put'] == 'C')
                    ].sort_values('Total Premium ($)', ascending=False)
                    
                    if not symbol_flows.empty:
                        top_flow = symbol_flows.iloc[0]
                        strike = top_flow['Strike Price']
                        expiry = top_flow['Expiry Dates']
                        current_price = row['Close']
                        otm_distance = ((strike - current_price) / current_price * 100) if current_price > 0 else 0
                        
                        st.markdown(f"""
                        <div style="background: rgba(76, 175, 80, 0.1); padding: 8px; margin: 2px; border-radius: 5px; border-left: 4px solid #4CAF50;">
                            <strong>{row['Symbol']}</strong>    
                            <span style="color: #666;">Flow#: {int(row['Flow Count'])}</span>   
                            <span style="color: #4CAF50; font-weight: bold;">${row['Call Premium']/1000000:.1f}M</span><br>
                            <span style="color: #888; font-size: 0.9em;">${strike:.0f} CALL • {expiry}</span><br>
                            <span style="color: #999; font-size: 0.8em;">Stock: ${current_price:.2f} | OTM: +{otm_distance:.1f}%</span>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("No bullish flows found")
        
        with col2:
            st.markdown("### 🔴 Bearish: " + str(len(bearish_df)))
            
            if not bearish_df.empty:
                for _, row in bearish_df.head(15).iterrows():
                    # Get detailed flow for this symbol
                    symbol_flows = tech_flow[
                        (tech_flow['Symbol'] == row['Symbol']) & 
                        (tech_flow['Call/Put'] == 'P')
                    ].sort_values('Total Premium ($)', ascending=False)
                    
                    if not symbol_flows.empty:
                        top_flow = symbol_flows.iloc[0]
                        strike = top_flow['Strike Price']
                        expiry = top_flow['Expiry Dates']
                        current_price = row['Close']
                        otm_distance = ((current_price - strike) / current_price * 100) if current_price > 0 else 0
                        
                        st.markdown(f"""
                        <div style="background: rgba(244, 67, 54, 0.1); padding: 8px; margin: 2px; border-radius: 5px; border-left: 4px solid #f44336;">
                            <strong>{row['Symbol']}</strong>    
                            <span style="color: #666;">Flow#: {int(row['Flow Count'])}</span>   
                            <span style="color: #f44336; font-weight: bold;">-${row['Put Premium']/1000000:.1f}M</span><br>
                            <span style="color: #888; font-size: 0.9em;">${strike:.0f} PUT • {expiry}</span><br>
                            <span style="color: #999; font-size: 0.8em;">Stock: ${current_price:.2f} | OTM: +{otm_distance:.1f}%</span>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("No bearish flows found")

    with tab2:
        # Smart Money tab
        if 'tech_flow' in locals():
            display_smart_money_flows(tech_flow)
        else:
            st.info("Please run the Overview tab first to load data.")

    with tab3:
        # Custom Search tab
        st.markdown("### 🔍 Custom Stock Search")
        st.markdown("Enter up to 5 stock symbols separated by commas (e.g., AAPL, MSFT, GOOGL)")
        
        # Input for custom symbols
        custom_input = st.text_input(
            "Stock Symbols",
            placeholder="Enter symbols like: AAPL, MSFT, GOOGL, AMZN, TSLA",
            help="Enter up to 5 stock symbols separated by commas"
        )
        
        # Custom filters
        col1, col2, col3 = st.columns(3)
        with col1:
            custom_min_premium = st.selectbox(
                "Min Premium", 
                [100000, 300000, 500000, 1000000, 2000000], 
                index=1,
                format_func=lambda x: f"${x/1000000:.1f}M"
            )
        with col2:
            custom_days_limit = st.selectbox("Max Days to Expiry", [7, 14, 30, 45, 90], index=3)
        with col3:
            custom_otm_only = st.checkbox("OTM Only", value=True)
        
        if st.button("🔍 Search Flows", type="primary"):
            if custom_input:
                # Parse input
                custom_symbols = [s.strip().upper() for s in custom_input.split(',') if s.strip()]
                
                # Limit to 5 symbols
                if len(custom_symbols) > 5:
                    st.warning("Only the first 5 symbols will be processed.")
                    custom_symbols = custom_symbols[:5]
                
                # Display custom flows
                display_custom_flows(custom_symbols, custom_min_premium, custom_days_limit, custom_otm_only)
            else:
                st.warning("Please enter at least one stock symbol.")

        # Enhanced High Volume Summary Tab with Advanced Insights
    
    with tab4:
        st.markdown("### 📊 Advanced High Volume Flow Analytics")
        st.markdown("*Deep market intelligence from institutional option activity*")
    
        with st.spinner("Fetching and analyzing options flow data..."):
            urls = [
                "https://www.cboe.com/us/options/market_statistics/symbol_data/csv/?mkt=cone",
                "https://www.cboe.com/us/options/market_statistics/symbol_data/csv/?mkt=opt",
                "https://www.cboe.com/us/options/market_statistics/symbol_data/csv/?mkt=ctwo",
                "https://www.cboe.com/us/options/market_statistics/symbol_data/csv/?mkt=exo"
            ]
    
            data = fetch_data_from_urls(urls)
            if data.empty:
                st.warning("No options flow data fetched.")
                return
    
            high_vol_df = filter_high_volume_flows(data)
    
        if high_vol_df.empty:
            st.info("No options meet the high volume criteria.")
        else:
            # 🔥 NEW: Enhanced Insights Section
            st.markdown("---")
            st.markdown("### 🧠 Market Intelligence Dashboard")
            
            # Calculate advanced metrics
            total_premium = high_vol_df['total_premium'].sum()
            total_options = high_vol_df['option_count'].sum()
            
            # NEW: Sentiment Analysis
            bullish_stocks = high_vol_df[high_vol_df['net_sentiment'] > 0.2]  # Strong bullish
            bearish_stocks = high_vol_df[high_vol_df['net_sentiment'] < -0.2]  # Strong bearish
            neutral_stocks = high_vol_df[abs(high_vol_df['net_sentiment']) <= 0.2]
            
            # NEW: Flow Concentration Analysis
            top_10_premium = high_vol_df.head(10)['total_premium'].sum()
            concentration_ratio = (top_10_premium / total_premium) * 100
            
            # NEW: Unusual Activity Detection
            avg_premium = high_vol_df['total_premium'].mean()
            unusual_activity = high_vol_df[high_vol_df['total_premium'] > avg_premium * 3]
            
            # 📊 Key Insights Grid
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("💰 Total Premium", f"${total_premium / 1000000:.1f}M")
                st.metric("🎯 Market Sentiment", 
                         f"{len(bullish_stocks)}-{len(neutral_stocks)}-{len(bearish_stocks)}", 
                         help="Bull-Neutral-Bear count")
            
            with col2:
                st.metric("📊 Active Symbols", len(high_vol_df))
                flow_intensity = "🔥 High" if concentration_ratio > 60 else "🟡 Medium" if concentration_ratio > 40 else "🟢 Distributed"
                st.metric("⚡ Flow Concentration", f"{concentration_ratio:.1f}%", 
                         delta=flow_intensity, delta_color="inverse")
            
            with col3:
                st.metric("🚨 Unusual Activity", len(unusual_activity))
                if len(unusual_activity) > 0:
                    biggest_flow = unusual_activity.iloc[0]
                    st.metric("🐋 Biggest Whale", f"{biggest_flow['Symbol']}", 
                             delta=f"${biggest_flow['total_premium']/1000000:.1f}M")
                else:
                    st.metric("🐋 Biggest Flow", "None")
            
            with col4:
                # NEW: Risk-On vs Risk-Off indicator
                risk_on_premium = bullish_stocks['total_premium'].sum()
                risk_off_premium = bearish_stocks['total_premium'].sum()
                
                if risk_on_premium > risk_off_premium * 1.5:
                    market_regime = "🚀 Risk On"
                    regime_color = "normal"
                elif risk_off_premium > risk_on_premium * 1.5:
                    market_regime = "🛡️ Risk Off"
                    regime_color = "inverse"
                else:
                    market_regime = "⚖️ Mixed"
                    regime_color = "off"
                
                st.metric("🌍 Market Regime", market_regime, delta_color=regime_color)
                net_flow = (risk_on_premium - risk_off_premium) / 1000000
                st.metric("📈 Net Bull Flow", f"${net_flow:.1f}M")
    
            # 🎯 NEW: Actionable Alerts Section
            st.markdown("---")
            st.markdown("### 🚨 Trading Alerts & Opportunities")
            
            alert_col1, alert_col2 = st.columns(2)
            
            with alert_col1:
                st.markdown("#### 🔥 High Conviction Plays")
                
                # Strong bullish with high volume
                strong_bulls = bullish_stocks[bullish_stocks['net_sentiment'] > 0.5].head(5)
                
                for _, stock in strong_bulls.iterrows():
                    call_dominance = (stock['call_premium'] / stock['total_premium']) * 100
                    st.markdown(f"""
                    <div style="background: linear-gradient(90deg, rgba(76,175,80,0.1), rgba(129,199,132,0.05)); 
                               padding: 10px; margin: 5px 0; border-radius: 8px; 
                               border-left: 4px solid #4CAF50;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <strong style="font-size: 1.1em;">{stock['Symbol']}</strong>
                                <span style="color: #4CAF50; margin-left: 10px;">📈 {call_dominance:.0f}% Calls</span>
                            </div>
                            <div style="text-align: right;">
                                <strong style="color: #2E7D32;">${stock['total_premium']/1000000:.1f}M</strong><br>
                                <small style="color: #666;">{stock['option_count']} flows</small>
                            </div>
                        </div>
                        <div style="margin-top: 5px; font-size: 0.9em; color: #555;">
                            💡 <strong>Signal:</strong> Strong institutional buying detected
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            with alert_col2:
                st.markdown("#### ⚠️ Risk-Off Signals")
                
                # Strong bearish with high volume
                strong_bears = bearish_stocks[bearish_stocks['net_sentiment'] < -0.5].head(5)
                
                for _, stock in strong_bears.iterrows():
                    put_dominance = (stock['put_premium'] / stock['total_premium']) * 100
                    st.markdown(f"""
                    <div style="background: linear-gradient(90deg, rgba(244,67,54,0.1), rgba(229,115,115,0.05)); 
                               padding: 10px; margin: 5px 0; border-radius: 8px; 
                               border-left: 4px solid #f44336;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <strong style="font-size: 1.1em;">{stock['Symbol']}</strong>
                                <span style="color: #f44336; margin-left: 10px;">📉 {put_dominance:.0f}% Puts</span>
                            </div>
                            <div style="text-align: right;">
                                <strong style="color: #C62828;">${stock['total_premium']/1000000:.1f}M</strong><br>
                                <small style="color: #666;">{stock['option_count']} flows</small>
                            </div>
                        </div>
                        <div style="margin-top: 5px; font-size: 0.9em; color: #555;">
                            💡 <strong>Signal:</strong> Hedging or bearish positioning
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
    
            # 📈 NEW: Advanced Visualizations
            st.markdown("---")
            st.markdown("### 📊 Advanced Analytics")
            
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                # Sentiment Distribution
                sentiment_bins = pd.cut(high_vol_df['net_sentiment'], 
                                      bins=[-1, -0.5, -0.2, 0.2, 0.5, 1], 
                                      labels=['Strong Bear', 'Weak Bear', 'Neutral', 'Weak Bull', 'Strong Bull'])
                sentiment_counts = sentiment_bins.value_counts()
                
                fig_sentiment = px.pie(
                    values=sentiment_counts.values,
                    names=sentiment_counts.index,
                    title="Market Sentiment Distribution",
                    color_discrete_map={
                        'Strong Bear': '#B71C1C',
                        'Weak Bear': '#F44336', 
                        'Neutral': '#9E9E9E',
                        'Weak Bull': '#8BC34A',
                        'Strong Bull': '#2E7D32'
                    }
                )
                fig_sentiment.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_sentiment, use_container_width=True)
            
            with viz_col2:
                # Volume vs Premium Efficiency
                high_vol_df['premium_per_option'] = high_vol_df['total_premium'] / high_vol_df['option_count']
                
                fig_efficiency = px.scatter(
                    high_vol_df.head(20),
                    x='total_volume',
                    y='premium_per_option',
                    size='total_premium',
                    color='net_sentiment',
                    hover_name='Symbol',
                    title="Premium Efficiency Analysis",
                    labels={
                        'total_volume': 'Total Volume',
                        'premium_per_option': 'Premium per Contract ($)',
                        'net_sentiment': 'Sentiment'
                    },
                    color_continuous_scale='RdYlGn',
                    color_continuous_midpoint=0
                )
                st.plotly_chart(fig_efficiency, use_container_width=True)
    
            # 🔍 NEW: Sector Analysis
            st.markdown("---")
            st.markdown("### 🏭 Sector Flow Analysis")
            
            # Map symbols to sectors (you can expand this mapping)
            sector_mapping = {
                # Tech
                'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology', 'META': 'Technology',
                'NVDA': 'Technology', 'TSLA': 'Technology', 'AMZN': 'Technology', 'NFLX': 'Technology',
                
                # Finance
                'JPM': 'Financial', 'BAC': 'Financial', 'GS': 'Financial', 'MS': 'Financial',
                
                # Healthcare
                'JNJ': 'Healthcare', 'PFE': 'Healthcare', 'MRK': 'Healthcare', 'UNH': 'Healthcare',
                
                # Energy
                'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy',
                
                # ETFs
                'SPY': 'Market ETF', 'QQQ': 'Tech ETF', 'TQQQ': 'Leveraged ETF'
            }
            
            high_vol_df['Sector'] = high_vol_df['Symbol'].map(sector_mapping).fillna('Other')
            
            sector_analysis = high_vol_df.groupby('Sector').agg({
                'total_premium': 'sum',
                'call_premium': 'sum', 
                'put_premium': 'sum',
                'Symbol': 'count'
            }).rename(columns={'Symbol': 'Stock Count'})
            
            sector_analysis['Net Sentiment'] = (
                (sector_analysis['call_premium'] - sector_analysis['put_premium']) / 
                sector_analysis['total_premium']
            ).round(3)
            
            sector_analysis = sector_analysis.sort_values('total_premium', ascending=False)
            
            # Sector performance table
            st.markdown("#### 🏆 Sector Rankings by Flow")
            
            sector_display = sector_analysis.copy()
            sector_display['total_premium'] = sector_display['total_premium'].apply(lambda x: f"${x/1000000:.1f}M")
            sector_display['call_premium'] = sector_display['call_premium'].apply(lambda x: f"${x/1000000:.1f}M")
            sector_display['put_premium'] = sector_display['put_premium'].apply(lambda x: f"${x/1000000:.1f}M")
            
            st.dataframe(sector_display, use_container_width=True)
    
            # 📋 NEW: Export Enhanced Data
            st.markdown("---")
            st.markdown("### 📥 Export & Alerts")
            
            export_col1, export_col2, export_col3 = st.columns(3)
            
            with export_col1:
                # Enhanced CSV export
                if st.button("📊 Export Full Analysis", type="primary"):
                    enhanced_df = high_vol_df.copy()
                    enhanced_df['Market_Regime'] = market_regime
                    enhanced_df['Alert_Level'] = enhanced_df.apply(lambda row: 
                        'HIGH' if row['total_premium'] > avg_premium * 3 else
                        'MEDIUM' if abs(row['net_sentiment']) > 0.3 else 'LOW', axis=1)
                    
                    csv = enhanced_df.to_csv(index=False)
                    st.download_button(
                        label="Download Enhanced CSV",
                        data=csv,
                        file_name=f"enhanced_options_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv"
                    )
            
            with export_col2:
                # Watchlist export
                if st.button("⭐ Export Watchlist"):
                    watchlist_symbols = list(strong_bulls['Symbol']) + list(strong_bears['Symbol'])
                    watchlist_data = f"# High Priority Options Flow Watchlist - {datetime.now().strftime('%Y-%m-%d')}\n"
                    watchlist_data += f"# Generated from {len(high_vol_df)} analyzed flows\n"
                    watchlist_data += f"# Market Regime: {market_regime}\n\n"
                    watchlist_data += "# BULLISH SIGNALS:\n"
                    for symbol in strong_bulls['Symbol']:
                        watchlist_data += f"{symbol}\n"
                    watchlist_data += "\n# BEARISH SIGNALS:\n"
                    for symbol in strong_bears['Symbol']:
                        watchlist_data += f"{symbol}\n"
                    
                    st.download_button(
                        label="Download Watchlist",
                        data=watchlist_data,
                        file_name=f"options_watchlist_{datetime.now().strftime('%Y%m%d')}.txt",
                        mime="text/plain"
                    )
            
            with export_col3:
                # Alert setup
                alert_threshold = st.selectbox(
                    "Set Alert Threshold", 
                    ['$1M+', '$2M+', '$5M+', '$10M+'],
                    help="Get notified when flows exceed threshold"
                )
                
                if st.button("🔔 Setup Alert"):
                    threshold_map = {'$1M+': 1000000, '$2M+': 2000000, '$5M+': 5000000, '$10M+': 10000000}
                    threshold = threshold_map[alert_threshold]
                    big_flows = high_vol_df[high_vol_df['total_premium'] >= threshold]
                    
                    if len(big_flows) > 0:
                        st.success(f"🚨 ALERT: {len(big_flows)} flows exceed {alert_threshold}!")
                        for _, flow in big_flows.head(3).iterrows():
                            st.warning(f"🐋 {flow['Symbol']}: ${flow['total_premium']/1000000:.1f}M premium")
                    else:
                        st.info(f"No flows currently exceed {alert_threshold}")
    
            # 📊 Enhanced Data Table
            st.markdown("---")
            st.markdown("### 📋 Detailed Flow Analysis")
            
            # Add more calculated columns
            enhanced_display = high_vol_df.copy()
            enhanced_display['Flow_Efficiency'] = (enhanced_display['total_premium'] / enhanced_display['option_count']).round(0)
            enhanced_display['Sentiment_Grade'] = enhanced_display['net_sentiment'].apply(lambda x: 
                'A+' if x > 0.7 else 'A' if x > 0.4 else 'B+' if x > 0.2 else 
                'B' if x > -0.2 else 'C+' if x > -0.4 else 'C' if x > -0.7 else 'D')
            
            # Format for display
            display_cols = ['Symbol', 'total_premium', 'call_premium', 'put_premium', 'total_volume', 
                           'option_count', 'net_sentiment', 'Flow_Efficiency', 'Sentiment_Grade']
            
            format_dict = {
                'total_premium': '${:,.0f}',
                'call_premium': '${:,.0f}',
                'put_premium': '${:,.0f}',
                'total_volume': '{:,.0f}',
                'option_count': '{:,.0f}',
                'net_sentiment': '{:.3f}',
                'Flow_Efficiency': '${:,.0f}'
            }
            
            st.dataframe(
                enhanced_display[display_cols].head(30).style.format(format_dict),
                use_container_width=True,
                height=400
            )
    
    

    # Auto-refresh (only for overview tab)
    if st.session_state.auto_refresh:
        st_autorefresh(interval=REFRESH_INTERVAL * 1000, key="optionsflowrefresh")

if __name__ == "__main__":
    main()
