import pandas as pd
import requests
from io import StringIO
from datetime import datetime
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor
import streamlit as st
import yfinance as yf
import time
from streamlit_autorefresh import st_autorefresh  # New import for auto-refresh
import plotly.express as px  # For charts

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
        else:
            pass

        return df
    except requests.RequestException as e:
        pass
    except pd.errors.ParserError as e:
        pass
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
        'Expiration': grouped['Expiration'].apply(lambda x: x[0] if x else pd.NaT)  # For compatibility
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

# List of top 50 tech stocks by market cap (as of recent data)
tech_stocks = [
    # Top 20 tech stocks by market cap (no duplicates)
    "MSFT", "AAPL", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "TSM", "AVGO", "ASML",
    "ORCL", "ADBE", "CRM", "INTC", "CSCO", "AMD", "QCOM", "TXN", "AMAT", "LRCX",
    # Top 5 financials
    "JPM", "GS", "MS", "BAC", "WFC",
    # Top 5 industrials/energy
    "CAT", "DE", "HON", "GE", "MMM",
    # Top 5 consumer/defensive
    "PEP", "KO", "COST", "WMT", "TGT",
    # ETFs and leveraged products (no duplicates)
    "QQQ", "SPY", "TQQQ", "SQQQ", "SMH", "UVXY", "IBIT"
]

def main():
    st.set_page_config(page_title="Options Flow Analyzer", page_icon="📊", layout="wide")
    
    # Initialize session state if needed
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

    st.title("🚀 Options Flow: delayed data by 15mins")

    # Fetch stock prices
    with st.spinner("Fetching stock prices..."):
        price_data = []
        for symbol in tech_stocks:
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
            except Exception as e:
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
        st.warning("No tech stock options flow data found.")
        return

    # Display last updated in sidebar
    with st.sidebar:
        st.markdown(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Compute weighted average strikes and sentiment
    def weighted_avg_strike(group):
        if group['Total Premium ($)'].sum() == 0:
            return 0
        return (group['Strike Price'] * group['Total Premium ($)']).sum() / group['Total Premium ($)'].sum()

    avg_call_strike = tech_flow[tech_flow['Call/Put'] == 'C'].groupby('Symbol').apply(weighted_avg_strike).rename('Avg Call Strike')
    avg_put_strike = tech_flow[tech_flow['Call/Put'] == 'P'].groupby('Symbol').apply(weighted_avg_strike).rename('Avg Put Strike')

    # Compute sentiment
    symbol_group = tech_flow.groupby(['Symbol', 'Call/Put'])['Total Premium ($)'].sum().unstack(fill_value=0)
    symbol_group.columns = ['Call Premium', 'Put Premium']
    symbol_group['Total Premium'] = symbol_group['Call Premium'] + symbol_group['Put Premium']
    symbol_group['Net Sentiment'] = (symbol_group['Call Premium'] - symbol_group['Put Premium']) / symbol_group['Total Premium'].replace(0, 1)
    
    # Calculate flow counts
    flow_counts = tech_flow.groupby('Symbol').size().rename('Flow Count')
    
    sentiment_df = symbol_group.reset_index()
    sentiment_df = sentiment_df.set_index('Symbol').join(avg_call_strike).join(avg_put_strike).join(flow_counts).reset_index().fillna(0)
    
    # Merge with prices
    merged_df = prices_df.merge(sentiment_df, on='Symbol', how='left').fillna(0)
    
    # Filter only stocks with actual options flow
    merged_df = merged_df[merged_df['Total Premium'] > 0]
    
    # Separate bullish and bearish flows
    bullish_df = merged_df[merged_df['Net Sentiment'] > 0].sort_values('Call Premium', ascending=False)
    bearish_df = merged_df[merged_df['Net Sentiment'] < 0].sort_values('Put Premium', ascending=False)
    
    # Tabs for organization
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Overview", "🟢 Bulls", "🔴 Bears", "📊 All Flows", "📉 Charts"])

    with tab1:
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
        st.info(f"📊 Filtered to {len(tech_flow)} flows from {len(flow_data)} total flows")

    with tab2:
        st.markdown(f"### 🟢 Bullish: {len(bullish_df)}")
        
        if not bullish_df.empty:
            for i, (_, row) in enumerate(bullish_df.head(20).iterrows()):
                # Get the detailed flows for this symbol
                symbol_flows = tech_flow[
                    (tech_flow['Symbol'] == row['Symbol']) & 
                    (tech_flow['Call/Put'] == 'C')
                ].sort_values('Total Premium ($)', ascending=False)
                
                if not symbol_flows.empty:
                    top_flow = symbol_flows.iloc[0]
                    strike = top_flow['Strike Price']
                    expiry = top_flow['Expiry Dates']
                    
                    # Get current stock price for context
                    current_price = row['Close']
                    otm_distance = ((strike - current_price) / current_price * 100) if current_price > 0 else 0
                    
                    # Display the card structure always visible
                    st.markdown(f"""
                    <div style="background: rgba(76, 175, 80, 0.1); padding: 8px; margin: 2px; border-radius: 5px; border-left: 4px solid #4CAF50;">
                        <strong>{row['Symbol']}</strong>    
                        <span style="color: #666;">Flow#: {int(row['Flow Count'])}</span>   
                        <span style="color: #4CAF50; font-weight: bold;">${row['Call Premium']/1000000:.1f}M</span><br>
                        <span style="color: #888; font-size: 0.9em;">${strike:.0f} CALL • {expiry}</span><br>
                        <span style="color: #999; font-size: 0.8em;">Stock: ${current_price:.2f} | OTM: +{otm_distance:.1f}%</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Expander for detailed flows
                    with st.expander(f"Show detailed flows for {row['Symbol']}"):
                        st.dataframe(symbol_flows[['Strike Price', 'Total Premium ($)', 'Expiry Dates', 'Days to Expiry']], use_container_width=True)
        else:
            st.info("No bullish flows found")

    with tab3:
        st.markdown(f"### 🔴 Bearish: {len(bearish_df)}")
        
        if not bearish_df.empty:
            for i, (_, row) in enumerate(bearish_df.head(20).iterrows()):
                # Get the detailed flows for this symbol
                symbol_flows = tech_flow[
                    (tech_flow['Symbol'] == row['Symbol']) & 
                    (tech_flow['Call/Put'] == 'P')
                ].sort_values('Total Premium ($)', ascending=False)
                
                if not symbol_flows.empty:
                    top_flow = symbol_flows.iloc[0]
                    strike = top_flow['Strike Price']
                    expiry = top_flow['Expiry Dates']
                    
                    # Get current stock price for context
                    current_price = row['Close']
                    otm_distance = ((current_price - strike) / current_price * 100) if current_price > 0 else 0
                    
                    # Display the card structure always visible
                    st.markdown(f"""
                    <div style="background: rgba(244, 67, 54, 0.1); padding: 8px; margin: 2px; border-radius: 5px; border-left: 4px solid #f44336;">
                        <strong>{row['Symbol']}</strong>    
                        <span style="color: #666;">Flow#: {int(row['Flow Count'])}</span>   
                        <span style="color: #f44336; font-weight: bold;">-${row['Put Premium']/1000000:.1f}M</span><br>
                        <span style="color: #888; font-size: 0.9em;">${strike:.0f} PUT • {expiry}</span><br>
                        <span style="color: #999; font-size: 0.8em;">Stock: ${current_price:.2f} | OTM: +{otm_distance:.1f}%</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Expander for detailed flows
                    with st.expander(f"Show detailed flows for {row['Symbol']}"):
                        st.dataframe(symbol_flows[['Strike Price', 'Total Premium ($)', 'Expiry Dates', 'Days to Expiry']], use_container_width=True)
        else:
            st.info("No bearish flows found")

    with tab4:
        st.markdown("### All Flows")
        if not merged_df.empty:
            detailed_all = merged_df[['Symbol', 'Call Premium', 'Put Premium', 'Total Premium', 'Flow Count', 'Change %', 'Close']].copy()
            detailed_all['Call Premium'] = detailed_all['Call Premium'].apply(lambda x: f"${x:,.0f}")
            detailed_all['Put Premium'] = detailed_all['Put Premium'].apply(lambda x: f"${x:,.0f}")
            detailed_all['Total Premium'] = detailed_all['Total Premium'].apply(lambda x: f"${x:,.0f}")
            detailed_all['Change %'] = detailed_all['Change %'].apply(lambda x: f"{x:.2f}%")
            detailed_all['Stock Price'] = detailed_all['Close'].apply(lambda x: f"${x:.2f}")
            detailed_all = detailed_all.drop('Close', axis=1)
            st.dataframe(detailed_all, use_container_width=True)
        else:
            st.info("No flows found")

    with tab5:
        st.markdown("### Sentiment Charts")
        if not sentiment_df.empty:
            fig = px.bar(sentiment_df.sort_values('Total Premium', ascending=False).head(20),
                         x='Symbol', y=['Call Premium', 'Put Premium'],
                         title="Premium Breakdown by Symbol",
                         barmode='stack')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data for charts")

    # Trigger auto-refresh if enabled
    if st.session_state.auto_refresh:
        st_autorefresh(interval=REFRESH_INTERVAL * 1000, key="optionsflowrefresh")

if __name__ == "__main__":
    main()
