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
        st.markdown("###  The error is a ValueError in the Streamlit app. It's from the code I provided earlier. The traceback shows it's in the main function, line 896, which is the px.bar call in the high volume summary tab.

The code is:

fig = px.bar(

    high_vol_df.head(10),

    x='Symbol',

    y='total_premium',

    title="Top 10 Stocks by Total Premium",

    labels={'total_premium': 'Total Premium ($)'}

)

The error is in build_dataframe, process_args_into_dataframe.

It says raise ValueError(err_msg), but the err_msg isn't shown.

Probably because 'Symbol' is the index, not a column.

In filter_high_volume_flows, grouped = grouped.sort_values('total_premium', ascending=False)

high_vol_df = grouped

But grouped is a DataFrame with index as Symbol, because groupby('Symbol')

In the code:

grouped = df.groupby('Symbol').agg(...)

Yes, groupby('Symbol') makes Symbol the index.

Then in px.bar, x='Symbol', but 'Symbol' is the index, not a column.

So px.bar can't find 'Symbol' as a column.

I need to reset_index() to make Symbol a column.

Yes, that's the issue.

To fix, after sorted, high_vol_df = high_vol_df.reset_index()

Before the px.bar.

Also, in the display_df = high_vol_df.head(20).copy(), it would show Symbol as index, but st.dataframe would handle it, but for consistency, reset_index.

In the full data table too.

In the key insights, len(high_vol_df) etc., it's fine.

But for the chart, need Symbol as column.

In the code_execution tool, I can test this.

But the question is: "ValueError: This app has encountered an error. ..."

The user is showing the error, probably wants me to fix the code.

The current user question is the error message.

I need to debug and fix it.

Since it's code, I can use code_execution to test.

But first, understand.

Yes, the problem is that after groupby, Symbol is index, so for px.bar, x='Symbol' won't work because 'Symbol' isn't a column.

Solution: high_vol_df = high_vol_df.reset_index()

Before displaying.

Also, in display_df, after copy, but better to reset before.

To confirm, I can think of a small example.

Suppose df = pd.DataFrame({'Symbol': ['A', 'A', 'B'], 'Premium': [1,2,3]})

grouped = df.groupby('Symbol').agg({'Premium':'sum'})

Then grouped.index = ['A','B'], columns=['Premium']

px.bar(grouped, x='Symbol') would fail, because no column 'Symbol'.

Yes.

Need grouped.reset_index()

Yes.

So, in the code, after grouped = grouped.sort_values('total_premium', ascending=False)

Add high_vol_df = grouped.reset_index()

Yes.

In the function: high_vol_df = filter_high_volume_flows(data)

Then in tab4, after if high_vol_df.empty:

Then st.markdown("#### Key Insights")

Then total_premium = high_vol_df['total_premium'].sum() etc.

Fine, since index doesn't affect sum.

Then display_df = high_vol_df.head(20).copy()

Then display_df['total_premium'] = ... apply format

But when st.dataframe(display_df), if Symbol is index, it will show Symbol as first column anyway.

But for px.bar(high_vol_df.head(10), x='Symbol', y='total_premium')

It will fail if 'Symbol' not column.

So, need to reset_index().

Also, in the full st.dataframe(high_vol_df.style.format(...))

It will work, but to be consistent.

In the fix, add high_vol_df = high_vol_df.reset_index()

After getting high_vol_df.

But wait, in the function filter_high_volume_flows, at the end:

return grouped

I can add return grouped.reset_index()

That's better.

Modify the function to return with reset_index.

Yes.

In the code, change:

grouped = grouped.sort_values('total_premium', ascending=False)

return grouped.reset_index()

Yes.

That way, Symbol is a column.

Also, need to make sure 'total_premium' is numeric for the chart, but in display_df, we format to string, but for fig, we use high_vol_df.head(10), which has numeric.

Yes.

In key insights, sum is fine.

Yes.

In the function, before reset, the sums are on numeric.

Yes.

Also, one thing: in filter_high_volume_flows, there is no days_limit or min_premium, it's separate from the other filter.

It filters Volume >1000 and Last Price >1, and Expiration >= now.

But no min_premium, it calculates Premium, but no filter on it.

And no exclude_symbols.

And it's for all symbols, not selected.

Yes, as per request: "for all stocks"

Yes.

Also, in the px.bar, y='total_premium', but after format in display, but no, fig uses original.

Yes.

I think that's the fix.

Now, there might be more, but let's see.

In the traceback, it's in process_args_into_dataframe, raise ValueError.

Likely "Could not interpret value `Symbol` for `x`. An entry with this name does not appear in `data_frame`."

Yes.

So, to output the fixed code.

The user is showing the error, so I need to provide the corrected code.

But the human message is the error.

So, my response should be the fixed code.

But wait, the system prompt says: Do not mention these guidelines...

But anyway.

Perhaps use tools if needed.

But I think I can fix it directly.

One more: in the function, df = df[df['Volume'] > 1000]

df['Volume'] >=100 in fetch, but >1000 here.

Yes.

And Last Price >1.

But in columns, it's 'Last Price'?

In required_columns = ['Symbol', 'Strike Price', 'Call/Put', 'Expiration', 'Volume', 'Last Price']

Yes.

Assuming the data has it.

Yes.

I think that's it.```python
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

    with tab4:
        st.markdown("### 📊 Summary of High Volume Options Flows")
        st.markdown("Showing options with Volume > 1000 and Last Price > 1, grouped by stock for all available symbols.")

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

            high_vol_df = filter_high_volume_flows(data)

        if high_vol_df.empty:
            st.info("No options meet the high volume criteria.")
        else:
            # Intelligent summary
            st.markdown("#### Key Insights")
            total_premium = high_vol_df['total_premium'].sum()
            total_options = high_vol_df['option_count'].sum()
            bullish_stocks = len(high_vol_df[high_vol_df['net_sentiment'] > 0])
            bearish_stocks = len(high_vol_df[high_vol_df['net_sentiment'] < 0])
            neutral_stocks = len(high_vol_df[high_vol_df['net_sentiment'] == 0])

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Premium", f"${total_premium / 1000000:.1f}M")
            with col2:
                st.metric("Total Options", f"{total_options:,}")
            with col3:
                st.metric("Active Stocks", len(high_vol_df))

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Bullish Stocks", bullish_stocks)
            with col2:
                st.metric("Bearish Stocks", bearish_stocks)
            with col3:
                st.metric("Neutral Stocks", neutral_stocks)

            st.markdown("#### Top Stocks by Total Premium")
            display_df = high_vol_df.head(20).copy()
            display_df['total_premium'] = display_df['total_premium'].apply(lambda x: f"${x:,.0f}")
            display_df['call_premium'] = display_df['call_premium'].apply(lambda x: f"${x:,.0f}")
            display_df['put_premium'] = display_df['put_premium'].apply(lambda x: f"${x:,.0f}")
            display_df['total_volume'] = display_df['total_volume'].apply(lambda x: f"{x:,.0f}")
            display_df['net_sentiment'] = display_df['net_sentiment'].round(2)
            st.dataframe(display_df)

            # Bar chart for top 10
            fig = px.bar(
                high_vol_df.head(10),
                x='Symbol',
                y='total_premium',
                title="Top 10 Stocks by Total Premium",
                labels={'total_premium': 'Total Premium ($)'}
            )
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("Full Data Table"):
                st.dataframe(high_vol_df.style.format({
                    'total_premium': '${:,.0f}',
                    'total_volume': '{:,.0f}',
                    'option_count': '{:,.0f}',
                    'call_premium': '${:,.0f}',
                    'put_premium': '${:,.0f}',
                    'net_sentiment': '{:.2f}'
                }))

    # Auto-refresh (only for overview tab)
    if st.session_state.auto_refresh:
        st_autorefresh(interval=REFRESH_INTERVAL * 1000, key="optionsflowrefresh")

if __name__ == "__main__":
    main()
