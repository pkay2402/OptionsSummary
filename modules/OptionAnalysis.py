import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta
import os

# Function to load and process CSV file
def load_csv(file):
    try:
        df = pd.read_csv(file)
        
        # Define expected columns
        required_columns = ['Ticker', 'Expiration Date', 'Contract Type', 'Strike Price', 
                           'Reference Price', 'Size', 'Premium Price', 'Side Code', 
                           'Is Opening Position', 'Money Type', 'Is Unusual', 'Is Golden Sweep']
        
        # Map columns by position if names don't match
        if not all(col in df.columns for col in required_columns):
            column_mapping = {
                2: 'Ticker', 3: 'Expiration Date', 4: 'Days Until Expiration',
                5: 'Strike Price', 6: 'Contract Type', 7: 'Reference Price',
                8: 'Size', 9: 'Option Price', 12: 'Premium Price', 19: 'Side Code',
                15: 'Is Unusual', 16: 'Is Golden Sweep', 17: 'Is Opening Position',
                18: 'Money Type'
            }
            df.columns = [column_mapping.get(i, col) for i, col in enumerate(df.columns)]
        
        # Convert and clean data
        df['Expiration Date'] = pd.to_datetime(df['Expiration Date'], errors='coerce')
        numeric_cols = ['Strike Price', 'Reference Price', 'Size', 'Premium Price']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace('[\$,]', '', regex=True), 
                                  errors='coerce')
        
        categorical_cols = ['Ticker', 'Contract Type', 'Side Code', 'Is Unusual', 
                          'Is Golden Sweep', 'Is Opening Position', 'Money Type']
        for col in categorical_cols:
            df[col] = df[col].fillna('N/A').astype(str).str.strip().str.upper()
        
        # Filter for OTM and opening positions
        df = df[(df['Money Type'] == 'OUT_THE_MONEY') & (df['Is Opening Position'] == 'YES')]
        
        return df
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return None

# Function to split content for Discord
def split_for_discord(content, max_length=2000):
    if len(content) <= max_length:
        return [content]
    
    parts = []
    current_part = ""
    for line in content.split('\n'):
        if len(current_part) + len(line) + 1 > max_length:
            parts.append(current_part.strip())
            current_part = line
        else:
            current_part += f"\n{line}" if current_part else line
    
    if current_part:
        parts.append(current_part.strip())
    
    return parts

# Function to send content to Discord
def send_to_discord(content, webhook_url):
    try:
        parts = split_for_discord(content)
        for i, part in enumerate(parts, 1):
            payload = {"content": f"**Part {i}/{len(parts)}**\n{part}"}
            response = requests.post(webhook_url, json=payload)
            if response.status_code != 204:
                return f"Failed to send part {i}: {response.status_code}"
        return f"Sent newsletter in {len(parts)} parts."
    except Exception as e:
        return f"Discord send error: {e}"

# Function to generate newsletter
def generate_newsletter(df, top_n=10, min_premium=250000, side_codes=['A', 'AA'], 
                      tickers=None, sort_by='Premium Price'):
    if df is None or df.empty:
        return "No valid data for newsletter."
    
    today = pd.to_datetime("today").normalize()
    max_date = today + timedelta(days=32)  # Next 2 weeks
    exclude_tickers = {'SPY', 'QQQ', 'SPX', 'SPXW', 'IWM', 'NDX', 'RUT'}
    
    # Filter data
    filtered_df = df[
        (df['Expiration Date'] > today) &
        (df['Expiration Date'] <= max_date) &
        (df['Premium Price'] >= min_premium) &
        (df['Side Code'].isin(side_codes)) &
        (~df['Ticker'].isin(exclude_tickers))
    ]
    
    if tickers:
        filtered_df = filtered_df[filtered_df['Ticker'].isin(tickers)]
    
    if filtered_df.empty:
        return "No OTM flows match the criteria."
    
    # Initialize newsletter
    newsletter = f"📊 **Options Flow Newsletter - {today.strftime('%B %d, %Y')}** 📊\n\n"
    newsletter += "🔍 Analyzing OUT-THE-MONEY opening positions for the next 4 weeks\n\n"
    
    # Long Plays (Bullish: CALL A/AA or PUT B/BB)
    long_df = filtered_df[
        ((filtered_df['Contract Type'] == 'CALL') & (filtered_df['Side Code'].isin(['A', 'AA']))) |
        ((filtered_df['Contract Type'] == 'PUT') & (filtered_df['Side Code'].isin(['B', 'BB'])))
    ]
    
    newsletter += "**📈 Top Long Plays**\n"
    if long_df.empty:
        newsletter += "No bullish OTM flows detected.\n"
    else:
        if sort_by == 'Premium Price':
            long_df = long_df.sort_values('Premium Price', ascending=False)
        else:
            long_df = long_df.sort_values(['Ticker', 'Premium Price'])
        
        for _, row in long_df.head(top_n).iterrows():
            move_pct = abs((row['Strike Price'] - row['Reference Price']) / 
                          row['Reference Price'] * 100)
            flags = []
            if row['Is Unusual'] == 'YES':
                flags.append("UNUSUAL")
            if row['Is Golden Sweep'] == 'YES':
                flags.append("GOLDEN")
            flags_str = f" [{' '.join(flags)}]" if flags else ""
            
            newsletter += (
                f"• **{row['Ticker']}** {row['Contract Type']} ${row['Strike Price']:,.2f} "
                f"exp {row['Expiration Date'].strftime('%Y-%m-%d')} - "
                f"${row['Premium Price']:,.0f} ({row['Size']} contracts, "
                f"{move_pct:.1f}% move, 🟢 {row['Side Code']}){flags_str}\n"
            )
    
    newsletter += "\n"
    
    # Short Plays (Bearish: CALL B/BB or PUT A/AA)
    short_df = filtered_df[
        ((filtered_df['Contract Type'] == 'CALL') & (filtered_df['Side Code'].isin(['B', 'BB']))) |
        ((filtered_df['Contract Type'] == 'PUT') & (filtered_df['Side Code'].isin(['A', 'AA'])))
    ]
    
    newsletter += "**📉 Top Short Plays**\n"
    if short_df.empty:
        newsletter += "No bearish OTM flows detected.\n"
    else:
        if sort_by == 'Premium Price':
            short_df = short_df.sort_values('Premium Price', ascending=False)
        else:
            short_df = short_df.sort_values(['Ticker', 'Premium Price'])
        
        for _, row in short_df.head(top_n).iterrows():
            move_pct = abs((row['Strike Price'] - row['Reference Price']) / 
                          row['Reference Price'] * 100)
            flags = []
            if row['Is Unusual'] == 'YES':
                flags.append("UNUSUAL")
            if row['Is Golden Sweep'] == 'YES':
                flags.append("GOLDEN")
            flags_str = f" [{' '.join(flags)}]" if flags else ""
            
            newsletter += (
                f"• **{row['Ticker']}** {row['Contract Type']} ${row['Strike Price']:,.2f} "
                f"exp {row['Expiration Date'].strftime('%Y-%m-%d')} - "
                f"${row['Premium Price']:,.0f} ({row['Size']} contracts, "
                f"{move_pct:.1f}% move, 🔴 {row['Side Code']}){flags_str}\n"
            )
    
    newsletter += "\n⚠️ **For educational purposes only. Not financial advice.**"
    return newsletter

# Function to generate Twitter newsletter
def generate_twitter_newsletter(df, top_n=5, min_premium=500000, side_codes=['A', 'AA'], 
                               tickers=None, sort_by='Premium Price'):
    if df is None or df.empty:
        return "No valid data for Twitter newsletter."
    
    today = pd.to_datetime("today").normalize()
    max_date = today + timedelta(days=32)
    exclude_tickers = {'SPY', 'QQQ', 'SPX', 'SPXW', 'IWM', 'NDX', 'RUT'}
    
    # Filter data
    filtered_df = df[
        (df['Expiration Date'] > today) &
        (df['Expiration Date'] <= max_date) &
        (df['Premium Price'] >= min_premium) &
        (df['Side Code'].isin(side_codes)) &
        (~df['Ticker'].isin(exclude_tickers))
    ]
    
    if tickers:
        filtered_df = filtered_df[filtered_df['Ticker'].isin(tickers)]
    
    if filtered_df.empty:
        return "No OTM flows match the criteria for Twitter."
    
    # Twitter newsletter - shorter format
    twitter_post = f"📊 OPTIONS FLOW ALERT - {today.strftime('%m/%d/%Y')}\n\n"
    twitter_post += "🎯 TOP OTM PLAYS:\n\n"
    
    # Combine long and short plays, sorted by premium
    all_flows = filtered_df.copy()
    if sort_by == 'Premium Price':
        all_flows = all_flows.sort_values('Premium Price', ascending=False)
    else:
        all_flows = all_flows.sort_values(['Ticker', 'Premium Price'])
    
    for i, (_, row) in enumerate(all_flows.head(top_n).iterrows(), 1):
        # Determine sentiment
        is_bullish = ((row['Contract Type'] == 'CALL' and row['Side Code'] in ['A', 'AA']) or
                     (row['Contract Type'] == 'PUT' and row['Side Code'] in ['B', 'BB']))
        sentiment = "🟢" if is_bullish else "🔴"
        
        move_pct = abs((row['Strike Price'] - row['Reference Price']) / row['Reference Price'] * 100)
        
        # Flags
        flags = []
        if row['Is Unusual'] == 'YES':
            flags.append("🔥")
        if row['Is Golden Sweep'] == 'YES':
            flags.append("⚡")
        flag_str = " ".join(flags)
        
        twitter_post += (f"{i}. {sentiment} ${row['Ticker']} "
                        f"{row['Contract Type']} ${row['Strike Price']:,.0f}\n"
                        f"   Exp: {row['Expiration Date'].strftime('%m/%d')} | "
                        f"${row['Premium Price']/1000000:.1f}M | "
                        f"{move_pct:.0f}% move {flag_str}\n\n")
    
    twitter_post += "⚠️ Educational only. Not financial advice.\n"
    twitter_post += "#OptionsFlow #Trading #StockMarket"
    
    return twitter_post

# Function to display flows for a specific symbol
def display_symbol_flows(df, symbol):
    if df is None or df.empty:
        st.warning("No data available.")
        return
    
    today = pd.to_datetime("today").normalize()
    symbol_df = df[
        (df['Ticker'] == symbol.upper()) &
        (df['Is Opening Position'] == 'YES') &
        (df['Premium Price'] > 250000) &
        (df['Expiration Date'] > today)
    ].sort_values(['Expiration Date', 'Premium Price'])
    
    if symbol_df.empty:
        st.warning(f"No qualifying flows for {symbol}.")
        return
    
    st.subheader(f"Flows for {symbol} (Premium > $250K)")
    for _, row in symbol_df.iterrows():
        move_pct = abs((row['Strike Price'] - row['Reference Price']) / 
                      row['Reference Price'] * 100)
        sentiment = (
            "🟢" if (row['Contract Type'] == 'CALL' and row['Side Code'] in ['A', 'AA']) or
                    (row['Contract Type'] == 'PUT' and row['Side Code'] in ['B', 'BB']) else
            "🔴" if (row['Contract Type'] == 'CALL' and row['Side Code'] in ['B', 'BB']) or
                    (row['Contract Type'] == 'PUT' and row['Side Code'] in ['A', 'AA']) else "N/A"
        )
        flags = []
        if row['Is Unusual'] == 'YES':
            flags.append("UNUSUAL")
        if row['Is Golden Sweep'] == 'YES':
            flags.append("GOLDEN")
        flags_str = f" [{' '.join(flags)}]" if flags else ""
        
        st.markdown(
            f"• **{row['Contract Type']}** | Strike: ${row['Strike Price']:,.2f} | "
            f"Exp: {row['Expiration Date'].strftime('%Y-%m-%d')} | "
            f"Premium: ${row['Premium Price']:,.0f} | Contracts: {row['Size']} | "
            f"Move: {move_pct:.1f}% | Sentiment: {sentiment} | Side: {row['Side Code']}{flags_str}"
        )
    
    with st.expander("Raw Data"):
        st.dataframe(symbol_df)

# Function to display repeat flows grouped by ticker
def display_repeat_flows(df, min_premium=30000):
    if df is None or df.empty:
        st.warning("No data available.")
        return
    
    # Filter by minimum premium and opening positions
    filtered_df = df[
        (df['Premium Price'] >= min_premium) &
        (df['Is Opening Position'] == 'YES')
    ].copy()
    
    if filtered_df.empty:
        st.warning(f"No flows found with premium >= ${min_premium:,}")
        return
    
    # Create a composite key for similar contracts
    filtered_df['Contract_Key'] = (
        filtered_df['Ticker'] + '_' + 
        filtered_df['Contract Type'] + '_' + 
        filtered_df['Strike Price'].astype(str) + '_' + 
        filtered_df['Expiration Date'].dt.strftime('%Y-%m-%d')
    )
    
    # Group by ticker and contract key to find repeats
    grouped = filtered_df.groupby(['Ticker', 'Contract_Key']).agg({
        'Premium Price': ['count', 'sum', 'mean'],
        'Size': 'sum',
        'Contract Type': 'first',
        'Strike Price': 'first',
        'Expiration Date': 'first',
        'Reference Price': 'first',
        'Side Code': lambda x: list(x),
        'Is Unusual': lambda x: 'YES' if 'YES' in x.values else 'NO',
        'Is Golden Sweep': lambda x: 'YES' if 'YES' in x.values else 'NO'
    }).reset_index()
    
    # Flatten column names
    grouped.columns = ['Ticker', 'Contract_Key', 'Flow_Count', 'Total_Premium', 'Avg_Premium',
                      'Total_Contracts', 'Contract_Type', 'Strike_Price', 'Expiration_Date',
                      'Reference_Price', 'Side_Codes', 'Has_Unusual', 'Has_Golden']
    
    # Filter for repeat flows (more than 1 occurrence)
    repeat_flows = grouped[grouped['Flow_Count'] > 1].copy()
    
    if repeat_flows.empty:
        st.warning("No repeat flows found with the specified criteria.")
        return
    
    # Sort by total premium
    repeat_flows = repeat_flows.sort_values('Total_Premium', ascending=False)
    
    st.subheader(f"🔄 Repeat Flows (Premium >= ${min_premium:,})")
    st.markdown(f"Found **{len(repeat_flows)}** contracts with multiple flows")
    
    # Group by ticker for display
    for ticker in repeat_flows['Ticker'].unique():
        ticker_flows = repeat_flows[repeat_flows['Ticker'] == ticker]
        
        with st.expander(f"📈 {ticker} ({len(ticker_flows)} repeat contracts)", expanded=True):
            for _, row in ticker_flows.iterrows():
                move_pct = abs((row['Strike_Price'] - row['Reference_Price']) / 
                              row['Reference_Price'] * 100)
                
                # Determine predominant sentiment
                side_codes = row['Side_Codes']
                bullish_sides = sum(1 for side in side_codes if side in ['A', 'AA'])
                bearish_sides = sum(1 for side in side_codes if side in ['B', 'BB'])
                
                if row['Contract_Type'] == 'CALL':
                    if bullish_sides > bearish_sides:
                        sentiment = "🟢 Bullish"
                    elif bearish_sides > bullish_sides:
                        sentiment = "🔴 Bearish"
                    else:
                        sentiment = "⚪ Mixed"
                else:  # PUT
                    if bullish_sides > bearish_sides:
                        sentiment = "🔴 Bearish"
                    elif bearish_sides > bullish_sides:
                        sentiment = "🟢 Bullish"
                    else:
                        sentiment = "⚪ Mixed"
                
                flags = []
                if row['Has_Unusual'] == 'YES':
                    flags.append("🔥 UNUSUAL")
                if row['Has_Golden'] == 'YES':
                    flags.append("⚡ GOLDEN")
                flags_str = f" [{' '.join(flags)}]" if flags else ""
                
                # Display the flow information
                st.markdown(f"""
                **{row['Contract_Type']} ${row['Strike_Price']:,.2f}** exp {row['Expiration_Date'].strftime('%Y-%m-%d')}
                - **{row['Flow_Count']} flows** totaling **${row['Total_Premium']:,.0f}** 
                - Avg Premium: ${row['Avg_Premium']:,.0f} | Total Contracts: {row['Total_Contracts']:,}
                - Move Required: {move_pct:.1f}% | Sentiment: {sentiment}
                - Side Codes: {', '.join(map(str, row['Side_Codes']))}{flags_str}
                """)
                st.divider()
    
    # Summary statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Repeat Contracts", len(repeat_flows))
    
    with col2:
        total_premium = repeat_flows['Total_Premium'].sum()
        st.metric("Total Premium", f"${total_premium/1000000:.1f}M")
    
    with col3:
        avg_flows_per_contract = repeat_flows['Flow_Count'].mean()
        st.metric("Avg Flows per Contract", f"{avg_flows_per_contract:.1f}")
    
    with col4:
        most_active = repeat_flows.loc[repeat_flows['Flow_Count'].idxmax(), 'Ticker']
        max_flows = repeat_flows['Flow_Count'].max()
        st.metric("Most Active", f"{most_active} ({max_flows} flows)")
    
    # Show raw data
    with st.expander("Raw Repeat Flows Data"):
        display_df = repeat_flows.copy()
        display_df['Side_Codes'] = display_df['Side_Codes'].apply(lambda x: ', '.join(map(str, x)))
        st.dataframe(display_df, use_container_width=True)

# Add this function after the existing functions and before the main() function

def calculate_trade_score(row, ticker_daily_flow):
    """
    Calculate a composite score for each trade based on:
    - Premium size (weight: 40%)
    - Number of sweeps/frequency (weight: 30%) 
    - Percentage of daily flow (weight: 20%)
    - Special flags (weight: 10%)
    """
    # Base score from premium (normalized to 0-100)
    premium_score = min(100, (row['Total_Premium'] / 10000000) * 100)  # $10M = 100 points
    
    # Frequency score (normalized to 0-100)
    frequency_score = min(100, (row['Flow_Count'] / 50) * 100)  # 50 sweeps = 100 points
    
    # Daily flow percentage score (normalized to 0-100)
    daily_flow_pct = (row['Total_Premium'] / ticker_daily_flow) * 100
    daily_flow_score = min(100, daily_flow_pct * 5)  # 20% of daily flow = 100 points
    
    # Special flags bonus
    flags_score = 0
    if row['Has_Unusual'] == 'YES':
        flags_score += 10
    if row['Has_Golden'] == 'YES':
        flags_score += 15
    
    # Weighted composite score
    composite_score = (
        premium_score * 0.4 +
        frequency_score * 0.3 +
        daily_flow_score * 0.2 +
        flags_score * 0.1
    )
    
    return composite_score, daily_flow_pct

def display_top_trades_summary(df, min_premium=250000, min_flows=3):
    """
    Display top scoring trade for each ticker
    """
    if df is None or df.empty:
        st.warning("No data available.")
        return
    
    today = pd.to_datetime("today").normalize()
    exclude_tickers = {'SPY', 'QQQ', 'SPX', 'SPXW', 'IWM', 'NDX', 'RUT'}
    
    # Filter for relevant data
    filtered_df = df[
        (df['Premium Price'] >= min_premium) &
        (df['Is Opening Position'] == 'YES') &
        (~df['Ticker'].isin(exclude_tickers)) &
        (df['Expiration Date'] > today)
    ].copy()
    
    if filtered_df.empty:
        st.warning(f"No flows found with premium >= ${min_premium:,}")
        return
    
    # Create contract grouping key
    filtered_df['Contract_Key'] = (
        filtered_df['Ticker'] + '_' + 
        filtered_df['Contract Type'] + '_' + 
        filtered_df['Strike Price'].astype(str) + '_' + 
        filtered_df['Expiration Date'].dt.strftime('%Y-%m-%d')
    )
    
    # Group by ticker and contract to get sweep counts
    contract_groups = filtered_df.groupby(['Ticker', 'Contract_Key']).agg({
        'Premium Price': ['count', 'sum'],
        'Size': 'sum',
        'Contract Type': 'first',
        'Strike Price': 'first',
        'Expiration Date': 'first',
        'Reference Price': 'first',
        'Side Code': lambda x: list(x),
        'Is Unusual': lambda x: 'YES' if 'YES' in x.values else 'NO',
        'Is Golden Sweep': lambda x: 'YES' if 'YES' in x.values else 'NO'
    }).reset_index()
    
    # Flatten column names
    contract_groups.columns = ['Ticker', 'Contract_Key', 'Flow_Count', 'Total_Premium', 
                              'Total_Contracts', 'Contract_Type', 'Strike_Price', 
                              'Expiration_Date', 'Reference_Price', 'Side_Codes', 
                              'Has_Unusual', 'Has_Golden']
    
    # Filter for minimum number of flows
    contract_groups = contract_groups[contract_groups['Flow_Count'] >= min_flows]
    
    if contract_groups.empty:
        st.warning(f"No contracts found with >= {min_flows} flows")
        return
    
    # Calculate daily flow by ticker
    ticker_daily_flows = filtered_df.groupby('Ticker')['Premium Price'].sum()
    
    # Calculate scores for each contract
    scores_data = []
    for _, row in contract_groups.iterrows():
        ticker_daily_flow = ticker_daily_flows.get(row['Ticker'], 1)  # Avoid division by zero
        score, daily_flow_pct = calculate_trade_score(row, ticker_daily_flow)
        
        # Determine sentiment
        side_codes = row['Side_Codes']
        bullish_sides = sum(1 for side in side_codes if side in ['A', 'AA'])
        bearish_sides = sum(1 for side in side_codes if side in ['B', 'BB'])
        
        if row['Contract_Type'] == 'CALL':
            if bullish_sides > bearish_sides:
                sentiment = "📈 Bullish"
            elif bearish_sides > bullish_sides:
                sentiment = "📉 Bearish"
            else:
                sentiment = "⚪ Mixed"
        else:  # PUT
            if bullish_sides > bearish_sides:
                sentiment = "📉 Bearish"
            elif bearish_sides > bullish_sides:
                sentiment = "📈 Bullish"
            else:
                sentiment = "⚪ Mixed"
        
        scores_data.append({
            'Ticker': row['Ticker'],
            'Score': score,
            'Contract_Type': row['Contract_Type'],
            'Strike_Price': row['Strike_Price'],
            'Expiration_Date': row['Expiration_Date'],
            'Flow_Count': row['Flow_Count'],
            'Total_Premium': row['Total_Premium'],
            'Daily_Flow_Pct': daily_flow_pct,
            'Sentiment': sentiment,
            'Has_Unusual': row['Has_Unusual'],
            'Has_Golden': row['Has_Golden']
        })
    
    if not scores_data:
        st.warning("No qualifying trades found for scoring.")
        return
    
    # Convert to DataFrame and get top trade per ticker
    scores_df = pd.DataFrame(scores_data)
    top_trades = scores_df.loc[scores_df.groupby('Ticker')['Score'].idxmax()].sort_values('Score', ascending=False)
    
    # Generate the summary output
    st.subheader("📊 TOP TRADES SUMMARY - Highest Score Trade for Each Ticker")
    st.markdown("=" * 80)
    
    summary_text = ""
    for _, trade in top_trades.head(15).iterrows():  # Top 15 tickers
        # Format premium
        if trade['Total_Premium'] >= 1000000:
            premium_str = f"${trade['Total_Premium']/1000000:.2f}M"
        else:
            premium_str = f"${trade['Total_Premium']/1000:.0f}K"
        
        # Special flags
        flags = []
        if trade['Has_Unusual'] == 'YES':
            flags.append("UNUSUAL")
        if trade['Has_Golden'] == 'YES':
            flags.append("GOLDEN")
        flags_str = f" [{' '.join(flags)}]" if flags else ""
        
        line = (f"  ${trade['Ticker']} {trade['Sentiment']} "
                f"${trade['Strike_Price']:,.2f} {trade['Contract_Type']} "
                f"Exp: {trade['Expiration_Date'].strftime('%Y-%m-%d')} "
                f"({trade['Flow_Count']} sweeps) → {premium_str} "
                f"({trade['Daily_Flow_Pct']:.1f}% of daily flow) "
                f"[score: {trade['Score']:.1f}]{flags_str}")
        
        summary_text += line + "\n"
        st.markdown(line)
    
    st.markdown("=" * 80)
    
    # Add copy-to-clipboard functionality
    with st.expander("📋 Copy Summary Text"):
        full_summary = f"📊 TOP TRADES SUMMARY - Highest Score Trade for Each Ticker\n{'='*80}\n{summary_text}{'='*80}"
        st.text_area("Copy this text:", full_summary, height=400)
    
    # Show scoring methodology
    with st.expander("ℹ️ Scoring Methodology"):
        st.markdown("""
        **Trade Score Calculation:**
        - **Premium Size (40%)**: Higher premiums get higher scores (max at $10M)
        - **Sweep Frequency (30%)**: More sweeps indicate sustained interest (max at 50 sweeps)
        - **Daily Flow % (20%)**: Higher percentage of ticker's daily flow (max at 20%)
        - **Special Flags (10%)**: Bonus for UNUSUAL (+10) and GOLDEN SWEEP (+15)
        
        **Sentiment Logic:**
        - CALL + A/AA sides = Bullish 📈
        - CALL + B/BB sides = Bearish 📉
        - PUT + A/AA sides = Bearish 📉
        - PUT + B/BB sides = Bullish 📈
        """)
    
    # Summary statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Unique Tickers", len(top_trades))
    
    with col2:
        total_premium = top_trades['Total_Premium'].sum()
        st.metric("Total Premium", f"${total_premium/1000000:.1f}M")
    
    with col3:
        avg_score = top_trades['Score'].mean()
        st.metric("Average Score", f"{avg_score:.1f}")
    
    with col4:
        top_score = top_trades['Score'].max()
        st.metric("Highest Score", f"{top_score:.1f}")

# Function to analyze symbol flows and project path
def analyze_symbol_flows(df, symbols, min_premium=100000):
    """
    Analyze option flows for specific symbols and project price path
    """
    if df is None or df.empty:
        st.warning("No data available.")
        return
    
    today = pd.to_datetime("today").normalize()
    next_friday = today + timedelta(days=(4 - today.weekday()) % 7)  # Next Friday
    if next_friday <= today:
        next_friday += timedelta(days=7)
    
    for symbol in symbols:
        symbol = symbol.upper().strip()
        
        # Filter flows for this symbol
        symbol_df = df[
            (df['Ticker'] == symbol) &
            (df['Is Opening Position'] == 'YES') &
            (df['Premium Price'] >= min_premium) &
            (df['Expiration Date'] > today)
        ].copy()
        
        if symbol_df.empty:
            st.warning(f"No qualifying flows found for {symbol} (Premium >= ${min_premium:,})")
            continue
        
        st.subheader(f"📊 {symbol} Flow Analysis & Projected Path")
        
        # Get current reference price (assuming it's the most recent)
        current_price = symbol_df['Reference Price'].iloc[0]
        
        # Separate calls and puts
        calls_df = symbol_df[symbol_df['Contract Type'] == 'CALL']
        puts_df = symbol_df[symbol_df['Contract Type'] == 'PUT']
        
        # Calculate bullish vs bearish sentiment
        bullish_premium = 0
        bearish_premium = 0
        
        # CALL A/AA = Bullish, CALL B/BB = Bearish
        # PUT A/AA = Bearish, PUT B/BB = Bullish
        for _, row in symbol_df.iterrows():
            if row['Contract Type'] == 'CALL':
                if row['Side Code'] in ['A', 'AA']:
                    bullish_premium += row['Premium Price']
                elif row['Side Code'] in ['B', 'BB']:
                    bearish_premium += row['Premium Price']
            else:  # PUT
                if row['Side Code'] in ['A', 'AA']:
                    bearish_premium += row['Premium Price']
                elif row['Side Code'] in ['B', 'BB']:
                    bullish_premium += row['Premium Price']
        
        total_premium = bullish_premium + bearish_premium
        if total_premium > 0:
            bullish_pct = (bullish_premium / total_premium) * 100
            bearish_pct = (bearish_premium / total_premium) * 100
        else:
            bullish_pct = bearish_pct = 0
        
        # Overall sentiment
        if bullish_pct > 60:
            overall_sentiment = "🟢 BULLISH"
            sentiment_strength = "Strong" if bullish_pct > 75 else "Moderate"
        elif bearish_pct > 60:
            overall_sentiment = "🔴 BEARISH"
            sentiment_strength = "Strong" if bearish_pct > 75 else "Moderate"
        else:
            overall_sentiment = "🟡 NEUTRAL/MIXED"
            sentiment_strength = "Balanced"
        
        # Display current status
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Current Price", f"${current_price:.2f}")
        
        with col2:
            st.metric("Overall Sentiment", overall_sentiment.split()[1], delta=f"{sentiment_strength}")
        
        with col3:
            st.metric("Bullish Premium", f"${bullish_premium/1000000:.1f}M", delta=f"{bullish_pct:.1f}%")
        
        with col4:
            st.metric("Bearish Premium", f"${bearish_premium/1000000:.1f}M", delta=f"{bearish_pct:.1f}%")
        
        # Calculate key levels for next week projection
        next_week_expiry = symbol_df[
            (symbol_df['Expiration Date'] >= next_friday) &
            (symbol_df['Expiration Date'] <= next_friday + timedelta(days=7))
        ]
        
        if not next_week_expiry.empty:
            # Get most active strikes
            call_strikes = calls_df['Strike Price'].value_counts().head(10)
            put_strikes = puts_df['Strike Price'].value_counts().head(10)
            
            # Calculate weighted average target based on premium flow
            weighted_call_target = 0
            weighted_put_target = 0
            call_premium_total = 0
            put_premium_total = 0
            
            # Weight call strikes by premium (bullish flows only)
            bullish_calls = calls_df[calls_df['Side Code'].isin(['A', 'AA'])]
            for _, row in bullish_calls.iterrows():
                if row['Strike Price'] > current_price:  # OTM calls
                    weighted_call_target += row['Strike Price'] * row['Premium Price']
                    call_premium_total += row['Premium Price']
            
            # Weight put strikes by premium (bullish flows only - put selling)
            bullish_puts = puts_df[puts_df['Side Code'].isin(['B', 'BB'])]
            for _, row in bullish_puts.iterrows():
                if row['Strike Price'] < current_price:  # OTM puts
                    weighted_put_target += row['Strike Price'] * row['Premium Price']
                    put_premium_total += row['Premium Price']
            
            # Calculate projected targets
            upside_target = weighted_call_target / call_premium_total if call_premium_total > 0 else current_price
            downside_support = weighted_put_target / put_premium_total if put_premium_total > 0 else current_price
            
            # Find resistance and support levels
            otm_call_strikes = sorted([strike for strike in call_strikes.index if strike > current_price])
            otm_put_strikes = sorted([strike for strike in put_strikes.index if strike < current_price], reverse=True)
            
            resistance_1 = otm_call_strikes[0] if otm_call_strikes else current_price * 1.02
            resistance_2 = otm_call_strikes[1] if len(otm_call_strikes) > 1 else current_price * 1.05
            
            support_1 = otm_put_strikes[0] if otm_put_strikes else current_price * 0.98
            support_2 = otm_put_strikes[1] if len(otm_put_strikes) > 1 else current_price * 0.95
            
            # Display projection
            st.markdown("### 🎯 Next Week Projection")
            
            if bullish_pct > bearish_pct:
                move_direction = "higher"
                primary_target = upside_target
                move_pct = ((primary_target - current_price) / current_price) * 100
            else:
                move_direction = "lower"
                primary_target = downside_support
                move_pct = ((primary_target - current_price) / current_price) * 100
            
            st.markdown(f"""
            **{symbol} Analysis (Current: ${current_price:.2f})**
            
            Based on options flow analysis:
            - **Overall Bias**: {overall_sentiment} ({sentiment_strength})
            - **Expected Direction**: Likely to move {move_direction}
            - **Primary Target**: ${primary_target:.2f} ({move_pct:+.1f}%)
            - **Next Friday Expiry**: {next_friday.strftime('%Y-%m-%d')}
            
            **Key Levels for Next Week:**
            - 🔴 **Resistance 1**: ${resistance_1:.2f} ({((resistance_1 - current_price) / current_price * 100):+.1f}%)
            - 🔴 **Resistance 2**: ${resistance_2:.2f} ({((resistance_2 - current_price) / current_price * 100):+.1f}%)
            - 🟢 **Support 1**: ${support_1:.2f} ({((support_1 - current_price) / current_price * 100):+.1f}%)
            - 🟢 **Support 2**: ${support_2:.2f} ({((support_2 - current_price) / current_price * 100):+.1f}%)
            """)
            
            # Show reasoning
            with st.expander("📈 Analysis Details"):
                st.markdown(f"""
                **Flow Breakdown:**
                - Total Premium Analyzed: ${total_premium/1000000:.1f}M
                - Bullish Flows: ${bullish_premium/1000000:.1f}M ({bullish_pct:.1f}%)
                - Bearish Flows: ${bearish_premium/1000000:.1f}M ({bearish_pct:.1f}%)
                
                **Most Active Strikes (Calls):**
                """)
                for strike, count in call_strikes.head(5).items():
                    direction = "🔴" if strike > current_price else "🟢"
                    st.markdown(f"- {direction} ${strike:.2f}: {count} contracts")
                
                st.markdown("**Most Active Strikes (Puts):**")
                for strike, count in put_strikes.head(5).items():
                    direction = "🟢" if strike < current_price else "🔴"
                    st.markdown(f"- {direction} ${strike:.2f}: {count} contracts")
        
        # Show detailed flows
        with st.expander(f"📊 Detailed Flows for {symbol}"):
            # Sort by premium descending
            display_df = symbol_df.sort_values('Premium Price', ascending=False)
            
            for _, row in display_df.head(20).iterrows():  # Show top 20 flows
                move_pct = abs((row['Strike Price'] - row['Reference Price']) / 
                              row['Reference Price'] * 100)
                
                # Determine sentiment for this specific flow
                if row['Contract Type'] == 'CALL':
                    if row['Side Code'] in ['A', 'AA']:
                        flow_sentiment = "🟢 Bullish"
                    else:
                        flow_sentiment = "🔴 Bearish"
                else:  # PUT
                    if row['Side Code'] in ['A', 'AA']:
                        flow_sentiment = "🔴 Bearish"
                    else:
                        flow_sentiment = "🟢 Bullish"
                
                flags = []
                if row['Is Unusual'] == 'YES':
                    flags.append("🔥 UNUSUAL")
                if row['Is Golden Sweep'] == 'YES':
                    flags.append("⚡ GOLDEN")
                flags_str = f" [{' '.join(flags)}]" if flags else ""
                
                st.markdown(f"""
                **{row['Contract Type']} ${row['Strike Price']:,.2f}** exp {row['Expiration Date'].strftime('%Y-%m-%d')}
                - Premium: ${row['Premium Price']:,.0f} | Contracts: {row['Size']:,} | {flow_sentiment}
                - Move Required: {move_pct:.1f}% | Side: {row['Side Code']}{flags_str}
                """)
        
        st.divider()

# Main Streamlit app
def main():
    st.set_page_config(page_title="Options Flow Analyzer", page_icon="📊", layout="wide")
    st.title("🔍 Options Flow Analyzer")
    st.markdown("Generate a newsletter or view OTM option flows for the next 2 weeks.")
    
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    if uploaded_file:
        with st.spinner("Loading data..."):
            df = load_csv(uploaded_file)
        
        if df is not None:
            # Add the new tab here
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["Newsletter", "Symbol Flows", "Repeat Flows", "Top Trades Summary", "Symbol Analysis"])
            
            with tab1:
                st.subheader("Generate Newsletter")
                
                # Create two columns for different newsletter types
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 📄 Full Newsletter")
                    top_n = st.number_input("Number of Flows", min_value=1, max_value=50, value=20, key="full_top_n")
                    min_premium = st.number_input("Min Premium ($)", min_value=0, value=250000, key="full_premium")
                    side_codes = st.multiselect("Side Codes", ['A', 'AA', 'B', 'BB'], default=['A', 'AA'], key="full_sides")
                    with st.expander("Ticker Filter", expanded=False):
                        tickers = st.multiselect("Tickers", df['Ticker'].unique(), default=df['Ticker'].unique(), key="full_tickers")
                    sort_by = st.selectbox("Sort By", ["Premium Price", "Ticker"], key="full_sort")
                    
                    webhook_url = st.text_input(
                        "Discord Webhook URL", 
                        value=os.environ.get("DISCORD_WEBHOOK_URL", "https://discord.com/api/webhooks/1379692595961401406/D4v1I-h-7YKrk5KUutByAlBfheBfZmMbKydoX6_gcVnXM9AQYZXgC4twC-1T69O1MZ7h"), 
                        type="password"
                    )
                    send_discord = st.checkbox("Send to Discord")
                    
                    if st.button("Generate Full Newsletter"):
                        with st.spinner("Generating..."):
                            newsletter = generate_newsletter(df, top_n, min_premium, side_codes, 
                                                          tickers, sort_by)
                            st.markdown(f"```\n{newsletter}\n```")
                            if send_discord and webhook_url:
                                result = send_to_discord(newsletter, webhook_url)
                                st.info(result)
                
                with col2:
                    st.markdown("### 🐦 Twitter Newsletter")
                    twitter_top_n = st.number_input("Number of Flows", min_value=1, max_value=10, value=5, key="twitter_top_n")
                    twitter_min_premium = st.number_input("Min Premium ($)", min_value=0, value=500000, key="twitter_premium")
                    twitter_side_codes = st.multiselect("Side Codes", ['A', 'AA', 'B', 'BB'], default=['A', 'AA'], key="twitter_sides")
                    with st.expander("Ticker Filter", expanded=False):
                        twitter_tickers = st.multiselect("Tickers", df['Ticker'].unique(), default=df['Ticker'].unique(), key="twitter_tickers")
                    twitter_sort_by = st.selectbox("Sort By", ["Premium Price", "Ticker"], key="twitter_sort")
                    
                    if st.button("Generate Twitter Post"):
                        with st.spinner("Generating Twitter post..."):
                            twitter_newsletter = generate_twitter_newsletter(df, twitter_top_n, twitter_min_premium, 
                                                                           twitter_side_codes, twitter_tickers, twitter_sort_by)
                            st.markdown("**Copy this for Twitter:**")
                            st.text_area("Twitter Post", twitter_newsletter, height=400, key="twitter_output")

            with tab2:
                st.subheader("View Symbol Flows")
                symbol = st.text_input("Enter Symbol (e.g., AAPL)").upper()
                if symbol:
                    display_symbol_flows(df, symbol)
            
            with tab3:
                st.subheader("🔄 Repeat Flows Analysis")
                st.markdown("Identify contracts with multiple flows throughout the day")
                
                # Controls for repeat flows
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    repeat_min_premium = st.number_input(
                        "Minimum Premium ($)", 
                        min_value=0, 
                        value=15000, 
                        step=5000,
                        key="repeat_premium"
                    )
                
                with col2:
                    st.markdown("**Instructions:**")
                    st.markdown("- Shows contracts that appeared multiple times")
                    st.markdown("- Groups by ticker, contract type, strike, and expiration")
                    st.markdown("- Useful for spotting accumulation patterns")
                
                if st.button("Analyze Repeat Flows", key="analyze_repeats"):
                    with st.spinner("Analyzing repeat flows..."):
                        display_repeat_flows(df, repeat_min_premium)
                else:
                    st.info("Click 'Analyze Repeat Flows' to see contracts with multiple flows")
            
            # NEW TAB - Top Trades Summary
            with tab4:
                st.subheader("🏆 Top Trades Summary")
                st.markdown("Highest scoring trade for each ticker based on premium, frequency, and flow percentage")
                
                # Controls for top trades
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    summary_min_premium = st.number_input(
                        "Minimum Premium ($)", 
                        min_value=0, 
                        value=250000, 
                        step=50000,
                        key="summary_premium"
                    )
                    
                    min_flows = st.number_input(
                        "Minimum Sweeps", 
                        min_value=1, 
                        value=3, 
                        step=1,
                        key="min_flows"
                    )
                
                with col2:
                    st.markdown("**How it works:**")
                    st.markdown("- Finds the highest scoring contract for each ticker")
                    st.markdown("- Score based on premium size, sweep count, and daily flow %")
                    st.markdown("- Excludes index ETFs (SPY, QQQ, etc.)")
                    st.markdown("- Shows sentiment based on contract type and side codes")
                
                if st.button("Generate Top Trades Summary", key="generate_summary"):
                    with st.spinner("Calculating trade scores..."):
                        display_top_trades_summary(df, summary_min_premium, min_flows)
                else:
                    st.info("Click 'Generate Top Trades Summary' to see the highest scoring trades")
            
            # NEW TAB - Symbol Analysis
            with tab5:
                st.subheader("🎯 Symbol Analysis & Price Projection")
                st.markdown("Analyze option flows for specific symbols and get projected price path for next week")
                
                # Symbol input
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    symbols_input = st.text_input(
                        "Enter Symbol(s) (comma-separated)", 
                        placeholder="e.g., QQQ, AAPL, TSLA",
                        help="Enter one or more stock symbols separated by commas"
                    )
                    
                    min_premium_analysis = st.number_input(
                        "Minimum Premium ($)", 
                        min_value=0, 
                        value=100000, 
                        step=25000,
                        key="analysis_premium"
                    )
                
                with col2:
                    st.markdown("**Analysis includes:**")
                    st.markdown("- Overall sentiment from flows")
                    st.markdown("- Next week price projection")
                    st.markdown("- Key support/resistance levels")
                    st.markdown("- Most active strike prices")
                
                if symbols_input:
                    symbols = [s.strip() for s in symbols_input.split(',') if s.strip()]
                    
                    if st.button("Analyze Symbols", key="analyze_symbols"):
                        with st.spinner(f"Analyzing flows for {', '.join(symbols)}..."):
                            analyze_symbol_flows(df, symbols, min_premium_analysis)
                    else:
                        st.info(f"Click 'Analyze Symbols' to analyze flows for: {', '.join(symbols)}")
                else:
                    st.info("Enter one or more symbols to analyze their option flows and price projection")

if __name__ == "__main__":
    main()
