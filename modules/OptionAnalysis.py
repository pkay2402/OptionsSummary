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

def calculate_trade_score(row, ticker_daily_flow, all_premiums):
    """
    Enhanced scoring algorithm for option trades based on:
    - Premium size (30%): Normalized against market distribution
    - Flow frequency/consistency (25%): Number of sweeps and timing
    - Daily flow dominance (20%): Percentage of ticker's daily flow
    - Market impact potential (15%): Move required vs premium invested
    - Special flags and urgency (10%): Unusual activity, golden sweeps, time decay
    """
    
    # 1. Premium Score (30% weight) - Use percentile ranking
    premium_percentile = (all_premiums <= row['Total_Premium']).mean() * 100
    premium_score = min(100, premium_percentile)
    
    # 2. Flow Frequency Score (25% weight) - Reward consistency
    base_frequency_score = min(100, (row['Flow_Count'] / 20) * 100)  # 20 sweeps = 100 points
    
    # Bonus for high frequency (suggests institutional accumulation)
    if row['Flow_Count'] >= 15:
        frequency_multiplier = 1.3
    elif row['Flow_Count'] >= 10:
        frequency_multiplier = 1.15
    elif row['Flow_Count'] >= 5:
        frequency_multiplier = 1.05
    else:
        frequency_multiplier = 1.0
    
    frequency_score = min(100, base_frequency_score * frequency_multiplier)
    
    # 3. Daily Flow Dominance Score (20% weight)
    daily_flow_pct = (row['Total_Premium'] / ticker_daily_flow) * 100
    dominance_score = min(100, daily_flow_pct * 3)  # 33% of daily flow = 100 points
    
    # 4. Market Impact Potential Score (15% weight)
    # Consider move required vs premium invested (efficiency ratio)
    move_required = abs((row['Strike_Price'] - row.get('Reference_Price', row['Strike_Price'])) / 
                       row.get('Reference_Price', row['Strike_Price'])) * 100
    
    if move_required > 0:
        # Efficiency: Lower move required for higher premium = higher score
        # Sweet spot: 2-8% moves with significant premium
        if 2 <= move_required <= 8:
            impact_score = 100
        elif move_required < 2:
            impact_score = 70  # Too close to money, less upside
        elif move_required <= 15:
            impact_score = max(20, 100 - (move_required - 8) * 5)  # Penalize excessive moves
        else:
            impact_score = 20  # Very ambitious moves
    else:
        impact_score = 50  # Default if we can't calculate
    
    # 5. Special Flags and Urgency Score (10% weight)
    flags_score = 0
    
    # Premium-weighted flags (larger trades get more weight)
    premium_weight = min(2.0, row['Total_Premium'] / 1000000)  # $1M+ gets full weight
    
    if row['Has_Unusual'] == 'YES':
        flags_score += 25 * premium_weight
    if row['Has_Golden'] == 'YES':
        flags_score += 35 * premium_weight
    
    # Time decay urgency (closer expirations get bonus for immediate moves)
    try:
        days_to_exp = (row['Expiration_Date'] - pd.Timestamp.now()).days
        if days_to_exp <= 7:
            flags_score += 20  # Weekly expiration urgency
        elif days_to_exp <= 14:
            flags_score += 10  # Bi-weekly urgency
    except:
        pass
    
    flags_score = min(100, flags_score)
    
    # 6. Calculate weighted composite score
    composite_score = (
        premium_score * 0.30 +
        frequency_score * 0.25 +
        dominance_score * 0.20 +
        impact_score * 0.15 +
        flags_score * 0.10
    )
    
    # 7. Apply final adjustments
    # Boost score for perfect storm scenarios
    if (premium_percentile > 90 and row['Flow_Count'] >= 10 and daily_flow_pct > 25):
        composite_score = min(100, composite_score * 1.15)  # "Perfect storm" bonus
    
    # Slight penalty for extremely high move requirements (>20%)
    if move_required > 20:
        composite_score *= 0.9
    
    return round(composite_score, 1), daily_flow_pct, move_required

def display_top_trades_summary(df, min_premium=250000, min_flows=3):
    """
    Display top scoring trade for each ticker with enhanced analysis
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
    
    # Calculate daily flow by ticker and get all premiums for percentile calculation
    ticker_daily_flows = filtered_df.groupby('Ticker')['Premium Price'].sum()
    all_premiums = contract_groups['Total_Premium'].values
    
    # Calculate enhanced scores for each contract
    scores_data = []
    for _, row in contract_groups.iterrows():
        ticker_daily_flow = ticker_daily_flows.get(row['Ticker'], 1)  # Avoid division by zero
        score, daily_flow_pct, move_required = calculate_trade_score(row, ticker_daily_flow, all_premiums)
        
        # Enhanced sentiment analysis
        side_codes = row['Side_Codes']
        bullish_sides = sum(1 for side in side_codes if side in ['A', 'AA'])
        bearish_sides = sum(1 for side in side_codes if side in ['B', 'BB'])
        
        # Determine sentiment with confidence level
        if row['Contract_Type'] == 'CALL':
            if bullish_sides > bearish_sides:
                sentiment = "📈 BULLISH"
                confidence = "High" if bullish_sides / len(side_codes) > 0.75 else "Moderate"
            elif bearish_sides > bullish_sides:
                sentiment = "📉 BEARISH"
                confidence = "High" if bearish_sides / len(side_codes) > 0.75 else "Moderate"
            else:
                sentiment = "⚪ MIXED"
                confidence = "Low"
        else:  # PUT
            if bullish_sides > bearish_sides:
                sentiment = "📉 BEARISH"  # Put buying is bearish
                confidence = "High" if bullish_sides / len(side_codes) > 0.75 else "Moderate"
            elif bearish_sides > bullish_sides:
                sentiment = "📈 BULLISH"  # Put selling is bullish
                confidence = "High" if bearish_sides / len(side_codes) > 0.75 else "Moderate"
            else:
                sentiment = "⚪ MIXED"
                confidence = "Low"
        
        # Calculate days to expiration
        days_to_exp = (row['Expiration_Date'] - today).days
        
        scores_data.append({
            'Ticker': row['Ticker'],
            'Score': score,
            'Contract_Type': row['Contract_Type'],
            'Strike_Price': row['Strike_Price'],
            'Reference_Price': row['Reference_Price'],
            'Expiration_Date': row['Expiration_Date'],
            'Days_to_Exp': days_to_exp,
            'Flow_Count': row['Flow_Count'],
            'Total_Premium': row['Total_Premium'],
            'Daily_Flow_Pct': daily_flow_pct,
            'Move_Required': move_required,
            'Sentiment': sentiment,
            'Confidence': confidence,
            'Has_Unusual': row['Has_Unusual'],
            'Has_Golden': row['Has_Golden']
        })
    
    if not scores_data:
        st.warning("No qualifying trades found for scoring.")
        return
    
    # Convert to DataFrame and get top trade per ticker
    scores_df = pd.DataFrame(scores_data)
    top_trades = scores_df.loc[scores_df.groupby('Ticker')['Score'].idxmax()].sort_values('Score', ascending=False)
    
    # Enhanced summary header with market analysis
    st.subheader("🏆 TOP TRADES SUMMARY - AI-Enhanced Scoring Algorithm")
    
    # Market overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_tickers = len(top_trades)
        st.metric("📊 Unique Tickers", total_tickers)
    
    with col2:
        total_premium = top_trades['Total_Premium'].sum()
        st.metric("💰 Total Premium", f"${total_premium/1000000:.1f}M")
    
    with col3:
        avg_score = top_trades['Score'].mean()
        score_quality = "Excellent" if avg_score > 80 else "Good" if avg_score > 65 else "Fair"
        st.metric("⭐ Avg Quality Score", f"{avg_score:.1f}", delta=score_quality)
    
    with col4:
        bullish_count = sum(1 for sentiment in top_trades['Sentiment'] if 'BULLISH' in sentiment)
        bearish_count = sum(1 for sentiment in top_trades['Sentiment'] if 'BEARISH' in sentiment)
        market_bias = "Bullish" if bullish_count > bearish_count else "Bearish" if bearish_count > bullish_count else "Mixed"
        st.metric("📈 Market Bias", market_bias, delta=f"{bullish_count}B/{bearish_count}B")
    
    st.markdown("---")
    
    # Tiered display based on score ranges
    st.markdown("### 🌟 TIER 1: PREMIUM OPPORTUNITIES (Score 80+)")
    tier1_trades = top_trades[top_trades['Score'] >= 80]
    if not tier1_trades.empty:
        display_trade_tier(tier1_trades, "🔥")
    else:
        st.markdown("*No Tier 1 opportunities found*")
    
    st.markdown("### ⭐ TIER 2: STRONG OPPORTUNITIES (Score 65-79)")
    tier2_trades = top_trades[(top_trades['Score'] >= 65) & (top_trades['Score'] < 80)]
    if not tier2_trades.empty:
        display_trade_tier(tier2_trades, "💪")
    else:
        st.markdown("*No Tier 2 opportunities found*")
    
    st.markdown("### 📊 TIER 3: MODERATE OPPORTUNITIES (Score 50-64)")
    tier3_trades = top_trades[(top_trades['Score'] >= 50) & (top_trades['Score'] < 65)]
    if not tier3_trades.empty:
        display_trade_tier(tier3_trades, "👀")
    else:
        st.markdown("*No Tier 3 opportunities found*")
    
    # Show lower scored trades in expandable section
    lower_trades = top_trades[top_trades['Score'] < 50]
    if not lower_trades.empty:
        with st.expander(f"📋 Lower Scored Trades ({len(lower_trades)} tickers)", expanded=False):
            display_trade_tier(lower_trades, "⚠️")
    
    # Enhanced copy-to-clipboard functionality with multiple formats
    with st.expander("📋 Copy Professional Summary (Multiple Formats)", expanded=True):
        summary_formats = generate_professional_summary(top_trades)
        
        # Create tabs for different formats
        format_tabs = st.tabs(["Full Report", "Discord", "Twitter", "Quick Summary"])
        
        with format_tabs[0]:
            st.markdown("**📄 Complete comprehensive report with all tiers**")
            st.text_area("Full Report:", summary_formats['Full Report'], height=500, key="full_report")
        
        with format_tabs[1]:
            st.markdown("**💬 Optimized for Discord sharing (under 2000 chars)**")
            st.text_area("Discord Format:", summary_formats['Discord'], height=300, key="discord_format")
        
        with format_tabs[2]:
            st.markdown("**🐦 Optimized for Twitter (under 280 chars)**")
            st.text_area("Twitter Format:", summary_formats['Twitter'], height=150, key="twitter_format")
        
        with format_tabs[3]:
            st.markdown("**⚡ Quick one-liner summary**")
            st.text_area("Quick Summary:", summary_formats['Quick Summary'], height=100, key="quick_summary")
    
    # Enhanced scoring methodology
    with st.expander("🧠 Enhanced AI Scoring Methodology"):
        st.markdown("""
        **🚀 New Enhanced Scoring Algorithm (5 Components):**
        
        **1. Premium Percentile (30%)** 📊
        - Uses market distribution percentiles instead of fixed thresholds
        - Adapts to current market conditions automatically
        - 90th percentile+ = Premium opportunities
        
        **2. Flow Consistency (25%)** 🔄
        - Rewards multiple sweeps (institutional accumulation pattern)
        - Frequency multipliers: 15+ sweeps (+30%), 10+ sweeps (+15%)
        - Consistent flow indicates conviction
        
        **3. Daily Flow Dominance (20%)** 💪
        - Percentage of ticker's total daily options flow
        - 33%+ dominance = Maximum score
        - Shows relative importance vs other activity
        
        **4. Market Impact Efficiency (15%)** 🎯
        - Move required vs premium invested ratio
        - Sweet spot: 2-8% moves = Optimal efficiency
        - Penalizes excessive moves (>20%)
        
        **5. Special Signals & Urgency (10%)** ⚡
        - Premium-weighted unusual activity detection
        - Golden sweep identification with size consideration
        - Time decay urgency (weekly expirations get bonus)
        
        **🎖️ Quality Tiers:**
        - **Tier 1 (80+)**: Premium opportunities with multiple strong signals
        - **Tier 2 (65-79)**: Strong opportunities with good conviction
        - **Tier 3 (50-64)**: Moderate opportunities worth monitoring
        
        **🔥 Perfect Storm Bonus (+15%):**
        - Premium >90th percentile + 10+ sweeps + 25%+ daily flow dominance
        """)

def display_trade_tier(trades_df, emoji):
    """Helper function to display trades in a consistent format"""
    for _, trade in trades_df.iterrows():
        # Format premium
        if trade['Total_Premium'] >= 1000000:
            premium_str = f"${trade['Total_Premium']/1000000:.2f}M"
        else:
            premium_str = f"${trade['Total_Premium']/1000:.0f}K"
        
        # Special flags with enhanced display
        flags = []
        if trade['Has_Unusual'] == 'YES':
            flags.append("🔥 UNUSUAL")
        if trade['Has_Golden'] == 'YES':
            flags.append("⚡ GOLDEN")
        flags_str = f" [{' '.join(flags)}]" if flags else ""
        
        # Time urgency indicator
        if trade['Days_to_Exp'] <= 7:
            urgency = "🚨 WEEKLY"
        elif trade['Days_to_Exp'] <= 14:
            urgency = "⏰ 2-WEEK"
        else:
            urgency = f"{trade['Days_to_Exp']}d"
        
        # Confidence indicator
        conf_indicator = "🎯" if trade['Confidence'] == "High" else "📊" if trade['Confidence'] == "Moderate" else "❓"
        
        line = (f"{emoji} **${trade['Ticker']}** {trade['Sentiment']} {conf_indicator} "
                f"${trade['Strike_Price']:,.0f} {trade['Contract_Type']} "
                f"({trade['Flow_Count']} sweeps) → {premium_str} "
                f"[{trade['Move_Required']:.1f}% move] [{urgency}] "
                f"**Score: {trade['Score']:.1f}**{flags_str}")
        
        st.markdown(line)

def generate_professional_summary(top_trades):
    """Generate a comprehensive professional summary for copying to Discord/Twitter"""
    timestamp = pd.Timestamp.now().strftime('%B %d, %Y')
    
    # Create multiple format options
    formats = {}
    
    # 1. FULL COMPREHENSIVE REPORT
    full_summary = f"📊 OPTIONS FLOW INTELLIGENCE REPORT - {timestamp}\n"
    full_summary += "="*80 + "\n\n"
    
    # Market overview
    total_premium = top_trades['Total_Premium'].sum()
    avg_score = top_trades['Score'].mean()
    bullish_count = sum(1 for sentiment in top_trades['Sentiment'] if 'BULLISH' in sentiment)
    bearish_count = sum(1 for sentiment in top_trades['Sentiment'] if 'BEARISH' in sentiment)
    mixed_count = len(top_trades) - bullish_count - bearish_count
    
    full_summary += f"MARKET OVERVIEW:\n"
    full_summary += f"• Total Premium Analyzed: ${total_premium/1000000:.1f}M across {len(top_trades)} tickers\n"
    full_summary += f"• Average Quality Score: {avg_score:.1f}/100\n"
    full_summary += f"• Directional Bias: {bullish_count} Bullish, {bearish_count} Bearish, {mixed_count} Mixed\n\n"
    
    # All tiers with comprehensive data
    tier1 = top_trades[top_trades['Score'] >= 80]
    tier2 = top_trades[(top_trades['Score'] >= 65) & (top_trades['Score'] < 80)]
    tier3 = top_trades[(top_trades['Score'] >= 50) & (top_trades['Score'] < 65)]
    lower_trades = top_trades[top_trades['Score'] < 50]
    
    def format_trade_line(trade, include_score=True):
        premium_str = f"${trade['Total_Premium']/1000000:.2f}M" if trade['Total_Premium'] >= 1000000 else f"${trade['Total_Premium']/1000:.0f}K"
        flags = []
        if trade['Has_Unusual'] == 'YES':
            flags.append("UNUSUAL")
        if trade['Has_Golden'] == 'YES':
            flags.append("GOLDEN")
        flags_str = f" [{' '.join(flags)}]" if flags else ""
        
        score_str = f" [Score: {trade['Score']:.1f}]" if include_score else ""
        
        return (f"  • ${trade['Ticker']} {trade['Sentiment'].split()[1]} "
                f"${trade['Strike_Price']:,.0f} {trade['Contract_Type']} "
                f"({trade['Flow_Count']} sweeps) → {premium_str} "
                f"[{trade['Move_Required']:.1f}% move]{score_str}{flags_str}")
    
    # Add all tiers
    if not tier1.empty:
        full_summary += f"🌟 TIER 1 OPPORTUNITIES (Score 80+) - {len(tier1)} trades:\n"
        for _, trade in tier1.iterrows():
            full_summary += format_trade_line(trade) + "\n"
        full_summary += "\n"
    
    if not tier2.empty:
        full_summary += f"⭐ TIER 2 OPPORTUNITIES (Score 65-79) - {len(tier2)} trades:\n"
        for _, trade in tier2.iterrows():
            full_summary += format_trade_line(trade) + "\n"
        full_summary += "\n"
    
    if not tier3.empty:
        full_summary += f"📊 TIER 3 OPPORTUNITIES (Score 50-64) - {len(tier3)} trades:\n"
        for _, trade in tier3.iterrows():
            full_summary += format_trade_line(trade) + "\n"
        full_summary += "\n"
    
    if not lower_trades.empty:
        full_summary += f"📋 ADDITIONAL FLOWS (Score <50) - {len(lower_trades)} trades:\n"
        for _, trade in lower_trades.iterrows():
            full_summary += format_trade_line(trade) + "\n"
        full_summary += "\n"
    
    full_summary += "="*80 + "\n"
    full_summary += "Generated by Enhanced AI Options Flow Analysis\n"
    full_summary += "⚠️ Not financial advice."
    
    formats['Full Report'] = full_summary
    
    # 2. DISCORD OPTIMIZED (Under 2000 chars per message)
    discord_summary = f"📊 **OPTIONS FLOW REPORT** - {timestamp}\n\n"
    discord_summary += f"💰 **${total_premium/1000000:.1f}M** total premium | **{len(top_trades)}** tickers | Avg Score: **{avg_score:.1f}**\n"
    discord_summary += f"📈 **{bullish_count}** Bullish | 📉 **{bearish_count}** Bearish | ⚪ **{mixed_count}** Mixed\n\n"
    
    # Top opportunities only for Discord (space constraints)
    top_opportunities = top_trades.head(20)  # Top 20 trades
    
    discord_summary += "**🏆 TOP OPPORTUNITIES:**\n"
    for _, trade in top_opportunities.iterrows():
        premium_str = f"${trade['Total_Premium']/1000000:.1f}M" if trade['Total_Premium'] >= 1000000 else f"${trade['Total_Premium']/1000:.0f}K"
        
        # Use emojis for sentiment
        sentiment_emoji = "📈" if "BULLISH" in trade['Sentiment'] else "📉" if "BEARISH" in trade['Sentiment'] else "⚪"
        
        # Flags as emojis
        flags_emoji = ""
        if trade['Has_Unusual'] == 'YES':
            flags_emoji += "🔥"
        if trade['Has_Golden'] == 'YES':
            flags_emoji += "⚡"
        
        discord_summary += (f"{sentiment_emoji} **${trade['Ticker']}** "
                          f"${trade['Strike_Price']:,.0f} {trade['Contract_Type']} "
                          f"({trade['Flow_Count']}x) {premium_str} [{trade['Score']:.0f}] {flags_emoji}\n")
    
    discord_summary += f"\n⚠️ Educational only. Not financial advice."
    
    formats['Discord'] = discord_summary
    
    # 3. TWITTER OPTIMIZED (Under 280 chars)
    twitter_summary = f"📊 OPTIONS FLOW ALERT {timestamp.split(',')[0]}\n\n"
    twitter_summary += f"💰${total_premium/1000000:.1f}M across {len(top_trades)} tickers\n"
    twitter_summary += f"📈{bullish_count} Bullish 📉{bearish_count} Bearish\n\n"
    
    # Top 5 for Twitter
    twitter_summary += "🏆 TOP PLAYS:\n"
    for i, (_, trade) in enumerate(top_trades.head(5).iterrows(), 1):
        premium_str = f"${trade['Total_Premium']/1000000:.1f}M" if trade['Total_Premium'] >= 1000000 else f"${trade['Total_Premium']/1000:.0f}K"
        sentiment_emoji = "📈" if "BULLISH" in trade['Sentiment'] else "📉"
        flags_emoji = "🔥" if trade['Has_Unusual'] == 'YES' else ""
        
        twitter_summary += f"{i}. {sentiment_emoji} ${trade['Ticker']} ${trade['Strike_Price']:,.0f} {premium_str} {flags_emoji}\n"
    
    twitter_summary += "\n#OptionsFlow #Trading"
    
    formats['Twitter'] = twitter_summary
    
    # 4. QUICK SUMMARY (One-liner style)
    quick_summary = f"📊 {timestamp}: ${total_premium/1000000:.1f}M flow across {len(top_trades)} tickers | "
    quick_summary += f"{bullish_count}📈 {bearish_count}📉 | TOP: "
    
    for _, trade in top_trades.head(3).iterrows():
        sentiment_emoji = "📈" if "BULLISH" in trade['Sentiment'] else "📉"
        premium_str = f"${trade['Total_Premium']/1000000:.1f}M" if trade['Total_Premium'] >= 1000000 else f"${trade['Total_Premium']/1000:.0f}K"
        quick_summary += f"{sentiment_emoji}${trade['Ticker']} {premium_str} | "
    
    quick_summary = quick_summary.rstrip(" | ")
    
    formats['Quick Summary'] = quick_summary
    
    return formats

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
        
        # Calculate bullish vs bearish premium (unchanged)
        bullish_premium = 0
        bearish_premium = 0
        
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
        
        # Overall sentiment (unchanged)
        if bullish_pct > 60:
            overall_sentiment = "🟢 BULLISH"
            sentiment_strength = "Strong" if bullish_pct > 75 else "Moderate"
        elif bearish_pct > 60:
            overall_sentiment = "🔴 BEARISH"
            sentiment_strength = "Strong" if bearish_pct > 75 else "Moderate"
        else:
            overall_sentiment = "🟡 NEUTRAL/MIXED"
            sentiment_strength = "Balanced"
        
        # Display current status (unchanged)
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Current Price", f"${current_price:.2f}")
        
        with col2:
            st.metric("Overall Sentiment", overall_sentiment.split()[1], delta=f"{sentiment_strength}")
        
        with col3:
            st.metric("Bullish Premium", f"${bullish_premium/1000000:.1f}M", delta=f"{bullish_pct:.1f}%")
        
        with col4:
            st.metric("Bearish Premium", f"${bearish_premium/1000000:.1f}M", delta=f"{bearish_pct:.1f}%")
        
        # Weekly positioning breakdown
        st.markdown("### 📅 Weekly Options Positioning")
        
        # Group flows by expiration week
        symbol_df['Week_End'] = symbol_df['Expiration Date'].dt.to_period('W').dt.end_time.dt.date
        weekly_groups = symbol_df.groupby('Week_End')
        
        # Display weekly breakdown
        for week_end, week_data in weekly_groups:
            week_total_premium = week_data['Premium Price'].sum()
            week_str = week_end.strftime('%Y-%m-%d')
            
            with st.expander(f"📊 Week of {week_str} - ${week_total_premium/1000000:.1f}M Total Premium", expanded=True):
                
                # Separate by contract type and side
                call_buys = week_data[(week_data['Contract Type'] == 'CALL') & (week_data['Side Code'].isin(['A', 'AA']))]
                call_sells = week_data[(week_data['Contract Type'] == 'CALL') & (week_data['Side Code'].isin(['B', 'BB']))]
                put_buys = week_data[(week_data['Contract Type'] == 'PUT') & (week_data['Side Code'].isin(['A', 'AA']))]
                put_sells = week_data[(week_data['Contract Type'] == 'PUT') & (week_data['Side Code'].isin(['B', 'BB']))]
                
                # Create 4 columns for the breakdown
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown("**🟢 CALL BUYS (Bullish)**")
                    if not call_buys.empty:
                        call_buy_premium = call_buys['Premium Price'].sum()
                        call_buy_contracts = call_buys['Size'].sum()
                        st.metric("Premium", f"${call_buy_premium/1000000:.1f}M")
                        st.metric("Contracts", f"{call_buy_contracts:,}")
                        
                        # Show top strikes
                        top_call_strikes = call_buys.groupby('Strike Price')['Premium Price'].sum().sort_values(ascending=False).head(3)
                        for strike, premium in top_call_strikes.items():
                            move_pct = ((strike - current_price) / current_price) * 100
                            st.markdown(f"• ${strike:.0f} ({move_pct:+.1f}%): ${premium/1000:.0f}K")
                    else:
                        st.markdown("*No call buying*")
                
                with col2:
                    st.markdown("**🔴 CALL SELLS (Bearish)**")
                    if not call_sells.empty:
                        call_sell_premium = call_sells['Premium Price'].sum()
                        call_sell_contracts = call_sells['Size'].sum()
                        st.metric("Premium", f"${call_sell_premium/1000000:.1f}M")
                        st.metric("Contracts", f"{call_sell_contracts:,}")
                        
                        # Show top strikes
                        top_call_sell_strikes = call_sells.groupby('Strike Price')['Premium Price'].sum().sort_values(ascending=False).head(3)
                        for strike, premium in top_call_sell_strikes.items():
                            move_pct = ((strike - current_price) / current_price) * 100
                            st.markdown(f"• ${strike:.0f} ({move_pct:+.1f}%): ${premium/1000:.0f}K")
                    else:
                        st.markdown("*No call selling*")
                
                with col3:
                    st.markdown("**🔴 PUT BUYS (Bearish)**")
                    if not put_buys.empty:
                        put_buy_premium = put_buys['Premium Price'].sum()
                        put_buy_contracts = put_buys['Size'].sum()
                        st.metric("Premium", f"${put_buy_premium/1000000:.1f}M")
                        st.metric("Contracts", f"{put_buy_contracts:,}")
                        
                        # Show top strikes
                        top_put_strikes = put_buys.groupby('Strike Price')['Premium Price'].sum().sort_values(ascending=False).head(3)
                        for strike, premium in top_put_strikes.items():
                            move_pct = ((strike - current_price) / current_price) * 100
                            st.markdown(f"• ${strike:.0f} ({move_pct:+.1f}%): ${premium/1000:.0f}K")
                    else:
                        st.markdown("*No put buying*")
                
                with col4:
                    st.markdown("**🟢 PUT SELLS (Bullish)**")
                    if not put_sells.empty:
                        put_sell_premium = put_sells['Premium Price'].sum()
                        put_sell_contracts = put_sells['Size'].sum()
                        st.metric("Premium", f"${put_sell_premium/1000000:.1f}M")
                        st.metric("Contracts", f"{put_sell_contracts:,}")
                        
                        # Show top strikes
                        top_put_sell_strikes = put_sells.groupby('Strike Price')['Premium Price'].sum().sort_values(ascending=False).head(3)
                        for strike, premium in top_put_sell_strikes.items():
                            move_pct = ((strike - current_price) / current_price) * 100
                            st.markdown(f"• ${strike:.0f} ({move_pct:+.1f}%): ${premium/1000:.0f}K")
                    else:
                        st.markdown("*No put selling*")
                
                # Week summary
                bullish_flow = (call_buys['Premium Price'].sum() if not call_buys.empty else 0) + \
                              (put_sells['Premium Price'].sum() if not put_sells.empty else 0)
                bearish_flow = (call_sells['Premium Price'].sum() if not call_sells.empty else 0) + \
                              (put_buys['Premium Price'].sum() if not put_buys.empty else 0)
                
                if bullish_flow + bearish_flow > 0:
                    bullish_pct_week = (bullish_flow / (bullish_flow + bearish_flow)) * 100
                    bearish_pct_week = (bearish_flow / (bullish_flow + bearish_flow)) * 100
                    
                    if bullish_pct_week > 60:
                        week_bias = f"🟢 **BULLISH** ({bullish_pct_week:.0f}%)"
                    elif bearish_pct_week > 60:
                        week_bias = f"🔴 **BEARISH** ({bearish_pct_week:.0f}%)"
                    else:
                        week_bias = f"🟡 **MIXED** (Bull: {bullish_pct_week:.0f}%, Bear: {bearish_pct_week:.0f}%)"
                    
                    st.markdown(f"**Week Bias**: {week_bias}")
                
                st.divider()
        
        # Overall positioning summary
        with st.expander("📈 Overall Positioning Summary"):
            # Total flows by type
            total_call_buys = symbol_df[(symbol_df['Contract Type'] == 'CALL') & (symbol_df['Side Code'].isin(['A', 'AA']))]['Premium Price'].sum()
            total_call_sells = symbol_df[(symbol_df['Contract Type'] == 'CALL') & (symbol_df['Side Code'].isin(['B', 'BB']))]['Premium Price'].sum()
            total_put_buys = symbol_df[(symbol_df['Contract Type'] == 'PUT') & (symbol_df['Side Code'].isin(['A', 'AA']))]['Premium Price'].sum()
            total_put_sells = symbol_df[(symbol_df['Contract Type'] == 'PUT') & (symbol_df['Side Code'].isin(['B', 'BB']))]['Premium Price'].sum()
            
            st.markdown("**Flow Type Breakdown:**")
            st.markdown(f"- 🟢 **Call Buying**: ${total_call_buys/1000000:.1f}M ({total_call_buys/total_premium*100:.0f}%)")
            st.markdown(f"- 🔴 **Call Selling**: ${total_call_sells/1000000:.1f}M ({total_call_sells/total_premium*100:.0f}%)")
            st.markdown(f"- 🔴 **Put Buying**: ${total_put_buys/1000000:.1f}M ({total_put_buys/total_premium*100:.0f}%)")
            st.markdown(f"- 🟢 **Put Selling**: ${total_put_sells/1000000:.1f}M ({total_put_sells/total_premium*100:.0f}%)")
            
            # Net positioning
            net_bullish = total_call_buys + total_put_sells
            net_bearish = total_call_sells + total_put_buys
            st.markdown(f"\n**Net Positioning:**")
            st.markdown(f"- 🟢 **Net Bullish Flows**: ${net_bullish/1000000:.1f}M")
            st.markdown(f"- 🔴 **Net Bearish Flows**: ${net_bearish/1000000:.1f}M")
        
        # Show detailed flows (unchanged)
        with st.expander(f"📊 Detailed Flows for {symbol}"):
            display_df = symbol_df.sort_values('Premium Price', ascending=False)
            
            for _, row in display_df.head(20).iterrows():
                move_pct = abs((row['Strike Price'] - row['Reference Price']) / 
                              row['Reference Price'] * 100)
                
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
                        value=2, 
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
