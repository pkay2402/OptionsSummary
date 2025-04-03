import streamlit as st
import pandas as pd
from datetime import datetime
import requests
import os

# Function to load and process the CSV file
def load_csv(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        
        # Map columns by position if needed
        if not all(col in df.columns for col in ['Ticker', 'Expiration Date', 'Contract Type']):
            positional_mapping = {
                'Trade ID': 0,
                'Trade Time': 1,
                'Ticker': 2,
                'Expiration Date': 3,
                'Days Until Expiration': 4,
                'Strike Price': 5,
                'Contract Type': 6,
                'Reference Price': 7,
                'Size': 8,
                'Option Price': 9,
                'Ask Price': 10,
                'Bid Price': 11,
                'Premium Price': 12,
                'Trade Type': 13,
                'Consolidation Type': 14,
                'Is Unusual': 15,
                'Is Golden Sweep': 16,
                'Is Opening Position': 17,
                'Money Type': 18,
                'Side Code': 19  # Added Side Code to mapping
            }
            
            new_columns = df.columns.tolist()
            renamed_df = df.copy()
            renamed_df.columns = [positional_mapping.get(i, col) for i, col in enumerate(new_columns)]
            df = renamed_df

        # Convert 'Expiration Date' to datetime for filtering
        df['Expiration Date'] = pd.to_datetime(df['Expiration Date'])
        
        # Ensure numeric columns are properly converted
        numeric_columns = ['Days Until Expiration', 'Strike Price', 'Reference Price', 
                         'Size', 'Option Price', 'Premium Price']
        
        for col in numeric_columns:
            if col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].astype(str).str.replace('$', '').str.replace(',', '')
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Ensure categorical columns are strings and handle Side Code
        categorical_columns = ['Trade Type', 'Ticker', 'Contract Type', 'Is Unusual', 
                             'Is Golden Sweep', 'Is Opening Position', 'Money Type', 'Side Code']
        for col in categorical_columns:
            if col in df.columns:
                df[col] = df[col].fillna('N/A').astype(str).str.strip()
                if col == 'Ticker':
                    df[col] = df[col].str.upper()
        
        # Filter for OUT_THE_MONEY flows only and Is Opening Position == Yes
        df = df[(df['Money Type'] == 'OUT_THE_MONEY') & (df['Is Opening Position'] == 'Yes')]
        
        return df
    except Exception as e:
        st.error(f"Failed to load CSV: {e}")
        return None

# Function to split content for Discord
def split_message(content: str, max_length: int = 2000) -> list:
    """
    Split content into parts under max_length characters while preserving line breaks.
    """
    if len(content) <= max_length:
        return [content]
    
    messages = []
    current_part = ""
    
    for line in content.split('\n'):
        if len(current_part) + len(line) + 1 > max_length:
            if current_part:
                messages.append(current_part.strip())
            current_part = line
        else:
            current_part += f"\n{line}" if current_part else line
            
    if current_part:
        messages.append(current_part.strip())
        
    # Handle case where a single line is too long
    final_messages = []
    for msg in messages:
        if len(msg) > max_length:
            for i in range(0, len(msg), max_length):
                final_messages.append(msg[i:i + max_length])
        else:
            final_messages.append(msg)
            
    return final_messages

# Function to send content to Discord with splitting
def send_to_discord(content, webhook_url):
    DISCORD_CHAR_LIMIT = 2000
    try:
        parts = split_message(content, DISCORD_CHAR_LIMIT)
        
        for i, part in enumerate(parts, 1):
            data = {"content": f"Part {i}/{len(parts)}:\n{part}"}
            response = requests.post(webhook_url, json=data)
            if response.status_code != 204:
                return f"Failed to send part {i} to Discord: {response.status_code} {response.text}"
        
        return f"Newsletter sent successfully to Discord in {len(parts)} parts."
    except Exception as e:
        return f"Error sending to Discord: {e}"

def generate_newsletter(df, top_n_aggressive_flows, premium_price, side_codes, tickers, sort_by):
    if df is None:
        return "No data available for newsletter generation."
    
    if df.empty:
        return "No OUT_THE_MONEY flows found in the dataset."
    
    current_date = pd.to_datetime("today")
    current_date_str = current_date.strftime("%b %d, %Y")
    
    exclude_tickers = ['SPX', 'SPY', 'IWM', 'QQQ']
    
    newsletter = f"📈 OUT-THE-MONEY OPTIONS FLOW SUMMARY - {current_date_str} 📈\n\n"
    
    # Section 1: Market Update (SPY & QQQ)
    newsletter += "=== MARKET UPDATE (OTM FLOWS) ===\n"
    market_tickers = ['SPY', 'QQQ', 'DIA', 'IWM','VXX','SMH']
    market_df = df[df['Ticker'].isin(market_tickers)].copy()
    
    if market_df.empty:
        newsletter += "No OTM market index flows (SPY/QQQ) detected today.\n\n"
    else:
        max_expiry = current_date + pd.Timedelta(days=7)
        market_df = market_df[(market_df['Expiration Date'] <= max_expiry) & 
                            (market_df['Expiration Date'] > current_date)]
        
        if market_df.empty:
            newsletter += "No near-term (next week) OTM flows for SPY/QQQ detected after excluding today.\n"
        else:
            call_premium = market_df[market_df['Contract Type'] == 'CALL']['Premium Price'].sum()
            put_premium = market_df[market_df['Contract Type'] == 'PUT']['Premium Price'].sum()
            total_volume = market_df['Size'].sum()
            pc_ratio = put_premium / call_premium if call_premium > 0 else float('inf')
            
            sentiment = "🟢" if pc_ratio < 0.7 else "🔴" if pc_ratio > 1.5 else "NEUTRAL"
            newsletter += f"Market Sentiment: {sentiment}\n"
            newsletter += f"Total Premium: ${call_premium + put_premium:,.2f}\n"
            newsletter += f"Total Contracts: {total_volume:,}\n"
            newsletter += f"Put/Call Premium Ratio: {pc_ratio:.2f}\n\n"
            
            key_flows = market_df.nlargest(7, 'Premium Price')
            newsletter += "Key Market Flows:\n"
            for idx, flow in key_flows.iterrows():
                move_pct = abs((flow['Strike Price'] - flow['Reference Price']) / flow['Reference Price'] * 100)
                side = flow['Side Code'] if pd.notna(flow['Side Code']) and flow['Side Code'] != 'N/A' else "N/A"
                sentiment = ("🟢" if flow['Contract Type'] == 'CALL' and flow['Side Code'] in ['A', 'AA'] else
                            "🔴" if flow['Contract Type'] == 'CALL' and flow['Side Code'] in ['B', 'BB'] else
                            "🔴" if flow['Contract Type'] == 'PUT' and flow['Side Code'] in ['A', 'AA'] else
                            "🟢" if flow['Contract Type'] == 'PUT' and flow['Side Code'] in ['B', 'BB'] else "N/A")
                flags = []
                if flow['Is Unusual'] == 'Yes':
                    flags.append("UNUSUAL")
                if flow['Is Golden Sweep'] == 'Yes':
                    flags.append("GOLDEN SWEEP")
                if flow['Is Opening Position'] == 'Yes':
                    flags.append("OPENING")
                flags_str = f" [{' '.join(flags)}]" if flags else ""
                newsletter += (f"• {flow['Ticker']} {flow['Contract Type']} ${flow['Strike Price']:,.2f} "
                             f"exp {flow['Expiration Date'].date()} - ${flow['Premium Price']:,.2f} "
                             f"({flow['Size']} contracts, {move_pct:.1f}% move, {sentiment}, {side}){flags_str}\n")
        newsletter += "\n"
    
    # Section 2: Aggressive Flows
    newsletter += "=== OTM FLOWS ===\n"
    aggressive_df = df[
        (df['Expiration Date'] > current_date) &
        (df['Premium Price'] >= premium_price) &
        (df['Side Code'].isin(side_codes)) &
        (df['Ticker'].isin(tickers)) &
        (~df['Ticker'].isin(exclude_tickers))
    ]
    
    if aggressive_df.empty:
        newsletter += "No aggressive OTM flows detected after applying the filters.\n\n"
    else:
        if sort_by == "Premium Price":
            aggressive_df = aggressive_df.sort_values(by=["Premium Price", "Ticker"], ascending=[False, True])
            newsletter += "Top Aggressive Flows (Sorted by Premium Price):\n"
            for idx, flow in aggressive_df.head(top_n_aggressive_flows).iterrows():
                move_pct = abs((flow['Strike Price'] - flow['Reference Price']) / flow['Reference Price'] * 100)
                side = flow['Side Code'] if pd.notna(flow['Side Code']) and flow['Side Code'] != 'N/A' else "N/A"
                sentiment = ("🟢" if flow['Contract Type'] == 'CALL' and flow['Side Code'] in ['A', 'AA'] else
                            "🔴" if flow['Contract Type'] == 'CALL' and flow['Side Code'] in ['B', 'BB'] else
                            "🔴" if flow['Contract Type'] == 'PUT' and flow['Side Code'] in ['A', 'AA'] else
                            "🟢" if flow['Contract Type'] == 'PUT' and flow['Side Code'] in ['B', 'BB'] else "N/A")
                flags = []
                if flow['Is Unusual'] == 'Yes':
                    flags.append("UNUSUAL")
                if flow['Is Golden Sweep'] == 'Yes':
                    flags.append("GOLDEN SWEEP")
                if flow['Is Opening Position'] == 'Yes':
                    flags.append("OPENING")
                flags_str = f" [{' '.join(flags)}]" if flags else ""
                newsletter += (f"• {flow['Ticker']} {flow['Contract Type']} ${flow['Strike Price']:,.2f} "
                             f"exp {flow['Expiration Date'].date()} - ${flow['Premium Price']:,.2f} "
                             f"({flow['Size']} contracts, {move_pct:.1f}% move, {sentiment}, {side}){flags_str}\n")
        else:
            aggressive_df = aggressive_df.sort_values(['Ticker', 'Premium Price'], ascending=[True, False])
            newsletter += "Top Aggressive Flows (Grouped by Ticker):\n"
            grouped_aggressive_df = aggressive_df.groupby('Ticker')
            for ticker, group in grouped_aggressive_df:
                newsletter += f"\nTicker: {ticker}\n"
                for idx, flow in group.head(top_n_aggressive_flows).iterrows():
                    move_pct = abs((flow['Strike Price'] - flow['Reference Price']) / flow['Reference Price'] * 100)
                    side = flow['Side Code'] if pd.notna(flow['Side Code']) and flow['Side Code'] != 'N/A' else "N/A"
                    sentiment = ("🟢" if flow['Contract Type'] == 'CALL' and flow['Side Code'] in ['A', 'AA'] else
                                "🔴" if flow['Contract Type'] == 'CALL' and flow['Side Code'] in ['B', 'BB'] else
                                "🔴" if flow['Contract Type'] == 'PUT' and flow['Side Code'] in ['A', 'AA'] else
                                "🟢" if flow['Contract Type'] == 'PUT' and flow['Side Code'] in ['B', 'BB'] else "N/A")
                    flags = []
                    if flow['Is Unusual'] == 'Yes':
                        flags.append("UNUSUAL")
                    if flow['Is Golden Sweep'] == 'Yes':
                        flags.append("GOLDEN SWEEP")
                    if flow['Is Opening Position'] == 'Yes':
                        flags.append("OPENING")
                    flags_str = f" [{' '.join(flags)}]" if flags else ""
                    newsletter += (f"• {flow['Ticker']} {flow['Contract Type']} ${flow['Strike Price']:,.2f} "
                                 f"exp {flow['Expiration Date'].date()} - ${flow['Premium Price']:,.2f} "
                                 f"({flow['Size']} contracts, {move_pct:.1f}% move, {sentiment}, {side}){flags_str}\n")
        newsletter += "\n"
    
    newsletter += "Only for education purpose!"
    return newsletter

def display_symbol_flows(df, symbol):
    """Display flows for a specific symbol with opening positions, premium > 250000, and expiration date > today's date"""
    if df is None or df.empty:
        st.warning("No data available to display symbol flows.")
        return
    
    # Filter for specific symbol, opening positions, premium > 250000, and expiration date > today's date
    today = pd.to_datetime("today")
    symbol_df = df[
        (df['Ticker'] == symbol.upper()) &
        (df['Is Opening Position'] == 'Yes') &
        (df['Premium Price'] > 250000) &
        (df['Expiration Date'] > today)
    ].copy()
    
    if symbol_df.empty:
        st.warning(f"No opening position flows found for {symbol} with premium > $250,000 and expiration date > today")
        return
    
    # Sort by Expiration Date (ascending) and Premium Price (descending)
    symbol_df = symbol_df.sort_values(by=['Expiration Date', 'Premium Price'], ascending=[True, False])
    
    st.subheader(f"Opening Position Flows for {symbol} (Premium > $250,000, Expiration Date > Today)")
    
    # Format the display
    for idx, flow in symbol_df.iterrows():
        move_pct = abs((flow['Strike Price'] - flow['Reference Price']) / flow['Reference Price'] * 100)
        side = flow['Side Code'] if pd.notna(flow['Side Code']) and flow['Side Code'] != 'N/A' else "N/A"
        sentiment = ("🟢" if flow['Contract Type'] == 'CALL' and flow['Side Code'] in ['A', 'AA'] else
                    "🔴" if flow['Contract Type'] == 'CALL' and flow['Side Code'] in ['B', 'BB'] else
                    "🔴" if flow['Contract Type'] == 'PUT' and flow['Side Code'] in ['A', 'AA'] else
                    "🟢" if flow['Contract Type'] == 'PUT' and flow['Side Code'] in ['B', 'BB'] else "N/A")
        flags = []
        if flow['Is Unusual'] == 'Yes':
            flags.append("UNUSUAL")
        if flow['Is Golden Sweep'] == 'Yes':
            flags.append("GOLDEN SWEEP")
        flags_str = f" [{' '.join(flags)}]" if flags else ""
        
        flow_str = (
            f"• {flow['Contract Type']} | "
            f"Strike: ${flow['Strike Price']:,.2f} | "
            f"Exp: {flow['Expiration Date'].date()} | "
            f"Premium: ${flow['Premium Price']:,.2f} | "
            f"Contracts: {flow['Size']} | "
            f"Move: {move_pct:.1f}% | "
            f"Sentiment: {sentiment} | "
            f"Side: {side} {flags_str}"
        )
        
        st.markdown(flow_str)
    
    # Display raw data table as an option
    with st.expander("View Raw Data"):
        st.dataframe(symbol_df)

def main():
    st.set_page_config(
        page_title="Smart Options Flow Analyzer",
        page_icon="📈",
        layout="wide"
    )
    
    default_webhook = os.environ.get("DISCORD_WEBHOOK_URL", "https://discord.com/api/webhooks/1341974407102595082/HTKke4FEZIQe6Xd9AUv2IgVDJp0yx89Uhosv_iM-7BZBTn2jk2T-dP_TFbX2PgMuF75D")
    
    st.title("🔍 Smart Options Flow Analyzer")
    st.markdown("Generate a newsletter or view specific symbol flows for OUT-THE-MONEY options")
     
    uploaded_file = st.file_uploader("Upload your options flow CSV file", type=["csv"])
    
    if uploaded_file is not None:
        with st.spinner("Processing options data..."):
            df = load_csv(uploaded_file)
            
            if df is not None:
                # Tabs for different views
                tab1, tab2 = st.tabs(["Newsletter Generator", "Individual Symbol Flows"])
                
                # Tab 1: Newsletter Generator (original functionality)
                with tab1:
                    st.subheader("Options Flow Newsletter")
                    st.markdown("Generate a newsletter summarizing today's OUT-THE-MONEY flows for your Discord group")
                    
                    top_n_aggressive_flows = st.number_input("Number of Flows to Display", min_value=1, max_value=200, value=100)
                    premium_price = st.number_input("Minimum Premium Price", min_value=0, value=150000)
                    side_codes = st.multiselect("Side Codes", options=['A', 'AA', 'B', 'BB'], default=['AA', 'A'])
                    with st.expander("Select Tickers", expanded=False):
                        tickers = st.multiselect("Tickers", options=df['Ticker'].unique().tolist(), default=df['Ticker'].unique().tolist())
                    sort_by = st.selectbox("Sort By", options=["Ticker", "Premium Price"])
                    
                    with st.expander("Discord Integration"):
                        webhook_url = st.text_input(
                            "Discord Webhook URL (Newsletter)",
                            value=default_webhook,
                            type="password",
                            help="Enter your Discord webhook URL to send the newsletter"
                        )
                        send_to_discord_enabled = st.checkbox("Send newsletter to Discord", value=False)
                    
                    if st.button("Generate Newsletter", type="primary"):
                        with st.spinner("Generating newsletter..."):
                            newsletter_content = generate_newsletter(df, top_n_aggressive_flows, premium_price, side_codes, tickers, sort_by)
                            st.markdown(f"```\n{newsletter_content}\n```")
                            
                            if send_to_discord_enabled:
                                with st.spinner("Sending newsletter to Discord..."):
                                    discord_result = send_to_discord(newsletter_content, webhook_url)
                                    st.info(discord_result)
                
                # Tab 2: Individual Symbol Flows
                with tab2:
                    st.subheader("View Individual Symbol Flows")
                    st.markdown("View opening position flows with premium > $250,000 for a specific symbol")
                    
                    symbol_input = st.text_input("Enter Symbol (e.g., AAPL)", "").upper()
                    if symbol_input:
                        display_symbol_flows(df, symbol_input)

if __name__ == "__main__":
    main()
