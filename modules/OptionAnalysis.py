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
                'Money Type': 18
            }
            
            # Create a new DataFrame with renamed columns
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
                # If column is object type, clean currency symbols and commas
                if df[col].dtype == 'object':
                    df[col] = df[col].astype(str).str.replace('$', '').str.replace(',', '')
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Ensure categorical columns are strings
        for col in ['Trade Type', 'Ticker', 'Contract Type', 'Is Unusual', 'Is Golden Sweep', 'Is Opening Position', 'Money Type']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
                if col == 'Ticker':
                    df[col] = df[col].str.upper()
        
        # Filter for OUT_THE_MONEY flows only
        df = df[df['Money Type'] == 'OUT_THE_MONEY']
        
        return df
    except Exception as e:
        st.error(f"Failed to load CSV: {e}")
        return None

# Function to send content to Discord with splitting
def send_to_discord(content, webhook_url):
    DISCORD_CHAR_LIMIT = 2000
    try:
        if len(content) <= DISCORD_CHAR_LIMIT:
            data = {"content": content}
            response = requests.post(webhook_url, json=data)
            if response.status_code == 204:
                return "Newsletter sent successfully to Discord."
            else:
                return f"Failed to send newsletter to Discord: {response.status_code} {response.text}"
        else:
            # Split content into parts
            parts = []
            current_part = ""
            lines = content.split("\n")
            
            for line in lines:
                if len(current_part) + len(line) + 1 > DISCORD_CHAR_LIMIT:
                    parts.append(current_part.strip())
                    current_part = line + "\n"
                else:
                    current_part += line + "\n"
            
            if current_part.strip():
                parts.append(current_part.strip())
            
            # Send each part
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
    
    # Define exclude tickers
    exclude_tickers = ['SPX', 'SPY', 'IWM', 'QQQ']
    
    # Newsletter Header
    newsletter = f"📈 OUT-THE-MONEY OPTIONS FLOW SUMMARY - {current_date_str} 📈\n\n"
    
    # Section 1: Market Update (SPY & QQQ)
    newsletter += "=== MARKET UPDATE (OTM FLOWS) ===\n"
    market_tickers = ['SPY', 'QQQ']
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
            
            key_flows = market_df.nlargest(3, 'Premium Price')
            newsletter += "Key Market Flows:\n"
            for idx, flow in key_flows.iterrows():
                move_pct = abs((flow['Strike Price'] - flow['Reference Price']) / flow['Reference Price'] * 100)
                side = ("AA" if flow.get('Side Code') == 'AA' else 
                        "BB" if flow.get('Side Code') == 'BB' else 
                        "A" if flow.get('Side Code') == 'A' else 
                        "B" if flow.get('Side Code') == 'B' else "N/A")
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
    
    # Section 2: Aggressive Flows (Premium $100K+, Side = AA or BB)
    newsletter += "=== OTM FLOWS ===\n"
    if 'Side Code' in df.columns:
        aggressive_df = df[
            (df['Expiration Date'] > current_date) &
            (df['Premium Price'] >= premium_price) &
            (df['Side Code'].isin(side_codes)) &
            (df['Ticker'].isin(tickers)) &
            (~df['Ticker'].isin(exclude_tickers))
        ]
        
        if sort_by == "Premium Price":
            aggressive_df = aggressive_df.sort_values('Premium Price', ascending=False)
        else:
            aggressive_df = aggressive_df.sort_values(['Ticker', 'Premium Price'], ascending=[True, False])
    else:
        aggressive_df = pd.DataFrame()  # Empty DataFrame if 'Side Code' column is not present
    
    if aggressive_df.empty:
        newsletter += "No aggressive OTM flows detected after applying the filters.\n\n"
    else:
        if sort_by == "Premium Price":
            newsletter += "Top Aggressive Flows (Sorted by Premium Price):\n"
            for idx, flow in aggressive_df.head(top_n_aggressive_flows).iterrows():
                move_pct = abs((flow['Strike Price'] - flow['Reference Price']) / flow['Reference Price'] * 100)
                side = "AA" if flow['Side Code'] == 'AA' else "BB" if flow['Side Code'] == 'BB' else "N/A"
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
            newsletter += "Top Aggressive Flows (Grouped by Ticker):\n"
            grouped_aggressive_df = aggressive_df.groupby('Ticker')
            for ticker, group in grouped_aggressive_df:
                newsletter += f"\nTicker: {ticker}\n"
                for idx, flow in group.head(top_n_aggressive_flows).iterrows():
                    move_pct = abs((flow['Strike Price'] - flow['Reference Price']) / flow['Reference Price'] * 100)
                    side = "AA" if flow['Side Code'] == 'AA' else "BB" if flow['Side Code'] == 'BB' else "N/A"
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
    
    # Footer
    #newsletter += "Generated by Smart Options Flow Analyzer\n"
    newsletter += "Only for education purpose!"
    
    return newsletter

# Main function
def main():
    st.set_page_config(
        page_title="Smart Options Flow Analyzer",
        page_icon="📈",
        layout="wide"
    )
    
    default_webhook = os.environ.get("DISCORD_WEBHOOK_URL", "https://discord.com/api/webhooks/1341974407102595082/HTKke4FEZIQe6Xd9AUv2IgVDJp0yx89Uhosv_iM-7BZBTn2jk2T-dP_TFbX2PgMuF75D")
    
    st.title("🔍 Smart Options Flow Analyzer")
    st.markdown("Generate a newsletter summarizing today's OUT-THE-MONEY options flows")
     
    uploaded_file = st.file_uploader("Upload your options flow CSV file", type=["csv"])
    
    if uploaded_file is not None:
        with st.spinner("Processing options data..."):
            df = load_csv(uploaded_file)
            
            if df is not None:
                st.subheader("Options Flow Newsletter")
                st.markdown("Generate a newsletter summarizing today's OUT-THE-MONEY flows for your Discord group")
                
                top_n_aggressive_flows = st.number_input("Number of Flows to Display", min_value=1, max_value=100, value=50)
                premium_price = st.number_input("Minimum Premium Price", min_value=0, value=100000)
                side_codes = st.multiselect("Side Codes", options=['A', 'AA', 'B', 'BB'], default=['AA', 'A'])
                with st.expander("Select Tickers", expanded=False):
                    tickers = st.multiselect("Tickers", options=df['Ticker'].unique().tolist(), default=df['Ticker'].unique().tolist())
                sort_by = st.selectbox("Sort By", options=["Premium Price", "Ticker"])
                
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

if __name__ == "__main__":
    main()
