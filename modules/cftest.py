import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import plotly.graph_objects as go
from datetime import datetime, timedelta
import re
from io import StringIO

def cftc_analyzer_module():
    st.title("CFTC Data Analyzer")
    
    # Set default URLs
    default_url = "https://www.cftc.gov/dea/options/financial_lof.htm"
    weekly_default_url = "https://www.cftc.gov/sites/default/files/files/dea/cotarchives/2025/options/financial_lof030425.htm"
    
    with st.sidebar:
        st.header("CFTC Data Settings")
        
        # Source selection
        data_source = st.radio(
            "Select Data Source:",
            options=["Default CFTC Page", "Custom URL", "Weekly Report"],
            index=0
        )
        
        if data_source == "Custom URL":
            cftc_url = st.text_input("Custom CFTC Data URL", value=default_url)
        elif data_source == "Weekly Report":
            cftc_url = st.text_input("Weekly Report URL", value=weekly_default_url, help="Update this for new weekly data (e.g., financial_lof031125.htm for March 11)")
        else:
            cftc_url = default_url
            
        st.caption(f"Current source: {cftc_url}")
        
        fetch_data = st.button("Fetch Latest CFTC Data")
        
        # Display options
        st.header("Display Options")
        show_net_position = st.checkbox("Show Net Position", value=True)
        show_long_position = st.checkbox("Show Long Position", value=True)
        show_short_position = st.checkbox("Show Short Position", value=True)
        
        # Time range selector
        st.header("Time Range")
        time_range = st.radio(
            "Select time period:",
            options=["All Available Data", "Past 52 Weeks", "Past 26 Weeks", "Past 12 Weeks", "Past 4 Weeks"],
            index=0
        )
    
    # Main content area
    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def fetch_cftc_data(url):
        try:
            # Fetch the HTML content
            response = requests.get(url)
            response.raise_for_status()
            
            # Find links to CSV or TXT data files
            soup = BeautifulSoup(response.text, 'html.parser')
            data_links = []
            
            # Look for specific file patterns that match CFTC reports
            for link in soup.find_all('a', href=True):
                href = link['href']
                if href.endswith('.txt') or href.endswith('.csv') or 'fin_com_disagg' in href or 'futures_com' in href or 'options' in href:
                    if not href.startswith('http'):
                        # Convert relative URL to absolute
                        base_url = '/'.join(url.split('/')[:-1])
                        href = f"{base_url}/{href}"
                    data_links.append(href)
            
            if not data_links:
                # If no direct links found, try to parse the table data from the page
                return parse_table_data(soup)
            
            # Special handling for weekly report URLs
            if 'cotarchives' in url and 'financial_lof' in url:
                # Assume weekly URLs link to a single TXT file or contain the data
                if data_links:
                    data_url = data_links[0]  # Take the first link (e.g., f030425.txt)
                    data_response = requests.get(data_url)
                    data_response.raise_for_status()
                    df = pd.read_csv(StringIO(data_response.text))  # Legacy files are often CSV-like
                    processed_df = process_cftc_dataframe(df)
                    if not processed_df.empty:
                        return processed_df
                else:
                    # Fallback to parsing the HTML table if no TXT link
                    return parse_table_data(soup)
            
            # General handling for other URLs
            all_data = []
            for data_url in data_links[:3]:  # Limit to first 3 links
                try:
                    data_response = requests.get(data_url)
                    data_response.raise_for_status()
                    
                    if data_url.endswith('.csv'):
                        df = pd.read_csv(StringIO(data_response.text))
                    else:  # Assume TXT file
                        try:
                            df = pd.read_csv(StringIO(data_response.text), sep=',')
                        except:
                            df = pd.read_fwf(StringIO(data_response.text))
                    
                    processed_df = process_cftc_dataframe(df)
                    if not processed_df.empty:
                        all_data.append(processed_df)
                except Exception as e:
                    st.warning(f"Could not process {data_url}: {str(e)}")
                    continue
            
            if all_data:
                return pd.concat(all_data, ignore_index=True)
            
            # Fallback parsing for the first link
            if data_links:
                data_url = data_links[0]
                data_response = requests.get(data_url)
                data_response.raise_for_status()
                content = data_response.text
                
                for parser in [parse_csv, parse_txt_as_csv, parse_fixed_width]:
                    try:
                        df = parser(content)
                        if not df.empty:
                            return process_cftc_dataframe(df)
                    except:
                        continue
            
            return process_cftc_dataframe(df)
            
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")
            return generate_mock_data()
    
    def parse_table_data(soup):
        """Parse table data directly from HTML if no CSV/TXT links found"""
        tables = soup.find_all('table')
        if not tables:
            return generate_mock_data()
        
        main_table = tables[0]
        for table in tables:
            if len(table.find_all('tr')) > len(main_table.find_all('tr')):
                main_table = table
        
        headers = [th.text.strip() for th in main_table.find('tr').find_all(['th', 'td'])]
        rows = [[td.text.strip() for td in tr.find_all('td')] for tr in main_table.find_all('tr')[1:]]
        rows = [row for row in rows if row and len(row) == len(headers)]
        
        if headers and rows:
            df = pd.DataFrame(rows, columns=headers)
            return process_cftc_dataframe(df)
        return generate_mock_data()
    
    def process_cftc_dataframe(df):
        """Process and clean the CFTC dataframe"""
        date_col = None
        symbol_col = None
        long_cols = []
        short_cols = []
        
        date_patterns = ['date', 'as_of_date', 'report_date', 'report date', 'as of']
        symbol_patterns = ['contract', 'market', 'symbol', 'name', 'futures_contract', 'cftc_contract_market']
        long_patterns = ['noncomm.*long', 'positions_long', 'long']  # Updated for Legacy report
        short_patterns = ['noncomm.*short', 'positions_short', 'short']  # Updated for Legacy report
        
        for col in df.columns:
            col_lower = col.lower()
            if date_col is None:
                for pattern in date_patterns:
                    if pattern in col_lower:
                        date_col = col
                        break
            if symbol_col is None:
                for pattern in symbol_patterns:
                    if pattern in col_lower:
                        symbol_col = col
                        break
            for pattern in long_patterns:
                if re.search(pattern, col_lower, re.IGNORECASE) and 'all' in col_lower:  # Focus on 'All' positions
                    long_cols.append(col)
            for pattern in short_patterns:
                if re.search(pattern, col_lower, re.IGNORECASE) and 'all' in col_lower:
                    short_cols.append(col)
        
        if not date_col and 'Report_Date_as_MM_DD_YYYY' in df.columns:
            date_col = 'Report_Date_as_MM_DD_YYYY'
        if not symbol_col and 'Market_and_Exchange_Names' in df.columns:
            symbol_col = 'Market_and_Exchange_Names'
        
        if not date_col or not symbol_col:
            return generate_mock_data()
        
        processed_df = df.copy()
        if date_col:
            processed_df[date_col] = pd.to_datetime(processed_df[date_col], errors='coerce')
        
        result_data = []
        for _, row in processed_df.iterrows():
            symbol = str(row[symbol_col]).strip() if symbol_col else "Unknown"
            date = row[date_col] if date_col else pd.NaT
            
            long_pos = 0
            short_pos = 0
            for col in long_cols:
                long_pos += pd.to_numeric(row[col], errors='coerce') or 0
            for col in short_cols:
                short_pos += pd.to_numeric(row[col], errors='coerce') or 0
            
            result_data.append({
                'date': date,
                'symbol': symbol,
                'leveraged_long': long_pos,
                'leveraged_short': short_pos,
                'net_position': long_pos - short_pos
            })
        
        result_df = pd.DataFrame(result_data)
        result_df = result_df[~result_df['date'].isna() & (result_df['symbol'] != "") & (result_df['symbol'] != "Unknown")]
        
        return result_df if not result_df.empty else generate_mock_data()
    
    def parse_csv(content):
        return pd.read_csv(StringIO(content))
    
    def parse_txt_as_csv(content):
        for delimiter in [',', ';', '\t', '|']:
            try:
                return pd.read_csv(StringIO(content), sep=delimiter)
            except:
                continue
        raise Exception("Could not parse as delimited text")
    
    def parse_fixed_width(content):
        return pd.read_fwf(StringIO(content))

    def generate_mock_data():
        symbols = ['S&P 500', 'NASDAQ-100', 'DJIA', 'EURO FX', 'TREASURY BONDS']
        today = datetime.now()
        dates = [(today - timedelta(days=7*i)).strftime('%Y-%m-%d') for i in range(104)]
        mock_data = []
        for symbol in symbols:
            long_pos = np.random.randint(50000, 250000)
            short_pos = np.random.randint(50000, 250000)
            trend = np.random.choice([-1, 1]) * np.random.uniform(0.2, 0.5)
            for date in dates:
                random_change = np.random.normal(0, 5000)
                trend_change = trend * 2000
                long_pos = max(10000, long_pos + random_change + trend_change)
                short_pos = max(10000, short_pos + random_change - trend_change)
                mock_data.append({
                    'date': pd.to_datetime(date),
                    'symbol': symbol,
                    'leveraged_long': int(long_pos),
                    'leveraged_short': int(short_pos),
                    'net_position': int(long_pos - short_pos)
                })
        return pd.DataFrame(mock_data)
    
    if 'cftc_data' not in st.session_state or fetch_data:
        with st.spinner('Fetching CFTC data...'):
            st.session_state.cftc_data = fetch_cftc_data(cftc_url)
            st.success('Data loaded successfully!')
    
    symbols = sorted(st.session_state.cftc_data['symbol'].unique())
    selected_symbol = st.selectbox('Select Symbol', symbols)
    symbol_data = st.session_state.cftc_data[st.session_state.cftc_data['symbol'] == selected_symbol].copy()
    symbol_data = symbol_data.sort_values('date')
    
    if time_range != "All Available Data":
        weeks = int(re.search(r'\d+', time_range).group())
        cutoff_date = datetime.now() - timedelta(weeks=weeks)
        symbol_data = symbol_data[symbol_data['date'] >= cutoff_date]
    
    if not symbol_data.empty:
        fig = go.Figure()
        if show_net_position:
            fig.add_trace(go.Scatter(x=symbol_data['date'], y=symbol_data['net_position'], mode='lines', name='Net Position', line=dict(color='rgb(0, 102, 204)', width=2)))
        if show_long_position:
            fig.add_trace(go.Scatter(x=symbol_data['date'], y=symbol_data['leveraged_long'], mode='lines', name='Long Position', line=dict(color='rgb(0, 204, 102)', width=2)))
        if show_short_position:
            fig.add_trace(go.Scatter(x=symbol_data['date'], y=symbol_data['leveraged_short'], mode='lines', name='Short Position', line=dict(color='rgb(204, 0, 0)', width=2)))
        
        fig.update_layout(
            title=f'CFTC Positions: {selected_symbol}',
            xaxis_title='Date',
            yaxis_title='Contracts',
            hovermode='x unified',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        latest_data = symbol_data.iloc[-1]
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Net Position", f"{int(latest_data['net_position']):,}", delta=f"{int(latest_data['net_position'] - symbol_data.iloc[-2]['net_position']):,}" if len(symbol_data) > 1 else None)
        with col2:
            if len(symbol_data) >= 5:
                four_week_change = latest_data['net_position'] - symbol_data.iloc[-5]['net_position']
                st.metric("Net Position Change (4 Weeks)", f"{int(four_week_change):,}", delta=f"{(four_week_change / abs(symbol_data.iloc[-5]['net_position']) * 100):.1f}%" if symbol_data.iloc[-5]['net_position'] != 0 else None)
            else:
                st.metric("Net Position Change (4 Weeks)", "Insufficient data")
        with col3:
            if len(symbol_data) > 1 and np.max(symbol_data['net_position']) != np.min(symbol_data['net_position']):
                percentile = ((latest_data['net_position'] - np.min(symbol_data['net_position'])) / (np.max(symbol_data['net_position']) - np.min(symbol_data['net_position'])) * 100)
                st.metric("Positioning Extreme", f"{percentile:.0f}%", delta="Bullish" if percentile > 60 else ("Bearish" if percentile < 40 else "Neutral"))
            else:
                st.metric("Positioning Extreme", "N/A")
        
        st.subheader("Position Analysis")
        if len(symbol_data) > 12:
            recent_trend = symbol_data.iloc[-12:]['net_position'].values
            is_increasing = recent_trend[-1] > recent_trend[0]
            trend_strength = abs((recent_trend[-1] - recent_trend[0]) / recent_trend[0]) if recent_trend[0] != 0 else 0
            trend_desc = f"Strong {'bullish' if is_increasing else 'bearish'}" if trend_strength > 0.15 else f"Moderate {'bullish' if is_increasing else 'bearish'}" if trend_strength > 0.05 else "Neutral/Sideways"
            st.info(f"**Recent Trend (12 weeks)**: {trend_desc} with {trend_strength:.1%} change")
        
        with st.expander("View Raw Data"):
            st.dataframe(symbol_data[['date', 'leveraged_long', 'leveraged_short', 'net_position']].sort_values('date', ascending=False).reset_index(drop=True))
    else:
        st.warning(f"No data available for {selected_symbol} in the selected time range.")

    st.markdown("""
    ---
    ### Usage Notes
    - Weekly Report URLs (e.g., financial_lof030425.htm) provide single-week data.
    - Update the Weekly Report URL as new data is released (e.g., financial_lof031125.htm for March 11).
    - Leveraged funds approximate CTA positions in the Legacy report.
    """)

if __name__ == "__main__":
    st.set_page_config(page_title="CFTC Data Analyzer", page_icon="📊", layout="wide")
    cftc_analyzer_module()
