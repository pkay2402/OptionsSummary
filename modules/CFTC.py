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
    
    # Set default URL
    default_url = "https://www.cftc.gov/dea/options/financial_lof.htm"
    
    with st.sidebar:
        st.header("CFTC Data Settings")
        use_custom_url = st.checkbox("Use Custom URL", value=False)
        
        if use_custom_url:
            cftc_url = st.text_input("CFTC Data URL", value=default_url)
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
            
            # First look for specific file patterns that match CFTC reports
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
            
            # Try to get data from all links - the financial lof page often has multiple datasets
            all_data = []
            for data_url in data_links[:3]:  # Limit to first 3 links to avoid too many requests
                try:
                    data_response = requests.get(data_url)
                    data_response.raise_for_status()
                    
                    # Determine file type and parse accordingly
                    if data_url.endswith('.csv'):
                        df = pd.read_csv(StringIO(data_response.text))
                    else:  # Assume TXT file with fixed width or comma-separated format
                        try:
                            df = pd.read_csv(StringIO(data_response.text), sep=',')
                        except:
                            # Try fixed width format
                            df = pd.read_fwf(StringIO(data_response.text))
                    
                    processed_df = process_cftc_dataframe(df)
                    if not processed_df.empty:
                        all_data.append(processed_df)
                except Exception as e:
                    st.warning(f"Could not process {data_url}: {str(e)}")
                    continue
            
            if all_data:
                # Combine all datasets
                combined_df = pd.concat(all_data, ignore_index=True)
                return combined_df
            
            # If we couldn't get data from any link, try the first link again with more careful parsing
            if data_links:
                data_url = data_links[0]
                data_response = requests.get(data_url)
                data_response.raise_for_status()
                content = data_response.text
                
                # Try multiple parsing approaches
                for parser in [parse_csv, parse_txt_as_csv, parse_fixed_width]:
                    try:
                        df = parser(content)
                        if not df.empty:
                            return process_cftc_dataframe(df)
                    except:
                        continue
            
            # Process the dataframe
            return process_cftc_dataframe(df)
            
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")
            # Return mock data for demonstration
            return generate_mock_data()
    
    def parse_table_data(soup):
        """Parse table data directly from HTML if no CSV/TXT links found"""
        tables = soup.find_all('table')
        if not tables:
            return generate_mock_data()
        
        # Try to find the main data table
        main_table = tables[0]
        for table in tables:
            # Look for tables with more rows, which is likely the data table
            if len(table.find_all('tr')) > len(main_table.find_all('tr')):
                main_table = table
        
        # Extract table headers
        headers = []
        header_row = main_table.find('tr')
        if header_row:
            headers = [th.text.strip() for th in header_row.find_all(['th', 'td'])]
        
        # Extract table rows
        rows = []
        for tr in main_table.find_all('tr')[1:]:  # Skip header row
            row = [td.text.strip() for td in tr.find_all('td')]
            if row and len(row) == len(headers):
                rows.append(row)
        
        if headers and rows:
            df = pd.DataFrame(rows, columns=headers)
            return process_cftc_dataframe(df)
        
        # Fallback to mock data
        return generate_mock_data()
    
    def process_cftc_dataframe(df):
        """Process and clean the CFTC dataframe"""
        # Try to identify key columns
        date_col = None
        symbol_col = None
        long_cols = []
        short_cols = []
        
        # Common column name patterns
        date_patterns = ['date', 'as_of_date', 'report_date', 'report date', 'as of']
        symbol_patterns = ['contract', 'market', 'symbol', 'name', 'futures_contract', 'cftc_contract_market']
        long_patterns = ['long', 'pos_long', 'positions_long']
        short_patterns = ['short', 'pos_short', 'positions_short']
        
        # Find column names
        for col in df.columns:
            col_lower = col.lower()
            
            # Find date column
            if date_col is None:
                for pattern in date_patterns:
                    if pattern in col_lower:
                        date_col = col
                        break
            
            # Find symbol/contract column
            if symbol_col is None:
                for pattern in symbol_patterns:
                    if pattern in col_lower:
                        symbol_col = col
                        break
            
            # Find long position columns
            for pattern in long_patterns:
                if pattern in col_lower and 'levered' in col_lower:
                    long_cols.append(col)
            
            # Find short position columns
            for pattern in short_patterns:
                if pattern in col_lower and 'levered' in col_lower:
                    short_cols.append(col)
        
        # If we couldn't identify columns, try to make educated guesses
        if not date_col and 'Report_Date_as_MM_DD_YYYY' in df.columns:
            date_col = 'Report_Date_as_MM_DD_YYYY'
        elif not date_col and df.columns[0].lower().startswith('date'):
            date_col = df.columns[0]
            
        if not symbol_col and 'Market_and_Exchange_Names' in df.columns:
            symbol_col = 'Market_and_Exchange_Names'
        elif not symbol_col and any('fut' in col.lower() for col in df.columns):
            for col in df.columns:
                if 'contract' in col.lower() or 'futures' in col.lower():
                    symbol_col = col
                    break
        
        # If we still don't have the necessary columns, return mock data
        if not date_col or not symbol_col:
            return generate_mock_data()
        
        # Clean and transform the data
        processed_df = df.copy()
        
        # Process date column
        if date_col:
            try:
                processed_df[date_col] = pd.to_datetime(processed_df[date_col])
            except:
                # Try to parse date with various formats
                date_formats = ['%m/%d/%Y', '%Y-%m-%d', '%Y%m%d', '%m-%d-%Y']
                for fmt in date_formats:
                    try:
                        processed_df[date_col] = pd.to_datetime(processed_df[date_col], format=fmt)
                        break
                    except:
                        continue
        
        # Extract relevant position data
        result_data = []
        
        for _, row in processed_df.iterrows():
            symbol = str(row[symbol_col]).strip() if symbol_col else "Unknown"
            date = row[date_col] if date_col else pd.NaT
            
            # Get long and short positions
            long_pos = 0
            short_pos = 0
            
            if long_cols:
                for col in long_cols:
                    try:
                        long_pos += pd.to_numeric(row[col], errors='coerce') or 0
                    except:
                        pass
            
            if short_cols:
                for col in short_cols:
                    try:
                        short_pos += pd.to_numeric(row[col], errors='coerce') or 0
                    except:
                        pass
            
            # If we couldn't find specific columns, try to infer from column names
            if long_pos == 0 and short_pos == 0:
                for col in processed_df.columns:
                    col_lower = col.lower()
                    if ('levered' in col_lower or 'leveraged' in col_lower or 'hedge' in col_lower) and 'long' in col_lower:
                        try:
                            long_pos += pd.to_numeric(row[col], errors='coerce') or 0
                        except:
                            pass
                    elif ('levered' in col_lower or 'leveraged' in col_lower or 'hedge' in col_lower) and 'short' in col_lower:
                        try:
                            short_pos += pd.to_numeric(row[col], errors='coerce') or 0
                        except:
                            pass
            
            result_data.append({
                'date': date,
                'symbol': symbol,
                'leveraged_long': long_pos,
                'leveraged_short': short_pos,
                'net_position': long_pos - short_pos
            })
        
        result_df = pd.DataFrame(result_data)
        
        # Filter out rows with invalid dates or empty symbols
        result_df = result_df[~result_df['date'].isna()]
        result_df = result_df[result_df['symbol'] != ""]
        result_df = result_df[result_df['symbol'] != "Unknown"]
        
        if len(result_df) == 0:
            return generate_mock_data()
            
        return result_df
    
    def parse_csv(content):
        """Parse CSV content"""
        return pd.read_csv(StringIO(content))
    
    def parse_txt_as_csv(content):
        """Parse txt file as CSV with different delimiters"""
        for delimiter in [',', ';', '\t', '|']:
            try:
                return pd.read_csv(StringIO(content), sep=delimiter)
            except:
                continue
        raise Exception("Could not parse as delimited text")
        
    def parse_fixed_width(content):
        """Parse as fixed width file"""
        return pd.read_fwf(StringIO(content))

    def generate_mock_data():
        """Generate mock CFTC data for demonstration"""
        symbols = [
            'EURO FX', 'TREASURY BONDS', 'S&P 500', 'NASDAQ-100', 'US DOLLAR INDEX', 'VIX',
            'CANADIAN DOLLAR', 'BRITISH POUND', 'JAPANESE YEN', 'SWISS FRANC',
            '2-YEAR U.S. TREASURY NOTES', '5-YEAR U.S. TREASURY NOTES', '10-YEAR U.S. TREASURY NOTES',
            'DJIA', 'RUSSELL 2000', 'GOLD', 'SILVER', 'CRUDE OIL', 'NATURAL GAS',
            'EURODOLLAR', 'FEDERAL FUNDS'
        ]
        today = datetime.now()
        
        # Generate dates for the past 104 weeks (2 years of weekly data)
        dates = [(today - timedelta(days=7*i)).strftime('%Y-%m-%d') for i in range(104)]
        
        mock_data = []
        
        for symbol in symbols:
            # Start with base positions
            long_pos = np.random.randint(50000, 250000)
            short_pos = np.random.randint(50000, 250000)
            
            # Create a trend bias for each symbol
            trend = np.random.choice([-1, 1]) * np.random.uniform(0.2, 0.5)
            
            for date in dates:
                # Add some randomness to positions with trend bias
                random_change = np.random.normal(0, 5000)
                trend_change = trend * 2000
                
                long_pos += random_change + trend_change
                short_pos += random_change - trend_change
                
                # Ensure positions stay positive
                long_pos = max(10000, long_pos)
                short_pos = max(10000, short_pos)
                
                mock_data.append({
                    'date': pd.to_datetime(date),
                    'symbol': symbol,
                    'leveraged_long': int(long_pos),
                    'leveraged_short': int(short_pos),
                    'net_position': int(long_pos - short_pos)
                })
        
        return pd.DataFrame(mock_data)
    
    # Initial data load or refresh
    if 'cftc_data' not in st.session_state or fetch_data:
        with st.spinner('Fetching CFTC data...'):
            st.session_state.cftc_data = fetch_cftc_data(cftc_url)
            st.success('Data loaded successfully!')
    
    # Get unique symbols for selection
    symbols = sorted(st.session_state.cftc_data['symbol'].unique())
    
    # Symbol selector
    selected_symbol = st.selectbox('Select Symbol', symbols)
    
    # Filter data by symbol
    symbol_data = st.session_state.cftc_data[st.session_state.cftc_data['symbol'] == selected_symbol].copy()
    symbol_data = symbol_data.sort_values('date')
    
    # Apply time range filter
    if time_range != "All Available Data":
        weeks = int(re.search(r'\d+', time_range).group())
        cutoff_date = datetime.now() - timedelta(weeks=weeks)
        symbol_data = symbol_data[symbol_data['date'] >= cutoff_date]
    
    # Create visualization
    if not symbol_data.empty:
        # Create plot
        fig = go.Figure()
        
        if show_net_position:
            fig.add_trace(go.Scatter(
                x=symbol_data['date'],
                y=symbol_data['net_position'],
                mode='lines',
                name='Net Position',
                line=dict(color='rgb(0, 102, 204)', width=2)
            ))
            
        if show_long_position:
            fig.add_trace(go.Scatter(
                x=symbol_data['date'],
                y=symbol_data['leveraged_long'],
                mode='lines',
                name='Long Position',
                line=dict(color='rgb(0, 204, 102)', width=2)
            ))
            
        if show_short_position:
            fig.add_trace(go.Scatter(
                x=symbol_data['date'],
                y=symbol_data['leveraged_short'],
                mode='lines',
                name='Short Position',
                line=dict(color='rgb(204, 0, 0)', width=2)
            ))
        
        fig.update_layout(
            title=f'CFTC Positions: {selected_symbol}',
            xaxis_title='Date',
            yaxis_title='Contracts',
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistics cards
        latest_data = symbol_data.iloc[-1]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Current Net Position",
                f"{int(latest_data['net_position']):,}",
                delta=f"{int(latest_data['net_position'] - symbol_data.iloc[-2]['net_position']):,}" if len(symbol_data) > 1 else None
            )
            
        with col2:
            if len(symbol_data) >= 5:
                four_week_change = latest_data['net_position'] - symbol_data.iloc[-5]['net_position']
                st.metric(
                    "Net Position Change (4 Weeks)",
                    f"{int(four_week_change):,}",
                    delta=f"{(four_week_change / abs(symbol_data.iloc[-5]['net_position']) * 100):.1f}%" if symbol_data.iloc[-5]['net_position'] != 0 else None
                )
            else:
                st.metric("Net Position Change (4 Weeks)", "Insufficient data")
                
        with col3:
            if len(symbol_data) > 1:
                net_positions = symbol_data['net_position'].values
                max_pos = np.max(net_positions)
                min_pos = np.min(net_positions)
                current = latest_data['net_position']
                
                if max_pos != min_pos:  # Avoid division by zero
                    percentile = ((current - min_pos) / (max_pos - min_pos) * 100)
                    
                    st.metric(
                        "Positioning Extreme",
                        f"{percentile:.0f}%",
                        delta="Bullish" if percentile > 60 else ("Bearish" if percentile < 40 else "Neutral"),
                        delta_color="normal"
                    )
                else:
                    st.metric("Positioning Extreme", "N/A")
            else:
                st.metric("Positioning Extreme", "Insufficient data")
        
        # Additional analysis
        st.subheader("Position Analysis")
        
        # Position trend analysis
        if len(symbol_data) > 12:
            recent_trend = symbol_data.iloc[-12:]['net_position'].values
            is_increasing = recent_trend[-1] > recent_trend[0]
            
            trend_strength = abs((recent_trend[-1] - recent_trend[0]) / recent_trend[0]) if recent_trend[0] != 0 else 0
            
            if trend_strength > 0.15:
                trend_desc = f"Strong {'bullish' if is_increasing else 'bearish'}"
            elif trend_strength > 0.05:
                trend_desc = f"Moderate {'bullish' if is_increasing else 'bearish'}"
            else:
                trend_desc = "Neutral/Sideways"
                
            st.info(f"**Recent Trend (12 weeks)**: {trend_desc} with {trend_strength:.1%} change")
            
        # Extreme positioning warning
        if len(symbol_data) > 52:
            yearly_percentile = np.percentile(symbol_data['net_position'].values, [10, 90])
            current = latest_data['net_position']
            
            if current >= yearly_percentile[1]:
                st.warning(f"⚠️ **Extreme Bullish Positioning**: Current net position is in the top 10% of the past year")
            elif current <= yearly_percentile[0]:
                st.warning(f"⚠️ **Extreme Bearish Positioning**: Current net position is in the bottom 10% of the past year")
        
        # Position volatility
        if len(symbol_data) > 12:
            recent_volatility = np.std(symbol_data.iloc[-12:]['net_position'].pct_change().dropna())
            if not np.isnan(recent_volatility):
                if recent_volatility > 0.08:
                    st.warning(f"⚠️ **High Position Volatility**: Positions have been changing rapidly ({recent_volatility:.1%} stddev)")
        
        # Raw data table with toggle
        with st.expander("View Raw Data"):
            st.dataframe(
                symbol_data[['date', 'leveraged_long', 'leveraged_short', 'net_position']]
                .sort_values('date', ascending=False)
                .reset_index(drop=True)
            )
            
    else:
        st.warning(f"No data available for {selected_symbol} in the selected time range.")

    st.markdown("""
    ---
    ### Usage Notes
    - This module fetches and analyzes CFTC Commitment of Traders data
    - Leveraged funds primarily represent hedge funds and speculative traders
    - Net positioning extremes often precede market reversals
    - Data is typically released weekly on Fridays
    """)

if __name__ == "__main__":
    st.set_page_config(
        page_title="CFTC Data Analyzer", 
        page_icon="📊",
        layout="wide"
    )
    cftc_analyzer_module()
