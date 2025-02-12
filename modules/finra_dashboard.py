import pandas as pd
import streamlit as st
import requests
import io
from datetime import datetime, timedelta

def download_finra_short_sale_data(date):
    url = f"https://cdn.finra.org/equity/regsho/daily/CNMSshvol{date}.txt"
    response = requests.get(url)
    return response.text if response.status_code == 200 else None

def process_finra_short_sale_data(data):
    if not data:
        return pd.DataFrame()
    df = pd.read_csv(io.StringIO(data), delimiter="|")
    return df[df["Symbol"].str.len() <= 4]

def calculate_metrics(row, total_volume):
    short_volume = row.get('ShortVolume', 0)
    short_exempt_volume = row.get('ShortExemptVolume', 0)
    
    bought_volume = short_volume
    sold_volume = total_volume - short_volume - short_exempt_volume
    
    buy_to_sell_ratio = bought_volume / sold_volume if sold_volume > 0 else float('inf')
    short_volume_ratio = (short_volume + short_exempt_volume) / total_volume if total_volume > 0 else 0
    
    return {
        'total_volume': total_volume,
        'bought_volume': bought_volume,
        'sold_volume': sold_volume,
        'buy_to_sell_ratio': round(buy_to_sell_ratio, 2),
        'short_volume_ratio': round(short_volume_ratio, 4)
    }

def analyze_symbol(symbol, lookback_days=20, threshold=1.5):
    results = []
    significant_days = 0
    
    for i in range(lookback_days):
        date = (datetime.now() - timedelta(days=i)).strftime("%Y%m%d")
        data = download_finra_short_sale_data(date)
        
        if data:
            df = process_finra_short_sale_data(data)
            symbol_data = df[df['Symbol'] == symbol]
            
            if not symbol_data.empty:
                row = symbol_data.iloc[0]
                total_volume = row.get('TotalVolume', 0)
                metrics = calculate_metrics(row, total_volume)
                
                if metrics['buy_to_sell_ratio'] > threshold:
                    significant_days += 1
                
                metrics['date'] = date
                results.append(metrics)
    
    df_results = pd.DataFrame(results)
    if not df_results.empty:
        df_results['date'] = pd.to_datetime(df_results['date'], format='%Y%m%d')
        df_results = df_results.sort_values('date', ascending=False)
    
    return df_results, significant_days

def find_patterns(lookback_days=5, min_volume=1000000, pattern_type="accumulation"):
    """Find stocks showing accumulation or distribution patterns"""
    pattern_data = {}
    
    for i in range(lookback_days):
        date = (datetime.now() - timedelta(days=i)).strftime("%Y%m%d")
        data = download_finra_short_sale_data(date)
        
        if data:
            df = process_finra_short_sale_data(data)
            df = df[df['TotalVolume'] > min_volume]
            
            for _, row in df.iterrows():
                symbol = row['Symbol']
                total_volume = row.get('TotalVolume', 0)
                metrics = calculate_metrics(row, total_volume)
                
                if symbol not in pattern_data:
                    pattern_data[symbol] = {
                        'dates': [],
                        'volumes': [],
                        'ratios': [],
                        'total_volume': 0,
                        'days_pattern': 0
                    }
                
                pattern_data[symbol]['dates'].append(date)
                pattern_data[symbol]['volumes'].append(total_volume)
                pattern_data[symbol]['ratios'].append(metrics['buy_to_sell_ratio'])
                pattern_data[symbol]['total_volume'] += total_volume
                
                # For accumulation, look for high buy/sell ratio
                # For distribution, look for low buy/sell ratio
                if pattern_type == "accumulation" and metrics['buy_to_sell_ratio'] > 1.5:
                    pattern_data[symbol]['days_pattern'] += 1
                elif pattern_type == "distribution" and metrics['buy_to_sell_ratio'] < 0.7:
                    pattern_data[symbol]['days_pattern'] += 1
    
    results = []
    for symbol, data in pattern_data.items():
        if len(data['dates']) >= 3:  # Must have at least 3 days of data
            avg_ratio = sum(data['ratios']) / len(data['ratios'])
            avg_volume = data['total_volume'] / len(data['dates'])
            
            if data['days_pattern'] >= 2:  # At least 2 days showing pattern
                results.append({
                    'Symbol': symbol,
                    'Avg Daily Volume': int(avg_volume),
                    'Avg Buy/Sell Ratio': round(avg_ratio, 2),
                    'Days Showing Pattern': data['days_pattern'],
                    'Total Volume': int(data['total_volume']),
                    'Latest Ratio': data['ratios'][0],
                    'Volume Trend': 'Increasing' if data['volumes'][0] > avg_volume else 'Decreasing'
                })
    
    # Sort based on pattern type
    if pattern_type == "accumulation":
        results = sorted(results, 
                        key=lambda x: (x['Days Showing Pattern'], x['Avg Buy/Sell Ratio'], x['Total Volume']), 
                        reverse=True)
    else:
        results = sorted(results, 
                        key=lambda x: (x['Days Showing Pattern'], -x['Avg Buy/Sell Ratio'], x['Total Volume']), 
                        reverse=True)
    
    return pd.DataFrame(results[:20])

def run():
    st.title("FINRA Short Sale Analysis")
    
    tab1, tab2, tab3 = st.tabs(["Single Stock Analysis", "Accumulation Patterns", "Distribution Patterns"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            symbol = st.text_input("Enter Symbol", "SPY").strip().upper()
            lookback_days = st.slider("Lookback Days", 1, 30, 20)
        with col2:
            threshold = st.number_input("Buy/Sell Ratio Threshold", 
                                      min_value=1.0, 
                                      max_value=5.0, 
                                      value=1.5,
                                      step=0.1)
        
        if st.button("Analyze Stock"):
            results_df, significant_days = analyze_symbol(symbol, lookback_days, threshold)
            
            if not results_df.empty:
                st.subheader("Summary")
                avg_ratio = results_df['buy_to_sell_ratio'].mean()
                max_ratio = results_df['buy_to_sell_ratio'].max()
                
                metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                with metrics_col1:
                    st.metric("Average Buy/Sell Ratio", f"{avg_ratio:.2f}")
                with metrics_col2:
                    st.metric("Max Buy/Sell Ratio", f"{max_ratio:.2f}")
                with metrics_col3:
                    st.metric(f"Days Above {threshold}", significant_days)
                
                st.subheader("Daily Analysis")
                def highlight_significant(row):
                    if row['buy_to_sell_ratio'] > threshold:
                        return ['background-color: rgba(144, 238, 144, 0.3)'] * len(row)
                    return [''] * len(row)
                
                display_df = results_df.copy()
                for col in ['total_volume', 'bought_volume', 'sold_volume']:
                    display_df[col] = display_df[col].astype(int)
                
                styled_df = display_df.style.apply(highlight_significant, axis=1)
                st.dataframe(styled_df)
    
    with tab2:
        st.subheader("Top 20 Stocks Showing Accumulation")
        col1, col2 = st.columns(2)
        with col1:
            acc_min_volume = st.number_input("Minimum Daily Volume (Accumulation)", 
                                           value=1000000, 
                                           step=500000,
                                           format="%d",
                                           key="acc_vol")
        
        if st.button("Find Accumulation Patterns"):
            accumulation_df = find_patterns(
                lookback_days=5,
                min_volume=acc_min_volume,
                pattern_type="accumulation"
            )
            
            def highlight_acc_pattern(row):
                color = 'rgba(144, 238, 144, 0.3)' if row['Volume Trend'] == 'Increasing' else ''
                return [f'background-color: {color}'] * len(row)
            
            styled_df = accumulation_df.style.apply(highlight_acc_pattern, axis=1)
            st.dataframe(styled_df)
    
    with tab3:
        st.subheader("Top 20 Stocks Showing Distribution")
        col1, col2 = st.columns(2)
        with col1:
            dist_min_volume = st.number_input("Minimum Daily Volume (Distribution)", 
                                            value=1000000, 
                                            step=500000,
                                            format="%d",
                                            key="dist_vol")
        
        if st.button("Find Distribution Patterns"):
            distribution_df = find_patterns(
                lookback_days=5,
                min_volume=dist_min_volume,
                pattern_type="distribution"
            )
            
            def highlight_dist_pattern(row):
                color = 'rgba(255, 182, 193, 0.3)' if row['Volume Trend'] == 'Increasing' else ''  # Light red
                return [f'background-color: {color}'] * len(row)
            
            styled_df = distribution_df.style.apply(highlight_dist_pattern, axis=1)
            st.dataframe(styled_df)

if __name__ == "__main__":
    run()
