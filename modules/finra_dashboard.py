import pandas as pd
import streamlit as st
import requests
import io
from datetime import datetime, timedelta
import yfinance as yf

# Existing functions unchanged unless noted
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
    
    bought_volume = short_volume + short_exempt_volume
    sold_volume = total_volume - bought_volume
    
    buy_to_sell_ratio = bought_volume / sold_volume if sold_volume > 0 else float('inf')
    short_volume_ratio = bought_volume / total_volume if total_volume > 0 else 0
    
    return {
        'total_volume': total_volume,
        'bought_volume': bought_volume,
        'sold_volume': sold_volume,
        'buy_to_sell_ratio': round(buy_to_sell_ratio, 2),
        'short_volume_ratio': round(short_volume_ratio, 4)
    }

def validate_pattern(symbol, dates, pattern_type="accumulation"):
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(start=min(dates), end=max(dates))
        if hist.empty:
            return False
        latest_price = hist['Close'].iloc[-1]
        avg_price = hist['Close'].mean()
        return latest_price > avg_price if pattern_type == "accumulation" else latest_price < avg_price
    except Exception:
        return False

def find_patterns(lookback_days=10, min_volume=1000000, pattern_type="accumulation", use_price_validation=False):
    pattern_data = {}
    # Adjust for trading days (skip weekends)
    for i in range(lookback_days * 2):  # Double to account for non-trading days
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
                    pattern_data[symbol] = {'dates': [], 'volumes': [], 'ratios': [], 'total_volume': 0, 'days_pattern': 0}
                pattern_data[symbol]['dates'].append(date)
                pattern_data[symbol]['volumes'].append(total_volume)
                pattern_data[symbol]['ratios'].append(metrics['buy_to_sell_ratio'])
                pattern_data[symbol]['total_volume'] += total_volume
                if pattern_type == "accumulation" and metrics['buy_to_sell_ratio'] > 1.5:  # Match original threshold
                    pattern_data[symbol]['days_pattern'] += 1
                elif pattern_type == "distribution" and metrics['buy_to_sell_ratio'] < 0.7:  # Match original
                    pattern_data[symbol]['days_pattern'] += 1
    
    results = []
    for symbol, data in pattern_data.items():
        if len(data['dates']) >= 3 and data['days_pattern'] >= 2:  # Match original requirements
            if not use_price_validation or validate_pattern(symbol, data['dates'], pattern_type):
                avg_ratio = sum(data['ratios']) / len(data['ratios'])
                avg_volume = data['total_volume'] / len(data['dates'])
                results.append({
                    'Symbol': symbol,
                    'Avg Daily Volume': int(avg_volume),
                    'Avg Buy/Sell Ratio': round(avg_ratio, 2),
                    'Days Showing Pattern': data['days_pattern'],
                    'Total Volume': int(data['total_volume']),
                    'Latest Ratio': data['ratios'][0],
                    'Volume Trend': 'Increasing' if data['volumes'][0] > avg_volume else 'Decreasing'
                })
    
    if pattern_type == "accumulation":
        results = sorted(results, key=lambda x: (x['Days Showing Pattern'], x['Avg Buy/Sell Ratio'], x['Total Volume']), reverse=True)
    else:
        results = sorted(results, key=lambda x: (x['Days Showing Pattern'], -x['Avg Buy/Sell Ratio'], x['Total Volume']), reverse=True)
    return pd.DataFrame(results[:40])

# New function for sector rotation
def get_sector_rotation(lookback_days=10, min_volume=50000):
    # Simplified sector mapping (expand with full S&P 500 sector data)
    sector_map = {
        'XLK': 'S&P Technology',
        'XLF': 'Financials',
        'XLE': 'Energy',
        'XLP': 'Consumer Staples',
        'XLY': 'Consumer Discretionary',
        'XLV': 'Healthcare',
        'XLI': 'Industrials',
        'XLU': 'Utilities',
        'XLRE': 'Real Estate',
        'XRT': 'Retail',
        'SMH': 'Semis',
        'QQQ': 'Nasdaq 100'
        # Add more symbols or fetch from yfinance
    }
    
    sector_data = {}
    for i in range(lookback_days):
        date = (datetime.now() - timedelta(days=i)).strftime("%Y%m%d")
        data = download_finra_short_sale_data(date)
        if data:
            df = process_finra_short_sale_data(data)
            df = df[df['TotalVolume'] > min_volume]
            for _, row in df.iterrows():
                symbol = row['Symbol']
                if symbol in sector_map:
                    sector = sector_map[symbol]
                    total_volume = row.get('TotalVolume', 0)
                    metrics = calculate_metrics(row, total_volume)
                    
                    if sector not in sector_data:
                        sector_data[sector] = {'dates': [], 'ratios': [], 'volumes': [], 'symbols': set()}
                    sector_data[sector]['dates'].append(date)
                    sector_data[sector]['ratios'].append(metrics['buy_to_sell_ratio'])
                    sector_data[sector]['volumes'].append(total_volume)
                    sector_data[sector]['symbols'].add(symbol)
    
    results = []
    for sector, data in sector_data.items():
        if len(data['dates']) >= 3:  # Require data for at least 3 days
            avg_ratio = sum(data['ratios']) / len(data['ratios'])
            avg_volume = sum(data['volumes']) / len(data['volumes'])
            ratio_trend = (data['ratios'][0] - data['ratios'][-1]) / len(data['ratios'])  # Daily change in ratio
            volume_trend = 'Increasing' if data['volumes'][0] > avg_volume else 'Decreasing'
            results.append({
                'Sector': sector,
                'Avg Buy/Sell Ratio': round(avg_ratio, 2),
                'Ratio Trend': 'Up' if ratio_trend > 0 else 'Down',
                'Avg Daily Volume': int(avg_volume),
                'Volume Trend': volume_trend,
                'Symbols Count': len(data['symbols']),
                'Latest Ratio': data['ratios'][0]
            })
    
    # Sort by ratio trend and volume for buying/selling signals
    buying_sectors = sorted([r for r in results if r['Ratio Trend'] == 'Up'], 
                            key=lambda x: (x['Avg Buy/Sell Ratio'], x['Avg Daily Volume']), reverse=True)
    selling_sectors = sorted([r for r in results if r['Ratio Trend'] == 'Down'], 
                             key=lambda x: (x['Avg Buy/Sell Ratio'], x['Avg Daily Volume']))
    
    return pd.DataFrame(buying_sectors[:7]), pd.DataFrame(selling_sectors[:7])

def run():
    st.title("FINRA Short Sale Analysis")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Single Stock Analysis", "Accumulation Patterns", "Distribution Patterns", "Sector Rotation"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            symbol = st.text_input("Enter Symbol", "SPY").strip().upper()
            lookback_days = st.slider("Lookback Days", 1, 30, 20)
        with col2:
            threshold = st.number_input("Buy/Sell Ratio Threshold", min_value=1.0, max_value=5.0, value=1.5, step=0.1)
        
        if st.button("Analyze Stock"):
            results_df, significant_days = analyze_symbol(symbol, lookback_days, threshold)  # Assume this function exists
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
                    return ['background-color: rgba(144, 238, 144, 0.3)' if row['buy_to_sell_ratio'] > threshold else ''] * len(row)
                display_df = results_df.copy()
                for col in ['total_volume', 'bought_volume', 'sold_volume']:
                    display_df[col] = display_df[col].astype(int)
                styled_df = display_df.style.apply(highlight_significant, axis=1)
                st.dataframe(styled_df)
    
    with tab2:
        st.subheader("Top 40 Stocks Showing Accumulation")
        col1, col2 = st.columns(2)
        with col1:
            acc_min_volume = st.number_input("Minimum Daily Volume (Accumulation)", value=1000000, step=500000, format="%d", key="acc_vol")
        with col2:
            acc_use_validation = st.checkbox("Use Price Validation", value=False, key="acc_val")
        
        if st.button("Find Accumulation Patterns"):
            accumulation_df = find_patterns(lookback_days=10, min_volume=acc_min_volume, pattern_type="accumulation", use_price_validation=acc_use_validation)
            if not accumulation_df.empty:
                def highlight_acc_pattern(row):
                    return [f'background-color: rgba(144, 238, 144, 0.3)' if row['Volume Trend'] == 'Increasing' else ''] * len(row)
                styled_df = accumulation_df.style.apply(highlight_acc_pattern, axis=1)
                st.dataframe(styled_df)
            else:
                st.write("No accumulation patterns detected with current filters.")
    
    with tab3:
        st.subheader("Top 40 Stocks Showing Distribution")
        col1, col2 = st.columns(2)
        with col1:
            dist_min_volume = st.number_input("Minimum Daily Volume (Distribution)", value=1000000, step=500000, format="%d", key="dist_vol")
        with col2:
            dist_use_validation = st.checkbox("Use Price Validation", value=False, key="dist_val")
        
        if st.button("Find Distribution Patterns"):
            distribution_df = find_patterns(lookback_days=10, min_volume=dist_min_volume, pattern_type="distribution", use_price_validation=dist_use_validation)
            if not distribution_df.empty:
                def highlight_dist_pattern(row):
                    return [f'background-color: rgba(255, 182, 193, 0.3)' if row['Volume Trend'] == 'Increasing' else ''] * len(row)
                styled_df = distribution_df.style.apply(highlight_dist_pattern, axis=1)
                st.dataframe(styled_df)
            else:
                st.write("No distribution patterns detected with current filters.")
    
    with tab4:
        st.subheader("Sector Rotation Analysis")
        col1, col2 = st.columns(2)
        with col1:
            rot_min_volume = st.number_input("Minimum Daily Volume (Sector Rotation)", value=50000, step=10000, format="%d", key="rot_vol")
        with col2:
            rot_lookback_days = st.slider("Lookback Days (Sector Rotation)", 1, 30, 10, key="rot_days")
        
        if st.button("Analyze Sector Rotation"):
            buying_df, selling_df = get_sector_rotation(lookback_days=rot_lookback_days, min_volume=rot_min_volume)
            st.write("### Sectors Being Bought (Money In)")
            def highlight_buying(row):
                return ['background-color: rgba(144, 238, 144, 0.3)'] * len(row)
            if not buying_df.empty:
                styled_buying = buying_df.style.apply(highlight_buying, axis=1)
                st.dataframe(styled_buying)
            else:
                st.write("No sectors with clear buying trends detected.")
            st.write("### Sectors Being Sold (Money Out)")
            def highlight_selling(row):
                return ['background-color: rgba(255, 182, 193, 0.3)'] * len(row)
            if not selling_df.empty:
                styled_selling = selling_df.style.apply(highlight_selling, axis=1)
                st.dataframe(styled_selling)
            else:
                st.write("No sectors with clear selling trends detected.")

if __name__ == "__main__":
    run()
