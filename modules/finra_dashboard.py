import pandas as pd
import streamlit as st
import requests
import io
from datetime import datetime, timedelta
import yfinance as yf
import plotly.express as px
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Sector mapping
sector_mapping = {
    'Index/Sectors': [
        'SPY',  # SPY
        'QQQ',  # QQQ
        'DIA',  # DIA
        'IWM',  # IWM
        'VXX',  # VXX
        'SMH',  # SMH
        'IVV',  # IVV
        'VTI',  # VTI
        'VOO',  # VOO
        'XLK',  # XLK
        'VUG',  # VUG
        'XLF',  # XLF
        'XLV',  # XLV
        'XLY',  # XLY
        'XLC',  # XLC
        'XLI',  # XLI
        'XLE',  # XLE
        'XLB',  # XLB
        'XLU',  # XLU
        'XLRE', # XLRE
        'XLP',  # XLP
        'XBI',  # XBI
        'XOP',  # XOP
        'XME',  # XME
        'XRT',  # XRT
        'XHB',  # XHB
        'XWEB', # XWEB
        'XLC',  # XLC
        'RSP',  # RSP
    ],
    'Technology': [
        'AAPL',  # Apple
        'MSFT',  # Microsoft
        'NVDA',  # NVIDIA
        'AMZN',  # Amazon
        'GOOGL', # Alphabet Class A
        'META',  # Meta Platforms
        'TSLA',  # Tesla
        'PLTR',  # Palantir
        'ORCL',  # Oracle
        'AMD',   # Advanced Micro Devices
        'NFLX',  # Netflix
        'ADBE',  # Adobe
        'CRM',   # Salesforce
        'INTC',  # Intel
        'CSCO',  # Cisco Systems
        'QCOM',  # Qualcomm
        'TXN',   # Texas Instruments
        'INTU',  # Intuit
        'IBM',   # IBM
        'NOW',   # ServiceNow
        'AVGO',  # Broadcom
        'UBER',  # Uber Technologies
        'SNOW',  # Snowflake
        'DELL',  # Dell Technologies
        'PANW'   # Palo Alto Networks
    ],
    'Financials': [
        'JPM',   # JPMorgan Chase
        'V',     # Visa
        'MA',    # Mastercard
        'BAC',   # Bank of America
        'WFC',   # Wells Fargo
        'GS',    # Goldman Sachs
        'MS',    # Morgan Stanley
        'C',     # Citigroup
        'AXP',   # American Express
        'SCHW',  # Charles Schwab
        'COF',   # Capital One
        'MET',   # MetLife
        'AIG',   # American International Group
        'BK',    # Bank of New York Mellon
        'BLK',   # BlackRock
        'TFC',   # Truist Financial
        'USB',   # U.S. Bancorp
        'PNC',   # PNC Financial Services
        'CME',   # CME Group
        'SPGI',  # S&P Global
        'ICE',   # Intercontinental Exchange
        'MCO',   # Moody's
        'AON',   # Aon
        'PYPL',  # PayPal
        'SQ'     # Square (Block)
    ],
    'Healthcare': [
        'LLY',   # Eli Lilly
        'UNH',   # UnitedHealth Group
        'JNJ',   # Johnson & Johnson
        'PFE',   # Pfizer
        'MRK',   # Merck
        'ABBV',  # AbbVie
        'TMO',   # Thermo Fisher Scientific
        'AMGN',  # Amgen
        'GILD',  # Gilead Sciences
        'CVS',   # CVS Health
        'MDT',   # Medtronic
        'BMY',   # Bristol Myers Squibb
        'ABT',   # Abbott Laboratories
        'DHR',   # Danaher
        'ISRG',  # Intuitive Surgical
        'SYK',   # Stryker
        'REGN',  # Regeneron Pharmaceuticals
        'VRTX',  # Vertex Pharmaceuticals
        'CI',    # Cigna
        'ZTS',   # Zoetis
        'BDX',   # Becton Dickinson
        'HCA',   # HCA Healthcare
        'EW',    # Edwards Lifesciences
        'DXCM',  # DexCom
        'BIIB'   # Biogen
    ],
    'Consumer': [
        'WMT',   # Walmart
        'PG',    # Procter & Gamble
        'KO',    # Coca-Cola
        'PEP',   # PepsiCo
        'COST',  # Costco
        'MCD',   # McDonald's
        'DIS',   # Walt Disney
        'NKE',   # Nike
        'SBUX',  # Starbucks
        'LOW',   # Lowe's
        'TGT',   # Target
        'HD',    # Home Depot
        'CL',    # Colgate-Palmolive
        'MO',    # Altria Group
        'KHC',   # Kraft Heinz
        'PM',    # Philip Morris
        'TJX',   # TJX Companies
        'DG',    # Dollar General
        'DLTR',  # Dollar Tree
        'YUM',   # Yum! Brands
        'GIS',   # General Mills
        'KMB',   # Kimberly-Clark
        'MNST',  # Monster Beverage
        'EL',    # Estee Lauder
        'CMG'    # Chipotle Mexican Grill
    ],
    'Energy': [
        'XOM',   # Exxon Mobil
        'CVX',   # Chevron
        'COP',   # ConocoPhillips
        'SLB',   # Schlumberger
        'EOG',   # EOG Resources
        'MPC',   # Marathon Petroleum
        'PSX',   # Phillips 66
        'OXY',   # Occidental Petroleum
        'VLO',   # Valero Energy
        'PXD',   # Pioneer Natural Resources
        'HES',   # Hess
        'WMB',   # Williams Companies
        'KMI',   # Kinder Morgan
        'OKE',   # ONEOK
        'HAL',   # Halliburton
        'BKR',   # Baker Hughes
        'FANG',  # Diamondback Energy
        'DVN',   # Devon Energy
        'TRGP',  # Targa Resources
        'APA',   # APA Corporation
        'EQT',   # EQT Corporation
        'MRO',   # Marathon Oil
        'NOV',   # NOV Inc.
        'FTI',   # TechnipFMC
        'RRC'    # Range Resources
    ],
    'Industrials': [
        'CAT',   # Caterpillar
        'DE',    # Deere & Company
        'UPS',   # United Parcel Service
        'FDX',   # FedEx
        'BA',    # Boeing
        'HON',   # Honeywell
        'UNP',   # Union Pacific
        'MMM',   # 3M
        'GE',    # General Electric
        'LMT',   # Lockheed Martin
        'RTX',   # RTX Corporation
        'GD',    # General Dynamics
        'CSX',   # CSX Corporation
        'NSC',   # Norfolk Southern
        'WM',    # Waste Management
        'ETN',   # Eaton
        'ITW',   # Illinois Tool Works
        'EMR',   # Emerson Electric
        'PH',    # Parker-Hannifin
        'ROK',   # Rockwell Automation
        'CARR',  # Carrier Global
        'OTIS',  # Otis Worldwide
        'IR',    # Ingersoll Rand
        'CMI',   # Cummins
        'FAST'   # Fastenal
    ],
    'User Defined': []  # Catch-all for unmapped symbols
}

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
                    pattern_data[symbol] = {'dates': [], 'volumes': [], 'ratios': [], 'total_volume': 0, 'days_pattern': 0}
                pattern_data[symbol]['dates'].append(date)
                pattern_data[symbol]['volumes'].append(total_volume)
                pattern_data[symbol]['ratios'].append(metrics['buy_to_sell_ratio'])
                pattern_data[symbol]['total_volume'] += total_volume
                if pattern_type == "accumulation" and metrics['buy_to_sell_ratio'] > 1.5:
                    pattern_data[symbol]['days_pattern'] += 1
                elif pattern_type == "distribution" and metrics['buy_to_sell_ratio'] < 0.7:
                    pattern_data[symbol]['days_pattern'] += 1
    
    results = []
    for symbol, data in pattern_data.items():
        if len(data['dates']) >= 3 and data['days_pattern'] >= 2:
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

def get_sector_rotation(lookback_days=10, min_volume=50000):
    sector_map = {
        'QQQ': 'Technology',
        'VGT': 'Technology',
        'IVW': 'Large-Cap Growth',
        'VUG': 'Large-Cap Growth',
        'SCHG': 'Large-Cap Growth',
        'MGK': 'Large-Cap Growth',
        'VOOG': 'Large-Cap Growth',
        'IWP': 'Mid-Cap Growth',
        'VOT': 'Mid-Cap Growth',
        'EFG': 'International Growth',
        'VTV': 'Large-Cap Value',
        'IVE': 'Large-Cap Value',
        'SCHV': 'Large-Cap Value',
        'IWD': 'Large-Cap Value',
        'VLUE': 'Large-Cap Value',
        'VOE': 'Mid-Cap Value',
        'IWS': 'Mid-Cap Value',
        'VBR': 'Small-Cap Value',
        'DHS': 'High Dividend Value',
        'RPV': 'S&P Pure Value'
    }
    sector_etfs = yf.Tickers(list(sector_map.keys()))
    sector_performance = {ticker: ticker_obj.history(period=f"{lookback_days}d")['Close'].pct_change().mean() 
                          for ticker, ticker_obj in sector_etfs.tickers.items()}
    
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
        if len(data['dates']) >= 3:
            avg_ratio = sum(data['ratios']) / len(data['ratios'])
            avg_volume = sum(data['volumes']) / len(data['volumes'])
            ratio_trend = (data['ratios'][0] - data['ratios'][-1]) / len(data['ratios'])
            volume_trend = 'Increasing' if data['volumes'][0] > avg_volume else 'Decreasing'
            results.append({
                'Sector': sector,
                'Symbol': ', '.join(data['symbols']),
                'Avg Buy/Sell Ratio': round(avg_ratio, 2),
                'Ratio Trend': 'Up' if ratio_trend > 0 else 'Down',
                'Avg Daily Volume': int(avg_volume),
                'Volume Trend': volume_trend,
                'Symbols Count': len(data['symbols']),
                'Latest Ratio': data['ratios'][0],
                'Sector Performance': round(sector_performance.get(sector.split()[0], 0) * 100, 2)
            })
    
    buying_sectors = sorted([r for r in results if r['Ratio Trend'] == 'Up'], 
                            key=lambda x: (x['Avg Buy/Sell Ratio'], x['Avg Daily Volume']), reverse=True)
    selling_sectors = sorted([r for r in results if r['Ratio Trend'] == 'Down'], 
                             key=lambda x: (x['Avg Buy/Sell Ratio'], x['Avg Daily Volume']))
    
    return pd.DataFrame(buying_sectors[:7]), pd.DataFrame(selling_sectors[:7])

def analyze_portfolio(symbols, lookback_days=20):
    portfolio_results = []
    for symbol in symbols:
        df, significant_days = analyze_symbol(symbol, lookback_days, threshold=1.5)
        if not df.empty:
            for _, row in df.iterrows():
                portfolio_results.append({
                    'Symbol': symbol,
                    'Date': row['date'],
                    'Avg Buy/Sell Ratio': row['buy_to_sell_ratio'],
                    'Bought Volume': int(row['bought_volume']),
                    'Sold Volume': int(row['sold_volume']),
                    'Significant Days': significant_days,
                    'Latest Volume': int(row['total_volume'])
                })
    portfolio_df = pd.DataFrame(portfolio_results)
    if not portfolio_df.empty:
        portfolio_df = portfolio_df.sort_values(by=['Avg Buy/Sell Ratio', 'Bought Volume'], ascending=[False, False])
    return portfolio_df

def check_alerts(df_results, symbol, threshold=2.0):
    if not df_results.empty and df_results['buy_to_sell_ratio'].max() > threshold:
        st.warning(f"Alert: {symbol} has a Buy/Sell Ratio above {threshold} on {df_results['date'].iloc[0].strftime('%Y-%m-%d')}!")

def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

def plot_correlation_heatmap(symbols, lookback_days=20):
    data = {}
    for symbol in symbols:
        df, _ = analyze_symbol(symbol, lookback_days)
        if not df.empty:
            data[symbol] = df['buy_to_sell_ratio']
    corr_df = pd.DataFrame(data).corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_df, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

def run():
    st.markdown("""
        <style>
        .stButton>button {
            background-color: #4CAF50;
            color: white;
        }
        .stTabs [data-baseweb="tab"] {
            font-size: 16px;
        }
        </style>
        """, unsafe_allow_html=True)
    
    st.title("FINRA Short Sale Analysis")
    
    with st.sidebar:
        portfolio_symbols = st.text_area("User Defined Symbols for Multi Stock Analysis (comma-separated)", 
                                         "AAPL, MSFT, NVDA, AMZN, GOOGL, META, TSLA, PLTR, LLY, JPM, AVGO, UNH, V, WMT, XOM, MA, JNJ, PG, HD, COST, ORCL, CVX, BAC, KO, PEP, ABBV, "
                                         "MRK, AMD, NFLX, ADBE, CRM, INTC, CSCO, PFE, TMO, MCD, DIS, WFC, QCOM, LIN, GE, AXP, CAT, IBM, VZ, GS, MS, PM, LOW, NEE, RTX, BA, HON, UNP, "
                                         "T, BLK, MDT, SBUX, LMT, AMGN, GILD, CVS, DE, TGT, AMT, BKNG, SO, DUK, PYPL, UPS, C, COP, MMM, ACN, ABT, DHR, TMUS, TXN, MDLZ, BMY, INTU, NKE, "
                                         "CL, MO, KHC, EMR, GM, F, FDX, "
                                         "GD, BK, SPG, CHTR, USB, MET, AIG, COF, DOW, SCHW, CMCSA, SPY, QQQ, VOO, IVV, XLK, VUG, XLV, XLF, IWF, VTI").split(",")
        portfolio_symbols = [s.strip().upper() for s in portfolio_symbols if s.strip()]
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Single Stock Analysis", "Accumulation Patterns", "Distribution Patterns", "Sector Rotation", "Multi Stock Analysis"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            symbol = st.text_input("Enter Symbol", "SPY").strip().upper()
            lookback_days = st.slider("Lookback Days", 1, 30, 20)
        with col2:
            threshold = st.number_input("Buy/Sell Ratio Threshold", min_value=1.0, max_value=5.0, value=1.5, step=0.1)
        
        if st.button("Analyze Stock"):
            with st.spinner(f"Analyzing {symbol}..."):
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
                
                check_alerts(results_df, symbol)
                
                fig = px.line(results_df, x='date', y='buy_to_sell_ratio', title=f"{symbol} Buy/Sell Ratio Over Time",
                              hover_data=['total_volume', 'short_volume_ratio'])
                fig.add_hline(y=threshold, line_dash="dash", line_color="red", annotation_text=f"Threshold: {threshold}")
                st.plotly_chart(fig)
                
                st.subheader("Daily Analysis")
                def highlight_significant(row):
                    return ['background-color: rgba(144, 238, 144, 0.3)' if row['buy_to_sell_ratio'] > threshold else ''] * len(row)
                display_df = results_df.copy()
                for col in ['total_volume', 'bought_volume', 'sold_volume']:
                    display_df[col] = display_df[col].astype(int)
                styled_df = display_df.style.apply(highlight_significant, axis=1)
                st.dataframe(styled_df)
                
                csv = convert_df_to_csv(results_df)
                st.download_button("Download Results as CSV", csv, f"{symbol}_analysis.csv", "text/csv")
    
    with tab2:
        st.subheader("Top 40 Stocks Showing Accumulation")
        col1, col2 = st.columns(2)
        with col1:
            acc_min_volume = st.number_input("Minimum Daily Volume (Accumulation)", value=1000000, step=500000, format="%d", key="acc_vol")
        with col2:
            acc_use_validation = st.checkbox("Use Price Validation", value=False, key="acc_val")
        
        if st.button("Find Accumulation Patterns"):
            with st.spinner("Finding patterns..."):
                accumulation_df = find_patterns(lookback_days=10, min_volume=acc_min_volume, pattern_type="accumulation", use_price_validation=acc_use_validation)
            if not accumulation_df.empty:
                def highlight_acc_pattern(row):
                    return [f'background-color: rgba(144, 238, 144, 0.3)' if row['Volume Trend'] == 'Increasing' else ''] * len(row)
                styled_df = accumulation_df.style.apply(highlight_acc_pattern, axis=1)
                st.dataframe(styled_df)
                
                fig = px.bar(accumulation_df.head(10), x='Symbol', y='Avg Buy/Sell Ratio', 
                             color='Volume Trend', title="Top 10 Accumulation Patterns")
                st.plotly_chart(fig)
                
                csv = convert_df_to_csv(accumulation_df)
                st.download_button("Download Accumulation Patterns as CSV", csv, "accumulation_patterns.csv", "text/csv")
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
            with st.spinner("Finding patterns..."):
                distribution_df = find_patterns(lookback_days=10, min_volume=dist_min_volume, pattern_type="distribution", use_price_validation=dist_use_validation)
            if not distribution_df.empty:
                def highlight_dist_pattern(row):
                    return [f'background-color: rgba(255, 182, 193, 0.3)' if row['Volume Trend'] == 'Increasing' else ''] * len(row)
                styled_df = distribution_df.style.apply(highlight_dist_pattern, axis=1)
                st.dataframe(styled_df)
                
                fig = px.bar(distribution_df.head(10), x='Symbol', y='Avg Buy/Sell Ratio', 
                             color='Volume Trend', title="Top 10 Distribution Patterns")
                st.plotly_chart(fig)
                
                csv = convert_df_to_csv(distribution_df)
                st.download_button("Download Distribution Patterns as CSV", csv, "distribution_patterns.csv", "text/csv")
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
            with st.spinner("Analyzing sector rotation..."):
                buying_df, selling_df = get_sector_rotation(lookback_days=rot_lookback_days, min_volume=rot_min_volume)
            st.write("### Sectors Being Bought (Money In)")
            if not buying_df.empty:
                def highlight_buying(row):
                    return ['background-color: rgba(144, 238, 144, 0.3)'] * len(row)
                styled_buying = buying_df.style.apply(highlight_buying, axis=1)
                st.dataframe(styled_buying)
                
                fig = px.pie(buying_df, values='Avg Daily Volume', names='Sector', 
                             title="Buying Sectors by Volume")
                st.plotly_chart(fig)
                
                csv = convert_df_to_csv(buying_df)
                st.download_button("Download Buying Sectors as CSV", csv, "buying_sectors.csv", "text/csv")
            else:
                st.write("No sectors with clear buying trends detected.")
            
            st.write("### Sectors Being Sold (Money Out)")
            if not selling_df.empty:
                def highlight_selling(row):
                    return ['background-color: rgba(255, 182, 193, 0.3)'] * len(row)
                styled_selling = selling_df.style.apply(highlight_selling, axis=1)
                st.dataframe(styled_selling)
                
                csv = convert_df_to_csv(selling_df)
                st.download_button("Download Selling Sectors as CSV", csv, "selling_sectors.csv", "text/csv")
            else:
                st.write("No sectors with clear selling trends detected.")
    
    with tab5:
        st.subheader("Multi Stock Analysis")
        col1, col2 = st.columns(2)
        with col1:
            port_lookback_days = st.slider("Lookback Days (Portfolio)", 1, 30, 1, key="port_days")
        with col2:
            #st.write(f"Analyzing: {', '.join(portfolio_symbols)}")
            st.write(f"Analyzing: ")
        
        if st.button("Analyze Portfolio"):
            with st.spinner("Analyzing stocks..."):
                portfolio_df = analyze_portfolio(portfolio_symbols, port_lookback_days)
            
            if not portfolio_df.empty:
                # Assign sectors to each symbol
                def get_sector(symbol):
                    for sector, symbols in sector_mapping.items():
                        if symbol in symbols:
                            return sector
                    return 'User Defined'
                
                portfolio_df['Sector'] = portfolio_df['Symbol'].apply(get_sector)
                
                # Create tabs for each sector
                sector_tabs = st.tabs(list(sector_mapping.keys()))
                
                for tab, sector in zip(sector_tabs, sector_mapping.keys()):
                    with tab:
                        sector_df = portfolio_df[portfolio_df['Sector'] == sector]
                        if not sector_df.empty:
                            st.dataframe(sector_df)
                            
                            # Sector-specific summary
                            sector_bought = sector_df['Bought Volume'].sum()
                            sector_sold = sector_df['Sold Volume'].sum()
                            st.write(f"Total Bought Volume: {sector_bought:,}")
                            st.write(f"Total Sold Volume: {sector_sold:,}")
                            st.write("Dark Pools: " + 
                                   ("Bullish" if sector_bought > sector_sold else "Bearish"))
                            
                            # Sector visualization
                            fig = px.bar(sector_df, x='Symbol', y='Avg Buy/Sell Ratio',
                                       title=f"{sector} Buy/Sell Ratios",
                                       color='Significant Days')
                            st.plotly_chart(fig)
                            
                            # Export option for sector
                            csv = convert_df_to_csv(sector_df)
                            st.download_button(f"Download {sector} Analysis",
                                            csv, 
                                            f"portfolio_{sector.lower()}_analysis.csv",
                                            "text/csv")
                
                # Overall portfolio summary
                st.write("### Overall Portfolio Summary")
                total_bought_volume = portfolio_df['Bought Volume'].sum()
                total_sold_volume = portfolio_df['Sold Volume'].sum()
                st.write(f"Total Bought Volume: {total_bought_volume:,}")
                st.write(f"Total Sold Volume: {total_sold_volume:,}")
                st.write("Dark Pools: " + 
                       ("Bullish" if total_bought_volume > total_sold_volume else "Bearish"))
                
                # Overall visualization
                fig = px.bar(portfolio_df, x='Symbol', y='Avg Buy/Sell Ratio',
                            title="Portfolio Buy/Sell Ratios by Sector",
                            color='Sector')
                st.plotly_chart(fig)
                
                # Correlation Heatmap
                if st.button("Show Portfolio Correlation"):
                    plot_correlation_heatmap(portfolio_symbols, port_lookback_days)
                
                # Export full portfolio
                csv = convert_df_to_csv(portfolio_df)
                st.download_button("Download Full Multi Stock Analysis",
                                csv,
                                "portfolio_analysis.csv",
                                "text/csv")
            else:
                st.write("No data available for the selected portfolio.")

if __name__ == "__main__":
    run()
