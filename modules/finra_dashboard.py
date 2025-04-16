import pandas as pd
import streamlit as st
import requests
import io
from datetime import datetime, timedelta
import plotly.express as px
import logging
import sqlite3
from typing import Optional, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Output to console
    ]
)
logger = logging.getLogger(__name__)

# Theme mapping for tabs
theme_mapping = {
    "Indexes": [
        "SPY", "QQQ", "IWM", "DIA", "SMH"
    ],
    "Volatility": [
        "VXX", "VIXY", "UVXY"
    ],
    "Bonds": [
        "TLT", "IEF", "SHY", "LQD", "HYG", "AGG"
    ],
    "Commodities": [
        "SPY", "GLD", "SLV", "USO", "UNG", "DBA", "DBB", "DBC"
    ],
    "Metals": [
        "GLD", "SLV", "GDX", "GDXJ", "IAU", "SIVR"
    ],
    "Real Estate": [
        "VNQ", "IYR", "XHB", "XLF", "SPG", "PLD", "AMT", "DLR"
    ],
    "Consumer Discretionary": [
        "AMZN", "TSLA", "HD", "NKE", "MCD", "DIS", "LOW", "TGT", "LULU"
    ],
    "Consumer Staples": [
        "PG", "KO", "PEP", "WMT", "COST", "CL", "KMB", "MDLZ", "GIS"
    ],
    "Utilities": [
        "XLU", "DUK", "SO", "D", "NEE", "EXC", "AEP", "SRE", "ED"
    ],
    "Telecommunications": [
        "XLC", "T", "VZ", "TMUS", "S", "DISH", "LUMN", "VOD"
    ],
    "Materials": [
        "XLB", "XME", "XLI", "FCX", "NUE", "DD", "APD", "LIN", "IFF"
    ],
    "Transportation": [
        "UPS", "FDX", "DAL", "UAL", "LUV", "CSX", "NSC", "KSU", "WAB"
    ],
    "Aerospace & Defense": [
        "LMT", "BA", "NOC", "RTX", "GD", "HII", "LHX", "COL", "TXT"
    ],
    "Retail": [
        "AMZN", "WMT", "TGT", "COST", "HD", "LOW", "TJX", "M", "KSS"
    ],
    "Automotive": [
        "TSLA", "F", "GM", "RIVN", "LCID", "NIO", "XPEV", "BYDDF", "FCAU"
    ],
    "Pharmaceuticals": [
        "PFE", "MRK", "JNJ", "ABBV", "BMY", "GILD", "AMGN", "LLY", "VRTX"
    ],
    "Biotechnology": [
        "AMGN", "REGN", "ILMN", "VRTX", "CRSP", "MRNA", "BMRN", "ALNY",
        "SRPT", "EDIT", "NTLA", "BEAM", "BLUE", "FATE", "SANA"
    ],
    "Insurance": [
        "AIG", "PRU", "MET", "UNM", "LNC", "TRV", "CINF", "PGR", "ALL"
    ],
    "Technology": [
        "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD", 
        "ORCL", "CRM", "ADBE", "INTC", "CSCO", "QCOM", "TXN", "IBM", 
        "NOW", "AVGO", "INTU", "PANW", "SNOW"
    ],
    "Financials": [
        "JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "C", "AXP", "SCHW",
        "COF", "MET", "AIG", "BK", "BLK", "TFC", "USB", "PNC", "CME", "SPGI"
    ],
    "Healthcare": [
        "LLY", "UNH", "JNJ", "PFE", "MRK", "ABBV", "TMO", "AMGN", "GILD", 
        "CVS", "MDT", "BMY", "ABT", "DHR", "ISRG", "SYK", "REGN", "VRTX", 
        "CI", "ZTS"
    ],
    "Consumer": [
        "WMT", "PG", "KO", "PEP", "COST", "MCD", "DIS", "NKE", "SBUX", 
        "LOW", "TGT", "HD", "CL", "MO", "KHC", "PM", "TJX", "DG", "DLTR", "YUM"
    ],
    "Energy": [
        "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "OXY", "VLO", 
        "PXD", "HES", "WMB", "KMI", "OKE", "HAL", "BKR", "FANG", "DVN", 
        "TRGP", "APA"
    ],
    "Industrials": [
        "CAT", "DE", "UPS", "FDX", "BA", "HON", "UNP", "MMM", "GE", "LMT",
        "RTX", "GD", "CSX", "NSC", "WM", "ETN", "ITW", "EMR", "PH", "ROK"
    ],
    "Semiconductors": [
        "NVDA", "AMD", "QCOM", "TXN", "INTC", "AVGO", "ASML", "KLAC", 
        "LRCX", "AMAT", "ADI", "MCHP", "ON", "STM", "MPWR", "TER", "ENTG", 
        "SWKS", "QRVO", "LSCC"
    ],
    "Cybersecurity": [
        "CRWD", "PANW", "ZS", "FTNT", "S", "OKTA", "CYBR", "RPD", "NET",
        "QLYS", "TENB", "VRNS", "SPLK", "CHKP", "FEYE", "DDOG", "ESTC", 
        "FSLY", "MIME", "KNBE"
    ],
    "Quantum Computing": [
        "IBM", "GOOGL", "MSFT", "RGTI", "IONQ", "QUBT", "HON", "QCOM", 
        "INTC", "AMAT", "MKSI", "NTNX", "DWave", "XERI", "QTUM", "FORM", 
        "LMT", "BA", "NOC", "ACN"
    ],
    "Clean Energy": [
        "TSLA", "ENPH", "FSLR", "NEE", "PLUG", "SEDG", "RUN", "SHLS", 
        "ARRY", "NOVA", "BE", "BLDP", "FCEL", "CWEN", "NEP", "DTE", "AES", 
        "EIX", "SRE", "SPWR"
    ],
    "Artificial Intelligence": [
        "NVDA", "GOOGL", "MSFT", "AMD", "PLTR", "SNOW", "AI", "CRM", "IBM",
        "AAPL", "ADBE", "MSCI", "DELL", "BIDU", "UPST", "C3AI", "PATH", 
        "SOUN", "VRNT", "ANSS"
    ],
    "Biotechnology": [
        "MRNA", "CRSP", "VRTX", "REGN", "ILMN", "AMGN", "NBIX", "BIIB", 
        "INCY", "GILD", "BMRN", "ALNY", "SRPT", "BEAM", "NTLA", "EDIT", 
        "BLUE", "SANA", "FATE", "KRYS"
    ]
}

def download_finra_short_sale_data(date: str) -> Optional[str]:
    url = f"https://cdn.finra.org/equity/regsho/daily/CNMSshvol{date}.txt"
    response = requests.get(url)
    return response.text if response.status_code == 200 else None

def process_finra_short_sale_data(data: Optional[str]) -> pd.DataFrame:
    if not data:
        return pd.DataFrame()
    df = pd.read_csv(io.StringIO(data), delimiter="|")
    return df[df["Symbol"].str.len() <= 4]

def calculate_metrics(row: pd.Series, total_volume: float) -> dict:
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

@st.cache_data(ttl=11520)
def analyze_symbol(symbol: str, lookback_days: int = 20, threshold: float = 1.5) -> tuple[pd.DataFrame, int]:
    results = []
    significant_days = 0
    cumulative_bought = 0
    cumulative_sold = 0
    
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
                
                cumulative_bought += metrics['bought_volume']
                cumulative_sold += metrics['sold_volume']
                
                if metrics['buy_to_sell_ratio'] > threshold:
                    significant_days += 1
                
                metrics['date'] = date
                metrics['cumulative_bought'] = cumulative_bought
                metrics['cumulative_sold'] = cumulative_sold
                results.append(metrics)
    
    df_results = pd.DataFrame(results)
    if not df_results.empty:
        df_results['date'] = pd.to_datetime(df_results['date'], format='%Y%m%d')
        df_results = df_results.sort_values('date', ascending=False)
    
    return df_results, significant_days

def get_latest_data(symbols: List[str] = None) -> tuple[pd.DataFrame, Optional[str]]:
    for i in range(7):  # Check the last 7 days
        date = (datetime.now() - timedelta(days=i)).strftime("%Y%m%d")
        data = download_finra_short_sale_data(date)
        if data:
            df = process_finra_short_sale_data(data)
            if not df.empty:
                if symbols:
                    df = df[df['Symbol'].isin(symbols)]
                return df, date
    return pd.DataFrame(), None

def generate_recommendations() -> tuple[pd.DataFrame, pd.DataFrame, Optional[str]]:
    """
    Analyze FINRA data across themes to recommend top 5 long and short candidates.
    Ensures no overlap between long and short candidates.
    Returns two DataFrames (longs, shorts) and the date of the data.
    """
    # Get unique symbols across all themes and assign a single theme per symbol
    symbol_themes = {}
    for theme, symbols in theme_mapping.items():
        for symbol in symbols:
            if symbol not in symbol_themes:  # Assign to the first theme encountered
                symbol_themes[symbol] = theme
    
    all_symbols = list(symbol_themes.keys())
    
    # Fetch latest data
    latest_df, latest_date = get_latest_data(all_symbols)
    if latest_df.empty or latest_date is None:
        return pd.DataFrame(), pd.DataFrame(), None
    
    # Calculate metrics for each stock
    metrics_list = []
    for _, row in latest_df.iterrows():
        total_volume = row.get('TotalVolume', 0)
        metrics = calculate_metrics(row, total_volume)
        metrics['Symbol'] = row['Symbol']
        metrics['TotalVolume'] = total_volume
        metrics_list.append(metrics)
    
    df = pd.DataFrame(metrics_list)
    
    # Remove duplicates by symbol, keeping the first occurrence
    df = df.drop_duplicates(subset=['Symbol'], keep='first')
    
    # Calculate theme strength
    theme_factors = {}
    for theme, symbols in theme_mapping.items():
        theme_data = df[df['Symbol'].isin(symbols)]
        if not theme_data.empty:
            total_bought = theme_data['bought_volume'].sum()
            total_sold = theme_data['sold_volume'].sum()
            bullish_factor = min(total_bought / (total_sold + 1), 2.0)  # Avoid division by zero, cap at 2
            bearish_factor = min(total_sold / (total_bought + 1), 2.0)
            theme_factors[theme] = {'bullish': bullish_factor, 'bearish': bearish_factor}
        else:
            theme_factors[theme] = {'bullish': 1.0, 'bearish': 1.0}  # Neutral if no data
    
    # Assign themes to symbols
    df['Theme'] = df['Symbol'].map(symbol_themes)
    df['buy_to_sell_ratio'] = df['buy_to_sell_ratio'].replace(float('inf'), 5.0)  # Cap infinite ratios
    df['norm_ratio'] = df['buy_to_sell_ratio'].clip(upper=5.0)  # Normalize for scoring
    df['buy_volume_ratio'] = df['bought_volume'] / df['total_volume']
    df['sell_volume_ratio'] = df['sold_volume'] / df['total_volume']
    df['theme_bullish_factor'] = df['Theme'].map(lambda x: theme_factors.get(x, {'bullish': 1.0})['bullish'])
    df['theme_bearish_factor'] = df['Theme'].map(lambda x: theme_factors.get(x, {'bearish': 1.0})['bearish'])
    
    # Long score: High buy/sell ratio, high buy volume, strong theme
    df['long_score'] = (
        (df['norm_ratio'] * 0.5) +
        (df['buy_volume_ratio'] * 0.3) +
        (df['theme_bullish_factor'] * 0.2)
    ) * (df['total_volume'] / df['total_volume'].max())  # Weight by volume
    
    # Short score: Low buy/sell ratio (high inverse), high sell volume, weak theme
    df['short_score'] = (
        ((1 / df['norm_ratio'].replace(0, 0.1)) * 0.5) +
        (df['sell_volume_ratio'] * 0.3) +
        (df['theme_bearish_factor'] * 0.2)
    ) * (df['total_volume'] / df['total_volume'].max())  # Weight by volume
    
    # Add a preference score to penalize stocks that could fit both categories
    df['preference_score'] = df['long_score'] - df['short_score']  # Positive means better for long, negative for short
    
    # Select top candidates with diversity constraint and mutual exclusion
    def select_diverse_candidates(df, score_column, top_n=5, max_per_theme=2, exclude_symbols=None):
        if exclude_symbols is None:
            exclude_symbols = set()
        df_filtered = df[~df['Symbol'].isin(exclude_symbols)]
        df_sorted = df_filtered.sort_values(by=score_column, ascending=False)
        selected = []
        theme_count = {}
        
        for _, row in df_sorted.iterrows():
            theme = row['Theme']
            if theme not in theme_count:
                theme_count[theme] = 0
            if theme_count[theme] < max_per_theme and len(selected) < top_n:
                selected.append(row)
                theme_count[theme] += 1
        
        return pd.DataFrame(selected)
    
    # Select long candidates first
    long_candidates = select_diverse_candidates(df, 'long_score', top_n=5, max_per_theme=2)
    
    # Exclude long candidates from short selection
    long_symbols = set(long_candidates['Symbol'].values) if not long_candidates.empty else set()
    short_candidates = select_diverse_candidates(df, 'short_score', top_n=5, max_per_theme=2, exclude_symbols=long_symbols)
    
    # Format output
    columns = ['Symbol', 'Theme', 'buy_to_sell_ratio', 'bought_volume', 'sold_volume', 'total_volume', 'long_score']
    long_df = long_candidates[columns].copy() if not long_candidates.empty else pd.DataFrame(columns=columns)
    columns = ['Symbol', 'Theme', 'buy_to_sell_ratio', 'bought_volume', 'sold_volume', 'total_volume', 'short_score']
    short_df = short_candidates[columns].copy() if not short_candidates.empty else pd.DataFrame(columns=columns)
    
    # Round scores and volumes
    if not long_df.empty:
        long_df['long_score'] = long_df['long_score'].round(4)
        for col in ['bought_volume', 'sold_volume', 'total_volume']:
            long_df[col] = long_df[col].astype(int)
    if not short_df.empty:
        short_df['short_score'] = short_df['short_score'].round(4)
        for col in ['bought_volume', 'sold_volume', 'total_volume']:
            short_df[col] = short_df[col].astype(int)
    
    return long_df, short_df, latest_date

def generate_theme_summary() -> tuple[pd.DataFrame, pd.DataFrame, Optional[str]]:
    """
    Analyze FINRA data to determine bullish and bearish themes.
    Returns two DataFrames (bullish themes, bearish themes) and the date of the data.
    """
    # Get unique symbols across all themes and assign a single theme per symbol
    symbol_themes = {}
    for theme, symbols in theme_mapping.items():
        for symbol in symbols:
            if symbol not in symbol_themes:  # Assign to the first theme encountered
                symbol_themes[symbol] = theme
    
    all_symbols = list(symbol_themes.keys())
    
    # Fetch latest data
    latest_df, latest_date = get_latest_data(all_symbols)
    if latest_df.empty or latest_date is None:
        return pd.DataFrame(), pd.DataFrame(), None
    
    # Calculate metrics for each stock
    metrics_list = []
    for _, row in latest_df.iterrows():
        total_volume = row.get('TotalVolume', 0)
        metrics = calculate_metrics(row, total_volume)
        metrics['Symbol'] = row['Symbol']
        metrics['TotalVolume'] = total_volume
        metrics_list.append(metrics)
    
    df = pd.DataFrame(metrics_list)
    
    # Remove duplicates by symbol, keeping the first occurrence
    df = df.drop_duplicates(subset=['Symbol'], keep='first')
    
    # Assign themes to symbols
    df['Theme'] = df['Symbol'].map(symbol_themes)
    
    # Calculate metrics per theme
    theme_summary = []
    for theme in theme_mapping.keys():
        theme_data = df[df['Theme'] == theme]
        if not theme_data.empty:
            total_bought = theme_data['bought_volume'].sum()
            total_sold = theme_data['sold_volume'].sum()
            total_volume = theme_data['total_volume'].sum()
            avg_buy_sell_ratio = theme_data['buy_to_sell_ratio'].mean()
            num_stocks = len(theme_data)
            
            # Calculate bought-to-sold ratio
            bought_to_sold_ratio = total_bought / (total_sold + 1)  # Avoid division by zero
            sold_to_bought_ratio = total_sold / (total_bought + 1)  # For bearish sorting
            
            theme_summary.append({
                'Theme': theme,
                'Total Bought Volume': int(total_bought),
                'Total Sold Volume': int(total_sold),
                'Bought-to-Sold Ratio': round(bought_to_sold_ratio, 2),
                'Sold-to-Bought Ratio': round(sold_to_bought_ratio, 2),
                'Average Buy/Sell Ratio': round(avg_buy_sell_ratio, 2),
                'Total Volume': int(total_volume),
                'Number of Stocks': num_stocks
            })
    
    summary_df = pd.DataFrame(theme_summary)
    
    # Split into bullish and bearish themes
    bullish_df = summary_df[summary_df['Bought-to-Sold Ratio'] > 1].sort_values(by='Bought-to-Sold Ratio', ascending=False)
    bearish_df = summary_df[summary_df['Bought-to-Sold Ratio'] <= 1].sort_values(by='Sold-to-Bought Ratio', ascending=False)
    
    return bullish_df, bearish_df, latest_date

def check_alerts(df_results: pd.DataFrame, symbol: str, threshold: float = 2.0) -> None:
    if not df_results.empty and df_results['buy_to_sell_ratio'].max() > threshold:
        st.warning(f"Alert: {symbol} has a Buy/Sell Ratio above {threshold} on {df_results['date'].iloc[0].strftime('%Y-%m-%d')}!")

# Database functions
def setup_stock_database() -> None:
    conn = sqlite3.connect('stock_data.db')
    cursor = conn.cursor()
    
    cursor.execute('DROP TABLE IF EXISTS stocks')
    logger.info("Dropped existing `stocks` table (if it existed).")
    
    cursor.execute('''
        CREATE TABLE stocks (
            symbol TEXT PRIMARY KEY,
            price REAL,
            market_cap REAL,
            last_updated TEXT
        )
    ''')
    logger.info("Created `stocks` table with correct schema.")
    
    conn.commit()
    conn.close()

def check_and_setup_database() -> None:
    try:
        conn = sqlite3.connect('stock_data.db')
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='stocks'")
        if not cursor.fetchone():
            conn.close()
            setup_stock_database()
            return
        
        cursor.execute("PRAGMA table_info(stocks)")
        columns = [info[1] for info in cursor.fetchall()]
        if not all(col in columns for col in ['symbol', 'price', 'market_cap', 'last_updated']):
            conn.close()
            setup_stock_database()
            return
        
        conn.close()
    except Exception as e:
        logger.error(f"Error checking database: {e}")
        setup_stock_database()

check_and_setup_database()

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
    
    # Initialize session state
    if 'analysis_results' not in st.session_state:
        st.session_state['analysis_results'] = {}
    if 'recommendations' not in st.session_state:
        st.session_state['recommendations'] = None
    if 'theme_summary' not in st.session_state:
        st.session_state['theme_summary'] = None
    
    # with st.sidebar:
    #     portfolio_symbols = st.text_area("User Defined Symbols (comma-separated)", 
    #                                    "AAPL, MSFT, NVDA, AMZN, GOOGL, META, TSLA, JPM, V, MA, BAC, LLY, UNH, JNJ").split(",")
    #     portfolio_symbols = [s.strip().upper() for s in portfolio_symbols if s.strip()]
        
    #     st.subheader("Data Management")
    #     update_button = st.button("Update Stock Data", use_container_width=True)
    #     if update_button:
    #         with st.spinner("Updating stock data..."):
    #             logger.info("Stock data update triggered (function not implemented).")
    #         st.success("Stock data updated successfully!")
    
    # Create tabs: Recommendations + Theme Summary + Single Stock + Theme-based tabs
    tab_names = ["Recommendations", "Theme Summary", "Single Stock"] + list(theme_mapping.keys())
    tabs = st.tabs(tab_names)
    
    # Recommendations Tab
    with tabs[0]:
        st.subheader("Top Long and Short Recommendations")
        st.write("Based on a holistic analysis of FINRA short sale data across all themes.")
        
        if st.button("Generate Recommendations"):
            with st.spinner("Analyzing data for recommendations..."):
                long_df, short_df, latest_date = generate_recommendations()
                st.session_state['recommendations'] = {'long_df': long_df, 'short_df': short_df, 'date': latest_date}
        
        if st.session_state['recommendations']:
            recommendations = st.session_state['recommendations']
            long_df = recommendations['long_df']
            short_df = recommendations['short_df']
            latest_date = recommendations['date']
            
            if latest_date:
                st.write(f"Data analyzed for: {latest_date}")
            
            # Long Candidates
            st.write("### Top 5 Long Candidates")
            if not long_df.empty:
                def highlight_long(row):
                    return ['background-color: rgba(144, 238, 144, 0.3)'] * len(row)
                
                st.dataframe(long_df.style.apply(highlight_long, axis=1).format({
                    'buy_to_sell_ratio': '{:.2f}',
                    'bought_volume': '{:,.0f}',
                    'sold_volume': '{:,.0f}',
                    'total_volume': '{:,.0f}',
                    'long_score': '{:.4f}'
                }))
            else:
                st.write("No long candidates found with current data.")
            
            # Short Candidates
            st.write("### Top 5 Short Candidates")
            if not short_df.empty:
                def highlight_short(row):
                    return ['background-color: rgba(255, 182, 193, 0.3)'] * len(row)
                
                st.dataframe(short_df.style.apply(highlight_short, axis=1).format({
                    'buy_to_sell_ratio': '{:.2f}',
                    'bought_volume': '{:,.0f}',
                    'sold_volume': '{:,.0f}',
                    'total_volume': '{:,.0f}',
                    'short_score': '{:.4f}'
                }))
            else:
                st.write("No short candidates found with current data.")
            
            # Visualization
            if not long_df.empty or not short_df.empty:
                # Combine data for plotting
                plot_data = []
                if not long_df.empty:
                    long_plot = long_df[['Symbol', 'long_score']].copy()
                    long_plot['Type'] = 'Long'
                    long_plot['Score'] = long_plot['long_score']
                    plot_data.append(long_plot[['Symbol', 'Score', 'Type']])
                if not short_df.empty:
                    short_plot = short_df[['Symbol', 'short_score']].copy()
                    short_plot['Type'] = 'Short'
                    short_plot['Score'] = short_plot['short_score']
                    plot_data.append(short_plot[['Symbol', 'Score', 'Type']])
                
                if plot_data:
                    plot_df = pd.concat(plot_data, ignore_index=True)
                    fig = px.bar(plot_df, x='Symbol', y='Score', color='Type',
                                title="Top Long and Short Candidates by Score",
                                color_discrete_map={'Long': '#4CAF50', 'Short': '#F44336'},
                                hover_data=['Symbol', 'Score'])
                    fig.update_layout(barmode='group', xaxis_tickangle=-45)
                    st.plotly_chart(fig)
    
    # Theme Summary Tab
    with tabs[1]:
        st.subheader("Theme Sentiment Summary")
        st.write("Analysis of overall bullish and bearish themes based on FINRA short sale data.")
        
        if st.button("Generate Theme Summary"):
            with st.spinner("Analyzing theme sentiment..."):
                bullish_df, bearish_df, latest_date = generate_theme_summary()
                st.session_state['theme_summary'] = {'bullish_df': bullish_df, 'bearish_df': bearish_df, 'date': latest_date}
        
        if st.session_state['theme_summary']:
            theme_summary = st.session_state['theme_summary']
            bullish_df = theme_summary['bullish_df']
            bearish_df = theme_summary['bearish_df']
            latest_date = theme_summary['date']
            
            if latest_date:
                st.write(f"Data analyzed for: {latest_date}")
            
            # Bullish Themes
            st.write("### Bullish Themes (Bought Volume > Sold Volume)")
            if not bullish_df.empty:
                def highlight_bullish(row):
                    return ['background-color: rgba(144, 238, 144, 0.3)'] * len(row)
                
                st.dataframe(bullish_df.style.apply(highlight_bullish, axis=1).format({
                    'Bought-to-Sold Ratio': '{:.2f}',
                    'Average Buy/Sell Ratio': '{:.2f}',
                    'Total Bought Volume': '{:,.0f}',
                    'Total Sold Volume': '{:,.0f}',
                    'Total Volume': '{:,.0f}',
                    'Number of Stocks': '{:d}'
                }))
            else:
                st.write("No bullish themes found with current data.")
            
            # Bearish Themes
            st.write("### Bearish Themes (Sold Volume >= Bought Volume)")
            if not bearish_df.empty:
                def highlight_bearish(row):
                    return ['background-color: rgba(255, 182, 193, 0.3)'] * len(row)
                
                st.dataframe(bearish_df.style.apply(highlight_bearish, axis=1).format({
                    'Bought-to-Sold Ratio': '{:.2f}',
                    'Average Buy/Sell Ratio': '{:.2f}',
                    'Total Bought Volume': '{:,.0f}',
                    'Total Sold Volume': '{:,.0f}',
                    'Total Volume': '{:,.0f}',
                    'Number of Stocks': '{:d}'
                }))
            else:
                st.write("No bearish themes found with current data.")
            
            # Visualization
            if not bullish_df.empty or not bearish_df.empty:
                plot_data = []
                if not bullish_df.empty:
                    bullish_plot = bullish_df[['Theme', 'Bought-to-Sold Ratio']].copy()
                    bullish_plot['Sentiment'] = 'Bullish'
                    bullish_plot['Ratio'] = bullish_plot['Bought-to-Sold Ratio']
                    plot_data.append(bullish_plot[['Theme', 'Ratio', 'Sentiment']])
                if not bearish_df.empty:
                    bearish_plot = bearish_df[['Theme', 'Bought-to-Sold Ratio']].copy()
                    bearish_plot['Sentiment'] = 'Bearish'
                    bearish_plot['Ratio'] = bearish_plot['Bought-to-Sold Ratio']
                    plot_data.append(bearish_plot[['Theme', 'Ratio', 'Sentiment']])
                
                if plot_data:
                    plot_df = pd.concat(plot_data, ignore_index=True)
                    fig = px.bar(plot_df, x='Theme', y='Ratio', color='Sentiment',
                                title="Theme Sentiment by Bought-to-Sold Volume Ratio",
                                color_discrete_map={'Bullish': '#4CAF50', 'Bearish': '#F44336'},
                                hover_data=['Theme', 'Ratio'])
                    fig.update_layout(barmode='group', xaxis_tickangle=-45)
                    fig.add_hline(y=1.0, line_dash="dash", line_color="gray", annotation_text="Neutral Line")
                    st.plotly_chart(fig)
    
    # Single Stock Tab
    with tabs[2]:
        col1, col2 = st.columns(2)
        with col1:
            symbol = st.text_input("Enter Symbol", "SPY").strip().upper()
            lookback_days = st.slider("Lookback Days", 1, 30, 20)
        with col2:
            threshold = st.number_input("Buy/Sell Ratio Threshold", min_value=1.0, max_value=5.0, value=1.5, step=0.1)
        
        if st.button("Analyze Stock"):
            with st.spinner(f"Analyzing {symbol}..."):
                results_df, significant_days = analyze_symbol(symbol, lookback_days, threshold)
                st.session_state['analysis_results'][symbol] = {'df': results_df, 'significant_days': significant_days}
        
        if symbol in st.session_state['analysis_results']:
            results_df = st.session_state['analysis_results'][symbol]['df']
            significant_days = st.session_state['analysis_results'][symbol]['significant_days']
            
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
                              hover_data=['total_volume', 'short_volume_ratio', 'cumulative_bought', 'cumulative_sold'])
                fig.add_hline(y=threshold, line_dash="dash", line_color="red", annotation_text=f"Threshold: {threshold}")
                st.plotly_chart(fig)
                
                st.subheader("Daily Analysis")
                display_df = results_df.copy()
                for col in ['total_volume', 'bought_volume', 'sold_volume', 'cumulative_bought', 'cumulative_sold']:
                    display_df[col] = display_df[col].astype(int)
                
                # Highlight rows based on cumulative_bought vs cumulative_sold
                def highlight_row(row):
                    color = 'background-color: rgba(144, 238, 144, 0.3)' if row['cumulative_bought'] > row['cumulative_sold'] else 'background-color: rgba(255, 182, 193, 0.3)'
                    return [color] * len(row)
                
                st.dataframe(display_df.style.apply(highlight_row, axis=1))
            else:
                st.write(f"No data available for {symbol}.")
    
    # Theme-based Tabs
        # Theme-based Tabs
    for i, theme in enumerate(theme_mapping.keys(), 3):
        with tabs[i]:
            st.subheader(f"{theme} - Latest Day FINRA Data")
            symbols = theme_mapping[theme]
            
            # Fetch latest data for the theme's symbols
            latest_df, latest_date = get_latest_data(symbols)
            
            if latest_df.empty or latest_date is None:
                st.write(f"No data available for {theme} stocks on the latest day.")
                continue
            
            st.write(f"Showing data for: {latest_date}")
            
            # Process the data
            metrics_list = []
            for _, row in latest_df.iterrows():
                total_volume = row.get('TotalVolume', 0)
                metrics = calculate_metrics(row, total_volume)
                metrics['Symbol'] = row['Symbol']
                metrics['TotalVolume'] = total_volume
                metrics_list.append(metrics)
            
            theme_df = pd.DataFrame(metrics_list)
            theme_df = theme_df.sort_values(by=['buy_to_sell_ratio', 'total_volume'], ascending=[False, False])
            
            if not theme_df.empty:
                # Prepare display DataFrame
                display_df = theme_df.copy()
                for col in ['total_volume', 'bought_volume', 'sold_volume']:
                    display_df[col] = display_df[col].astype(int)
                
                # Calculate cumulative bought and sold for display (not for highlighting)
                display_df['cumulative_bought'] = display_df['bought_volume'].cumsum()
                display_df['cumulative_sold'] = display_df['sold_volume'].cumsum()
                
                # Highlight rows based on individual stock's bought_volume vs sold_volume
                def highlight_row(row):
                    color = 'background-color: rgba(144, 238, 144, 0.3)' if row['bought_volume'] > row['sold_volume'] else 'background-color: rgba(255, 182, 193, 0.3)'
                    return [color] * len(row)
                
                # Display the table
                column_order = ['Symbol', 'buy_to_sell_ratio', 'total_volume', 'bought_volume', 'sold_volume', 'short_volume_ratio', 'cumulative_bought', 'cumulative_sold']
                st.dataframe(display_df[column_order].style.apply(highlight_row, axis=1).format({
                    'buy_to_sell_ratio': '{:.2f}',
                    'short_volume_ratio': '{:.4f}',
                    'total_volume': '{:,.0f}',
                    'bought_volume': '{:,.0f}',
                    'sold_volume': '{:,.0f}',
                    'cumulative_bought': '{:,.0f}',
                    'cumulative_sold': '{:,.0f}'
                }))
                
                # Calculate and display summary
                total_bought_volume = theme_df['bought_volume'].sum()
                total_sold_volume = theme_df['sold_volume'].sum()
                st.write("### Summary")
                st.write(f"Total Bought Volume: {total_bought_volume:,.0f}")
                st.write(f"Total Sold Volume: {total_sold_volume:,.0f}")
                st.write(f"Dark Pools: {'Bullish' if total_bought_volume > total_sold_volume else 'Bearish'}")
                
                # Visualization
                fig = px.bar(theme_df, x='Symbol', y='buy_to_sell_ratio',
                            title=f"{theme} Buy/Sell Ratios for {latest_date}",
                            hover_data=['total_volume', 'bought_volume', 'sold_volume'])
                fig.update_layout(barmode='group', xaxis_tickangle=-45)
                st.plotly_chart(fig)
            else:
                st.write(f"No records found for {theme} stocks on {latest_date}.")

if __name__ == "__main__":
    run()
