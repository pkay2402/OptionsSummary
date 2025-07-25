import pandas as pd
import requests
from io import StringIO
from datetime import datetime, timedelta
import logging
import sqlite3
from typing import List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Theme mapping
theme_mapping = {
    "Indexes": [
        "SPY", "QQQ", "IWM", "DIA", "SMH"
    ],
    "Bull Leverage ETF": [
        "SPXL", "UPRO", "TQQQ", "SOXL", "UDOW", "FAS", "SPUU", "TNA"
    ],
    "Bear Leverage ETF": [
        "SQQQ", "SPXS", "SOXS", "SDOW", "FAZ","SPDN", "TZA", "SPXU"
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
    "Nuclear Power": [
        "CEG", "NNE", "GEV", "OKLO", "UUUU", "ASPI", "CCJ"
    ],
    "Crypto": [
        "IBIT", "FBTC", "MSTR", "COIN", "HOOD", "ETHU"
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
        "XLC", "T", "VZ", "TMUS", "S", "LUMN", "VOD"
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
        "XLE", "HES", "WMB", "KMI", "OKE", "HAL", "BKR", "FANG", "DVN",
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
        "INTC", "AMAT", "MKSI", "NTNX", "XERI", "QTUM", "FORM",
        "LMT", "BA", "NOC", "ACN"
    ],
    "Clean Energy": [
        "TSLA", "ENPH", "FSLR", "NEE", "PLUG", "SEDG", "RUN", "SHLS",
        "ARRY", "NOVA", "BE", "BLDP", "FCEL", "CWEN", "DTE", "AES",
        "EIX", "SRE"
    ],
    "Artificial Intelligence": [
        "NVDA", "GOOGL", "MSFT", "AMD", "PLTR", "SNOW", "AI", "CRM", "IBM",
        "AAPL", "ADBE", "MSCI", "DELL", "BIDU", "UPST", "AI", "PATH",
        "SOUN", "VRNT", "ANSS"
    ],
    "Biotechnology": [
        "MRNA", "CRSP", "VRTX", "REGN", "ILMN", "AMGN", "NBIX", "BIIB",
        "INCY", "GILD", "BMRN", "ALNY", "SRPT", "BEAM", "NTLA", "EDIT",
        "BLUE", "SANA", "VKTX", "KRYS"
    ]
}

def setup_option_database() -> None:
    conn = sqlite3.connect('option_sentiment.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sentiments (
            symbol TEXT,
            date DATE,
            call_volume REAL,
            put_volume REAL,
            net_sentiment REAL,
            PRIMARY KEY (symbol, date)
        )
    ''')
    conn.commit()
    conn.close()

setup_option_database()

def validate_csv_content_type(response: requests.Response) -> bool:
    return 'text/csv' in response.headers.get('Content-Type', '')

def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df['Volume'] >= 100]
    df['Expiration'] = pd.to_datetime(df['Expiration'])
    df = df[df['Expiration'].dt.date >= datetime.now().date()]
    return df

def fetch_data_from_url(url: str) -> Optional[pd.DataFrame]:
    try:
        response = requests.get(url)
        response.raise_for_status()
        if validate_csv_content_type(response):
            csv_data = StringIO(response.text)
            df = pd.read_csv(csv_data)
            return apply_filters(df)
        else:
            logger.warning(f"Data from {url} is not in CSV format. Skipping...")
    except Exception as e:
        logger.error(f"Error fetching or processing data from {url}: {e}")
    return None

def fetch_data_from_urls(urls: List[str]) -> pd.DataFrame:
    data_frames = []
    for url in urls:
        df = fetch_data_from_url(url)
        if df is not None:
            data_frames.append(df)
    return pd.concat(data_frames, ignore_index=True) if data_frames else pd.DataFrame()

def calculate_and_store_sentiments(df: pd.DataFrame) -> None:
    conn = sqlite3.connect('option_sentiment.db')
    today = datetime.now().date().strftime('%Y-%m-%d')
    grouped = df.groupby(['Symbol', 'Call/Put'])['Volume'].sum().reset_index()
    symbols = grouped['Symbol'].unique()
    for symbol in symbols:
        call_vol = grouped[(grouped['Symbol'] == symbol) & (grouped['Call/Put'] == 'C')]['Volume'].sum()
        put_vol = grouped[(grouped['Symbol'] == symbol) & (grouped['Call/Put'] == 'P')]['Volume'].sum()
        total = call_vol + put_vol
        if total > 0:
            net = (call_vol - put_vol) / total
        else:
            net = 0
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO sentiments (symbol, date, call_volume, put_volume, net_sentiment)
            VALUES (?, ?, ?, ?, ?)
        ''', (symbol, today, call_vol, put_vol, net))
    conn.commit()
    conn.close()

def get_historical_sentiment(symbol: str, lookback_days: int = 20) -> pd.DataFrame:
    conn = sqlite3.connect('option_sentiment.db')
    start_date = (datetime.now().date() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
    df = pd.read_sql_query(
        "SELECT * FROM sentiments WHERE symbol = ? AND date >= ? ORDER BY date DESC",
        conn,
        params=(symbol, start_date)
    )
    conn.close()
    return df

def get_today_sentiments() -> pd.DataFrame:
    conn = sqlite3.connect('option_sentiment.db')
    today = datetime.now().date().strftime('%Y-%m-%d')
    df = pd.read_sql_query(
        "SELECT * FROM sentiments WHERE date = ?",
        conn,
        params=(today,)
    )
    conn.close()
    return df

def get_signal(ratio: float) -> str:
    if ratio > 1.5:
        return 'Buy'
    elif ratio > 1.0:
        return 'Add'
    else:
        return 'Trim'

def style_signal(val: str):
    color = 'green' if val in ['Buy', 'Add'] else 'red'
    return f'color: {color}; font-weight: bold'

def run():
    import streamlit as st

    st.markdown("""
        <style>
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 5px;
            padding: 8px 16px;
        }
        .stTabs [data-baseweb="tab"] {
            font-size: 16px;
            padding: 10px;
        }
        .stSelectbox {
            max-width: 300px;
        }
        .stDataFrame {
            font-size: 14px;
        }
        </style>
        """, unsafe_allow_html=True)
    
    st.title("Options Flow Analysis")
    
    urls = [
        "https://www.cboe.com/us/options/market_statistics/symbol_data/csv/?mkt=cone",
        "https://www.cboe.com/us/options/market_statistics/symbol_data/csv/?mkt=opt",
        "https://www.cboe.com/us/options/market_statistics/symbol_data/csv/?mkt=ctwo",
        "https://www.cboe.com/us/options/market_statistics/symbol_data/csv/?mkt=exo"
    ]
    
    with st.spinner("Fetching data..."):
        data = fetch_data_from_urls(urls)
        if not data.empty:
            calculate_and_store_sentiments(data)
    
    tabs = st.tabs(["Single Stock", "Stock Summary", "Watchlist Summary"])
    
    with tabs[0]:
        st.subheader("Single Stock Analysis")
        col1, col2 = st.columns(2)
        with col1:
            symbol = st.text_input("Enter Symbol", "NVDA").strip().upper()
            lookback_days = st.slider("Lookback Days", 1, 30, 20)
        if st.button("Analyze Stock"):
            with st.spinner(f"Analyzing {symbol}..."):
                df = get_historical_sentiment(symbol, lookback_days)
                if not df.empty:
                    df['total_volume'] = df['call_volume'] + df['put_volume']
                    df['ratio'] = df['call_volume'] / df['put_volume'].replace(0, 1)
                    df['ratio'] = df['ratio'].replace(float('inf'), 5.0).clip(upper=5.0)
                    df['pct_avg'] = (df['call_volume'] / df['total_volume'] * 100).round(0).astype(int)
                    df['bot_pct'] = df['pct_avg']
                    df['signal'] = df['ratio'].apply(get_signal)
                    
                    total_call = df['call_volume'].sum()
                    total_put = df['put_volume'].sum()
                    total_vol = total_call + total_put
                    aggregate_ratio = total_call / total_put if total_put > 0 else 5.0
                    
                    st.subheader("Summary Metrics")
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Total Call Volume", f"{total_call:,.0f}")
                    col2.metric("Total Put Volume", f"{total_put:,.0f}")
                    col3.metric("Aggregate Ratio", f"{aggregate_ratio:.2f}")
                    
                    display_df = df.copy()
                    display_df['call_volume'] = display_df['call_volume'].apply(lambda x: f"{x:,.0f}")
                    display_df['put_volume'] = display_df['put_volume'].apply(lambda x: f"{x:,.0f}")
                    display_df['total_volume'] = display_df['total_volume'].apply(lambda x: f"{x:,.0f}")
                    display_df['pct_avg'] = display_df['pct_avg'].apply(lambda x: f"{x}%")
                    display_df['bot_pct'] = display_df['bot_pct'].apply(lambda x: f"{x}%")
                    display_df['ratio'] = display_df['ratio'].round(2)
                    
                    columns = ['date', 'call_volume', 'put_volume', 'pct_avg', 'ratio', 'bot_pct', 'total_volume', 'signal']
                    st.dataframe(display_df[columns].style.applymap(style_signal, subset=['signal']))
                else:
                    st.write(f"No data available for {symbol}.")
    
    with tabs[1]:
        st.subheader("Stock Summary")
        default_excluded = ["SPX", "SPXW", "VIX", "SPY"]
        excluded_symbols = st.text_input(
            "Exclude Symbols (comma-separated)",
            value=", ".join(default_excluded)
        )
        excluded_symbols = [s.strip() for s in excluded_symbols.split(",") if s.strip()]
        
        if st.button("Generate Stock Summary"):
            with st.spinner("Analyzing..."):
                df = get_today_sentiments()
                if not df.empty:
                    if excluded_symbols:
                        df = df[~df['symbol'].isin(excluded_symbols)]
                    df['total_volume'] = df['call_volume'] + df['put_volume']
                    df['ratio'] = df['call_volume'] / df['put_volume'].replace(0, 1)
                    df['ratio'] = df['ratio'].replace(float('inf'), 5.0).clip(upper=5.0)
                    
                    high_buy_df = df[df['ratio'] > 2].copy()
                    high_sell_df = df[df['ratio'] < 0.5].copy()
                    
                    for d in [high_buy_df, high_sell_df]:
                        d['pct_avg'] = (d['call_volume'] / d['total_volume'] * 100).round(0).astype(int)
                        d['bot_pct'] = d['pct_avg']
                        d['signal'] = d['ratio'].apply(get_signal)
                        d['call_volume'] = d['call_volume'].apply(lambda x: f"{x:,.0f}")
                        d['put_volume'] = d['put_volume'].apply(lambda x: f"{x:,.0f}")
                        d['total_volume'] = d['total_volume'].apply(lambda x: f"{x:,.0f}")
                        d['pct_avg'] = d['pct_avg'].apply(lambda x: f"{x}%")
                        d['bot_pct'] = d['bot_pct'].apply(lambda x: f"{x}%")
                        d['ratio'] = d['ratio'].round(2)
                    
                    st.write("### High Buy Stocks (Ratio > 2)")
                    if not high_buy_df.empty:
                        columns = ['symbol', 'call_volume', 'put_volume', 'pct_avg', 'ratio', 'bot_pct', 'total_volume', 'signal']
                        st.dataframe(high_buy_df[columns].style.applymap(style_signal, subset=['signal']))
                    else:
                        st.write("No high buy stocks found.")
                    
                    st.write("### High Sell Stocks (Ratio < 0.5)")
                    if not high_sell_df.empty:
                        columns = ['symbol', 'call_volume', 'put_volume', 'pct_avg', 'ratio', 'bot_pct', 'total_volume', 'signal']
                        st.dataframe(high_sell_df[columns].style.applymap(style_signal, subset=['signal']))
                    else:
                        st.write("No high sell stocks found.")
                else:
                    st.write("No data available for today.")
    
    with tabs[2]:
        st.subheader("Watchlist Summary")
        selected_theme = st.selectbox("Select Watchlist (Theme)", list(theme_mapping.keys()), index=0)
        if st.button("Generate Watchlist Summary"):
            with st.spinner(f"Analyzing {selected_theme}..."):
                symbols = theme_mapping[selected_theme]
                df = get_today_sentiments()
                if not df.empty:
                    df = df[df['symbol'].isin(symbols)]
                    if not df.empty:
                        df['total_volume'] = df['call_volume'] + df['put_volume']
                        df['ratio'] = df['call_volume'] / df['put_volume'].replace(0, 1)
                        df['ratio'] = df['ratio'].replace(float('inf'), 5.0).clip(upper=5.0)
                        df['pct_avg'] = (df['call_volume'] / df['total_volume'] * 100).round(0).astype(int)
                        df['bot_pct'] = df['pct_avg']
                        df['signal'] = df['ratio'].apply(get_signal)
                        
                        total_call = df['call_volume'].sum()
                        total_put = df['put_volume'].sum()
                        aggregate_ratio = total_call / total_put if total_put > 0 else 5.0
                        
                        st.subheader("Summary Metrics")
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Total Call Volume", f"{total_call:,.0f}")
                        col2.metric("Total Put Volume", f"{total_put:,.0f}")
                        col3.metric("Aggregate Ratio", f"{aggregate_ratio:.2f}")
                        
                        display_df = df.copy()
                        display_df['call_volume'] = display_df['call_volume'].apply(lambda x: f"{x:,.0f}")
                        display_df['put_volume'] = display_df['put_volume'].apply(lambda x: f"{x:,.0f}")
                        display_df['total_volume'] = display_df['total_volume'].apply(lambda x: f"{x:,.0f}")
                        display_df['pct_avg'] = display_df['pct_avg'].apply(lambda x: f"{x}%")
                        display_df['bot_pct'] = display_df['bot_pct'].apply(lambda x: f"{x}%")
                        display_df['ratio'] = display_df['ratio'].round(2)
                        
                        columns = ['symbol', 'call_volume', 'put_volume', 'pct_avg', 'ratio', 'bot_pct', 'total_volume', 'signal']
                        st.dataframe(display_df[columns].style.applymap(style_signal, subset=['signal']))
                    else:
                        st.write(f"No data available for {selected_theme} symbols today.")
                else:
                    st.write("No data available for today.")

if __name__ == "__main__":
    run()
