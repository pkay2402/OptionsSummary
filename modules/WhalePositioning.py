import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pytz
import logging
from tenacity import retry, stop_after_attempt, wait_exponential
from concurrent.futures import ThreadPoolExecutor

# Setup logging
logging.basicConfig(filename='options_range.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

@st.cache_data(ttl=300)
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def get_weekly_expirations(ticker_symbol, num_weeks=4):
    """Fetch weekly expirations"""
    try:
        ticker = yf.Ticker(ticker_symbol)
        expirations = ticker.options
        eastern = pytz.timezone('US/Eastern')
        now = datetime.now(eastern)
        today = now.date()
        
        weekly_exps = []
        for exp in expirations:
            exp_date = datetime.strptime(exp, '%Y-%m-%d').date()
            days_to_exp = (exp_date - today).days
            if exp_date.weekday() == 4 and 0 <= days_to_exp <= (num_weeks * 7 + 7):
                weekly_exps.append({'date': exp, 'days': days_to_exp})
        
        result = sorted(weekly_exps, key=lambda x: x['days'])[:num_weeks]
        logging.info(f"Fetched {len(result)} weekly expirations for {ticker_symbol}")
        return result
    except Exception as e:
        logging.error(f"Error fetching expirations: {str(e)}")
        st.error(f"Error fetching expirations: {str(e)}")
        return []

def fetch_option_chain(ticker, exp, price_min, price_max):
    """Fetch option chain data"""
    try:
        opt = ticker.option_chain(exp['date'])
        calls = opt.calls[(opt.calls['strike'] >= price_min) & (opt.calls['strike'] <= price_max)][['strike', 'openInterest', 'volume']].copy()
        calls['type'] = 'call'
        calls['expiry_date'] = exp['date']
        
        puts = opt.puts[(opt.puts['strike'] >= price_min) & (opt.puts['strike'] <= price_max)][['strike', 'openInterest', 'volume']].copy()
        puts['type'] = 'put'
        puts['expiry_date'] = exp['date']
        
        return calls, puts
    except Exception as e:
        logging.warning(f"Error fetching option chain for {exp['date']}: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()

@st.cache_data(ttl=60)  # Refresh every minute during market hours
def fetch_options_data(ticker_symbol, expirations, price_range_pct=10):
    """Fetch volume and OI data"""
    try:
        ticker = yf.Ticker(ticker_symbol)
        hist = ticker.history(period='1d', interval='1m')
        current_price = hist['Close'].iloc[-1] if not hist.empty else None
        if not current_price:
            raise ValueError("Unable to fetch current price")
        
        price_min = current_price * (1 - price_range_pct/100)
        price_max = current_price * (1 + price_range_pct/100)
        
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(lambda exp: fetch_option_chain(ticker, exp, price_min, price_max), expirations))
        options_data = [item for sublist in results for item in sublist if not item.empty]
        
        if not options_data:
            raise ValueError("No option data within price range")
        
        df = pd.concat(options_data)
        df['total_activity'] = df['volume'].fillna(0) + df['openInterest'].fillna(0)
        pivot_df = df.pivot_table(index=['strike', 'expiry_date'], columns='type', 
                                 values=['total_activity'], aggfunc='sum').fillna(0)
        pivot_df.columns = ['call_activity', 'put_activity']
        pivot_df['total_activity'] = pivot_df['call_activity'] + pivot_df['put_activity']
        
        return pivot_df, current_price
    except Exception as e:
        logging.error(f"Error fetching options data: {str(e)}")
        st.error(f"Error fetching options data: {str(e)}")
        return pd.DataFrame(), None

def calculate_trading_range(pivot_df, current_price, expiry_date, activity_threshold=0.8):
    """Calculate expected trading range for a given expiry"""
    if pivot_df.empty:
        return None, None
    
    expiry_df = pivot_df.xs(expiry_date, level='expiry_date')
    total_activity = expiry_df['total_activity'].sum()
    if total_activity == 0:
        return current_price, current_price
    
    # Sort by strike and compute cumulative activity
    expiry_df = expiry_df.sort_index()
    expiry_df['cumulative_activity'] = expiry_df['total_activity'].cumsum() / total_activity
    
    # Find range containing threshold% of activity
    lower_bound = expiry_df[expiry_df['cumulative_activity'] >= (1 - activity_threshold) / 2].index[0]
    upper_bound = expiry_df[expiry_df['cumulative_activity'] <= 1 - (1 - activity_threshold) / 2].index[-1]
    
    return lower_bound, upper_bound

def plot_activity(pivot_df, current_price, ticker_symbol, weekly_target):
    """Plot activity distribution"""
    if pivot_df.empty:
        return None
    
    weekly_df = pivot_df.xs(weekly_target, level='expiry_date')
    fig = go.Figure()
    fig.add_trace(go.Bar(x=weekly_df.index, y=weekly_df['call_activity'], name='Call Activity', marker_color='green', opacity=0.7))
    fig.add_trace(go.Bar(x=weekly_df.index, y=weekly_df['put_activity'], name='Put Activity', marker_color='red', opacity=0.7))
    fig.add_vline(x=current_price, line_dash="dash", line_color="blue", annotation_text=f"Current: ${current_price:.2f}")
    fig.update_layout(
        title=f"Options Activity for {ticker_symbol} (Next Week: {weekly_target})",
        xaxis_title="Strike Price",
        yaxis_title="Total Activity (Volume + OI)",
        barmode='stack',
        legend=dict(x=0.01, y=0.99)
    )
    return fig

def validate_ticker(ticker_symbol):
    """Validate ticker symbol"""
    try:
        ticker = yf.Ticker(ticker_symbol)
        ticker.info
        return True
    except:
        logging.warning(f"Invalid ticker symbol: {ticker_symbol}")
        st.error(f"Invalid ticker symbol: {ticker_symbol}")
        return False

def run():
    st.markdown("<h1 style='text-align: center;'>Weekly Options Range Analysis</h1>", unsafe_allow_html=True)
    st.sidebar.markdown("### Configuration")
    ticker_input = st.sidebar.text_input("Ticker Symbol", value='SPY').upper()
    price_range_pct = st.sidebar.slider("Price Range (%)", 5, 20, 10)
    num_weeks = st.sidebar.slider("Weeks to Analyze", 1, 4, 4)
    refresh_rate = st.sidebar.slider("Refresh Rate (seconds)", 30, 300, 60)
    
    if st.sidebar.button("Start Analysis"):
        if not validate_ticker(ticker_input):
            return
        
        weekly_exps = get_weekly_expirations(ticker_input, num_weeks)
        if not weekly_exps:
            st.error("No weekly expirations found")
            return
        
        placeholder = st.empty()
        while True:
            with st.spinner(f'Analyzing {ticker_input} options data...'):
                pivot_df, current_price = fetch_options_data(ticker_input, weekly_exps, price_range_pct)
                if pivot_df.empty or current_price is None:
                    st.error("No data available")
                    break
                
                # Weekly range for next expiry
                weekly_target = weekly_exps[0]['date']
                lower_bound, upper_bound = calculate_trading_range(pivot_df, current_price, weekly_target)
                
                # Display results
                with placeholder.container():
                    st.subheader(f"{ticker_input} - Current Price: ${current_price:.2f}")
                    st.markdown(f"**Traders expect {ticker_input} to trade between ${lower_bound:.2f} and ${upper_bound:.2f} next week ({weekly_target})**")
                    
                    fig = plot_activity(pivot_df, current_price, ticker_input, weekly_target)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
            
            # Refresh during market hours
            eastern = pytz.timezone('US/Eastern')
            now = datetime.now(eastern)
            market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
            market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
            if market_open <= now <= market_close:
                st.write(f"Updating in {refresh_rate} seconds...")
                st.experimental_rerun()
                time.sleep(refresh_rate)
            else:
                st.write("Market is closed. Showing latest data.")
                break

if __name__ == "__main__":
    run()
