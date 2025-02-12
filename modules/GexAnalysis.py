import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def run():
# Set page config
st.set_page_config(
    page_title="Options GEX Analysis",
    page_icon="📈",
    layout="wide"
)

# Title and description
st.title("Options Gamma Exposure (GEX) Analysis")
st.markdown("""
This application analyzes and visualizes the Gamma Exposure (GEX) for stock options.
GEX helps understand potential price magnetism and resistance levels based on options market positioning.
""")

def calculate_gamma(S, K, T, r, sigma, option_type='call'):
    if T <= 0.001:
        T = 0.001
    
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    gamma = np.exp(-d1**2/2) / (S*sigma*np.sqrt(2*np.pi*T))
    return gamma

def find_best_expiration(ticker, min_days, max_days):
    expirations = ticker.options
    today = datetime.now()
    
    best_oi = 0
    best_exp = None
    
    expiration_data = []
    for exp in expirations:
        exp_date = datetime.strptime(exp, '%Y-%m-%d')
        days_to_exp = (exp_date - today).days
        
        if min_days <= days_to_exp <= max_days:
            try:
                opt = ticker.option_chain(exp)
                total_oi = opt.calls['openInterest'].sum() + opt.puts['openInterest'].sum()
                expiration_data.append({
                    'date': exp,
                    'days': days_to_exp,
                    'oi': total_oi
                })
                
                if total_oi > best_oi:
                    best_oi = total_oi
                    best_exp = exp
            except Exception as e:
                st.warning(f"Error processing expiration {exp}: {str(e)}")
                continue
    
    return best_exp, pd.DataFrame(expiration_data)

def fetch_gex_data(ticker_symbol, expiration, price_range_pct, threshold):
    try:
        ticker = yf.Ticker(ticker_symbol)
        hist = ticker.history(period='5d')
        if hist.empty:
            raise ValueError(f"No price data available for {ticker_symbol}")
        
        current_price = hist['Close'].iloc[-1]
        
        exp_date = datetime.strptime(expiration, '%Y-%m-%d')
        today = datetime.now()
        T = max((exp_date - today).days / 365, 0.001)
        
        opt = ticker.option_chain(expiration)
        
        # Determine strike spacing
        if current_price > 500:
            strike_spacing = 5
        elif current_price > 100:
            strike_spacing = 2.5
        else:
            strike_spacing = 1
            
        calls = opt.calls[['strike', 'openInterest', 'impliedVolatility']].copy()
        calls.loc[:, 'type'] = 'call'
        
        puts = opt.puts[['strike', 'openInterest', 'impliedVolatility']].copy()
        puts.loc[:, 'type'] = 'put'
        
        # Filter strikes based on user-selected range
        price_range = current_price * (price_range_pct / 100)
        calls = calls[
            (calls['strike'] >= current_price - price_range) & 
            (calls['strike'] <= current_price + price_range)
        ]
        puts = puts[
            (puts['strike'] >= current_price - price_range) & 
            (puts['strike'] <= current_price + price_range)
        ]
        
        calls = calls[calls['openInterest'] > 0].dropna()
        puts = puts[puts['openInterest'] > 0].dropna()
        
        options_data = pd.concat([calls, puts])
        
        gex_data = []
        for _, row in options_data.iterrows():
            K = row['strike']
            sigma = row['impliedVolatility']
            oi = row['openInterest']
            
            gamma = calculate_gamma(current_price, K, T, 0.05, sigma)
            gex = gamma * oi * current_price / 10000
            if row['type'] == 'put':
                gex = -gex
            
            gex_data.append({
                'strike': K,
                'gex': gex,
                'oi': oi,
                'type': row['type']
            })
        
        df = pd.DataFrame(gex_data)
        df['abs_gex'] = abs(df['gex'])
        
        filtered_df = df[df['abs_gex'] > threshold].drop('abs_gex', axis=1)
        
        return filtered_df, current_price
        
    except Exception as e:
        st.error(f"Error processing {ticker_symbol}: {str(e)}")
        return pd.DataFrame(), None

def plot_gex(gex_data, current_price, ticker_symbol):
    if gex_data.empty:
        st.warning("No significant GEX values found for the selected parameters")
        return
    
    fig, ax = plt.subplots(figsize=(15, 8))
    
    gex_data = gex_data.sort_values('strike')
    
    bars = ax.bar(gex_data['strike'], gex_data['gex'],
                 color=['green' if x >= 0 else 'red' for x in gex_data['gex']],
                 alpha=0.6, width=2.0)
    
    ax.axvline(x=current_price, color='blue', linestyle='--',
               label=f'Current Price: {current_price:.2f}')
    
    ax.set_title(f'Gamma Exposure (GEX) for {ticker_symbol}', fontsize=12, pad=20)
    ax.set_xlabel('Strike Price')
    ax.set_ylabel('GEX')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    significant_threshold = gex_data['gex'].abs().max() * 0.1
    for bar in bars:
        height = bar.get_height()
        if abs(height) >= significant_threshold:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}',
                    ha='center', va='bottom' if height >= 0 else 'top',
                    fontsize=8)
    
    stats_text = (f"Total GEX: {gex_data['gex'].sum():.1f}\n"
                 f"Max +GEX: {gex_data['gex'].max():.1f}\n"
                 f"Max -GEX: {gex_data['gex'].min():.1f}")
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig

# Sidebar controls
st.sidebar.header("Analysis Parameters")

# Ticker input with default popular tickers
popular_tickers = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'NVDA', 'TSLA']
ticker_input = st.sidebar.text_input("Enter Ticker Symbol:", value='SPY')
ticker_symbol = ticker_input.upper()

if st.sidebar.button("Analyze"):
    try:
        # Initialize ticker
        ticker = yf.Ticker(ticker_symbol)
        
        # Date range selection for expiration
        st.sidebar.subheader("Expiration Selection")
        min_days = st.sidebar.slider("Minimum Days to Expiration", 1, 30, 1)
        max_days = st.sidebar.slider("Maximum Days to Expiration", min_days, 60, 30)
        
        # Find best expiration
        best_exp, exp_data = find_best_expiration(ticker, min_days, max_days)
        
        if not best_exp:
            st.error("No suitable expiration dates found")
        else:
            # Display expiration data
            st.subheader("Available Expirations")
            st.dataframe(exp_data)
            
            # Let user select expiration
            selected_exp = st.selectbox(
                "Select Expiration Date",
                exp_data['date'].tolist(),
                index=exp_data['date'].tolist().index(best_exp) if best_exp in exp_data['date'].tolist() else 0
            )
            
            # Additional parameters
            col1, col2 = st.columns(2)
            with col1:
                price_range = st.slider("Price Range (%)", 5, 30, 15)
            with col2:
                gex_threshold = st.slider("GEX Threshold", 0.1, 20.0, 5.0)
            
            # Fetch and plot GEX data
            gex_data, current_price = fetch_gex_data(
                ticker_symbol,
                selected_exp,
                price_range,
                gex_threshold
            )
            
            if not gex_data.empty and current_price:
                # Display current price and statistics
                st.metric("Current Price", f"${current_price:.2f}")
                
                # Show GEX plot
                fig = plot_gex(gex_data, current_price, ticker_symbol)
                st.pyplot(fig)
                
                # Display raw data in expandable section
                with st.expander("View Raw GEX Data"):
                    st.dataframe(gex_data)
            
    except Exception as e:
        st.error(f"Error analyzing {ticker_symbol}: {str(e)}")

if __name__ == "__main__":
    run()
