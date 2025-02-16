import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def calculate_gamma(S, K, T, r, sigma, option_type='call'):
    if T <= 0.001:
        T = 0.001
    
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    gamma = np.exp(-d1**2/2) / (S*sigma*np.sqrt(2*np.pi*T))
    return gamma

def get_all_expirations(ticker):
    try:
        expirations = ticker.options
        today = datetime.now()
        
        expiration_data = []
        for exp in expirations:
            exp_date = datetime.strptime(exp, '%Y-%m-%d')
            days_to_exp = (exp_date - today).days
            
            try:
                opt = ticker.option_chain(exp)
                total_oi = opt.calls['openInterest'].sum() + opt.puts['openInterest'].sum()
                expiration_data.append({
                    'date': exp,
                    'days': days_to_exp,
                    'oi': total_oi
                })
            except Exception as e:
                st.warning(f"Error processing expiration {exp}: {str(e)}")
                continue
                
        return pd.DataFrame(expiration_data)
    except Exception as e:
        st.error(f"Error fetching expirations: {str(e)}")
        return pd.DataFrame()

def fetch_gex_data(ticker_symbol, expiration, price_range_pct, threshold, strike_spacing_override=None, risk_free_rate=0.05):
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
        
        # Use override if provided, otherwise calculate based on price
        if strike_spacing_override:
            strike_spacing = strike_spacing_override
        else:
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
            
            gamma = calculate_gamma(current_price, K, T, risk_free_rate, sigma)
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

def plot_gex(gex_data, current_price, ticker_symbol, bar_width=2.0, show_labels=True):
    if gex_data.empty:
        st.warning("No significant GEX values found for the selected parameters")
        return
    
    fig, ax = plt.subplots(figsize=(15, 8))
    
    gex_data = gex_data.sort_values('strike')
    
    bars = ax.bar(gex_data['strike'], gex_data['gex'],
                 color=['green' if x >= 0 else 'red' for x in gex_data['gex']],
                 alpha=0.6, width=bar_width)
    
    ax.axvline(x=current_price, color='blue', linestyle='--',
               label=f'Current Price: {current_price:.2f}')
    
    ax.set_title(f'Gamma Exposure (GEX) for {ticker_symbol}', fontsize=12, pad=20)
    ax.set_xlabel('Strike Price')
    ax.set_ylabel('GEX')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    if show_labels:
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

def run():
    st.title("Options Gamma Exposure (GEX) Analysis")
    st.markdown("""
    This application analyzes and visualizes the Gamma Exposure (GEX) for stock options.
    GEX helps understand potential price magnetism and resistance levels based on options market positioning.
    """)

    # Initialize variables
    gex_data = pd.DataFrame()
    current_price = None
    full_gex_data = pd.DataFrame()

    # Main parameters in sidebar
    st.sidebar.header("Analysis Parameters")

    # Ticker input
    ticker_input = st.sidebar.text_input("Enter Ticker Symbol:", value='SPY')
    ticker_symbol = ticker_input.upper()

    if st.sidebar.button("Load Ticker Data"):
        try:
            ticker = yf.Ticker(ticker_symbol)
            
            # Get all expirations
            exp_data = get_all_expirations(ticker)
            
            if not exp_data.empty:
                st.session_state['exp_data'] = exp_data
                st.session_state['ticker'] = ticker_symbol
                st.success(f"Successfully loaded data for {ticker_symbol}")
            else:
                st.error("No expiration dates found")
                return
                
        except Exception as e:
            st.error(f"Error loading {ticker_symbol}: {str(e)}")
            return

    if 'exp_data' in st.session_state:
        # Advanced Parameters
        with st.expander("Advanced Parameters", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                price_range = st.slider("Price Range (%)", 1, 50, 15)
                gex_threshold = st.slider("GEX Threshold", 0.1, 50.0, 5.0)
                strike_spacing = st.number_input("Strike Spacing Override (0 for auto)", 0.0, 100.0, 0.0)
                
            with col2:
                risk_free_rate = st.slider("Risk-Free Rate (%)", 0.0, 10.0, 5.0) / 100
                bar_width = st.slider("Bar Width", 0.5, 5.0, 2.0)
                show_labels = st.checkbox("Show Value Labels", True)

        # Expiration Selection
        st.subheader("Select Expiration")
        
        # Filter options
        col1, col2 = st.columns(2)
        with col1:
            min_days = st.number_input("Minimum Days to Expiration", 0, 365, 1)
        with col2:
            max_days = st.number_input("Maximum Days to Expiration", min_days, 365, 30)

        # Filter expiration data
        filtered_exp_data = st.session_state['exp_data'][
            (st.session_state['exp_data']['days'] >= min_days) &
            (st.session_state['exp_data']['days'] <= max_days)
        ]

        if not filtered_exp_data.empty:
            st.dataframe(filtered_exp_data)
            
            selected_exp = st.selectbox(
                "Select Expiration Date",
                filtered_exp_data['date'].tolist()
            )

            if st.button("Generate GEX Analysis"):
                # Fetch and plot GEX data
                try:
                    result = fetch_gex_data(
                        st.session_state['ticker'],
                        selected_exp,
                        price_range,
                        gex_threshold,
                        strike_spacing if strike_spacing > 0 else None,
                        risk_free_rate
                    )
                    
                    if len(result) == 3:
                        gex_data, current_price, full_gex_data = result
                    else:
                        gex_data, current_price = result
                        full_gex_data = gex_data.copy()
                    
                    if not gex_data.empty and current_price is not None:
                        # Display current price and statistics
                        st.metric("Current Price", f"${current_price:.2f}")
                        
                        # Show GEX plot
                        fig = plot_gex(gex_data, current_price, st.session_state['ticker'], 
                                     bar_width, show_labels)
                        st.pyplot(fig)
                        
                        # Add Summary Section
                        st.subheader("GEX Analysis Summary")
                        
                        # Calculate key metrics
                        total_gex = gex_data['gex'].sum()
                        max_positive_gex = gex_data['gex'].max()
                        max_negative_gex = gex_data['gex'].min()
                        
                        # Find strongest GEX level
                        if 'abs_gex' in full_gex_data.columns:
                            strongest_gex_strike = full_gex_data.loc[full_gex_data['abs_gex'].idxmax(), 'strike']
                        else:
                            full_gex_data['abs_gex'] = abs(full_gex_data['gex'])
                            strongest_gex_strike = full_gex_data.loc[full_gex_data['abs_gex'].idxmax(), 'strike']
                        
                        # Determine market positioning
                        market_bias = "Positive" if total_gex > 0 else "Negative"
                        gex_strength = "Strong" if abs(total_gex) > 20 else "Moderate" if abs(total_gex) > 10 else "Weak"
                        
                        # Display summary metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total GEX", f"{total_gex:.2f}")
                        with col2:
                            st.metric("Market Bias", market_bias)
                        with col3:
                            st.metric("GEX Strength", gex_strength)
                        
                        st.markdown("### Key Takeaways")
                        st.markdown(f"""
                        **Market Structure:**
                        - Overall GEX Bias: {market_bias} with {gex_strength.lower()} strength
                        - Strongest GEX Level: ${strongest_gex_strike:.2f}
                        - Largest Positive GEX: {max_positive_gex:.2f}
                        - Largest Negative GEX: {max_negative_gex:.2f}
                        
                        **What This Means:**
                        1. **Price Magnetism:**
                           - Strong GEX levels tend to act as price magnets
                           - ${strongest_gex_strike:.2f} is the strongest magnetic level
                        
                        2. **Market Movement:**
                           - Positive GEX suggests resistance to downward movement
                           - Negative GEX suggests resistance to upward movement
                           - Current bias suggests {'resistance to downward moves' if total_gex > 0 else 'resistance to upward moves'}
                        
                        3. **Volatility Implications:**
                           - {'High GEX levels typically suppress volatility' if abs(total_gex) > 20 else 'Moderate GEX levels suggest normal volatility' if abs(total_gex) > 10 else 'Low GEX levels may allow for larger price movements'}
                           - Expect {'more stable price action' if abs(total_gex) > 20 else 'normal price action' if abs(total_gex) > 10 else 'potentially volatile price action'}
                        
                        **Trading Considerations:**
                        - Look for potential price resistance at ${strongest_gex_strike:.2f}
                        - {'Consider mean reversion strategies near strong GEX levels' if abs(total_gex) > 20 else 'Monitor price action around GEX levels for potential support/resistance'}
                        - Volatility trades should account for {'suppression near GEX levels' if abs(total_gex) > 20 else 'normal conditions' if abs(total_gex) > 10 else 'potential expansion'}
                        
                        *Note: GEX analysis should be used in conjunction with other technical and fundamental indicators.*
                        """)
                        
                        # Display raw data in expandable section
                        with st.expander("View Raw GEX Data"):
                            st.dataframe(gex_data)
                except Exception as e:
                    st.error(f"Error generating GEX analysis: {str(e)}")
        else:
            st.warning("No expirations found within the selected date range")

if __name__ == "__main__":
    run()
