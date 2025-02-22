# modules/WhalePositioning.py
import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pytz

def get_weekly_expirations(ticker_symbol, num_weeks=8):
    """Fetch the next 8 weekly expirations"""
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
            if 0 <= days_to_exp <= (num_weeks * 7 + 7):
                weekly_exps.append({'date': exp, 'days': days_to_exp})
        
        return sorted(weekly_exps, key=lambda x: x['days'])[:num_weeks]
    except Exception as e:
        st.error(f"Error fetching weekly expirations: {str(e)}")
        return []

def fetch_whale_positions(ticker_symbol, expirations, price_range_pct=10):
    """Get calls/puts OI and volume within 10% of current price with price trend"""
    try:
        ticker = yf.Ticker(ticker_symbol)
        hist = ticker.history(period='1d', interval='1m')
        current_price = hist['Close'].iloc[-1] if not hist.empty else None
        
        if not current_price:
            raise ValueError("Unable to fetch current price")
            
        last_hour = hist.tail(60)
        price_trend = (last_hour['Close'].iloc[-1] - last_hour['Close'].iloc[0]) / last_hour['Close'].iloc[0] * 100
        
        whale_data = []
        price_min = current_price * (1 - price_range_pct/100)
        price_max = current_price * (1 + price_range_pct/100)
        
        for exp in expirations:
            opt = ticker.option_chain(exp['date'])
            calls = opt.calls[
                (opt.calls['strike'] >= price_min) & 
                (opt.calls['strike'] <= price_max)
            ][['strike', 'openInterest', 'volume']].copy()
            calls['type'] = 'call'
            calls['days_to_exp'] = exp['days']
            calls['expiry_date'] = exp['date']
            
            puts = opt.puts[
                (opt.puts['strike'] >= price_min) & 
                (opt.puts['strike'] <= price_max)
            ][['strike', 'openInterest', 'volume']].copy()
            puts['type'] = 'put'
            puts['days_to_exp'] = exp['days']
            puts['expiry_date'] = exp['date']
            
            whale_data.append(calls)
            whale_data.append(puts)
        
        df = pd.concat(whale_data)
        
        df['price_weight'] = 1 - (abs(df['strike'] - current_price) / (current_price * price_range_pct/100))
        df['time_weight'] = 1 / (df['days_to_exp'] + 1)
        df['weighted_volume'] = df['volume'] * df['price_weight'] * df['time_weight']
        
        agg_df = df.groupby(['strike', 'type', 'expiry_date']).agg({
            'openInterest': 'sum',
            'volume': 'sum',
            'weighted_volume': 'sum'
        }).reset_index()
        
        pivot_df = agg_df.pivot(index=['strike', 'expiry_date'], columns='type', 
                              values=['openInterest', 'volume', 'weighted_volume']).fillna(0)
        pivot_df.columns = ['call_oi', 'put_oi', 'call_vol', 'put_vol', 'call_wv', 'put_wv']
        
        pivot_df['total_oi'] = pivot_df['call_oi'] + pivot_df['put_oi']
        pivot_df['net_vol'] = pivot_df['call_vol'] - pivot_df['put_vol']
        pivot_df['net_wv'] = pivot_df['call_wv'] - pivot_df['put_wv']
        
        strike_df = pivot_df.groupby(level='strike').sum()
        
        return pivot_df, strike_df, current_price, price_trend
    except Exception as e:
        st.error(f"Error fetching whale positions: {str(e)}")
        return pd.DataFrame(), pd.DataFrame(), None, None

def predict_price_direction(current_price, whale_df, price_trend):
    """Smart prediction with expiration date"""
    if whale_df.empty:
        return "Insufficient data", None, None, "gray"
    
    call_wv_score = whale_df['call_wv'].sum()
    put_wv_score = whale_df['put_wv'].sum()
    net_wv_score = call_wv_score - put_wv_score
    
    call_oi_concentration = whale_df[whale_df.index.get_level_values('strike') > current_price]['call_oi'].sum()
    put_oi_concentration = whale_df[whale_df.index.get_level_values('strike') < current_price]['put_oi'].sum()
    
    direction_score = (
        (net_wv_score / max(abs(net_wv_score), 1)) * 0.4 +
        (price_trend / max(abs(price_trend), 0.1)) * 0.3 +
        ((call_oi_concentration - put_oi_concentration) / 
         max(call_oi_concentration + put_oi_concentration, 1)) * 0.3
    )
    
    if direction_score > 0:
        target_df = whale_df[whale_df.index.get_level_values('strike') > current_price]
        if not target_df.empty:
            target_row = target_df.loc[target_df['call_wv'].idxmax()]
            target_strike = target_row.name[0]
            target_expiry = target_row.name[1]
        else:
            target_strike = current_price
            target_expiry = whale_df.index.get_level_values('expiry_date')[0]
    else:
        target_df = whale_df[whale_df.index.get_level_values('strike') < current_price]
        if not target_df.empty:
            target_row = target_df.loc[target_df['put_wv'].idxmax()]
            target_strike = target_row.name[0]
            target_expiry = target_row.name[1]
        else:
            target_strike = current_price
            target_expiry = whale_df.index.get_level_values('expiry_date')[0]
    
    if direction_score > 0.3:
        direction = "Bullish"
        color = "green"
    elif direction_score < -0.3:
        direction = "Bearish"
        color = "red"
    else:
        direction = "Neutral"
        color = "gray"
    
    confidence = min(abs(direction_score) * 100, 100)
    
    return (f"{direction} toward ${target_strike:.2f} for {target_expiry} "
            f"(Confidence: {confidence:.0f}%)"), target_strike, target_expiry, color

def plot_volume_trend(whale_df, current_price, ticker_symbol):
    """Bar chart showing weighted volume trend"""
    if whale_df.empty:
        return None
    
    strike_df = whale_df.groupby(level='strike').sum()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.bar(strike_df.index - 0.2, strike_df['call_wv'], width=0.4, 
           color='green', alpha=0.7, label='Call Weighted Volume')
    ax.bar(strike_df.index + 0.2, strike_df['put_wv'], width=0.4, 
           color='red', alpha=0.7, label='Put Weighted Volume')
    
    ax.axvline(current_price, color='blue', linestyle='--', 
              label=f'Current Price: ${current_price:.2f}')
    
    ax.set_title(f"Weighted Volume Trend Analysis for {ticker_symbol} (±10% Price Range)")
    ax.set_xlabel("Strike Price")
    ax.set_ylabel("Weighted Volume")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    total_call_wv = strike_df['call_wv'].sum()
    total_put_wv = strike_df['put_wv'].sum()
    stats_text = f"Call WV: {total_call_wv:,.0f}\nPut WV: {total_put_wv:,.0f}"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig

def run():
    st.markdown("<h1 style='text-align: center;'>Smart Whale Positioning Analysis</h1>", unsafe_allow_html=True)
    
    ticker_symbol = st.sidebar.text_input("Ticker Symbol", value='SPY', key='whale_ticker').upper()
    analyze_button = st.sidebar.button("Analyze", key='whale_analyze')
    
    if analyze_button:
        with st.spinner('Analyzing whale positions...'):
            weekly_exps = get_weekly_expirations(ticker_symbol)
            
            if weekly_exps:
                whale_df, strike_df, current_price, price_trend = fetch_whale_positions(ticker_symbol, weekly_exps)
                
                if not whale_df.empty and current_price:
                    st.subheader(f"Options Data within 10% of Current Price (${current_price:.2f})")
                    display_df = strike_df[['call_oi', 'put_oi', 'total_oi', 'call_vol', 'put_vol', 'net_vol']]
                    st.dataframe(
                        display_df.style.format({
                            'call_oi': '{:,.0f}',
                            'put_oi': '{:,.0f}',
                            'total_oi': '{:,.0f}',
                            'call_vol': '{:,.0f}',
                            'put_vol': '{:,.0f}',
                            'net_vol': '{:+,.0f}'
                        }).background_gradient(subset=['total_oi'], cmap='Blues'),
                        use_container_width=True
                    )
                    
                    direction, target_strike, target_expiry, color = predict_price_direction(
                        current_price, whale_df, price_trend
                    )
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Current Price", f"${current_price:.2f}")
                        st.metric("Last Hour Trend", f"{price_trend:+.2f}%")
                    with col2:
                        st.markdown(f"""
                        <div style='background-color: {color}; color: white; padding: 10px; 
                        border-radius: 5px; text-align: center;'>
                        Predicted Direction: {direction}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.subheader("Weighted Volume Trend Analysis")
                    fig = plot_volume_trend(whale_df, current_price, ticker_symbol)
                    if fig:
                        st.pyplot(fig)
                    
                    st.subheader("Key Insights")
                    total_call_oi = strike_df['call_oi'].sum()
                    total_put_oi = strike_df['put_oi'].sum()
                    total_call_vol = strike_df['call_vol'].sum()
                    total_put_vol = strike_df['put_vol'].sum()
                    total_call_wv = strike_df['call_wv'].sum()
                    total_put_wv = strike_df['put_wv'].sum()
                    
                    st.markdown(f"""
                    - **Analysis Range**: Strikes between ${current_price*0.9:.2f} and ${current_price*1.1:.2f}
                    - **Open Interest**: 
                      - Calls: {total_call_oi:,.0f} contracts
                      - Puts: {total_put_oi:,.0f} contracts
                      - Call/Put Ratio: {total_call_oi/total_put_oi if total_put_oi > 0 else 'N/A':.2f}
                    - **Today's Volume**:
                      - Call Volume: {total_call_vol:,.0f} contracts
                      - Put Volume: {total_put_vol:,.0f} contracts
                      - Weighted Call Volume: {total_call_wv:,.0f}
                      - Weighted Put Volume: {total_put_wv:,.0f}
                    - **Momentum Factors**:
                      - Price Trend (1h): {price_trend:+.2f}%
                      - Volume Momentum: {total_call_wv - total_put_wv:+,.0f}
                    - **Prediction**: {direction}
                    - **Target Details**:
                      - Strike: ${target_strike:.2f}
                      - Expiration: {target_expiry}
                    - **Note**: Prediction targets the expiry with strongest weighted volume
                    """)
                else:
                    st.warning("No data available within the specified price range")
            else:
                st.warning("No weekly expirations found for this ticker")
