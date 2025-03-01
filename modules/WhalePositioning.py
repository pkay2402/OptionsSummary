# modules/WhalePositioning.py
import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pytz
from scipy.stats import norm
import math

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

def get_daily_expirations(ticker_symbol, num_days=5):
    """Fetch daily expirations for index ETFs like SPY, QQQ, SPX"""
    try:
        ticker = yf.Ticker(ticker_symbol)
        all_expirations = ticker.options
        eastern = pytz.timezone('US/Eastern')
        now = datetime.now(eastern)
        today = now.date()
        
        index_etfs = ['SPY', 'QQQ', '^SPX']
        if ticker_symbol in index_etfs:
            daily_exps = []
            for exp in all_expirations:
                exp_date = datetime.strptime(exp, '%Y-%m-%d').date()
                days_to_exp = (exp_date - today).days
                if 0 <= days_to_exp <= num_days:
                    daily_exps.append({'date': exp, 'days': days_to_exp})
            
            return sorted(daily_exps, key=lambda x: x['days'])
        else:
            return get_weekly_expirations(ticker_symbol)
    except Exception as e:
        st.error(f"Error fetching expirations: {str(e)}")
        return []

def get_intraday_context(ticker_symbol):
    """Get intraday context including time remaining until close"""
    eastern = pytz.timezone('US/Eastern')
    now = datetime.now(eastern)
    
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    
    if now < market_open:
        market_status = "pre-market"
        time_to_close = (market_close - now).total_seconds() / 3600
    elif now > market_close:
        market_status = "after-hours"
        time_to_close = 0
    else:
        market_status = "market-hours"
        time_to_close = (market_close - now).total_seconds() / 3600
    
    try:
        ticker = yf.Ticker(ticker_symbol)
        intraday = ticker.history(period='1d', interval='5m')
        
        open_price = intraday['Open'].iloc[0] if not intraday.empty else None
        high_price = intraday['High'].max() if not intraday.empty else None
        low_price = intraday['Low'].min() if not intraday.empty else None
        current_price = intraday['Close'].iloc[-1] if not intraday.empty else None
        yesterday_close = ticker.history(period='2d')['Close'].iloc[-2] if len(ticker.history(period='2d')) > 1 else None
        
        return {
            'market_status': market_status,
            'time_to_close': time_to_close,
            'day_progress': (now - market_open).total_seconds() / (market_close - market_open).total_seconds() if market_status == "market-hours" else None,
            'open_price': open_price,
            'high_price': high_price,
            'low_price': low_price,
            'current_price': current_price,
            'gap_pct': ((open_price / yesterday_close) - 1) * 100 if open_price and yesterday_close else None
        }
    except Exception as e:
        st.error(f"Error fetching intraday context: {str(e)}")
        return {}

def calculate_greeks(strike, current_price, days_to_exp, iv, risk_free_rate=0.04, option_type='call'):
    """Approximate Delta and Gamma using Black-Scholes"""
    T = days_to_exp / 365
    if T <= 0 or iv <= 0 or pd.isna(iv):
        return 0, 0
    
    S = current_price
    K = strike
    r = risk_free_rate
    sigma = iv
    
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    
    if option_type == 'call':
        delta = norm.cdf(d1)
    else:
        delta = norm.cdf(d1) - 1
    
    gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
    
    return delta, gamma

def fetch_whale_positions(ticker_symbol, expirations, price_range_pct=10):
    """Get calls/puts OI, volume, and Greeks within price range of current price with price trend"""
    try:
        ticker = yf.Ticker(ticker_symbol)
        hist = ticker.history(period='1d', interval='1m')
        current_price = hist['Close'].iloc[-1] if not hist.empty else None
        
        if not current_price:
            raise ValueError("Unable to fetch current price")
            
        last_hour = hist.tail(60)
        price_trend = (last_hour['Close'].iloc[-1] - last_hour['Close'].iloc[0]) / last_hour['Close'].iloc[0] * 100 if not last_hour.empty else 0
        
        whale_data = []
        price_min = current_price * (1 - price_range_pct/100)
        price_max = current_price * (1 + price_range_pct/100)
        
        for exp in expirations:
            opt = ticker.option_chain(exp['date'])
            for option_type, df in [('call', opt.calls), ('put', opt.puts)]:
                filtered_df = df[
                    (df['strike'] >= price_min) & 
                    (df['strike'] <= price_max)
                ][['strike', 'openInterest', 'volume', 'impliedVolatility', 'lastPrice']].copy()
                
                filtered_df['type'] = option_type
                filtered_df['days_to_exp'] = exp['days']
                filtered_df['expiry_date'] = exp['date']
                
                filtered_df['delta'] = filtered_df.apply(
                    lambda row: calculate_greeks(
                        row['strike'], current_price, row['days_to_exp'], 
                        row['impliedVolatility'], option_type=option_type)[0], axis=1
                )
                filtered_df['gamma'] = filtered_df.apply(
                    lambda row: calculate_greeks(
                        row['strike'], current_price, row['days_to_exp'], 
                        row['impliedVolatility'], option_type=option_type)[1], axis=1
                )
                filtered_df['delta_exposure'] = -filtered_df['delta'] * filtered_df['openInterest'] * 100
                filtered_df['gamma_exposure'] = -filtered_df['gamma'] * filtered_df['openInterest'] * 100
                
                whale_data.append(filtered_df)
        
        if not whale_data:
            return pd.DataFrame(), pd.DataFrame(), current_price, price_trend
        
        df = pd.concat(whale_data, ignore_index=True)
        
        df['price_weight'] = 1 - (abs(df['strike'] - current_price) / (current_price * price_range_pct/100))
        df['time_weight'] = 1 / (df['days_to_exp'] + 1)
        df['weighted_volume'] = df['volume'] * df['price_weight'] * df['time_weight']
        
        agg_df = df.groupby(['strike', 'type', 'expiry_date']).agg({
            'openInterest': 'sum',
            'volume': 'sum',
            'weighted_volume': 'sum',
            'delta_exposure': 'sum',
            'gamma_exposure': 'sum'
        }).reset_index()
        
        pivot_df = agg_df.pivot_table(
            index=['strike', 'expiry_date'], 
            columns='type', 
            values=['openInterest', 'volume', 'weighted_volume', 'delta_exposure', 'gamma_exposure'],
            fill_value=0
        )
        
        pivot_df.columns = [
            f"{col[1]}_{col[0]}" if col[1] else col[0] 
            for col in pivot_df.columns
        ]
        
        pivot_df['total_oi'] = pivot_df['call_openInterest'] + pivot_df['put_openInterest']
        pivot_df['net_vol'] = pivot_df['call_volume'] - pivot_df['put_volume']
        pivot_df['net_wv'] = pivot_df['call_weighted_volume'] - pivot_df['put_weighted_volume']
        pivot_df['net_dxoi'] = pivot_df['call_delta_exposure'] + pivot_df['put_delta_exposure']
        pivot_df['net_gxoi'] = pivot_df['call_gamma_exposure'] + pivot_df['put_gamma_exposure']
        
        strike_df = pivot_df.groupby('strike').sum()
        
        return pivot_df, strike_df, current_price, price_trend
    except Exception as e:
        st.error(f"Error fetching whale positions: {str(e)}")
        return pd.DataFrame(), pd.DataFrame(), None, None

def predict_price_direction(current_price, whale_df, price_trend):
    """Smart prediction with expiration date incorporating gamma and delta"""
    if whale_df.empty:
        return "Insufficient data", None, None, "gray"
    
    call_wv_score = whale_df['call_weighted_volume'].sum()
    put_wv_score = whale_df['put_weighted_volume'].sum()
    net_wv_score = call_wv_score - put_wv_score
    
    net_gxoi = whale_df['net_gxoi'].sum()
    net_dxoi = whale_df['net_dxoi'].sum()
    
    call_oi_above = whale_df[whale_df.index.get_level_values('strike') > current_price]['call_openInterest'].sum()
    put_oi_below = whale_df[whale_df.index.get_level_values('strike') < current_price]['put_openInterest'].sum()
    
    direction_score = (
        (net_wv_score / max(abs(net_wv_score), 1)) * 0.3 +
        (price_trend / max(abs(price_trend), 0.1)) * 0.2 +
        ((call_oi_above - put_oi_below) / max(call_oi_above + put_oi_below, 1)) * 0.2 +
        (net_gxoi / max(abs(net_gxoi), 1)) * 0.2 +
        (net_dxoi / max(abs(net_dxoi), 1)) * 0.1
    )
    
    if direction_score > 0:
        target_df = whale_df[whale_df.index.get_level_values('strike') > current_price]
        if not target_df.empty:
            target_row = target_df.loc[target_df['call_gamma_exposure'].idxmax()]
            target_strike = target_row.name[0]
            target_expiry = target_row.name[1]
        else:
            target_strike = current_price * 1.01
            target_expiry = whale_df.index.get_level_values('expiry_date')[0]
    else:
        target_df = whale_df[whale_df.index.get_level_values('strike') < current_price]
        if not target_df.empty:
            target_row = target_df.loc[target_df['put_gamma_exposure'].idxmax()]
            target_strike = target_row.name[0]
            target_expiry = target_row.name[1]
        else:
            target_strike = current_price * 0.99
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

def predict_eod_price(current_price, intraday_context, whale_df, time_weight=0.5):
    """Predict end-of-day price considering time decay of options"""
    if whale_df.empty or not intraday_context:
        return current_price, 0
    
    today = datetime.now(pytz.timezone('US/Eastern')).strftime('%Y-%m-%d')
    today_exp = whale_df[whale_df.index.get_level_values('expiry_date') == today]
    
    whale_df_weighted = today_exp if not today_exp.empty else whale_df
    
    call_pain = whale_df_weighted.apply(
        lambda row: row['call_openInterest'] * abs(row.name[0] - current_price), axis=1
    )
    put_pain = whale_df_weighted.apply(
        lambda row: row['put_openInterest'] * abs(row.name[0] - current_price), axis=1
    )
    total_pain = call_pain + put_pain
    
    pain_by_strike = total_pain.groupby(level='strike').sum()
    max_pain_strike = pain_by_strike.idxmin() if not pain_by_strike.empty else current_price
    
    time_factor = 1 - min(intraday_context.get('time_to_close', 6.5), 6.5) / 6.5
    
    eod_price = (current_price * (1 - time_factor * time_weight)) + (max_pain_strike * time_factor * time_weight)
    confidence = min(time_factor * 100 * (whale_df_weighted['total_oi'].sum() / 10000), 95)
    
    return eod_price, confidence

def analyze_order_flow(ticker_symbol, lookback_minutes=30):
    """Simulate options order flow (placeholder for real data)"""
    try:
        ticker = yf.Ticker(ticker_symbol)
        price_data = ticker.history(period='1d', interval='1m')
        current_price = price_data['Close'].iloc[-1] if not price_data.empty else None
        
        if not current_price:
            return pd.DataFrame(), 0, 0
        
        eastern = pytz.timezone('US/Eastern')
        now = datetime.now(eastern)
        
        atm_call_strike = round(current_price * 2) / 2
        atm_put_strike = atm_call_strike
        
        orders = []
        for i in range(lookback_minutes):
            time_ago = now - timedelta(minutes=i)
            weight = (lookback_minutes - i) / lookback_minutes
            
            if np.random.random() < 0.7 * weight:
                size = int(np.random.choice([10, 20, 50, 100, 200, 500, 1000], p=[0.2, 0.3, 0.2, 0.15, 0.1, 0.03, 0.02]))
                premium = round(np.random.uniform(0.5, 5.0), 2)
                strike = atm_call_strike + np.random.choice([-1.0, -0.5, 0, 0.5, 1.0], p=[0.1, 0.2, 0.4, 0.2, 0.1])
                side = np.random.choice(['buy', 'sell'], p=[0.7, 0.3])
                orders.append({
                    'time': time_ago.strftime('%H:%M:%S'),
                    'type': 'call',
                    'strike': strike,
                    'premium': premium,
                    'size': size,
                    'side': side,
                    'value': size * premium * 100,
                    'weight': weight
                })
            
            if np.random.random() < 0.7 * weight:
                size = int(np.random.choice([10, 20, 50, 100, 200, 500, 1000], p=[0.2, 0.3, 0.2, 0.15, 0.1, 0.03, 0.02]))
                premium = round(np.random.uniform(0.5, 5.0), 2)
                strike = atm_put_strike + np.random.choice([-1.0, -0.5, 0, 0.5, 1.0], p=[0.1, 0.2, 0.4, 0.2, 0.1])
                side = np.random.choice(['buy', 'sell'], p=[0.7, 0.3])
                orders.append({
                    'time': time_ago.strftime('%H:%M:%S'),
                    'type': 'put',
                    'strike': strike,
                    'premium': premium,
                    'size': size,
                    'side': side,
                    'value': size * premium * 100,
                    'weight': weight
                })
        
        flow_df = pd.DataFrame(orders)
        
        if flow_df.empty:
            return pd.DataFrame(), 0, 0
        
        flow_df['direction'] = flow_df.apply(
            lambda row: 1 if (row['side'] == 'buy' and row['type'] == 'call') or 
                             (row['side'] == 'sell' and row['type'] == 'put') else -1, 
            axis=1
        )
        
        flow_df['weighted_value'] = flow_df['value'] * flow_df['weight'] * flow_df['direction']
        net_flow = flow_df['weighted_value'].sum()
        
        call_flow = flow_df[flow_df['type'] == 'call']['value'].sum()
        put_flow = flow_df[flow_df['type'] == 'put']['value'].sum()
        flow_ratio = call_flow / put_flow if put_flow > 0 else float('inf')
        
        return flow_df, net_flow, flow_ratio
    
    except Exception as e:
        st.error(f"Error analyzing order flow: {str(e)}")
        return pd.DataFrame(), 0, 0

def predict_intraday_move(current_price, intraday_context, whale_df, flow_df, net_flow, flow_ratio):
    """Predict intraday price movement using all available signals"""
    if whale_df.empty:
        return "Insufficient data", current_price, None, "gray", 0
    
    day_progress = intraday_context.get('day_progress', 0.5) or 0.5
    time_to_close = intraday_context.get('time_to_close', 3)
    
    if day_progress < 0.2:
        gap_effect = (intraday_context.get('gap_pct', 0) or 0) * -0.3
        overnight_weight = 0.6
        flow_weight = 0.2
        gamma_weight = 0.2
    elif day_progress < 0.7:
        gap_effect = 0
        overnight_weight = 0.2
        flow_weight = 0.5
        gamma_weight = 0.3
    else:
        gap_effect = 0
        overnight_weight = 0.1
        flow_weight = 0.3
        gamma_weight = 0.6
    
    net_gxoi = whale_df['net_gxoi'].sum()
    net_dxoi = whale_df['net_dxoi'].sum()
    
    call_max_oi_strike = whale_df.groupby(level='strike')['call_openInterest'].sum().idxmax()
    put_max_oi_strike = whale_df.groupby(level='strike')['put_openInterest'].sum().idxmax()
    
    price_vs_call = (current_price - call_max_oi_strike) / current_price * 100
    price_vs_put = (current_price - put_max_oi_strike) / current_price * 100
    
    flow_signal = net_flow / 100000 if abs(net_flow) > 0 else 0
    flow_polarity = 1 if flow_ratio > 1.5 else (-1 if flow_ratio < 0.67 else 0)
    
    direction_score = (
        gap_effect +
        (net_gxoi / max(abs(net_gxoi), 1)) * gamma_weight +
        (net_dxoi / max(abs(net_dxoi), 1)) * overnight_weight +
        flow_signal * flow_weight * flow_polarity
    )
    
    if time_to_close < 1:
        direction_score *= min(1.5, 1 + (1 - time_to_close))
    
    expected_percent_move = direction_score * 0.2
    
    if direction_score > 0:
        target_price = current_price * (1 + expected_percent_move/100)
        movement = "up"
        color = "green"
    else:
        target_price = current_price * (1 + expected_percent_move/100)
        movement = "down"
        color = "red"
    
    confidence = min(abs(direction_score) * 100, 90)
    
    if time_to_close < 0.5:
        prediction = f"EOD target: ${target_price:.2f} ({expected_percent_move:+.2f}%)"
    elif time_to_close < 1:
        prediction = f"Next hour move {movement} to ${target_price:.2f} ({expected_percent_move:+.2f}%)"
    else:
        prediction = f"Intraday trend {movement} toward ${target_price:.2f} ({expected_percent_move:+.2f}%)"
    
    return prediction, target_price, movement, color, confidence

def plot_volume_trend(whale_df, current_price, ticker_symbol):
    """Bar chart showing weighted volume and gamma trend"""
    if whale_df.empty:
        return None
    
    strike_df = whale_df.groupby(level='strike').sum()
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    ax1.bar(strike_df.index - 0.2, strike_df['call_weighted_volume'], width=0.4, 
            color='green', alpha=0.7, label='Call Weighted Volume')
    ax1.bar(strike_df.index + 0.2, strike_df['put_weighted_volume'], width=0.4, 
            color='red', alpha=0.7, label='Put Weighted Volume')
    ax1.set_xlabel("Strike Price")
    ax1.set_ylabel("Weighted Volume")
    ax1.legend(loc='upper left')
    
    ax2 = ax1.twinx()
    ax2.plot(strike_df.index, strike_df['net_gxoi'], color='purple', 
             linestyle='--', label='Net Gamma Exposure', linewidth=5)
    ax2.set_ylabel("Net Gamma Exposure")
    ax2.legend(loc='upper right')
    
    ax1.axvline(current_price, color='blue', linestyle='--', 
               label=f'Current Price: ${current_price:.2f}')
    ax1.set_title(f"Volume and Gamma Trend Analysis for {ticker_symbol}")
    ax1.grid(True, alpha=0.3)
    
    total_call_wv = strike_df['call_weighted_volume'].sum()
    total_put_wv = strike_df['put_weighted_volume'].sum()
    stats_text = f"Call WV: {total_call_wv:,.0f}\nPut WV: {total_put_wv:,.0f}"
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig

def plot_intraday_heatmap(whale_df, current_price, ticker_symbol, metric='net_gxoi'):
    """Create a heatmap of option activity across strikes and expirations"""
    if whale_df.empty:
        return None
    
    heatmap_data = whale_df[metric].unstack(level='expiry_date').fillna(0)
    heatmap_data = heatmap_data.reindex(columns=sorted(heatmap_data.columns))
    
    from matplotlib.colors import LinearSegmentedColormap
    colors = ['red', 'white', 'green']
    cmap = LinearSegmentedColormap.from_list('gamma_cmap', colors, N=100)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    vmax = max(abs(heatmap_data.min().min()), abs(heatmap_data.max().max()))
    vmin = -vmax
    
    im = ax.imshow(heatmap_data, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
    
    ax.set_yticks(range(len(heatmap_data.index)))
    ax.set_yticklabels([f"${x:.1f}" for x in heatmap_data.index])
    
    ax.set_xticks(range(len(heatmap_data.columns)))
    expiry_labels = []
    for exp in heatmap_data.columns:
        exp_date = datetime.strptime(exp, '%Y-%m-%d').date()
        days_to_exp = (exp_date - datetime.now().date()).days
        expiry_labels.append("Today" if days_to_exp == 0 else "Tom" if days_to_exp == 1 else f"{days_to_exp}d")
    ax.set_xticklabels(expiry_labels)
    
    current_price_idx = np.searchsorted(heatmap_data.index, current_price, side='left')
    ax.axhline(y=current_price_idx, color='blue', linestyle='--', alpha=0.8)
    
    cbar = plt.colorbar(im)
    metric_labels = {
        'net_gxoi': 'Net Gamma Exposure',
        'net_dxoi': 'Net Delta Exposure',
        'total_oi': 'Total Open Interest',
        'net_wv': 'Net Weighted Volume'
    }
    cbar.set_label(metric_labels.get(metric, metric))
    
    metric_titles = {
        'net_gxoi': 'Gamma Exposure',
        'net_dxoi': 'Delta Exposure',
        'total_oi': 'Open Interest',
        'net_wv': 'Net Weighted Volume'
    }
    plt.title(f'{metric_titles.get(metric, metric)} Heatmap for {ticker_symbol}')
    plt.xlabel('Expiration')
    plt.ylabel('Strike Price')
    
    plt.tight_layout()
    return fig

def run_intraday_dashboard():
    st.markdown("<h1 style='text-align: center;'>Intraday Options Flow & Prediction</h1>", unsafe_allow_html=True)
    
    ticker_symbol = st.sidebar.text_input("Ticker Symbol", value='SPY', key='intraday_ticker').upper()
    refresh_seconds = st.sidebar.slider("Auto Refresh (seconds)", 0, 300, 60)
    analyze_button = st.sidebar.button("Analyze Now", key='intraday_analyze')
    
    refresh_placeholder = st.empty()
    
    if 'last_analysis' not in st.session_state:
        st.session_state.last_analysis = None
    
    current_time = datetime.now()
    should_refresh = (
        refresh_seconds > 0 and 
        st.session_state.last_analysis is not None and
        (current_time - st.session_state.last_analysis).total_seconds() >= refresh_seconds
    )
    
    if analyze_button or should_refresh:
        st.session_state.last_analysis = current_time
        
        with st.spinner('Analyzing real-time options data...'):
            intraday_context = get_intraday_context(ticker_symbol)
            
            market_status = intraday_context.get('market_status', 'unknown')
            market_status_color = "green" if market_status == 'market-hours' else "red" if market_status == 'after-hours' else "orange"
            market_hours_text = (f"Market Open - {intraday_context.get('time_to_close', 0):.1f} hours to close" 
                                if market_status == 'market-hours' else f"Market {market_status}")
            
            expirations = get_daily_expirations(ticker_symbol) if ticker_symbol in ['SPY', 'QQQ', 'IWM', 'DIA'] else get_weekly_expirations(ticker_symbol)
            whale_df, strike_df, current_price, price_trend = fetch_whale_positions(ticker_symbol, expirations, price_range_pct=5)
            flow_df, net_flow, flow_ratio = analyze_order_flow(ticker_symbol)
            
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col1:
                st.metric("Current Price", f"${current_price:.2f}" if current_price else "N/A")
                st.metric("Day Change", f"{price_trend:+.2f}%" if price_trend else "N/A")
                
                st.markdown(f"""
                <div style='background-color: {market_status_color}; color: white; padding: 10px; 
                border-radius: 5px; text-align: center;'>
                {market_hours_text}
                </div>
                """, unsafe_allow_html=True)
                
                if not flow_df.empty:
                    st.subheader("Recent Options Flow")
                    call_flow = flow_df[flow_df['type'] == 'call']['value'].sum()
                    put_flow = flow_df[flow_df['type'] == 'put']['value'].sum()
                    st.markdown(f"""
                    - Call Premium: ${call_flow:,.0f}
                    - Put Premium: ${put_flow:,.0f}
                    - Call/Put Ratio: {flow_ratio:.2f}
                    """)
                    
                    st.markdown("#### Large Orders (Last 30m)")
                    large_orders = flow_df[flow_df['size'] >= 100].sort_values('time', ascending=False).head(5)
                    for _, order in large_orders.iterrows():
                        st.markdown(f"""
                        {order['time']} - {order['side'].upper()} {order['size']} {ticker_symbol} 
                        {order['strike']} {order['type'].upper()} @ ${order['premium']:.2f}
                        """)
            
            with col2:
                if not whale_df.empty and current_price:
                    prediction, target_price, movement, color, confidence = predict_intraday_move(
                        current_price, intraday_context, whale_df, flow_df, net_flow, flow_ratio
                    )
                    
                    st.markdown(f"""
                    <div style='background-color: {color}; color: white; padding: 15px; 
                    border-radius: 5px; text-align: center; font-size: 18px;'>
                    <strong>Prediction: {prediction}</strong><br>
                    Confidence: {confidence:.0f}%
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if intraday_context.get('market_status') == 'market-hours':
                        eod_price, eod_confidence = predict_eod_price(current_price, intraday_context, whale_df)
                        eod_change = (eod_price - current_price) / current_price * 100
                        eod_color = "green" if eod_change > 0 else "red" if eod_change < 0 else "gray"
                        
                        st.markdown(f"""
                        <div style='background-color: {eod_color}; color: white; padding: 10px; 
                        border-radius: 5px; text-align: center; margin-top: 10px;'>
                        <strong>Projected EOD: ${eod_price:.2f} ({eod_change:+.2f}%)</strong><br>
                        Confidence: {eod_confidence:.0f}%
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.subheader("Strike Price Heat Map")
                    fig = plot_volume_trend(whale_df, current_price, ticker_symbol)
                    if fig:
                        st.pyplot(fig)
                    
                    if len(whale_df.index.get_level_values('expiry_date').unique()) > 1:
                        st.subheader("Gamma Exposure Across Expirations")
                        heatmap_fig = plot_intraday_heatmap(whale_df, current_price, ticker_symbol)
                        if heatmap_fig:
                            st.pyplot(heatmap_fig)
            
            with col3:
                st.subheader("Key Price Levels")
                if not strike_df.empty:
                    call_oi_by_strike = strike_df['call_openInterest'].sort_values(ascending=False)
                    put_oi_by_strike = strike_df['put_openInterest'].sort_values(ascending=False)
                    
                    resistance_levels = call_oi_by_strike[call_oi_by_strike.index > current_price].head(3)
                    support_levels = put_oi_by_strike[put_oi_by_strike.index < current_price].head(3)
                    
                    st.markdown("#### Resistance Levels (Call Walls)")
                    for strike, oi in resistance_levels.items():
                        distance = ((strike - current_price) / current_price) * 100
                        st.markdown(f"${strike:.2f} (+{distance:.2f}%) - {oi:,.0f} calls")
                    
                    st.markdown("#### Support Levels (Put Walls)")
                    for strike, oi in support_levels.items():
                        distance = ((current_price - strike) / current_price) * 100
                        st.markdown(f"${strike:.2f} (-{distance:.2f}%) - {oi:,.0f} puts")
                
                st.subheader("Gamma Exposure by Expiry")
                if not whale_df.empty:
                    gamma_by_exp = whale_df.groupby(level='expiry_date')['net_gxoi'].sum().sort_index()
                    exp_data = [
                        {
                            "Expiry": "Today" if (datetime.strptime(exp, '%Y-%m-%d').date() - datetime.now().date()).days == 0 else f"{(datetime.strptime(exp, '%Y-%m-%d').date() - datetime.now().date()).days}d ({exp})",
                            "Net Gamma": gamma
                        }
                        for exp, gamma in gamma_by_exp.items()
                    ]
                    gamma_df = pd.DataFrame(exp_data)
                    # Adjust background_gradient for compatibility
                    max_gamma = gamma_df['Net Gamma'].abs().max()
                    styled_df = gamma_df.style.format({'Net Gamma': '{:,.0f}'}).background_gradient(
                        subset=['Net Gamma'], cmap='coolwarm', vmin=-max_gamma, vmax=max_gamma
                    )
                    st.dataframe(styled_df, use_container_width=True)
        
        if refresh_seconds > 0:
            next_refresh = current_time + timedelta(seconds=refresh_seconds)
            refresh_placeholder.info(f"Auto-refreshing every {refresh_seconds} seconds. Next refresh at {next_refresh.strftime('%H:%M:%S')}")
    else:
        st.info("Click 'Analyze Now' to fetch real-time options data and predictions")
        if refresh_seconds > 0:
            refresh_placeholder.info(f"Auto-refresh set to {refresh_seconds} seconds")

def run_standard_analysis():
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
                    display_df = strike_df[['call_openInterest', 'put_openInterest', 'total_oi', 'call_volume', 'put_volume', 'net_vol',
                                           'call_delta_exposure', 'put_delta_exposure', 'net_dxoi', 'call_gamma_exposure', 'put_gamma_exposure', 'net_gxoi']]
                    st.dataframe(
                        display_df.style.format({
                            'call_openInterest': '{:,.0f}', 'put_openInterest': '{:,.0f}', 'total_oi': '{:,.0f}',
                            'call_volume': '{:,.0f}', 'put_volume': '{:,.0f}', 'net_vol': '{:+,.0f}',
                            'call_delta_exposure': '{:,.0f}', 'put_delta_exposure': '{:,.0f}', 'net_dxoi': '{:,.0f}',
                            'call_gamma_exposure': '{:,.0f}', 'put_gamma_exposure': '{:,.0f}', 'net_gxoi': '{:,.0f}'
                        }).background_gradient(subset=['total_oi', 'net_gxoi'], cmap='Blues'),
                        use_container_width=True
                    )
                    
                    direction, target_strike, target_expiry, color = predict_price_direction(current_price, whale_df, price_trend)
                    
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
                    
                    st.subheader("Weighted Volume and Gamma Trend Analysis")
                    fig = plot_volume_trend(whale_df, current_price, ticker_symbol)
                    if fig:
                        st.pyplot(fig)
                    
                    st.subheader("Key Insights")
                    total_call_oi = strike_df['call_openInterest'].sum()
                    total_put_oi = strike_df['put_openInterest'].sum()
                    total_call_vol = strike_df['call_volume'].sum()
                    total_put_vol = strike_df['put_volume'].sum()
                    total_call_wv = strike_df['call_weighted_volume'].sum()
                    total_put_wv = strike_df['put_weighted_volume'].sum()
                    net_gxoi = strike_df['net_gxoi'].sum()
                    net_dxoi = strike_df['net_dxoi'].sum()
                    
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
                    - **Dealer Exposure**:
                      - Net Gamma (GxOI): {net_gxoi:,.0f}
                      - Net Delta (DxOI): {net_dxoi:,.0f}
                    - **Momentum Factors**:
                      - Price Trend (1h): {price_trend:+.2f}%
                      - Volume Momentum: {total_call_wv - total_put_wv:+,.0f}
                    - **Prediction**: {direction}
                    - **Target Details**:
                      - Strike: ${target_strike:.2f}
                      - Expiration: {target_expiry}
                    - **Note**: Prediction incorporates gamma and delta exposure
                    """)
                else:
                    st.warning("No data available within the specified price range")
            else:
                st.warning("No weekly expirations found for this ticker")

def run():
    st.sidebar.title("Mode Selection")
    mode = st.sidebar.radio("Select Analysis Mode", ["Standard Whale Analysis", "Intraday Prediction"])
    
    if mode == "Standard Whale Analysis":
        run_standard_analysis()
    else:
        run_intraday_dashboard()

if __name__ == "__main__":
    run()
