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
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from scipy import stats

# Setup logging
logging.basicConfig(filename='whale_positioning.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

@st.cache_data(ttl=300)
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def get_weekly_expirations(ticker_symbol, num_weeks=8):
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
            if (exp_date.weekday() == 4 and 0 <= days_to_exp <= (num_weeks * 7 + 7)):
                weekly_exps.append({'date': exp, 'days': days_to_exp})
        
        result = sorted(weekly_exps, key=lambda x: x['days'])[:num_weeks]
        logging.info(f"Fetched {len(result)} weekly expirations for {ticker_symbol}") if result else logging.warning(f"No weekly expirations found for {ticker_symbol}")
        return result
    except Exception as e:
        logging.error(f"Error fetching expirations for {ticker_symbol}: {str(e)}")
        st.error(f"Error fetching weekly expirations: {str(e)}")
        return []

def fetch_option_chain(ticker, exp, price_min, price_max):
    """Fetch and process option chain"""
    try:
        opt = ticker.option_chain(exp['date'])
        calls = opt.calls[(opt.calls['strike'] >= price_min) & (opt.calls['strike'] <= price_max)][['strike', 'openInterest', 'volume', 'impliedVolatility']].copy()
        calls['type'] = 'call'
        calls['days_to_exp'] = exp['days']
        calls['expiry_date'] = exp['date']
        
        puts = opt.puts[(opt.puts['strike'] >= price_min) & (opt.puts['strike'] <= price_max)][['strike', 'openInterest', 'volume', 'impliedVolatility']].copy()
        puts['type'] = 'put'
        puts['days_to_exp'] = exp['days']
        puts['expiry_date'] = exp['date']
        
        return calls, puts
    except Exception as e:
        logging.warning(f"Error fetching option chain for {exp['date']}: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()

@st.cache_data(ttl=300)
def fetch_whale_positions(ticker_symbol, expirations, price_range_pct=10):
    """Get whale positions data"""
    try:
        ticker = yf.Ticker(ticker_symbol)
        hist = ticker.history(period='1d', interval='1m')
        current_price = hist['Close'].iloc[-1] if not hist.empty else None
        if not current_price:
            raise ValueError("Unable to fetch current price")
            
        last_hour = hist.tail(60)
        price_trend = (last_hour['Close'].iloc[-1] - last_hour['Close'].iloc[0]) / last_hour['Close'].iloc[0] * 100
        
        price_min = current_price * (1 - price_range_pct/100)
        price_max = current_price * (1 + price_range_pct/100)
        
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(lambda exp: fetch_option_chain(ticker, exp, price_min, price_max), expirations))
        whale_data = [item for sublist in results for item in sublist if not item.empty]
        
        if not whale_data:
            raise ValueError("No option data within price range")
        
        df = pd.concat(whale_data)
        df['price_weight'] = 1 - (abs(df['strike'] - current_price) / (current_price * price_range_pct/100))
        df['time_weight'] = 1 / (df['days_to_exp'] + 1)
        df['weighted_volume'] = df['volume'] * df['price_weight'] * df['time_weight']
        
        agg_df = df.groupby(['strike', 'type', 'expiry_date']).agg({
            'openInterest': 'sum',
            'volume': 'sum',
            'weighted_volume': 'sum',
            'impliedVolatility': 'mean'
        }).reset_index()
        
        pivot_df = agg_df.pivot(index=['strike', 'expiry_date'], columns='type', 
                              values=['openInterest', 'volume', 'weighted_volume', 'impliedVolatility']).fillna(0)
        pivot_df.columns = ['call_oi', 'put_oi', 'call_vol', 'put_vol', 'call_wv', 'put_wv', 'call_iv', 'put_iv']
        
        pivot_df['total_oi'] = pivot_df['call_oi'] + pivot_df['put_oi']
        pivot_df['net_vol'] = pivot_df['call_vol'] - pivot_df['put_vol']
        pivot_df['net_wv'] = pivot_df['call_wv'] - pivot_df['put_wv']
        
        strike_df = pivot_df.groupby(level='strike').sum()
        logging.info(f"Processed whale positions for {ticker_symbol}")
        return pivot_df, strike_df, current_price, price_trend
    except Exception as e:
        logging.error(f"Error fetching whale positions for {ticker_symbol}: {str(e)}")
        st.error(f"Error fetching whale positions: {str(e)}")
        return pd.DataFrame(), pd.DataFrame(), None, None

def predict_price_direction(current_price, whale_df, price_trend, weekly_expiries):
    """Enhanced prediction with multiple modeling techniques"""
    if whale_df.empty or len(whale_df) < 1:
        direction = "Neutral"
        target_strike = 0
        target_expiry = "N/A"
        confidence = 0
        color = "gray"
        weekly_message = monthly_message = "Weekly Target: N/A"
        message = f"{direction} toward ${target_strike:.2f} for {target_expiry} (Confidence: {confidence:.0f}%)<br>{weekly_message}<br>{monthly_message}"
        logging.warning("Whale DataFrame is empty; returning neutral prediction")
        return message, target_strike, target_expiry, color, confidence

    strikes = whale_df.index.get_level_values('strike')
    days_to_exp = pd.Series(
        whale_df.index.get_level_values('expiry_date').map(
            lambda x: (pd.to_datetime(x) - pd.Timestamp.now()).days
        ),
        index=whale_df.index
    )

    features = pd.DataFrame({
        'strike_distance': (strikes - current_price) / current_price,
        'call_put_ratio': whale_df['call_wv'] / (whale_df['put_wv'] + 1e-6),
        'oi_concentration': whale_df['call_oi'] + whale_df['put_oi'],
        'volume_momentum': whale_df['call_wv'] - whale_df['put_wv'],
        'iv_spread': whale_df['call_iv'] - whale_df['put_iv'],
        'time_decay': np.exp(-0.05 * days_to_exp),
        'price_trend': price_trend / 100
    }, index=whale_df.index)

    features = features.fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    
    if X_scaled.shape[0] == 0 or X_scaled.shape[1] != features.shape[1]:
        logging.error(f"Invalid X_scaled shape: {X_scaled.shape}; expected ({features.shape[0]}, {features.shape[1]})")
        raise ValueError("Feature scaling produced an invalid shape")

    target = np.where(strikes > current_price, 1, -1)
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    rf_model.fit(X_scaled, target)
    feature_importance = dict(zip(features.columns, rf_model.feature_importances_))

    rf_predictions = rf_model.predict(X_scaled)
    rf_prediction = np.mean(rf_predictions)

    volume_zscore = stats.zscore(whale_df['call_wv'] - whale_df['put_wv']).mean()
    oi_zscore = stats.zscore(whale_df['call_oi'] - whale_df['put_oi']).mean()
    statistical_score = (volume_zscore + oi_zscore) / 2

    vol_regime = (whale_df['call_iv'].mean() + whale_df['put_iv'].mean()) / 2
    iv_skew = (whale_df['call_iv'] - whale_df['put_iv']).mean()
    vol_adjusted_score = (whale_df['call_wv'].sum() - whale_df['put_wv'].sum()) * (1 - vol_regime)

    rf_weight = 0.4 + feature_importance.get('volume_momentum', 0.1)
    stat_weight = 0.3 * (1 + abs(statistical_score))
    vol_weight = 0.3 * (1 - vol_regime)
    total_weight = rf_weight + stat_weight + vol_weight
    direction_score = (
        (rf_prediction * rf_weight) +
        (statistical_score * stat_weight) +
        (vol_adjusted_score / max(abs(vol_adjusted_score), 1) * vol_weight)
    ) / total_weight

    def calculate_target_scores(df, direction, expiry_date=None):
        if expiry_date is not None:
            df_features = pd.DataFrame({
                'strike_distance': (df.index - current_price) / current_price,
                'call_put_ratio': df['call_wv'] / (df['put_wv'] + 1e-6),
                'oi_concentration': df['call_oi'] + df['put_oi'],
                'volume_momentum': df['call_wv'] - df['put_wv'],
                'iv_spread': df['call_iv'] - df['put_iv'],
                'time_decay': np.exp(-0.05 * (pd.to_datetime(expiry_date) - pd.Timestamp.now()).days),
                'price_trend': price_trend / 100
            }, index=df.index)
            days_to_exp_subset = pd.Series(
                (pd.to_datetime(expiry_date) - pd.Timestamp.now()).days,
                index=df.index
            )
        else:
            df_features = features.loc[df.index]
            days_to_exp_subset = days_to_exp.loc[df.index]

        df_scaled = scaler.transform(df_features)
        if len(df_scaled) != len(df):
            logging.error(f"Length mismatch: df_scaled ({len(df_scaled)}) vs df ({len(df)}), df index: {df.index}")
            raise ValueError("Scaled features length does not match DataFrame length")
        df['probability'] = rf_model.predict(df_scaled)
        df['weighted_score'] = (
            df['probability'] * 
            (df['call_wv'] if direction > 0 else df['put_wv']) * 
            np.exp(-0.05 * days_to_exp_subset)
        )
        # Log weighted scores for debugging
        logging.debug(f"Weighted scores for {expiry_date or 'target'}: {df['weighted_score'].to_dict()}")
        return df

    # Target selection with constraints
    if direction_score > 0:
        target_df = whale_df[strikes > current_price].copy()
        if not target_df.empty:
            target_df = calculate_target_scores(target_df, 1)
            target_row = target_df.loc[target_df['weighted_score'].idxmax()]
            target_strike = target_row.name[0]
            target_expiry = target_row.name[1]
        else:
            target_strike = current_price
            target_expiry = whale_df.index.get_level_values('expiry_date')[0]
    else:
        target_df = whale_df[strikes < current_price].copy()
        if not target_df.empty:
            target_df = calculate_target_scores(target_df, -1)
            target_row = target_df.loc[target_df['weighted_score'].idxmax()]
            target_strike = target_row.name[0]
            target_expiry = target_row.name[1]
        else:
            target_strike = current_price
            target_expiry = whale_df.index.get_level_values('expiry_date')[0]

    # Confidence and direction
    model_agreement = len([s for s in [rf_prediction, statistical_score, vol_adjusted_score] 
                         if np.sign(s) == np.sign(direction_score)]) / 3
    confidence = min(abs(direction_score) * 100 * model_agreement, 95)

    threshold = 0.3 * (1 + vol_regime)
    if direction_score > threshold:
        direction = "Bullish"
        color = "green"
    elif direction_score < -threshold:
        direction = "Bearish"
        color = "red"
    else:
        direction = "Neutral"
        color = "gray"
        # Override extreme targets for Neutral with low confidence
        if confidence < 25:  # Arbitrary threshold; adjust as needed
            target_strike = current_price
            target_expiry = whale_df.index.get_level_values('expiry_date')[0]
            logging.info(f"Neutral with low confidence ({confidence:.0f}%), resetting target to current price: ${current_price:.2f}")

    today = pd.Timestamp.now().date()
    next_week = today + timedelta(days=7)
    next_month = (today.replace(day=1) + timedelta(days=32)).replace(day=1)
    weekly_target = monthly_target = weekly_target_strike = monthly_target_strike = None

    for expiry in weekly_expiries:
        exp_date = datetime.strptime(expiry['date'], '%Y-%m-%d').date()
        if exp_date <= next_week and weekly_target is None:
            weekly_target = expiry['date']
        if exp_date <= next_month and monthly_target is None:
            monthly_target = expiry['date']

    # Weekly target with range constraint
    if weekly_target:
        weekly_df = whale_df.xs(weekly_target, level='expiry_date')
        if not weekly_df.empty:
            weekly_df = calculate_target_scores(weekly_df, direction_score, expiry_date=weekly_target)
            if direction_score > 0 and not weekly_df[weekly_df.index > current_price].empty:
                weekly_targets = weekly_df[weekly_df.index > current_price]
                weekly_target_strike = weekly_targets.loc[weekly_targets['weighted_score'].idxmax()].name
            elif direction_score <= 0 and not weekly_df[weekly_df.index < current_price].empty:
                weekly_targets = weekly_df[weekly_df.index < current_price]
                weekly_target_strike = weekly_targets.loc[weekly_targets['weighted_score'].idxmax()].name
            # Constrain weekly target to ±5% unless confidence is high
            if weekly_target_strike and confidence < 50:  # Adjust threshold as needed
                max_drop = current_price * 0.95  # -5%
                max_rise = current_price * 1.05  # +5%
                if weekly_target_strike < max_drop or weekly_target_strike > max_rise:
                    weekly_target_strike = current_price
                    logging.info(f"Weekly target ${weekly_target_strike:.2f} outside ±5% range, reset to ${current_price:.2f}")

    # Monthly target (less constrained)
    if monthly_target:
        monthly_df = whale_df.xs(monthly_target, level='expiry_date')
        if not monthly_df.empty:
            monthly_df = calculate_target_scores(monthly_df, direction_score, expiry_date=monthly_target)
            if direction_score > 0 and not monthly_df[monthly_df.index > current_price].empty:
                monthly_target_strike = monthly_df[monthly_df.index > current_price]['weighted_score'].idxmax()
            elif direction_score <= 0 and not monthly_df[monthly_df.index < current_price].empty:
                monthly_target_strike = monthly_df[monthly_df.index < current_price]['weighted_score'].idxmax()

    weekly_message = f"Weekly Target: ${weekly_target_strike:.2f} ({weekly_target})" if weekly_target_strike else "Weekly Target: N/A"
    monthly_message = f"Monthly Target: ${monthly_target_strike:.2f} ({monthly_target})" if monthly_target_strike else "Monthly Target: N/A"

    logging.info(f"Prediction for {direction}: Score={direction_score:.2f}, Confidence={confidence:.0f}%, Features={feature_importance}")
    return f"{direction} toward ${target_strike:.2f} for {target_expiry} (Confidence: {confidence:.0f}%)<br>{weekly_message}<br>{monthly_message}", target_strike, target_expiry, color, confidence

def plot_volume_trend(whale_df, current_price, ticker_symbol):
    """Interactive volume trend plot"""
    if whale_df.empty:
        return None
    strike_df = whale_df.groupby(level='strike').sum()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=strike_df.index, y=strike_df['call_wv'], name='Call Weighted Volume', marker_color='green', opacity=0.7))
    fig.add_trace(go.Bar(x=strike_df.index, y=strike_df['put_wv'], name='Put Weighted Volume', marker_color='red', opacity=0.7))
    fig.add_vline(x=current_price, line_dash="dash", line_color="blue", annotation_text=f"Current: ${current_price:.2f}")
    fig.update_layout(
        title=f"Weighted Volume Trend Analysis for {ticker_symbol}",
        xaxis_title="Strike Price",
        yaxis_title="Weighted Volume",
        barmode='group',
        legend=dict(x=0.01, y=0.99),
        annotations=[dict(x=0.02, y=0.98, xref="paper", yref="paper", 
                         text=f"Call WV: {strike_df['call_wv'].sum():,.0f}<br>Put WV: {strike_df['put_wv'].sum():,.0f}",
                         showarrow=False, bgcolor="white", opacity=0.8)]
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
    st.markdown("<h1 style='text-align: center;'>Smart Whale Positioning Analysis</h1>", unsafe_allow_html=True)
    st.sidebar.markdown("### Configuration")
    ticker_input = st.sidebar.text_input("Ticker Symbols (comma-separated)", value='SPY').upper()
    tickers = [t.strip() for t in ticker_input.split(',')]
    price_range_pct = st.sidebar.slider("Price Range (%)", 5, 20, 10)
    num_weeks = st.sidebar.slider("Weeks of Expirations", 4, 12, 8)
    analyze_button = st.sidebar.button("Analyze")
    
    if 'analysis_data' not in st.session_state:
        st.session_state.analysis_data = {}
    
    if analyze_button:
        valid_tickers = [t for t in tickers if validate_ticker(t)]
        if not valid_tickers:
            st.error("No valid tickers provided")
            return
        
        progress_bar = st.progress(0)
        step = 100 / (len(valid_tickers) * 3)
        
        for i, ticker_symbol in enumerate(valid_tickers):
            with st.spinner(f'Analyzing whale positions for {ticker_symbol}...'):
                weekly_exps = get_weekly_expirations(ticker_symbol, num_weeks)
                progress_bar.progress(int((i * 3 + 1) * step))
                
                if weekly_exps:
                    whale_df, strike_df, current_price, price_trend = fetch_whale_positions(ticker_symbol, weekly_exps, price_range_pct)
                    progress_bar.progress(int((i * 3 + 2) * step))
                    
                    if not whale_df.empty and current_price:
                        st.subheader(f"{ticker_symbol} - Options Data within {price_range_pct}% of Current Price (${current_price:.2f})")
                        display_df = strike_df[['call_oi', 'put_oi', 'total_oi', 'call_vol', 'put_vol', 'net_vol']]
                        st.dataframe(display_df.style.format("{:,.0f}").background_gradient(cmap='Blues'), use_container_width=True)
                        
                        direction, target_strike, target_expiry, color, confidence = predict_price_direction(current_price, whale_df, price_trend, weekly_exps)
                        
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
                            st.plotly_chart(fig, use_container_width=True)
                        
                        st.subheader("Key Insights")
                        total_call_oi = strike_df['call_oi'].sum()
                        total_put_oi = strike_df['put_oi'].sum()
                        total_call_vol = strike_df['call_vol'].sum()
                        total_put_vol = strike_df['put_vol'].sum()
                        total_call_wv = strike_df['call_wv'].sum()
                        total_put_wv = strike_df['put_wv'].sum()
                        avg_call_iv = whale_df['call_iv'].mean()
                        avg_put_iv = whale_df['put_iv'].mean()
                        insights = f"""
                        - **Analysis Range**: Strikes between ${current_price*(1-price_range_pct/100):.2f} and ${current_price*(1+price_range_pct/100):.2f}
                        - **Open Interest**: Calls: {total_call_oi:,.0f}, Puts: {total_put_oi:,.0f}, Ratio: {total_call_oi/total_put_oi if total_put_oi > 0 else 'N/A':.2f}
                        - **Today's Volume**: Calls: {total_call_vol:,.0f}, Puts: {total_put_vol:,.0f}, Weighted Calls: {total_call_wv:,.0f}, Weighted Puts: {total_put_wv:,.0f}
                        - **Implied Volatility**: Avg Call IV: {avg_call_iv:.2%}, Avg Put IV: {avg_put_iv:.2%}
                        - **Momentum Factors**: Price Trend (1h): {price_trend:+.2f}%, Volume Momentum: {total_call_wv - total_put_wv:+,.0f}
                        - **Prediction**: {direction} toward ${target_strike:.2f} for {target_expiry}
                        """
                        st.markdown(insights)
                        
                        csv = strike_df.to_csv(index=True)
                        st.download_button(f"Download {ticker_symbol} Data", csv, f"{ticker_symbol}_whale_data.csv", "text/csv")
                        st.session_state.analysis_data[ticker_symbol] = {
                            'current_price': current_price,
                            'direction': direction,
                            'confidence': confidence,
                            'target_strike': target_strike
                        }
                    else:
                        st.warning(f"No data available within the specified price range for {ticker_symbol}")
                else:
                    st.warning(f"No weekly expirations found for {ticker_symbol}")
                progress_bar.progress(int((i * 3 + 3) * step))
        
        if len(st.session_state.analysis_data) > 1:
            st.subheader("Comparative Analysis")
            comp_df = pd.DataFrame(st.session_state.analysis_data).T
            st.dataframe(comp_df.style.format({'current_price': '${:.2f}', 'target_strike': '${:.2f}', 'confidence': '{:.0f}%'}), use_container_width=True)

if __name__ == "__main__":
    run()
