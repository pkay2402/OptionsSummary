import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
import datetime

def calculate_gann_swings(data, swing_period=3, sensitivity=0.001):
    """
    Enhanced Gann swing calculation with adjustable sensitivity
    """
    highs = data['High'].values
    lows = data['Low'].values
    
    swings = pd.Series(index=data.index, dtype='object')
    swing_prices = pd.Series(index=data.index, dtype='float')
    
    # Initialize
    swings.iloc[:swing_period] = 'Neutral'
    last_direction = 'Neutral'
    min_price_change = np.mean(highs) * sensitivity
    
    for i in range(swing_period, len(highs)):
        window_highs = highs[i-swing_period:i+1]
        window_lows = lows[i-swing_period:i+1]
        
        if (window_highs[-1] > window_highs[-2] and 
            window_highs[-1] > window_highs[-3] and 
            last_direction != 'Up' and
            (window_highs[-1] - window_lows[-3]) > min_price_change):
            swings.iloc[i] = 'Up'
            swing_prices.iloc[i] = window_highs[-1]
            last_direction = 'Up'
            
        elif (window_lows[-1] < window_lows[-2] and 
              window_lows[-1] < window_lows[-3] and 
              last_direction != 'Down' and
              (window_highs[-3] - window_lows[-1]) > min_price_change):
            swings.iloc[i] = 'Down'
            swing_prices.iloc[i] = window_lows[-1]
            last_direction = 'Down'
            
        else:
            swings.iloc[i] = 'Neutral'
    
    return swings, swing_prices

def plot_trend_lines(ax, data, up_swings, down_swings):
    """Plot trend lines connecting swing points"""
    high_dates = up_swings.index.values
    high_prices = up_swings['Swing_Prices'].values
    for i in range(len(high_dates)-1):
        ax.plot([high_dates[i], high_dates[i+1]], 
                [high_prices[i], high_prices[i+1]], 
                '--', color='green', alpha=0.5)
    
    low_dates = down_swings.index.values
    low_prices = down_swings['Swing_Prices'].values
    for i in range(len(low_dates)-1):
        ax.plot([low_dates[i], low_dates[i+1]], 
                [low_prices[i], low_prices[i+1]], 
                '--', color='red', alpha=0.5)

def add_gann_time_factors(ax, data):
    """Add Gann time factor lines with improved visibility"""
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Different colors for each cycle
    cycles = [20, 40, 80]
    
    start_date = data.index[0]
    for idx, days in enumerate(cycles):
        current_date = start_date
        while current_date <= data.index[-1]:
            ax.axvline(x=current_date, color=colors[idx], linestyle=':', 
                      alpha=0.3, label=f'{days}-Day Cycle' if current_date == start_date else "")
            current_date = current_date + pd.Timedelta(days=days)

def plot_swing_chart(data, swings):
    """Create a separate swing chart"""
    fig_swing = plt.figure(figsize=(14, 3))
    ax_swing = fig_swing.add_subplot(111)
    
    swing_points = data[data['Gann_Swing'] != 'Neutral'].copy()
    dates = swing_points.index.values
    prices = swing_points['Swing_Prices'].values
    directions = swing_points['Gann_Swing'].values
    
    for i in range(len(dates)-1):
        color = 'green' if directions[i] == 'Up' else 'red'
        ax_swing.plot([dates[i], dates[i+1]], [prices[i], prices[i+1]], 
                     color=color, linewidth=1.5)
    
    ax_swing.set_title('Gann Swing Chart')
    ax_swing.grid(True, alpha=0.3)
    ax_swing.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax_swing.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig_swing

def run():
    st.title("Gann Swing Analysis")
    
    # Sidebar controls
    st.sidebar.header("Parameters")
    ticker = st.sidebar.text_input("Ticker Symbol", value="SPY")
    
    # Date inputs with default values
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=180)
    
    start_date = st.sidebar.date_input("Start Date", value=start_date)
    end_date = st.sidebar.date_input("End Date", value=end_date)
    
    swing_period = st.sidebar.slider("Swing Period", min_value=2, max_value=5, value=3)
    sensitivity = st.sidebar.slider("Sensitivity", min_value=0.0001, max_value=0.005, 
                                  value=0.001, format="%.4f")
    
    show_time_cycles = st.sidebar.multiselect("Show Time Cycles", 
                                            ["20-Day", "40-Day", "80-Day"],
                                            default=["40-Day"])
    
    try:
        # Download data
        data = yf.download(ticker, start=start_date, end=end_date)
        
        if data.empty:
            st.error(f"No data found for {ticker}")
            return
        
        # Calculate Gann swings
        swings, swing_prices = calculate_gann_swings(data, swing_period, sensitivity)
        data['Gann_Swing'] = swings
        data['Swing_Prices'] = swing_prices
        
        # Create main price chart
        fig_price = plt.figure(figsize=(14, 7))
        ax_price = fig_price.add_subplot(111)
        
        # Plot price
        ax_price.plot(data.index, data['Close'], label='Price', color='blue', alpha=0.7, linewidth=1)
        
        # Plot swing points
        up_swings = data[data['Gann_Swing'] == 'Up']
        down_swings = data[data['Gann_Swing'] == 'Down']
        
        ax_price.scatter(up_swings.index, up_swings['Swing_Prices'], 
                        color='green', marker='^', s=100, label='Swing High')
        ax_price.scatter(down_swings.index, down_swings['Swing_Prices'], 
                        color='red', marker='v', s=100, label='Swing Low')
        
        # Add trend lines
        plot_trend_lines(ax_price, data, up_swings, down_swings)
        
        # Add selected time cycles
        if show_time_cycles:
            add_gann_time_factors(ax_price, data)
        
        # Customize the plot
        ax_price.set_title(f'{ticker} Gann Swing Analysis ({swing_period}-Bar)', pad=20)
        ax_price.set_ylabel('Price')
        ax_price.grid(True, alpha=0.3)
        
        # Add legend
        legend_elements = ax_price.get_legend_handles_labels()[0]
        legend_elements.extend([
            Line2D([0], [0], color='green', linestyle='--', alpha=0.5, label='High Trend'),
            Line2D([0], [0], color='red', linestyle='--', alpha=0.5, label='Low Trend')
        ])
        ax_price.legend(handles=legend_elements)
        
        # Format dates
        ax_price.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax_price.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Create swing chart
        fig_swing = plot_swing_chart(data, swings)
        
        # Display charts in Streamlit
        st.pyplot(fig_price)
        st.pyplot(fig_swing)
        
        # Display statistics
        st.subheader("Swing Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Swing Points", 
                     len(up_swings) + len(down_swings))
        
        with col2:
            st.metric("Swing Highs", len(up_swings))
        
        with col3:
            st.metric("Swing Lows", len(down_swings))
            
    except Exception as e:
        st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    run()
