import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
import pytz

# Function to calculate DeMark Sequence Counter (improved)
def calculate_demark_sequence(df):
    # Make a copy to avoid modifying the original dataframe
    df_seq = df.copy()
    
    # Initialize columns for Setup and Countdown
    df_seq['Setup'] = 0
    df_seq['Countdown'] = 0
    df_seq['Setup_Perfected'] = False
    
    # Setup calculation (TD Sequential)
    setup_count = 0
    for i in range(4, len(df_seq)):
        # True TD Sequential compares close to close 4 bars earlier
        if df_seq['Close'].iloc[i] < df_seq['Close'].iloc[i - 4]:
            setup_count += 1
            if 1 <= setup_count <= 9:
                df_seq.loc[df_seq.index[i], 'Setup'] = setup_count
        else:
            # Reset if condition fails before reaching 9
            setup_count = 0
    
    # Mark perfected setups (close less than low of any of the prior 3 bars)
    for i in range(len(df_seq)):
        if df_seq['Setup'].iloc[i] == 9:
            # Check if setup is perfected
            if (i >= 3 and 
                (df_seq['Close'].iloc[i] < df_seq['Low'].iloc[i-3:i].min())):
                df_seq.loc[df_seq.index[i], 'Setup_Perfected'] = True
    
    # Countdown calculation
    countdown_active = False
    countdown_start_idx = 0
    countdown_count = 0
    
    for i in range(len(df_seq)):
        # Look for a completed setup (9) to start countdown
        if df_seq['Setup'].iloc[i] == 9 and not countdown_active:
            countdown_active = True
            countdown_start_idx = i
            continue
        
        # Only process countdown after a setup is found
        if countdown_active and i > countdown_start_idx:
            # TD Sequential countdown: close less than or equal to low 2 bars earlier
            if df_seq['Close'].iloc[i] <= df_seq['Low'].iloc[i - 2]:
                countdown_count += 1
                if countdown_count <= 13:  # Cap at 13
                    df_seq.loc[df_seq.index[i], 'Countdown'] = countdown_count
                
                # Reset after reaching 13
                if countdown_count == 13:
                    countdown_active = False
                    countdown_count = 0
    
    return df_seq

# Helper function to filter weekends for daily data
def filter_weekends(df, interval):
    if interval == '1d':
        # Keep only weekdays (Monday=0, Sunday=6)
        return df[df.index.dayofweek < 5]
    return df

# Function to create candlestick chart with optimized annotations
def create_demark_chart(df, ticker, interval='1d'):
    # Filter weekends for daily charts
    df = filter_weekends(df, interval)
    
    fig = go.Figure()
    
    # Add candlestick chart with improved appearance
    # Using proper properties for the Candlestick object
    candlestick = go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Candlestick',
        increasing=dict(line=dict(color='#26A69A', width=1), fillcolor='#B2DFDB'),
        decreasing=dict(line=dict(color='#EF5350', width=1), fillcolor='#FFCDD2')
    )
    
    # Set hover text for better information display
    hover_text = []
    for idx, row in df.iterrows():
        date_str = idx.strftime('%Y-%m-%d')
        if interval != '1d':
            date_str = idx.strftime('%Y-%m-%d %H:%M')
        
        text = (f"Date: {date_str}<br>" +
                f"Open: {row['Open']:.2f}<br>" +
                f"High: {row['High']:.2f}<br>" +
                f"Low: {row['Low']:.2f}<br>" +
                f"Close: {row['Close']:.2f}")
                
        # Add setup/countdown info if present
        if row['Setup'] > 0:
            text += f"<br>Setup: {int(row['Setup'])}"
            if row['Setup_Perfected']:
                text += " (Perfected)"
                
        if row['Countdown'] > 0:
            text += f"<br>Countdown: {int(row['Countdown'])}"
            
        hover_text.append(text)
    
    candlestick.text = hover_text
    candlestick.hoverinfo = "text"
    
    # Customize the hover label
    candlestick.hoverlabel = dict(
        bgcolor="white",
        font=dict(size=14, family="Arial")
    )
    
    fig.add_trace(candlestick)
    
    # Create lists to hold annotation data
    setup_annotations = []
    countdown_annotations = []
    
    # Prepare annotations
    for i, row in df.iterrows():
        # Setup annotations (green)
        if row['Setup'] > 0:
            color = 'darkgreen' if row['Setup_Perfected'] else 'limegreen'
            text = str(int(row['Setup']))
            
            setup_annotations.append({
                'x': i,
                'y': row['High'],
                'text': text,
                'color': color,
                'perfected': row['Setup_Perfected']
            })
        
        # Countdown annotations (red)
        if row['Countdown'] > 0:
            text = str(int(row['Countdown']))
            
            # Make 13 stand out more as it's the completion
            size = 16 if row['Countdown'] == 13 else 14
            weight = 'bold' if row['Countdown'] == 13 else 'normal'
            
            countdown_annotations.append({
                'x': i,
                'y': row['Low'],
                'text': text,
                'size': size,
                'weight': weight
            })
    
    # Calculate dynamic spacing factor based on price range
    price_range = df['High'].max() - df['Low'].min()
    up_spacing = 0.005 * (price_range / df['High'].mean())
    down_spacing = 0.005 * (price_range / df['Low'].mean())
    
    # Add annotations to chart with improved positioning
    for ann in setup_annotations:
        fig.add_annotation(
            x=ann['x'],
            y=ann['y'] * (1 + up_spacing),  # Dynamic position above the high
            text=ann['text'],
            showarrow=False,
            font=dict(
                size=14, 
                color=ann['color'], 
                weight='bold'
            ),
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor=ann['color'],
            borderwidth=1,
            borderpad=2,
            opacity=0.9
        )
        
        # Add a star for perfected setups
        if ann['perfected'] and ann['text'] == '9':
            fig.add_annotation(
                x=ann['x'],
                y=ann['y'] * (1 + up_spacing * 2),  # Position above the setup number
                text='★',
                showarrow=False,
                font=dict(size=16, color='gold'),
                bgcolor=None,
                borderpad=0
            )
    
    for ann in countdown_annotations:
        fig.add_annotation(
            x=ann['x'],
            y=ann['y'] * (1 - down_spacing),  # Dynamic position below the low
            text=ann['text'],
            showarrow=False,
            font=dict(
                size=ann['size'], 
                color='darkred' if ann['text'] == '13' else 'red', 
                weight=ann['weight']
            ),
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='red',
            borderwidth=1,
            borderpad=2,
            opacity=0.9
        )
    
    # Add volume bars at bottom with low opacity
    if 'Volume' in df.columns and not df['Volume'].isnull().all():
        # Calculate colors based on price movement
        volume_colors = np.where(df['Close'] >= df['Open'], '#B2DFDB', '#FFCDD2')
        
        # Add volume as a separate trace
        fig.add_trace(go.Bar(
            x=df.index,
            y=df['Volume'],
            name='Volume',
            marker=dict(
                color=volume_colors,
                opacity=0.5
            ),
            yaxis='y2'
        ))
        
        # Create a secondary y-axis for volume
        fig.update_layout(
            yaxis2=dict(
                title="Volume",
                domain=[0, 0.2],  # Bottom 20% of the chart
                showgrid=False
            ),
            yaxis=dict(
                domain=[0.25, 1]  # Top 75% of the chart
            )
        )
    
    # Enhance grid and background
    fig.update_layout(
        plot_bgcolor='rgba(250,250,250,0.9)',
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(200,200,200,0.2)',
            zeroline=False
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(200,200,200,0.2)',
            zeroline=False,
            autorange=True,
            fixedrange=False
        )
    )
    
    # Customize layout
    fig.update_layout(
        title=dict(
            text=f"{ticker} with DeMark TD Sequential",
            font=dict(size=24)
        ),
        yaxis_title="Price",
        xaxis_title="Date",
        xaxis_rangeslider_visible=False,
        template="plotly_white",
        legend_title="Legend",
        height=700,  # Increased height for better visibility
        margin=dict(t=80, b=50, l=50, r=50),
        hovermode="closest"  # Changed from "x unified" to "closest" for better compatibility
    )
    
    # Add custom buttons for different timeframes with safety checks
    buttons = []
    
    # Only add timeframe buttons if there's enough data
    if len(df) >= 30:
        buttons.append(dict(label="1M", method="relayout", args=[{"xaxis.range": [df.index[-30], df.index[-1]]}]))
    
    if len(df) >= 90:
        buttons.append(dict(label="3M", method="relayout", args=[{"xaxis.range": [df.index[-90], df.index[-1]]}]))
    
    if len(df) >= 180:
        buttons.append(dict(label="6M", method="relayout", args=[{"xaxis.range": [df.index[-180], df.index[-1]]}]))
    
    # For YTD check if we have data from start of year
    try:
        # Create YTD start date with the same timezone info as the index
        if hasattr(df.index[-1], 'tzinfo') and df.index[-1].tzinfo is not None:
            ytd_start = datetime(df.index[-1].year, 1, 1, tzinfo=df.index[-1].tzinfo)
        else:
            # If index is timezone-naive, create a naive datetime
            ytd_start = datetime(df.index[-1].year, 1, 1)
            
        # Ensure comparison works by checking timezone consistency
        first_date = df.index[0]
        if hasattr(first_date, 'tzinfo') != hasattr(ytd_start, 'tzinfo'):
            # If one has timezone and other doesn't, make them consistent
            if hasattr(first_date, 'tzinfo') and first_date.tzinfo is not None:
                # Convert ytd_start to same timezone
                ytd_start = pytz.timezone(str(first_date.tzinfo)).localize(datetime(df.index[-1].year, 1, 1))
            else:
                # Make first_date naive by replacing it with its naive equivalent
                first_date = datetime(first_date.year, first_date.month, first_date.day, 
                                     first_date.hour, first_date.minute, first_date.second)
        
        if first_date <= ytd_start:
            buttons.append(dict(label="YTD", method="relayout", args=[{"xaxis.range": [ytd_start, df.index[-1]]}]))
    except Exception as e:
        # If there's any error, skip this button
        print(f"Error creating YTD button: {e}")
    
    if len(df) >= 365:
        buttons.append(dict(label="1Y", method="relayout", args=[{"xaxis.range": [df.index[-365], df.index[-1]]}]))
    
    # Always add the "All" button
    buttons.append(dict(label="All", method="relayout", args=[{"xaxis.range": [df.index[0], df.index[-1]]}]))
    
    # Only add the updatemenus if we have buttons
    if buttons:
        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    buttons=buttons,
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.1,
                    xanchor="left",
                    y=1.1,
                    yanchor="top",
                    bgcolor='rgba(255,255,255,0.9)',
                    bordercolor='rgba(0,0,0,0.5)'
                )
            ]
        )
    
    return fig

# Main run function for module integration with Trading Tools Hub
def run():
    st.title("DeMark TD Sequential Indicator")
    st.markdown("<p style='text-align: center; font-style: italic;'>A technical analysis tool used to identify potential price exhaustion points</p>", unsafe_allow_html=True)
    
    # Create two columns for input controls
    col1, col2 = st.columns([1, 1])

    with col1:
        ticker = st.text_input("Enter Stock Ticker (e.g., AAPL)", "AAPL")
        interval = st.select_slider("Chart Interval", 
                                  options=["1d", "1h", "15m", "5m"],
                                  value="1d")

    with col2:
        start_date = st.date_input("Start Date", datetime.now() - timedelta(days=90))
        end_date = st.date_input("End Date", datetime.now())

    # Add a spinner during data loading
    with st.spinner('Fetching market data...'):
        try:
            if st.button("Generate Chart", type="primary"):
                # Check if ticker is valid
                if not ticker:
                    st.error("Please enter a valid ticker symbol.")
                else:
                    # Fetch data with error handling
                    try:
                        # For intraday data, limit to 60 days max due to API limitations
                        max_start_date = start_date
                        if interval in ["15m", "5m"]:
                            # For intraday data, limiting to last 60 days due to data provider limitations.
                            if (end_date - start_date).days > 60:
                                st.warning("For intraday data, limiting to last 60 days due to data provider limitations.")
                                max_start_date = end_date - timedelta(days=60)
                        
                        # Fetch data
                        stock = yf.Ticker(ticker)
                        # Add one day to end_date to ensure we include today's data
                        end_date_inclusive = end_date + timedelta(days=1)
                        df = stock.history(start=max_start_date, end=end_date_inclusive, interval=interval)
                        
                        if df.empty:
                            st.error(f"No data found for {ticker} in the selected date range.")
                        elif len(df) < 5:  # Need at least 5 bars for DeMark calculation
                            st.error(f"Not enough data points for {ticker}. Try extending the date range.")
                        else:
                            # Calculate DeMark Sequence
                            df_with_demark = calculate_demark_sequence(df)
                            
                            # Create and display chart
                            fig = create_demark_chart(df_with_demark, ticker, interval)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Summary statistics
                            st.subheader("DeMark Sequence Summary")
                            
                            # Count setups and countdowns
                            setup_count = len(df_with_demark[df_with_demark['Setup'] == 9])
                            perfected_setups = len(df_with_demark[df_with_demark['Setup_Perfected'] == True])
                            countdown_count = len(df_with_demark[df_with_demark['Countdown'] == 13])
                            
                            # Create metrics display
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Completed Setups (9)", setup_count)
                            col2.metric("Perfected Setups", perfected_setups)
                            col3.metric("Completed Countdowns (13)", countdown_count)
                            
                            # Show most recent signals
                            st.subheader("Recent Signals")
                            
                            # Get latest setup and countdown
                            latest_setup = df_with_demark[df_with_demark['Setup'] == 9].iloc[-1:] if not df_with_demark[df_with_demark['Setup'] == 9].empty else None
                            latest_countdown = df_with_demark[df_with_demark['Countdown'] == 13].iloc[-1:] if not df_with_demark[df_with_demark['Countdown'] == 13].empty else None
                            
                            if latest_setup is not None and len(latest_setup) > 0:
                                st.info(f"Latest Setup (9) completed on {latest_setup.index[0].date()} at price {latest_setup['Close'].values[0]:.2f}")
                            
                            if latest_countdown is not None and len(latest_countdown) > 0:
                                st.info(f"Latest Countdown (13) completed on {latest_countdown.index[0].date()} at price {latest_countdown['Close'].values[0]:.2f}")
                            
                            # Raw data display option
                            with st.expander("Show Raw Data"):
                                # Filter weekends from display if daily chart
                                display_df = filter_weekends(df_with_demark, interval) if interval == '1d' else df_with_demark
                                st.dataframe(display_df[['Open', 'High', 'Low', 'Close', 'Setup', 'Countdown', 'Setup_Perfected']])
                                
                            # Allow download of data
                            csv = df_with_demark.to_csv()
                            st.download_button(
                                label="Download Data as CSV",
                                data=csv,
                                file_name=f"{ticker}_demark_data.csv",
                                mime="text/csv",
                            )
                    
                    except Exception as e:
                        st.error(f"Error processing data: {str(e)}")
                        st.info("If using intraday intervals, try a smaller date range or check if the ticker symbol is correct.")
        
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")

    # Instructions section with tabs
    st.markdown("---")
    tabs = st.tabs(["How It Works", "Usage Instructions", "DeMark Theory"])

    with tabs[0]:
        st.markdown("""
        ### How DeMark TD Sequential Works
        
        TD Sequential is a technical analysis indicator developed by Tom DeMark to identify potential price reversals.
        
        The indicator consists of two phases:
        
        1. **Setup Phase (Green Numbers)**
            - A sequence of 9 consecutive closes lower than the close 4 bars earlier
            - Indicates potential buying opportunity when complete
            - A "perfected" setup (marked with ★) occurs when the close of bar 9 is less than the low of bars 6, 7, or 8
        
        2. **Countdown Phase (Red Numbers)**
            - Follows a completed Setup
            - Counts 13 instances where the close is less than or equal to the low 2 bars earlier
            - A completed countdown (13) signals a potential market reversal
        """)

    with tabs[1]:
        st.markdown("""
        ### How to Use This Tool
        
        1. Enter a valid stock ticker symbol (e.g., AAPL, MSFT, AMZN)
        2. Select your preferred chart interval (daily, hourly, etc.)
        3. Choose a date range (note: intraday data is limited to 60 days)
        4. Click "Generate Chart" to display the analysis
        
        **Chart Navigation:**
        - Use the timeframe buttons (1M, 3M, etc.) to quickly change view periods
        - Zoom by selecting areas on the chart
        - Hover over data points for detailed information
        - Download the data as CSV for further analysis
        """)

    with tabs[2]:
        st.markdown("""
        ### DeMark Theory and Trading Applications
        
        Tom DeMark's indicators are based on the concept of market exhaustion. The TD Sequential helps identify potential price turning points where trends may be exhausted.
        
        **Key Principles:**
        
        - The indicator works on any timeframe and any asset class
        - It aims to identify exhaustion in both uptrends and downtrends
        - The 9-count setup and 13-count countdown are the most significant signals
        - Perfected setups have a higher probability of resulting in meaningful reversals
        
        **Important Notes:**
        
        - This indicator should be used in conjunction with other analysis methods
        - Market context and volume analysis can improve signal quality
        - False signals can occur, particularly in strongly trending markets
        """)

# For testing this module directly
if __name__ == "__main__":
    run()
