# modules/SeasonalityAnalysis.py

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
import io
from datetime import datetime, date, timedelta
import time

# Month dictionary with full and abbreviated names
MONTHS = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12
}

def get_month_number(month_name):
    return MONTHS[month_name]

@st.cache_data(ttl=3600)  # Cache data for 1 hour
def fetch_stock_data(symbol, start_date, end_date):
    try:
        # Add error handling for invalid symbols
        if not symbol:
            st.error("Please enter a valid stock symbol")
            return None
            
        with st.spinner(f"Fetching data for {symbol.upper()}..."):
            # Enhanced data fetching
            data = yf.download(symbol, start=start_date, end=end_date, progress=False)
            
            if data.empty:
                st.error(f"No data found for {symbol.upper()}. Please check the symbol and try again.")
                return None
                
            # Add additional calculated fields
            data["Day"] = data.index.day
            data["Month"] = data.index.month
            data["Year"] = data.index.year
            data["DayOfYear"] = data.index.dayofyear
            data["DayOfWeek"] = data.index.dayofweek
            data["WeekOfYear"] = data.index.isocalendar().week
            data["Daily Return"] = data["Close"].pct_change()
            data["Log Return"] = np.log(data["Close"] / data["Close"].shift(1))
            data["Price"] = data["Close"]
            data["Volatility"] = data["Daily Return"].rolling(window=21).std()
            
            # Handle NaN values from pct_change
            data = data.dropna()
            
            return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

def calculate_seasonality(data, month_number, current_year):
    """
    Enhanced seasonality calculation with more robust statistics
    """
    historical_data = data[data.index.year < current_year].copy()
    current_data = data[data.index.year == current_year].copy()
    
    # Skip calculation if not enough historical data
    if len(historical_data) < 30:
        st.warning("Not enough historical data for robust analysis. Results may be less reliable.")
    
    historical_monthly = historical_data[historical_data["Month"] == month_number].copy()
    
    # Calculate daily average returns with improved statistics
    daily_avg_returns = historical_monthly.groupby("Day")["Daily Return"].agg([
        "mean",
        "std",
        "count",
        ("median", "median"),
        ("min", "min"),
        ("max", "max"),
        ("skew", lambda x: stats.skew(x)),
        ("kurtosis", lambda x: stats.kurtosis(x))
    ])
    
    # Calculate 95% confidence intervals
    confidence_level = 1.96
    daily_avg_returns["ci_lower"] = daily_avg_returns["mean"] - confidence_level * (
        daily_avg_returns["std"] / np.sqrt(daily_avg_returns["count"])
    )
    daily_avg_returns["ci_upper"] = daily_avg_returns["mean"] + confidence_level * (
        daily_avg_returns["std"] / np.sqrt(daily_avg_returns["count"])
    )
    
    # Cumulative monthly returns (to show pattern more clearly)
    daily_avg_returns["cumulative_mean"] = (1 + daily_avg_returns["mean"]).cumprod() - 1
    
    # Calculate current month returns if available
    current_monthly = current_data[current_data["Month"] == month_number].copy()
    current_returns = pd.Series(index=range(1, 32))  # Create a full month index
    
    if not current_monthly.empty:
        temp_current = current_monthly.groupby("Day")["Daily Return"].mean()
        # Fill the series with actual data
        for day in temp_current.index:
            current_returns[day] = temp_current[day]
            
        # Calculate cumulative return for current month
        current_cumulative = pd.Series(index=range(1, 32))
        cumul = 1.0
        for day in range(1, 32):
            if day in current_monthly["Day"].values:
                cumul *= (1 + current_returns[day])
            current_cumulative[day] = cumul - 1
    else:
        current_cumulative = pd.Series(index=range(1, 32))
    
    # Calculate monthly returns statistics
    monthly_returns = historical_monthly.groupby("Year")["Daily Return"].apply(
        lambda x: (1 + x).prod() - 1
    )
    
    # Enhanced statistical analysis
    t_stat, p_value = stats.ttest_1samp(monthly_returns, 0)
    
    # Add year-by-year breakdown of this month's performance
    yearly_performance = historical_monthly.groupby("Year").apply(
        lambda x: (1 + x["Daily Return"]).prod() - 1
    ).reset_index()
    yearly_performance.columns = ["Year", "Monthly Return"]
    yearly_performance = yearly_performance.sort_values("Year", ascending=False)
    
    # Calculate success rate (% of years with positive returns)
    success_rate = (monthly_returns > 0).mean()
    
    return {
        "historical_daily_returns": daily_avg_returns,
        "current_returns": current_returns,
        "current_cumulative": current_cumulative,
        "monthly_avg_return": monthly_returns.mean(),
        "monthly_median_return": monthly_returns.median(),
        "monthly_std": monthly_returns.std(),
        "current_month_return": current_returns.mean() if not current_monthly.empty else None,
        "t_statistic": t_stat,
        "p_value": p_value,
        "success_rate": success_rate,
        "yearly_performance": yearly_performance
    }

def plot_seasonality_with_current(seasonality_results, stock, month_name):
    """
    Enhanced seasonality chart with improved visuals and multiple views
    """
    # Create figure with subplots (daily returns and cumulative returns)
    fig = make_subplots(
        rows=2, 
        cols=1,
        subplot_titles=(
            f"{stock.upper()} Daily Returns - {month_name}",
            f"{stock.upper()} Cumulative Returns - {month_name}"
        ),
        vertical_spacing=0.12,
        shared_xaxes=True
    )
    
    # ------ Top plot: Daily returns ------
    # Historical average line
    fig.add_trace(
        go.Scatter(
            x=seasonality_results["historical_daily_returns"].index,
            y=seasonality_results["historical_daily_returns"]["mean"] * 100,  # Convert to percentage
            mode="lines",
            name="Historical Avg Return",
            line=dict(color="#1f77b4", width=2.5)
        ),
        row=1, col=1
    )
    
    # Confidence interval
    fig.add_trace(
        go.Scatter(
            x=seasonality_results["historical_daily_returns"].index,
            y=seasonality_results["historical_daily_returns"]["ci_upper"] * 100,
            mode="lines",
            line=dict(width=0),
            showlegend=False
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=seasonality_results["historical_daily_returns"].index,
            y=seasonality_results["historical_daily_returns"]["ci_lower"] * 100,
            mode="lines",
            line=dict(width=0),
            fillcolor="rgba(31, 119, 180, 0.2)",
            fill="tonexty",
            name="95% Confidence"
        ),
        row=1, col=1
    )
    
    # Current year data if available
    if not seasonality_results["current_returns"].dropna().empty:
        fig.add_trace(
            go.Scatter(
                x=seasonality_results["current_returns"].dropna().index,
                y=seasonality_results["current_returns"].dropna() * 100,
                mode="lines+markers",
                name=f"Current Year ({datetime.now().year})",
                line=dict(color="#d62728", width=2.5),
                marker=dict(size=8)
            ),
            row=1, col=1
        )
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
    
    # ------ Bottom plot: Cumulative returns ------
    # Historical cumulative
    fig.add_trace(
        go.Scatter(
            x=seasonality_results["historical_daily_returns"].index,
            y=seasonality_results["historical_daily_returns"]["cumulative_mean"] * 100,
            mode="lines",
            name="Avg Cumulative Return",
            line=dict(color="#2ca02c", width=2.5)
        ),
        row=2, col=1
    )
    
    # Current year cumulative if available
    if not seasonality_results["current_cumulative"].dropna().empty:
        fig.add_trace(
            go.Scatter(
                x=seasonality_results["current_cumulative"].dropna().index,
                y=seasonality_results["current_cumulative"].dropna() * 100,
                mode="lines+markers",
                name=f"Current Cumulative ({datetime.now().year})",
                line=dict(color="#ff7f0e", width=2.5),
                marker=dict(size=8)
            ),
            row=2, col=1
        )
    
    # Add zero line to bottom plot
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
    
    # Update layout
    fig.update_layout(
        title={
            'text': f"{stock.upper()} Seasonality Analysis - {month_name}",
            'y':0.98,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=20, color="#2c3e50")
        },
        height=700,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255, 255, 255, 0.5)",
            bordercolor="#DDDDDD",
            borderwidth=1
        ),
        margin=dict(l=50, r=50, t=100, b=50),
        plot_bgcolor="#F8F9FA",
        paper_bgcolor="#FFFFFF",
        hovermode="x unified"
    )
    
    fig.update_xaxes(
        tickmode="linear",
        tick0=1,
        dtick=1,
        title="Day of Month",
        gridcolor="#EEEEEE",
        zeroline=False,
        minor_showgrid=True,
        row=1, col=1
    )
    
    fig.update_xaxes(
        tickmode="linear",
        tick0=1,
        dtick=1,
        title="Day of Month",
        gridcolor="#EEEEEE",
        zeroline=False,
        row=2, col=1
    )
    
    fig.update_yaxes(
        title="Daily Return (%)",
        gridcolor="#EEEEEE",
        zeroline=False,
        tickformat=".2f",
        row=1, col=1
    )
    
    fig.update_yaxes(
        title="Cumulative Return (%)",
        gridcolor="#EEEEEE",
        zeroline=False,
        tickformat=".2f",
        row=2, col=1
    )
    
    return fig

def calculate_williams_true_seasonal(data, current_year, lookback_years=5):
    """
    Enhanced True Seasonal calculation with more robust statistics and improved forecasting
    """
    # Ensure we have enough data
    if data.index.year.nunique() < 2:
        st.warning("Not enough historical data for True Seasonal analysis. Results may not be reliable.")
    
    # Filter data by lookback period
    historical_data = data[data.index.year < current_year].copy()
    if lookback_years > 0:
        historical_data = historical_data[historical_data.index.year >= (current_year - lookback_years)].copy()
    current_data = data[data.index.year == current_year].copy()

    # Calculate seasonal average price for each day of the year with enhanced statistics
    seasonal_avg = historical_data.groupby("DayOfYear")["Price"].agg([
        "mean",
        "std",
        "count",
        ("median", "median"),
        ("min", "min"),
        ("max", "max")
    ]).rename(columns={"mean": "seasonal_avg_price"})
    
    # Calculate confidence intervals
    confidence_level = 1.96
    seasonal_avg["ci_lower"] = seasonal_avg["seasonal_avg_price"] - confidence_level * (
        seasonal_avg["std"] / np.sqrt(seasonal_avg["count"])
    )
    seasonal_avg["ci_upper"] = seasonal_avg["seasonal_avg_price"] + confidence_level * (
        seasonal_avg["std"] / np.sqrt(seasonal_avg["count"])
    )
    
    # Filter out days with too few data points
    seasonal_avg = seasonal_avg[seasonal_avg["count"] >= 2]

    # Calculate deviation for all data
    all_data = data.copy()
    all_data["Seasonal_Avg"] = all_data["DayOfYear"].map(seasonal_avg["seasonal_avg_price"])
    all_data["Deviation"] = (all_data["Price"] - all_data["Seasonal_Avg"]) / all_data["Seasonal_Avg"]
    
    # Calculate rolling average of deviation to smooth noise
    all_data["Smoothed_Deviation"] = all_data["Deviation"].rolling(window=5, center=True).mean()
    
    # Forward fill missing values at the boundaries
    all_data["Smoothed_Deviation"] = all_data["Smoothed_Deviation"].fillna(all_data["Deviation"])

    # Identify turning points with adaptive thresholds
    # Calculate historical volatility to set adaptive thresholds
    historical_volatility = historical_data["Daily Return"].std() * np.sqrt(252)  # Annualized
    base_threshold = 0.05  # Base threshold of 5%
    
    # Adjust threshold based on volatility (higher volatility = higher threshold)
    adaptive_threshold = base_threshold * (1 + historical_volatility)
    
    # Use smoothed deviation for signal generation
    all_data["Signal"] = 0
    all_data.loc[all_data["Smoothed_Deviation"] > adaptive_threshold, "Signal"] = 1  # Bullish
    all_data.loc[all_data["Smoothed_Deviation"] < -adaptive_threshold, "Signal"] = -1  # Bearish

    # Calculate signal changes to identify turning points
    all_data["Signal_Change"] = all_data["Signal"].diff()
    turning_points = all_data[all_data["Signal_Change"] != 0].copy()
    turning_points["Type"] = turning_points["Signal"].map({1: "Bullish", -1: "Bearish", 0: "Neutral"})
    
    # Add strength metric (how far from threshold)
    turning_points["Strength"] = abs(turning_points["Smoothed_Deviation"]) / adaptive_threshold
    
    # Predict upcoming turning points
    current_date = datetime.now()
    current_day_of_year = current_date.timetuple().tm_yday
    
    # Create DataFrame for future days
    upcoming_data = pd.DataFrame(index=range(current_day_of_year + 1, 366))
    upcoming_data["DayOfYear"] = upcoming_data.index
    
    # Map seasonal average prices to upcoming days
    upcoming_data["Seasonal_Avg"] = upcoming_data["DayOfYear"].map(
        lambda x: seasonal_avg.loc[x, "seasonal_avg_price"] if x in seasonal_avg.index else None
    )
    
    # Drop rows where we don't have historical seasonal data
    upcoming_data = upcoming_data.dropna(subset=["Seasonal_Avg"])
    
    # Calculate the expected deviation based on historical patterns
    # This uses a more sophisticated approach based on repeating seasonal patterns
    
    # First, get all historical deviations for each day of year
    day_deviation_map = {}
    for day in upcoming_data["DayOfYear"]:
        historical_deviations = all_data[all_data["DayOfYear"] == day]["Deviation"]
        if not historical_deviations.empty:
            day_deviation_map[day] = historical_deviations.median()  # Use median for robustness
    
    # Apply the expected deviations
    upcoming_data["Expected_Deviation"] = upcoming_data["DayOfYear"].map(
        lambda x: day_deviation_map.get(x, 0)
    )
    
    # Apply smoothing to expected deviations
    upcoming_data["Expected_Deviation"] = upcoming_data["Expected_Deviation"].rolling(window=5, center=True).mean()
    upcoming_data["Expected_Deviation"] = upcoming_data["Expected_Deviation"].fillna(method="bfill").fillna(method="ffill")

    # Generate signals using the same adaptive threshold
    upcoming_data["Signal"] = 0
    upcoming_data.loc[upcoming_data["Expected_Deviation"] > adaptive_threshold, "Signal"] = 1
    upcoming_data.loc[upcoming_data["Expected_Deviation"] < -adaptive_threshold, "Signal"] = -1
    
    # Calculate signal changes to identify upcoming turning points
    upcoming_data["Signal_Change"] = upcoming_data["Signal"].diff()
    upcoming_turning_points = upcoming_data[upcoming_data["Signal_Change"] != 0].copy()
    upcoming_turning_points["Type"] = upcoming_turning_points["Signal"].map({1: "Bullish", -1: "Bearish", 0: "Neutral"})
    
    # Add strength metric (how far from threshold)
    upcoming_turning_points["Strength"] = abs(upcoming_turning_points["Expected_Deviation"]) / adaptive_threshold
    
    # Add confidence metric based on historical consistency
    upcoming_turning_points["Confidence"] = upcoming_turning_points["DayOfYear"].map(
        lambda x: min(seasonal_avg.loc[x, "count"] / lookback_years, 1.0) if x in seasonal_avg.index else 0.5
    )

    # Add approximate dates for 2025
    def day_of_year_to_date(day_of_year):
        base_date = date(2025, 1, 1)
        return base_date + timedelta(days=day_of_year - 1)

    upcoming_turning_points["Date"] = upcoming_turning_points["DayOfYear"].apply(day_of_year_to_date)
    
    # Sort by date
    upcoming_turning_points = upcoming_turning_points.sort_values("DayOfYear")

    return {
        "seasonal_avg": seasonal_avg,
        "all_deviation": all_data["Deviation"],
        "smoothed_deviation": all_data["Smoothed_Deviation"],
        "turning_points": turning_points,
        "upcoming_turning_points": upcoming_turning_points,
        "current_deviation": all_data[all_data.index.year == current_year]["Deviation"],
        "adaptive_threshold": adaptive_threshold,
        "historical_volatility": historical_volatility
    }

def plot_williams_true_seasonal(wts_results, stock):
    """
    Enhanced True Seasonal chart with improved visuals
    """
    # Create figure with subplots
    fig = make_subplots(
        rows=2, 
        cols=1,
        subplot_titles=(
            f"{stock.upper()} - True Seasonal Deviation from Average",
            f"{stock.upper()} - Seasonal Price Pattern"
        ),
        vertical_spacing=0.12,
        shared_xaxes=True,
        specs=[[{"secondary_y": False}], [{"secondary_y": True}]]
    )
    
    # Get all_data with deviation for the top plot
    all_data = pd.DataFrame({
        'date': wts_results["smoothed_deviation"].index,
        'deviation': wts_results["smoothed_deviation"].values
    })
    
    # Add threshold lines
    threshold = wts_results["adaptive_threshold"]
    
    # Add main deviation line (smoothed)
    fig.add_trace(
        go.Scatter(
            x=all_data['date'],
            y=all_data['deviation'],
            mode="lines",
            name="Seasonal Deviation",
            line=dict(color="#1f77b4", width=2)
        ),
        row=1, col=1
    )
    
    # Add threshold lines
    fig.add_hline(y=threshold, line_dash="dash", line_color="green", row=1, col=1,
                 annotation_text="Bullish Threshold", annotation_position="top right")
    fig.add_hline(y=-threshold, line_dash="dash", line_color="red", row=1, col=1,
                 annotation_text="Bearish Threshold", annotation_position="bottom right")
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
    
    # Add turning points
    bullish_points = wts_results["turning_points"][wts_results["turning_points"]["Type"] == "Bullish"]
    bearish_points = wts_results["turning_points"][wts_results["turning_points"]["Type"] == "Bearish"]

    # Bullish turning points
    fig.add_trace(
        go.Scatter(
            x=bullish_points.index,
            y=bullish_points["Smoothed_Deviation"],
            mode="markers",
            name="Bullish Turning Points",
            marker=dict(color="green", size=10, symbol="triangle-up", line=dict(width=1, color="darkgreen"))
        ),
        row=1, col=1
    )
    
    # Bearish turning points
    fig.add_trace(
        go.Scatter(
            x=bearish_points.index,
            y=bearish_points["Smoothed_Deviation"],
            mode="markers",
            name="Bearish Turning Points",
            marker=dict(color="red", size=10, symbol="triangle-down", line=dict(width=1, color="darkred"))
        ),
        row=1, col=1
    )
    
    # Bottom plot: Seasonal price pattern
    # Get all unique days of year
    days_of_year = wts_results["seasonal_avg"].index.sort_values()
    seasonal_avg = wts_results["seasonal_avg"]
    
    # Create a date array for x-axis that covers a full year
    base_year = 2025
    dates = [date(base_year, 1, 1) + timedelta(days=d-1) for d in days_of_year if d <= 366]
    
    # Seasonal average price 
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=seasonal_avg.loc[days_of_year[days_of_year <= 366], "seasonal_avg_price"],
            mode="lines",
            name="Seasonal Price Pattern",
            line=dict(color="#2ca02c", width=2.5)
        ),
        row=2, col=1
    )
    
    # Add confidence intervals
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=seasonal_avg.loc[days_of_year[days_of_year <= 366], "ci_upper"],
            mode="lines",
            line=dict(width=0),
            showlegend=False
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=seasonal_avg.loc[days_of_year[days_of_year <= 366], "ci_lower"],
            mode="lines",
            line=dict(width=0),
            fillcolor="rgba(44, 160, 44, 0.2)",
            fill="tonexty",
            name="95% Confidence Interval"
        ),
        row=2, col=1
    )
    
    # Add markers for upcoming turning points on the bottom plot (on secondary y-axis)
    # Prepare upcoming turning points
    upcoming = wts_results["upcoming_turning_points"]
    
    if not upcoming.empty:
        # Bullish upcoming points
        bullish_upcoming = upcoming[upcoming["Type"] == "Bullish"]
        if not bullish_upcoming.empty:
            fig.add_trace(
                go.Scatter(
                    x=bullish_upcoming["Date"],
                    y=bullish_upcoming["Strength"],
                    mode="markers",
                    name="Upcoming Bullish Points",
                    marker=dict(
                        color="green", 
                        size=bullish_upcoming["Strength"] * 10,
                        symbol="triangle-up",
                        line=dict(width=1, color="darkgreen")
                    ),
                    text=[f"Confidence: {c:.1%}" for c in bullish_upcoming["Confidence"]],
                    hovertemplate="%{text}<br>Date: %{x}<br>Strength: %{y:.2f}",
                ),
                row=2, col=1,
                secondary_y=True
            )
        
        # Bearish upcoming points
        bearish_upcoming = upcoming[upcoming["Type"] == "Bearish"]
        if not bearish_upcoming.empty:
            fig.add_trace(
                go.Scatter(
                    x=bearish_upcoming["Date"],
                    y=bearish_upcoming["Strength"],
                    mode="markers",
                    name="Upcoming Bearish Points",
                    marker=dict(
                        color="red", 
                        size=bearish_upcoming["Strength"] * 10,
                        symbol="triangle-down",
                        line=dict(width=1, color="darkred")
                    ),
                    text=[f"Confidence: {c:.1%}" for c in bearish_upcoming["Confidence"]],
                    hovertemplate="%{text}<br>Date: %{x}<br>Strength: %{y:.2f}",
                ),
                row=2, col=1,
                secondary_y=True
            )
    
    # Update layout
    fig.update_layout(
        title={
            'text': f"{stock.upper()} True Seasonal Analysis",
            'y':0.98,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=20, color="#2c3e50")
        },
        height=800,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255, 255, 255, 0.5)",
            bordercolor="#DDDDDD",
            borderwidth=1
        ),
        margin=dict(l=50, r=50, t=100, b=50),
        plot_bgcolor="#F8F9FA",
        paper_bgcolor="#FFFFFF",
        hovermode="x unified"
    )
    
    # Update axes
    fig.update_xaxes(
        title="Date",
        gridcolor="#EEEEEE",
        zeroline=False,
        row=1, col=1
    )
    
    fig.update_xaxes(
        title="Date",
        gridcolor="#EEEEEE",
        zeroline=False,
        tickformat="%b %d",  # Month and day format for bottom chart
        dtick="M1",  # Monthly ticks
        row=2, col=1
    )
    
    fig.update_yaxes(
        title="Deviation from Seasonal Average",
        gridcolor="#EEEEEE",
        zeroline=False,
        tickformat=".1%",
        row=1, col=1
    )
    
    fig.update_yaxes(
        title="Price ($)",
        gridcolor="#EEEEEE",
        zeroline=False,
        row=2, col=1,
        secondary_y=False
    )
    
    fig.update_yaxes(
        title="Signal Strength",
        gridcolor="#EEEEEE",
        zeroline=False,
        range=[0, 3],
        row=2, col=1,
        secondary_y=True
    )
    
    return fig

def create_yearly_heatmap(data, stock_symbol):
    """
    Create a yearly seasonality heatmap to visualize monthly performance across years
    """
    # Prepare data for heatmap
    monthly_returns = data.groupby(["Year", "Month"])["Daily Return"].apply(
        lambda x: (1 + x).prod() - 1
    ).reset_index()
    
    # Pivot data for heatmap
    heatmap_data = monthly_returns.pivot(index="Year", columns="Month", values="Daily Return")
    
    # Replace month numbers with names
    month_mapper = {v: k[:3] for k, v in MONTHS.items()}  # Use abbreviated month names
    heatmap_data = heatmap_data.rename(columns=month_mapper)
    
    # Create heatmap figure
    fig = go.Figure()
    
    # Add heatmap trace
    fig.add_trace(go.Heatmap(
        z=heatmap_data.values * 100,  # Convert to percentage
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale=[
            [0, 'rgb(165,0,38)'],    # Dark red for worst returns
            [0.25, 'rgb(215,48,39)'], # Red
            [0.45, 'rgb(244,109,67)'], # Light red
            [0.5, 'rgb(255,255,255)'], # White for zero
            [0.55, 'rgb(116,173,209)'], # Light blue
            [0.75, 'rgb(69,117,180)'], # Blue
            [1, 'rgb(49,54,149)']     # Dark blue for best returns
        ],
        colorbar=dict(
            title="Return (%)",
            #titleside="right",
            tickformat=".1f"
        ),
        hovertemplate="Year: %{y}<br>Month: %{x}<br>Return: %{z:.2f}%<extra></extra>"
    ))
    
    # Add annotations showing the actual values
    for i, year in enumerate(heatmap_data.index):
        for j, month in enumerate(heatmap_data.columns):
            value = heatmap_data.iloc[i, j] * 100  # Convert to percentage
            color = "black" if abs(value) < 10 else "white"  # Choose text color based on cell color intensity
            fig.add_annotation(
                x=month,
                y=year,
                text=f"{value:.1f}%" if not np.isnan(value) else "",
                showarrow=False,
                font=dict(color=color, size=9)
            )
    
    # Update layout
    fig.update_layout(
        title={
            'text': f"{stock_symbol.upper()} Monthly Returns by Year (%)",
            'y':0.98,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=18, color="#2c3e50")
        },
        xaxis=dict(
            title="Month",
            side="top"
        ),
        yaxis=dict(
            title="Year",
            autorange="reversed"  # Most recent years at top
        ),
        margin=dict(l=50, r=50, t=80, b=30),
        height=400 + (len(heatmap_data.index) * 20),  # Adjust height based on number of years
        width=900,
        plot_bgcolor="#F8F9FA",
        paper_bgcolor="#FFFFFF"
    )
    
    return fig

def create_dow_seasonality(data, stock_symbol):
    """
    Create day-of-week seasonality analysis
    """
    # Map day numbers to names
    day_names = {
        0: "Monday",
        1: "Tuesday",
        2: "Wednesday",
        3: "Thursday",
        4: "Friday",
        5: "Saturday",
        6: "Sunday"
    }
    
    # Calculate average returns by day of week
    dow_returns = data.groupby("DayOfWeek")["Daily Return"].agg([
        "mean",
        "std",
        "count",
        ("median", "median"),
        ("positive", lambda x: (x > 0).mean())  # Percentage of positive days
    ])
    
    # Add day names
    dow_returns = dow_returns.reset_index()
    dow_returns["Day"] = dow_returns["DayOfWeek"].map(day_names)
    dow_returns = dow_returns.sort_values("DayOfWeek")
    
    # Create figure with two subplots: bar chart and success rate
    fig = make_subplots(
        rows=1, 
        cols=2,
        subplot_titles=(
            f"{stock_symbol.upper()} Average Return by Day of Week",
            f"{stock_symbol.upper()} Win Rate by Day of Week"
        ),
        specs=[[{"type": "bar"}, {"type": "bar"}]],
        column_widths=[0.6, 0.4]
    )
    
    # Add bar chart for average returns
    fig.add_trace(
        go.Bar(
            x=dow_returns["Day"],
            y=dow_returns["mean"] * 100,  # Convert to percentage
            error_y=dict(
                type="data",
                array=dow_returns["std"] * 100 / np.sqrt(dow_returns["count"]),
                visible=True,
                color="black",
                thickness=1,
                width=3
            ),
            marker=dict(
                color=dow_returns["mean"] * 100,
                colorscale=[
                    [0, "red"],
                    [0.5, "white"],
                    [1, "green"]
                ],
                line=dict(color="black", width=1),
                cmin=-0.5,  # Center colorscale
                cmax=0.5
            ),
            text=dow_returns["mean"] * 100,
            texttemplate="%{text:.2f}%",
            textposition="outside",
            name="Avg Return"
        ),
        row=1, col=1
    )
    
    # Add bar chart for win rate
    fig.add_trace(
        go.Bar(
            x=dow_returns["Day"],
            y=dow_returns["positive"] * 100,  # Convert to percentage
            marker=dict(
                color=dow_returns["positive"] * 100,
                colorscale=[
                    [0, "#FFCCCC"],  # Light red
                    [0.5, "#FFFF99"], # Yellow
                    [1, "#CCFFCC"]   # Light green
                ],
                line=dict(color="black", width=1),
                cmin=40,  # Scale from 40-60% for better color differentiation
                cmax=60
            ),
            text=dow_returns["positive"] * 100,
            texttemplate="%{text:.1f}%",
            textposition="outside",
            name="Win Rate"
        ),
        row=1, col=2
    )
    
    # Add a reference line at 50% for win rate
    fig.add_hline(
        y=50, 
        line_dash="dash", 
        line_color="gray",
        annotation_text="50% Win Rate",
        annotation_position="bottom right",
        row=1, col=2
    )
    
    # Add a reference line at 0% for returns
    fig.add_hline(
        y=0, 
        line_dash="dash", 
        line_color="gray",
        row=1, col=1
    )
    
    # Update layout
    fig.update_layout(
        title={
            'text': f"{stock_symbol.upper()} Day-of-Week Seasonality",
            'y':0.98,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=18, color="#2c3e50")
        },
        showlegend=False,
        height=500,
        width=900,
        plot_bgcolor="#F8F9FA",
        paper_bgcolor="#FFFFFF"
    )
    
    fig.update_yaxes(
        title="Average Return (%)",
        range=[
            min(dow_returns["mean"] * 100) * 1.5 if min(dow_returns["mean"] * 100) < 0 else -0.1,
            max(dow_returns["mean"] * 100) * 1.5
        ],
        row=1, col=1
    )
    
    fig.update_yaxes(
        title="Win Rate (%)",
        range=[
            min(40, min(dow_returns["positive"] * 100) * 0.95),
            max(60, max(dow_returns["positive"] * 100) * 1.05)
        ],
        row=1, col=2
    )
    
    return fig

def send_to_discord(webhook_url, message, fig):
    try:
        # Better error handling
        if not webhook_url or not webhook_url.startswith("https://discord.com/api/webhooks/"):
            st.error("Invalid Discord webhook URL. Please check your URL and try again.")
            return False
            
        with st.spinner("Sending to Discord..."):
            # Send text message
            payload = {"content": message}
            response = requests.post(webhook_url, json=payload)
            
            if response.status_code != 204:
                st.error(f"Error sending message to Discord: {response.status_code}")
                return False
            
            # Send chart as PNG
            buf = io.BytesIO()
            fig.write_image(buf, format="png", width=1200, height=800, scale=2)  # Higher resolution
            buf.seek(0)
            
            files = {"file": ("chart.png", buf, "image/png")}
            response = requests.post(webhook_url, files=files)
            
            if response.status_code != 204:
                st.error(f"Error sending chart to Discord: {response.status_code}")
                return False
                
            return True
            
    except Exception as e:
        st.error(f"Error sending to Discord: {e}")
        return False

def run():
    # Configure Streamlit page
    
    # Custom CSS for better UI
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem !important;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem !important;
        color: #34495e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem !important;
        font-weight: bold;
    }
    .metric-label {
        font-size: 1rem !important;
        color: #7f8c8d;
    }
    .info-box {
        background-color: #e3f2fd;
        border-left: 5px solid #2196f3;
        padding: 1rem;
        border-radius: 0.3rem;
        margin-bottom: 1rem;
    }
    .positive {color: #27ae60 !important; font-weight: bold;}
    .negative {color: #e74c3c !important; font-weight: bold;}
    .neutral {color: #7f8c8d !important;}
    .highlight {background-color: #ffeaa7; padding: 0.2rem 0.4rem; border-radius: 0.2rem;}
    </style>
    """, unsafe_allow_html=True)
    
    # Main header with logo
    st.markdown('<h1 class="main-header">📈 Stock Seasonality & True Seasonal Analysis Tool</h1>', unsafe_allow_html=True)
    
    # Create tabs for different analysis modes
    tabs = st.tabs([
        "📅 Seasonality Analysis",
        "🔄 True Seasonal Analysis",
        "ℹ️ About & Help"
    ])
    
    # Sidebar for common inputs across all tabs
    with st.sidebar:
        st.header("Analysis Settings")
        
        stock = st.text_input("Enter the stock symbol (e.g., TSLA):", value="AAPL").strip().upper()
        
        # Date range with default values
        col1, col2 = st.columns(2)
        with col1:
            start_year = st.number_input("Start Year:", min_value=2000, max_value=2024, value=2018)
            start_date = date(start_year, 1, 1)
        with col2:
            end_date = st.date_input("End Date:", date.today())
        
        # Advanced settings collapsible section
        with st.expander("Advanced Settings"):
            data_frequency = st.selectbox("Data Frequency:", ["Daily", "Weekly", "Monthly"], index=0)
            include_dividends = st.checkbox("Include Dividends", value=True)
            
            # Additional advanced settings based on analysis mode
            confidence_level = st.slider("Confidence Level (%):", min_value=80, max_value=99, value=95, step=1)
        
        # Discord integration
        st.header("Discord Integration")
        webhook_url = st.text_input("Discord Webhook URL (optional):", type="password")
        
        # Add a download section
        if 'data' in st.session_state:
            st.header("Download Data")
            if st.download_button(
                "Download Full Dataset (CSV)",
                st.session_state.data.to_csv().encode("utf-8"),
                f"{stock}_data.csv",
                "text/csv"
            ):
                st.success("Download started!")
    
    # Initialize session state for preserving data between interactions
    if 'data' not in st.session_state:
        st.session_state.data = None
    
    # Fetch data if stock symbol is provided
    if stock:
        # Attempt to fetch data only once and store in session state
        if st.session_state.data is None or st.session_state.last_stock != stock:
            data = fetch_stock_data(stock, start_date, end_date)
            if data is not None:
                st.session_state.data = data
                st.session_state.last_stock = stock
        
        data = st.session_state.data
    else:
        data = None
    
    # ----- Tab 1: Monthly Seasonality Analysis -----
    with tabs[0]:
        st.markdown('<h2 class="sub-header">Monthly Seasonality Analysis</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        This analysis examines how a stock typically performs in specific months based on historical data.
        It shows daily patterns within a month and calculates statistical significance to determine if the patterns are reliable.
        </div>
        """, unsafe_allow_html=True)
        
        if data is not None and not data.empty:
            current_year = datetime.now().year
            
            # Grid layout for month selection and analysis metrics
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                month_name = st.selectbox("Select a month for analysis:", list(MONTHS.keys()))
                month_number = get_month_number(month_name)
            
            with col2:
                years = sorted(data["Year"].unique(), reverse=True)
                comparison_year = st.selectbox("Compare with year:", [current_year] + list(years[:-1]), index=0)
            
            with col3:
                chart_type = st.selectbox("Chart Display:", ["Daily + Cumulative", "Daily Returns Only", "Cumulative Returns Only"], index=0)
            
            # Calculate seasonality
            seasonality_results = calculate_seasonality(data, month_number, comparison_year)
            
            # Display metrics in a nice grid
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <p class="metric-label">Historical Avg Return</p>
                    <p class="metric-value {'positive' if seasonality_results['monthly_avg_return'] > 0 else 'negative'}">
                        {seasonality_results['monthly_avg_return']:.2%}
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
            with col2:
                success_rate = seasonality_results['success_rate']
                st.markdown(f"""
                <div class="metric-card">
                    <p class="metric-label">Success Rate</p>
                    <p class="metric-value {'positive' if success_rate > 0.5 else 'negative'}">
                        {success_rate:.1%}
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
            with col3:
                p_value = seasonality_results['p_value']
                significant = p_value < 0.05
                st.markdown(f"""
                <div class="metric-card">
                    <p class="metric-label">Statistical Significance</p>
                    <p class="metric-value {'positive' if significant else 'neutral'}">
                        p-value: {p_value:.3f}
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
            with col4:
                if seasonality_results['current_month_return'] is not None:
                    current_return = seasonality_results['current_month_return']
                    st.markdown(f"""
                    <div class="metric-card">
                        <p class="metric-label">Current/Selected Year</p>
                        <p class="metric-value {'positive' if current_return > 0 else 'negative'}">
                            {current_return:.2%}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="metric-card">
                        <p class="metric-label">Current/Selected Year</p>
                        <p class="metric-value neutral">No data yet</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Interpretation of results
            with st.expander("📊 Interpretation of Results", expanded=True):
                st.markdown(f"""
                ### Key Insights for {month_name}:
                
                - **Historical Performance**: {stock} has an average return of {seasonality_results['monthly_avg_return']:.2%} in {month_name}.
                - **Reliability**: The monthly pattern is {"statistically significant" if significant else "not statistically significant"} (p-value: {p_value:.3f}).
                - **Success Rate**: {month_name} has been positive {success_rate:.1%} of the time historically.
                """)
                
                # Add specific insights based on the data
                if seasonality_results['monthly_avg_return'] > 0.02:
                    st.markdown(f"- **Strong Month**: {month_name} has historically been a **strong month** for {stock} with above-average returns.")
                elif seasonality_results['monthly_avg_return'] < -0.02:
                    st.markdown(f"- **Weak Month**: {month_name} has historically been a **weak month** for {stock} with below-average returns.")
                
                if success_rate > 0.7:
                    st.markdown(f"- **Highly Consistent**: {month_name} has been consistently positive ({success_rate:.1%} of years).")
                elif success_rate < 0.3:
                    st.markdown(f"- **Consistently Negative**: {month_name} has been consistently negative ({(1-success_rate):.1%} of years).")
            
            # Display seasonality chart
            fig = plot_seasonality_with_current(seasonality_results, stock, month_name)
            st.plotly_chart(fig, use_container_width=True)
            
            # Display historical performance by year
            st.subheader(f"Historical Performance: {month_name}")
            yearly_data = seasonality_results["yearly_performance"]
            
            # Create a visual representation of yearly performance
            yearly_fig = px.bar(
                yearly_data,
                x="Year",
                y="Monthly Return",
                color="Monthly Return",
                color_continuous_scale=px.colors.diverging.RdBu,
                color_continuous_midpoint=0,
                height=400
            )
            
            yearly_fig.update_layout(
                title=f"{stock} Performance in {month_name} by Year",
                yaxis_tickformat=".1%",
                plot_bgcolor="#F8F9FA",
                paper_bgcolor="#FFFFFF"
            )
            
            yearly_fig.add_hline(y=0, line_dash="dash", line_color="gray")
            
            st.plotly_chart(yearly_fig, use_container_width=True)
            
            # Additional analyses
            with st.expander("Additional Seasonality Analyses"):
                analysis_tab1, analysis_tab2 = st.tabs(["Monthly Heatmap", "Day-of-Week Analysis"])
                
                with analysis_tab1:
                    # Show monthly returns heatmap
                    heatmap_fig = create_yearly_heatmap(data, stock)
                    st.plotly_chart(heatmap_fig, use_container_width=True)
                
                with analysis_tab2:
                    # Show day of week analysis
                    dow_fig = create_dow_seasonality(data, stock)
                    st.plotly_chart(dow_fig, use_container_width=True)
            
            # Discord integration
            if webhook_url:
                with st.expander("Send to Discord"):
                    discord_message = f"""
**Seasonality Analysis for {stock}**
Month: {month_name}
Historical Avg Return: {seasonality_results['monthly_avg_return']:.2%}
Success Rate: {seasonality_results['success_rate']:.1%}
Statistical Significance: p-value = {seasonality_results['p_value']:.3f}
Analysis performed on {datetime.now().strftime('%Y-%m-%d %H:%M')}
                    """
                    
                    if st.button("Send Seasonality Analysis to Discord"):
                        if send_to_discord(webhook_url, discord_message, fig):
                            st.success("Successfully sent to Discord!")
        
        else:
            # Show a placeholder if no data is available yet
            if stock:
                st.info("Please enter a valid stock symbol above to see the analysis.")
            else:
                st.warning("Please enter a stock symbol to begin analysis.")
                
                # Add a demo button
                if st.button("Load Demo Data (AAPL)"):
                    st.session_state.last_stock = "AAPL"
                    st.experimental_rerun()
            
    # ----- Tab 2: True Seasonal Analysis -----
    with tabs[1]:
        st.markdown('<h2 class="sub-header">True Seasonal Analysis</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        True Seasonal Analysis identifies turning points where a stock's price significantly deviates from its typical seasonal pattern.
        This can help identify potential bullish and bearish opportunities based on seasonal tendencies.
        </div>
        """, unsafe_allow_html=True)
        
        if data is not None and not data.empty:
            current_year = datetime.now().year
            
            # Analysis parameters
            col1, col2 = st.columns(2)
            
            with col1:
                lookback_years = st.slider(
                    "Lookback Period (Years):", 
                    min_value=2, 
                    max_value=min(15, len(data["Year"].unique())), 
                    value=min(5, len(data["Year"].unique()) - 1)
                )
                
            with col2:
                smoothing_window = st.slider(
                    "Smoothing Window (Days):", 
                    min_value=1, 
                    max_value=21, 
                    value=5
                )
            
            # Calculate Williams True Seasonal results
            with st.spinner("Calculating True Seasonal patterns..."):
                wts_results = calculate_williams_true_seasonal(data, current_year, lookback_years)
            
            # Display key metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <p class="metric-label">Historical Volatility</p>
                    <p class="metric-value">
                        {wts_results['historical_volatility']:.1%}
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
            with col2:
                turning_point_count = len(wts_results["turning_points"])
                st.markdown(f"""
                <div class="metric-card">
                    <p class="metric-label">Detected Turning Points</p>
                    <p class="metric-value">
                        {turning_point_count}
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
            with col3:
                upcoming_count = len(wts_results["upcoming_turning_points"])
                st.markdown(f"""
                <div class="metric-card">
                    <p class="metric-label">Upcoming Turning Points</p>
                    <p class="metric-value highlight">
                        {upcoming_count}
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            # Display True Seasonal chart
            fig = plot_williams_true_seasonal(wts_results, stock)
            st.plotly_chart(fig, use_container_width=True)
            
            # Display upcoming turning points with enhanced visualization
            st.subheader("Upcoming Turning Points (2025)")
            
            upcoming_points = wts_results["upcoming_turning_points"].copy()
            
            if not upcoming_points.empty:
                # Sort by date and format the table
                upcoming_points = upcoming_points.sort_values("Date")
                
                # Format for display
                display_data = upcoming_points[["Date", "Type", "Expected_Deviation", "Strength", "Confidence"]]
                display_data = display_data.rename(columns={
                    "Expected_Deviation": "Expected Deviation",
                    "Strength": "Signal Strength",
                    "Confidence": "Confidence"
                })
                
                # Format values
                display_data["Expected Deviation"] = display_data["Expected Deviation"].map("{:.2%}".format)
                display_data["Signal Strength"] = display_data["Signal Strength"].map("{:.2f}".format)
                display_data["Confidence"] = display_data["Confidence"].map("{:.1%}".format)
                
                # Color-code the table
                def highlight_type(val):
                    if val == "Bullish":
                        return "background-color: #d4edda; color: #155724; font-weight: bold"
                    elif val == "Bearish":
                        return "background-color: #f8d7da; color: #721c24; font-weight: bold"
                    return ""
                
                # Display styled table
                st.dataframe(
                    display_data.style.applymap(highlight_type, subset=["Type"]),
                    use_container_width=True,
                    height=min(400, (len(display_data) + 1) * 35 + 3)
                )
                
                # Create a calendar view of upcoming turning points
                st.subheader("Calendar View")
                
                # Group data by month for calendar view
                upcoming_by_month = upcoming_points.copy()
                # Make sure Date is actually a datetime type
                if not pd.api.types.is_datetime64_any_dtype(upcoming_by_month["Date"]):
                # Convert to datetime if it's not already
                    upcoming_by_month["Date"] = pd.to_datetime(upcoming_by_month["Date"])
                upcoming_by_month["Month"] = upcoming_by_month["Date"].dt.month
                upcoming_by_month["MonthName"] = upcoming_by_month["Date"].dt.strftime("%B")
                
                months_with_events = upcoming_by_month["MonthName"].unique()
                
                # Create tabs for each month with events
                if len(months_with_events) > 0:
                    month_tabs = st.tabs(list(months_with_events))
                    
                    for i, month in enumerate(months_with_events):
                        with month_tabs[i]:
                            month_data = upcoming_by_month[upcoming_by_month["MonthName"] == month]
                            # Create a clean display of events
                            for _, event in month_data.iterrows():
                                event_type_color = "green" if event["Type"] == "Bullish" else "red"
                                event_date = event["Date"].strftime("%B %d, %Y")
                                
                                st.markdown(f"""
                                <div style="margin-bottom: 12px; padding: 10px; border-left: 4px solid {event_type_color}; background-color: #f8f9fa;">
                                    <span style="font-weight: bold; color: {event_type_color};">{event["Type"]} Signal</span> on <span style="font-weight: bold;">{event_date}</span><br>
                                    <span style="font-size: 0.9rem;">Expected Deviation: {event["Expected_Deviation"]:.2%} | Strength: {event["Strength"]:.2f} | Confidence: {event["Confidence"]:.1%}</span>
                                </div>
                                """, unsafe_allow_html=True)
                else:
                    st.info("No upcoming turning points detected in the forecast period.")
                 
            
            # Historical turning points
            with st.expander("Historical Turning Points"):
                hist_points = wts_results["turning_points"].copy()
                
                if not hist_points.empty:
                    # Add date as column
                    hist_points["Date"] = hist_points.index
                    
                    # Format for display
                    hist_display = hist_points[["Date", "Type", "Deviation", "Strength"]]
                    hist_display = hist_display.rename(columns={
                        "Deviation": "Deviation from Average",
                        "Strength": "Signal Strength"
                    })
                    
                    # Reset index to avoid ambiguity
                    hist_display = hist_display.drop(columns=["Date"])
                    hist_display = hist_display.reset_index()
                    
                    # Sort by date (most recent first)
                    hist_display = hist_display.sort_values("Date", ascending=False)
                    
                    # Format values
                    hist_display["Deviation from Average"] = hist_display["Deviation from Average"].map("{:.2%}".format)
                    hist_display["Signal Strength"] = hist_display["Signal Strength"].map("{:.2f}".format)
                    
                    # Display table
                    st.dataframe(
                        hist_display,
                        use_container_width=True,
                        height=min(500, (len(hist_display) + 1) * 35 + 3)
                    )
                else:
                    st.info("No historical turning points detected with the current settings.")
            
            # Discord integration
            if webhook_url:
                with st.expander("Send to Discord"):
                    discord_message = f"""
**True Seasonal Analysis for {stock}**
Lookback Years: {lookback_years}
Historical Volatility: {wts_results['historical_volatility']:.1%}
Detected Turning Points: {turning_point_count}
Upcoming Turning Points: {upcoming_count}
Analysis performed on {datetime.now().strftime('%Y-%m-%d %H:%M')}
                    """
                    
                    if st.button("Send True Seasonal Analysis to Discord"):
                        if send_to_discord(webhook_url, discord_message, fig):
                            st.success("Successfully sent to Discord!")
        
            else:
            # Show a placeholder if no data is available yet
                if stock:
                    st.info("Please enter a valid stock symbol above to see the analysis.")
                else:
                    st.warning("Please enter a stock symbol to begin analysis.")
    
    # ----- Tab 3: About & Help -----
    with tabs[2]:
        st.markdown('<h2 class="sub-header">About This Tool</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        ### Stock Seasonality & True Seasonal Analysis Tool
        
        This tool provides advanced analysis of stock price seasonality patterns through two main approaches:
        
        1. **Monthly Seasonality Analysis**: Examines historical performance patterns within specific months, including:
           - Daily return patterns
           - Cumulative return patterns
           - Statistical significance testing
           - Year-by-year performance breakdown
        
        2. **True Seasonal Analysis**: Identifies turning points where a stock significantly deviates from its seasonal pattern:
           - Detects historical bullish and bearish turning points
           - Forecasts upcoming turning points
           - Calculates signal strength and confidence levels
        
        ### How to Use This Tool
        
        1. Enter a stock symbol in the sidebar
        2. Select your analysis date range
        3. Choose between Monthly Seasonality or True Seasonal Analysis
        4. Adjust parameters as needed
        5. Optionally, connect to Discord to share results
        
        ### Understanding the Results
        
        #### Monthly Seasonality
        - **Historical Average Return**: The average return for the selected month across all years
        - **Success Rate**: The percentage of years the selected month had positive returns
        - **Statistical Significance**: p-value below 0.05 suggests the pattern is statistically significant
        - **Current Year**: How the stock is performing in the current year compared to historical patterns
        
        #### True Seasonal Analysis
        - **Deviation from Average**: How much the stock price deviates from its seasonal average
        - **Turning Points**: Where the stock crosses threshold levels indicating potential trend changes
        - **Signal Strength**: The magnitude of the deviation relative to the threshold
        - **Confidence**: The reliability of the forecast based on historical consistency
        
        ### Data Sources
        
        Stock data is retrieved from Yahoo Finance via the yfinance library.
        
        ### Disclaimer
        
        This tool is for informational purposes only and should not be considered financial advice. Past performance is not indicative of future results.
        """)
