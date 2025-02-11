import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go
import requests
import io
from datetime import datetime, date

# Month dictionary
MONTHS = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12
}

# Function to convert month name to number
def get_month_number(month_name):
    return MONTHS[month_name]

# Fetch historical stock data from Yahoo Finance
def fetch_stock_data(symbol, start_date, end_date):
    try:
        data = yf.download(symbol, start=start_date, end=end_date)
        data["Day"] = data.index.day
        data["Month"] = data.index.month
        data["Year"] = data.index.year
        data["Daily Return"] = data["Close"].pct_change()
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

def calculate_seasonality(data, month_number, current_year):
    # Separate current year and historical data
    historical_data = data[data.index.year < current_year].copy()
    current_data = data[data.index.year == current_year].copy()
    
    # Calculate historical seasonality
    historical_monthly = historical_data[historical_data["Month"] == month_number].copy()
    daily_avg_returns = historical_monthly.groupby("Day")["Daily Return"].agg([
        "mean",
        "std",
        "count"
    ])
    
    # Calculate confidence intervals (95%)
    confidence_level = 1.96
    daily_avg_returns["ci_lower"] = daily_avg_returns["mean"] - confidence_level * (
        daily_avg_returns["std"] / np.sqrt(daily_avg_returns["count"])
    )
    daily_avg_returns["ci_upper"] = daily_avg_returns["mean"] + confidence_level * (
        daily_avg_returns["std"] / np.sqrt(daily_avg_returns["count"])
    )
    
    # Get current year's data for the month
    current_monthly = current_data[current_data["Month"] == month_number].copy()
    current_returns = current_monthly.groupby("Day")["Daily Return"].mean()
    
    # Calculate statistical significance
    monthly_returns = historical_monthly.groupby("Year")["Daily Return"].mean()
    t_stat, p_value = stats.ttest_1samp(monthly_returns, 0)
    
    return {
        "historical_daily_returns": daily_avg_returns,
        "current_returns": current_returns,
        "monthly_avg_return": monthly_returns.mean(),
        "current_month_return": current_returns.mean() if not current_returns.empty else None,
        "t_statistic": t_stat,
        "p_value": p_value
    }

def plot_seasonality_with_current(seasonality_results, stock, month_name):
    fig = go.Figure()
    
    # Plot historical average
    fig.add_trace(go.Scatter(
        x=seasonality_results["historical_daily_returns"].index,
        y=seasonality_results["historical_daily_returns"]["mean"],
        mode="lines",
        name="Historical Average",
        line=dict(color="blue")
    ))
    
    # Add confidence interval
    fig.add_trace(go.Scatter(
        x=seasonality_results["historical_daily_returns"].index,
        y=seasonality_results["historical_daily_returns"]["ci_upper"],
        mode="lines",
        line=dict(width=0),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=seasonality_results["historical_daily_returns"].index,
        y=seasonality_results["historical_daily_returns"]["ci_lower"],
        mode="lines",
        line=dict(width=0),
        fillcolor="rgba(68, 68, 68, 0.3)",
        fill="tonexty",
        name="95% Confidence Interval"
    ))
    
    # Plot current year's returns if available
    if not seasonality_results["current_returns"].empty:
        fig.add_trace(go.Scatter(
            x=seasonality_results["current_returns"].index,
            y=seasonality_results["current_returns"],
            mode="lines+markers",
            name=f"Current Year ({datetime.now().year})",
            line=dict(color="red"),
            marker=dict(size=8)
        ))
    
    # Add zero line for reference
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    
    fig.update_layout(
        title=f"{stock.upper()} Seasonality - {month_name}",
        xaxis_title="Day of Month",
        yaxis_title="Return (%)",
        hovermode="x unified",
        xaxis=dict(tickmode="linear", tick0=1, dtick=1),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig

# Send data and chart to Discord webhook
def send_to_discord(webhook_url, message, fig):
    payload = {"content": message}
    requests.post(webhook_url, json=payload)
    
    buf = io.BytesIO()
    fig.write_image(buf, format="png")
    buf.seek(0)
    
    files = {"file": ("chart.png", buf, "image/png")}
    requests.post(webhook_url, files=files)

# Streamlit App
def main():
    st.title("Stock Seasonality & Analysis Tool")
    
    # Input parameters
    col1, col2 = st.columns(2)
    with col1:
        stock = st.text_input("Enter the stock symbol (e.g., TSLA):").strip()
        month_name = st.selectbox("Select a month:", list(MONTHS.keys()))
    with col2:
        start_date = st.date_input("Select Start Date", date(2011, 1, 1))
        end_date = st.date_input("Select End Date", date.today())
    
    webhook_url = st.text_input("Discord Webhook URL (optional):", type="password")
    
    if stock and month_name and start_date < end_date:
        month_number = get_month_number(month_name)
        with st.spinner("Fetching stock data..."):
            data = fetch_stock_data(stock, start_date, end_date)
        
        if data is not None:
            current_year = datetime.now().year
            seasonality_results = calculate_seasonality(data, month_number, current_year)
            
            # Display statistics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Historical Average Return", 
                         f"{seasonality_results['monthly_avg_return']:.2%}")
                if seasonality_results['current_month_return'] is not None:
                    st.metric("Current Year Return", 
                            f"{seasonality_results['current_month_return']:.2%}")
            
            with col2:
                st.metric("Statistical Significance", 
                         f"p-value: {seasonality_results['p_value']:.3f}")
                significant = seasonality_results['p_value'] < 0.05
                st.write(f"Seasonal effect is {'statistically significant' if significant else 'not statistically significant'}")
            
            # Plot
            fig = plot_seasonality_with_current(seasonality_results, stock, month_name)
            st.plotly_chart(fig)
            
            # Data Export Option
            csv_data = seasonality_results["historical_daily_returns"].to_csv().encode("utf-8")
            st.download_button("Download Historical Data as CSV", 
                             csv_data, 
                             "seasonality.csv", 
                             "text/csv")
            
            # Send to Discord Button
            if webhook_url:
                if st.button("Send to Discord"):
                    discord_message = f"""
                    Stock: {stock}
                    Month: {month_name}
                    Historical Avg Return: {seasonality_results['monthly_avg_return']:.2%}
                    Current Year Return: {seasonality_results['current_month_return']:.2%}
                    Statistical Significance: p-value = {seasonality_results['p_value']:.3f}
                    """
                    send_to_discord(webhook_url, discord_message, fig)
                    st.success("Sent to Discord!")
        else:
            st.error("Failed to fetch stock data. Please check the stock symbol and try again.")

if __name__ == "__main__":
    main()
