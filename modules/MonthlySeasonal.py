import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import requests
import datetime
import io

# Month dictionary
MONTHS = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12
}

# Convert month name to number
def get_month_number(month_name):
    return MONTHS[month_name]

# Fetch historical stock data from Yahoo Finance
def fetch_stock_data(symbol, start_date, end_date):
    try:
        data = yf.download(symbol, start=start_date, end=end_date)
        data["Year"] = data.index.year
        data["Month"] = data.index.month
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# Calculate Optuma-style seasonality index
def calculate_seasonality(data):
    monthly_avg_prices = data.groupby(["Year", "Month"])["Close"].mean().reset_index()
    
    yearly_avg_prices = monthly_avg_prices.groupby("Year")["Close"].mean().reset_index()
    yearly_avg_prices.rename(columns={"Close": "Yearly_Avg"}, inplace=True)
    
    seasonality_data = pd.merge(monthly_avg_prices, yearly_avg_prices, on="Year")
    seasonality_data["Seasonality Index"] = (seasonality_data["Close"] / seasonality_data["Yearly_Avg"]) * 100
    
    seasonality_index = seasonality_data.groupby("Month")["Seasonality Index"].mean()
    
    return seasonality_index

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
    
    stock = st.text_input("Enter the stock symbol (e.g., TSLA):").strip()
    start_date = st.date_input("Select Start Date", datetime.date(2011, 1, 1))
    end_date = st.date_input("Select End Date", datetime.date(2025, 1, 1))
    
    if stock and start_date < end_date:
        with st.spinner("Fetching stock data..."):
            data = fetch_stock_data(stock, start_date, end_date)
        
        if data is not None:
            seasonality_index = calculate_seasonality(data)
            
            st.header("Seasonality Index (Optuma Method)")
            st.subheader("Values above 100 indicate historically strong months, below 100 indicate weaker months.")
            
            # Interactive Chart
            fig = px.bar(
                x=[list(MONTHS.keys())[i-1] for i in seasonality_index.index], 
                y=seasonality_index.values,
                labels={"x": "Month", "y": "Seasonality Index"},
                title=f"{stock.upper()} Seasonality Index",
                color=seasonality_index.values
            )
            st.plotly_chart(fig)
            
            # Data Export Option
            csv_data = seasonality_index.to_csv().encode("utf-8")
            st.download_button("Download Data as CSV", csv_data, "seasonality.csv", "text/csv")
        else:
            st.error("Failed to fetch stock data. Please check the stock symbol and try again.")

if __name__ == "__main__":
    main()
