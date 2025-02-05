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

# Function to convert month name to number
def get_month_number(month_name):
    return MONTHS[month_name]

# Fetch historical stock data from Yahoo Finance
def fetch_stock_data(symbol, start_date, end_date):
    try:
        data = yf.download(symbol, start=start_date, end=end_date)
        data["Day"] = data.index.day
        data["Month"] = data.index.month
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# Calculate seasonality and volatility
def calculate_seasonality(data, month_number):
    data["Daily Return"] = data["Close"].pct_change()
    monthly_data = data[data["Month"] == month_number]
    daily_avg_returns = monthly_data.groupby("Day")["Daily Return"].mean()
    monthly_avg_return = daily_avg_returns.mean()
    volatility = monthly_data["Daily Return"].std()
    return daily_avg_returns, monthly_avg_return, volatility

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
    month_name = st.selectbox("Select a month:", list(MONTHS.keys()))
    start_date = st.date_input("Select Start Date", datetime.date(2011, 1, 1))
    end_date = st.date_input("Select End Date", datetime.date(2025, 1, 1))
    webhook_url = "https://discord.com/api/webhooks/1332242383458406401/6KXAsHFsvTKgDZyDimQ_ncrBx9vePgsOYxSRjga0mK-Zg2m404r65zzqdyL1bKCQRwVO"
    
    if stock and month_name and start_date < end_date:
        month_number = get_month_number(month_name)
        with st.spinner("Fetching stock data..."):
            data = fetch_stock_data(stock, start_date, end_date)
        
        if data is not None:
            daily_avg_returns, monthly_avg_return, volatility = calculate_seasonality(data, month_number)
            
            st.header(f"Average Return for {month_name}: {monthly_avg_return:.2%}")
            st.subheader(f"Volatility: {volatility:.2%}")
            
            # Interactive Chart
            fig = px.bar(x=daily_avg_returns.index, y=daily_avg_returns.values, labels={"x": "Day", "y": "Average Return (%)"},
                         title=f"{stock.upper()} Seasonality - {month_name}", color=daily_avg_returns.values)
            st.plotly_chart(fig)
            
            # Data Export Option
            csv_data = daily_avg_returns.to_csv().encode("utf-8")
            st.download_button("Download Data as CSV", csv_data, "seasonality.csv", "text/csv")
            
            # Send to Discord Button
            if webhook_url:
                if st.button("Send to Discord"):
                    discord_message = f"Stock: {stock}\nMonth: {month_name}\nAvg Return: {monthly_avg_return:.2%}\nVolatility: {volatility:.2%}"
                    send_to_discord(webhook_url, discord_message, fig)
                    st.success("Sent to Discord!")
        else:
            st.error("Failed to fetch stock data. Please check the stock symbol and try again.")

if __name__ == "__main__":
    main()
