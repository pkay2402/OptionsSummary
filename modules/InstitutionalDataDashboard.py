# modules/InstitutionalDataDashboard.py
import pandas as pd
import matplotlib.pyplot as plt
import io
import requests
from datetime import datetime, timedelta
from matplotlib.colors import LinearSegmentedColormap
import streamlit as st
import yfinance as yf

# Discord webhook URL (replace with your actual webhook URL)
DISCORD_WEBHOOK_URL = 'https://discord.com/api/webhooks/1332367135023956009/8HH_RiKnSP7R7l7mtFHOB8kJi7ATt0TKZRrh35D82zycKC7JrFVSMpgJUmHrnDQ4mQRw'

# Function to download FINRA short sale data for a specific date
def download_finra_short_sale_data(date):
    url = f"https://cdn.finra.org/equity/regsho/daily/CNMSshvol{date}.txt"
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        print(f"No data available for {date}")  # Log for debugging, not shown in UI
        return None

# Function to process the downloaded data
def process_finra_short_sale_data(data):
    if not data:
        return pd.DataFrame()
    
    # Convert the data into a DataFrame
    df = pd.read_csv(io.StringIO(data), delimiter="|")
    # Clean up the data (remove header/footer rows)
    df = df[df["Symbol"].str.len() <= 4]  # Filter out non-symbol rows
    return df

# Function to fetch stock descriptions using yfinance
def get_stock_info(symbol):
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        description = info.get('longName', info.get('shortName', 'N/A'))
        return description
    except:
        return 'N/A'

# Function to fetch historical closing price for a specific date
def get_historical_closing_price(symbol, date):
    try:
        # Convert date to the format yfinance expects (YYYY-MM-DD)
        formatted_date = datetime.strptime(date, "%Y%m%d").strftime("%Y-%m-%d")
        stock = yf.Ticker(symbol)
        historical_data = stock.history(start=formatted_date, end=(datetime.strptime(date, "%Y%m%d") + timedelta(days=1)).strftime("%Y-%m-%d"))
        if not historical_data.empty:
            return historical_data['Close'].iloc[0]
        else:
            return 'N/A'
    except Exception as e:
        print(f"Error fetching historical data for {symbol} on {date}: {e}")
        return 'N/A'

# Function to fetch the latest closing price
def get_latest_close_price(symbol):
    try:
        stock = yf.Ticker(symbol)
        latest_data = stock.history(period="1d")
        if not latest_data.empty:
            return latest_data['Close'].iloc[0]
        else:
            return 'N/A'
    except Exception as e:
        print(f"Error fetching latest close price for {symbol}: {e}")
        return 'N/A'

# Function to calculate forward performance (5-day and 10-day)
def get_forward_performance(symbol, transaction_date, days_forward):
    try:
        # Convert transaction_date to datetime
        transaction_date = datetime.strptime(transaction_date, "%Y%m%d")
        
        # Calculate end date (transaction_date + days_forward)
        end_date = (transaction_date + timedelta(days=days_forward)).strftime("%Y-%m-%d")
        
        # Fetch historical data
        stock = yf.Ticker(symbol)
        historical_data = stock.history(start=transaction_date.strftime("%Y-%m-%d"), end=end_date)
        
        # Check if data is available
        if not historical_data.empty and len(historical_data) >= 2:
            start_price = historical_data['Close'].iloc[0]  # Closing price on transaction date
            end_price = historical_data['Close'].iloc[-1]  # Closing price on the last available date
            return ((end_price - start_price) / start_price) * 100  # Percentage change
        else:
            return 'N/A'  # Data not available
    except Exception as e:
        print(f"Error calculating forward performance for {symbol}: {e}")
        return 'N/A'

# Function to find stocks with total volume over 10 million
def find_stocks_over_10_million(df, date):
    stocks_over_10_million = []

    for _, row in df.iterrows():
        total_volume = row.get('TotalVolume', 0)
        if total_volume > 10000000:
            short_volume = row.get('ShortVolume', 0)
            short_exempt_volume = row.get('ShortExemptVolume', 0)

            sold_volume = short_volume + short_exempt_volume
            bought_volume = total_volume - sold_volume

            buy_to_sell_ratio = bought_volume / sold_volume if sold_volume > 0 else float('inf')

            if buy_to_sell_ratio > 1:
                symbol = row['Symbol']
                description = get_stock_info(symbol)
                closing_price = get_historical_closing_price(symbol, date)
                latest_close = get_latest_close_price(symbol)

                # Calculate forward 5-day performance
                five_day_performance = get_forward_performance(symbol, date, days_forward=5)

                # Calculate forward 10-day performance
                ten_day_performance = get_forward_performance(symbol, date, days_forward=10)

                stocks_over_10_million.append({
                    'Date': date,
                    'Symbol': symbol,
                    'Description': description,
                    'ClosingPrice': closing_price,
                    'LatestClose': latest_close,
                    '5DayPerformance': five_day_performance,
                    '10DayPerformance': ten_day_performance,
                    'TotalVolume': total_volume,
                    'SoldVolume': sold_volume,
                    'BoughtVolume': bought_volume,
                    'BuyToSellRatio': buy_to_sell_ratio
                })

    # Sort by BuyToSellRatio in descending order and take top 10
    return sorted(stocks_over_10_million, key=lambda x: x['BuyToSellRatio'], reverse=True)[:10]

# Function to analyze user-defined stocks over the last 20 days
def analyze_user_stocks(user_stocks):
    results = []
    for i in range(20):
        date = (datetime.now() - timedelta(days=i)).strftime("%Y%m%d")
        data = download_finra_short_sale_data(date)
        if data:
            df = process_finra_short_sale_data(data)
            for symbol in user_stocks:
                stock_data = df[df['Symbol'] == symbol]
                if not stock_data.empty:
                    row = stock_data.iloc[0]
                    total_volume = row.get('TotalVolume', 0)
                    short_volume = row.get('ShortVolume', 0)
                    short_exempt_volume = row.get('ShortExemptVolume', 0)

                    sold_volume = short_volume + short_exempt_volume
                    bought_volume = total_volume - sold_volume

                    description = get_stock_info(symbol)
                    closing_price = get_historical_closing_price(symbol, date)
                    latest_close = get_latest_close_price(symbol)

                    # Calculate forward 5-day performance
                    five_day_performance = get_forward_performance(symbol, date, days_forward=5)

                    # Calculate forward 10-day performance
                    ten_day_performance = get_forward_performance(symbol, date, days_forward=10)

                    results.append({
                        'Date': date,
                        'Symbol': symbol,
                        'Description': description,
                        'ClosingPrice': closing_price,
                        'LatestClose': latest_close,
                        '5DayPerformance': five_day_performance,
                        '10DayPerformance': ten_day_performance,
                        'TotalVolume': total_volume,
                        'SoldVolume': sold_volume,
                        'BoughtVolume': bought_volume,
                        'BuyToSellRatio': bought_volume / sold_volume if sold_volume > 0 else float('inf')
                    })
    return pd.DataFrame(results)

# Function to find large trades (buy-to-sell ratio > 5 and total volume > 2 million)
def find_large_trades(df, date):
    large_trades = []

    for _, row in df.iterrows():
        total_volume = row.get('TotalVolume', 0)
        if total_volume > 2000000:
            short_volume = row.get('ShortVolume', 0)
            short_exempt_volume = row.get('ShortExemptVolume', 0)

            sold_volume = short_volume + short_exempt_volume
            bought_volume = total_volume - sold_volume

            buy_to_sell_ratio = bought_volume / sold_volume if sold_volume > 0 else float('inf')

            if buy_to_sell_ratio > 5:
                symbol = row['Symbol']
                description = get_stock_info(symbol)
                closing_price = get_historical_closing_price(symbol, date)
                latest_close = get_latest_close_price(symbol)

                # Calculate forward 5-day performance
                five_day_performance = get_forward_performance(symbol, date, days_forward=5)

                # Calculate forward 10-day performance
                ten_day_performance = get_forward_performance(symbol, date, days_forward=10)

                large_trades.append({
                    'Date': date,
                    'Symbol': symbol,
                    'Description': description,
                    'ClosingPrice': closing_price,
                    'LatestClose': latest_close,
                    '5DayPerformance': five_day_performance,
                    '10DayPerformance': ten_day_performance,
                    'TotalVolume': total_volume,
                    'SoldVolume': sold_volume,
                    'BoughtVolume': bought_volume,
                    'BuyToSellRatio': buy_to_sell_ratio
                })

    return pd.DataFrame(large_trades)

# Function to create a dashboard
def create_dashboard(df):
    plt.figure(figsize=(15, 10))

    # Plot: Table with highlighted high Buy-to-Sell Ratios (Top 10)
    table_data = df[['Symbol', 'Description', 'ClosingPrice', 'LatestClose', '5DayPerformance', '10DayPerformance', 'TotalVolume', 'BoughtVolume', 'SoldVolume', 'BuyToSellRatio']].round(2)

    # Custom colormap for highlighting
    colors = [(1, 1, 1), (0, 1, 0)]  # From white to green
    cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=100)

    # Normalize the 'BuyToSellRatio' for color mapping
    norm = plt.Normalize(df['BuyToSellRatio'].min(), df['BuyToSellRatio'].max())

    # Create colors for each row
    row_colors = plt.cm.ScalarMappable(norm=norm, cmap=cmap).to_rgba(df['BuyToSellRatio'])
    cell_colours = [[row_colors[i] for _ in range(len(table_data.columns))] for i in range(len(table_data))]

    # Create the table with colored cells
    the_table = plt.table(cellText=table_data.values, colLabels=table_data.columns, 
                          cellColours=cell_colours,
                          loc='center')
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(10)
    the_table.scale(1, 1.5)

    plt.title('Top 10 Stocks by Buy-to-Sell Ratio (High to Low)')
    plt.axis('off')

    plt.tight_layout()

    # Convert plot to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return buf

# Function to send chart to Discord using a webhook
def send_chart_to_discord_webhook(image_buffer, webhook_url):
    files = {'file': ('dashboard.png', image_buffer, 'image/png')}
    response = requests.post(webhook_url, files=files)
    if response.status_code == 204:
        print("Chart sent to Discord successfully!")
    else:
        print(f"Failed to send chart to Discord: {response.status_code} - {response.text}")

# Main function for the Institutional Data Dashboard
def run():
    st.title("Institutional Data Dashboard")

    # Date input for user to select the date
    selected_date = st.date_input("Select a date", datetime.now() - timedelta(days=1))
    formatted_date = selected_date.strftime("%Y%m%d")

    # Download and process data
    data = download_finra_short_sale_data(formatted_date)
    if data:
        df = process_finra_short_sale_data(data)
        if not df.empty:
            # Top 10 stocks by volume > 10 million
            top_stocks = find_stocks_over_10_million(df, formatted_date)
            top_stocks_df = pd.DataFrame(top_stocks)

            # Display the table in Streamlit
            st.write("Top 10 Stocks by Buy-to-Sell Ratio:")
            st.dataframe(top_stocks_df)

            # Create and display the dashboard
            image_buffer = create_dashboard(top_stocks_df)
            st.image(image_buffer, caption="Top 10 Stocks Dashboard")

            # Option to send the dashboard to Discord
            if st.button("Send to Discord"):
                send_chart_to_discord_webhook(image_buffer, DISCORD_WEBHOOK_URL)
                st.success("Chart sent to Discord!")
        else:
            st.warning("No valid data found for the selected date.")
    else:
        st.warning("Failed to download data for the selected date.")

    # User-defined stock filter
    st.sidebar.title("User-Defined Stock Analysis")
    user_stocks = st.sidebar.text_input("Enter up to 5 stock symbols (comma-separated)", "AAPL,TSLA,AMZN").strip().upper().split(",")[:5]
    if st.sidebar.button("Analyze User Stocks"):
        user_stocks_df = analyze_user_stocks(user_stocks)
        if not user_stocks_df.empty:
            st.write("Analysis of User-Defined Stocks Over Last 20 Days:")
            st.dataframe(user_stocks_df)
        else:
            st.warning("No data found for the selected stocks.")

    # Large trades filter
    st.sidebar.title("Large Trades Analysis")
    if st.sidebar.button("Find Large Trades"):
        large_trades_df = pd.DataFrame()
        for i in range(20):
            date = (datetime.now() - timedelta(days=i)).strftime("%Y%m%d")
            data = download_finra_short_sale_data(date)
            if data:
                df = process_finra_short_sale_data(data)
                large_trades = find_large_trades(df, date)
                large_trades_df = pd.concat([large_trades_df, large_trades])
        
        if not large_trades_df.empty:
            st.write("Large Trades (Buy-to-Sell Ratio > 5 and Total Volume > 2M):")
            st.dataframe(large_trades_df)
        else:
            st.warning("No large trades found in the last 20 days.")

# Run the app
if __name__ == "__main__":
    run()
