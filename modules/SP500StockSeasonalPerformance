import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
from typing import List, Tuple, Optional

# Month dictionary
MONTHS = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12
}

def get_month_number(month_name: str) -> int:
    """Convert month name to number."""
    return MONTHS[month_name]

def get_top_200_stocks() -> List[str]:
    """Fetch top 200 S&P 500 stocks from Wikipedia."""
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        sp500_df = tables[0]  # First table contains the stock list
        return sp500_df["Symbol"].tolist()[:200]  # Limit to top 200
    except Exception as e:
        st.error(f"Error fetching S&P 500 stocks: {e}")
        return []

def fetch_stock_data(symbol: str, start_date: datetime.date, end_date: datetime.date) -> Optional[pd.DataFrame]:
    """Fetch historical stock data from Yahoo Finance."""
    try:
        data = yf.download(symbol, start=start_date, end=end_date, progress=False)
        if data.empty:
            st.warning(f"No data available for {symbol}")
            return None
        data["Month"] = data.index.month
        data["Year"] = data.index.year
        return data
    except Exception as e:
        st.warning(f"Error fetching data for {symbol}: {e}")
        return None

def calculate_monthly_return(data: pd.DataFrame, month_number: int) -> Optional[float]:
    """Calculate average monthly return for a given month across all years."""
    if data is None or "Close" not in data.columns:
        return None

    try:
        # Filter for the specified month
        monthly_data = data[data["Month"] == month_number].copy()
        if len(monthly_data) < 2:
            return None

        # Group by year to calculate returns for each instance of the month
        monthly_returns = []
        for year in monthly_data["Year"].unique():
            year_data = monthly_data[monthly_data["Year"] == year]
            if len(year_data) >= 2:
                start_price = float(year_data["Close"].iloc[0])
                end_price = float(year_data["Close"].iloc[-1])
                if not (pd.isna(start_price) or pd.isna(end_price)):
                    monthly_return = (end_price - start_price) / start_price
                    monthly_returns.append(monthly_return)

        # Calculate average monthly return across all years
        if monthly_returns:
            return sum(monthly_returns) / len(monthly_returns)
        return None

    except (IndexError, ValueError, TypeError) as e:
        st.warning(f"Error calculating return: {e}")
        return None

def find_high_performing_stocks(
    month_name: str,
    start_date: datetime.date,
    end_date: datetime.date
) -> List[Tuple[str, float]]:
    """Find stocks with average return > 3% in selected month."""
    month_number = get_month_number(month_name)
    top_stocks = get_top_200_stocks()
    high_performers = []
    
    # Add progress bar
    progress_bar = st.progress(0)
    total_stocks = len(top_stocks)
    
    for idx, stock in enumerate(top_stocks):
        # Update progress
        progress = (idx + 1) / total_stocks
        progress_bar.progress(progress)
        
        data = fetch_stock_data(stock, start_date, end_date)
        if data is not None and not data.empty:
            avg_return = calculate_monthly_return(data, month_number)
            if avg_return is not None and avg_return > 0.03:  # 3% threshold
                high_performers.append((stock, avg_return))
    
    progress_bar.empty()  # Clear progress bar when done
    return sorted(high_performers, key=lambda x: x[1], reverse=True)

def main():
    """Main Streamlit application."""
    st.set_page_config(page_title="Stock Seasonality & Analysis Tool", layout="wide")
    st.title("S&P 500 Stocks Monthly Average Performance")
    
    # Add description
    st.markdown("""
    This tool analyzes the top 200 S&P 500 stocks to find those that historically 
    perform well in specific months. It identifies stocks with average monthly 
    returns greater than 3%.
    """)
    
    # Create three columns for input controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        month_name = st.selectbox("Select a month:", list(MONTHS.keys()))
    with col2:
        start_date = st.date_input(
            "Select Start Date",
            datetime.date(2011, 1, 1),
            min_value=datetime.date(2000, 1, 1),
            max_value=datetime.date.today()
        )
    with col3:
        end_date = st.date_input(
            "Select End Date",
            datetime.date.today(),
            min_value=start_date,
            max_value=datetime.date.today()
        )

    if st.button("Find High-Performing Stocks"):
        if start_date >= end_date:
            st.error("Start date must be before end date!")
            return
            
        with st.spinner("Analyzing top stocks..."):
            high_performers = find_high_performing_stocks(month_name, start_date, end_date)
            
            if high_performers:
                df = pd.DataFrame(high_performers, columns=["Stock", "Avg Monthly Return"])
                df["Avg Monthly Return"] = df["Avg Monthly Return"].map("{:.2%}".format)
                
                st.success(f"Found {len(high_performers)} stocks with >3% average return in {month_name}")
                st.dataframe(df, use_container_width=True)
                
                # Add download button
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name=f"high_performers_{month_name}.csv",
                    mime="text/csv"
                )
            else:
                st.warning(f"No stocks found with >3% average return in {month_name}.")

if __name__ == "__main__":
    main()
