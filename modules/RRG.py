import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.express as px

# Default parameters
DEFAULT_BENCHMARK = "SPY"
DEFAULT_STOCKS = ["AAPL", "META", "TSLA", "NVDA"]
PERIOD = "1y"  # Time period for historical data
MOMENTUM_WINDOW = 14  # Window for calculating momentum

def fetch_data(symbol, period=PERIOD):
    """Fetch historical price data for a symbol using yfinance."""
    try:
        data = yf.download(symbol, period=period)
        if data.empty:
            st.warning(f"No data received for {symbol}")
            return None
        return data['Close']
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {e}")
        return None

def calculate_relative_strength(stock_data, benchmark_data):
    """Calculate relative strength (stock price / benchmark price)."""
    if stock_data is None or benchmark_data is None:
        return None
    return (stock_data / benchmark_data).squeeze()  # Ensure it's a Series

def calculate_momentum(data, window=MOMENTUM_WINDOW):
    """Calculate momentum as the percentage change over a given window."""
    if data is None:
        return None
    return data.pct_change(window, fill_method=None).squeeze()  # Ensure it's a Series

def calculate_rrg_data(stocks, benchmark):
    """Calculate RRG data (relative strength and momentum) for the given stocks."""
    benchmark_data = fetch_data(benchmark)
    if benchmark_data is None:
        return None

    rrg_data = []
    for stock in stocks:
        stock_data = fetch_data(stock)
        if stock_data is None:
            continue

        # Calculate relative strength and momentum
        relative_strength = calculate_relative_strength(stock_data, benchmark_data)
        momentum = calculate_momentum(relative_strength)

        # Ensure relative_strength and momentum are valid
        if relative_strength is not None and momentum is not None:
            # Extract the last value from the Series
            last_relative_strength = relative_strength.iloc[-1]
            last_momentum = momentum.iloc[-1]

            # Check if the values are not NaN
            if not pd.isna(last_relative_strength) and not pd.isna(last_momentum):
                rrg_data.append({
                    "Stock": stock,
                    "Relative Strength": last_relative_strength,  # Store the numerical value
                    "Momentum": last_momentum  # Store the numerical value
                })

    return pd.DataFrame(rrg_data)

def plot_rrg(rrg_data):
    """Plot the Relative Rotation Graph using Plotly."""
    if rrg_data.empty:
        st.warning("No data available to plot.")
        return

    # Create the RRG plot
    fig = px.scatter(
        rrg_data,
        x="Relative Strength",
        y="Momentum",
        text="Stock",
        title="Relative Rotation Graph (RRG)",
        labels={
            "Relative Strength": "Relative Strength (vs Benchmark)",
            "Momentum": "Momentum"
        }
    )

    # Add quadrant lines
    fig.add_shape(
        type="line",
        x0=1, y0=rrg_data["Momentum"].min(), x1=1, y1=rrg_data["Momentum"].max(),
        line=dict(color="gray", dash="dash")
    )
    fig.add_shape(
        type="line",
        x0=rrg_data["Relative Strength"].min(), y0=0, x1=rrg_data["Relative Strength"].max(), y1=0,
        line=dict(color="gray", dash="dash")
    )

    # Add quadrant labels
    fig.add_annotation(
        x=0.5, y=0.5, text="Improving", showarrow=False, font=dict(size=14, color="blue")
    )
    fig.add_annotation(
        x=1.5, y=0.5, text="Leading", showarrow=False, font=dict(size=14, color="green")
    )
    fig.add_annotation(
        x=0.5, y=-0.5, text="Lagging", showarrow=False, font=dict(size=14, color="red")
    )
    fig.add_annotation(
        x=1.5, y=-0.5, text="Weakening", showarrow=False, font=dict(size=14, color="orange")
    )

    # Customize layout
    fig.update_traces(textposition="top center")
    fig.update_layout(
        xaxis_title="Relative Strength (vs Benchmark)",
        yaxis_title="Momentum",
        showlegend=False
    )

    st.plotly_chart(fig)

def run():
    """Main function to run the RRG app."""
    st.title("Relative Rotation Graph (RRG)")

    # User inputs
    st.sidebar.header("Configuration")
    benchmark = st.sidebar.text_input("Benchmark Index", DEFAULT_BENCHMARK)
    default_stocks = st.sidebar.text_input("Stocks to Analyze (comma-separated)", ", ".join(DEFAULT_STOCKS))
    stocks = [s.strip() for s in default_stocks.split(",")]

    # Fetch and calculate RRG data
    rrg_data = calculate_rrg_data(stocks, benchmark)
    if rrg_data is not None:
        st.write("### Relative Rotation Graph Data")
        st.dataframe(rrg_data)

        # Plot the RRG
        st.write("### Relative Rotation Graph")
        plot_rrg(rrg_data)

    st.write("This is the Relative Rotation Graph (RRG) application.")
