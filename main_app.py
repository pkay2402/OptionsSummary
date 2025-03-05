# Instructions for integrating the DeMark Sequential module into your Trading Tools Hub

# Step 1: Save the DeMark Sequential module
# Save the provided code as a file named "DemarkSequential.py" in your "modules" directory

# Step 2: Update your main app.py file
# Add the import statement for the new module:

import streamlit as st  # This MUST be the first Streamlit command
st.set_page_config(layout="wide", page_title="Trading Tools Hub")

from modules.flowSummary import run as flowSummary_run
from modules.MomentumSignals import run as MomentumSignals_run
from modules.MomentumETF import run as MomentumETF_run
from modules.GexAnalysis import run as GexAnalysis_run
from modules.finra_dashboard import run as finra_dashboard_run
from modules.TosScan import run as TosScan_run
from modules.StockAnalysis import run as StockAnalysis_run
from modules.Seasonality import run as Seasonality_run
from modules.SP500Performance import run as SP500Performance_run
from modules.StockTrendOscillator import show_trend_oscillator
from modules.GannSwing import run as GannSwing_run
from modules.GannCycleAnalysis import run as GannCycle_run
from modules.CFTC import cftc_analyzer_module
from modules.BlockTrade import Blocktrade_run
from modules.WhalePositioning import run as WhalePositioning_run
from modules.StockInsights import run as StockInsights_run
from modules.DemarkSequential import run as DemarkSequential_run  # Add this line

def add_buymeacoffee():
    try:
        st.sidebar.markdown("---")
        st.sidebar.markdown("""
            <div style="text-align: center;">
                <p>If you find these tools helpful, consider supporting the project:</p>
                <a href="https://www.buymeacoffee.com/tosalerts33" target="_blank">
                    <img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" 
                         alt="Buy Me A Coffee" 
                         style="height: 45px; width: 162px;">
                </a>
            </div>
            """, unsafe_allow_html=True)
    except Exception as e:
        st.sidebar.error(f"Error rendering Buy Me a Coffee button: {str(e)}")

def main():
    st.title("Trading Tools Hub")
    
    # Add custom CSS to ensure sidebar footer stays at bottom
    st.markdown("""
        <style>
        .css-1544g2n {
            padding-bottom: 60px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    add_buymeacoffee()
    
    app_selection = st.sidebar.selectbox("Choose the app:",
                                      ["Stock Insights",
                                       "FINRA Dashboard",
                                       "Block Trades",
                                       "Stock Trend Oscillator",
                                       "Flow Summary", 
                                       "Momentum Signals", 
                                       "Momentum ETF",
                                       "GEX Analysis",
                                       "Stock Analysis",
                                       "TOS Scanner",
                                       "Gann Swing Analysis",
                                       "Gann Cycle Analysis",
                                       "Seasonality",
                                       "S&P 500 Performance",
                                       "CFTC Data Analyzer",
                                       "Whale Positioning",
                                       "DeMark Sequential"])  # Add this option

    if app_selection == "Stock Trend Oscillator":
        show_trend_oscillator()
    elif app_selection == "Flow Summary":
        flowSummary_run()
    elif app_selection == "Block Trades":
        Blocktrade_run()
    elif app_selection == "Momentum Signals":
        MomentumSignals_run()
    elif app_selection == "Momentum ETF":
        MomentumETF_run()
    elif app_selection == "GEX Analysis":
        GexAnalysis_run()
    elif app_selection == "FINRA Dashboard":
        finra_dashboard_run()
    elif app_selection == "TOS Scanner":
        TosScan_run()
    elif app_selection == "Stock Analysis":
        StockAnalysis_run()
    elif app_selection == "Gann Swing Analysis":
        GannSwing_run()
    elif app_selection == "Gann Cycle Analysis":
        GannCycle_run()
    elif app_selection == "Seasonality":
        Seasonality_run()
    elif app_selection == "S&P 500 Performance":
        SP500Performance_run()
    elif app_selection == "CFTC Data Analyzer":
        cftc_analyzer_module()
    elif app_selection == "Whale Positioning":
        WhalePositioning_run()
    elif app_selection == "Stock Insights":
        StockInsights_run()
    elif app_selection == "DeMark Sequential":  # Add this condition
        DemarkSequential_run()

if __name__ == "__main__":
    main()
