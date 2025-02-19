# main.py
import streamlit as st
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
from modules.CFTC import cftc_analyzer_module
from modules.BlockTrade import Blocktrade_run  # Corrected case to match BlockTrade.py

# [add_buymeacoffee function remains unchanged]

def main():
    st.title("Trading Tools Hub")
    
    st.markdown("""
        <style>
        .css-1544g2n {
            padding-bottom: 60px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    app_selection = st.sidebar.selectbox("Choose the app:", 
                                      ["FINRA Dashboard",
                                       "Block Trades",  # Matches the if condition below
                                       "Stock Trend Oscillator",
                                       "Flow Summary", 
                                       "Momentum Signals", 
                                       "Momentum ETF",
                                       "GEX Analysis",
                                       "Stock Analysis",
                                       "TOS Scanner",
                                       "Gann Swing Analysis",
                                       "Seasonality",
                                       "S&P 500 Performance",
                                       "CFTC Data Analyzer"])
    
    if app_selection == "Stock Trend Oscillator":
        show_trend_oscillator()
    elif app_selection == "Flow Summary":
        flowSummary_run()
    elif app_selection == "Block Trades":  # This is correct and matches the selectbox option
        Blocktrade_run()  # This will now work with the corrected import
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
    elif app_selection == "Seasonality":
        Seasonality_run()
    elif app_selection == "S&P 500 Performance":
        SP500Performance_run()
    elif app_selection == "CFTC Data Analyzer":
        cftc_analyzer_module()
        
    add_buymeacoffee()

if __name__ == "__main__":
    main()
