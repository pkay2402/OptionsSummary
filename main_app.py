# main.py
from modules.flowSummary import run as flowSummary_run
from modules.MomentumSignals import run as MomentumSignals_run
from modules.MomentumETF import run as MomentumETF_run
from modules.IntradaySignals import run as IntradaySignals_run
from modules.InstitutionalDataDashboard import run as InstitutionalDataDashboard_run  # Add this line
import streamlit as st

def main():
    st.title("Trading Tools Hub")

    app_selection = st.sidebar.selectbox("Choose the app:", 
                                          ["Flow Summary", 
                                           "Momentum Signals", 
                                           "Momentum ETF", 
                                           "Intraday Signals",
                                           "Institutional Data Dashboard"])  # Add this option

    if app_selection == "Flow Summary":
        flowSummary_run()
    elif app_selection == "Momentum Signals":
        MomentumSignals_run()
    elif app_selection == "Momentum ETF":
        MomentumETF_run()
    elif app_selection == "Intraday Signals":
        IntradaySignals_run()
    elif app_selection == "Institutional Data Dashboard":  # Add this condition
        InstitutionalDataDashboard_run()

if __name__ == "__main__":
    main()
