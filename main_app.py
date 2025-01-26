from modules.flowSummary import run as flowSummary_run
from modules.MomentumSignals import run as MomentumSignals_run
from modules.MomentumETF import run as MomentumETF_run
from modules.IntradaySignals import run as IntradaySignals_run
from modules.InstitutionalDataDashboard import run as InstitutionalDataDashboard_run
from modules.StockInsights import run as StockInsights_run
from modules.WeekendRule import run as WeekendRule_run
from modules.TechnicalScriptsEducation import display_technical_scripts_education  # Import the new module
import streamlit as st

def main():
    st.title("Trading Tools Hub")

    # Add "Technical Scripts and Education" to the sidebar options
    app_selection = st.sidebar.selectbox("Choose the app:", 
                                          ["Flow Summary", 
                                           "Momentum Signals", 
                                           "Momentum ETF", 
                                           "Intraday Signals",
                                           "Institutional Data Dashboard",
                                           "Stock Insights", 
                                           "Weekend Rule",
                                           "Technical Scripts and Education"])  # Added new option

    # Route to the selected app
    if app_selection == "Flow Summary":
        flowSummary_run()
    elif app_selection == "Momentum Signals":
        MomentumSignals_run()
    elif app_selection == "Momentum ETF":
        MomentumETF_run()
    elif app_selection == "Intraday Signals":
        IntradaySignals_run()
    elif app_selection == "Institutional Data Dashboard":
        InstitutionalDataDashboard_run()
    elif app_selection == "Stock Insights":
        StockInsights_run()
    elif app_selection == "Weekend Rule":
        WeekendRule_run()
    elif app_selection == "Technical Scripts and Education":  # Added new condition
        display_technical_scripts_education()

if __name__ == "__main__":
    main()
