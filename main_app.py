from modules.flowSummary import run as flowSummary_run
from modules.MomentumSignals import run as MomentumSignals_run
from modules.MomentumETF import run as MomentumETF_run
from modules.IntradaySignals import run as IntradaySignals_run
from modules.InstitutionalDataDashboard import run as InstitutionalDataDashboard_run
#from modules.StockInsights import run as StockInsights_run
from modules.WeekendRule import run as WeekendRule_run
from modules.TechnicalScriptsEducation import display_technical_scripts_education
# New module imports
from modules.GexAnalysis import run as GexAnalysis_run
from modules.finra_dashboard import run as finra_dashboard_run
from modules.TosScan import run as TosScan_run
from modules.Stock_analysis import run as Stock_analysis_run
import streamlit as st

def main():
    st.title("Trading Tools Hub")
    
    app_selection = st.sidebar.selectbox("Choose the app:", 
                                      ["Flow Summary", 
                                       "Momentum Signals", 
                                       "Momentum ETF", 
                                       "Intraday Signals",
                                       "Institutional Data Dashboard",
                                       #"Stock Insights", 
                                       "Weekend Rule",
                                       "Technical Scripts and Education",
                                       "GEX Analysis",          # New option
                                       "FINRA Dashboard",       # New option
                                       "TOS Scanner",          # New option
                                       "Stock Analysis"])      # New option
    
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
    #elif app_selection == "Stock Insights":
        #StockInsights_run()
    elif app_selection == "Weekend Rule":
        WeekendRule_run()
    elif app_selection == "Technical Scripts and Education":
        display_technical_scripts_education()
    # New module routing
    elif app_selection == "GEX Analysis":
        GexAnalysis_run()
    elif app_selection == "FINRA Dashboard":
        finra_dashboard_run()
    elif app_selection == "TOS Scanner":
        TosScan_run()
    elif app_selection == "Stock Analysis":
        Stock_analysis_run()

    # Buy Me a Coffee Section
    st.sidebar.markdown("---")  # Add a separator
    st.sidebar.header("Support Me")
    st.sidebar.markdown(
        """
        If you find this app helpful, consider supporting me by buying me a coffee!
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        """
        <a href="https://www.buymeacoffee.com/yourusername" target="_blank">
            <img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" style="height: 40px !important; width: 145px !important;">
        </a>
        """,
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()
