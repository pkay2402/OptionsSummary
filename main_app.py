import streamlit as st
from modules import flowSummary, MomentumSignals, MomentumETF, IntradaySignals

def main():
    st.title("Trading Tools Hub")

    # Sidebar for navigation
    app_selection = st.sidebar.selectbox("Choose the app:", 
                                          ["Flow Summary", 
                                           "Momentum Signals", 
                                           "Momentum ETF", 
                                           "Intraday Signals"])

    if app_selection == "Flow Summary":
        flowSummary.run()
    elif app_selection == "Momentum Signals":
        MomentumSignals.run()
    elif app_selection == "Momentum ETF":
        MomentumETF.run()
    elif app_selection == "Intraday Signals":
        IntradaySignals.run()

if __name__ == "__main__":
    main()
