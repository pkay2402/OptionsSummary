import streamlit as st
from modules import flowSummary, MomentumSignals, MomentumETF, IntradaySignals, WeekendRule  # Added WeekendRule

# Set page configuration (must be the first Streamlit command)
st.set_page_config(page_title="Trading Tools Hub", layout="wide")

# Custom CSS to style buttons and improve visibility
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .icon-button {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 20px;
        margin: 10px;
        border-radius: 10px;
        background-color: #f0f2f6;
        color: black;
        transition: background-color 0.3s, transform 0.3s;
        cursor: pointer;
        text-align: center;
        height: 150px;
        border: 1px solid #ccc;
    }
    .icon-button:hover {
        background-color: #d1d9e6;
        transform: scale(1.05);
    }
    .icon-button i {
        font-size: 3em;
        margin-bottom: 10px;
    }
    .icon-button span {
        font-size: 1em;
    }
    </style>
""", unsafe_allow_html=True)

# Include Font Awesome for icons
st.markdown('<link rel="stylesheet" href="https://use.fontawesome.com/releases/v5.15.1/css/all.css" integrity="sha384-vp86vTRFVJgpjF9jiIGPEEqYqlDwgyBgEF109VFjmqGmIY/Y4HV4d3Gp2irVfcrp" crossorigin="anonymous">', unsafe_allow_html=True)

def main():
    st.title("Trading Tools Hub")

    # Create a radio button for module selection
    selected_app = st.radio(
        "Choose a module",
        ["Flow Summary", "Momentum Signals", "Momentum ETF", "Intraday Signals", "Weekend Rule"]  # Added WeekendRule
    )

    # Display the selected module's content
    if selected_app == "Flow Summary":
        flowSummary.run()
    elif selected_app == "Momentum Signals":
        MomentumSignals.run()
    elif selected_app == "Momentum ETF":
        MomentumETF.run()
    elif selected_app == "Intraday Signals":
        IntradaySignals.run()
    elif selected_app == "Weekend Rule":
        WeekendRule.run()

    # Back button to return to the main menu (clears the selection)
    if st.button("Back to Main Menu"):
        st.experimental_rerun()

if __name__ == "__main__":
    main()
