import streamlit as st
from modules import flowSummary, MomentumSignals, MomentumETF, IntradaySignals

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
        color: black; /* Ensure text color is visible */
        transition: background-color 0.3s, transform 0.3s;
        cursor: pointer;
        text-align: center;
        height: 150px; /* Fixed height for consistency */
    }
    .icon-button:hover {
        background-color: #d1d9e6;
        transform: scale(1.05); /* Slight zoom effect on hover */
    }
    .icon-button i {
        font-size: 3em;
        margin-bottom: 10px;
    }
    .icon-button span {
        font-size: 1em; /* Adjust text size */
    }
    </style>
""", unsafe_allow_html=True)

# Include Font Awesome for icons
st.markdown('<link rel="stylesheet" href="https://use.fontawesome.com/releases/v5.15.1/css/all.css" integrity="sha384-vp86vTRFVJgpjF9jiIGPEEqYqlDwgyBgEF109VFjmqGmIY/Y4HV4d3Gp2irVfcrp" crossorigin="anonymous">', unsafe_allow_html=True)

def main():
    st.title("Trading Tools Hub")

    # State to control which module is currently displayed
    if 'selected_app' not in st.session_state:
        st.session_state['selected_app'] = None

    # Use columns to layout icons
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button('<div class="icon-button"><i class="fas fa-chart-line"></i><span>Flow Summary</span></div>', key="flowSummary"):
            st.session_state['selected_app'] = 'Flow Summary'

    with col2:
        if st.button('<div class="icon-button"><i class="fas fa-tachometer-alt"></i><span>Momentum Signals</span></div>', key="momentumSignals"):
            st.session_state['selected_app'] = 'Momentum Signals'

    with col3:
        if st.button('<div class="icon-button"><i class="fas fa-globe"></i><span>Momentum ETF</span></div>', key="momentumETF"):
            st.session_state['selected_app'] = 'Momentum ETF'

    with col4:
        if st.button('<div class="icon-button"><i class="fas fa-clock"></i><span>Intraday Signals</span></div>', key="intradaySignals"):
            st.session_state['selected_app'] = 'Intraday Signals'

    # Display the selected module
    if st.session_state['selected_app']:
        st.write(f"### {st.session_state['selected_app']}")
        if st.session_state['selected_app'] == 'Flow Summary':
            flowSummary.run()
        elif st.session_state['selected_app'] == 'Momentum Signals':
            MomentumSignals.run()
        elif st.session_state['selected_app'] == 'Momentum ETF':
            MomentumETF.run()
        elif st.session_state['selected_app'] == 'Intraday Signals':
            IntradaySignals.run()

        # Add a back button to return to the main menu
        if st.button("Back to Main Menu"):
            st.session_state['selected_app'] = None

if __name__ == "__main__":
    main()
