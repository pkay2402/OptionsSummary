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

    # State to control which module is currently displayed
    if 'selected_app' not in st.session_state:
        st.session_state['selected_app'] = None

    # Use columns to layout icons
    col1, col2, col3, col4 = st.columns(4)

    # We'll use markdown to create clickable divs instead of buttons
    with col1:
        st.markdown('<div class="icon-button" onclick="runFlowSummary()"><i class="fas fa-chart-line"></i><span>Flow Summary</span></div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="icon-button" onclick="runMomentumSignals()"><i class="fas fa-tachometer-alt"></i><span>Momentum Signals</span></div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="icon-button" onclick="runMomentumETF()"><i class="fas fa-globe"></i><span>Momentum ETF</span></div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="icon-button" onclick="runIntradaySignals()"><i class="fas fa-clock"></i><span>Intraday Signals</span></div>', unsafe_allow_html=True)

    # Check for URL hash to determine which module to run
    if st.experimental_get_query_params().get('module'):
        module = st.experimental_get_query_params()['module'][0]
        st.session_state['selected_app'] = module

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
            st.experimental_set_query_params()

    # JavaScript to handle onclick events since Streamlit doesn't support them directly
    st.markdown('''
    <script>
    function runFlowSummary() {
        window.location.hash = '#Flow Summary';
        window.location.search = '?module=Flow%20Summary';
    }
    function runMomentumSignals() {
        window.location.hash = '#Momentum Signals';
        window.location.search = '?module=Momentum%20Signals';
    }
    function runMomentumETF() {
        window.location.hash = '#Momentum ETF';
        window.location.search = '?module=Momentum%20ETF';
    }
    function runIntradaySignals() {
        window.location.hash = '#Intraday Signals';
        window.location.search = '?module=Intraday%20Signals';
    }
    </script>
    ''', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
