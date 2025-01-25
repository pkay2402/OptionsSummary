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

# Dictionary to map module names to their run functions
MODULES = {
    "Flow Summary": flowSummary.run,
    "Momentum Signals": MomentumSignals.run,
    "Momentum ETF": MomentumETF.run,
    "Intraday Signals": IntradaySignals.run,
}

def main():
    st.title("Trading Tools Hub")

    # Initialize session state for navigation
    if "current_module" not in st.session_state:
        st.session_state.current_module = None

    # Sidebar for navigation
    with st.sidebar:
        st.header("Navigation")
        for module_name in MODULES.keys():
            if st.button(module_name):
                st.session_state.current_module = module_name

    # Display the selected module's content
    if st.session_state.current_module:
        try:
            # Run the selected module
            MODULES[st.session_state.current_module]()
        except Exception as e:
            st.error(f"An error occurred while running {st.session_state.current_module}: {e}")
            st.session_state.current_module = None  # Reset to main menu on error

        # Back button to return to the main menu
        if st.button("Back to Main Menu"):
            st.session_state.current_module = None
    else:
        # Main menu with module selection
        st.write("Welcome to the Trading Tools Hub! Select a module from the sidebar to get started.")

if __name__ == "__main__":
    main()
