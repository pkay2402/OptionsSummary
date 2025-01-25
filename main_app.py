import streamlit as st
from modules import flowSummary, MomentumSignals, MomentumETF, IntradaySignals

# Custom CSS to hide Streamlit's default menu and footer
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Custom CSS for icons
st.markdown("""
    <style>
    .icon {
        display: flex;
        flex-direction: column;
        align-items: center;
        padding: 20px;
        margin: 10px;
        border-radius: 10px;
        background-color: #f0f2f6;
        transition: background-color 0.3s; /* Smooth transition for hover effect */
    }
    .icon:hover {
        background-color: #d1d9e6;
    }
    .icon i {
        font-size: 3em;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

def main():
    st.title("Trading Tools Hub")

    # Use columns to layout icons
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("<i class='fas fa-chart-line'></i><br>Flow Summary", help="Analyze options flow", use_container_width=True):
            flowSummary.run()

    with col2:
        if st.button("<i class='fas fa-tachometer-alt'></i><br>Momentum Signals", help="View stock momentum", use_container_width=True):
            MomentumSignals.run()

    with col3:
        if st.button("<i class='fas fa-globe'></i><br>Momentum ETF", help="Check ETF momentum", use_container_width=True):
            MomentumETF.run()

    with col4:
        if st.button("<i class='fas fa-clock'></i><br>Intraday Signals", help="Intraday trading signals", use_container_width=True):
            IntradaySignals.run()

if __name__ == "__main__":
    main()
