import streamlit as st
from modules import flowSummary, MomentumSignals, MomentumETF, IntradaySignals

# Custom CSS to hide Streamlit's default menu and footer, and style buttons
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .icon-button {
        display: flex;
        flex-direction: column;
        align-items: center;
        padding: 20px;
        margin: 10px;
        border-radius: 10px;
        background-color: #f0f2f6;
        transition: background-color 0.3s;
        cursor: pointer;
    }
    .icon-button:hover {
        background-color: #d1d9e6;
    }
    .icon-button i {
        font-size: 3em;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Include Font Awesome for icons
st.markdown('<link rel="stylesheet" href="https://use.fontawesome.com/releases/v5.15.1/css/all.css" integrity="sha384-vp86vTRFVJgpjF9jiIGPEEqYqlDwgyBgEF109VFjmqGmIY/Y4HV4d3Gp2irVfcrp" crossorigin="anonymous">', unsafe_allow_html=True)

def main():
    st.title("Trading Tools Hub")

    # Use columns to layout icons
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown('<div class="icon-button" onclick="flowSummary.run()"><i class="fas fa-chart-line"></i><br>Flow Summary</div>', unsafe_allow_html=True)
        if st.button("Flow Summary", key="flowSummary"):
            flowSummary.run()

    with col2:
        st.markdown('<div class="icon-button" onclick="MomentumSignals.run()"><i class="fas fa-tachometer-alt"></i><br>Momentum Signals</div>', unsafe_allow_html=True)
        if st.button("Momentum Signals", key="momentumSignals"):
            MomentumSignals.run()

    with col3:
        st.markdown('<div class="icon-button" onclick="MomentumETF.run()"><i class="fas fa-globe"></i><br>Momentum ETF</div>', unsafe_allow_html=True)
        if st.button("Momentum ETF", key="momentumETF"):
            MomentumETF.run()

    with col4:
        st.markdown('<div class="icon-button" onclick="IntradaySignals.run()"><i class="fas fa-clock"></i><br>Intraday Signals</div>', unsafe_allow_html=True)
        if st.button("Intraday Signals", key="intradaySignals"):
            IntradaySignals.run()

if __name__ == "__main__":
    main()
