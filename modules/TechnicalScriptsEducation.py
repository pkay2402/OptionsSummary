import streamlit as st

# File paths for storing data
THINKORSWIM_SCRIPTS_FILE = "thinkorswim_scripts.txt"
THINKORSWIM_SCANS_FILE = "thinkorswim_scans.txt"
TRADINGVIEW_SCRIPTS_FILE = "tradingview_scripts.txt"
SESSION_RECORDINGS_FILE = "session_recordings.txt"

def load_data(file_path):
    """Load data from a text file."""
    try:
        with open(file_path, "r") as file:
            return [line.strip().split("|") for line in file.readlines()]
    except FileNotFoundError:
        return []

def display_technical_scripts_education():
    st.title("Technical Scripts and Education")

    # Section for ThinkorSwim Scripts
    st.header("ThinkorSwim Scripts")
    st.markdown("""
    ThinkorSwim scripts are custom scripts used in the ThinkorSwim platform for technical analysis and automation.
    """)
    thinkorswim_scripts = load_data(THINKORSWIM_SCRIPTS_FILE)
    for script in thinkorswim_scripts:
        with st.expander(f"ThinkorSwim Script: {script[0]}"):
            st.code(script[1], language="javascript")
            st.write(f"**Description:** {script[2]}")

    # Section for ThinkorSwim Scans
    st.header("ThinkorSwim Scans")
    st.markdown("""
    ThinkorSwim scans are pre-configured or custom scans for identifying trading opportunities.
    """)
    thinkorswim_scans = load_data(THINKORSWIM_SCANS_FILE)
    for scan in thinkorswim_scans:
        with st.expander(f"ThinkorSwim Scan: {scan[0]}"):
            st.write(f"**Link:** [Click here]({scan[1]})")
            st.write(f"**Description:** {scan[2]}")

    # Section for TradingView Scripts
    st.header("TradingView Scripts")
    st.markdown("""
    TradingView scripts are written in Pine Script and are used for creating custom indicators and strategies.
    """)
    tradingview_scripts = load_data(TRADINGVIEW_SCRIPTS_FILE)
    for script in tradingview_scripts:
        with st.expander(f"TradingView Script: {script[0]}"):
            st.code(script[1], language="javascript")
            st.write(f"**Description:** {script[2]}")

    # Section for Session Recordings
    st.header("Session Recordings")
    st.markdown("""
    Session recordings are videos or tutorials that provide educational content on trading strategies, tools, and techniques.
    """)
    session_recordings = load_data(SESSION_RECORDINGS_FILE)
    for recording in session_recordings:
        with st.expander(f"Session Recording: {recording[0]}"):
            st.write(f"**Link:** [Click here]({recording[1]})")
            st.write(f"**Description:** {recording[2]}")

# Run the module
if __name__ == "__main__":
    display_technical_scripts_education()
