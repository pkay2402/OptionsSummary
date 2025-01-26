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

def save_data(file_path, data):
    """Save data to a text file."""
    with open(file_path, "w") as file:
        for item in data:
            file.write("|".join(item) + "\n")

def display_technical_scripts_education():
    st.title("Technical Scripts and Education")

    # Section for ThinkorSwim Scripts
    st.header("ThinkorSwim Scripts")
    st.markdown("""
    ThinkorSwim scripts are custom scripts used in the ThinkorSwim platform for technical analysis and automation.
    """)
    with st.expander("Add a New ThinkorSwim Script"):
        script_name = st.text_input("Script Name", key="thinkorswim_script_name")
        script_code = st.text_area("Script Code", key="thinkorswim_script_code")
        script_description = st.text_area("Description", key="thinkorswim_script_description")
        if st.button("Save ThinkorSwim Script"):
            if script_name and script_code:
                data = load_data(THINKORSWIM_SCRIPTS_FILE)
                data.append([script_name, script_code, script_description])
                save_data(THINKORSWIM_SCRIPTS_FILE, data)
                st.success("ThinkorSwim Script saved!")
            else:
                st.warning("Please provide a script name and code.")

    thinkorswim_scripts = load_data(THINKORSWIM_SCRIPTS_FILE)
    for idx, script in enumerate(thinkorswim_scripts):
        with st.expander(f"ThinkorSwim Script: {script[0]}"):
            st.code(script[1], language="javascript")
            st.write(f"**Description:** {script[2]}")
            if st.button(f"Delete {script[0]}", key=f"delete_thinkorswim_script_{idx}"):
                thinkorswim_scripts.pop(idx)
                save_data(THINKORSWIM_SCRIPTS_FILE, thinkorswim_scripts)
                st.experimental_rerun()

    # Section for ThinkorSwim Scans
    st.header("ThinkorSwim Scans")
    st.markdown("""
    ThinkorSwim scans are pre-configured or custom scans for identifying trading opportunities.
    """)
    with st.expander("Add a New ThinkorSwim Scan"):
        scan_name = st.text_input("Scan Name", key="thinkorswim_scan_name")
        scan_link = st.text_input("Scan Link", key="thinkorswim_scan_link")
        scan_description = st.text_area("Description", key="thinkorswim_scan_description")
        if st.button("Save ThinkorSwim Scan"):
            if scan_name and scan_link:
                data = load_data(THINKORSWIM_SCANS_FILE)
                data.append([scan_name, scan_link, scan_description])
                save_data(THINKORSWIM_SCANS_FILE, data)
                st.success("ThinkorSwim Scan saved!")
            else:
                st.warning("Please provide a scan name and link.")

    thinkorswim_scans = load_data(THINKORSWIM_SCANS_FILE)
    for idx, scan in enumerate(thinkorswim_scans):
        with st.expander(f"ThinkorSwim Scan: {scan[0]}"):
            st.write(f"**Link:** [Click here]({scan[1]})")
            st.write(f"**Description:** {scan[2]}")
            if st.button(f"Delete {scan[0]}", key=f"delete_thinkorswim_scan_{idx}"):
                thinkorswim_scans.pop(idx)
                save_data(THINKORSWIM_SCANS_FILE, thinkorswim_scans)
                st.experimental_rerun()

    # Section for TradingView Scripts
    st.header("TradingView Scripts")
    st.markdown("""
    TradingView scripts are written in Pine Script and are used for creating custom indicators and strategies.
    """)
    with st.expander("Add a New TradingView Script"):
        tv_script_name = st.text_input("Script Name", key="tradingview_script_name")
        tv_script_code = st.text_area("Script Code", key="tradingview_script_code")
        tv_script_description = st.text_area("Description", key="tradingview_script_description")
        if st.button("Save TradingView Script"):
            if tv_script_name and tv_script_code:
                data = load_data(TRADINGVIEW_SCRIPTS_FILE)
                data.append([tv_script_name, tv_script_code, tv_script_description])
                save_data(TRADINGVIEW_SCRIPTS_FILE, data)
                st.success("TradingView Script saved!")
            else:
                st.warning("Please provide a script name and code.")

    tradingview_scripts = load_data(TRADINGVIEW_SCRIPTS_FILE)
    for idx, script in enumerate(tradingview_scripts):
        with st.expander(f"TradingView Script: {script[0]}"):
            st.code(script[1], language="javascript")
            st.write(f"**Description:** {script[2]}")
            if st.button(f"Delete {script[0]}", key=f"delete_tradingview_script_{idx}"):
                tradingview_scripts.pop(idx)
                save_data(TRADINGVIEW_SCRIPTS_FILE, tradingview_scripts)
                st.experimental_rerun()

    # Section for Session Recordings
    st.header("Session Recordings")
    st.markdown("""
    Session recordings are videos or tutorials that provide educational content on trading strategies, tools, and techniques.
    """)
    with st.expander("Add a New Session Recording"):
        recording_name = st.text_input("Recording Name", key="session_recording_name")
        recording_link = st.text_input("Recording Link", key="session_recording_link")
        recording_description = st.text_area("Description", key="session_recording_description")
        if st.button("Save Session Recording"):
            if recording_name and recording_link:
                data = load_data(SESSION_RECORDINGS_FILE)
                data.append([recording_name, recording_link, recording_description])
                save_data(SESSION_RECORDINGS_FILE, data)
                st.success("Session Recording saved!")
            else:
                st.warning("Please provide a recording name and link.")

    session_recordings = load_data(SESSION_RECORDINGS_FILE)
    for idx, recording in enumerate(session_recordings):
        with st.expander(f"Session Recording: {recording[0]}"):
            st.write(f"**Link:** [Click here]({recording[1]})")
            st.write(f"**Description:** {recording[2]}")
            if st.button(f"Delete {recording[0]}", key=f"delete_session_recording_{idx}"):
                session_recordings.pop(idx)
                save_data(SESSION_RECORDINGS_FILE, session_recordings)
                st.experimental_rerun()

# Run the module
if __name__ == "__main__":
    display_technical_scripts_education()
