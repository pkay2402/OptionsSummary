import streamlit as st
import imaplib
import email
import re
import datetime
import pandas as pd
from dateutil import parser
import sqlite3
from bs4 import BeautifulSoup
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fetch credentials from Streamlit Secrets
EMAIL_ADDRESS = st.secrets["EMAIL_ADDRESS"]
EMAIL_PASSWORD = st.secrets["EMAIL_PASSWORD"]

# Constants
SENDER_EMAIL = "alerts@thinkorswim.com"
OPTION_KEYWORDS = ["ETF_options", "UOP_Call", "call_swing", "put_swing"]
DEFAULT_DAYS_LOOKBACK = 3

def init_database():
    """Initialize SQLite database for storing option signals"""
    conn = sqlite3.connect('option_signals.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS option_signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            raw_symbol TEXT NOT NULL,
            readable_symbol TEXT NOT NULL,
            date_received DATETIME NOT NULL,
            signal_type TEXT NOT NULL,
            keyword TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(raw_symbol, signal_type, date_received, keyword)
        )
    ''')
    
    conn.commit()
    conn.close()

def connect_to_email():
    """Establish email connection"""
    try:
        mail = imaplib.IMAP4_SSL('imap.gmail.com')
        mail.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        return mail
    except Exception as e:
        logger.error(f"Email connection failed: {e}")
        raise

def parse_email_body(msg):
    """Parse email body with HTML handling"""
    try:
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() in ["text/plain", "text/html"]:
                    body = part.get_payload(decode=True).decode()
                    if part.get_content_type() == "text/html":
                        soup = BeautifulSoup(body, "html.parser", from_encoding='utf-8')
                        return soup.get_text(separator=' ', strip=True)
                    return body
        else:
            body = msg.get_payload(decode=True).decode()
            if msg.get_content_type() == "text/html":
                soup = BeautifulSoup(body, "html.parser", from_encoding='utf-8')
                return soup.get_text(separator=' ', strip=True)
            return body
    except Exception as e:
        logger.error(f"Error parsing email body: {e}")
        return ""

def parse_option_symbol(option_symbol):
    """Parse option symbol into readable format"""
    try:
        symbol = option_symbol.lstrip('.')
        pattern = r'([A-Z]+)(\d{2})(\d{2})(\d{2})([CP])([\d_]+)'
        match = re.match(pattern, symbol)
        
        if match:
            ticker, year, month, day, opt_type, strike = match.groups()
            month_name = datetime.datetime.strptime(month, '%m').strftime('%b').upper()
            strike = strike.replace('_', '.')
            option_type = 'CALL' if opt_type == 'C' else 'PUT'
            return f"{ticker} {day} {month_name} 20{year} {strike} {option_type}"
    
    except Exception as e:
        logger.error(f"Error parsing option symbol {option_symbol}: {e}")

    return option_symbol

def store_option_signals_in_db(option_data):
    """Store option signals in database"""
    if not option_data:
        return 0
    
    conn = sqlite3.connect('option_signals.db')
    cursor = conn.cursor()
    
    inserted_count = 0
    for raw_symbol, readable_symbol, date_received, signal_type, keyword in option_data:
        try:
            cursor.execute('''
                INSERT OR IGNORE INTO option_signals 
                (raw_symbol, readable_symbol, date_received, signal_type, keyword)
                VALUES (?, ?, ?, ?, ?)
            ''', (raw_symbol, readable_symbol, date_received, signal_type, keyword))
            
            if cursor.rowcount > 0:
                inserted_count += 1
                
        except Exception as e:
            logger.error(f"Error inserting record: {e}")
    
    conn.commit()
    conn.close()
    return inserted_count

def extract_option_symbols_from_email(keyword, days_lookback=DEFAULT_DAYS_LOOKBACK):
    """Extract option symbols from email alerts and store in database"""
    try:
        mail = connect_to_email()
        mail.select('inbox')

        # Calculate date range
        today = datetime.date.today()
        start_date = today - datetime.timedelta(days=days_lookback-1)
        date_since = start_date.strftime("%d-%b-%Y")
        
        search_criteria = f'(FROM "{SENDER_EMAIL}" SUBJECT "{keyword}" SINCE "{date_since}")'
        _, data = mail.search(None, search_criteria)

        option_data = []
        
        for num in data[0].split():
            _, email_data = mail.fetch(num, '(RFC822)')
            msg = email.message_from_bytes(email_data[0][1])
            
            email_datetime = parser.parse(msg['Date'])
            email_date = email_datetime.date()
            
            # Skip weekends and emails outside date range
            if email_date < start_date or email_datetime.weekday() >= 5:
                continue

            body = parse_email_body(msg)
            symbols = re.findall(r'New symbols:\s*([\.\w,\s]+)\s*were added to\s*(' + re.escape(keyword) + ')', body)
            
            if symbols:
                for symbol_group in symbols:
                    extracted_symbols = symbol_group[0].replace(" ", "").split(",")
                    signal_type = symbol_group[1]
                    for symbol in extracted_symbols:
                        if symbol:
                            readable_symbol = parse_option_symbol(symbol)
                            option_data.append([symbol, readable_symbol, email_datetime, signal_type, keyword])

        mail.close()
        mail.logout()

        # Store in database
        inserted_count = store_option_signals_in_db(option_data)
        logger.info(f"Inserted {inserted_count} new records for {keyword}")
        
        return len(option_data), inserted_count

    except Exception as e:
        logger.error(f"Error extracting option symbols for {keyword}: {e}")
        return 0, 0

def get_option_signals_from_db(keyword=None, days_back=None):
    """Retrieve option signals from database"""
    conn = sqlite3.connect('option_signals.db')
    
    query = "SELECT * FROM option_signals"
    params = []
    
    where_conditions = []
    
    if keyword:
        where_conditions.append("keyword = ?")
        params.append(keyword)
    
    if days_back:
        cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days_back)
        where_conditions.append("date_received >= ?")
        params.append(cutoff_date)
    
    if where_conditions:
        query += " WHERE " + " AND ".join(where_conditions)
    
    query += " ORDER BY date_received DESC"
    
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    
    return df

def main():
    """Main Streamlit application"""
    st.title("📈 Option Signals Email Extractor")
    st.markdown("---")
    
    # Initialize database
    init_database()
    
    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Settings")
        days_lookback = st.slider("Days to Extract", 1, 7, DEFAULT_DAYS_LOOKBACK)
        
        st.markdown("---")
        st.subheader("📧 Extract New Signals")
        
        selected_keywords = st.multiselect(
            "Select Keywords to Extract",
            OPTION_KEYWORDS,
            default=OPTION_KEYWORDS
        )
        
        if st.button("🔄 Extract Emails", type="primary"):
            if selected_keywords:
                with st.spinner("Extracting emails..."):
                    progress_bar = st.progress(0)
                    total_extracted = 0
                    total_inserted = 0
                    
                    for i, keyword in enumerate(selected_keywords):
                        extracted, inserted = extract_option_symbols_from_email(keyword, days_lookback)
                        total_extracted += extracted
                        total_inserted += inserted
                        progress_bar.progress((i + 1) / len(selected_keywords))
                    
                    st.success(f"✅ Extraction complete!\n\n"
                             f"📊 Total signals found: {total_extracted}\n"
                             f"💾 New signals stored: {total_inserted}")
            else:
                st.warning("Please select at least one keyword to extract.")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📊 View Stored Signals")
        view_keyword = st.selectbox("Filter by Keyword", ["All"] + OPTION_KEYWORDS)
        view_days = st.slider("Show Last N Days", 1, 30, 7)
    
    with col2:
        st.subheader("📈 Database Stats")
        # Get database statistics
        conn = sqlite3.connect('option_signals.db')
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM option_signals")
        total_records = cursor.fetchone()[0]
        
        cursor.execute("SELECT keyword, COUNT(*) FROM option_signals GROUP BY keyword")
        keyword_counts = cursor.fetchall()
        
        conn.close()
        
        st.metric("Total Records", total_records)
        
        if keyword_counts:
            st.write("**Records by Keyword:**")
            for keyword, count in keyword_counts:
                st.write(f"• {keyword}: {count}")
    
    # Display signals
    st.markdown("---")
    st.subheader("📋 Option Signals")
    
    # Get data from database
    filter_keyword = None if view_keyword == "All" else view_keyword
    df = get_option_signals_from_db(filter_keyword, view_days)
    
    if not df.empty:
        # Format the dataframe for display
        display_df = df.copy()
        display_df['date_received'] = pd.to_datetime(display_df['date_received']).dt.strftime('%Y-%m-%d %H:%M')
        display_df = display_df[['readable_symbol', 'signal_type', 'keyword', 'date_received']]
        display_df.columns = ['Option Symbol', 'Signal Type', 'Keyword', 'Date Received']
        
        st.dataframe(display_df, use_container_width=True)
        
        # Download button
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Data as CSV",
            data=csv,
            file_name=f"option_signals_{datetime.date.today()}.csv",
            mime="text/csv"
        )
        
        # Summary statistics
        st.subheader("📊 Summary")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Signals", len(df))
        with col2:
            unique_symbols = df['readable_symbol'].nunique()
            st.metric("Unique Symbols", unique_symbols)
        with col3:
            latest_signal = pd.to_datetime(df['date_received']).max()
            if pd.notna(latest_signal):
                st.metric("Latest Signal", latest_signal.strftime('%Y-%m-%d'))
    else:
        st.info("No option signals found with the current filters.")
    
    # Footer
    st.markdown("---")
    st.markdown(f"*Last updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")

if __name__ == "__main__":
    main()
