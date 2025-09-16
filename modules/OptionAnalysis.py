import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta
import os
import sqlite3
import hashlib
import yfinance as yf
import time

try:
    import pytz
    TIMEZONE_AVAILABLE = True
except ImportError:
    TIMEZONE_AVAILABLE = False
    st.warning("⚠️ pytz not installed. Install with: pip install pytz for proper timezone handling")

def get_technical_indicators(symbol):
    """Get current price and key technical indicators for a symbol"""
    try:
        # Get stock data for the last 252 trading days (1 year) to calculate 200 SMA
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="1y")
        
        if hist.empty:
            return "No data"
        
        # Get current close price
        current_close = hist['Close'].iloc[-1]
        
        # Calculate moving averages
        ema_8 = hist['Close'].ewm(span=8).mean().iloc[-1]
        ema_21 = hist['Close'].ewm(span=21).mean().iloc[-1]
        sma_50 = hist['Close'].rolling(window=50).mean().iloc[-1]
        sma_200 = hist['Close'].rolling(window=200).mean().iloc[-1]
        
        # Format the output
        return f"${current_close:.2f} | 8E: {ema_8:.2f} | 21E: {ema_21:.2f} | 50S: {sma_50:.2f} | 200S: {sma_200:.2f}"
        
    except Exception as e:
        return "Error fetching data"

# Top stocks to track in database
TOP_STOCKS = list(dict.fromkeys([
    # Mega-cap tech & growth
    'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN','GEV','APP', 'NVDA', 'TSLA', 'META', 'ORCL', 'CRM', 'ADBE', 'INTC', 'QCOM', 'TXN', 'AVGO', 'AMD', 'SNOW', 'NOW', 'SHOP', 'PLTR', 'UBER', 'PANW', 'DDOG', 'MDB', 'MRVL', 'ASML', 'VRT', 'APP', 'TEM', 'MSTR', 'RDDT', 'HOOD', 'COIN', 'NBIS', 'SMCI', 'ARM',
    # Financials
    'JPM', 'GS', 'BAC', 'MS', 'WFC', 'C', 'PYPL', 'V', 'MA', 'AXP', 'SCHW', 'BLK',
    # Healthcare & Pharma
    'UNH', 'JNJ', 'LLY', 'ABBV', 'PFE', 'MRNA', 'BNTX', 'GILD', 'BIIB', 'AMGN', 'REGN', 'VRTX', 'KVUE',
    # Consumer & Retail
    'HD', 'LOW', 'COST', 'WMT', 'TGT', 'NKE', 'SBUX', 'DIS', 'NFLX', 'WBD', 'PARA', 'ROKU', 'SPOT', 'ETSY', 'CAVA',
    # Energy & Industrials
    'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'OXY', 'DVN', 'MRO', 'GE', 'BA', 'CAT', 'DE', 'HON', 'MMM', 'LMT',
    # ETFs & Index proxies
    'SPY', 'QQQ', 'IWM', 'DIA', 'SMH','UVXY', 'XLK', 'XLF', 'XLV', 'XLE', 'XLI', 'XLY', 'XLP', 'XLU', 'XLB', 'XLRE', 'XLC', 'VIX', 'UVXY', 'VXX', 'SVXY', 'IBIT', 'ETHA',
    # Autos & EV
    'F', 'GM', 'RIVN', 'LCID', 'NIO', 'XPEV', 'LI', 'BYD', 'VFS',
    # China/Asia
    'BABA', 'JD', 'PDD', 'TCEHY', 'NTES', 'BIDU', 'NU',
    # Fintech & Trading
    'SOFI', 'DKNG', 'LYFT',
    # Social Media & Tech
    'SNAP', 'PINS', 'TTD', 'ABNB',
    # Cybersecurity & Cloud
    'CRWD', 'ZS',
    # Energy & Clean Tech
    'ENPH', 'PLUG',
    # Crypto & Mining
    'MARA', 'RIOT',
    # Misc High Volume
    'XYZ'
]))

@st.cache_data(ttl=900)  # Cache for 15 minutes
def get_stock_technicals(symbol):
    """Get current price and key moving averages for a stock"""
    try:
        ticker = yf.Ticker(symbol)
        # Get last 200 days of data to calculate moving averages
        hist = ticker.history(period="200d")
        
        if hist.empty:
            return None
        
        # Get latest close price
        current_price = hist['Close'].iloc[-1]
        
        # Calculate moving averages
        ema_8 = hist['Close'].ewm(span=8).mean().iloc[-1]
        ema_21 = hist['Close'].ewm(span=21).mean().iloc[-1]
        sma_50 = hist['Close'].rolling(window=50).mean().iloc[-1]
        sma_200 = hist['Close'].rolling(window=200).mean().iloc[-1]
        
        # Format the output
        return {
            'price': f"${current_price:.2f}",
            'ema8': f"${ema_8:.2f}" if pd.notna(ema_8) else "N/A",
            'ema21': f"${ema_21:.2f}" if pd.notna(ema_21) else "N/A", 
            'sma50': f"${sma_50:.2f}" if pd.notna(sma_50) else "N/A",
            'sma200': f"${sma_200:.2f}" if pd.notna(sma_200) else "N/A",
            'formatted': f"${current_price:.2f} | 8EMA: ${ema_8:.2f} | 21EMA: ${ema_21:.2f} | 50SMA: ${sma_50:.2f} | 200SMA: ${sma_200:.2f}"
        }
    except Exception as e:
        return None

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_batch_technicals(symbols_list):
    """Fetch technical data for multiple symbols efficiently"""
    results = {}
    
    # Batch process symbols to avoid overwhelming the API
    batch_size = 10
    for i in range(0, len(symbols_list), batch_size):
        batch = symbols_list[i:i + batch_size]
        
        for symbol in batch:
            try:
                tech_data = get_stock_technicals(symbol)
                results[symbol] = tech_data['formatted'] if tech_data else 'Data unavailable'
            except Exception as e:
                results[symbol] = 'Error fetching data'
        
        # Small delay between batches to be respectful to the API
        if i + batch_size < len(symbols_list):
            time.sleep(0.1)
    
    return results

def init_flow_database():
    """Initialize SQLite database for flow storage"""
    conn = sqlite3.connect('flow_database.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS flows (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_date TEXT,
            order_type TEXT,
            symbol TEXT,
            strike TEXT,
            expiry TEXT,
            contracts INTEGER,
            premium REAL,
            implied_volatility REAL,
            data_hash TEXT UNIQUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_open INTEGER DEFAULT 1,
            days_to_expiry INTEGER,
            position_status TEXT DEFAULT 'Open'
        )
    ''')
    
    # Create indexes for better performance
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_symbol ON flows(symbol)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_trade_date ON flows(trade_date)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_order_type ON flows(order_type)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_expiry ON flows(expiry)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_is_open ON flows(is_open)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_position_status ON flows(position_status)')
    
    # Add columns to existing table if they don't exist
    try:
        cursor.execute('ALTER TABLE flows ADD COLUMN is_open INTEGER DEFAULT 1')
    except sqlite3.OperationalError:
        pass  # Column already exists
    
    try:
        cursor.execute('ALTER TABLE flows ADD COLUMN days_to_expiry INTEGER')
    except sqlite3.OperationalError:
        pass  # Column already exists
    
    try:
        cursor.execute('ALTER TABLE flows ADD COLUMN position_status TEXT DEFAULT "Open"')
    except sqlite3.OperationalError:
        pass  # Column already exists
    
    # Update existing rows that might have NULL values in new columns
    cursor.execute('''
        UPDATE flows 
        SET is_open = 1, 
            position_status = 'Open',
            days_to_expiry = CAST(julianday(expiry) - julianday('now') AS INTEGER)
        WHERE is_open IS NULL OR position_status IS NULL OR days_to_expiry IS NULL
    ''')
    
    conn.commit()
    conn.close()

def migrate_existing_data():
    """Migrate existing data to add position tracking"""
    conn = sqlite3.connect('flow_database.db')
    cursor = conn.cursor()
    
    # Check if we have any data
    cursor.execute('SELECT COUNT(*) FROM flows')
    total_rows = cursor.fetchone()[0]
    
    if total_rows > 0:
        # Update position status for all existing rows
        cursor.execute('''
            UPDATE flows 
            SET days_to_expiry = CAST(julianday(expiry) - julianday('now') AS INTEGER),
                position_status = CASE 
                    WHEN date(expiry) < date('now') THEN 'Expired'
                    WHEN CAST(julianday(expiry) - julianday('now') AS INTEGER) <= 0 THEN 'Expires Today'
                    WHEN CAST(julianday(expiry) - julianday('now') AS INTEGER) <= 7 THEN 'Expires This Week'
                    WHEN CAST(julianday(expiry) - julianday('now') AS INTEGER) <= 30 THEN 'Expires This Month'
                    ELSE 'Open'
                END,
                is_open = CASE 
                    WHEN date(expiry) < date('now') THEN 0
                    ELSE 1
                END
            WHERE 1=1
        ''')
        
        conn.commit()
    
    conn.close()
    return total_rows

def get_position_analysis():
    """Analyze position matching and provide detailed position tracking"""
    conn = sqlite3.connect('flow_database.db')
    
    # Get detailed position analysis
    query = '''
        SELECT 
            symbol,
            strike,
            expiry,
            SUM(CASE WHEN order_type IN ('Calls Bought', 'Puts Bought') THEN contracts ELSE 0 END) as total_bought,
            SUM(CASE WHEN order_type IN ('Calls Sold', 'Puts Sold') THEN contracts ELSE 0 END) as total_sold,
            SUM(CASE WHEN order_type = 'Calls Bought' THEN contracts ELSE 0 END) as calls_bought,
            SUM(CASE WHEN order_type = 'Calls Sold' THEN contracts ELSE 0 END) as calls_sold,
            SUM(CASE WHEN order_type = 'Puts Bought' THEN contracts ELSE 0 END) as puts_bought,
            SUM(CASE WHEN order_type = 'Puts Sold' THEN contracts ELSE 0 END) as puts_sold,
            COUNT(*) as trade_count,
            MIN(trade_date) as first_trade,
            MAX(trade_date) as last_trade,
            SUM(premium) as total_premium,
            days_to_expiry,
            position_status
        FROM flows 
        GROUP BY symbol, strike, expiry
        HAVING trade_count > 0
        ORDER BY symbol, expiry, strike
    '''
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if df.empty:
        return df
    
    # Calculate net positions and position types
    df['net_contracts'] = df['total_bought'] - df['total_sold']
    df['net_calls'] = df['calls_bought'] - df['calls_sold']
    df['net_puts'] = df['puts_bought'] - df['puts_sold']
    
    # Determine position type
    def determine_position_type(row):
        if row['trade_count'] == 1:
            return 'Single Trade'
        elif row['net_contracts'] == 0:
            return 'Fully Offset'
        elif abs(row['net_contracts']) < min(row['total_bought'], row['total_sold']):
            return 'Partially Offset'
        elif row['net_contracts'] > 0:
            return 'Net Long'
        else:
            return 'Net Short'
    
    df['position_type'] = df.apply(determine_position_type, axis=1)
    
    # Calculate offset percentage
    df['offset_percentage'] = df.apply(
        lambda row: min(row['total_bought'], row['total_sold']) / max(row['total_bought'], row['total_sold']) * 100 
        if row['trade_count'] > 1 and max(row['total_bought'], row['total_sold']) > 0 else 0, 
        axis=1
    )
    
    return df

def smart_flow_interpretation(df):
    """
    Smart interpretation of flow direction using multiple factors:
    - Side code patterns
    - Implied volatility levels
    - Block size clustering
    - Time proximity
    - Premium levels
    """
    if df is None or df.empty:
        return df
    
    df = df.copy()
    df['smart_order_type'] = df.apply(lambda row: determine_smart_order_type(row, df), axis=1)
    return df

def determine_smart_order_type(row, full_df):
    """
    Determine the likely true order type using contextual analysis
    """
    symbol = row['Ticker']
    strike = row['Strike Price']
    contract_type = row['Contract Type']
    side_code = row['Side Code']
    trade_time = row['Trade Time']
    size = row['Size']
    iv = row.get('Implied Volatility', 0)
    premium = row.get('Premium Price', 0)
    
    # Default interpretation based on side code
    if contract_type == 'CALL':
        default_order = 'Calls Bought' if side_code in ['A', 'AA'] else 'Calls Sold'
    else:
        default_order = 'Puts Bought' if side_code in ['A', 'AA'] else 'Puts Sold'
    
    # Look for clustering patterns (same strike, similar times, large sizes)
    if pd.notna(trade_time):
        time_window = pd.Timedelta(minutes=30)
        similar_flows = full_df[
            (full_df['Ticker'] == symbol) &
            (full_df['Strike Price'] == strike) &
            (full_df['Contract Type'] == contract_type) &
            (full_df['Size'] >= 1000) &  # Large institutional flows
            (abs(full_df['Trade Time'] - trade_time) <= time_window)
        ]
        
        if len(similar_flows) >= 3:  # Multiple large flows at same strike
            total_volume = similar_flows['Size'].sum()
            avg_iv = similar_flows['Implied Volatility'].mean()
            
            # High volume clustering with high IV suggests institutional positioning
            if total_volume >= 5000 and avg_iv > 50:
                # Look at the flow pattern
                side_b_volume = similar_flows[similar_flows['Side Code'] == 'B']['Size'].sum()
                side_a_volume = similar_flows[similar_flows['Side Code'] == 'A']['Size'].sum()
                
                # If predominantly one side, it's likely institutional positioning
                if side_b_volume > side_a_volume * 2:  # Mostly "B" (bid hits)
                    # Large bid hits often indicate institutional buying despite being marked as "sold"
                    if contract_type == 'CALL':
                        return 'Calls Bought'  # Override: likely institutional call buying
                    else:
                        return 'Puts Bought'   # Override: likely institutional put buying
                        
                elif side_a_volume > side_b_volume * 2:  # Mostly "A" (ask lifts)
                    # Ask lifts confirm buying pressure
                    if contract_type == 'CALL':
                        return 'Calls Bought'
                    else:
                        return 'Puts Bought'
    
    # High IV with large size suggests aggressive positioning
    if iv > 75 and size >= 2000:
        if contract_type == 'CALL' and side_code == 'B':
            return 'Calls Bought'  # Override: likely aggressive call buying despite bid hit
        elif contract_type == 'PUT' and side_code == 'B':
            return 'Puts Bought'   # Override: likely aggressive put buying despite bid hit
    
    # Default to original interpretation
    return default_order

def store_flows_in_database(df):
    """Store flows for top 30 stocks in database with enhanced criteria"""
    if df is None or df.empty:
        return 0
    
    # Filter for top 30 stocks only
    df_filtered = df[df['Ticker'].isin(TOP_STOCKS)].copy()
    
    # Enhanced filtering criteria:
    # 1. Original: 900+ contracts AND $200K+ premium, OR
    # 2. New: $1M+ premium regardless of contract size
    criteria_1 = (df_filtered['Size'] >= 900) & (df_filtered['Premium Price'] >= 200000)
    criteria_2 = df_filtered['Premium Price'] >= 900000
    
    df_filtered = df_filtered[criteria_1 | criteria_2].copy()
    
    if df_filtered.empty:
        return 0
    
    # Apply smart flow interpretation
    df_filtered = smart_flow_interpretation(df_filtered)
    
    conn = sqlite3.connect('flow_database.db')
    cursor = conn.cursor()
    
    stored_count = 0
    
    for _, row in df_filtered.iterrows():
        try:
            # Create data hash to prevent duplicates
            data_string = f"{row.get('Ticker', '')}{row.get('Strike Price', '')}{row.get('Contract Type', '')}{row.get('Expiration Date', '')}{row.get('Size', '')}"
            data_hash = hashlib.md5(data_string.encode()).hexdigest()
            
            # Determine order type based on contract type and side code
            contract_type = str(row.get('Contract Type', '')).upper()
            side_code = str(row.get('Side Code', '')).upper()
            
            # Use smart interpretation (already applied to df_filtered)
            # The order_type field now contains the smart interpretation result
            order_type = row.get('order_type', '')
            if not order_type:
                # Fallback to simple interpretation if smart interpretation failed
                if contract_type == 'CALL':
                    order_type = 'Calls Bought' if side_code in ['A', 'AA'] else 'Calls Sold'
                elif contract_type == 'PUT':
                    order_type = 'Puts Bought' if side_code in ['A', 'AA'] else 'Puts Sold'
                else:
                    continue  # Skip unknown contract types
            
            # Format strike price with proper decimal places
            strike_price = row.get('Strike Price', 0)
            # Show one decimal place if needed, otherwise show whole number
            if strike_price % 1 == 0:
                strike = f"{strike_price:.0f}{'C' if contract_type == 'CALL' else 'P'}"
            else:
                strike = f"{strike_price:.1f}{'C' if contract_type == 'CALL' else 'P'}"
            
            # Handle expiration date
            exp_date = row.get('Expiration Date')
            if pd.isna(exp_date):
                continue
            
            if isinstance(exp_date, str):
                try:
                    exp_date = pd.to_datetime(exp_date)
                except:
                    continue
            
            expiry = exp_date.strftime('%Y-%m-%d')
            
            # Get trade date from CSV Trade Time and convert to EST/EDT
            trade_time = row.get('Trade Time')
            if pd.isna(trade_time) or trade_time is None:
                continue  # Skip rows without trade time
            
            if isinstance(trade_time, str):
                try:
                    trade_time = pd.to_datetime(trade_time)
                except:
                    continue  # Skip rows with invalid trade time
            
            if isinstance(trade_time, pd.Timestamp):
                if TIMEZONE_AVAILABLE:
                    # Convert to Eastern Time (market timezone)
                    eastern = pytz.timezone('US/Eastern')
                    
                    # Convert to Eastern time (handles EST/EDT automatically)
                    if trade_time.tz is not None:
                        # Already timezone-aware, convert to Eastern
                        eastern_time = trade_time.astimezone(eastern)
                    else:
                        # Timezone-naive, assume UTC
                        utc_time = pytz.utc.localize(trade_time)
                        eastern_time = utc_time.astimezone(eastern)
                    
                    # Format with proper timezone abbreviation
                    trade_date = eastern_time.strftime('%Y-%m-%d %H:%M %Z')
                else:
                    # Fallback without timezone conversion
                    trade_date = trade_time.strftime('%Y-%m-%d %H:%M UTC')
            else:
                continue  # Skip rows with invalid trade time format
            
            # Calculate premium
            premium = float(row.get('Premium Price', 0))
            
            cursor.execute('''
                INSERT OR IGNORE INTO flows 
                (trade_date, order_type, symbol, strike, expiry, contracts, premium, implied_volatility, data_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade_date,
                order_type,
                row.get('Ticker'),
                strike,
                expiry,
                int(row.get('Size', 0)),
                premium,
                float(row.get('IV', 0)),
                data_hash
            ))
            
            if cursor.rowcount > 0:
                stored_count += 1
                
        except Exception as e:
            continue  # Skip problematic rows
    
    conn.commit()
    conn.close()
    
    return stored_count

def get_flows_from_database(symbol_filter=None, order_type_filter=None, date_from=None, date_to=None, include_technicals=True, limit=None):
    """Retrieve flows from database with filters and optional technical indicators"""
    conn = sqlite3.connect('flow_database.db')
    
    query = "SELECT trade_date, order_type, symbol, strike, expiry, contracts, premium FROM flows WHERE 1=1"
    params = []
    
    if symbol_filter and symbol_filter != "All":
        query += " AND symbol = ?"
        params.append(symbol_filter)
    
    if order_type_filter and order_type_filter != "All":
        query += " AND order_type = ?"
        params.append(order_type_filter)
    
    if date_from:
        query += " AND trade_date >= ?"
        if hasattr(date_from, 'strftime'):
            params.append(date_from.strftime('%Y-%m-%d'))
        else:
            params.append(str(date_from))
    
    if date_to:
        query += " AND trade_date <= ?"
        if hasattr(date_to, 'strftime'):
            params.append((date_to + timedelta(days=1)).strftime('%Y-%m-%d'))
        else:
            # If it's already a string, just use it as is
            params.append(str(date_to))
    
    query += " ORDER BY trade_date DESC"
    
    # Add limit for performance
    if limit:
        query += f" LIMIT {limit}"
    
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    
    # Add technical indicators column only if requested and dataframe is not too large
    if include_technicals and not df.empty and len(df) <= 1000:  # Limit technical data for large datasets
        # Get unique symbols to minimize API calls
        unique_symbols = df['symbol'].unique()
        
        # Only fetch technicals if we have a reasonable number of symbols
        if len(unique_symbols) <= 50:  # Limit to 50 symbols max
            # Use batch processing for better performance
            with st.spinner(f"Fetching technical data for {len(unique_symbols)} symbols..."):
                technicals_dict = get_batch_technicals(list(unique_symbols))
            
            # Add technical data to dataframe
            df['Technicals'] = df['symbol'].apply(
                lambda x: technicals_dict.get(x, 'Data unavailable')
            )
        else:
            # Too many symbols, skip technicals to avoid performance issues
            df['Technicals'] = 'Too many symbols - technicals disabled'
    elif include_technicals and len(df) > 1000:
        # Dataset too large, skip technicals
        df['Technicals'] = 'Large dataset - technicals disabled'
    
    return df

def get_todays_flows_summary():
    """Get consolidated summary of today's flows from database"""
    conn = sqlite3.connect('flow_database.db')
    
    today = datetime.now().strftime('%Y-%m-%d')
    
    # Get today's flows with detailed information
    query = """
        SELECT symbol, order_type, strike, contracts, COUNT(*) as flow_count, 
               SUM(contracts) as total_contracts, AVG(contracts) as avg_contracts, 
               MIN(trade_date) as first_flow, MAX(trade_date) as last_flow
        FROM flows 
        WHERE substr(trade_date, 1, 10) = ?
        GROUP BY symbol, order_type, strike, contracts
        ORDER BY total_contracts DESC
    """
    
    detailed_flows = pd.read_sql_query(query, conn, params=[today])
    
    # Get summary by symbol and order type
    summary_query = """
        SELECT symbol, order_type, COUNT(*) as flow_count, SUM(contracts) as total_contracts, 
               AVG(contracts) as avg_contracts, MIN(contracts) as min_contracts, 
               MAX(contracts) as max_contracts, MIN(trade_date) as first_flow, MAX(trade_date) as last_flow
        FROM flows 
        WHERE substr(trade_date, 1, 10) = ?
        GROUP BY symbol, order_type
        ORDER BY total_contracts DESC
    """
    
    summary_df = pd.read_sql_query(summary_query, conn, params=[today])
    
    # Get overall stats for today
    overall_query = """
        SELECT 
            COUNT(*) as total_flows,
            COUNT(DISTINCT symbol) as unique_symbols,
            SUM(contracts) as total_contracts,
            SUM(CASE WHEN order_type LIKE '%Calls%' THEN contracts ELSE 0 END) as call_contracts,
            SUM(CASE WHEN order_type LIKE '%Puts%' THEN contracts ELSE 0 END) as put_contracts,
            SUM(CASE WHEN order_type LIKE '%Bought%' THEN contracts ELSE 0 END) as bought_contracts,
            SUM(CASE WHEN order_type LIKE '%Sold%' THEN contracts ELSE 0 END) as sold_contracts,
            MIN(contracts) as min_contract_size,
            MAX(contracts) as max_contract_size,
            AVG(contracts) as avg_contract_size
        FROM flows 
        WHERE substr(trade_date, 1, 10) = ?
    """
    
    overall_stats = pd.read_sql_query(overall_query, conn, params=[today])
    
    conn.close()
    
    return summary_df, overall_stats, detailed_flows

def format_todays_summary(summary_df, overall_stats, detailed_flows):
    """Format today's flow summary into readable text with notable strikes and contract sizes"""
    if summary_df.empty:
        return "📭 No flows stored for today yet."
    
    stats = overall_stats.iloc[0]
    
    summary_text = f"""
📊 **TODAY'S FLOW SUMMARY** - {datetime.now().strftime('%Y-%m-%d')}

**🎯 OVERALL STATISTICS:**
• Total Flows: {stats['total_flows']}
• Unique Symbols: {stats['unique_symbols']}
• Total Contracts: {stats['total_contracts']:,}
• Call Contracts: {stats['call_contracts']:,} ({stats['call_contracts']/stats['total_contracts']*100:.1f}%)
• Put Contracts: {stats['put_contracts']:,} ({stats['put_contracts']/stats['total_contracts']*100:.1f}%)
• Bought vs Sold: {stats['bought_contracts']:,} bought | {stats['sold_contracts']:,} sold
• Contract Size Range: {stats['min_contract_size']:,} - {stats['max_contract_size']:,} (Avg: {stats['avg_contract_size']:,.0f})

**📈 TOP ACTIVITY BY SYMBOL:**
"""
    
    # Group by symbol for cleaner display
    symbol_summary = summary_df.groupby('symbol').agg({
        'flow_count': 'sum',
        'total_contracts': 'sum',
        'avg_contracts': 'mean',
        'min_contracts': 'min',
        'max_contracts': 'max'
    }).sort_values('total_contracts', ascending=False).head(10)
    
    for symbol, row in symbol_summary.iterrows():
        symbol_flows = summary_df[summary_df['symbol'] == symbol]
        
        # Get breakdown by order type
        call_flows = symbol_flows[symbol_flows['order_type'].str.contains('Calls')]
        put_flows = symbol_flows[symbol_flows['order_type'].str.contains('Puts')]
        
        call_contracts = call_flows['total_contracts'].sum() if not call_flows.empty else 0
        put_contracts = put_flows['total_contracts'].sum() if not put_flows.empty else 0
        
        bias = "📈 CALL HEAVY" if call_contracts > put_contracts * 1.5 else "📉 PUT HEAVY" if put_contracts > call_contracts * 1.5 else "⚖️ MIXED"
        
        # Get notable strikes and contract sizes for this symbol
        symbol_detailed = detailed_flows[detailed_flows['symbol'] == symbol].sort_values('total_contracts', ascending=False)
        
        # Find most notable strikes (top 2)
        notable_strikes = []
        if not symbol_detailed.empty:
            top_flows = symbol_detailed.head(2)
            for _, flow in top_flows.iterrows():
                strike_clean = flow['strike'].replace('C', '').replace('P', '')
                contract_type = '📞' if 'C' in flow['strike'] else '📻'
                order_action = '🟢BUY' if 'Bought' in flow['order_type'] else '🔴SELL'
                notable_strikes.append(f"{contract_type}{strike_clean} ({order_action} {flow['contracts']:,})")
        
        # Get contract size range for this symbol
        min_size = int(row['min_contracts'])
        max_size = int(row['max_contracts'])
        size_range = f"{min_size:,}" if min_size == max_size else f"{min_size:,}-{max_size:,}"
        
        summary_text += f"""
**{symbol}** {bias}
  └─ {int(row['flow_count'])} flows | {int(row['total_contracts']):,} total contracts | Size range: {size_range}
  └─ Calls: {call_contracts:,} | Puts: {put_contracts:,}"""
        
        # Add notable strikes if available
        if notable_strikes:
            strikes_text = " | ".join(notable_strikes[:2])  # Limit to top 2 to keep clean
            summary_text += f"""
  └─ 🎯 Notable: {strikes_text}"""
    
    return summary_text

def generate_symbol_interpretation(flows_df, symbol):
    """Generate interpretation summary for a specific symbol's flows"""
    if flows_df.empty:
        return f"📭 No flows found for {symbol} in the selected time period."
    
    symbol_flows = flows_df[flows_df['symbol'] == symbol]
    if symbol_flows.empty:
        return f"📭 No flows found for {symbol} in the selected time period."
    
    # Analyze the flows
    total_flows = len(symbol_flows)
    call_flows = symbol_flows[symbol_flows['order_type'].str.contains('Calls')]
    put_flows = symbol_flows[symbol_flows['order_type'].str.contains('Puts')]
    bought_flows = symbol_flows[symbol_flows['order_type'].str.contains('Bought')]
    sold_flows = symbol_flows[symbol_flows['order_type'].str.contains('Sold')]
    
    total_contracts = symbol_flows['contracts'].sum()
    call_contracts = call_flows['contracts'].sum() if not call_flows.empty else 0
    put_contracts = put_flows['contracts'].sum() if not put_flows.empty else 0
    bought_contracts = bought_flows['contracts'].sum() if not bought_flows.empty else 0
    sold_contracts = sold_flows['contracts'].sum() if not sold_flows.empty else 0
    
    # Determine bias and sentiment
    if call_contracts > put_contracts * 1.5:
        direction_bias = "📈 BULLISH BIAS"
        bias_explanation = f"Call activity dominates with {call_contracts:,} vs {put_contracts:,} put contracts"
    elif put_contracts > call_contracts * 1.5:
        direction_bias = "📉 BEARISH BIAS" 
        bias_explanation = f"Put activity dominates with {put_contracts:,} vs {call_contracts:,} call contracts"
    else:
        direction_bias = "⚖️ NEUTRAL/MIXED"
        bias_explanation = f"Balanced activity with {call_contracts:,} call and {put_contracts:,} put contracts"
    
    # Calculate detailed buying/selling by type for better sentiment analysis
    call_bought = symbol_flows[symbol_flows['order_type'] == 'Calls Bought']['contracts'].sum()
    call_sold = symbol_flows[symbol_flows['order_type'] == 'Calls Sold']['contracts'].sum()
    put_bought = symbol_flows[symbol_flows['order_type'] == 'Puts Bought']['contracts'].sum()
    put_sold = symbol_flows[symbol_flows['order_type'] == 'Puts Sold']['contracts'].sum()
    
    # Calculate true market sentiment based on bullish vs bearish actions
    bullish_activity = call_bought + put_sold  # Buying calls + Selling puts = Bullish
    bearish_activity = put_bought + call_sold  # Buying puts + Selling calls = Bearish
    
    total_activity = bullish_activity + bearish_activity
    if total_activity > 0:
        bullish_percentage = (bullish_activity / total_activity) * 100
        bearish_percentage = (bearish_activity / total_activity) * 100
        
        if bullish_percentage > 65:
            market_sentiment = "🟢 BULLISH SENTIMENT"
            sentiment_explanation = f"Strong bullish sentiment: {bullish_percentage:.1f}% bullish activity ({call_bought:,} calls bought + {put_sold:,} puts sold)"
        elif bearish_percentage > 65:
            market_sentiment = "🔴 BEARISH SENTIMENT"
            sentiment_explanation = f"Strong bearish sentiment: {bearish_percentage:.1f}% bearish activity ({put_bought:,} puts bought + {call_sold:,} calls sold)"
        elif bullish_percentage > bearish_percentage:
            market_sentiment = "🟢 MILDLY BULLISH"
            sentiment_explanation = f"Mildly bullish sentiment: {bullish_percentage:.1f}% vs {bearish_percentage:.1f}% bearish activity"
        elif bearish_percentage > bullish_percentage:
            market_sentiment = "🔴 MILDLY BEARISH"
            sentiment_explanation = f"Mildly bearish sentiment: {bearish_percentage:.1f}% vs {bullish_percentage:.1f}% bullish activity"
        else:
            market_sentiment = "🟡 NEUTRAL SENTIMENT"
            sentiment_explanation = f"Neutral sentiment: {bullish_percentage:.1f}% bullish vs {bearish_percentage:.1f}% bearish activity"
    else:
        market_sentiment = "� NO CLEAR SENTIMENT"
        sentiment_explanation = "No clear sentiment can be determined from available data"
    
    # Analyze pure buying vs selling pressure (separate from sentiment)
    if bought_contracts > sold_contracts * 1.3:
        flow_pressure = "⬆️ STRONG INFLOW"
        pressure_explanation = f"Heavy buying pressure with {bought_contracts:,} bought vs {sold_contracts:,} sold"
    elif sold_contracts > bought_contracts * 1.3:
        flow_pressure = "⬇️ STRONG OUTFLOW"
        pressure_explanation = f"Heavy selling pressure with {sold_contracts:,} sold vs {bought_contracts:,} bought"
    else:
        flow_pressure = "↔️ BALANCED FLOW"
        pressure_explanation = f"Balanced flow with {bought_contracts:,} bought and {sold_contracts:,} sold"
    
    # Get notable strikes
    strike_activity = symbol_flows.groupby('strike')['contracts'].sum().sort_values(ascending=False)
    top_strikes = strike_activity.head(3)
    
    # Get size distribution
    avg_size = symbol_flows['contracts'].mean()
    min_size = symbol_flows['contracts'].min()
    max_size = symbol_flows['contracts'].max()
    
    # Recent activity (latest flows)
    latest_flows = symbol_flows.sort_values('trade_date', ascending=False).head(3)
    
    interpretation = f"""
🎯 **{symbol} FLOW INTERPRETATION**

**📊 ACTIVITY OVERVIEW:**
• Total Flows: {total_flows}
• Total Contracts: {total_contracts:,}
• Size Range: {min_size:,} - {max_size:,} (Avg: {avg_size:,.0f})

**🎭 MARKET SENTIMENT:**
• {market_sentiment}
• {flow_pressure}

**📈 DETAILED ANALYSIS:**
• {bias_explanation}
• {sentiment_explanation}
• {pressure_explanation}

**📊 BREAKDOWN BY ORDER TYPE:**
• Calls Bought: {call_bought:,} contracts
• Calls Sold: {call_sold:,} contracts
• Puts Bought: {put_bought:,} contracts  
• Puts Sold: {put_sold:,} contracts

**🎯 MOST ACTIVE STRIKES:**"""
    
    for strike, contracts in top_strikes.items():
        strike_clean = strike.replace('C', '').replace('P', '')
        contract_type = "📞 Call" if 'C' in strike else "📻 Put"
        interpretation += f"\n• {contract_type} ${strike_clean}: {contracts:,} contracts"
    
    if not latest_flows.empty:
        interpretation += f"\n\n**⏰ RECENT ACTIVITY:**"
        for _, flow in latest_flows.iterrows():
            strike_clean = flow['strike'].replace('C', '').replace('P', '')
            contract_type = "📞" if 'C' in flow['strike'] else "📻"
            action_color = "🟢" if 'Bought' in flow['order_type'] else "🔴"
            interpretation += f"\n• {contract_type} ${strike_clean} - {action_color} {flow['order_type']} {flow['contracts']:,} contracts"
    
    return interpretation

def get_trending_bullish_stocks(days_back=1, min_flows=2):
    """Get trending bullish stocks based on latest trading day available in database"""
    conn = sqlite3.connect('flow_database.db')
    
    # First, find the latest trading day available in the database
    latest_day_query = "SELECT MAX(substr(trade_date, 1, 10)) as latest_day FROM flows"
    latest_day_result = pd.read_sql_query(latest_day_query, conn)
    
    if latest_day_result.empty or latest_day_result['latest_day'].iloc[0] is None:
        conn.close()
        return pd.DataFrame()
    
    latest_trading_day = latest_day_result['latest_day'].iloc[0]
    
    query = """
        SELECT 
            symbol,
            COUNT(*) as total_flows,
            SUM(contracts) as total_contracts,
            SUM(premium) as total_premium,
            SUM(CASE WHEN order_type = 'Calls Bought' THEN contracts ELSE 0 END) as calls_bought,
            SUM(CASE WHEN order_type = 'Calls Sold' THEN contracts ELSE 0 END) as calls_sold,
            SUM(CASE WHEN order_type = 'Puts Bought' THEN contracts ELSE 0 END) as puts_bought,
            SUM(CASE WHEN order_type = 'Puts Sold' THEN contracts ELSE 0 END) as puts_sold,
            MAX(trade_date) as latest_activity
        FROM flows 
        WHERE substr(trade_date, 1, 10) = ?
        GROUP BY symbol
        HAVING COUNT(*) >= ?
        ORDER BY symbol
    """
    
    df = pd.read_sql_query(query, conn, params=[latest_trading_day, min_flows])
    conn.close()
    
    if df.empty:
        return pd.DataFrame()
    
    # Calculate bullish sentiment for latest trading day only
    df['bullish_activity'] = df['calls_bought'] + df['puts_sold']
    df['bearish_activity'] = df['puts_bought'] + df['calls_sold']
    df['total_activity'] = df['bullish_activity'] + df['bearish_activity']
    
    # Filter out stocks with no activity
    df = df[df['total_activity'] > 0].copy()
    
    if df.empty:
        return pd.DataFrame()
    
    df['bullish_percentage'] = (df['bullish_activity'] / df['total_activity']) * 100
    df['net_bullish_contracts'] = df['bullish_activity'] - df['bearish_activity']
    
    # Simple scoring for latest day trending
    df['volume_weight'] = df['total_contracts'] / df['total_contracts'].max()
    df['premium_weight'] = df['total_premium'] / df['total_premium'].max()
    df['flow_frequency_weight'] = df['total_flows'] / df['total_flows'].max()
    df['bullish_strength'] = df['bullish_percentage'] / 100
    
    # Latest day bullish score
    df['bullish_score'] = (
        df['bullish_strength'] * 0.4 +     # 40% weight on bullish percentage
        df['volume_weight'] * 0.3 +        # 30% weight on contract volume
        df['premium_weight'] * 0.2 +       # 20% weight on premium volume
        df['flow_frequency_weight'] * 0.1  # 10% weight on flow frequency
    )
    
    # Filter for meaningful bullish activity today
    bullish_df = df[
        (df['bullish_percentage'] >= 60) &     # Must be at least 60% bullish today
        (df['net_bullish_contracts'] > 0) &    # Must have net bullish contracts
        (df['total_contracts'] >= 1000)        # Minimum volume threshold
    ].copy()
    
    bullish_df = bullish_df.sort_values('bullish_score', ascending=False)
    
    return bullish_df

def get_trending_bearish_stocks(days_back=1, min_flows=2):
    """Get trending bearish stocks based on latest trading day available in database"""
    conn = sqlite3.connect('flow_database.db')
    
    # First, find the latest trading day available in the database
    latest_day_query = "SELECT MAX(substr(trade_date, 1, 10)) as latest_day FROM flows"
    latest_day_result = pd.read_sql_query(latest_day_query, conn)
    
    if latest_day_result.empty or latest_day_result['latest_day'].iloc[0] is None:
        conn.close()
        return pd.DataFrame()
    
    latest_trading_day = latest_day_result['latest_day'].iloc[0]
    
    query = """
        SELECT 
            symbol,
            COUNT(*) as total_flows,
            SUM(contracts) as total_contracts,
            SUM(premium) as total_premium,
            SUM(CASE WHEN order_type = 'Calls Bought' THEN contracts ELSE 0 END) as calls_bought,
            SUM(CASE WHEN order_type = 'Calls Sold' THEN contracts ELSE 0 END) as calls_sold,
            SUM(CASE WHEN order_type = 'Puts Bought' THEN contracts ELSE 0 END) as puts_bought,
            SUM(CASE WHEN order_type = 'Puts Sold' THEN contracts ELSE 0 END) as puts_sold,
            MAX(trade_date) as latest_activity
        FROM flows 
        WHERE substr(trade_date, 1, 10) = ?
        GROUP BY symbol
        HAVING COUNT(*) >= ?
        ORDER BY symbol
    """
    
    df = pd.read_sql_query(query, conn, params=[latest_trading_day, min_flows])
    conn.close()
    
    if df.empty:
        return pd.DataFrame()
    
    # Calculate bearish sentiment for latest trading day only
    df['bullish_activity'] = df['calls_bought'] + df['puts_sold']
    df['bearish_activity'] = df['puts_bought'] + df['calls_sold']
    df['total_activity'] = df['bullish_activity'] + df['bearish_activity']
    
    # Filter out stocks with no activity
    df = df[df['total_activity'] > 0].copy()
    
    if df.empty:
        return pd.DataFrame()
    
    df['bearish_percentage'] = (df['bearish_activity'] / df['total_activity']) * 100
    df['net_bearish_contracts'] = df['bearish_activity'] - df['bullish_activity']
    
    # Simple scoring for latest day trending
    df['volume_weight'] = df['total_contracts'] / df['total_contracts'].max()
    df['premium_weight'] = df['total_premium'] / df['total_premium'].max()
    df['flow_frequency_weight'] = df['total_flows'] / df['total_flows'].max()
    df['bearish_strength'] = df['bearish_percentage'] / 100
    
    # Latest day bearish score
    df['bearish_score'] = (
        df['bearish_strength'] * 0.4 +     # 40% weight on bearish percentage
        df['volume_weight'] * 0.3 +        # 30% weight on contract volume
        df['premium_weight'] * 0.2 +       # 20% weight on premium volume
        df['flow_frequency_weight'] * 0.1  # 10% weight on flow frequency
    )
    
    # Filter for meaningful bearish activity today
    bearish_df = df[
        (df['bearish_percentage'] >= 60) &     # Must be at least 60% bearish today
        (df['net_bearish_contracts'] > 0) &    # Must have net bearish contracts
        (df['total_contracts'] >= 1000)        # Minimum volume threshold
    ].copy()
    
    bearish_df = bearish_df.sort_values('bearish_score', ascending=False)
    
    return bearish_df

# Function to load and process CSV file
def load_csv(file):
    try:
        df = pd.read_csv(file)
        
        # Define expected columns
        required_columns = ['Trade Time', 'Ticker', 'Expiration Date', 'Contract Type', 'Strike Price', 
                           'Reference Price', 'Size', 'Premium Price', 'Side Code', 
                           'Is Opening Position', 'Money Type', 'Is Unusual', 'Is Golden Sweep', 'Implied Volatility']
        
        # Map columns by position if names don't match
        if not all(col in df.columns for col in required_columns):
            column_mapping = {
                1: 'Trade Time', 2: 'Ticker', 3: 'Expiration Date', 4: 'Days Until Expiration',
                5: 'Strike Price', 6: 'Contract Type', 7: 'Reference Price',
                8: 'Size', 9: 'Option Price', 12: 'Premium Price', 19: 'Side Code',
                15: 'Is Unusual', 16: 'Is Golden Sweep', 17: 'Is Opening Position',
                18: 'Money Type', 20: 'Implied Volatility'
            }
            df.columns = [column_mapping.get(i, col) for i, col in enumerate(df.columns)]
        
        # Convert and clean data
        df['Trade Time'] = pd.to_datetime(df['Trade Time'], errors='coerce', utc=True)
        df['Expiration Date'] = pd.to_datetime(df['Expiration Date'], errors='coerce')
        numeric_cols = ['Strike Price', 'Reference Price', 'Size', 'Premium Price', 'Implied Volatility']
        for col in numeric_cols:
            if col == 'Implied Volatility':
                # Handle IV percentage format (remove % if present)
                df[col] = pd.to_numeric(df[col].astype(str).str.replace('%', ''), errors='coerce')
            else:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace('[\$,]', '', regex=True), 
                                      errors='coerce')
        
        categorical_cols = ['Ticker', 'Contract Type', 'Side Code', 'Is Unusual', 
                          'Is Golden Sweep', 'Is Opening Position', 'Money Type']
        for col in categorical_cols:
            df[col] = df[col].fillna('N/A').astype(str).str.strip().str.upper()
        
        # Filter for OTM and opening positions
        df = df[(df['Money Type'] == 'OUT_THE_MONEY') & (df['Is Opening Position'] == 'YES')]
        
        return df
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return None

# Function to split content for Discord
def split_for_discord(content, max_length=2000):
    if len(content) <= max_length:
        return [content]
    
    parts = []
    current_part = ""
    for line in content.split('\n'):
        if len(current_part) + len(line) + 1 > max_length:
            parts.append(current_part.strip())
            current_part = line
        else:
            current_part += f"\n{line}" if current_part else line
    
    if current_part:
        parts.append(current_part.strip())
    
    return parts

# Function to send content to Discord
def send_to_discord(content, webhook_url):
    try:
        parts = split_for_discord(content)
        for i, part in enumerate(parts, 1):
            payload = {"content": f"**Part {i}/{len(parts)}**\n{part}"}
            response = requests.post(webhook_url, json=payload)
            if response.status_code != 204:
                return f"Failed to send part {i}: {response.status_code}"
        return f"Sent newsletter in {len(parts)} parts."
    except Exception as e:
        return f"Discord send error: {e}"

# Function to generate dashboard newsletter
def generate_newsletter(df, top_n=10, min_premium=250000, side_codes=['A', 'AA'], 
                      tickers=None, sort_by='Premium Price'):
    if df is None or df.empty:
        return "No valid data for newsletter."
    
    today = pd.to_datetime("today").normalize()
    max_date = today + timedelta(days=2048)  # Next 4 weeks
    
    # Initialize newsletter
    newsletter = f"📊 **DAILY OPTIONS FLOW DASHBOARD - {today.strftime('%B %d, %Y')}** 📊\n"
    newsletter += "═" * 60 + "\n"
    newsletter += "🎯 Professional Options Flow Analysis & Market Intelligence\n\n"
    
    # === SECTION 1: MAJOR INDICES OVERVIEW ===
    newsletter += "█ **1. MAJOR INDICES POSITIONING** █\n"
    newsletter += "─" * 40 + "\n"
    
    major_indices = ['SPY', 'QQQ', 'IWM', 'DIA', 'SMH']
    index_flows = df[df['Ticker'].isin(major_indices)].copy()
    
    if not index_flows.empty:
        for index in major_indices:
            ticker_flows = index_flows[index_flows['Ticker'] == index]
            if not ticker_flows.empty:
                # Calculate total premium and bias
                total_premium = ticker_flows['Premium Price'].sum()
                call_premium = ticker_flows[ticker_flows['Contract Type'] == 'CALL']['Premium Price'].sum()
                put_premium = ticker_flows[ticker_flows['Contract Type'] == 'PUT']['Premium Price'].sum()
                
                call_put_ratio = call_premium / put_premium if put_premium > 0 else float('inf')
                bias = "🟢 BULLISH" if call_put_ratio > 1.5 else "🔴 BEARISH" if call_put_ratio < 0.67 else "⚪ NEUTRAL"
                
                newsletter += f"**{index}**: ${total_premium/1000000:.1f}M total | C/P Ratio: {call_put_ratio:.2f} {bias}\n"
        newsletter += "\n"
    
    # === SECTION 2: VIX POSITIONING SUMMARY ===
    newsletter += "█ **2. VIX POSITIONING & VOLATILITY** █\n"
    newsletter += "─" * 40 + "\n"
    
    vix_related = df[df['Ticker'].str.contains('VIX|UVXY|VXX|SVXY', na=False)].copy()
    if not vix_related.empty:
        vix_flows = vix_related.sort_values('Premium Price', ascending=False).head(5)
        for _, row in vix_flows.iterrows():
            exp_str = row['Expiration Date'].strftime('%m/%d') if pd.notnull(row['Expiration Date']) else 'N/A'
            
            # Determine buy/sell action for VIX
            if row['Contract Type'] == 'CALL' and row['Side Code'] in ['A', 'AA']:
                action = "BUY CALL"
                sentiment = "📈 Vol Up"
            elif row['Contract Type'] == 'CALL' and row['Side Code'] in ['B', 'BB']:
                action = "SELL CALL"
                sentiment = "📉 Vol Down"
            elif row['Contract Type'] == 'PUT' and row['Side Code'] in ['A', 'AA']:
                action = "BUY PUT"
                sentiment = "📉 Vol Down"
            elif row['Contract Type'] == 'PUT' and row['Side Code'] in ['B', 'BB']:
                action = "SELL PUT"
                sentiment = "📈 Vol Up"
            else:
                action = f"{row['Side Code']} {row['Contract Type']}"
                sentiment = "⚪"
            
            newsletter += f"• {row['Ticker']} {action} ${row['Strike Price']:.0f} exp {exp_str} - ${row['Premium Price']/1000:.0f}K {sentiment}\n"
    else:
        newsletter += "• No significant VIX positioning detected\n"
    newsletter += "\n"
    
    # === SECTION 3: S&P 500 SECTOR ETFs ===
    newsletter += "█ **3. S&P 500 SECTOR ETF FLOWS** █\n"
    newsletter += "─" * 40 + "\n"
    
    sector_etfs = ['XLK', 'XLF', 'XLV', 'XLE', 'XLI', 'XLY', 'XLP', 'XLU', 'XLB', 'XLRE', 'XLC']
    sector_flows = df[
        df['Ticker'].isin(sector_etfs) & 
        (df['Premium Price'] >= min_premium) &
        (df['Expiration Date'] > today)
    ].copy()
    
    if not sector_flows.empty:
        # Sort by premium and show individual flows with strike and date
        sector_flows = sector_flows.sort_values('Premium Price', ascending=False).head(6)
        
        for _, row in sector_flows.iterrows():
            flags = "🔥" if row['Is Unusual'] == 'YES' else ""
            flags += "⚡" if row['Is Golden Sweep'] == 'YES' else ""
            
            # Determine buy/sell action for clarity
            if row['Contract Type'] == 'CALL' and row['Side Code'] in ['A', 'AA']:
                action = "BUY CALL"
                sentiment = "📈"
            elif row['Contract Type'] == 'CALL' and row['Side Code'] in ['B', 'BB']:
                action = "SELL CALL"
                sentiment = "�"
            elif row['Contract Type'] == 'PUT' and row['Side Code'] in ['A', 'AA']:
                action = "BUY PUT"
                sentiment = "📉"
            elif row['Contract Type'] == 'PUT' and row['Side Code'] in ['B', 'BB']:
                action = "SELL PUT"
                sentiment = "📈"
            else:
                action = f"{row['Side Code']} {row['Contract Type']}"
                sentiment = "⚪"
            
            newsletter += (f"• **{row['Ticker']}** {action} ${row['Strike Price']:,.0f} "
                          f"exp {row['Expiration Date'].strftime('%m/%d')} - "
                          f"${row['Premium Price']/1000:.0f}K {sentiment} {flags}\n")
    else:
        newsletter += "• No significant sector ETF flows detected\n"
    newsletter += "\n"
    
    # === SECTION 4: EXTREME BULLISH STOCKS ===
    newsletter += "█ **4. EXTREME BULLISH POSITIONS** █\n"
    newsletter += "─" * 40 + "\n"
    
    exclude_indices = {'SPY', 'QQQ', 'SPX', 'SPXW', 'IWM', 'NDX', 'RUT', 'DIA', 'SMH'}
    bullish_df = df[
        ~df['Ticker'].isin(exclude_indices) &
        (df['Expiration Date'] > today) &
        (df['Expiration Date'] <= max_date) &
        (df['Premium Price'] >= min_premium) &
        (((df['Contract Type'] == 'CALL') & (df['Side Code'].isin(['A', 'AA']))) |
         ((df['Contract Type'] == 'PUT') & (df['Side Code'].isin(['B', 'BB']))))
    ].copy()
    
    if not bullish_df.empty:
        bullish_df = bullish_df.sort_values('Premium Price', ascending=False).head(top_n)
        for _, row in bullish_df.iterrows():
            move_pct = abs((row['Strike Price'] - row['Reference Price']) / row['Reference Price'] * 100)
            flags = "🔥" if row['Is Unusual'] == 'YES' else ""
            flags += "⚡" if row['Is Golden Sweep'] == 'YES' else ""
            
            # Determine buy/sell action for clarity
            if row['Contract Type'] == 'CALL' and row['Side Code'] in ['A', 'AA']:
                action = "BUY CALL"
            elif row['Contract Type'] == 'PUT' and row['Side Code'] in ['B', 'BB']:
                action = "SELL PUT"
            else:
                action = f"{row['Side Code']} {row['Contract Type']}"
            
            newsletter += (f"• **{row['Ticker']}** {action} ${row['Strike Price']:,.0f} "
                          f"exp {row['Expiration Date'].strftime('%m/%d')} - "
                          f"${row['Premium Price']/1000:.0f}K ({move_pct:.0f}% move) {flags}\n")
    newsletter += "\n"
    
    # === SECTION 5: EXTREME BEARISH STOCKS ===
    newsletter += "█ **5. EXTREME BEARISH POSITIONS** █\n"
    newsletter += "─" * 40 + "\n"
    
    bearish_df = df[
        ~df['Ticker'].isin(exclude_indices) &
        (df['Expiration Date'] > today) &
        (df['Expiration Date'] <= max_date) &
        (df['Premium Price'] >= min_premium) &
        (((df['Contract Type'] == 'CALL') & (df['Side Code'].isin(['B', 'BB']))) |
         ((df['Contract Type'] == 'PUT') & (df['Side Code'].isin(['A', 'AA']))))
    ].copy()
    
    if not bearish_df.empty:
        bearish_df = bearish_df.sort_values('Premium Price', ascending=False).head(top_n)
        for _, row in bearish_df.iterrows():
            move_pct = abs((row['Strike Price'] - row['Reference Price']) / row['Reference Price'] * 100)
            flags = "🔥" if row['Is Unusual'] == 'YES' else ""
            flags += "⚡" if row['Is Golden Sweep'] == 'YES' else ""
            
            # Determine buy/sell action for clarity
            if row['Contract Type'] == 'CALL' and row['Side Code'] in ['B', 'BB']:
                action = "SELL CALL"
            elif row['Contract Type'] == 'PUT' and row['Side Code'] in ['A', 'AA']:
                action = "BUY PUT"
            else:
                action = f"{row['Side Code']} {row['Contract Type']}"
            
            newsletter += (f"• **{row['Ticker']}** {action} ${row['Strike Price']:,.0f} "
                          f"exp {row['Expiration Date'].strftime('%m/%d')} - "
                          f"${row['Premium Price']/1000:.0f}K ({move_pct:.0f}% move) {flags}\n")
    newsletter += "\n"
    
    # === SECTION 6: UNUSUAL ACTIVITY ALERTS ===
    newsletter += "█ **6. UNUSUAL ACTIVITY ALERTS** █\n"
    newsletter += "─" * 40 + "\n"
    
    unusual_flows = df[
        (df['Is Unusual'] == 'YES') &
        (df['Premium Price'] >= min_premium/2) &  # Lower threshold for unusual
        (df['Expiration Date'] > today)
    ].sort_values('Premium Price', ascending=False).head(8)
    
    if not unusual_flows.empty:
        for _, row in unusual_flows.iterrows():
            move_pct = abs((row['Strike Price'] - row['Reference Price']) / row['Reference Price'] * 100)
            
            # Determine buy/sell action for clarity
            if row['Contract Type'] == 'CALL' and row['Side Code'] in ['A', 'AA']:
                action = "BUY CALL"
                sentiment = "📈"
            elif row['Contract Type'] == 'CALL' and row['Side Code'] in ['B', 'BB']:
                action = "SELL CALL"
                sentiment = "📉"
            elif row['Contract Type'] == 'PUT' and row['Side Code'] in ['A', 'AA']:
                action = "BUY PUT"
                sentiment = "📉"
            elif row['Contract Type'] == 'PUT' and row['Side Code'] in ['B', 'BB']:
                action = "SELL PUT"
                sentiment = "📈"
            else:
                action = f"{row['Side Code']} {row['Contract Type']}"
                sentiment = "⚪"
            
            newsletter += (f"🔥 **{row['Ticker']}** {action} ${row['Strike Price']:,.0f} "
                          f"exp {row['Expiration Date'].strftime('%m/%d')} - "
                          f"${row['Premium Price']/1000:.0f}K {sentiment}\n")
    else:
        newsletter += "• No significant unusual activity detected\n"
    newsletter += "\n"
    
    # === SECTION 7: HIGH VOLUME CONCENTRATION ===
    newsletter += "█ **7. HIGH VOLUME CONCENTRATIONS** █\n"
    newsletter += "─" * 40 + "\n"
    
    high_volume = df[
        (df['Size'] >= 1000) &  # High contract volume
        (df['Premium Price'] >= min_premium) &
        (df['Expiration Date'] > today)
    ].groupby('Ticker').agg({
        'Premium Price': 'sum',
        'Size': 'sum',
        'Contract Type': lambda x: list(x)
    }).sort_values('Premium Price', ascending=False).head(5)
    
    if not high_volume.empty:
        for ticker, row in high_volume.iterrows():
            total_contracts = row['Size']
            total_premium = row['Premium Price']
            call_count = sum(1 for ct in row['Contract Type'] if ct == 'CALL')
            put_count = len(row['Contract Type']) - call_count
            bias = "📈 CALL HEAVY" if call_count > put_count * 1.5 else "📉 PUT HEAVY" if put_count > call_count * 1.5 else "⚖️ MIXED"
            
            newsletter += (f"• **{ticker}**: {total_contracts:,} contracts, "
                          f"${total_premium/1000000:.1f}M premium {bias}\n")
    else:
        newsletter += "• No high volume concentrations detected\n"
    newsletter += "\n"
    
    # === SECTION 8: WEEKLY EXPIRATION FOCUS ===
    newsletter += "█ **8. WEEKLY EXPIRATION FOCUS** █\n"
    newsletter += "─" * 40 + "\n"
    
    weekly_exp = today + timedelta(days=7 - today.weekday() + 4)  # Next Friday
    weekly_flows = df[
        (df['Expiration Date'] == weekly_exp) &
        (df['Premium Price'] >= min_premium/2)
    ].sort_values('Premium Price', ascending=False).head(6)
    
    if not weekly_flows.empty:
        newsletter += f"⏰ **Expiring {weekly_exp.strftime('%m/%d/%Y')}**:\n"
        for _, row in weekly_flows.iterrows():
            # Determine buy/sell action for clarity
            if row['Contract Type'] == 'CALL' and row['Side Code'] in ['A', 'AA']:
                action = "BUY CALL"
                sentiment = "📈"
            elif row['Contract Type'] == 'CALL' and row['Side Code'] in ['B', 'BB']:
                action = "SELL CALL"
                sentiment = "📉"
            elif row['Contract Type'] == 'PUT' and row['Side Code'] in ['A', 'AA']:
                action = "BUY PUT"
                sentiment = "📉"
            elif row['Contract Type'] == 'PUT' and row['Side Code'] in ['B', 'BB']:
                action = "SELL PUT"
                sentiment = "📈"
            else:
                action = f"{row['Side Code']} {row['Contract Type']}"
                sentiment = "⚪"
            
            newsletter += (f"• {row['Ticker']} {action} ${row['Strike Price']:,.0f} - "
                          f"${row['Premium Price']/1000:.0f}K {sentiment}\n")
    else:
        newsletter += f"• No significant weekly flows for {weekly_exp.strftime('%m/%d')}\n"
    newsletter += "\n"
    
    # === SECTION 9: CROSS-SECTION ANALYSIS ===
    newsletter += "█ **9. CROSS-SECTION ANALYSIS** █\n"
    newsletter += "─" * 40 + "\n"
    
    # Define stock ecosystems and themes
    ecosystems = {
        "AI/Chip Theme": ["NVDA", "AMD", "TSM", "ASML", "AVGO", "QCOM", "MU", "INTC", "ARM", "MRVL"],
        "Cloud/Software": ["MSFT", "GOOGL", "CRWD", "AMZN", "CRM", "ORCL", "ADBE", "NOW", "SNOW", "PLTR"],
        "EV/Auto Theme": ["TSLA", "RIVN", "LCID", "F", "GM", "NIO", "XPEV", "LI", "BYD"],
        "Social/Meta": ["META", "SNAP", "PINS", "RDDT", "ROKU", "SPOT"],
        "Fintech/Banks": ["JPM", "BAC", "GS", "MS", "WFC", "C", "PYPL", "XYZ", "COIN", "HOOD"],
        "Energy/Oil": ["XOM", "CVX", "COP", "EOG", "SLB", "OXY", "DVN", "MRO"],
        "Biotech/Pharma": ["JNJ", "PFE", "MRNA", "BNTX", "GILD", "BIIB", "AMGN", "REGN"],
        "Retail/Consumer": ["AMZN", "WMT", "TGT", "COST", "HD", "LOW", "NKE", "SBUX"],
        "Streaming/Media": ["NFLX", "DIS", "WBD", "PARA", "ROKU", "SPOT"],
        "Crypto Proxy": ["COIN", "MSTR", "RIOT", "MARA", "IBIT", "ETHA"]
    }
    
    # Get significant flows for analysis (lower threshold for theme detection)
    theme_flows = df[
        (df['Premium Price'] >= min_premium/2) &
        (df['Expiration Date'] > today)
    ].copy()
    
    if not theme_flows.empty:
        detected_themes = []
        
        for theme, tickers in ecosystems.items():
            # Find flows in this theme
            theme_data = theme_flows[theme_flows['Ticker'].isin(tickers)]
            
            if len(theme_data) >= 2:  # At least 2 flows to constitute a theme
                total_premium = theme_data['Premium Price'].sum()
                bullish_flows = theme_data[
                    ((theme_data['Contract Type'] == 'CALL') & (theme_data['Side Code'].isin(['A', 'AA']))) |
                    ((theme_data['Contract Type'] == 'PUT') & (theme_data['Side Code'].isin(['B', 'BB'])))
                ]
                bearish_flows = theme_data[
                    ((theme_data['Contract Type'] == 'CALL') & (theme_data['Side Code'].isin(['B', 'BB']))) |
                    ((theme_data['Contract Type'] == 'PUT') & (theme_data['Side Code'].isin(['A', 'AA'])))
                ]
                
                bullish_premium = bullish_flows['Premium Price'].sum()
                bearish_premium = bearish_flows['Premium Price'].sum()
                
                # Determine theme bias
                if bullish_premium > bearish_premium * 1.5:
                    bias = "📈 BULLISH"
                    bias_ratio = f"{bullish_premium/1000000:.1f}M vs {bearish_premium/1000000:.1f}M"
                elif bearish_premium > bullish_premium * 1.5:
                    bias = "📉 BEARISH" 
                    bias_ratio = f"{bearish_premium/1000000:.1f}M vs {bullish_premium/1000000:.1f}M"
                else:
                    bias = "⚖️ MIXED"
                    bias_ratio = f"${total_premium/1000000:.1f}M total"
                
                # Get top tickers in theme
                top_theme_tickers = theme_data.groupby('Ticker')['Premium Price'].sum().sort_values(ascending=False).head(3)
                ticker_list = ", ".join([f"{ticker}(${premium/1000000:.1f}M)" for ticker, premium in top_theme_tickers.items()])
                
                detected_themes.append({
                    'theme': theme,
                    'bias': bias,
                    'ratio': bias_ratio,
                    'tickers': ticker_list,
                    'total_premium': total_premium,
                    'flow_count': len(theme_data)
                })
        
        # Sort by total premium and show top themes
        detected_themes.sort(key=lambda x: x['total_premium'], reverse=True)
        
        if detected_themes:
            for theme_info in detected_themes[:5]:  # Show top 5 themes
                newsletter += (f"🎯 **{theme_info['theme']}**: {theme_info['bias']} "
                              f"({theme_info['flow_count']} flows, {theme_info['ratio']})\n")
                newsletter += f"   └─ Key Players: {theme_info['tickers']}\n\n"
        else:
            newsletter += "• No clear thematic patterns detected\n"
    
    # Cross-asset correlations
    newsletter += "**🔄 Cross-Asset Signals:**\n"
    
    # VIX vs Equity flows
    vix_flows = df[df['Ticker'].str.contains('VIX|UVXY|VXX', na=False)]
    equity_flows = df[~df['Ticker'].isin(['SPY', 'QQQ', 'IWM', 'DIA']) & 
                     ~df['Ticker'].str.contains('VIX|UVXY|VXX', na=False)]
    
    if not vix_flows.empty and not equity_flows.empty:
        vix_call_buying = vix_flows[(vix_flows['Contract Type'] == 'CALL') & 
                                   (vix_flows['Side Code'].isin(['A', 'AA']))]['Premium Price'].sum()
        
        equity_put_buying = equity_flows[(equity_flows['Contract Type'] == 'PUT') & 
                                        (equity_flows['Side Code'].isin(['A', 'AA']))]['Premium Price'].sum()
        
        if vix_call_buying > 0 and equity_put_buying > 0:
            correlation = "🔄 VIX calls + Equity puts = Defensive positioning"
            newsletter += f"• {correlation}\n"
    
    # Sector rotation signals
    sector_etfs = ['XLK', 'XLF', 'XLV', 'XLE', 'XLI', 'XLY', 'XLP', 'XLU', 'XLB', 'XLRE', 'XLC']
    sector_data = df[df['Ticker'].isin(sector_etfs) & (df['Premium Price'] >= min_premium/4)]
    
    if not sector_data.empty:
        sector_summary = []
        for sector in sector_etfs:
            sector_flows = sector_data[sector_data['Ticker'] == sector]
            if not sector_flows.empty:
                bullish_premium = sector_flows[
                    ((sector_flows['Contract Type'] == 'CALL') & (sector_flows['Side Code'].isin(['A', 'AA']))) |
                    ((sector_flows['Contract Type'] == 'PUT') & (sector_flows['Side Code'].isin(['B', 'BB'])))
                ]['Premium Price'].sum()
                
                bearish_premium = sector_flows[
                    ((sector_flows['Contract Type'] == 'CALL') & (sector_flows['Side Code'].isin(['B', 'BB']))) |
                    ((sector_flows['Contract Type'] == 'PUT') & (sector_flows['Side Code'].isin(['A', 'AA'])))
                ]['Premium Price'].sum()
                
                net_flow = bullish_premium - bearish_premium
                sector_summary.append({'sector': sector, 'net_flow': net_flow})
        
        if sector_summary:
            sector_summary.sort(key=lambda x: abs(x['net_flow']), reverse=True)
            top_rotation = sector_summary[:3]
            rotation_text = " | ".join([f"{s['sector']}({'📈' if s['net_flow'] > 0 else '📉'})" for s in top_rotation])
            newsletter += f"• Sector Rotation: {rotation_text}\n"
    
    newsletter += "\n"
    
    return newsletter

# Function to generate Twitter newsletter
def generate_twitter_newsletter(df, top_n=5, min_premium=500000, side_codes=['A', 'AA'], 
                               tickers=None, sort_by='Premium Price'):
    if df is None or df.empty:
        return "No valid data for Twitter newsletter."
    
    today = pd.to_datetime("today").normalize()
    max_date = today + timedelta(days=32)
    exclude_indices = {'SPY', 'QQQ', 'SPX', 'SPXW', 'IWM', 'NDX', 'RUT', 'DIA', 'SMH'}
    
    # Twitter newsletter - concise dashboard format
    twitter_post = f"📊 DAILY FLOW DASHBOARD - {today.strftime('%m/%d/%Y')}\n"
    twitter_post += "═" * 35 + "\n\n"
    
    # 1. INDICES OVERVIEW
    major_indices = ['SPY', 'QQQ', 'IWM', 'DIA', 'SMH']
    index_flows = df[df['Ticker'].isin(major_indices)]
    
    if not index_flows.empty:
        twitter_post += "🎯 INDICES:\n"
        for index in ['SPY', 'QQQ', 'IWM']:  # Top 3 for Twitter
            ticker_flows = index_flows[index_flows['Ticker'] == index]
            if not ticker_flows.empty:
                total_premium = ticker_flows['Premium Price'].sum()
                call_premium = ticker_flows[ticker_flows['Contract Type'] == 'CALL']['Premium Price'].sum()
                put_premium = ticker_flows[ticker_flows['Contract Type'] == 'PUT']['Premium Price'].sum()
                
                if put_premium > 0:
                    ratio = call_premium / put_premium
                    bias = "📈" if ratio > 1.5 else "📉" if ratio < 0.67 else "⚪"
                    twitter_post += f"{index}: {bias} ${total_premium/1000000:.1f}M\n"
        twitter_post += "\n"
    
    # 2. TOP BULLISH PLAYS
    bullish_df = df[
        ~df['Ticker'].isin(exclude_indices) &
        (df['Expiration Date'] > today) &
        (df['Premium Price'] >= min_premium) &
        (((df['Contract Type'] == 'CALL') & (df['Side Code'].isin(['A', 'AA']))) |
         ((df['Contract Type'] == 'PUT') & (df['Side Code'].isin(['B', 'BB']))))
    ].sort_values('Premium Price', ascending=False).head(3)
    
    if not bullish_df.empty:
        twitter_post += "📈 BULLISH:\n"
        for _, row in bullish_df.iterrows():
            flags = "🔥" if row['Is Unusual'] == 'YES' else ""
            twitter_post += f"{row['Ticker']} ${row['Strike Price']:,.0f}{row['Contract Type'][0]} ${row['Premium Price']/1000:.0f}K {flags}\n"
        twitter_post += "\n"
    
    # 3. TOP BEARISH PLAYS
    bearish_df = df[
        ~df['Ticker'].isin(exclude_indices) &
        (df['Expiration Date'] > today) &
        (df['Premium Price'] >= min_premium) &
        (((df['Contract Type'] == 'CALL') & (df['Side Code'].isin(['B', 'BB']))) |
         ((df['Contract Type'] == 'PUT') & (df['Side Code'].isin(['A', 'AA']))))
    ].sort_values('Premium Price', ascending=False).head(3)
    
    if not bearish_df.empty:
        twitter_post += "� BEARISH:\n"
        for _, row in bearish_df.iterrows():
            flags = "�" if row['Is Unusual'] == 'YES' else ""
            twitter_post += f"{row['Ticker']} ${row['Strike Price']:,.0f}{row['Contract Type'][0]} ${row['Premium Price']/1000:.0f}K {flags}\n"
        twitter_post += "\n"
    
    # 4. UNUSUAL ACTIVITY
    unusual_flows = df[
        (df['Is Unusual'] == 'YES') &
        (df['Premium Price'] >= min_premium/2)
    ].sort_values('Premium Price', ascending=False).head(2)
    
    if not unusual_flows.empty:
        twitter_post += "🔥 UNUSUAL:\n"
        for _, row in unusual_flows.iterrows():
            sentiment = "📈" if ((row['Contract Type'] == 'CALL' and row['Side Code'] in ['A', 'AA']) or 
                               (row['Contract Type'] == 'PUT' and row['Side Code'] in ['B', 'BB'])) else "📉"
            twitter_post += f"{row['Ticker']} ${row['Strike Price']:,.0f}{row['Contract Type'][0]} ${row['Premium Price']/1000:.0f}K {sentiment}\n"
        twitter_post += "\n"
    
    twitter_post += "⚠️ Educational only. Not financial advice.\n"
    twitter_post += "#OptionsFlow #Trading #FlowDashboard"
    
    return twitter_post

# Function to display flows for a specific symbol
def display_symbol_flows(df, symbol):
    if df is None or df.empty:
        st.warning("No data available.")
        return
    
    today = pd.to_datetime("today").normalize()
    symbol_df = df[
        (df['Ticker'] == symbol.upper()) &
        (df['Is Opening Position'] == 'YES') &
        (df['Premium Price'] > 250000) &
        (df['Expiration Date'] > today)
    ].sort_values(['Expiration Date', 'Premium Price'])
    
    if symbol_df.empty:
        st.warning(f"No qualifying flows for {symbol}.")
        return
    
    st.subheader(f"Flows for {symbol} (Premium > $250K)")
    for _, row in symbol_df.iterrows():
        move_pct = abs((row['Strike Price'] - row['Reference Price']) / 
                      row['Reference Price'] * 100)
        sentiment = (
            "🟢" if (row['Contract Type'] == 'CALL' and row['Side Code'] in ['A', 'AA']) or
                    (row['Contract Type'] == 'PUT' and row['Side Code'] in ['B', 'BB']) else
            "🔴" if (row['Contract Type'] == 'CALL' and row['Side Code'] in ['B', 'BB']) or
                    (row['Contract Type'] == 'PUT' and row['Side Code'] in ['A', 'AA']) else "N/A"
        )
        flags = []
        if row['Is Unusual'] == 'YES':
            flags.append("UNUSUAL")
        if row['Is Golden Sweep'] == 'YES':
            flags.append("GOLDEN")
        flags_str = f" [{' '.join(flags)}]" if flags else ""
        
        st.markdown(
            f"• **{row['Contract Type']}** | Strike: ${row['Strike Price']:,.2f} | "
            f"Exp: {row['Expiration Date'].strftime('%Y-%m-%d')} | "
            f"Premium: ${row['Premium Price']:,.0f} | Contracts: {row['Size']} | "
            f"Move: {move_pct:.1f}% | Sentiment: {sentiment} | Side: {row['Side Code']}{flags_str}"
        )
    
    with st.expander("Raw Data"):
        st.dataframe(symbol_df)

# Function to display repeat flows grouped by ticker
def display_repeat_flows(df, min_premium=30000):
    if df is None or df.empty:
        st.warning("No data available.")
        return
    
    # Filter by minimum premium and opening positions
    filtered_df = df[
        (df['Premium Price'] >= min_premium) &
        (df['Is Opening Position'] == 'YES')
    ].copy()
    
    if filtered_df.empty:
        st.warning(f"No flows found with premium >= ${min_premium:,}")
        return
    
    # Create a composite key for similar contracts
    # Ensure Strike Price is numeric and Expiration Date is datetime
    filtered_df['Strike_Price_Clean'] = pd.to_numeric(filtered_df['Strike Price'], errors='coerce').fillna(0)
    filtered_df['Expiration_Date_Clean'] = pd.to_datetime(filtered_df['Expiration Date'], errors='coerce')
    
    filtered_df['Contract_Key'] = (
        filtered_df['Ticker'] + '_' + 
        filtered_df['Contract Type'] + '_' + 
        filtered_df['Strike_Price_Clean'].astype(str) + '_' + 
        filtered_df['Expiration_Date_Clean'].dt.strftime('%Y-%m-%d')
    )
    
    # Group by ticker and contract key to find repeats
    grouped = filtered_df.groupby(['Ticker', 'Contract_Key']).agg({
        'Premium Price': ['count', 'sum', 'mean'],
        'Size': 'sum',
        'Contract Type': 'first',
        'Strike_Price_Clean': 'first',  # Use cleaned version
        'Expiration_Date_Clean': 'first',  # Use cleaned version
        'Reference Price': 'first',
        'Side Code': lambda x: list(x),
        'Is Unusual': lambda x: 'YES' if 'YES' in x.values else 'NO',
        'Is Golden Sweep': lambda x: 'YES' if 'YES' in x.values else 'NO'
    }).reset_index()
    
    # Flatten column names
    grouped.columns = ['Ticker', 'Contract_Key', 'Flow_Count', 'Total_Premium', 'Avg_Premium',
                      'Total_Contracts', 'Contract_Type', 'Strike_Price', 'Expiration_Date',
                      'Reference_Price', 'Side_Codes', 'Has_Unusual', 'Has_Golden']
    
    # Filter for repeat flows (more than 1 occurrence)
    repeat_flows = grouped[grouped['Flow_Count'] > 1].copy()
    
    if repeat_flows.empty:
        st.warning("No repeat flows found with the specified criteria.")
        return
    
    # Sort by total premium
    repeat_flows = repeat_flows.sort_values('Total_Premium', ascending=False)
    
    st.subheader(f"🔄 Repeat Flows (Premium >= ${min_premium:,})")
    st.markdown(f"Found **{len(repeat_flows)}** contracts with multiple flows")
    
    # Group by ticker for display
    for ticker in repeat_flows['Ticker'].unique():
        ticker_flows = repeat_flows[repeat_flows['Ticker'] == ticker]
        
        with st.expander(f"📈 {ticker} ({len(ticker_flows)} repeat contracts)", expanded=True):
            for _, row in ticker_flows.iterrows():
                move_pct = abs((row['Strike_Price'] - row['Reference_Price']) / 
                              row['Reference_Price'] * 100)
                
                # Determine predominant sentiment
                side_codes = row['Side_Codes']
                bullish_sides = sum(1 for side in side_codes if side in ['A', 'AA'])
                bearish_sides = sum(1 for side in side_codes if side in ['B', 'BB'])
                
                if row['Contract_Type'] == 'CALL':
                    if bullish_sides > bearish_sides:
                        sentiment = "🟢 Bullish"
                    elif bearish_sides > bullish_sides:
                        sentiment = "🔴 Bearish"
                    else:
                        sentiment = "⚪ Mixed"
                else:  # PUT
                    if bullish_sides > bearish_sides:
                        sentiment = "🔴 Bearish"
                    elif bearish_sides > bullish_sides:
                        sentiment = "🟢 Bullish"
                    else:
                        sentiment = "⚪ Mixed"
                
                flags = []
                if row['Has_Unusual'] == 'YES':
                    flags.append("🔥 UNUSUAL")
                if row['Has_Golden'] == 'YES':
                    flags.append("⚡ GOLDEN")
                flags_str = f" [{' '.join(flags)}]" if flags else ""
                
                # Display the flow information
                exp_date_str = row['Expiration_Date'].strftime('%Y-%m-%d') if hasattr(row['Expiration_Date'], 'strftime') else str(row['Expiration_Date'])
                st.markdown(f"""
                **{row['Contract_Type']} ${row['Strike_Price']:,.2f}** exp {exp_date_str}
                - **{row['Flow_Count']} flows** totaling **${row['Total_Premium']:,.0f}** 
                - Avg Premium: ${row['Avg_Premium']:,.0f} | Total Contracts: {row['Total_Contracts']:,}
                - Move Required: {move_pct:.1f}% | Sentiment: {sentiment}
                - Side Codes: {', '.join(map(str, row['Side_Codes']))}{flags_str}
                """)
                st.divider()
    
    # Summary statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Repeat Contracts", len(repeat_flows))
    
    with col2:
        total_premium = repeat_flows['Total_Premium'].sum()
        st.metric("Total Premium", f"${total_premium/1000000:.1f}M")
    
    with col3:
        avg_flows_per_contract = repeat_flows['Flow_Count'].mean()
        st.metric("Avg Flows per Contract", f"{avg_flows_per_contract:.1f}")
    
    with col4:
        most_active = repeat_flows.loc[repeat_flows['Flow_Count'].idxmax(), 'Ticker']
        max_flows = repeat_flows['Flow_Count'].max()
        st.metric("Most Active", f"{most_active} ({max_flows} flows)")
    
    # Enhanced sharing summary
    with st.expander("📋 Easy Sharing Summary", expanded=True):
        sharing_summary = generate_repeat_flows_summary(repeat_flows)
        
        # Streamlined single format
        st.markdown("**📊 Professional Report Format**")
        st.text_area("Copy & Share:", sharing_summary['Professional'], height=300, key="repeat_summary")
    
    # Show raw data
    with st.expander("Raw Repeat Flows Data"):
        display_df = repeat_flows.copy()
        display_df['Side_Codes'] = display_df['Side_Codes'].apply(lambda x: ', '.join(map(str, x)))
        st.dataframe(display_df, use_container_width=True)

def detect_options_strategy(flows_df):
    """
    Detect and articulate common options strategies from simultaneous flows
    """
    if len(flows_df) < 2:
        return None
    
    # Group flows by expiration and contract type
    flows_list = []
    for _, flow in flows_df.iterrows():
        flows_list.append({
            'type': flow['Contract Type'],
            'strike': flow['Strike Price'],
            'side': flow['Side Code'],
            'premium': flow['Premium Price'],
            'exp_date': flow['Expiration Date']
        })
    
    # Sort flows by strike price for easier analysis
    flows_list.sort(key=lambda x: x['strike'])
    
    # Check for common strategies
    if len(flows_list) == 2:
        flow1, flow2 = flows_list[0], flows_list[1]
        
        # Same expiration date required for most strategies
        if flow1['exp_date'] == flow2['exp_date']:
            
            # Call Debit Spread (Bull Call Spread): Buy lower call + Sell higher call
            if (flow1['type'] == 'CALL' and flow2['type'] == 'CALL' and
                flow1['side'] in ['A', 'AA'] and flow2['side'] in ['B', 'BB'] and
                flow1['strike'] < flow2['strike']):
                net_cost = flow1['premium'] - flow2['premium']
                return f"🟢 **BULLISH CALL DEBIT SPREAD** - Buy ${flow1['strike']:.0f}C, Sell ${flow2['strike']:.0f}C (Net Cost: ${net_cost/1000:.0f}K)"
            
            # Call Credit Spread (Bear Call Spread): Sell lower call + Buy higher call  
            elif (flow1['type'] == 'CALL' and flow2['type'] == 'CALL' and
                  flow1['side'] in ['B', 'BB'] and flow2['side'] in ['A', 'AA'] and
                  flow1['strike'] < flow2['strike']):
                net_credit = flow1['premium'] - flow2['premium']
                return f"🔴 **BEARISH CALL CREDIT SPREAD** - Sell ${flow1['strike']:.0f}C, Buy ${flow2['strike']:.0f}C (Net Credit: ${net_credit/1000:.0f}K)"
            
            # Put Debit Spread (Bear Put Spread): Buy higher put + Sell lower put
            elif (flow1['type'] == 'PUT' and flow2['type'] == 'PUT' and
                  flow1['side'] in ['B', 'BB'] and flow2['side'] in ['A', 'AA'] and
                  flow1['strike'] < flow2['strike']):
                net_cost = flow2['premium'] - flow1['premium']
                return f"🔴 **BEARISH PUT DEBIT SPREAD** - Buy ${flow2['strike']:.0f}P, Sell ${flow1['strike']:.0f}P (Net Cost: ${net_cost/1000:.0f}K)"
            
            # Put Credit Spread (Bull Put Spread): Sell higher put + Buy lower put
            elif (flow1['type'] == 'PUT' and flow2['type'] == 'PUT' and
                  flow1['side'] in ['A', 'AA'] and flow2['side'] in ['B', 'BB'] and
                  flow1['strike'] < flow2['strike']):
                net_credit = flow2['premium'] - flow1['premium']
                return f"🟢 **BULLISH PUT CREDIT SPREAD** - Sell ${flow2['strike']:.0f}P, Buy ${flow1['strike']:.0f}P (Net Credit: ${net_credit/1000:.0f}K)"
            
            # Straddle: Same strike, same exp, both call and put
            elif (flow1['strike'] == flow2['strike'] and
                  ((flow1['type'] == 'CALL' and flow2['type'] == 'PUT') or
                   (flow1['type'] == 'PUT' and flow2['type'] == 'CALL'))):
                if flow1['side'] in ['A', 'AA'] and flow2['side'] in ['A', 'AA']:
                    total_cost = flow1['premium'] + flow2['premium']
                    return f"⚡ **LONG STRADDLE** - Buy ${flow1['strike']:.0f} Call & Put (Total Cost: ${total_cost/1000:.0f}K, expects big move)"
                elif flow1['side'] in ['B', 'BB'] and flow2['side'] in ['B', 'BB']:
                    total_credit = flow1['premium'] + flow2['premium']
                    return f"💤 **SHORT STRADDLE** - Sell ${flow1['strike']:.0f} Call & Put (Total Credit: ${total_credit/1000:.0f}K, expects small move)"
            
            # Strangle: Different strikes, same exp, call and put
            elif (flow1['strike'] != flow2['strike'] and
                  ((flow1['type'] == 'CALL' and flow2['type'] == 'PUT') or
                   (flow1['type'] == 'PUT' and flow2['type'] == 'CALL'))):
                if flow1['side'] in ['A', 'AA'] and flow2['side'] in ['A', 'AA']:
                    total_cost = flow1['premium'] + flow2['premium']
                    return f"⚡ **LONG STRANGLE** - Buy calls & puts at different strikes (Total Cost: ${total_cost/1000:.0f}K, expects big move)"
                elif flow1['side'] in ['B', 'BB'] and flow2['side'] in ['B', 'BB']:
                    total_credit = flow1['premium'] + flow2['premium']
                    return f"💤 **SHORT STRANGLE** - Sell calls & puts at different strikes (Total Credit: ${total_credit/1000:.0f}K, expects small move)"
    
    # Multi-leg strategies for 3+ flows
    elif len(flows_list) >= 3:
        # Check for Iron Condor or other complex strategies
        call_flows = [f for f in flows_list if f['type'] == 'CALL']
        put_flows = [f for f in flows_list if f['type'] == 'PUT']
        
        if len(call_flows) == 2 and len(put_flows) == 2:
            return "🦅 **IRON CONDOR or COMPLEX STRATEGY** - Multiple strikes and types (4-leg strategy)"
        elif len(call_flows) >= 2:
            return "📈 **COMPLEX CALL STRATEGY** - Multiple call strikes (multi-leg)"
        elif len(put_flows) >= 2:
            return "📉 **COMPLEX PUT STRATEGY** - Multiple put strikes (multi-leg)"
    
    # Default for unrecognized patterns
    call_count = sum(1 for f in flows_list if f['type'] == 'CALL')
    put_count = sum(1 for f in flows_list if f['type'] == 'PUT')
    buy_count = sum(1 for f in flows_list if f['side'] in ['A', 'AA'])
    sell_count = sum(1 for f in flows_list if f['side'] in ['B', 'BB'])
    
    if call_count > put_count:
        return f"📞 **CALL-FOCUSED STRATEGY** - {call_count} calls, {put_count} puts ({buy_count} buys, {sell_count} sells)"
    elif put_count > call_count:
        return f"📱 **PUT-FOCUSED STRATEGY** - {put_count} puts, {call_count} calls ({buy_count} buys, {sell_count} sells)"
    else:
        return f"⚖️ **BALANCED STRATEGY** - Equal calls/puts ({buy_count} buys, {sell_count} sells)"

# Function to identify high-premium, high-quantity flows with multiple occurrences
def display_high_volume_flows(df, min_premium=40000, min_quantity=900, time_grouping="1 Second", summary_top_n=30):
    """
    Identify flows where premium > threshold and quantity > threshold
    and multiple flows came for the same stocks at the same time
    """
    if df is None or df.empty:
        st.warning("No data available.")
        return
    
    today = pd.to_datetime("today").normalize()
    
    # Filter for high premium and high quantity flows
    high_volume_df = df[
        (df['Premium Price'] >= min_premium) &
        (df['Size'] >= min_quantity) &
        (df['Expiration Date'] > today)
    ].copy()
    
    if high_volume_df.empty:
        st.warning(f"No flows found with premium >= ${min_premium:,} and quantity >= {min_quantity}")
        return
    
    # Add time grouping using Trade Time field to find flows at exact same time
    # Check if Trade Time column exists, otherwise fall back to date grouping
    if 'Trade Time' in high_volume_df.columns:
        time_column = 'Trade Time'
        # Parse the trade time and round based on selected time grouping
        high_volume_df['Trade_Time_Clean'] = pd.to_datetime(high_volume_df[time_column], errors='coerce')
        
        # Map time grouping to pandas frequency
        time_freq_map = {
            "1 Second": "S",
            "5 Seconds": "5S", 
            "10 Seconds": "10S",
            "30 Seconds": "30S",
            "1 Minute": "T"
        }
        freq = time_freq_map.get(time_grouping, "S")
        
        # Round to selected time window to group flows that happened within the same window
        high_volume_df['Trade_Time_Rounded'] = high_volume_df['Trade_Time_Clean'].dt.round(freq)
        grouping_column = 'Trade_Time_Rounded'
    else:
        time_column = 'Date'  
        high_volume_df['Trade_Time_Clean'] = pd.to_datetime(high_volume_df.get('Date', today.date() if hasattr(today, 'date') else today))
        high_volume_df['Trade_Time_Rounded'] = high_volume_df['Trade_Time_Clean']
        grouping_column = 'Trade_Time_Rounded'
    
    # Group by ticker and rounded trade time to find stocks with multiple simultaneous flows
    ticker_time_groups = high_volume_df.groupby(['Ticker', grouping_column]).agg({
        'Premium Price': ['count', 'sum', 'mean'],
        'Size': ['count', 'sum', 'mean'],
        'Contract Type': lambda x: list(x),
        'Strike Price': lambda x: list(x),
        'Expiration Date': lambda x: list(x),
        'Reference Price': 'first',
        'Side Code': lambda x: list(x),
        'Is Unusual': lambda x: 'YES' if 'YES' in x.values else 'NO',
        'Is Golden Sweep': lambda x: 'YES' if 'YES' in x.values else 'NO'
    }).reset_index()
    
    # Flatten column names
    ticker_time_groups.columns = [
        'Ticker', 'Trade_Time', 'Flow_Count', 'Total_Premium', 'Avg_Premium',
        'Contract_Count', 'Total_Quantity', 'Avg_Quantity', 'Contract_Types',
        'Strike_Prices', 'Expiration_Dates', 'Reference_Price', 'Side_Codes',
        'Has_Unusual', 'Has_Golden'
    ]
    
    # Filter for tickers with multiple high-volume flows at same time (2 or more)
    multiple_flows = ticker_time_groups[ticker_time_groups['Flow_Count'] >= 2].copy()
    
    if multiple_flows.empty:
        st.warning("No stocks found with multiple high-volume flows at the same trade time meeting the criteria.")
        return
    
    # Sort by total premium
    multiple_flows = multiple_flows.sort_values('Total_Premium', ascending=False)
    
    st.subheader(f"🚨 High-Volume Flow Clusters (Within {time_grouping})")
    st.markdown(f"**Criteria:** Premium ≥ ${min_premium:,} AND Quantity ≥ {min_quantity:,} contracts")
    st.markdown(f"**Grouping:** Multiple flows for same stock within {time_grouping.lower()} time windows")
    st.markdown(f"**Found:** {len(multiple_flows)} time-clustered flow groups")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Stocks with Clusters", len(multiple_flows))
    
    with col2:
        total_premium = multiple_flows['Total_Premium'].sum()
        st.metric("Total Premium", f"${total_premium/1000000:.1f}M")
    
    with col3:
        total_contracts = multiple_flows['Total_Quantity'].sum()
        st.metric("Total Contracts", f"{total_contracts:,}")
    
    with col4:
        avg_flows_per_stock = multiple_flows['Flow_Count'].mean()
        st.metric("Avg Flows per Stock", f"{avg_flows_per_stock:.1f}")
    
    # Create tabular display data
    table_data = []
    
    for _, row in multiple_flows.iterrows():
        # Determine overall sentiment based on premium flows
        contract_types = row['Contract_Types']
        side_codes = row['Side_Codes']
        
        # We need to get back to individual premium data for accurate sentiment
        # Get individual flows for this ticker and time group
        ticker_flows = high_volume_df[
            (high_volume_df['Ticker'] == row['Ticker']) & 
            (high_volume_df['Trade_Time_Rounded'] == row['Trade_Time'])
        ]
        
        # Calculate bullish vs bearish premium
        bullish_premium = 0
        bearish_premium = 0
        
        for _, flow in ticker_flows.iterrows():
            premium = flow['Premium Price']
            
            # Use smart order type for sentiment analysis
            order_type = flow.get('order_type', '')
            
            # Bullish flows: Call buying or Put selling
            if 'Calls Bought' in order_type or 'Puts Sold' in order_type:
                bullish_premium += premium
            # Bearish flows: Call selling or Put buying  
            elif 'Calls Sold' in order_type or 'Puts Bought' in order_type:
                bearish_premium += premium
        
        # Determine sentiment based on premium weight
        total_premium = bullish_premium + bearish_premium
        if total_premium > 0:
            bullish_pct = (bullish_premium / total_premium) * 100
            bearish_pct = (bearish_premium / total_premium) * 100
            
            if bullish_pct > 65:
                overall_sentiment = "� BULLISH"
            elif bearish_pct > 65:
                overall_sentiment = "🔴 BEARISH"
            else:
                overall_sentiment = "⚪ MIXED"
        else:
            overall_sentiment = "⚪ UNKNOWN"
        
        # Flags
        flags = []
        if row['Has_Unusual'] == 'YES':
            flags.append("🔥")
        if row['Has_Golden'] == 'YES':
            flags.append("⚡")
        flags_str = " ".join(flags) if flags else ""
        
        # Calculate average move required
        strike_prices = row['Strike_Prices']
        ref_price = row['Reference_Price']
        
        moves_required = []
        for strike in strike_prices:
            if ref_price and ref_price > 0:
                move_pct = abs((strike - ref_price) / ref_price) * 100
                moves_required.append(move_pct)
        
        avg_move_required = sum(moves_required) / len(moves_required) if moves_required else 0
        
        # Format trade time
        trade_time_str = row['Trade_Time'].strftime('%H:%M:%S') if pd.notna(row['Trade_Time']) else 'Unknown'
        
        # Create contract breakdown summary with individual premiums
        contract_summary = []
        ticker_flows = high_volume_df[
            (high_volume_df['Ticker'] == row['Ticker']) & 
            (high_volume_df['Trade_Time_Rounded'] == row['Trade_Time'])
        ]
        
        for _, flow in ticker_flows.iterrows():
            ct = flow['Contract Type']
            strike = flow['Strike Price']
            side = flow['Side Code']
            premium = flow['Premium Price']
            exp_date = flow['Expiration Date']
            
            # Format expiration date
            if pd.notna(exp_date):
                exp_str = pd.to_datetime(exp_date).strftime('%m/%d')
            else:
                exp_str = 'N/A'
            
            sentiment_icon = "🟢" if ((ct == 'CALL' and side in ['A', 'AA']) or (ct == 'PUT' and side in ['B', 'BB'])) else "🔴"
            # More readable format with spaces and clear separators
            contract_summary.append(f"{sentiment_icon} {ct} ${strike:.0f} ({side}) {exp_str} ${premium/1000:.0f}K")
        
        contract_breakdown = " • ".join(contract_summary)
        
        # Detect and articulate options strategies
        strategy_description = detect_options_strategy(ticker_flows)
        if strategy_description:
            contract_breakdown = f"{contract_breakdown}\n📊 **Strategy:** {strategy_description}"
        
        # Calculate average implied volatility
        avg_iv = ticker_flows['IV'].mean() if 'IV' in ticker_flows.columns and not ticker_flows['IV'].isna().all() else 0
        
        # Add premium breakdown for better analysis
        premium_breakdown = f"🟢${bullish_premium/1000:.0f}K vs 🔴${bearish_premium/1000:.0f}K"
        
        # Add to table data
        table_data.append({
            'Ticker': row['Ticker'],
            'Trade Time': trade_time_str,
            'Flows': row['Flow_Count'],
            'Total Premium': f"${row['Total_Premium']:,.0f}",
            'Premium (M)': f"${row['Total_Premium']/1000000:.1f}M",
            'Total Contracts': f"{row['Total_Quantity']:,}",
            'Avg Premium/Flow': f"${row['Avg_Premium']:,.0f}",
            'Move Required': f"{avg_move_required:.1f}%",
            'Avg IV': f"{avg_iv:.1f}%" if avg_iv > 0 else "N/A",
            'Sentiment': overall_sentiment,
            'Premium Breakdown': premium_breakdown,
            'Flags': flags_str,
            'Contract Breakdown': contract_breakdown,
            'Reference Price': f"${ref_price:.2f}" if ref_price else "N/A"
        })
    
    # Convert to DataFrame and display
    results_df = pd.DataFrame(table_data)
    
    # Display main summary table
    st.markdown("### 📊 High-Volume Flow Clusters Summary Table")
    
    # Create display table with key columns
    display_cols = ['Ticker', 'Trade Time', 'Flows', 'Premium (M)', 'Total Contracts', 
                   'Move Required', 'Avg IV', 'Sentiment', 'Premium Breakdown', 'Flags']
    
    # Style the dataframe
    styled_df = results_df[display_cols].copy()
    st.dataframe(styled_df, use_container_width=True, hide_index=True)
    
    # Detailed breakdown table
    with st.expander("� Detailed Contract Breakdown", expanded=False):
        detailed_cols = ['Ticker', 'Trade Time', 'Total Premium', 'Reference Price', 'Contract Breakdown']
        st.dataframe(results_df[detailed_cols], use_container_width=True, hide_index=True)
    
    # Create sharing summary
    with st.expander("📋 High-Volume Flow Summary for Sharing", expanded=True):
        sharing_summary = generate_high_volume_summary(results_df, min_premium, min_quantity, summary_top_n)
        
        # Streamlined format selection
        format_choice = st.selectbox("Choose Format:", 
                                   ["Professional", "Discord", "Twitter", "Table"], 
                                   key="hv_format_choice")
        
        selected_format = sharing_summary[format_choice]
        if format_choice == "Twitter":
            char_count = len(selected_format)
            st.text_area(f"Twitter Format ({char_count}/280 chars):", 
                        selected_format, height=200, key="selected_hv_summary")
            if char_count > 280:
                st.warning(f"⚠️ Tweet is {char_count - 280} characters too long!")
        else:
            st.text_area(f"{format_choice} Format:", 
                        selected_format, height=300, key="selected_hv_summary")
    
    # Raw data
    with st.expander("Raw High-Volume Flow Data"):
        display_df = multiple_flows.copy()
        # Convert list columns to strings for display
        for col in ['Contract_Types', 'Strike_Prices', 'Side_Codes']:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: ', '.join(map(str, x)) if isinstance(x, list) else str(x))
        
        st.dataframe(display_df, use_container_width=True)

def generate_high_volume_summary(results_df, min_premium, min_quantity, top_n=30):
    """Generate different formats of high-volume flow summary with strategy detection"""
    
    if results_df.empty:
        return {'Professional': 'No data found', 'Discord': 'No data found', 'Twitter': 'No data found'}
    
    total_stocks = len(results_df)
    total_premium = pd.to_numeric(results_df['Total Premium'], errors='coerce').fillna(0).sum()
    total_contracts = pd.to_numeric(results_df['Total Contracts'], errors='coerce').fillna(0).sum()
    
    # Get configurable number of top stocks by premium
    top_stocks = results_df.head(top_n)
    
    # Professional format - Clean and detailed
    professional = f"""📊 HIGH-VOLUME OPTIONS FLOW ANALYSIS
⏰ {pd.Timestamp.now().strftime('%m/%d/%Y %I:%M %p EST')}
📋 Criteria: Premium ≥ ${min_premium/1000:.0f}K | Quantity ≥ {min_quantity:,}

🎯 MARKET OVERVIEW:
• Stocks with Flow Clusters: {total_stocks}
• Total Premium Involved: ${total_premium/1000000:.1f}M
• Total Contracts: {int(total_contracts) if not pd.isna(total_contracts) else 0:,}

🏆 TOP FLOW CLUSTERS (Top {min(top_n, len(results_df))} of {total_stocks}):"""
    
    for i, (_, stock) in enumerate(top_stocks.iterrows(), 1):
        # Build the contract breakdown in your preferred format
        contract_breakdown = str(stock.get('Contract Breakdown', ''))
        strategy = str(stock.get('Strategy', ''))
        
        professional += f"""

{stock['Ticker']} - {contract_breakdown}"""
        
        if strategy and strategy != 'Multi-flow pattern':
            professional += f"""
📊 **Strategy:** {strategy}"""
        
        professional += "\n"
    
    professional += f"""

💡 MARKET IMPLICATIONS:
This coordinated high-volume activity suggests institutional positioning
and significant price movement expectations. Monitor these names for
potential breakouts or major announcements."""
    
    # Discord format - Formatted for easy reading
    discord = f"""🚨 **HIGH-VOLUME FLOW ALERT** 🚨

📊 **SCAN RESULTS** ({pd.Timestamp.now().strftime('%I:%M %p')})
🔍 Min Premium: **${min_premium/1000:.0f}K** | Min Quantity: **{min_quantity:,}**

🎯 **FOUND:** {total_stocks} stocks • **${total_premium/1000000:.1f}M** total premium

**🔥 TOP FLOWS:**"""
    
    for i, (_, stock) in enumerate(top_stocks.iterrows(), 1):
        sentiment_emoji = "🟢" if "Bullish" in str(stock['Sentiment']) else "🔴" if "Bearish" in str(stock['Sentiment']) else "🟡"
        discord += f"""
**{i}. ${stock['Ticker']}** {sentiment_emoji} `{stock['Trade Time']}`
   💰 **$0.0M** • {stock['Flows']} flows • 0 contracts
   � Move needed: **{stock['Move Required']}**"""
        
        # Add strategy if detected
        if 'Strategy' in stock and stock['Strategy'] and stock['Strategy'] != 'Multi-flow pattern':
            discord += f"""
   🧠 **{stock['Strategy']}**"""
    
    discord += f"""

🎯 **Watch these names for big moves!** 
*Institutional money is positioning...*"""
    
    # Twitter format - Concise with key info
    twitter_lines = [
        f"🚨 HIGH-VOLUME FLOW SCAN",
        f"⏰ {pd.Timestamp.now().strftime('%I:%M %p')}",
        "",
        f"🎯 {total_stocks} stocks • ${total_premium/1000000:.1f}M premium",
        f"📊 Min: ${min_premium/1000:.0f}K premium • {min_quantity:,} contracts",
        ""
    ]
    
    # Add top 2 flows for Twitter
    for i, (_, stock) in enumerate(top_stocks.head(2).iterrows(), 1):
        sentiment_emoji = "🟢" if "Bullish" in str(stock['Sentiment']) else "�" if "Bearish" in str(stock['Sentiment']) else "⚡"
        twitter_lines.append(f"{i}. ${stock['Ticker']} {sentiment_emoji} $0.0M")
    
    twitter_lines.extend([
        "",
        "#OptionsFlow #BigMoney #InstitutionalFlow"
    ])
    
    twitter = "\n".join(twitter_lines)
    
    # Table format for easy copy-paste
    table_format = f"""HIGH-VOLUME FLOW TABLE - {pd.Timestamp.now().strftime('%m/%d %I:%M %p')}
Criteria: Premium ≥ ${min_premium/1000:.0f}K | Quantity ≥ {min_quantity:,}

"""
    
    # Create clean table
    table_format += "TICKER | TIME     | PREMIUM | FLOWS | CONTRACTS | MOVE REQ | SENTIMENT\n"
    table_format += "-------|----------|---------|-------|-----------|----------|----------\n"
    
    for _, stock in top_stocks.iterrows():
        sentiment_short = str(stock['Sentiment'])[:8] + "..." if len(str(stock['Sentiment'])) > 8 else str(stock['Sentiment'])
        
        # Convert all values to safe strings
        ticker = str(stock['Ticker'])[:6]
        time_str = str(stock['Trade Time'])[-8:]
        premium_str = "0.0"  # Placeholder
        flows_str = str(stock['Flows'])
        contracts_str = "0"  # Placeholder  
        move_req_str = str(stock['Move Required'])[:8]
        
        table_format += f"{ticker:<6} | {time_str} | ${premium_str}M | {flows_str:>5} | {contracts_str:>9} | {move_req_str:>8} | {sentiment_short}\n"
    
    return {
        'Professional': professional,
        'Discord': discord,
        'Twitter': twitter,
        'Table': table_format
    }

# Add this function after the existing functions and before the main() function

def calculate_trade_score(row, ticker_daily_flow, all_premiums):
    """
    Enhanced scoring algorithm for option trades based on:
    - Premium size (30%): Normalized against market distribution
    - Flow frequency/consistency (25%): Number of sweeps and timing
    - Daily flow dominance (20%): Percentage of ticker's daily flow
    - Market impact potential (15%): Move required vs premium invested
    - Special flags and urgency (10%): Unusual activity, golden sweeps, time decay
    """
    
    # 1. Premium Score (30% weight) - Use percentile ranking
    premium_percentile = (all_premiums <= row['Total_Premium']).mean() * 100
    premium_score = min(100, premium_percentile)
    
    # 2. Flow Frequency Score (25% weight) - Reward consistency
    base_frequency_score = min(100, (row['Flow_Count'] / 20) * 100)  # 20 sweeps = 100 points
    
    # Bonus for high frequency (suggests institutional accumulation)
    if row['Flow_Count'] >= 15:
        frequency_multiplier = 1.3
    elif row['Flow_Count'] >= 10:
        frequency_multiplier = 1.15
    elif row['Flow_Count'] >= 5:
        frequency_multiplier = 1.05
    else:
        frequency_multiplier = 1.0
    
    frequency_score = min(100, base_frequency_score * frequency_multiplier)
    
    # 3. Daily Flow Dominance Score (20% weight)
    daily_flow_pct = (row['Total_Premium'] / ticker_daily_flow) * 100
    dominance_score = min(100, daily_flow_pct * 3)  # 33% of daily flow = 100 points
    
    # 4. Market Impact Potential Score (15% weight)
    # Consider move required vs premium invested (efficiency ratio)
    move_required = abs((row['Strike_Price'] - row.get('Reference_Price', row['Strike_Price'])) / 
                       row.get('Reference_Price', row['Strike_Price'])) * 100
    
    if move_required > 0:
        # Efficiency: Lower move required for higher premium = higher score
        # Sweet spot: 2-8% moves with significant premium
        if 2 <= move_required <= 8:
            impact_score = 100
        elif move_required < 2:
            impact_score = 70  # Too close to money, less upside
        elif move_required <= 15:
            impact_score = max(20, 100 - (move_required - 8) * 5)  # Penalize excessive moves
        else:
            impact_score = 20  # Very ambitious moves
    else:
        impact_score = 50  # Default if we can't calculate
    
    # 5. Special Flags and Urgency Score (10% weight)
    flags_score = 0
    
    # Premium-weighted flags (larger trades get more weight)
    premium_weight = min(2.0, row['Total_Premium'] / 1000000)  # $1M+ gets full weight
    
    if row['Has_Unusual'] == 'YES':
        flags_score += 25 * premium_weight
    if row['Has_Golden'] == 'YES':
        flags_score += 35 * premium_weight
    
    # Time decay urgency (closer expirations get bonus for immediate moves)
    try:
        # Ensure Expiration_Date is a datetime object
        if isinstance(row['Expiration_Date'], str):
            exp_date = pd.to_datetime(row['Expiration_Date'])
        else:
            exp_date = row['Expiration_Date']
        days_to_exp = (exp_date - pd.Timestamp.now()).days
        if days_to_exp <= 7:
            flags_score += 20  # Weekly expiration urgency
        elif days_to_exp <= 14:
            flags_score += 10  # Bi-weekly urgency
    except:
        pass
    
    flags_score = min(100, flags_score)
    
    # 6. Calculate weighted composite score
    composite_score = (
        premium_score * 0.30 +
        frequency_score * 0.25 +
        dominance_score * 0.20 +
        impact_score * 0.15 +
        flags_score * 0.10
    )
    
    # 7. Apply final adjustments
    # Boost score for perfect storm scenarios
    if (premium_percentile > 90 and row['Flow_Count'] >= 10 and daily_flow_pct > 25):
        composite_score = min(100, composite_score * 1.15)  # "Perfect storm" bonus
    
    # Slight penalty for extremely high move requirements (>20%)
    if move_required > 20:
        composite_score *= 0.9
    
    return round(composite_score, 1), daily_flow_pct, move_required

def display_top_trades_summary(df, min_premium=250000, min_flows=2):
    """
    Display top scoring trade for each ticker with enhanced analysis
    """
    if df is None or df.empty:
        st.warning("No data available.")
        return
    
    today = pd.to_datetime("today").normalize()
    exclude_tickers = {'SPY', 'QQQ', 'SPX', 'SPXW', 'IWM', 'NDX', 'RUT'}
    
    # Filter for relevant data
    filtered_df = df[
        (df['Premium Price'] >= min_premium) &
        (df['Is Opening Position'] == 'YES') &
        (~df['Ticker'].isin(exclude_tickers)) &
        (df['Expiration Date'] > today)
    ].copy()
    
    if filtered_df.empty:
        st.warning(f"No flows found with premium >= ${min_premium:,}")
        return
    
    # Create contract grouping key
    # Ensure Strike Price is numeric and Expiration Date is datetime
    filtered_df['Strike_Price_Clean'] = pd.to_numeric(filtered_df['Strike Price'], errors='coerce').fillna(0)
    filtered_df['Expiration_Date_Clean'] = pd.to_datetime(filtered_df['Expiration Date'], errors='coerce')
    
    filtered_df['Contract_Key'] = (
        filtered_df['Ticker'] + '_' + 
        filtered_df['Contract Type'] + '_' + 
        filtered_df['Strike_Price_Clean'].astype(str) + '_' + 
        filtered_df['Expiration_Date_Clean'].dt.strftime('%Y-%m-%d')
    )
    
    # Group by ticker and contract to get sweep counts
    contract_groups = filtered_df.groupby(['Ticker', 'Contract_Key']).agg({
        'Premium Price': ['count', 'sum'],
        'Size': 'sum',
        'Contract Type': 'first',
        'Strike_Price_Clean': 'first',  # Use cleaned version
        'Expiration_Date_Clean': 'first',  # Use cleaned version
        'Reference Price': 'first',
        'Side Code': lambda x: list(x),
        'Is Unusual': lambda x: 'YES' if 'YES' in x.values else 'NO',
        'Is Golden Sweep': lambda x: 'YES' if 'YES' in x.values else 'NO'
    }).reset_index()
    
    # Flatten column names
    contract_groups.columns = ['Ticker', 'Contract_Key', 'Flow_Count', 'Total_Premium', 
                              'Total_Contracts', 'Contract_Type', 'Strike_Price', 
                              'Expiration_Date', 'Reference_Price', 'Side_Codes', 
                              'Has_Unusual', 'Has_Golden']
    
    # Filter for minimum number of flows
    contract_groups = contract_groups[contract_groups['Flow_Count'] >= min_flows]
    
    if contract_groups.empty:
        st.warning(f"No contracts found with >= {min_flows} flows")
        return
    
    # Calculate daily flow by ticker and get all premiums for percentile calculation
    ticker_daily_flows = filtered_df.groupby('Ticker')['Premium Price'].sum()
    all_premiums = contract_groups['Total_Premium'].values
    
    # Calculate enhanced scores for each contract
    scores_data = []
    for _, row in contract_groups.iterrows():
        ticker_daily_flow = ticker_daily_flows.get(row['Ticker'], 1)  # Avoid division by zero
        score, daily_flow_pct, move_required = calculate_trade_score(row, ticker_daily_flow, all_premiums)
        
        # Enhanced sentiment analysis
        side_codes = row['Side_Codes']
        bullish_sides = sum(1 for side in side_codes if side in ['A', 'AA'])
        bearish_sides = sum(1 for side in side_codes if side in ['B', 'BB'])
        
        # Determine sentiment with confidence level
        if row['Contract_Type'] == 'CALL':
            if bullish_sides > bearish_sides:
                sentiment = "📈 BULLISH"
                confidence = "High" if bullish_sides / len(side_codes) > 0.75 else "Moderate"
            elif bearish_sides > bullish_sides:
                sentiment = "📉 BEARISH"
                confidence = "High" if bearish_sides / len(side_codes) > 0.75 else "Moderate"
            else:
                sentiment = "⚪ MIXED"
                confidence = "Low"
        else:  # PUT
            if bullish_sides > bearish_sides:
                sentiment = "📉 BEARISH"  # Put buying is bearish
                confidence = "High" if bullish_sides / len(side_codes) > 0.75 else "Moderate"
            elif bearish_sides > bullish_sides:
                sentiment = "📈 BULLISH"  # Put selling is bullish
                confidence = "High" if bearish_sides / len(side_codes) > 0.75 else "Moderate"
            else:
                sentiment = "⚪ MIXED"
                confidence = "Low"
        
        # Calculate days to expiration
        # Calculate days to expiration
        if isinstance(row['Expiration_Date'], str):
            exp_date = pd.to_datetime(row['Expiration_Date'])
        else:
            exp_date = row['Expiration_Date']
        days_to_exp = (exp_date - today).days
        
        scores_data.append({
            'Ticker': row['Ticker'],
            'Score': score,
            'Contract_Type': row['Contract_Type'],
            'Strike_Price': row['Strike_Price'],
            'Reference_Price': row['Reference_Price'],
            'Expiration_Date': row['Expiration_Date'],
            'Days_to_Exp': days_to_exp,
            'Flow_Count': row['Flow_Count'],
            'Total_Premium': row['Total_Premium'],
            'Daily_Flow_Pct': daily_flow_pct,
            'Move_Required': move_required,
            'Sentiment': sentiment,
            'Confidence': confidence,
            'Has_Unusual': row['Has_Unusual'],
            'Has_Golden': row['Has_Golden']
        })
    
    if not scores_data:
        st.warning("No qualifying trades found for scoring.")
        return
    
    # Convert to DataFrame and get top trade per ticker
    scores_df = pd.DataFrame(scores_data)
    top_trades = scores_df.loc[scores_df.groupby('Ticker')['Score'].idxmax()].sort_values('Score', ascending=False)
    
    # Enhanced summary header with market analysis
    st.subheader("🏆 TOP TRADES SUMMARY - AI-Enhanced Scoring Algorithm")
    
    # Market overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_tickers = len(top_trades)
        st.metric("📊 Unique Tickers", total_tickers)
    
    with col2:
        total_premium = top_trades['Total_Premium'].sum()
        st.metric("💰 Total Premium", f"${total_premium/1000000:.1f}M")
    
    with col3:
        avg_score = top_trades['Score'].mean()
        score_quality = "Excellent" if avg_score > 80 else "Good" if avg_score > 65 else "Fair"
        st.metric("⭐ Avg Quality Score", f"{avg_score:.1f}", delta=score_quality)
    
    with col4:
        bullish_count = sum(1 for sentiment in top_trades['Sentiment'] if 'BULLISH' in sentiment)
        bearish_count = sum(1 for sentiment in top_trades['Sentiment'] if 'BEARISH' in sentiment)
        market_bias = "Bullish" if bullish_count > bearish_count else "Bearish" if bearish_count > bullish_count else "Mixed"
        st.metric("📈 Market Bias", market_bias, delta=f"{bullish_count}B/{bearish_count}B")
    
    st.markdown("---")
    
    # Tiered display based on score ranges
    st.markdown("### 🌟 TIER 1: PREMIUM OPPORTUNITIES (Score 80+)")
    tier1_trades = top_trades[top_trades['Score'] >= 80]
    if not tier1_trades.empty:
        display_trade_tier(tier1_trades, "🔥")
    else:
        st.markdown("*No Tier 1 opportunities found*")
    
    st.markdown("### ⭐ TIER 2: STRONG OPPORTUNITIES (Score 65-79)")
    tier2_trades = top_trades[(top_trades['Score'] >= 65) & (top_trades['Score'] < 80)]
    if not tier2_trades.empty:
        display_trade_tier(tier2_trades, "💪")
    else:
        st.markdown("*No Tier 2 opportunities found*")
    
    st.markdown("### 📊 TIER 3: MODERATE OPPORTUNITIES (Score 50-64)")
    tier3_trades = top_trades[(top_trades['Score'] >= 50) & (top_trades['Score'] < 65)]
    if not tier3_trades.empty:
        display_trade_tier(tier3_trades, "👀")
    else:
        st.markdown("*No Tier 3 opportunities found*")
    
    # Show lower scored trades in expandable section
    lower_trades = top_trades[top_trades['Score'] < 50]
    if not lower_trades.empty:
        with st.expander(f"📋 Lower Scored Trades ({len(lower_trades)} tickers)", expanded=False):
            display_trade_tier(lower_trades, "⚠️")
    
    # Enhanced copy-to-clipboard functionality with multiple formats
    with st.expander("📋 Copy Professional Summary", expanded=True):
        summary_formats = generate_professional_summary(top_trades)
        
        # Simple dropdown instead of tabs
        format_type = st.selectbox("Select Format:", 
                                 ["Full Report", "Discord", "Twitter", "Quick Summary"], 
                                 key="summary_format_choice")
        
        selected_summary = summary_formats[format_type]
        
        if format_type == "Twitter":
            char_count = len(selected_summary)
            st.text_area(f"Twitter Format ({char_count}/280 chars):", 
                        selected_summary, height=150, key="selected_summary")
            if char_count > 280:
                st.warning(f"⚠️ Tweet is {char_count - 280} characters too long!")
        else:
            height = 500 if format_type == "Full Report" else 200
            st.text_area(f"{format_type}:", selected_summary, height=height, key="selected_summary")
    
    # Enhanced scoring methodology
    with st.expander("🧠 Enhanced AI Scoring Methodology"):
        st.markdown("""
        **🚀 New Enhanced Scoring Algorithm (5 Components):**
        
        **1. Premium Percentile (30%)** 📊
        - Uses market distribution percentiles instead of fixed thresholds
        - Adapts to current market conditions automatically
        - 90th percentile+ = Premium opportunities
        
        **2. Flow Consistency (25%)** 🔄
        - Rewards multiple sweeps (institutional accumulation pattern)
        - Frequency multipliers: 15+ sweeps (+30%), 10+ sweeps (+15%)
        - Consistent flow indicates conviction
        
        **3. Daily Flow Dominance (20%)** 💪
        - Percentage of ticker's total daily options flow
        - 33%+ dominance = Maximum score
        - Shows relative importance vs other activity
        
        **4. Market Impact Efficiency (15%)** 🎯
        - Move required vs premium invested ratio
        - Sweet spot: 2-8% moves = Optimal efficiency
        - Penalizes excessive moves (>20%)
        
        **5. Special Signals & Urgency (10%)** ⚡
        - Premium-weighted unusual activity detection
        - Golden sweep identification with size consideration
        - Time decay urgency (weekly expirations get bonus)
        
        **🎖️ Quality Tiers:**
        - **Tier 1 (80+)**: Premium opportunities with multiple strong signals
        - **Tier 2 (65-79)**: Strong opportunities with good conviction
        - **Tier 3 (50-64)**: Moderate opportunities worth monitoring
        
        **🔥 Perfect Storm Bonus (+15%):**
        - Premium >90th percentile + 10+ sweeps + 25%+ daily flow dominance
        """)

def display_trade_tier(trades_df, emoji):
    """Helper function to display trades in a consistent format"""
    for _, trade in trades_df.iterrows():
        # Format premium
        if trade['Total_Premium'] >= 1000000:
            premium_str = f"${trade['Total_Premium']/1000000:.2f}M"
        else:
            premium_str = f"${trade['Total_Premium']/1000:.0f}K"
        
        # Special flags with enhanced display
        flags = []
        if trade['Has_Unusual'] == 'YES':
            flags.append("🔥 UNUSUAL")
        if trade['Has_Golden'] == 'YES':
            flags.append("⚡ GOLDEN")
        flags_str = f" [{' '.join(flags)}]" if flags else ""
        
        # Time urgency indicator
        if trade['Days_to_Exp'] <= 7:
            urgency = "🚨 WEEKLY"
        elif trade['Days_to_Exp'] <= 14:
            urgency = "⏰ 2-WEEK"
        else:
            urgency = f"{trade['Days_to_Exp']}d"
        
        # Confidence indicator
        conf_indicator = "🎯" if trade['Confidence'] == "High" else "📊" if trade['Confidence'] == "Moderate" else "❓"
        
        line = (f"{emoji} **${trade['Ticker']}** {trade['Sentiment']} {conf_indicator} "
                f"${trade['Strike_Price']:,.0f} {trade['Contract_Type']} "
                f"({trade['Flow_Count']} sweeps) → {premium_str} "
                f"[{trade['Move_Required']:.1f}% move] [{urgency}] "
                f"**Score: {trade['Score']:.1f}**{flags_str}")
        
        st.markdown(line)

def generate_professional_summary(top_trades):
    """Generate a comprehensive professional summary for copying to Discord/Twitter"""
    timestamp = pd.Timestamp.now().strftime('%B %d, %Y')
    
    # Create multiple format options
    formats = {}
    
    # 1. FULL COMPREHENSIVE REPORT
    full_summary = f"📊 OPTIONS FLOW INTELLIGENCE REPORT - {timestamp}\n"
    full_summary += "="*80 + "\n\n"
    
    # Market overview
    total_premium = top_trades['Total_Premium'].sum()
    avg_score = top_trades['Score'].mean()
    bullish_count = sum(1 for sentiment in top_trades['Sentiment'] if 'BULLISH' in sentiment)
    bearish_count = sum(1 for sentiment in top_trades['Sentiment'] if 'BEARISH' in sentiment)
    mixed_count = len(top_trades) - bullish_count - bearish_count
    
    full_summary += f"MARKET OVERVIEW:\n"
    full_summary += f"• Total Premium Analyzed: ${total_premium/1000000:.1f}M across {len(top_trades)} tickers\n"
    full_summary += f"• Average Quality Score: {avg_score:.1f}/100\n"
    full_summary += f"• Directional Bias: {bullish_count} Bullish, {bearish_count} Bearish, {mixed_count} Mixed\n\n"
    
    # All tiers with comprehensive data
    tier1 = top_trades[top_trades['Score'] >= 80]
    tier2 = top_trades[(top_trades['Score'] >= 65) & (top_trades['Score'] < 80)]
    tier3 = top_trades[(top_trades['Score'] >= 50) & (top_trades['Score'] < 65)]
    lower_trades = top_trades[top_trades['Score'] < 50]
    
    def format_trade_line(trade, include_score=True):
        premium_str = f"${trade['Total_Premium']/1000000:.2f}M" if trade['Total_Premium'] >= 1000000 else f"${trade['Total_Premium']/1000:.0f}K"
        flags = []
        if trade['Has_Unusual'] == 'YES':
            flags.append("UNUSUAL")
        if trade['Has_Golden'] == 'YES':
            flags.append("GOLDEN")
        flags_str = f" [{' '.join(flags)}]" if flags else ""
        
        score_str = f" [Score: {trade['Score']:.1f}]" if include_score else ""
        
        return (f"  • ${trade['Ticker']} {trade['Sentiment'].split()[1]} "
                f"${trade['Strike_Price']:,.0f} {trade['Contract_Type']} "
                f"({trade['Flow_Count']} sweeps) → {premium_str} "
                f"[{trade['Move_Required']:.1f}% move]{score_str}{flags_str}")
    
    # Add all tiers
    if not tier1.empty:
        full_summary += f"🌟 TIER 1 OPPORTUNITIES (Score 80+) - {len(tier1)} trades:\n"
        for _, trade in tier1.iterrows():
            full_summary += format_trade_line(trade) + "\n"
        full_summary += "\n"
    
    if not tier2.empty:
        full_summary += f"⭐ TIER 2 OPPORTUNITIES (Score 65-79) - {len(tier2)} trades:\n"
        for _, trade in tier2.iterrows():
            full_summary += format_trade_line(trade) + "\n"
        full_summary += "\n"
    
    if not tier3.empty:
        full_summary += f"📊 TIER 3 OPPORTUNITIES (Score 50-64) - {len(tier3)} trades:\n"
        for _, trade in tier3.iterrows():
            full_summary += format_trade_line(trade) + "\n"
        full_summary += "\n"
    
    if not lower_trades.empty:
        full_summary += f"📋 ADDITIONAL FLOWS (Score <50) - {len(lower_trades)} trades:\n"
        for _, trade in lower_trades.iterrows():
            full_summary += format_trade_line(trade) + "\n"
        full_summary += "\n"
    
    full_summary += "="*80 + "\n"
    full_summary += "Generated by Enhanced AI Options Flow Analysis\n"
    full_summary += "⚠️ Not financial advice."
    
    formats['Full Report'] = full_summary
    
    # 2. DISCORD OPTIMIZED (Under 2000 chars per message)
    discord_summary = f"📊 **OPTIONS FLOW REPORT** - {timestamp}\n\n"
    discord_summary += f"💰 **${total_premium/1000000:.1f}M** total premium | **{len(top_trades)}** tickers | Avg Score: **{avg_score:.1f}**\n"
    discord_summary += f"📈 **{bullish_count}** Bullish | 📉 **{bearish_count}** Bearish | ⚪ **{mixed_count}** Mixed\n\n"
    
    # Top opportunities only for Discord (space constraints)
    top_opportunities = top_trades.head(20)  # Top 20 trades
    
    discord_summary += "**🏆 TOP OPPORTUNITIES:**\n"
    for _, trade in top_opportunities.iterrows():
        premium_str = f"${trade['Total_Premium']/1000000:.1f}M" if trade['Total_Premium'] >= 1000000 else f"${trade['Total_Premium']/1000:.0f}K"
        
        # Use emojis for sentiment
        sentiment_emoji = "📈" if "BULLISH" in trade['Sentiment'] else "📉" if "BEARISH" in trade['Sentiment'] else "⚪"
        
        # Flags as emojis
        flags_emoji = ""
        if trade['Has_Unusual'] == 'YES':
            flags_emoji += "🔥"
        if trade['Has_Golden'] == 'YES':
            flags_emoji += "⚡"
        
        discord_summary += (f"{sentiment_emoji} **${trade['Ticker']}** "
                          f"${trade['Strike_Price']:,.0f} {trade['Contract_Type']} "
                          f"({trade['Flow_Count']}x) {premium_str} [{trade['Score']:.0f}] {flags_emoji}\n")
    
    discord_summary += f"\n⚠️ Educational only. Not financial advice."
    
    formats['Discord'] = discord_summary
    
    # 3. TWITTER OPTIMIZED (Under 280 chars)
    twitter_summary = f"📊 OPTIONS FLOW ALERT {timestamp.split(',')[0]}\n\n"
    twitter_summary += f"💰${total_premium/1000000:.1f}M across {len(top_trades)} tickers\n"
    twitter_summary += f"📈{bullish_count} Bullish 📉{bearish_count} Bearish\n\n"
    
    # Top 5 for Twitter
    twitter_summary += "🏆 TOP PLAYS:\n"
    for i, (_, trade) in enumerate(top_trades.head(5).iterrows(), 1):
        premium_str = f"${trade['Total_Premium']/1000000:.1f}M" if trade['Total_Premium'] >= 1000000 else f"${trade['Total_Premium']/1000:.0f}K"
        sentiment_emoji = "📈" if "BULLISH" in trade['Sentiment'] else "📉"
        flags_emoji = "🔥" if trade['Has_Unusual'] == 'YES' else ""
        
        twitter_summary += f"{i}. {sentiment_emoji} ${trade['Ticker']} ${trade['Strike_Price']:,.0f} {premium_str} {flags_emoji}\n"
    
    twitter_summary += "\n#OptionsFlow #Trading"
    
    formats['Twitter'] = twitter_summary
    
    # 4. QUICK SUMMARY (One-liner style)
    quick_summary = f"📊 {timestamp}: ${total_premium/1000000:.1f}M flow across {len(top_trades)} tickers | "
    quick_summary += f"{bullish_count}📈 {bearish_count}📉 | TOP: "
    
    for _, trade in top_trades.head(3).iterrows():
        sentiment_emoji = "📈" if "BULLISH" in trade['Sentiment'] else "📉"
        premium_str = f"${trade['Total_Premium']/1000000:.1f}M" if trade['Total_Premium'] >= 1000000 else f"${trade['Total_Premium']/1000:.0f}K"
        quick_summary += f"{sentiment_emoji}${trade['Ticker']} {premium_str} | "
    
    quick_summary = quick_summary.rstrip(" | ")
    
    formats['Quick Summary'] = quick_summary
    
    return formats

# Function to analyze symbol flows and project path
def analyze_symbol_flows(df, symbols, min_premium=100000):
    """
    Analyze option flows for specific symbols and project price path
    """
    if df is None or df.empty:
        st.warning("No data available.")
        return
    
    today = pd.to_datetime("today").normalize()
    next_friday = today + timedelta(days=(4 - today.weekday()) % 7)  # Next Friday
    if next_friday <= today:
        next_friday += timedelta(days=7)
    
    for symbol in symbols:
        symbol = symbol.upper().strip()
        
        # Filter flows for this symbol
        symbol_df = df[
            (df['Ticker'] == symbol) &
            (df['Is Opening Position'] == 'YES') &
            (df['Premium Price'] >= min_premium) &
            (df['Expiration Date'] > today)
        ].copy()
        
        if symbol_df.empty:
            st.warning(f"No qualifying flows found for {symbol} (Premium >= ${min_premium:,})")
            continue
        
        st.subheader(f"📊 {symbol} Flow Analysis & Projected Path")
        
        # Get current reference price (assuming it's the most recent)
        current_price = symbol_df['Reference Price'].iloc[0]
        
        # Separate calls and puts
        calls_df = symbol_df[symbol_df['Contract Type'] == 'CALL']
        puts_df = symbol_df[symbol_df['Contract Type'] == 'PUT']
        
        # Calculate bullish vs bearish premium (unchanged)
        bullish_premium = 0
        bearish_premium = 0
        
        for _, row in symbol_df.iterrows():
            if row['Contract Type'] == 'CALL':
                if row['Side Code'] in ['A', 'AA']:
                    bullish_premium += row['Premium Price']
                elif row['Side Code'] in ['B', 'BB']:
                    bearish_premium += row['Premium Price']
            else:  # PUT
                if row['Side Code'] in ['A', 'AA']:
                    bearish_premium += row['Premium Price']
                elif row['Side Code'] in ['B', 'BB']:
                    bullish_premium += row['Premium Price']
        
        total_premium = bullish_premium + bearish_premium
        if total_premium > 0:
            bullish_pct = (bullish_premium / total_premium) * 100
            bearish_pct = (bearish_premium / total_premium) * 100
        else:
            bullish_pct = bearish_pct = 0
        
        # Overall sentiment (unchanged)
        if bullish_pct > 60:
            overall_sentiment = "🟢 BULLISH"
            sentiment_strength = "Strong" if bullish_pct > 75 else "Moderate"
        elif bearish_pct > 60:
            overall_sentiment = "🔴 BEARISH"
            sentiment_strength = "Strong" if bearish_pct > 75 else "Moderate"
        else:
            overall_sentiment = "🟡 NEUTRAL/MIXED"
            sentiment_strength = "Balanced"
        
        # Display current status (unchanged)
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Current Price", f"${current_price:.2f}")
        
        with col2:
            st.metric("Overall Sentiment", overall_sentiment.split()[1], delta=f"{sentiment_strength}")
        
        with col3:
            st.metric("Bullish Premium", f"${bullish_premium/1000000:.1f}M", delta=f"{bullish_pct:.1f}%")
        
        with col4:
            st.metric("Bearish Premium", f"${bearish_premium/1000000:.1f}M", delta=f"{bearish_pct:.1f}%")
        
        # Weekly positioning breakdown
        st.markdown("### 📅 Weekly Options Positioning")
        
        # Group flows by expiration week
        symbol_df['Week_End'] = symbol_df['Expiration Date'].dt.to_period('W').dt.end_time.dt.date
        weekly_groups = symbol_df.groupby('Week_End')
        
        # Display weekly breakdown
        for week_end, week_data in weekly_groups:
            week_total_premium = week_data['Premium Price'].sum()
            week_str = week_end.strftime('%Y-%m-%d')
            
            with st.expander(f"📊 Week of {week_str} - ${week_total_premium/1000000:.1f}M Total Premium", expanded=True):
                
                # Separate by contract type and side
                call_buys = week_data[(week_data['Contract Type'] == 'CALL') & (week_data['Side Code'].isin(['A', 'AA']))]
                call_sells = week_data[(week_data['Contract Type'] == 'CALL') & (week_data['Side Code'].isin(['B', 'BB']))]
                put_buys = week_data[(week_data['Contract Type'] == 'PUT') & (week_data['Side Code'].isin(['A', 'AA']))]
                put_sells = week_data[(week_data['Contract Type'] == 'PUT') & (week_data['Side Code'].isin(['B', 'BB']))]
                
                # Create 4 columns for the breakdown
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown("**🟢 CALL BUYS (Bullish)**")
                    if not call_buys.empty:
                        call_buy_premium = call_buys['Premium Price'].sum()
                        call_buy_contracts = call_buys['Size'].sum()
                        st.metric("Premium", f"${call_buy_premium/1000000:.1f}M")
                        st.metric("Contracts", f"{call_buy_contracts:,}")
                        
                        # Show top strikes
                        top_call_strikes = call_buys.groupby('Strike Price')['Premium Price'].sum().sort_values(ascending=False).head(3)
                        for strike, premium in top_call_strikes.items():
                            move_pct = ((strike - current_price) / current_price) * 100
                            st.markdown(f"• ${strike:.0f} ({move_pct:+.1f}%): ${premium/1000:.0f}K")
                    else:
                        st.markdown("*No call buying*")
                
                with col2:
                    st.markdown("**🔴 CALL SELLS (Bearish)**")
                    if not call_sells.empty:
                        call_sell_premium = call_sells['Premium Price'].sum()
                        call_sell_contracts = call_sells['Size'].sum()
                        st.metric("Premium", f"${call_sell_premium/1000000:.1f}M")
                        st.metric("Contracts", f"{call_sell_contracts:,}")
                        
                        # Show top strikes
                        top_call_sell_strikes = call_sells.groupby('Strike Price')['Premium Price'].sum().sort_values(ascending=False).head(3)
                        for strike, premium in top_call_sell_strikes.items():
                            move_pct = ((strike - current_price) / current_price) * 100
                            st.markdown(f"• ${strike:.0f} ({move_pct:+.1f}%): ${premium/1000:.0f}K")
                    else:
                        st.markdown("*No call selling*")
                
                with col3:
                    st.markdown("**🔴 PUT BUYS (Bearish)**")
                    if not put_buys.empty:
                        put_buy_premium = put_buys['Premium Price'].sum()
                        put_buy_contracts = put_buys['Size'].sum()
                        st.metric("Premium", f"${put_buy_premium/1000000:.1f}M")
                        st.metric("Contracts", f"{put_buy_contracts:,}")
                        
                        # Show top strikes
                        top_put_strikes = put_buys.groupby('Strike Price')['Premium Price'].sum().sort_values(ascending=False).head(3)
                        for strike, premium in top_put_strikes.items():
                            move_pct = ((strike - current_price) / current_price) * 100
                            st.markdown(f"• ${strike:.0f} ({move_pct:+.1f}%): ${premium/1000:.0f}K")
                    else:
                        st.markdown("*No put buying*")
                
                with col4:
                    st.markdown("**🟢 PUT SELLS (Bullish)**")
                    if not put_sells.empty:
                        put_sell_premium = put_sells['Premium Price'].sum()
                        put_sell_contracts = put_sells['Size'].sum()
                        st.metric("Premium", f"${put_sell_premium/1000000:.1f}M")
                        st.metric("Contracts", f"{put_sell_contracts:,}")
                        
                        # Show top strikes
                        top_put_sell_strikes = put_sells.groupby('Strike Price')['Premium Price'].sum().sort_values(ascending=False).head(3)
                        for strike, premium in top_put_sell_strikes.items():
                            move_pct = ((strike - current_price) / current_price) * 100
                            st.markdown(f"• ${strike:.0f} ({move_pct:+.1f}%): ${premium/1000:.0f}K")
                    else:
                        st.markdown("*No put selling*")
                
                # Week summary
                bullish_flow = (call_buys['Premium Price'].sum() if not call_buys.empty else 0) + \
                              (put_sells['Premium Price'].sum() if not put_sells.empty else 0)
                bearish_flow = (call_sells['Premium Price'].sum() if not call_sells.empty else 0) + \
                              (put_buys['Premium Price'].sum() if not put_buys.empty else 0)
                
                if bullish_flow + bearish_flow > 0:
                    bullish_pct_week = (bullish_flow / (bullish_flow + bearish_flow)) * 100
                    bearish_pct_week = (bearish_flow / (bullish_flow + bearish_flow)) * 100
                    
                    if bullish_pct_week > 60:
                        week_bias = f"🟢 **BULLISH** ({bullish_pct_week:.0f}%)"
                    elif bearish_pct_week > 60:
                        week_bias = f"🔴 **BEARISH** ({bearish_pct_week:.0f}%)"
                    else:
                        week_bias = f"🟡 **MIXED** (Bull: {bullish_pct_week:.0f}%, Bear: {bearish_pct_week:.0f}%)"
                    
                    st.markdown(f"**Week Bias**: {week_bias}")
                
                st.divider()
        
        # Overall positioning summary
        with st.expander("📈 Overall Positioning Summary"):
            # Total flows by type
            total_call_buys = symbol_df[(symbol_df['Contract Type'] == 'CALL') & (symbol_df['Side Code'].isin(['A', 'AA']))]['Premium Price'].sum()
            total_call_sells = symbol_df[(symbol_df['Contract Type'] == 'CALL') & (symbol_df['Side Code'].isin(['B', 'BB']))]['Premium Price'].sum()
            total_put_buys = symbol_df[(symbol_df['Contract Type'] == 'PUT') & (symbol_df['Side Code'].isin(['A', 'AA']))]['Premium Price'].sum()
            total_put_sells = symbol_df[(symbol_df['Contract Type'] == 'PUT') & (symbol_df['Side Code'].isin(['B', 'BB']))]['Premium Price'].sum()
            
            st.markdown("**Flow Type Breakdown:**")
            st.markdown(f"- 🟢 **Call Buying**: ${total_call_buys/1000000:.1f}M ({total_call_buys/total_premium*100:.0f}%)")
            st.markdown(f"- 🔴 **Call Selling**: ${total_call_sells/1000000:.1f}M ({total_call_sells/total_premium*100:.0f}%)")
            st.markdown(f"- 🔴 **Put Buying**: ${total_put_buys/1000000:.1f}M ({total_put_buys/total_premium*100:.0f}%)")
            st.markdown(f"- 🟢 **Put Selling**: ${total_put_sells/1000000:.1f}M ({total_put_sells/total_premium*100:.0f}%)")
            
            # Net positioning
            net_bullish = total_call_buys + total_put_sells
            net_bearish = total_call_sells + total_put_buys
            st.markdown(f"\n**Net Positioning:**")
            st.markdown(f"- 🟢 **Net Bullish Flows**: ${net_bullish/1000000:.1f}M")
            st.markdown(f"- 🔴 **Net Bearish Flows**: ${net_bearish/1000000:.1f}M")
        
        # Show detailed flows (unchanged)
        with st.expander(f"📊 Detailed Flows for {symbol}"):
            display_df = symbol_df.sort_values('Premium Price', ascending=False)
            
            for _, row in display_df.head(20).iterrows():
                move_pct = abs((row['Strike Price'] - row['Reference Price']) / 
                              row['Reference Price'] * 100)
                
                if row['Contract Type'] == 'CALL':
                    if row['Side Code'] in ['A', 'AA']:
                        flow_sentiment = "🟢 Bullish"
                    else:
                        flow_sentiment = "🔴 Bearish"
                else:  # PUT
                    if row['Side Code'] in ['A', 'AA']:
                        flow_sentiment = "🔴 Bearish"
                    else:
                        flow_sentiment = "🟢 Bullish"
                
                flags = []
                if row['Is Unusual'] == 'YES':
                    flags.append("🔥 UNUSUAL")
                if row['Is Golden Sweep'] == 'YES':
                    flags.append("⚡ GOLDEN")
                flags_str = f" [{' '.join(flags)}]" if flags else ""
                
                st.markdown(f"""
                **{row['Contract Type']} ${row['Strike Price']:,.2f}** exp {row['Expiration Date'].strftime('%Y-%m-%d')}
                - Premium: ${row['Premium Price']:,.0f} | Contracts: {row['Size']:,} | {flow_sentiment}
                - Move Required: {move_pct:.1f}% | Side: {row['Side Code']}{flags_str}
                """)
        
        st.divider()

def generate_repeat_flows_summary(repeat_flows):
    """Generate sharing-friendly summaries of repeat flows analysis"""
    if repeat_flows.empty:
        return {
            'Professional': "No repeat flows found with the specified criteria.",
            'Social': "No repeat flows found.",
            'Quick': "No repeat flows."
        }
    
    timestamp = pd.Timestamp.now().strftime('%B %d, %Y')
    
    # Calculate key metrics
    total_contracts = len(repeat_flows)
    total_premium = repeat_flows['Total_Premium'].sum()
    avg_flows_per_contract = repeat_flows['Flow_Count'].mean()
    top_ticker = repeat_flows.loc[repeat_flows['Total_Premium'].idxmax(), 'Ticker']
    top_premium = repeat_flows['Total_Premium'].max()
    
    # Count sentiment distribution
    bullish_count = 0
    bearish_count = 0
    mixed_count = 0
    
    for _, row in repeat_flows.iterrows():
        side_codes = row['Side_Codes']
        bullish_sides = sum(1 for side in side_codes if side in ['A', 'AA'])
        bearish_sides = sum(1 for side in side_codes if side in ['B', 'BB'])
        
        if row['Contract_Type'] == 'CALL':
            if bullish_sides > bearish_sides:
                bullish_count += 1
            elif bearish_sides > bullish_sides:
                bearish_count += 1
            else:
                mixed_count += 1
        else:  # PUT
            if bullish_sides > bearish_sides:
                bearish_count += 1  # Put buying is bearish
            elif bearish_sides > bullish_sides:
                bullish_count += 1  # Put selling is bullish
            else:
                mixed_count += 1
    
    # Sort by total premium for display
    top_repeat_flows = repeat_flows.sort_values('Total_Premium', ascending=False)
    
    # 1. PROFESSIONAL SUMMARY
    professional_summary = f"📊 REPEAT FLOWS ANALYSIS REPORT - {timestamp}\n"
    professional_summary += "="*70 + "\n\n"
    
    professional_summary += "EXECUTIVE SUMMARY:\n"
    professional_summary += f"• Total Repeat Contracts: {total_contracts}\n"
    professional_summary += f"• Total Premium Volume: ${total_premium/1000000:.2f}M\n"
    professional_summary += f"• Average Sweeps per Contract: {avg_flows_per_contract:.1f}\n"
    professional_summary += f"• Sentiment Distribution: {bullish_count} Bullish, {bearish_count} Bearish, {mixed_count} Mixed\n"
    professional_summary += f"• Top Activity: ${top_ticker} (${top_premium/1000000:.2f}M)\n\n"
    
    professional_summary += "KEY INSTITUTIONAL ACCUMULATION PATTERNS:\n"
    professional_summary += "-" * 50 + "\n"
    
    # Group by ticker for professional display
    for ticker in top_repeat_flows['Ticker'].unique()[:10]:  # Top 10 tickers
        ticker_flows = top_repeat_flows[top_repeat_flows['Ticker'] == ticker]
        ticker_total = ticker_flows['Total_Premium'].sum()
        
        professional_summary += f"\n{ticker} - ${ticker_total/1000000:.2f}M total premium ({len(ticker_flows)} contracts)\n"
        
        for _, row in ticker_flows.head(3).iterrows():  # Top 3 contracts per ticker
            move_pct = abs((row['Strike_Price'] - row['Reference_Price']) / row['Reference_Price'] * 100)
            
            # Determine sentiment
            side_codes = row['Side_Codes']
            bullish_sides = sum(1 for side in side_codes if side in ['A', 'AA'])
            bearish_sides = sum(1 for side in side_codes if side in ['B', 'BB'])
            
            if row['Contract_Type'] == 'CALL':
                sentiment = "BULLISH" if bullish_sides > bearish_sides else "BEARISH" if bearish_sides > bullish_sides else "MIXED"
            else:
                sentiment = "BEARISH" if bullish_sides > bearish_sides else "BULLISH" if bearish_sides > bullish_sides else "MIXED"
            
            flags = []
            if row['Has_Unusual'] == 'YES':
                flags.append("UNUSUAL")
            if row['Has_Golden'] == 'YES':
                flags.append("GOLDEN")
            flags_str = f" [{' '.join(flags)}]" if flags else ""
            
            exp_date_str = row['Expiration_Date'].strftime('%Y-%m-%d') if hasattr(row['Expiration_Date'], 'strftime') else str(row['Expiration_Date'])
            professional_summary += (f"  • {row['Contract_Type']} ${row['Strike_Price']:,.0f} "
                                   f"exp {exp_date_str} - "
                                   f"{row['Flow_Count']} sweeps, ${row['Total_Premium']/1000000:.2f}M, "
                                   f"{move_pct:.1f}% move, {sentiment}{flags_str}\n")
    
    professional_summary += "\n" + "="*70 + "\n"
    professional_summary += "Analysis shows repeated institutional interest in specific strike prices,\n"
    professional_summary += "suggesting potential accumulation or hedging activity.\n"
    professional_summary += "⚠️ For educational purposes only. Not financial advice."
    
    # 2. SOCIAL MEDIA SUMMARY (Discord/Slack optimized)
    social_summary = f"🔄 **REPEAT FLOWS ALERT** - {timestamp.split(',')[0]}\n\n"
    social_summary += f"📊 **{total_contracts}** contracts with multiple sweeps\n"
    social_summary += f"💰 **${total_premium/1000000:.1f}M** total premium\n"
    social_summary += f"📈 **{bullish_count}** Bullish | 📉 **{bearish_count}** Bearish | ⚪ **{mixed_count}** Mixed\n\n"
    
    social_summary += "**🏆 TOP ACCUMULATION PATTERNS:**\n"
    
    for i, (_, row) in enumerate(top_repeat_flows.head(15).iterrows(), 1):
        # Determine sentiment emoji
        side_codes = row['Side_Codes']
        bullish_sides = sum(1 for side in side_codes if side in ['A', 'AA'])
        bearish_sides = sum(1 for side in side_codes if side in ['B', 'BB'])
        
        if row['Contract_Type'] == 'CALL':
            sentiment_emoji = "📈" if bullish_sides > bearish_sides else "📉" if bearish_sides > bullish_sides else "⚪"
        else:
            sentiment_emoji = "📉" if bullish_sides > bearish_sides else "📈" if bearish_sides > bullish_sides else "⚪"
        
        # Premium display
        premium_str = f"${row['Total_Premium']/1000000:.1f}M" if row['Total_Premium'] >= 1000000 else f"${row['Total_Premium']/1000:.0f}K"
        
        # Flags as emojis
        flags_emoji = ""
        if row['Has_Unusual'] == 'YES':
            flags_emoji += "🔥"
        if row['Has_Golden'] == 'YES':
            flags_emoji += "⚡"
        
        move_pct = abs((row['Strike_Price'] - row['Reference_Price']) / row['Reference_Price'] * 100)
        
        social_summary += (f"{i}. {sentiment_emoji} **${row['Ticker']}** "
                         f"${row['Strike_Price']:,.0f} {row['Contract_Type']} "
                         f"({row['Flow_Count']}x sweeps) → {premium_str} "
                         f"[{move_pct:.0f}%] {flags_emoji}\n")
    
    social_summary += f"\n💡 Avg {avg_flows_per_contract:.1f} sweeps per contract suggests institutional accumulation\n"
    social_summary += "⚠️ Educational only. #OptionsFlow #InstitutionalFlow"
    
    # 3. QUICK LIST
    quick_summary = f"🔄 REPEAT FLOWS {timestamp.split(',')[0]}: {total_contracts} contracts, ${total_premium/1000000:.1f}M | "
    
    # Top 5 quick mentions
    top_5 = top_repeat_flows.head(5)
    quick_mentions = []
    
    for _, row in top_5.iterrows():
        side_codes = row['Side_Codes']
        bullish_sides = sum(1 for side in side_codes if side in ['A', 'AA'])
        bearish_sides = sum(1 for side in side_codes if side in ['B', 'BB'])
        
        if row['Contract_Type'] == 'CALL':
            sentiment_emoji = "📈" if bullish_sides > bearish_sides else "📉"
        else:
            sentiment_emoji = "📉" if bullish_sides > bearish_sides else "📈"
        
        premium_str = f"${row['Total_Premium']/1000000:.1f}M" if row['Total_Premium'] >= 1000000 else f"${row['Total_Premium']/1000:.0f}K"
        
        quick_mentions.append(f"{sentiment_emoji}${row['Ticker']}({row['Flow_Count']}x){premium_str}")
    
    quick_summary += " | ".join(quick_mentions)
    
    return {
        'Professional': professional_summary,
        'Social': social_summary,
        'Quick': quick_summary
    }

# Main Streamlit app
def main():
    st.set_page_config(page_title="Options Flow Analyzer", page_icon="📊", layout="wide")
    st.title("🔍 Options Flow Analyzer")
    st.markdown("Generate a newsletter or view OTM option flows for the next 2 weeks.")
    
    # Main tabs (always available)
    main_tab1, main_tab2 = st.tabs(["📊 Flow Analysis", "🗄️ Flow Database"])
    
    with main_tab1:
        st.markdown("### 📤 Upload CSV for Analysis")
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        
        if uploaded_file:
            with st.spinner("Loading data..."):
                df = load_csv(uploaded_file)
            
            if df is not None:
                # Store in session state for Flow Database tab access
                st.session_state.uploaded_df = df
                
                # Analysis tabs (only when CSV is uploaded)
                tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Newsletter", "Symbol Flows", "Repeat Flows", "Top Trades Summary", "Symbol Analysis", "High-Volume Clusters"])
                
                with tab1:
                    st.subheader("📊 Generate Dashboard Newsletter")
                    
                    # Create two columns for different newsletter types
                    col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 📄 Full Dashboard")
                    st.markdown("*Comprehensive 8-section flow analysis*")
                    top_n = st.number_input("Flows per Section", min_value=1, max_value=50, value=8, key="full_top_n")
                    min_premium = st.number_input("Min Premium ($)", min_value=0, value=250000, key="full_premium")
                    
                    st.markdown("**Dashboard Sections:**")
                    st.markdown("1. 🎯 Major Indices (SPY,QQQ,IWM,DIA,SMH)")
                    st.markdown("2. 📊 VIX & Volatility Positioning")
                    st.markdown("3. 🏭 S&P 500 Sector ETFs")
                    st.markdown("4. 📈 Extreme Bullish Stocks")
                    st.markdown("5. 📉 Extreme Bearish Stocks")
                    st.markdown("6. 🔥 Unusual Activity Alerts")
                    st.markdown("7. 📊 High Volume Concentrations")
                    st.markdown("8. ⏰ Weekly Expiration Focus")
                    
                    webhook_url = st.text_input(
                        "Discord Webhook URL", 
                        value=os.environ.get("DISCORD_WEBHOOK_URL", "https://discord.com/api/webhooks/1379692595961401406/D4v1I-h-7YKrk5KUutByAlBfheBfZmMbKydoX6_gcVnXM9AQYZXgC4twC-1T69O1MZ7h"), 
                        type="password"
                    )
                    send_discord = st.checkbox("Send to Discord")
                    
                    if st.button("🚀 Generate Dashboard Newsletter", type="primary"):
                        with st.spinner("Generating comprehensive dashboard..."):
                            newsletter = generate_newsletter(df, top_n, min_premium)
                            st.markdown("### 📋 Generated Newsletter:")
                            st.text_area("Newsletter Content", newsletter, height=500, key="newsletter_output")
                            if send_discord and webhook_url:
                                result = send_to_discord(newsletter, webhook_url)
                                st.success(result)
                
                with col2:
                    st.markdown("### 🐦 Twitter Dashboard")
                    st.markdown("*Condensed dashboard for social media*")
                    twitter_top_n = st.number_input("Flows per Section", min_value=1, max_value=10, value=3, key="twitter_top_n")
                    twitter_min_premium = st.number_input("Min Premium ($)", min_value=0, value=500000, key="twitter_premium")
                    
                    st.markdown("**Twitter Sections:**")
                    st.markdown("• 🎯 Top 3 Indices Overview")
                    st.markdown("• 📈 Top 3 Bullish Plays")
                    st.markdown("• 📉 Top 3 Bearish Plays") 
                    st.markdown("• 🔥 Top 2 Unusual Alerts")
                    
                    if st.button("📱 Generate Twitter Dashboard"):
                        with st.spinner("Generating Twitter dashboard..."):
                            twitter_newsletter = generate_twitter_newsletter(df, twitter_top_n, twitter_min_premium)
                            st.markdown("### 🐦 Twitter Post:")
                            st.text_area("Twitter Content", twitter_newsletter, height=350, key="twitter_output")
                            
                            # Character count
                            char_count = len(twitter_newsletter)
                            if char_count <= 280:
                                st.success(f"✅ Perfect! {char_count}/280 characters")
                            elif char_count <= 1400:
                                st.warning(f"⚠️ Thread needed: {char_count} characters ({char_count//280 + 1} tweets)")
                            else:
                                st.error(f"❌ Too long: {char_count} characters - consider shortening")
                
                # Add dashboard preview/explanation
                st.markdown("---")
                st.markdown("### 📊 **Dashboard Newsletter Features**")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("""
                    **🎯 Professional Structure**
                    - 8 comprehensive sections
                    - Visual separators and emojis
                    - Clear section headers
                    - Professional formatting
                    """)
                
                with col2:
                    st.markdown("""
                    **📈 Market Intelligence**
                    - Index positioning analysis
                    - Sector rotation insights
                    - Volatility positioning (VIX)
                    - Weekly expiration focus
                    """)
                
                with col3:
                    st.markdown("""
                    **🔥 Advanced Analytics**
                    - Unusual activity detection
                    - High volume concentrations
                    - Bullish/bearish extremes
                    - Smart filtering & ranking
                    """)

            with tab2:
                st.subheader("View Symbol Flows")
                symbol = st.text_input("Enter Symbol (e.g., AAPL)").upper()
                if symbol:
                    display_symbol_flows(df, symbol)
            
            with tab3:
                st.subheader("🔄 Repeat Flows Analysis")
                st.markdown("Identify contracts with multiple flows throughout the day")
                
                # Controls for repeat flows
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    repeat_min_premium = st.number_input(
                        "Minimum Premium ($)", 
                        min_value=0, 
                        value=300000, 
                        step=5000,
                        key="repeat_premium"
                    )
                
                with col2:
                    st.markdown("**Instructions:**")
                    st.markdown("- Shows contracts that appeared multiple times")
                    st.markdown("- Groups by ticker, contract type, strike, and expiration")
                    st.markdown("- Useful for spotting accumulation patterns")
                
                if st.button("Analyze Repeat Flows", key="analyze_repeats"):
                    with st.spinner("Analyzing repeat flows..."):
                        display_repeat_flows(df, repeat_min_premium)
                else:
                    st.info("Click 'Analyze Repeat Flows' to see contracts with multiple flows")
            
            # NEW TAB - Top Trades Summary
            with tab4:
                st.subheader("🏆 Top Trades Summary")
                st.markdown("Highest scoring trade for each ticker based on premium, frequency, and flow percentage")
                
                # Controls for top trades
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    summary_min_premium = st.number_input(
                        "Minimum Premium ($)", 
                        min_value=0, 
                        value=250000, 
                        step=50000,
                        key="summary_premium"
                    )
                    
                    min_flows = st.number_input(
                        "Minimum Sweeps", 
                        min_value=1, 
                        value=2, 
                        step=1,
                        key="min_flows"
                    )
                
                with col2:
                    st.markdown("**How it works:**")
                    st.markdown("- Finds the highest scoring contract for each ticker")
                    st.markdown("- Score based on premium size, sweep count, and daily flow %")
                    st.markdown("- Excludes index ETFs (SPY, QQQ, etc.)")
                    st.markdown("- Shows sentiment based on contract type and side codes")
                
                if st.button("Generate Top Trades Summary", key="generate_summary"):
                    with st.spinner("Calculating trade scores..."):
                        display_top_trades_summary(df, summary_min_premium, min_flows)
                else:
                    st.info("Click 'Generate Top Trades Summary' to see the highest scoring trades")
            
            # NEW TAB - Symbol Analysis
            with tab5:
                st.subheader("🎯 Symbol Analysis & Price Projection")
                st.markdown("Analyze option flows for specific symbols and get projected price path for next week")
                
                # Symbol input
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    symbols_input = st.text_input(
                        "Enter Symbol(s) (comma-separated)", 
                        placeholder="e.g., QQQ, AAPL, TSLA",
                        help="Enter one or more stock symbols separated by commas"
                    )
                    
                    min_premium_analysis = st.number_input(
                        "Minimum Premium ($)", 
                        min_value=0, 
                        value=100000, 
                        step=25000,
                        key="analysis_premium"
                    )
                
                with col2:
                    st.markdown("**Analysis includes:**")
                    st.markdown("- Overall sentiment from flows")
                    st.markdown("- Next week price projection")
                    st.markdown("- Key support/resistance levels")
                    st.markdown("- Most active strike prices")
                
                if symbols_input:
                    symbols = [s.strip() for s in symbols_input.split(',') if s.strip()]
                    
                    if st.button("Analyze Symbols", key="analyze_symbols"):
                        with st.spinner(f"Analyzing flows for {', '.join(symbols)}..."):
                            analyze_symbol_flows(df, symbols, min_premium_analysis)
                    else:
                        st.info(f"Click 'Analyze Symbols' to analyze flows for: {', '.join(symbols)}")
                else:
                    st.info("Enter one or more symbols to analyze their option flows and price projection")
            
            # NEW TAB - High-Volume Flow Clusters
            with tab6:
                st.subheader("🚨 High-Volume Flow Clusters")
                st.markdown("Identify stocks where multiple large flows occurred simultaneously")
                
                # Controls for high-volume analysis
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    cluster_min_premium = st.number_input(
                        "Minimum Premium ($)", 
                        min_value=0, 
                        value=40000, 
                        step=10000,
                        key="cluster_premium",
                        help="Filter flows by minimum premium amount"
                    )
                    
                    cluster_min_quantity = st.number_input(
                        "Minimum Quantity", 
                        min_value=1, 
                        value=900, 
                        step=100,
                        key="cluster_quantity",
                        help="Filter flows by minimum contract quantity"
                    )
                    
                    time_grouping = st.selectbox(
                        "Time Grouping",
                        ["1 Second", "5 Seconds", "10 Seconds", "30 Seconds", "1 Minute"],
                        index=0,
                        key="time_grouping",
                        help="Group flows within this time window as 'simultaneous'"
                    )
                    
                    summary_top_n = st.number_input(
                        "Flows to Show in Summary",
                        min_value=1,
                        max_value=100,
                        value=70,
                        step=5,
                        key="summary_top_n",
                        help="Number of top flows to include in sharing summary"
                    )
                
                with col2:
                    st.markdown("**What this analyzes:**")
                    st.markdown("- 🎯 **Large Flows**: Premium ≥ threshold AND quantity ≥ threshold")
                    st.markdown("- ⏰ **Same Trade Time**: Multiple flows for same stock at EXACT same time")
                    st.markdown("- 🧠 **Coordinated Activity**: Identifies simultaneous institutional moves")
                    st.markdown("- 📈 **Sentiment Analysis**: Bullish/bearish direction from synchronized flows")
                    
                    st.markdown("**Perfect for finding:**")
                    st.markdown("- Block trades split across multiple orders")
                    st.markdown("- Coordinated institutional entries at same moment")
                    st.markdown("- High-conviction simultaneous positioning")
                    st.markdown("- Algorithmic trading clusters")
                
                if st.button("🔍 Find Same-Time Flow Clusters", key="find_clusters", type="primary"):
                    with st.spinner(f"Analyzing flows occurring within {time_grouping.lower()} time windows..."):
                        display_high_volume_flows(df, cluster_min_premium, cluster_min_quantity, time_grouping, summary_top_n)
                else:
                    st.info(f"Click 'Find Same-Time Flow Clusters' to identify stocks with simultaneous large flows (grouped by {time_grouping.lower()})")
                    
                    # Show example of what we're looking for
                    with st.expander("💡 Example of High-Volume Cluster", expanded=False):
                        st.markdown("""
                        **Example Scenario:**
                        
                        **AAPL** - Apple Inc.
                        - Flow 1: CALL $180 - Premium: $85,000 - Quantity: 1,200 contracts
                        - Flow 2: CALL $185 - Premium: $120,000 - Quantity: 1,500 contracts  
                        - Flow 3: PUT $170 - Premium: $95,000 - Quantity: 1,100 contracts
                        
                        **Analysis Results:**
                        - ✅ All flows meet criteria (Premium > $200K, Quantity > 900)
                        - 🎯 3 flows for same stock = cluster detected
                        - 📊 Total premium: $300,000
                        - 🧠 Mixed signals: Call buying + Put buying = Volatility play
                        """)
        else:
            st.info("👆 Upload a CSV file to start analyzing options flow data")
    
    with main_tab2:
        #st.subheader("🗄️ Flow Database")
        #st.markdown("Store and analyze flows for (minimum 900 contracts + $200K premium) with advanced filtering")
        
        # Initialize database
        init_flow_database()
        
        # Migrate existing data and show debug info
        with st.expander("🔧 Database Status", expanded=False):
            total_rows = migrate_existing_data()
            
            # Show current database status
            conn = sqlite3.connect('flow_database.db')
            cursor = conn.cursor()
            
            cursor.execute('SELECT COUNT(*) FROM flows')
            total_flows = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM flows WHERE is_open = 1')
            open_flows = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM flows WHERE is_open = 0')
            closed_flows = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(DISTINCT symbol) FROM flows')
            unique_symbols = cursor.fetchone()[0]
            
            conn.close()
            
            st.markdown(f"""
            **Database Status:**
            - Total Flows in Database: {total_flows}
            - Open Positions: {open_flows}
            - Closed/Expired: {closed_flows}
            - Unique Symbols: {unique_symbols}
            """)
            
            if total_flows == 0:
                st.warning("📭 No flows in database. Upload CSV data in Flow Analysis tab and store flows.")
            elif open_flows == 0 and closed_flows > 0:
                st.info("📊 All flows are expired/closed. Adjust date filters to see historical data.")
        
        # Two main sections
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### 📥 Store New Flows")
            #st.info("⚡ Stores flows meeting either: 900+ contracts + $200K+ premium OR $900K+ premium (any size)")
            
            # Show enhanced criteria details
            with st.expander("📋 Enhanced Storage Criteria", expanded=False):
                st.markdown("**Flows are stored if they meet EITHER of these criteria:**")
                st.markdown("1. **Volume + Premium**: 900+ contracts AND $200K+ premium")
                st.markdown("2. **High Premium**: $900K+ premium (regardless of contract count)")
                st.markdown("")
                st.markdown("**Why the enhancement?**")
                st.markdown("• Captures expensive single-contract flows (high strike/IV)")
                st.markdown("• Ensures no large institutional flows are missed")
                st.markdown("• Better coverage of high-value options activity")
            
            # Show which stocks are tracked
            with st.expander("📊 Tracked Stocks", expanded=False):
                stock_cols = st.columns(3)
                for i, stock in enumerate(TOP_STOCKS):
                    with stock_cols[i % 3]:
                        st.write(f"• {stock}")
            
            # Check if we have uploaded data in the Flow Analysis tab
            upload_section = st.empty()
            if 'uploaded_df' not in st.session_state:
                with upload_section.container():
                    st.warning("📤 No CSV data available. Upload a CSV in the 'Flow Analysis' tab first.")
                    if st.button("🔄 Check for Uploaded Data", key="check_upload"):
                        st.rerun()
            else:
                if st.button("💾 Store Institutional Flows (Enhanced Criteria)", type="primary"):
                    with st.spinner("Storing institutional flows in database..."):
                        count = store_flows_in_database(st.session_state.uploaded_df)
                    if count > 0:
                        st.success(f"✅ Stored {count} institutional flows in database!")
                    else:
                        st.warning("⚠️ No flows found meeting criteria (Top 30 stocks + enhanced filtering)")
        
        # Performance and display controls
        perf_cols = st.columns([2, 1, 1, 1])
        with perf_cols[0]:
            st.markdown("### 🔍 Filter & View Database")
        with perf_cols[1]:
            include_technicals = st.checkbox("📈 Show Technicals", value=True, help="Disable for faster loading with large datasets")
        with perf_cols[2]:
            limit_results = st.selectbox("📊 Results Limit", [500, 1000, 2000, 5000, "All"], index=0, help="Limit results for better performance")
        with perf_cols[3]:
            if st.button("🔄 Refresh Data"):
                st.cache_data.clear()
                st.rerun()
        
        # Filter controls
        filter_cols = st.columns(5)
        
        with filter_cols[0]:
            symbol_options = ["All"] + TOP_STOCKS
            symbol_filter = st.selectbox("Symbol", symbol_options, key="db_symbol_filter")
        
        with filter_cols[1]:
            order_type_options = ["All", "Calls Bought", "Calls Sold", "Puts Bought", "Puts Sold"]
            order_type_filter = st.selectbox("Order Type", order_type_options, key="db_order_filter")
        
        # Removed position filter - keeping columns aligned
        with filter_cols[2]:
            pass  # Empty column for layout
        
        with filter_cols[3]:
            date_from = st.date_input("From Date", value=datetime.now() - timedelta(days=5), key="db_date_from")
        
        with filter_cols[4]:
            date_to = st.date_input("To Date", value=datetime.now(), key="db_date_to")
    
        # Load flows from database (simplified without position tracking)
        flows_df = get_flows_from_database(
            symbol_filter=symbol_filter if symbol_filter != "All" else None,
            order_type_filter=order_type_filter if order_type_filter != "All" else None,
            date_from=date_from.strftime('%Y-%m-%d'),
            date_to=date_to.strftime('%Y-%m-%d'),
            include_technicals=include_technicals,
            limit=None if limit_results == "All" else limit_results
        )
        
        # Automatically load and display filtered data (reactive filtering)
        with st.spinner("Loading flows from database..."):
            flows_df = get_flows_from_database(
                symbol_filter=symbol_filter if symbol_filter != "All" else None,
                order_type_filter=order_type_filter if order_type_filter != "All" else None,
                date_from=date_from,
                date_to=date_to,
                include_technicals=include_technicals,
                limit=None if limit_results == "All" else int(limit_results)
            )
        
        # Show interpretation when specific symbol is selected
        if symbol_filter != "All":
            with st.expander(f"🧠 {symbol_filter} Flow Interpretation", expanded=True):
                interpretation = generate_symbol_interpretation(flows_df, symbol_filter)
                st.markdown(interpretation)
        
        # Display results
        if not flows_df.empty:
            # Calculate total premium for summary
            total_premium = flows_df['premium'].sum()
            
            # Summary stats with premium
            stats_cols = st.columns(5)
            with stats_cols[0]:
                st.metric("Total Flows", len(flows_df))
            with stats_cols[1]:
                unique_symbols = flows_df['symbol'].nunique()
                st.metric("Unique Symbols", unique_symbols)
            with stats_cols[2]:
                # Format premium display
                if total_premium >= 1000000:
                    premium_display = f"${total_premium/1000000:.1f}M"
                else:
                    premium_display = f"${total_premium/1000:.0f}K"
                st.metric("Total Premium", premium_display)
            with stats_cols[3]:
                calls_count = flows_df[flows_df['order_type'].str.contains('Calls')].shape[0]
                st.metric("Call Flows", calls_count)
            with stats_cols[4]:
                puts_count = flows_df[flows_df['order_type'].str.contains('Puts')].shape[0]
                st.metric("Put Flows", puts_count)
            
            # Add trending sections if showing all symbols
            if symbol_filter == "All":
                st.markdown("---")
                
                # Get latest trading day for display headers
                conn = sqlite3.connect('flow_database.db')
                latest_day_query = "SELECT MAX(substr(trade_date, 1, 10)) as latest_day FROM flows"
                latest_day_result = pd.read_sql_query(latest_day_query, conn)
                conn.close()
                
                if not latest_day_result.empty and latest_day_result['latest_day'].iloc[0] is not None:
                    latest_trading_day = latest_day_result['latest_day'].iloc[0]
                    # Format date for display (e.g., "Sep 13" or "Friday")
                    try:
                        date_obj = datetime.strptime(latest_trading_day, '%Y-%m-%d')
                        day_display = date_obj.strftime('%b %d')
                    except:
                        day_display = "Latest Day"
                else:
                    day_display = "Latest Day"
                
                trending_cols = st.columns(2)
                
                with trending_cols[0]:
                    st.markdown(f"### 📈 {day_display}'s Top Bullish flows")
                    with st.spinner(f"Analyzing {day_display.lower()}'s bullish flows..."):
                        bullish_stocks = get_trending_bullish_stocks(days_back=1, min_flows=2)
                    
                    if not bullish_stocks.empty:
                        for idx, row in bullish_stocks.head(10).iterrows():
                            col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                            
                            with col1:
                                st.write(f"**{row['symbol']}**")
                            with col2:
                                st.write(f"🟢 {row['bullish_percentage']:.1f}%")
                            with col3:
                                st.write(f"{row['total_flows']} flows")
                            with col4:
                                st.write(f"{row['total_contracts']:,} contracts")
                    else:
                        st.info(f"No bullish stocks found in {day_display.lower()}'s flows")
                
                with trending_cols[1]:
                    st.markdown(f"### 📉 {day_display}'s Top Bearish flows")
                    with st.spinner(f"Analyzing {day_display.lower()}'s bearish flows..."):
                        bearish_stocks = get_trending_bearish_stocks(days_back=1, min_flows=2)
                    
                    if not bearish_stocks.empty:
                        for idx, row in bearish_stocks.head(10).iterrows():
                            col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                            
                            with col1:
                                st.write(f"**{row['symbol']}**")
                            with col2:
                                st.write(f"🔴 {row['bearish_percentage']:.1f}%")
                            with col3:
                                st.write(f"{row['total_flows']} flows")
                            with col4:
                                st.write(f"{row['total_contracts']:,} contracts")
                    else:
                        st.info(f"No bearish stocks found in {day_display.lower()}'s flows")
                
                st.markdown("---")
            
            # Format the data for display with premium information
            display_df = flows_df.copy()
            
            # Format premium for better readability
            display_df['premium_formatted'] = display_df['premium'].apply(
                lambda x: f"${x/1000000:.2f}M" if x >= 1000000 else f"${x/1000:.0f}K"
            )
            
            # Select and rename columns for display
            columns_to_show = ['trade_date', 'order_type', 'symbol', 'strike', 'expiry', 'contracts', 'premium_formatted']
            column_names = ['Trade Date', 'Order Type', 'Symbol', 'Strike', 'Expiry', 'Contracts', 'Premium']
            
            if 'Technicals' in display_df.columns:
                columns_to_show.append('Technicals')
                column_names.append('Technicals')
            
            display_df = display_df[columns_to_show]
            display_df.columns = column_names
            
            # Apply styling based on bullish vs bearish sentiment (not just bought vs sold)
            def style_order_type(val):
                # Bullish activities (green): Calls Bought, Puts Sold
                if val in ['Calls Bought', 'Puts Sold']:
                    return 'background-color: #d4edda; color: #155724; font-weight: bold; border-radius: 4px; padding: 2px 8px;'
                # Bearish activities (red): Calls Sold, Puts Bought
                elif val in ['Calls Sold', 'Puts Bought']:
                    return 'background-color: #f8d7da; color: #721c24; font-weight: bold; border-radius: 4px; padding: 2px 8px;'
                return ''
            
            # Display with enhanced formatting
            st.markdown("### 📊 Flow Database Results")
            
            styled_df = display_df.style.applymap(style_order_type, subset=['Order Type'])
            st.dataframe(styled_df, use_container_width=True, height=400)
            
            # Export option
            csv_data = display_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Filtered Data as CSV",
                data=csv_data,
                file_name=f"flow_database_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
            
        else:
            st.info("No flows found matching the current filters.")

if __name__ == "__main__":
    main()
