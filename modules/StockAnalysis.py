import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configure Streamlit page
#st.set_page_config(page_title="Stock Trading Setup Scanner", layout="wide")

class TechnicalAnalyzer:
    def __init__(self, symbol, period="3mo"):
        self.symbol = symbol
        self.period = period
        self.data = None
        self.signals = {}
        
    def fetch_data(self):
        """Fetch stock data from yfinance"""
        try:
            ticker = yf.Ticker(self.symbol)
            self.data = ticker.history(period=self.period)
            return True
        except Exception as e:
            st.error(f"Error fetching data for {self.symbol}: {str(e)}")
            return False
    
    def calculate_indicators(self):
        """Calculate technical indicators using pandas and numpy"""
        if self.data is None or len(self.data) < 50:
            return False
            
        df = self.data.copy()
        
        # Moving Averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        
        # RSI Calculation
        def calculate_rsi(prices, window=14):
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        
        df['RSI'] = calculate_rsi(df['Close'])
        
        # MACD Calculation
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        # Bollinger Bands
        def calculate_bollinger_bands(prices, window=20, num_std=2):
            rolling_mean = prices.rolling(window=window).mean()
            rolling_std = prices.rolling(window=window).std()
            upper_band = rolling_mean + (rolling_std * num_std)
            lower_band = rolling_mean - (rolling_std * num_std)
            return upper_band, rolling_mean, lower_band
        
        df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = calculate_bollinger_bands(df['Close'])
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        
        # Support and Resistance levels
        df['Resistance'] = df['High'].rolling(window=20).max()
        df['Support'] = df['Low'].rolling(window=20).min()
        
        # Drop any NaN values to avoid issues
        df = df.dropna()
        
        self.data = df
        return True
    
    def detect_patterns(self):
        """Detect various technical patterns"""
        if self.data is None:
            return {}
            
        df = self.data
        current_price = df['Close'].iloc[-1]
        current_volume = df['Volume'].iloc[-1]
        avg_volume = df['Volume_SMA'].iloc[-1]
        
        patterns = {}
        
        # 1. Breakout Setup
        resistance_level = df['Resistance'].iloc[-2]
        if current_price > resistance_level * 1.001:  # 0.1% above resistance
            patterns['Breakout'] = {
                'signal': 'BUY',
                'entry': current_price,
                'stop_loss': resistance_level * 0.98,
                'short_target': current_price * 1.03,
                'medium_target': current_price * 1.08,
                'confidence': 'HIGH' if current_volume > avg_volume * 1.5 else 'MEDIUM'
            }
        
        # 2. Pullback to Support
        support_level = df['Support'].iloc[-2]
        sma_20 = df['SMA_20'].iloc[-1]
        if (current_price <= support_level * 1.02 and 
            current_price > support_level * 0.98 and
            current_price > sma_20):
            patterns['Support_Bounce'] = {
                'signal': 'BUY',
                'entry': current_price,
                'stop_loss': support_level * 0.97,
                'short_target': current_price * 1.04,
                'medium_target': current_price * 1.10,
                'confidence': 'MEDIUM'
            }
        
        # 3. RSI Oversold Bounce
        rsi = df['RSI'].iloc[-1]
        rsi_prev = df['RSI'].iloc[-2]
        if rsi < 35 and rsi > rsi_prev and current_price > df['SMA_20'].iloc[-1]:
            patterns['RSI_Oversold'] = {
                'signal': 'BUY',
                'entry': current_price,
                'stop_loss': current_price * 0.95,
                'short_target': current_price * 1.05,
                'medium_target': current_price * 1.12,
                'confidence': 'MEDIUM'
            }
        
        # 4. MACD Bullish Crossover
        macd = df['MACD'].iloc[-1]
        macd_signal = df['MACD_Signal'].iloc[-1]
        macd_prev = df['MACD'].iloc[-2]
        macd_signal_prev = df['MACD_Signal'].iloc[-2]
        
        if (macd > macd_signal and macd_prev <= macd_signal_prev and macd < 0):
            patterns['MACD_Bullish'] = {
                'signal': 'BUY',
                'entry': current_price,
                'stop_loss': current_price * 0.94,
                'short_target': current_price * 1.06,
                'medium_target': current_price * 1.15,
                'confidence': 'MEDIUM'
            }
        
        # 5. Bollinger Band Squeeze Breakout
        bb_upper = df['BB_Upper'].iloc[-1]
        bb_lower = df['BB_Lower'].iloc[-1]
        bb_width = (bb_upper - bb_lower) / df['BB_Middle'].iloc[-1]
        bb_width_avg = ((df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']).rolling(20).mean().iloc[-1]
        
        if bb_width < bb_width_avg * 0.8 and current_price > bb_upper:
            patterns['BB_Breakout'] = {
                'signal': 'BUY',
                'entry': current_price,
                'stop_loss': df['BB_Middle'].iloc[-1],
                'short_target': current_price * 1.04,
                'medium_target': current_price * 1.10,
                'confidence': 'HIGH' if current_volume > avg_volume * 1.3 else 'MEDIUM'
            }
        
        # 6. Moving Average Golden Cross (shorter timeframe)
        ema_12 = df['EMA_12'].iloc[-1]
        ema_26 = df['EMA_26'].iloc[-1]
        ema_12_prev = df['EMA_12'].iloc[-2]
        ema_26_prev = df['EMA_26'].iloc[-2]
        
        if ema_12 > ema_26 and ema_12_prev <= ema_26_prev:
            patterns['Golden_Cross'] = {
                'signal': 'BUY',
                'entry': current_price,
                'stop_loss': ema_26 * 0.97,
                'short_target': current_price * 1.05,
                'medium_target': current_price * 1.12,
                'confidence': 'MEDIUM'
            }
        
        # 7. Gap Fill Opportunity (Both Long and Short)
        if len(df) > 1:
            prev_close = df['Close'].iloc[-2]
            today_open = df['Open'].iloc[-1]
            gap_percent = abs(today_open - prev_close) / prev_close
            
            if gap_percent > 0.02:  # 2% gap
                # Gap down - potential long setup
                if today_open < prev_close and current_price > (today_open + prev_close) / 2:
                    patterns['Gap_Fill_Long'] = {
                        'signal': 'BUY',
                        'entry': current_price,
                        'stop_loss': today_open * 0.98,
                        'short_target': prev_close * 0.99,
                        'medium_target': prev_close * 1.02,
                        'confidence': 'MEDIUM'
                    }
                # Gap up - potential short setup
                elif today_open > prev_close and current_price < (today_open + prev_close) / 2:
                    patterns['Gap_Fill_Short'] = {
                        'signal': 'SHORT',
                        'entry': current_price,
                        'stop_loss': today_open * 1.02,
                        'short_target': prev_close * 1.01,
                        'medium_target': prev_close * 0.99,
                        'confidence': 'MEDIUM'
                    }
        
        # 8. Resistance Rejection (Short Setup)
        if current_price <= resistance_level * 0.999 and current_price >= resistance_level * 0.985:
            patterns['Resistance_Rejection'] = {
                'signal': 'SHORT',
                'entry': current_price,
                'stop_loss': resistance_level * 1.02,
                'short_target': current_price * 0.97,
                'medium_target': current_price * 0.92,
                'confidence': 'HIGH' if current_volume > avg_volume * 1.5 else 'MEDIUM'
            }
        
        # 9. RSI Overbought Rejection (Short Setup)
        if rsi > 70 and rsi < rsi_prev and current_price < df['SMA_20'].iloc[-1]:
            patterns['RSI_Overbought'] = {
                'signal': 'SHORT',
                'entry': current_price,
                'stop_loss': current_price * 1.05,
                'short_target': current_price * 0.95,
                'medium_target': current_price * 0.88,
                'confidence': 'MEDIUM'
            }
        
        # 10. MACD Bearish Crossover (Short Setup)
        if (macd < macd_signal and macd_prev >= macd_signal_prev and macd > 0):
            patterns['MACD_Bearish'] = {
                'signal': 'SHORT',
                'entry': current_price,
                'stop_loss': current_price * 1.06,
                'short_target': current_price * 0.94,
                'medium_target': current_price * 0.85,
                'confidence': 'MEDIUM'
            }
        
        # 11. Death Cross (Short Setup)
        if ema_12 < ema_26 and ema_12_prev >= ema_26_prev:
            patterns['Death_Cross'] = {
                'signal': 'SHORT',
                'entry': current_price,
                'stop_loss': ema_12 * 1.03,
                'short_target': current_price * 0.95,
                'medium_target': current_price * 0.88,
                'confidence': 'MEDIUM'
            }
        
        # 12. Bollinger Band Upper Rejection (Short Setup)
        if (current_price >= bb_upper * 0.995 and 
            current_price <= bb_upper * 1.005 and 
            df['Close'].iloc[-2] > bb_upper):
            patterns['BB_Upper_Rejection'] = {
                'signal': 'SHORT',
                'entry': current_price,
                'stop_loss': bb_upper * 1.02,
                'short_target': df['BB_Middle'].iloc[-1],
                'medium_target': bb_lower,
                'confidence': 'MEDIUM'
            }
        
        # 13. Failed Breakout (Short Setup)
        if (df['High'].iloc[-2] > resistance_level and 
            current_price < resistance_level * 0.98 and
            current_volume > avg_volume):
            patterns['Failed_Breakout'] = {
                'signal': 'SHORT',
                'entry': current_price,
                'stop_loss': resistance_level * 1.01,
                'short_target': current_price * 0.96,
                'medium_target': support_level * 1.02,
                'confidence': 'HIGH'
            }
        
        # 14. Double Top Pattern (Short Setup)
        highs = df['High'].rolling(5).max()
        if len(highs) >= 10:
            recent_high = highs.iloc[-1]
            prev_high = highs.iloc[-6:-1].max()
            if (abs(recent_high - prev_high) / prev_high < 0.02 and  # Within 2%
                current_price < recent_high * 0.97):
                patterns['Double_Top'] = {
                    'signal': 'SHORT',
                    'entry': current_price,
                    'stop_loss': recent_high * 1.02,
                    'short_target': current_price * 0.95,
                    'medium_target': current_price * 0.90,
                    'confidence': 'HIGH'
                }
        
        return patterns
    
    def create_chart(self):
        """Create interactive chart with indicators"""
        if self.data is None:
            return None
            
        df = self.data.tail(50)  # Last 50 days
        
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            subplot_titles=('Price & Technical Indicators', 'Volume', 'RSI', 'MACD'),
            row_heights=[0.5, 0.15, 0.175, 0.175],
            specs=[[{"secondary_y": False}],
                   [{"secondary_y": False}],
                   [{"secondary_y": False}],
                   [{"secondary_y": False}]]
        )
        
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='Price',
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Moving averages
        fig.add_trace(
            go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20', 
                      line=dict(color='orange', width=1)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50', 
                      line=dict(color='blue', width=1)),
            row=1, col=1
        )
        
        # Bollinger Bands
        fig.add_trace(
            go.Scatter(x=df.index, y=df['BB_Upper'], name='BB Upper', 
                      line=dict(color='gray', dash='dash', width=1),
                      fill=None),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['BB_Lower'], name='BB Lower', 
                      line=dict(color='gray', dash='dash', width=1),
                      fill='tonexty', fillcolor='rgba(128,128,128,0.1)'),
            row=1, col=1
        )
        
        # Volume on separate subplot
        colors = ['red' if df['Close'].iloc[i] < df['Open'].iloc[i] else 'green' 
                 for i in range(len(df))]
        
        fig.add_trace(
            go.Bar(x=df.index, y=df['Volume'], name='Volume', 
                  marker_color=colors, opacity=0.7, showlegend=False),
            row=2, col=1
        )
        
        # Volume moving average
        fig.add_trace(
            go.Scatter(x=df.index, y=df['Volume_SMA'], name='Vol MA', 
                      line=dict(color='black', width=1)),
            row=2, col=1
        )
        
        # RSI
        fig.add_trace(
            go.Scatter(x=df.index, y=df['RSI'], name='RSI', 
                      line=dict(color='purple', width=2), showlegend=False),
            row=3, col=1
        )
        
        # RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", 
                     annotation_text="Overbought", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", 
                     annotation_text="Oversold", row=3, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", row=3, col=1)
        
        # MACD
        fig.add_trace(
            go.Scatter(x=df.index, y=df['MACD'], name='MACD', 
                      line=dict(color='blue', width=2)),
            row=4, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal', 
                      line=dict(color='red', width=2)),
            row=4, col=1
        )
        
        # MACD Histogram
        colors_macd = ['green' if val >= 0 else 'red' for val in df['MACD_Hist']]
        fig.add_trace(
            go.Bar(x=df.index, y=df['MACD_Hist'], name='Histogram', 
                  marker_color=colors_macd, opacity=0.6, showlegend=False),
            row=4, col=1
        )
        
        # Add zero line for MACD
        fig.add_hline(y=0, line_dash="dash", line_color="black", row=4, col=1)
        
        # Update layout
        fig.update_layout(
            title=f'{self.symbol} - Technical Analysis',
            xaxis_rangeslider_visible=False,
            height=900,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Update y-axes labels
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        fig.update_yaxes(title_text="RSI", row=3, col=1, range=[0, 100])
        fig.update_yaxes(title_text="MACD", row=4, col=1)
        
        # Remove x-axis labels for all but bottom chart
        fig.update_xaxes(showticklabels=False, row=1, col=1)
        fig.update_xaxes(showticklabels=False, row=2, col=1)
        fig.update_xaxes(showticklabels=False, row=3, col=1)
        fig.update_xaxes(title_text="Date", row=4, col=1)
        
        return fig

def run():
    """Run the Stock Trading Setup Scanner application"""
    st.title("🎯 Stock Trading Setup Scanner")
    st.markdown("Analyze stocks for technical trading setups with price targets")
    
    # Sidebar for inputs
    st.sidebar.header("Configuration")
    
    # Default stock list
    default_stocks = "AAPL,GOOGL,MSFT,TSLA,NVDA,AMZN,META,NFLX,AMD,PLTR,ADBE,ADI,ADP,AMAT,ASML,AVGO,JPM,CRM,CSCO,GS,DELL,EBAY,FTNT,CEG,IBM,MMM,KLAC,LRCX,MCHP,FSLR,MRVL,MU,NOW,NTAP,ORCL,PANW,PAYX,PLTR,PTC,QCOM,GE,TEAM,QQQ,TXN,UNH,WDAY,SPY,ZS"
    
    # Stock input
    stock_input = st.sidebar.text_area(
        "Enter stock symbols (comma-separated):",
        value=default_stocks,
        help="Enter stock symbols separated by commas"
    )
    
    # Time period
    period = st.sidebar.selectbox(
        "Analysis Period:",
        ["1mo", "3mo", "6mo", "1y"],
        index=1
    )
    
    # Minimum confidence filter
    min_confidence = st.sidebar.selectbox(
        "Minimum Confidence Level:",
        ["ALL", "MEDIUM", "HIGH"],
        index=0
    )
    
    if st.sidebar.button("🔍 Scan for Setups", type="primary"):
        stock_list = [s.strip().upper() for s in stock_input.split(",") if s.strip()]
        
        if not stock_list:
            st.error("Please enter at least one stock symbol")
            return
        
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = []
        
        for i, symbol in enumerate(stock_list):
            status_text.text(f"Analyzing {symbol}...")
            progress_bar.progress((i + 1) / len(stock_list))
            
            analyzer = TechnicalAnalyzer(symbol, period)
            
            if analyzer.fetch_data() and analyzer.calculate_indicators():
                patterns = analyzer.detect_patterns()
                
                if patterns:
                    for pattern_name, pattern_data in patterns.items():
                        if min_confidence == "ALL" or pattern_data['confidence'] == min_confidence or (min_confidence == "MEDIUM" and pattern_data['confidence'] in ["MEDIUM", "HIGH"]):
                            results.append({
                                'Symbol': symbol,
                                'Pattern': pattern_name,
                                'Signal': pattern_data['signal'],
                                'Entry': f"${pattern_data['entry']:.2f}",
                                'Stop Loss': f"${pattern_data['stop_loss']:.2f}",
                                'Short Target': f"${pattern_data['short_target']:.2f}",
                                'Medium Target': f"${pattern_data['medium_target']:.2f}",
                                'Confidence': pattern_data['confidence'],
                                'Risk/Reward': f"{((pattern_data['short_target'] - pattern_data['entry']) / (pattern_data['entry'] - pattern_data['stop_loss'])):.2f}:1"
                            })
        
        status_text.empty()
        progress_bar.empty()
        
        # Display results
        if results:
            st.success(f"Found {len(results)} trading setups!")
            
            # Convert to DataFrame
            df_results = pd.DataFrame(results)
            
            # Display summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Setups", len(results))
            with col2:
                st.metric("High Confidence", len(df_results[df_results['Confidence'] == 'HIGH']))
            with col3:
                st.metric("Buy Signals", len(df_results[df_results['Signal'] == 'BUY']))
            with col4:
                st.metric("Short Signals", len(df_results[df_results['Signal'] == 'SHORT']))
            
            # Results table
            st.subheader("📊 Trading Setups")
            
            # Color code by confidence
            def highlight_confidence(row):
                if row['Confidence'] == 'HIGH':
                    return ['background-color: #d4edda'] * len(row)
                elif row['Confidence'] == 'MEDIUM':
                    return ['background-color: #fff3cd'] * len(row)
                else:
                    return [''] * len(row)
            
            styled_df = df_results.style.apply(highlight_confidence, axis=1)
            st.dataframe(styled_df, use_container_width=True)
            
            # Detailed analysis for selected stock
            st.subheader("📈 Detailed Analysis")
            selected_symbol = st.selectbox("Select a stock for detailed chart:", 
                                         sorted(list(set(df_results['Symbol']))))
            
            if selected_symbol:
                analyzer = TechnicalAnalyzer(selected_symbol, period)
                if analyzer.fetch_data() and analyzer.calculate_indicators():
                    chart = analyzer.create_chart()
                    if chart:
                        st.plotly_chart(chart, use_container_width=True)
                    
                    # Current stock info
                    current_data = analyzer.data.iloc[-1]
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Current Price", f"${current_data['Close']:.2f}")
                    with col2:
                        st.metric("RSI", f"{current_data['RSI']:.1f}")
                    with col3:
                        st.metric("Volume vs Avg", f"{(current_data['Volume']/current_data['Volume_SMA']):.1f}x")
                    with col4:
                        change_pct = ((current_data['Close'] - current_data['Open']) / current_data['Open']) * 100
                        st.metric("Day Change", f"{change_pct:.2f}%")
        
        else:
            st.warning("No trading setups found with the current criteria. Try adjusting the confidence level or stock list.")
    
    # Information section
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📚 Long Patterns")
    st.sidebar.markdown("""
    - **Breakout**: Price above resistance
    - **Support Bounce**: Price near support level
    - **RSI Oversold**: RSI < 35 with upward momentum
    - **MACD Bullish**: MACD crosses above signal
    - **BB Breakout**: Bollinger Band squeeze breakout
    - **Golden Cross**: EMA 12 crosses above EMA 26
    - **Gap Fill Long**: Gap down reversion
    """)
    
    st.sidebar.markdown("### 📉 Short Patterns")
    st.sidebar.markdown("""
    - **Resistance Rejection**: Price fails at resistance
    - **RSI Overbought**: RSI > 70 with downward momentum
    - **MACD Bearish**: MACD crosses below signal
    - **Death Cross**: EMA 12 crosses below EMA 26
    - **BB Upper Rejection**: Rejection at upper band
    - **Failed Breakout**: Breakout failure with volume
    - **Double Top**: Classic reversal pattern
    - **Gap Fill Short**: Gap up reversion
    """)
    
    st.sidebar.markdown("### ⚠️ Risk Disclaimer")
    st.sidebar.markdown("""
    This tool is for educational purposes only. 
    Always do your own research and manage risk appropriately.
    Past performance does not guarantee future results.
    """)

def main():
    """Main entry point for the application"""
    run()

if __name__ == "__main__":
    main()
