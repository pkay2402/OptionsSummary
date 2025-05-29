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

# Sector/Theme stock definitions
SECTOR_STOCKS = {
    "Technology": {
        "description": "Major technology companies including FAANG and chip makers",
        "stocks": ["AAPL", "GOOGL", "MSFT", "META", "AMZN", "NFLX", "NVDA", "AMD", "TSM", "ADBE", "CRM", "ORCL", "CSCO", "NOW", "TEAM", "WDAY", "PANW", "ZS", "FTNT", "SNOW", "PLTR", "AMAT", "LRCX", "KLAC", "MCHP", "MRVL", "QCOM", "TXN", "ADI", "ASML"]
    },
    "Index": {
        "description": "Major Indexes and ETFs",
        "stocks": ["SPY", "QQQ", "IWM", "DIA", "SMH", "XLF", "XLI", "XLY", "XLC", "XLB", "XLC", "XLI", "XLF", "XLY", "XBI", "ARKK"]
    },
    "Crypto": {
        "description": "Cryptocurrency and blockchain-related companies",
        "stocks": ["IBIT", "ETHA", "MSTR", "COIN", "RIOT", "MARA", "HUT", "BTBT", "BITX","ETHU","XXRP"]
    },
    "Healthcare & Biotech": {
        "description": "Healthcare, pharmaceuticals, and biotechnology companies",
        "stocks": ["JNJ", "PFE", "UNH", "ABBV", "TMO", "DHR", "ABT", "BMY", "MRK", "LLY", "GILD", "AMGN", "BIIB", "REGN", "VRTX", "ILMN", "MRNA", "BNTX", "ZTS", "DXCM"]
    },
    "Financial Services": {
        "description": "Banks, investment firms, and financial technology",
        "stocks": ["JPM", "BAC", "WFC", "GS", "MS", "C", "USB", "PNC", "TFC", "COF", "AXP", "BLK", "SPGI", "ICE", "CME", "V", "MA", "PYPL", "SQ", "COIN"]
    },
    "Clean Energy": {
        "description": "Solar, wind, battery, and renewable energy companies",
        "stocks": ["TSLA", "ENPH", "SEDG", "FSLR", "SPWR", "RUN", "BE", "PLUG", "BLDP", "FUV", "NEE", "AEP", "SO", "DUK", "EXC", "ORSTED", "ICLN", "PBW", "QCLN"]
    },
    "Nuclear Power": {
        "description": "Nuclear energy, uranium mining, and nuclear technology",
        "stocks": ["CEG", "NEE", "EXC", "DUK", "SO", "CCJ", "UEC", "UUUU", "DNN", "LTBR", "SMR", "BWXT", "BWX", "NNE", "OKLO", "VST"]
    },
    "Infrastructure": {
        "description": "Construction, materials, industrial equipment, and utilities",
        "stocks": ["CAT", "DE", "HON", "GE", "MMM", "EMR", "ITW", "PH", "ROK", "ETN", "FLR", "JCI", "IR", "DOV", "XYL", "AWK", "WM", "RSG", "WCN"]
    },
    "Consumer Discretionary": {
        "description": "Retail, automotive, entertainment, and luxury goods",
        "stocks": ["AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "LOW", "TJX", "F", "GM", "ABNB", "UBER", "DIS", "NFLX", "CMCSA", "T", "VZ", "LVMUY", "MC"]
    },
    "Commodities & Materials": {
        "description": "Mining, oil & gas, precious metals, and raw materials",
        "stocks": ["XOM", "CVX", "COP", "SLB", "HAL", "EOG", "KMI", "OKE", "ET", "GOLD", "NEM", "FCX", "AA", "X", "CLF", "VALE", "RIO", "BHP", "SCCO"]
    },
    "Communication Services": {
        "description": "Telecom, media, social media, and communication platforms",
        "stocks": ["GOOGL", "META", "DIS", "NFLX", "CMCSA", "T", "VZ", "CHTR", "TMUS", "TWTR", "SNAP", "PINS", "ROKU", "SPOT", "ZM", "DOCU", "YELP"]
    },
    "Real Estate & REITs": {
        "description": "Real estate investment trusts and property companies",
        "stocks": ["AMT", "PLD", "CCI", "EQIX", "PSA", "EXR", "WELL", "DLR", "O", "SBAC", "AVB", "EQR", "VTR", "MAA", "ESS", "UDR", "CPT", "ARE", "BXP"]
    },
    "Aerospace & Defense": {
        "description": "Aerospace, defense, and aviation companies",
        "stocks": ["BA", "LMT", "RTX", "NOC", "GD", "LHX", "TDG", "LDOS", "TXT", "HON", "UTX", "CW", "KTOS", "AVAV", "HEI", "WWD"]
    },
    "Artificial Intelligence": {
        "description": "AI-focused companies and AI infrastructure providers",
        "stocks": ["NVDA", "MSFT", "GOOGL", "META", "AMZN", "TSLA", "AMD", "INTC", "CRM", "ORCL", "IBM", "PLTR", "AI", "SOUN", "BBAI", "UPST", "PATH", "SNOW", "MDB", "DDOG"]
    },
    "Cybersecurity": {
        "description": "Cybersecurity and information security companies",
        "stocks": ["PANW", "CRWD", "ZS", "FTNT", "OKTA", "S", "CYBR","TENB", "QLYS", "VRNS", "RPD", "SAIL"]
    },
    "Cloud Computing": {
        "description": "Cloud infrastructure and software-as-a-service companies",
        "stocks": ["MSFT", "AMZN", "GOOGL", "CRM", "ORCL", "ADBE", "NOW", "TEAM", "WDAY", "SNOW", "MDB", "DDOG", "TWLO", "ZM", "DOCU", "OKTA", "PLTR", "NET"]
    },
    "Gaming & Entertainment": {
        "description": "Video game companies and entertainment platforms",
        "stocks": ["MSFT", "SONY", "NFLX", "DIS", "ATVI", "EA", "TTWO", "RBLX", "U", "ZNGA", "DKNG", "PENN", "LVS", "MGM", "WYNN", "CZR"]
    },
    "Electric Vehicles": {
        "description": "Electric vehicle manufacturers and EV supply chain",
        "stocks": ["TSLA", "F", "GM", "RIVN", "LCID", "NIO", "XPEV", "LI", "BYD", "CHPT", "BLNK", "EVGO", "WKHS", "FSR", "GOEV", "RIDE", "ALB", "LAC", "LTHM"]
    },
    "Semiconductors": {
        "description": "Chip manufacturers and semiconductor equipment companies",
        "stocks": ["NVDA", "AMD", "INTC", "TSM", "ASML", "AMAT", "LRCX", "KLAC", "ADI", "MCHP", "MRVL", "QCOM", "TXN", "MU", "WDC", "STM", "NXPI", "SWKS", "MPWR", "CRUS"]
    },
    "Space & Satellites": {
        "description": "Space exploration, satellites, and aerospace technology",
        "stocks": ["BA", "LMT", "NOC", "MAXR", "IRDM", "GSAT", "SPCE", "ASTR", "RKLB", "ASTS", "PL", "GILT", "HOL", "VACQ"]
    },
    "Robotics & Automation": {
        "description": "Robotics, automation, and industrial technology",
        "stocks": ["ABB", "ROK", "EMR", "HON", "FANUY", "IRBT", "ISRG", "ZIXI", "KUKA", "OMCL", "GRMN", "TER", "KEYS", "NOVT"]
    },
    "ETFs - Broad Market": {
        "description": "Major market ETFs and index funds",
        "stocks": ["SPY", "QQQ", "IWM", "VTI", "VOO", "VEA", "VWO", "EFA", "EEM", "GLD", "SLV", "TLT", "HYG", "LQD", "ARKK", "ARKW", "ARKG"]
    }
}

class TechnicalAnalyzer:
    def __init__(self, symbol, period="1yr"):
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
        
        # 5. Bollinger Band Squeeze Breakout (improved)
        bb_upper = df['BB_Upper'].iloc[-1]
        bb_lower = df['BB_Lower'].iloc[-1]
        bb_middle = df['BB_Middle'].iloc[-1]
        bb_width = (bb_upper - bb_lower) / bb_middle
        bb_width_prev = ((df['BB_Upper'].iloc[-10:-1] - df['BB_Lower'].iloc[-10:-1]) / df['BB_Middle'].iloc[-10:-1]).mean()

        # BB Squeeze with breakout confirmation (multiple confirmations needed)
        if bb_width < bb_width_prev * 0.8:  # Bands are narrowing (volatility decreasing)
            if (current_price > bb_upper and 
                df['RSI'].iloc[-1] < 70 and  # Not extremely overbought
                df['Close'].iloc[-2] < df['Close'].iloc[-1] and  # Price is rising
                current_volume > avg_volume * 1.3):  # Volume confirmation
                
                patterns['BB_Breakout'] = {
                    'signal': 'BUY',
                    'entry': current_price,
                    'stop_loss': bb_middle,
                    'short_target': current_price * 1.04,
                    'medium_target': current_price * 1.10,
                    'confidence': 'HIGH' if df['MACD'].iloc[-1] > df['MACD_Signal'].iloc[-1] else 'MEDIUM'
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
        
        # 12. Bollinger Band Upper Rejection (improved with multiple confirmations)
        if (current_price >= bb_upper * 0.98 and 
            df['RSI'].iloc[-1] > 65 and  # Overbought condition
            df['MACD'].iloc[-1] < df['MACD_Signal'].iloc[-1] and  # MACD bearish
            df['Close'].iloc[-1] < df['Close'].iloc[-2] and  # Price is falling
            current_volume > avg_volume):  # Volume confirmation
            
            patterns['BB_Upper_Rejection'] = {
                'signal': 'SHORT',
                'entry': current_price,
                'stop_loss': bb_upper * 1.02,
                'short_target': df['BB_Middle'].iloc[-1],
                'medium_target': bb_lower,
                'confidence': 'HIGH' if df['RSI'].iloc[-1] > 75 else 'MEDIUM'
            }
        
        # Add missing BB lower bounce pattern with confirmations
        if (current_price <= bb_lower * 1.02 and 
            df['RSI'].iloc[-1] < 35 and  # Oversold condition
            df['MACD'].iloc[-1] > df['MACD_Signal'].iloc[-1] and  # MACD bullish
            df['Close'].iloc[-1] > df['Close'].iloc[-2] and  # Price is rising
            current_volume > avg_volume):  # Volume confirmation
            
            patterns['BB_Lower_Bounce'] = {
                'signal': 'BUY',
                'entry': current_price,
                'stop_loss': bb_lower * 0.98,
                'short_target': df['BB_Middle'].iloc[-1],
                'medium_target': bb_upper,
                'confidence': 'HIGH' if df['RSI'].iloc[-1] < 25 else 'MEDIUM'
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
    st.title("🎯 Sector-Based Stock Trading Setup Scanner")
    st.markdown("Select sectors/themes and analyze stocks for technical trading setups")
    
    # Sidebar for inputs
    st.sidebar.header("🔧 Configuration")
    
    # Create tabs for better organization
    tab1, tab2, tab3 = st.sidebar.tabs(["Sectors", "Stocks", "Analysis"])
    
    # TAB 1: SECTOR SELECTION
    with tab1:
        selected_sectors = []
        
        # Add a select all option
        if st.checkbox("Select All Sectors", key="select_all"):
            selected_sectors = list(SECTOR_STOCKS.keys())
        else:
            # Group sectors into categories for better organization
            tech_sectors = ["Technology", "Semiconductors", "Cybersecurity", "Cloud Computing", "Artificial Intelligence"]
            energy_sectors = ["Clean Energy", "Nuclear Power", "Infrastructure"]
            finance_sectors = ["Financial Services", "Real Estate & REITs"]
            consumer_sectors = ["Consumer Discretionary", "Gaming & Entertainment"]
            industrial_sectors = ["Aerospace & Defense", "Robotics & Automation", "Space & Satellites"]
            other_sectors = ["Healthcare & Biotech", "Commodities & Materials", "Communication Services", "Electric Vehicles", "ETFs - Broad Market"]
            
            # Individual sectors that don't need "Select All" option
            st.write("### Market Sectors")
            if st.checkbox("Index", key="index_sector", help=SECTOR_STOCKS["Index"]["description"]):
                selected_sectors.append("Index")
            
            if st.checkbox("Crypto", key="crypto_sector", help=SECTOR_STOCKS["Crypto"]["description"]):
                selected_sectors.append("Crypto")
            
            # Create category selection with bulk selection options for multi-sector categories
            st.write("### Technology & Software")
            if st.checkbox("Select All Tech", key="tech_all"):
                selected_sectors.extend(tech_sectors)
                st.info(f"Selected all {len(tech_sectors)} tech sectors")
            else:
                for i, sector in enumerate(tech_sectors):
                    if st.checkbox(sector, key=f"tech_{i}", help=SECTOR_STOCKS[sector]["description"]):
                        selected_sectors.append(sector)
            
            st.write("### Energy & Infrastructure")
            if st.checkbox("Select All Energy", key="energy_all"):
                selected_sectors.extend(energy_sectors)
                st.info(f"Selected all {len(energy_sectors)} energy sectors")
            else:
                for i, sector in enumerate(energy_sectors):
                    if st.checkbox(sector, key=f"energy_{i}", help=SECTOR_STOCKS[sector]["description"]):
                        selected_sectors.append(sector)
            
            st.write("### Finance & Real Estate")
            if st.checkbox("Select All Finance", key="finance_all"):
                selected_sectors.extend(finance_sectors)
                st.info(f"Selected all {len(finance_sectors)} finance sectors")
            else:
                for i, sector in enumerate(finance_sectors):
                    if st.checkbox(sector, key=f"finance_{i}", help=SECTOR_STOCKS[sector]["description"]):
                        selected_sectors.append(sector)
            
            st.write("### Consumer & Entertainment")
            if st.checkbox("Select All Consumer", key="consumer_all"):
                selected_sectors.extend(consumer_sectors)
                st.info(f"Selected all {len(consumer_sectors)} consumer sectors")
            else:
                for i, sector in enumerate(consumer_sectors):
                    if st.checkbox(sector, key=f"consumer_{i}", help=SECTOR_STOCKS[sector]["description"]):
                        selected_sectors.append(sector)
            
            st.write("### Industrial & Defense")
            if st.checkbox("Select All Industrial", key="industrial_all"):
                selected_sectors.extend(industrial_sectors)
                st.info(f"Selected all {len(industrial_sectors)} industrial sectors")
            else:
                for i, sector in enumerate(industrial_sectors):
                    if st.checkbox(sector, key=f"industrial_{i}", help=SECTOR_STOCKS[sector]["description"]):
                        selected_sectors.append(sector)
            
            st.write("### Other Sectors")
            if st.checkbox("Select All Other", key="other_all"):
                selected_sectors.extend(other_sectors)
                st.info(f"Selected all {len(other_sectors)} other sectors")
            else:
                for i, sector in enumerate(other_sectors):
                    if st.checkbox(sector, key=f"other_{i}", help=SECTOR_STOCKS[sector]["description"]):
                        selected_sectors.append(sector)
        
        # Show selected sectors count in a more prominent way
        if selected_sectors:
            total_stocks = sum(len(SECTOR_STOCKS[sector]["stocks"]) for sector in selected_sectors)
            st.success(f"📈 Selected {len(selected_sectors)} sectors with {total_stocks} stocks")
    
    # TAB 2: STOCK SELECTION
    with tab2:
        # Show sector breakdown in an expander
        if selected_sectors:
            with st.expander("View Selected Stocks"):
                for sector in selected_sectors:
                    st.write(f"**{sector}** ({len(SECTOR_STOCKS[sector]['stocks'])} stocks)")
                    st.write(", ".join(SECTOR_STOCKS[sector]["stocks"][:10]))
                    if len(SECTOR_STOCKS[sector]["stocks"]) > 10:
                        st.write(f"... and {len(SECTOR_STOCKS[sector]['stocks']) - 10} more")
        
        custom_stocks = st.text_input(
            "Enter custom stock symbols",
            help="Example: AAPL, MSFT, GOOG"
        )
        
        scan_option = st.radio(
            "Scan option",
            ["All selected sector stocks", "Only custom stocks", "Both"],
            horizontal=True
        )
        
        # Determine which stocks to analyze
        stocks_to_scan = []
        if scan_option == "All selected sector stocks" or scan_option == "Both":
            for sector in selected_sectors:
                stocks_to_scan.extend(SECTOR_STOCKS[sector]["stocks"])
            # Remove duplicates while preserving order
            stocks_to_scan = list(dict.fromkeys(stocks_to_scan))
        
        if (scan_option == "Only custom stocks" or scan_option == "Both") and custom_stocks:
            custom_list = [stock.strip().upper() for stock in custom_stocks.split(",") if stock.strip()]
            if scan_option == "Only custom stocks":
                stocks_to_scan = custom_list
            else:
                # For "Both" option, add custom stocks and remove duplicates
                stocks_to_scan.extend(custom_list)
                stocks_to_scan = list(dict.fromkeys(stocks_to_scan))
        
        # Display the number of stocks to be scanned
        if stocks_to_scan:
            st.info(f"🔍 Ready to scan {len(stocks_to_scan)} stocks")
        else:
            st.warning("No stocks selected for scanning.")
    
    # TAB 3: ANALYSIS OPTIONS
    with tab3:
        # Time period selection
        time_period = st.selectbox(
            "Time period",
            ["1mo", "3mo", "6mo", "1y"],
            index=3,
            help="Data timeframe for technical analysis"
        )
        
        # Add filter preferences before scanning
        default_signal_types = st.multiselect(
            "Default signal types",
            ["BUY", "SHORT"],
            default=["BUY", "SHORT"]
        )
        
        default_confidence = st.multiselect(
            "Default confidence levels",
            ["HIGH", "MEDIUM"],
            default=["HIGH"]
        )
        
        # Scan button - Make it prominent
        scan_button = st.button("🔍 Scan for Trading Setups", type="primary", use_container_width=True)
    
    # Main content area
    if not scan_button:
        st.info("👈 Configure your scan settings, then click 'Scan for Trading Setups' to begin")
        st.image("https://images.unsplash.com/photo-1611974789855-9c2a0a7236a3?ixlib=rb-4.0.3&auto=format&fit=crop&w=1000&q=80", 
                 caption="Trading chart visualization")
    else:
        if not stocks_to_scan:
            st.warning("Please select at least one sector or enter custom stock symbols.")
        else:
            # Run the scan
            st.header("🔍 Scanning Results")
            progress_bar = st.progress(0)
            
            # Store results for stocks with patterns
            results = []
            
            for i, symbol in enumerate(stocks_to_scan):
                # Update progress
                progress = (i + 1) / len(stocks_to_scan)
                progress_bar.progress(progress)
                
                with st.spinner(f"Analyzing {symbol}..."):
                    analyzer = TechnicalAnalyzer(symbol, time_period)
                    if analyzer.fetch_data() and analyzer.calculate_indicators():
                        patterns = analyzer.detect_patterns()
                        if patterns:
                            results.append({
                                "symbol": symbol,
                                "patterns": patterns,
                                "analyzer": analyzer
                            })
            
            # Display results
            if results:
                st.success(f"Found trading setups in {len(results)} stocks!")
                
                # Filter options
                col1, col2 = st.columns(2)
                with col1:
                    signal_filter = st.multiselect(
                        "Filter by signal type",
                        ["BUY", "SHORT"],
                        default=default_signal_types
                    )
                
                with col2:
                    confidence_filter = st.multiselect(
                        "Filter by confidence level",
                        ["HIGH", "MEDIUM"],
                        default=default_confidence
                    )
                
                # Create a consolidated list of all setups for ranking
                all_setups = []
                for result in results:
                    symbol = result["symbol"]
                    patterns = result["patterns"]
                    
                    # Apply filters
                    filtered_patterns = {
                        name: info for name, info in patterns.items()
                        if info["signal"] in signal_filter and info["confidence"] in confidence_filter
                    }
                    
                    for pattern_name, pattern_info in filtered_patterns.items():
                        # Calculate risk-reward ratio (medium target)
                        if pattern_info["signal"] == "BUY":
                            risk = (pattern_info["entry"] - pattern_info["stop_loss"]) / pattern_info["entry"]
                            reward = (pattern_info["medium_target"] - pattern_info["entry"]) / pattern_info["entry"]
                        else:  # SHORT
                            risk = (pattern_info["stop_loss"] - pattern_info["entry"]) / pattern_info["entry"]
                            reward = (pattern_info["entry"] - pattern_info["medium_target"]) / pattern_info["entry"]
                        
                        risk_reward_ratio = reward / risk if risk > 0 else 0
                        
                        # Score setup based on confidence and risk-reward ratio
                        confidence_score = 10 if pattern_info["confidence"] == "HIGH" else 5
                        setup_score = confidence_score * risk_reward_ratio
                        
                        all_setups.append({
                            "symbol": symbol,
                            "pattern": pattern_name,
                            "signal": pattern_info["signal"],
                            "confidence": pattern_info["confidence"],
                            "entry": pattern_info["entry"],
                            "stop_loss": pattern_info["stop_loss"],
                            "medium_target": pattern_info["medium_target"],
                            "risk_reward": risk_reward_ratio,
                            "score": setup_score
                        })
                
                # Split into long and short setups
                long_setups = [setup for setup in all_setups if setup["signal"] == "BUY"]
                short_setups = [setup for setup in all_setups if setup["signal"] == "SHORT"]
                
                # Sort by score (highest first)
                long_setups.sort(key=lambda x: x["score"], reverse=True)
                short_setups.sort(key=lambda x: x["score"], reverse=True)
                
                # Show summary tabs for Long and Short setups
                setup_tabs = st.tabs(["🟢 Top BUY Setups", "🔴 Top SHORT Setups"])
                
                with setup_tabs[0]:
                    if long_setups:
                        st.write(f"### Top {min(10, len(long_setups))} BUY Setups")
                        
                        # Create a DataFrame for better display
                        long_df = pd.DataFrame([{
                            "Symbol": setup["symbol"],
                            "Pattern": setup["pattern"].replace("_", " "),
                            "Confidence": setup["confidence"],
                            "Entry": f"${setup['entry']:.2f}",
                            "Stop Loss": f"${setup['stop_loss']:.2f}",
                            "Target": f"${setup['medium_target']:.2f}",
                            "Risk/Reward": f"{setup['risk_reward']:.2f}"
                        } for setup in long_setups[:10]])
                        
                        # Add styling
                        st.dataframe(
                            long_df,
                            column_config={
                                "Symbol": st.column_config.TextColumn("Symbol", width="medium"),
                                "Pattern": st.column_config.TextColumn("Pattern", width="medium"),
                                "Confidence": st.column_config.TextColumn("Confidence", width="small"),
                                "Entry": st.column_config.TextColumn("Entry", width="small"),
                                "Stop Loss": st.column_config.TextColumn("Stop Loss", width="small"),
                                "Target": st.column_config.TextColumn("Target", width="small"),
                                "Risk/Reward": st.column_config.TextColumn("R/R", width="small"),
                            },
                            hide_index=True,
                            use_container_width=True
                        )
                    else:
                        st.info("No BUY setups found with the current filters.")
                
                with setup_tabs[1]:
                    if short_setups:
                        st.write(f"### Top {min(10, len(short_setups))} SHORT Setups")
                        
                        # Create a DataFrame for better display
                        short_df = pd.DataFrame([{
                            "Symbol": setup["symbol"],
                            "Pattern": setup["pattern"].replace("_", " "),
                            "Confidence": setup["confidence"],
                            "Entry": f"${setup['entry']:.2f}",
                            "Stop Loss": f"${setup['stop_loss']:.2f}",
                            "Target": f"${setup['medium_target']:.2f}",
                            "Risk/Reward": f"{setup['risk_reward']:.2f}"
                        } for setup in short_setups[:10]])
                        
                        # Add styling
                        st.dataframe(
                            short_df,
                            column_config={
                                "Symbol": st.column_config.TextColumn("Symbol", width="medium"),
                                "Pattern": st.column_config.TextColumn("Pattern", width="medium"),
                                "Confidence": st.column_config.TextColumn("Confidence", width="small"),
                                "Entry": st.column_config.TextColumn("Entry", width="small"),
                                "Stop Loss": st.column_config.TextColumn("Stop Loss", width="small"),
                                "Target": st.column_config.TextColumn("Target", width="small"),
                                "Risk/Reward": st.column_config.TextColumn("R/R", width="small"),
                            },
                            hide_index=True,
                            use_container_width=True
                        )
                    else:
                        st.info("No SHORT setups found with the current filters.")

            # Add section divider
            st.markdown("---")
            st.subheader("Detailed Analysis")
            st.write("Click on any stock below to see detailed chart and pattern analysis:")
            
            # Display individual detailed results
            for result in results:
                symbol = result["symbol"]
                patterns = result["patterns"]
                
                # Apply filters
                filtered_patterns = {
                    name: info for name, info in patterns.items()
                    if info["signal"] in signal_filter and info["confidence"] in confidence_filter
                }
                
                if filtered_patterns:
                    with st.expander(f"{symbol} - {len(filtered_patterns)} setups found"):
                        # Display chart
                        st.plotly_chart(result["analyzer"].create_chart(), use_container_width=True)
                        
                        # Display pattern details
                        for pattern_name, pattern_info in filtered_patterns.items():
                            signal_color = "green" if pattern_info["signal"] == "BUY" else "red"
                            confidence_color = "green" if pattern_info["confidence"] == "HIGH" else "orange"
                            
                            st.markdown(f"""
                            ### {pattern_name.replace('_', ' ')} Pattern
                            - Signal: <span style='color:{signal_color};font-weight:bold'>{pattern_info['signal']}</span>
                            - Confidence: <span style='color:{confidence_color};font-weight:bold'>{pattern_info['confidence']}</span>
                            - Entry Price: ${pattern_info['entry']:.2f}
                            - Stop Loss: ${pattern_info['stop_loss']:.2f} ({((pattern_info['stop_loss']/pattern_info['entry'])-1)*100:.1f}%)
                            - Short Target: ${pattern_info['short_target']:.2f} ({((pattern_info['short_target']/pattern_info['entry'])-1)*100:.1f}%)
                            - Medium Target: ${pattern_info['medium_target']:.2f} ({((pattern_info['medium_target']/pattern_info['entry'])-1)*100:.1f}%)
                            """, unsafe_allow_html=True)
            else:
                st.warning("No trading setups found with the current criteria. Try selecting different sectors or time period.")

# Run the app
if __name__ == "__main__":
    run()
