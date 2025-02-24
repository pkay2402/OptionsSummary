import streamlit as st
from datetime import datetime, timedelta
import math
import matplotlib.pyplot as plt
import pandas as pd
import altair as alt

class GannCycleAnalyzer:
    def __init__(self):
        self.gann_cycles = [30, 45, 90, 120, 144, 180, 270, 360]
        self.gann_angles = {'1x1': 1, '2x1': 2, '1x2': 0.5}
    
    def analyze(self, stock_symbol, low_price, low_date):
        """
        Perform Gann cycle analysis based on the given inputs.
        
        Args:
            stock_symbol (str): The stock symbol
            low_price (float): The low price point
            low_date (str): Date of the low price in 'YYYY-MM-DD' format
            
        Returns:
            dict: Results of the Gann cycle analysis
        """
        start_date = datetime.strptime(low_date, '%Y-%m-%d')
        results = {
            'Stock Symbol': stock_symbol, 
            'Low Price': low_price, 
            'Low Date': low_date, 
            'Time Cycles': {}, 
            'Price Targets': {}
        }

        # Calculate time cycles
        for cycle in self.gann_cycles:
            target_date = start_date + timedelta(days=cycle)
            results['Time Cycles'][cycle] = target_date.strftime('%Y-%m-%d')

        # Calculate price targets for each angle
        for angle_name, angle_value in self.gann_angles.items():
            price_targets = {}
            for cycle in self.gann_cycles:
                price_move = angle_value * cycle
                target_price = low_price + price_move
                price_targets[cycle] = round(target_price, 2)
            results['Price Targets'][angle_name] = price_targets

        # Calculate square values
        results['Price Square'] = round(math.sqrt(low_price) ** 2, 2)
        results['Next Square Target'] = round((math.sqrt(low_price) + 1) ** 2, 2)
        
        return results
    
    def get_time_cycle_df(self, results):
        """Convert time cycle results to a DataFrame for display"""
        data = []
        for cycle, date in results['Time Cycles'].items():
            data.append({
                'Cycle (Days)': cycle,
                'Target Date': date
            })
        return pd.DataFrame(data)
    
    def get_price_target_df(self, results):
        """Convert price target results to a DataFrame for display"""
        data = []
        for angle, targets in results['Price Targets'].items():
            for cycle, price in targets.items():
                data.append({
                    'Angle': angle,
                    'Cycle (Days)': cycle,
                    'Target Date': results['Time Cycles'][cycle],
                    'Price Target ($)': price
                })
        return pd.DataFrame(data)
    
    def plot_interactive(self, results):
        """Create an interactive Altair chart for the results"""
        df = self.get_price_target_df(results)
        
        # Convert date strings to datetime for proper ordering
        df['Target Date'] = pd.to_datetime(df['Target Date'])
        
        # Create the base chart
        chart = alt.Chart(df).encode(
            x=alt.X('Target Date:T', title='Target Date'),
            y=alt.Y('Price Target ($):Q', title='Price Target ($)'),
            color=alt.Color('Angle:N', legend=alt.Legend(title='Gann Angle')),
            tooltip=['Angle', 'Cycle (Days)', 'Target Date', 'Price Target ($)']
        )
        
        # Create points and lines
        points = chart.mark_circle(size=60)
        lines = chart.mark_line()
        
        # Add horizontal lines for price square and next square
        price_square = alt.Chart(pd.DataFrame([{
            'y': results['Price Square'],
            'label': f"Price Square: ${results['Price Square']}"
        }])).mark_rule(color='gray', strokeDash=[4, 4]).encode(
            y='y:Q',
            tooltip=['label']
        )
        
        next_square = alt.Chart(pd.DataFrame([{
            'y': results['Next Square Target'],
            'label': f"Next Square: ${results['Next Square Target']}"
        }])).mark_rule(color='green', strokeDash=[4, 4]).encode(
            y='y:Q',
            tooltip=['label']
        )
        
        # Combine the charts
        return (points + lines + price_square + next_square).properties(
            title=f"Gann Cycle Analysis for {results['Stock Symbol']} (Low: ${results['Low Price']} on {results['Low Date']})",
            width=700,
            height=400
        ).configure_axis(
            labelFontSize=12,
            titleFontSize=14
        ).configure_title(
            fontSize=16
        )
    
    def plot_matplotlib(self, results):
        """Create a matplotlib visualization for the results"""
        cycles = list(results['Time Cycles'].keys())
        dates = [datetime.strptime(d, '%Y-%m-%d') for d in results['Time Cycles'].values()]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot price targets for each angle
        for angle, targets in results['Price Targets'].items():
            prices = [targets[cycle] for cycle in cycles]
            ax.plot(dates, prices, marker='o', linewidth=2, label=f'{angle} Angle')
        
        # Add horizontal lines for price square and next square target
        ax.axhline(y=results['Price Square'], color='gray', linestyle='--', linewidth=1.5, label=f"Price Square: ${results['Price Square']}")
        ax.axhline(y=results['Next Square Target'], color='green', linestyle='--', linewidth=1.5, label=f"Next Square: ${results['Next Square Target']}")
        
        # Format the plot
        ax.set_title(f"Gann Cycle Analysis for {results['Stock Symbol']}\n(Low: ${results['Low Price']} on {results['Low Date']})", fontsize=14, fontweight='bold')
        ax.set_xlabel("Target Dates", fontsize=12)
        ax.set_ylabel("Price Targets ($)", fontsize=12)
        
        # Format date ticks
        plt.gcf().autofmt_xdate()
        
        # Add grid, legend and adjust layout
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(fontsize=10, framealpha=0.8)
        plt.tight_layout()
        
        return fig

def run():
    """
    Main function to run the Gann Cycle Analysis module.
    This follows the same pattern as your other modules.
    """
    st.subheader("📈 Gann Cycle Analysis")
    
    # Create two columns for the layout
    input_col, info_col = st.columns([2, 1])
    
    with input_col:
        # Create input form
        with st.form("gann_input_form"):
            st.subheader("Enter Stock Details")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                stock_symbol = st.text_input("Stock Symbol", value="NVDA")
            
            with col2:
                low_price = st.number_input("Low Price ($)", min_value=0.01, value=90.69, format="%.2f")
            
            with col3:
                low_date = st.date_input("Date of Low", value=datetime(2024, 8, 5))
            
            submit_button = st.form_submit_button("Calculate Gann Cycles", use_container_width=True)
    
    with info_col:
        st.info("""
        ### Gann Cycle Analysis
        
        This tool calculates time and price projections based on W.D. Gann's methods.
        
        Enter a significant low point with its date to project future price targets 
        and time cycles.
        
        The analysis includes:
        - Key time cycles (30, 45, 90, 120, 144, 180, 270, 360 days)
        - Price targets using Gann angles (1x1, 2x1, 1x2)
        - Price square calculations
        """)
    
    # Store the analyzer in session state
    if 'gann_analyzer' not in st.session_state:
        st.session_state.gann_analyzer = GannCycleAnalyzer()
        st.session_state.gann_results = None
    
    # When form is submitted, perform analysis
    if submit_button:
        low_date_str = low_date.strftime('%Y-%m-%d')
        analyzer = st.session_state.gann_analyzer
        
        try:
            results = analyzer.analyze(stock_symbol, low_price, low_date_str)
            st.session_state.gann_results = results
            st.success(f"Analysis completed for {stock_symbol}")
        except Exception as e:
            st.error(f"Error performing analysis: {str(e)}")
    
    # Display results if available
    if st.session_state.gann_results:
        results = st.session_state.gann_results
        analyzer = st.session_state.gann_analyzer
        
        # Display header with stock info
        st.markdown(f"""
        <div style="background-color:#f0f2f6; padding:10px; border-radius:5px; margin: 10px 0;">
            <h3 style="margin:0">Analysis for {results['Stock Symbol']}</h3>
            <p style="margin:5px 0 0 0">Low Price: <b>${results['Low Price']}</b> on <b>{results['Low Date']}</b></p>
        </div>
        """, unsafe_allow_html=True)
        
        tabs = st.tabs(["Visualization", "Time Cycles", "Price Targets", "Export"])
        
        with tabs[0]:
            # Show visualization (both interactive and static options)
            visual_type = st.radio("Chart Type:", ["Interactive", "Static"], horizontal=True)
            
            if visual_type == "Interactive":
                st.altair_chart(analyzer.plot_interactive(results), use_container_width=True)
            else:
                fig = analyzer.plot_matplotlib(results)
                st.pyplot(fig)
            
            # Display square values below the chart
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Price Square", f"${results['Price Square']}")
            with col2:
                st.metric("Next Square Target", f"${results['Next Square Target']}", 
                         delta=f"{round(results['Next Square Target'] - results['Price Square'], 2)}")
        
        with tabs[1]:
            # Display time cycle data
            time_cycle_df = analyzer.get_time_cycle_df(results)
            st.dataframe(time_cycle_df, use_container_width=True, hide_index=True)
        
        with tabs[2]:
            # Display price targets with option to filter by angle
            price_df = analyzer.get_price_target_df(results)
            
            angle_filter = st.selectbox("Filter by Angle:", ["All"] + list(results['Price Targets'].keys()))
            
            if angle_filter != "All":
                filtered_df = price_df[price_df['Angle'] == angle_filter]
            else:
                filtered_df = price_df
            
            st.dataframe(
                filtered_df.sort_values(['Angle', 'Cycle (Days)']),
                use_container_width=True,
                hide_index=True
            )
        
        with tabs[3]:
            # Export options
            st.download_button(
                label="Download Full Analysis as CSV",
                data=price_df.to_csv(index=False),
                file_name=f"gann_analysis_{results['Stock Symbol']}_{results['Low Date']}.csv",
                mime="text/csv"
            )
            
            # Also offer option for Excel
            # (This would need additional processing in a real app)
            st.download_button(
                label="Download as Excel",
                data=price_df.to_csv(index=False),  # Placeholder - would be Excel in real app
                file_name=f"gann_analysis_{results['Stock Symbol']}_{results['Low Date']}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

# For testing the module directly
if __name__ == "__main__":
    run()
