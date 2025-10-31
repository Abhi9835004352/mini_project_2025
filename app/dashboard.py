"""
Streamlit Dashboard for FOGA vs HBRF Compiler Optimization Comparison
A modern, interactive web interface for comparing compiler flag optimization approaches

Run with: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import os
import subprocess
import time
from datetime import datetime
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Compiler Optimizer Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #2e3340;
    }
    .winner-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin: 20px 0;
    }
    .section-header {
        color: #667eea;
        font-size: 28px;
        font-weight: bold;
        margin: 30px 0 20px 0;
        border-bottom: 3px solid #667eea;
        padding-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'comparison_data' not in st.session_state:
    st.session_state.comparison_data = None
if 'optimization_running' not in st.session_state:
    st.session_state.optimization_running = False

def load_comparison_data():
    """Load comparison results from JSON file"""
    if os.path.exists('comparison_results.json'):
        with open('comparison_results.json', 'r') as f:
            return json.load(f)
    return None

def load_hbrf_data():
    """Load HBRF detailed results"""
    if os.path.exists('hbrf_results.json'):
        with open('hbrf_results.json', 'r') as f:
            return json.load(f)
    return None

def create_execution_time_chart(data):
    """Create interactive execution time comparison chart"""
    methods = ['-O1', '-O2', '-O3', 'FOGA', 'HBRF']
    times = []
    colors = []
    
    for method in methods:
        if method.startswith('-'):
            time_val = data['baseline'].get(method, float('inf'))
            colors.append('#3498db' if method == '-O1' else '#2ecc71' if method == '-O2' else '#f39c12')
        else:
            time_val = data[method].get('best_time', float('inf'))
            colors.append('#e74c3c' if method == 'FOGA' else '#9b59b6')
        
        times.append(time_val if time_val != float('inf') else 0)
    
    fig = go.Figure(data=[
        go.Bar(
            x=methods,
            y=times,
            marker_color=colors,
            text=[f'{t:.6f}s' if t > 0 else 'Failed' for t in times],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Time: %{y:.6f}s<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title={
            'text': '⚡ Execution Time Comparison',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24, 'color': '#667eea'}
        },
        xaxis_title='Optimization Method',
        yaxis_title='Execution Time (seconds)',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=500,
        showlegend=False,
        hovermode='x unified'
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    
    return fig

def create_optimization_time_chart(data):
    """Create optimization time comparison"""
    fig = go.Figure()
    
    foga_time = data['FOGA'].get('total_time', 0)
    hbrf_time = data['HBRF'].get('total_time', 0)
    
    fig.add_trace(go.Bar(
        x=['FOGA', 'HBRF'],
        y=[foga_time, hbrf_time],
        marker_color=['#e74c3c', '#9b59b6'],
        text=[f'{foga_time:.1f}s', f'{hbrf_time:.1f}s'],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Time: %{y:.2f}s<extra></extra>'
    ))
    
    fig.update_layout(
        title={
            'text': '⏱️ Optimization Time Comparison',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24, 'color': '#667eea'}
        },
        xaxis_title='Method',
        yaxis_title='Time (seconds)',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=400,
        showlegend=False
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    
    return fig

def create_speedup_chart(data):
    """Create speedup comparison chart"""
    o3_time = data['baseline'].get('-O3', float('inf'))
    
    if o3_time == float('inf') or o3_time == 0:
        return None
    
    methods = ['-O1', '-O2', 'FOGA', 'HBRF']
    speedups = []
    
    for method in methods:
        if method.startswith('-'):
            time_val = data['baseline'].get(method, float('inf'))
        else:
            time_val = data[method].get('best_time', float('inf'))
        
        if time_val != float('inf') and time_val > 0:
            speedup = ((o3_time - time_val) / o3_time) * 100
            speedups.append(speedup)
        else:
            speedups.append(0)
    
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
    
    fig = go.Figure(data=[
        go.Bar(
            x=methods,
            y=speedups,
            marker_color=colors,
            text=[f'{s:+.2f}%' for s in speedups],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Speedup: %{y:+.2f}%<extra></extra>'
        )
    ])
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    fig.update_layout(
        title={
            'text': '📈 Speedup vs -O3 Baseline',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24, 'color': '#667eea'}
        },
        xaxis_title='Method',
        yaxis_title='Speedup (%)',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=400,
        showlegend=False
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    
    return fig

def create_evaluations_chart(data):
    """Create evaluations comparison"""
    foga_evals = data['FOGA'].get('evaluations', 0)
    hbrf_evals = data['HBRF'].get('evaluations', 0)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=['FOGA', 'HBRF'],
        y=[foga_evals, hbrf_evals],
        marker_color=['#e74c3c', '#9b59b6'],
        text=[str(foga_evals), str(hbrf_evals)],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Evaluations: %{y}<extra></extra>'
    ))
    
    fig.update_layout(
        title={
            'text': '🔢 Total Evaluations',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24, 'color': '#667eea'}
        },
        xaxis_title='Method',
        yaxis_title='Number of Evaluations',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=400,
        showlegend=False
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    
    return fig

def create_flag_importance_chart(hbrf_data):
    """Create flag importance visualization"""
    if not hbrf_data or 'flag_importance' not in hbrf_data:
        return None
    
    importance = hbrf_data['flag_importance']
    sorted_flags = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:15]
    
    flags = [f[0] for f in sorted_flags]
    values = [f[1] for f in sorted_flags]
    
    fig = go.Figure(data=[
        go.Bar(
            y=flags,
            x=values,
            orientation='h',
            marker_color='#9b59b6',
            text=[f'{v:.4f}' for v in values],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title={
            'text': '🎯 Top 15 Most Important Flags (HBRF Analysis)',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24, 'color': '#667eea'}
        },
        xaxis_title='Importance Score',
        yaxis_title='Compiler Flag',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=600,
        showlegend=False
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)')
    fig.update_yaxes(showgrid=False)
    
    return fig

def create_efficiency_radar(data):
    """Create radar chart for efficiency comparison"""
    o3_time = data['baseline'].get('-O3', 1)
    
    foga_exec = data['FOGA'].get('best_time', float('inf'))
    hbrf_exec = data['HBRF'].get('best_time', float('inf'))
    
    foga_opt_time = data['FOGA'].get('total_time', 1)
    hbrf_opt_time = data['HBRF'].get('total_time', 1)
    
    foga_evals = data['FOGA'].get('evaluations', 1)
    hbrf_evals = data['HBRF'].get('evaluations', 1)
    
    # Normalize scores (0-100, higher is better)
    categories = ['Execution<br>Speed', 'Optimization<br>Speed', 'Sample<br>Efficiency', 'Simplicity', 'Robustness']
    
    foga_scores = [
        100 * (o3_time / foga_exec) if foga_exec != float('inf') else 0,  # Execution
        100 * (hbrf_opt_time / foga_opt_time),  # Optimization speed
        100 * (hbrf_evals / foga_evals),  # Sample efficiency
        70,  # Simplicity (GA is simpler conceptually)
        90   # Robustness (GA is more robust)
    ]
    
    hbrf_scores = [
        100 * (o3_time / hbrf_exec) if hbrf_exec != float('inf') else 0,  # Execution
        100,  # Optimization speed (baseline)
        100,  # Sample efficiency (baseline)
        60,   # Simplicity (HBRF is more complex)
        80    # Robustness
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=foga_scores,
        theta=categories,
        fill='toself',
        name='FOGA',
        line_color='#e74c3c',
        fillcolor='rgba(231, 76, 60, 0.3)'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=hbrf_scores,
        theta=categories,
        fill='toself',
        name='HBRF',
        line_color='#9b59b6',
        fillcolor='rgba(155, 89, 182, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 110],
                gridcolor='rgba(128,128,128,0.2)'
            ),
            bgcolor='rgba(0,0,0,0)'
        ),
        showlegend=True,
        title={
            'text': '🎯 Multi-Dimensional Efficiency Analysis',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24, 'color': '#667eea'}
        },
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

# Main app
def main():
    # Header
    st.markdown("""
        <h1 style='text-align: center; color: #667eea; font-size: 48px; margin-bottom: 10px;'>
            🚀 Compiler Optimizer Comparison Dashboard
        </h1>
        <p style='text-align: center; color: #a0a0a0; font-size: 18px; margin-bottom: 40px;'>
            FOGA (Genetic Algorithm) vs HBRF (Hybrid Bayesian-Random Forest)
        </p>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/speed.png", width=80)
        st.markdown("## 🎛️ Control Panel")
        
        st.markdown("---")
        
        # File upload
        st.markdown("### 📁 Upload Source File")
        uploaded_file = st.file_uploader("Choose a C/C++ file", type=['c', 'cpp', 'cc'])
        
        if uploaded_file:
            with open(f"uploaded_{uploaded_file.name}", "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"✅ {uploaded_file.name} uploaded!")
        
        st.markdown("---")
        
        # Run optimization
        st.markdown("### ⚙️ Run Optimization")
        
        if st.button("🧬 Run FOGA Only", use_container_width=True):
            if uploaded_file:
                with st.spinner("Running FOGA..."):
                    subprocess.run(['python3', 'foga.py', f'uploaded_{uploaded_file.name}'])
                st.success("FOGA completed!")
            else:
                st.error("Please upload a file first")
        
        if st.button("🔬 Run HBRF Only", use_container_width=True):
            if uploaded_file:
                with st.spinner("Running HBRF..."):
                    subprocess.run(['python3', 'hbrf_optimizer.py', f'uploaded_{uploaded_file.name}'])
                st.success("HBRF completed!")
                st.session_state.comparison_data = load_comparison_data()
            else:
                st.error("Please upload a file first")
        
        if st.button("🏁 Run Full Comparison", type="primary", use_container_width=True):
            if uploaded_file:
                with st.spinner("Running full comparison... This may take a while..."):
                    subprocess.run(['python3', 'compare_optimizers.py', f'uploaded_{uploaded_file.name}'])
                st.success("Comparison completed!")
                st.session_state.comparison_data = load_comparison_data()
                st.rerun()
            else:
                st.error("Please upload a file first")
        
        st.markdown("---")
        
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.session_state.comparison_data = load_comparison_data()
            st.rerun()
        
        st.markdown("---")
        st.markdown("### 📊 About")
        st.info("""
        **FOGA**: Genetic Algorithm approach from research paper
        
        **HBRF**: Novel hybrid approach combining Random Forests and Bayesian Optimization
        
        Compares 114 GCC compiler flags across multiple dimensions.
        """)
    
    # Main content
    data = st.session_state.comparison_data or load_comparison_data()
    
    if data is None:
        st.markdown("""
            <div style='text-align: center; padding: 100px 20px;'>
                <h2 style='color: #667eea;'>👋 Welcome!</h2>
                <p style='color: #a0a0a0; font-size: 18px;'>
                    Upload a C/C++ file and run the comparison to see results here.
                </p>
                <p style='color: #a0a0a0;'>
                    Or if you've already run a comparison, click 'Refresh Data' in the sidebar.
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # Show example/demo mode
        with st.expander("📚 See Example Results (Demo Mode)", expanded=False):
            st.info("This shows example data. Upload your file to see real results!")
            data = {
                'baseline': {'-O1': 2.5, '-O2': 1.8, '-O3': 1.2},
                'FOGA': {'best_time': 1.15, 'total_time': 2700, 'evaluations': 2770},
                'HBRF': {'best_time': 1.14, 'total_time': 240, 'evaluations': 160},
                'winner': 'HBRF'
            }
    
    if data:
        # Winner announcement
        winner = data.get('winner', 'Unknown')
        if winner != 'TIE':
            st.markdown(f"""
                <div class='winner-card'>
                    🏆 WINNER: {winner}
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class='winner-card' style='background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);'>
                    🤝 RESULT: TIE
                </div>
            """, unsafe_allow_html=True)
        
        # Key metrics
        st.markdown("<div class='section-header'>📊 Key Performance Indicators</div>", unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            foga_time = data['FOGA'].get('best_time', 0)
            st.metric(
                "🧬 FOGA Execution Time",
                f"{foga_time:.6f}s",
                delta=None
            )
        
        with col2:
            hbrf_time = data['HBRF'].get('best_time', 0)
            delta = ((foga_time - hbrf_time) / foga_time * 100) if foga_time > 0 else 0
            st.metric(
                "🔬 HBRF Execution Time",
                f"{hbrf_time:.6f}s",
                delta=f"{delta:.2f}% vs FOGA",
                delta_color="inverse"
            )
        
        with col3:
            foga_opt = data['FOGA'].get('total_time', 0)
            st.metric(
                "⏱️ FOGA Opt. Time",
                f"{foga_opt:.1f}s"
            )
        
        with col4:
            hbrf_opt = data['HBRF'].get('total_time', 0)
            speedup = (foga_opt / hbrf_opt) if hbrf_opt > 0 else 0
            st.metric(
                "⏱️ HBRF Opt. Time",
                f"{hbrf_opt:.1f}s",
                delta=f"{speedup:.1f}x faster"
            )
        
        # Charts
        st.markdown("<div class='section-header'>📈 Performance Visualizations</div>", unsafe_allow_html=True)
        
        tab1, tab2, tab3, tab4 = st.tabs(["⚡ Execution Times", "🎯 Analysis", "🔢 Evaluations", "🏆 Efficiency"])
        
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(create_execution_time_chart(data), use_container_width=True)
            with col2:
                speedup_chart = create_speedup_chart(data)
                if speedup_chart:
                    st.plotly_chart(speedup_chart, use_container_width=True)
        
        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(create_optimization_time_chart(data), use_container_width=True)
            with col2:
                st.plotly_chart(create_evaluations_chart(data), use_container_width=True)
            
            # Flag importance
            hbrf_data = load_hbrf_data()
            if hbrf_data:
                flag_chart = create_flag_importance_chart(hbrf_data)
                if flag_chart:
                    st.plotly_chart(flag_chart, use_container_width=True)
        
        with tab3:
            foga_evals = data['FOGA'].get('evaluations', 0)
            hbrf_evals = data['HBRF'].get('evaluations', 0)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("FOGA Evaluations", f"{foga_evals:,}")
            with col2:
                st.metric("HBRF Evaluations", f"{hbrf_evals:,}")
            with col3:
                reduction = ((foga_evals - hbrf_evals) / foga_evals * 100) if foga_evals > 0 else 0
                st.metric("Reduction", f"{reduction:.1f}%")
            
            st.plotly_chart(create_evaluations_chart(data), use_container_width=True)
        
        with tab4:
            st.plotly_chart(create_efficiency_radar(data), use_container_width=True)
            
            # Detailed comparison table
            st.markdown("### 📋 Detailed Comparison")
            comparison_df = pd.DataFrame({
                'Metric': ['Execution Time', 'Optimization Time', 'Evaluations', 'Sample Efficiency', 'Speedup vs -O3'],
                'FOGA': [
                    f"{data['FOGA'].get('best_time', 0):.6f}s",
                    f"{data['FOGA'].get('total_time', 0):.1f}s",
                    f"{data['FOGA'].get('evaluations', 0):,}",
                    "Baseline",
                    f"{data.get('improvements', {}).get('FOGA_vs_O3', 0):.2f}%"
                ],
                'HBRF': [
                    f"{data['HBRF'].get('best_time', 0):.6f}s",
                    f"{data['HBRF'].get('total_time', 0):.1f}s",
                    f"{data['HBRF'].get('evaluations', 0):,}",
                    f"{((data['FOGA'].get('evaluations', 1) / data['HBRF'].get('evaluations', 1))):.1f}x better",
                    f"{data.get('improvements', {}).get('HBRF_vs_O3', 0):.2f}%"
                ]
            })
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        
        # Download section
        st.markdown("<div class='section-header'>💾 Export Results</div>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if os.path.exists('comparison_results.json'):
                with open('comparison_results.json', 'r') as f:
                    st.download_button(
                        "📄 Download JSON",
                        f.read(),
                        "comparison_results.json",
                        "application/json",
                        use_container_width=True
                    )
        
        with col2:
            if os.path.exists('comparison_chart.png'):
                with open('comparison_chart.png', 'rb') as f:
                    st.download_button(
                        "📊 Download Chart",
                        f.read(),
                        "comparison_chart.png",
                        "image/png",
                        use_container_width=True
                    )
        
        with col3:
            # Generate report
            report = f"""
            Compiler Optimization Comparison Report
            ========================================
            Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            
            Winner: {data.get('winner', 'Unknown')}
            
            Execution Times:
            - FOGA: {data['FOGA'].get('best_time', 0):.6f}s
            - HBRF: {data['HBRF'].get('best_time', 0):.6f}s
            
            Optimization Times:
            - FOGA: {data['FOGA'].get('total_time', 0):.1f}s
            - HBRF: {data['HBRF'].get('total_time', 0):.1f}s
            
            Evaluations:
            - FOGA: {data['FOGA'].get('evaluations', 0)}
            - HBRF: {data['HBRF'].get('evaluations', 0)}
            """
            
            st.download_button(
                "📝 Download Report",
                report,
                "optimization_report.txt",
                "text/plain",
                use_container_width=True
            )

if __name__ == "__main__":
    main()