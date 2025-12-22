"""
page_upload_eda.py - Upload and EDA Page
Data upload, exploratory data analysis, and issue detection
"""

import streamlit as st
import pandas as pd
from ui_components import render_header, render_section_header, render_metric_card, render_alert
from caching import load_csv, cached_analyze_data, cached_detect_issues, get_df_hash
from data_utils import plot_correlation_heatmap, plot_distributions, plot_categorical_distributions, plot_target_distribution

def page_upload_eda():
    render_header("📤", "Upload & Analyze", "Upload your dataset and explore its characteristics")
    
    # File Upload
    render_section_header("📁", "Dataset Upload")
    
    uploaded_file = st.file_uploader(
        "Drag and drop your CSV file here",
        type=['csv'],
        help="Supported format: CSV"
    )
    
    if uploaded_file:
        try:
            with st.spinner("Loading dataset... This may take a moment for large files."):
                df = load_csv(uploaded_file)
            st.session_state.df = df
            st.session_state.file_name = uploaded_file.name
            st.session_state.df_clean = None
            st.session_state.results = None
            render_alert(f"Successfully loaded {uploaded_file.name} ({len(df):,} rows)", "success")
        except Exception as e:
            render_alert(f"Error loading file: {str(e)}", "error")
            return
    
    if st.session_state.df is None:
        render_alert("Upload a CSV file to begin your machine learning journey", "info")
        return
    
    df = st.session_state.df
    df_hash = get_df_hash(df)
    
    # Use cached analysis
    with st.spinner("Analyzing dataset..."):
        metadata = cached_analyze_data(df_hash, df)
    
    # Metrics Row
    render_section_header("📊", "Dataset Overview")
    
    cols = st.columns(4)
    metrics = [
        (f"{metadata['rows']:,}", "Total Rows", "📈"),
        (f"{metadata['columns']}", "Columns", "📋"),
        (f"{len(metadata['numeric_columns'])}", "Numeric", "🔢"),
        (f"{len(metadata['categorical_columns'])}", "Categorical", "🏷️")
    ]
    for col, (value, label, icon) in zip(cols, metrics):
        with col:
            render_metric_card(value, label, icon)
    
    # Data Preview
    render_section_header("👀", "Data Preview")
    with st.expander("View First 10 Rows", expanded=True):
        st.dataframe(df.head(10), width='stretch', height=350)
    
    with st.expander("Column Information"):
        col_info = pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes.astype(str).values,
            'Non-Null': df.count().values,
            'Missing': df.isnull().sum().values,
            'Unique': df.nunique().values
        })
        st.dataframe(col_info, width='stretch')
    
    # Target Selection
    render_section_header("🎯", "Target Column")
    
    target_col = st.selectbox(
        "Select the target variable for classification:",
        options=['-- Select --'] + df.columns.tolist(),
        index=0 if st.session_state.target_col is None else df.columns.tolist().index(st.session_state.target_col) + 1
    )
    
    if target_col and target_col != '-- Select --':
        st.session_state.target_col = target_col
        st.plotly_chart(plot_target_distribution(df, target_col), width='stretch')
    
    # Issue Detection (cached)
    render_section_header("⚡", "Data Quality Check")
    
    target_for_issues = target_col if target_col and target_col != '-- Select --' else None
    with st.spinner("Checking data quality..."):
        issues = cached_detect_issues(df_hash, df, target_for_issues)
    st.session_state.issues = issues
    
    if not issues['has_issues']:
        render_alert("Excellent! No major data quality issues detected.", "success")
    else:
        if issues['missing_values']:
            with st.expander(f"🔴 Missing Values ({len(issues['missing_values'])} columns)", expanded=True):
                for col, info in issues['missing_values'].items():
                    st.markdown(f"""
                    <div style="display: flex; justify-content: space-between; align-items: center; padding: 0.75rem; background: rgba(239,68,68,0.1); border-radius: 8px; margin-bottom: 0.5rem;">
                        <span style="color: #fca5a5; font-weight: 500;">{col}</span>
                        <span style="color: rgba(255,255,255,0.6);">{info['count']} missing ({info['percentage']}%)</span>
                    </div>
                    """, unsafe_allow_html=True)
        
        if issues['outliers']:
            with st.expander(f"🟠 Outliers Detected ({len(issues['outliers'])} columns)"):
                for col, info in issues['outliers'].items():
                    st.markdown(f"""
                    <div style="display: flex; justify-content: space-between; padding: 0.75rem; background: rgba(245,158,11,0.1); border-radius: 8px; margin-bottom: 0.5rem;">
                        <span style="color: #fcd34d; font-weight: 500;">{col}</span>
                        <span style="color: rgba(255,255,255,0.6);">{info['count']} outliers ({info['percentage']}%)</span>
                    </div>
                    """, unsafe_allow_html=True)
        
        if 'class_imbalance' in issues and issues['class_imbalance'].get('is_imbalanced'):
            render_alert(f"Class imbalance detected: {issues['class_imbalance']['majority_ratio']}% majority class", "warning")
    
    # Visualizations
    render_section_header("📈", "Visual Analysis")
    
    tab1, tab2, tab3 = st.tabs(["🔥 Correlation", "📊 Distributions", "🏷️ Categories"])
    
    with tab1:
        if metadata['numeric_columns']:
            st.plotly_chart(plot_correlation_heatmap(df), width='stretch')
        else:
            render_alert("No numeric columns for correlation analysis", "info")
    
    with tab2:
        if metadata['numeric_columns']:
            figs = plot_distributions(df)
            cols = st.columns(2)
            for i, (name, fig) in enumerate(figs[:6]):
                with cols[i % 2]:
                    st.plotly_chart(fig, width='stretch')
        else:
            render_alert("No numeric columns for distribution plots", "info")
    
    with tab3:
        if metadata['categorical_columns']:
            figs = plot_categorical_distributions(df)
            cols = st.columns(2)
            for i, (name, fig) in enumerate(figs[:6]):
                with cols[i % 2]:
                    st.plotly_chart(fig, width='stretch')
        else:
            render_alert("No categorical columns", "info")
    
    # Proceed
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if target_col and target_col != '-- Select --':
            if st.button("🚀 Continue to Preprocessing", width='stretch', type="primary"):
                st.session_state.current_page = "Preprocessing"
                st.rerun()
        else:
            render_alert("Please select a target column to proceed", "warning")



