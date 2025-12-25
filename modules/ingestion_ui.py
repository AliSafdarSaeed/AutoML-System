"""
ingestion_ui.py - Data Ingestion Page

Handles file upload with drop zone and displays Data Health Dashboard.
Preserves existing logic from views/page_upload_eda.py (upload section only).
"""

import streamlit as st
import pandas as pd
from typing import Optional

from .components import (
    render_page_header,
    render_section_header,
    render_metric_card,
    render_alert,
    render_proceed_button
)
from caching import load_csv, cached_analyze_data, get_df_hash


def page_ingestion() -> None:
    """
    Render the data ingestion page with file upload and health dashboard.
    """
    render_page_header(
        "Data Ingestion",
        "Upload your dataset to begin the analysis pipeline"
    )
    
    # Drop Zone Section
    render_section_header("Upload Dataset")
    
    # Check if we need to show a new file uploader or if data already loaded
    current_file = st.session_state.get('file_name')
    
    uploaded_file = st.file_uploader(
        "Drag and drop your CSV file here, or click to browse",
        type=['csv'],
        help="Supported format: CSV",
        label_visibility="collapsed"
    )
    
    # Process new file upload
    if uploaded_file is not None:
        # Check if this is a new file (different from what's already loaded)
        is_new_file = (current_file != uploaded_file.name) or (st.session_state.df is None)
        
        if is_new_file:
            try:
                with st.status("Processing upload...", expanded=True) as status:
                    st.write("Loading dataset...")
                    df = load_csv(uploaded_file)
                    st.write("Validating data structure...")
                    st.write(f"Found {df.shape[0]:,} rows and {df.shape[1]} columns")
                    
                    # Store in session state
                    st.session_state.df = df
                    st.session_state.file_name = uploaded_file.name
                    # Reset downstream states when new file is uploaded
                    st.session_state.df_clean = None
                    st.session_state.results = None
                    st.session_state.target_col = None
                    st.session_state.issues = None
                    st.session_state.preprocess_config = {}
                    
                    status.update(label="Upload complete!", state="complete")
                
                st.toast(f"Loaded {uploaded_file.name} successfully", icon="✅")
                
            except Exception as e:
                render_alert(f"Error loading file: {str(e)}", "error")
                return
    
    # Guard clause - show info if no data
    if st.session_state.df is None:
        render_alert("Upload a CSV file to continue", "info")
        return
    
    df = st.session_state.df
    df_hash = get_df_hash(df)
    
    # Data Health Dashboard
    with st.spinner("Analyzing dataset..."):
        metadata = cached_analyze_data(df_hash, df)
    
    render_section_header("Data Health Dashboard")
    
    # KPI Metrics Row
    cols = st.columns(4)
    
    with cols[0]:
        render_metric_card(f"{metadata['rows']:,}", "Total Rows")
    
    with cols[1]:
        render_metric_card(f"{metadata['columns']}", "Columns")
    
    with cols[2]:
        # Calculate missing percentage
        total_cells = metadata['rows'] * metadata['columns']
        missing_cells = df.isnull().sum().sum()
        missing_pct = (missing_cells / total_cells * 100) if total_cells > 0 else 0
        render_metric_card(f"{missing_pct:.1f}%", "Missing Data")
    
    with cols[3]:
        render_metric_card(f"{len(metadata['numeric_columns'])}", "Numeric")
    
    # Data Preview
    render_section_header("Data Preview")
    
    with st.expander("First 10 Rows", expanded=True):
        st.dataframe(df.head(10), use_container_width=True, height=350)
    
    with st.expander("Column Information"):
        col_info = pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes.astype(str).values,
            'Non-Null': df.count().values,
            'Missing': df.isnull().sum().values,
            'Unique': df.nunique().values
        })
        st.dataframe(col_info, use_container_width=True)
    
    with st.expander("Summary Statistics"):
        # Get summary statistics for numerical columns
        numeric_df = df.select_dtypes(include=['float64', 'int64'])
        if not numeric_df.empty:
            summary_stats = numeric_df.describe().T
            summary_stats = summary_stats.round(2)
            st.dataframe(summary_stats, use_container_width=True)
        else:
            st.info("No numerical columns found in the dataset.")
    
    # Proceed to next step button
    render_proceed_button(
        next_page="EDA",
        label="Proceed to Explore Data",
        disabled=False
    )
