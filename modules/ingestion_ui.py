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
    render_alert
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
    
    uploaded_file = st.file_uploader(
        "Drag and drop your CSV file here, or click to browse",
        type=['csv'],
        help="Supported format: CSV",
        label_visibility="collapsed"
    )
    
    if uploaded_file:
        try:
            with st.status("Processing upload...", expanded=True) as status:
                st.write("Loading dataset...")
                df = load_csv(uploaded_file)
                st.write("Validating data structure...")
                
                # Store in session state
                st.session_state.df = df
                st.session_state.file_name = uploaded_file.name
            st.balloons()  # Process completion animation
            st.rerun()
            st.session_state.df_clean = None
            st.session_state.results = None
            
            status.update(label="Upload complete", state="complete")
            
            st.toast(f"Loaded {uploaded_file.name} successfully", icon="✅")
            
        except Exception as e:
            render_alert(f"Error loading file: {str(e)}", "error")
            return
    
    # Guard clause
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
    
    # Removed intra-page navigation as per user request
    pass
