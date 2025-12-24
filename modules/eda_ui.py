"""
eda_ui.py - Exploratory Data Analysis Page

Bento grid layout for visualizations and target selection.
Preserves existing logic from views/page_upload_eda.py (visualization section).
"""

import streamlit as st
from typing import Optional

from .components import (
    render_page_header,
    render_section_header,
    render_metric_card,
    render_alert
)
from caching import cached_analyze_data, cached_detect_issues, get_df_hash
from data_utils import (
    plot_correlation_heatmap,
    plot_distributions,
    plot_categorical_distributions,
    plot_target_distribution
)


def page_eda() -> None:
    """
    Render the EDA page with visualizations and target selection.
    """
    render_page_header(
        "Exploratory Analysis",
        "Visualize distributions, correlations, and select your target variable"
    )
    
    # Removed Back navigation    # Guard clause without navigation
    if st.session_state.df is None:
        render_alert("Please upload a dataset first in the Upload section", "warning")
        return
    
    df = st.session_state.df
    df_hash = get_df_hash(df)
    
    # Get cached metadata
    with st.spinner("Loading analysis..."):
        metadata = cached_analyze_data(df_hash, df)
    
    # Target Selection Section
    render_section_header("Target Variable")
    
    target_col = st.selectbox(
        "Select the target column for classification",
        options=['-- Select Target --'] + df.columns.tolist(),
        index=0 if st.session_state.target_col is None else df.columns.tolist().index(st.session_state.target_col) + 1,
        label_visibility="collapsed"
    )
    
    if target_col and target_col != '-- Select Target --':
        st.session_state.target_col = target_col
        
        # Target distribution
        col1, col2 = st.columns([2, 1])
        with col1:
            st.plotly_chart(plot_target_distribution(df, target_col), width='stretch')
        with col2:
            # Class balance info
            value_counts = df[target_col].value_counts()
            render_section_header("Class Distribution")
            for cls, count in value_counts.items():
                pct = count / len(df) * 100
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; padding: 0.5rem 0; border-bottom: 1px solid var(--border-default);">
                    <span style="color: var(--text-primary);">{cls}</span>
                    <span style="color: var(--text-muted);">{count:,} ({pct:.1f}%)</span>
                </div>
                """, unsafe_allow_html=True)
    
    # Visual Analysis - Bento Grid
    render_section_header("Visual Analysis")
    
    tab1, tab2, tab3 = st.tabs(["Correlation Matrix", "Numerical Distributions", "Categorical Distributions"])
    
    with tab1:
        if metadata['numeric_columns']:
            st.plotly_chart(plot_correlation_heatmap(df), width='stretch')
            
            # Interpretation helper
            st.markdown("""
            <div style="background: var(--bg-card); border: 1px solid var(--border-default); border-radius: var(--radius-md); padding: 1rem; margin-top: 1rem;">
                <strong style="color: var(--text-primary);">How to interpret:</strong>
                <p style="color: var(--text-secondary); margin: 0.5rem 0 0 0; font-size: 0.875rem;">
                    Values close to +1 indicate strong positive correlation. Values close to -1 indicate strong negative correlation. 
                    Values near 0 suggest no linear relationship. Consider removing highly correlated features (|r| > 0.9) to reduce multicollinearity.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            render_alert("No numeric columns available for correlation analysis", "info")
    
    with tab2:
        if metadata['numeric_columns']:
            figs = plot_distributions(df)
            cols = st.columns(2)
            for i, (name, fig) in enumerate(figs[:6]):
                with cols[i % 2]:
                    st.plotly_chart(fig, width='stretch')
        else:
            render_alert("No numeric columns available for distribution plots", "info")
    
    with tab3:
        if metadata['categorical_columns']:
            figs = plot_categorical_distributions(df)
            cols = st.columns(2)
            for i, (name, fig) in enumerate(figs[:6]):
                with cols[i % 2]:
                    st.plotly_chart(fig, width='stretch')
        else:
            render_alert("No categorical columns available", "info")
    
    # Issue Detection (cached)
    render_section_header("Data Quality Summary")
    
    target_for_issues = target_col if target_col and target_col != '-- Select Target --' else None
    with st.spinner("Checking data quality..."):
        issues = cached_detect_issues(df_hash, df, target_for_issues)
    st.session_state.issues = issues
    
    if not issues['has_issues']:
        render_alert("No major data quality issues detected", "success")
    else:
        # Summary counts
        missing_count = len(issues.get('missing_values', {}))
        outlier_count = len(issues.get('outliers', {}))
        
        cols = st.columns(3)
        with cols[0]:
            render_metric_card(str(missing_count), "Columns with Missing")
        with cols[1]:
            render_metric_card(str(outlier_count), "Columns with Outliers")
        with cols[2]:
            imbalanced = issues.get('class_imbalance', {}).get('is_imbalanced', False)
            render_metric_card("Yes" if imbalanced else "No", "Class Imbalance")
    
    # Removed navigation buttons as per user request
    if not (target_col and target_col != '-- Select Target --'):
        render_alert("Please select a target column to proceed", "warning")
