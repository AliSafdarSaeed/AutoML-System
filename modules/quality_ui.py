"""
quality_ui.py - Data Quality Gate Page

System check interface with issue cards and approval gate.
Preserves existing logic from views/page_preprocessing.py.
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any

from .components import (
    render_page_header,
    render_section_header,
    render_metric_card,
    render_alert,
    render_severity_badge,
    render_approval_gate
)
from data_utils import handle_outliers, apply_preprocessing, detect_issues
from .recommendations import (
    get_missing_value_recommendation,
    get_outlier_recommendation,
    get_encoding_recommendation
)


def _render_issue_card(
    col_name: str,
    issue_type: str,
    count: int,
    percentage: float,
    severity: str = "warning"
) -> None:
    """
    Render a single issue card row.
    
    Args:
        col_name: Column name with the issue
        issue_type: Type of issue (e.g., "missing", "outliers")
        count: Number of affected rows
        percentage: Percentage affected
        severity: 'critical' or 'warning'
    """
    badge_html = render_severity_badge(
        "CRITICAL" if severity == "critical" else "WARNING",
        severity
    )
    
    st.markdown(f"""
    <div class="issue-card">
        <div class="issue-card-left">
            <span class="issue-card-name">{col_name}</span>
            <span class="issue-card-detail">{count:,} {issue_type} ({percentage:.1f}%)</span>
        </div>
        <div>{badge_html}</div>
    </div>
    """, unsafe_allow_html=True)


def page_quality() -> None:
    """
    Render the data quality gate page with issue detection and fixes.
    """
    render_page_header(
        "Data Quality Check",
        "Review detected issues and configure preprocessing before training"
    )
    
    # Guard clauses without navigation buttons
    if st.session_state.df is None:
        render_alert("Please upload a dataset first in the Upload section", "warning")
        return
    
    if st.session_state.target_col is None:
        render_alert("Please select a target variable in the Explore section", "warning")
        return
    
    df = st.session_state.df.copy()
    target_col = st.session_state.target_col
    issues = st.session_state.issues or detect_issues(df, target_col)
    
    render_alert(
        "Review the detected issues below. Configure fixes and approve before proceeding to training.",
        "info"
    )
    
    config = st.session_state.preprocess_config.copy()
    
    # ===== MISSING VALUES =====
    render_section_header("Missing Values")
    
    missing_strategies = {}
    if issues.get('missing_values'):
        for col, info in issues['missing_values'].items():
            severity = "critical" if info['percentage'] > 20 else "warning"
            _render_issue_card(col, "missing", info['count'], info['percentage'], severity)
            
            # Get AI recommendation
            unique_ratio = df[col].nunique() / len(df) if len(df) > 0 else 0
            recommendation, reasoning, suggested_method = get_missing_value_recommendation(
                col, str(df[col].dtype), info['percentage'], unique_ratio
            )
            
            with st.expander(f"💡 {col} - Fix Configuration"):
                # Show smart recommendation
                st.info(f"{recommendation}\n\n**Reasoning:** {reasoning}", icon="🤖")
                
                c1, c2 = st.columns([1, 2])
                with c1:
                    fix = st.checkbox(
                        "Apply recommended fix?",
                        key=f"fix_{col}",
                        value=config.get('missing_value_strategies', {}).get(col) is not None
                    )
                with c2:
                    if fix:
                        opts = ['median', 'mean', 'drop'] if df[col].dtype in ['float64', 'int64'] else ['mode', 'drop']
                        default_idx = opts.index(suggested_method) if suggested_method in opts else 0
                        strategy = st.selectbox(
                            "Imputation method",
                            opts,
                            index=default_idx,
                            key=f"strat_{col}",
                            help=f"AI suggests: {suggested_method.upper()}"
                        )
                        missing_strategies[col] = strategy
    else:
        render_alert("No missing values detected", "success")
    
    config['missing_value_strategies'] = missing_strategies
    
    # ===== OUTLIERS =====
    render_section_header("Outliers")
    
    if issues.get('outliers'):
        # Show recommendation for first outlier column as example
        first_col = list(issues['outliers'].keys())[0]
        first_info = issues['outliers'][first_col]
        recommendation, reasoning, suggested_method = get_outlier_recommendation(
            first_col, first_info['count'], first_info['percentage'], len(df)
        )
        
        st.info(f"{recommendation}\n\n**Analysis:** {reasoning}", icon="🤖")
        
        for col, info in issues['outliers'].items():
            _render_issue_card(col, "outliers", info['count'], info['percentage'], "warning")
        
        outlier_cols = st.multiselect(
            "Select columns to handle",
            list(issues['outliers'].keys()),
            default=config.get('outlier_columns', []),
            label_visibility="collapsed"
        )
        
        if outlier_cols:
            default_strat = 0 if suggested_method == 'clip' else 1
            config['outlier_strategy'] = st.radio(
                "Outlier handling strategy",
                ['clip', 'remove'],
                index=default_strat,
                horizontal=True,
                help=f"AI suggests: {suggested_method.upper()} | Clip: cap at bounds, Remove: delete rows"
            )
            config['outlier_columns'] = outlier_cols
    else:
        render_alert("No significant outliers detected", "success")
    
    # ===== CATEGORICAL ENCODING =====
    render_section_header("Categorical Encoding")
    
    cat_cols = [c for c in df.select_dtypes(include=['object']).columns if c != target_col]
    encoding_strategies = {}
    
    if cat_cols:
        for col in cat_cols:
            unique = df[col].nunique()
            
            # Get AI recommendation
            recommendation, reasoning, suggested_method = get_encoding_recommendation(
                col, unique, len(df)
            )
            
            with st.expander(f"💡 {col} — {unique} unique values"):
                st.info(f"{recommendation}\n\n**Why?** {reasoning}", icon="🤖")
                
                default_idx = 0 if suggested_method == 'onehot' else 1
                encoding = st.selectbox(
                    "Encoding method",
                    ['onehot', 'ordinal'],
                    key=f"enc_{col}",
                    index=default_idx,
                    help=f"AI suggests: {suggested_method.upper()}"
                )
                encoding_strategies[col] = encoding
    else:
        render_alert("No categorical columns to encode", "success")
    
    config['encoding_strategies'] = encoding_strategies
    
    # ===== FEATURE SCALING =====
    render_section_header("Feature Scaling")
    
    scaling = st.radio(
        "Select scaling method",
        ['None', 'standard', 'minmax'],
        horizontal=True,
        format_func=lambda x: {'None': 'No Scaling', 'standard': 'StandardScaler', 'minmax': 'MinMaxScaler'}[x]
    )
    config['scaling_strategy'] = scaling if scaling != 'None' else None
    
    # ===== TRAIN-TEST SPLIT =====
    render_section_header("Train-Test Split")
    
    test_size_int = st.slider(
        "Test set size",
        min_value=10,
        max_value=40,
        value=int(config.get('test_size', 0.2) * 100),
        step=5,
        format="%d%%",
        help="Percentage of data to reserve for testing"
    )
    config['test_size'] = test_size_int / 100.0
    config['target_col'] = target_col
    
    st.session_state.preprocess_config = config
    
    # ===== APPROVAL GATE =====
    st.markdown("<br>", unsafe_allow_html=True)
    
    approved = render_approval_gate()
    
    # ===== ACTION BUTTONS =====
    st.markdown("<br>", unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns([1, 1, 1])
    
    with c1:
        if st.button("Reset", width='stretch'):
            st.session_state.df_clean = None
            st.session_state.preprocessing_log = []
            st.session_state.preprocess_config = {}
            st.rerun()
    
    with c3:
        if st.button("Apply", width='stretch', type="primary", disabled=not approved):
            with st.status("Applying preprocessing...", expanded=True) as status:
                try:
                    df_proc = df.copy()
                    log = []
                    
                    st.write("Handling outliers...")
                    if config.get('outlier_columns') and config.get('outlier_strategy'):
                        df_proc, ol_log = handle_outliers(
                            df_proc,
                            config['outlier_columns'],
                            config['outlier_strategy']
                        )
                        log.extend(ol_log)
                    
                    st.write("Applying transformations...")
                    df_proc, p_log = apply_preprocessing(df_proc, config)
                    log.extend(p_log)
                    log.append(f"Train/Test split: {int((1-test_size)*100)}% / {int(test_size*100)}%")
                    
                    st.session_state.df_clean = df_proc
                    st.session_state.preprocessing_log = log
                    
                    status.update(label="Preprocessing complete", state="complete")
                    
                    st.toast("Preprocessing applied successfully", icon="✅")
                    
                    # Navigate to training
                    st.session_state.current_page = "Training"
                    st.rerun()
                    
                except Exception as e:
                    status.update(label="Error", state="error")
                    render_alert(f"Error: {str(e)}", "error")
    
    # ===== PREVIEW (if already processed) =====
    if st.session_state.df_clean is not None:
        render_section_header("Preprocessed Data Preview")
        
        cols = st.columns(3)
        with cols[0]:
            render_metric_card(str(len(df)), "Original Rows")
        with cols[1]:
            render_metric_card(str(len(st.session_state.df_clean)), "Processed Rows")
        with cols[2]:
            render_metric_card(str(len(st.session_state.df_clean.columns)), "Features")
        
        st.dataframe(st.session_state.df_clean.head(10), width='stretch')
