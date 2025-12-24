"""page_preprocessing.py - Data Preprocessing Page"""
import streamlit as st
import pandas as pd
from ui_components import *
from data_utils import handle_outliers, apply_preprocessing, detect_issues

def page_preprocessing():
    render_header("🔧", "Data Preprocessing", "Clean and prepare your data for optimal model performance")
    
    if st.session_state.df is None:
        render_alert("Please upload a dataset first", "warning")
        return
    
    if st.session_state.target_col is None:
        render_alert("Please select a target column on the EDA page", "warning")
        return
    
    df = st.session_state.df.copy()
    target_col = st.session_state.target_col
    issues = st.session_state.issues or detect_issues(df, target_col)
    
    render_alert("Review the options below and customize your preprocessing pipeline. Changes are only applied when you click 'Apply'.", "info")
    
    config = st.session_state.preprocess_config.copy()
    
    # Missing Values
    render_section_header("🩹", "Handle Missing Values")
    
    missing_strategies = {}
    if issues['missing_values']:
        for col, info in issues['missing_values'].items():
            with st.expander(f"⚠️ {col} — {info['count']} missing ({info['percentage']}%)", expanded=True):
                c1, c2 = st.columns([1, 2])
                with c1:
                    fix = st.checkbox("Apply fix", key=f"fix_{col}", value=config.get('missing_value_strategies', {}).get(col) is not None)
                with c2:
                    if fix:
                        opts = ['median', 'mean', 'drop'] if df[col].dtype in ['float64', 'int64'] else ['mode', 'drop']
                        strategy = st.selectbox("Strategy", opts, key=f"strat_{col}")
                        missing_strategies[col] = strategy
    else:
        render_alert("No missing values detected!", "success")
    
    config['missing_value_strategies'] = missing_strategies
    
    # Outliers
    render_section_header("📉", "Handle Outliers")
    
    if issues['outliers']:
        outlier_cols = st.multiselect("Select columns to handle:", list(issues['outliers'].keys()), default=config.get('outlier_columns', []))
        if outlier_cols:
            config['outlier_strategy'] = st.radio("Strategy:", ['clip', 'remove'], horizontal=True, help="Clip: cap at bounds. Remove: delete rows.")
            config['outlier_columns'] = outlier_cols
    else:
        render_alert("No significant outliers detected!", "success")
    
    # Encoding
    render_section_header("🔤", "Encode Categorical Variables")
    
    cat_cols = [c for c in df.select_dtypes(include=['object']).columns if c != target_col]
    encoding_strategies = {}
    
    if cat_cols:
        for col in cat_cols:
            unique = df[col].nunique()
            with st.expander(f"🏷️ {col} — {unique} unique values"):
                encoding = st.selectbox("Encoding:", ['onehot', 'ordinal'], key=f"enc_{col}", index=0 if unique <= 10 else 1)
                encoding_strategies[col] = encoding
    else:
        render_alert("No categorical columns to encode!", "success")
    
    config['encoding_strategies'] = encoding_strategies
    
    # Scaling
    render_section_header("⚖️", "Feature Scaling")
    
    scaling = st.radio(
        "Select scaling method:",
        ['None', 'standard', 'minmax'],
        horizontal=True,
        format_func=lambda x: {'None': '❌ None', 'standard': '📊 StandardScaler', 'minmax': '📏 MinMaxScaler'}[x]
    )
    config['scaling_strategy'] = scaling if scaling != 'None' else None
    
    # Train-Test Split
    render_section_header("✂️", "Train-Test Split")
    
    test_size = st.slider("Test set size:", 0.1, 0.4, config.get('test_size', 0.2), 0.05, format="%.0f%%")
    config['test_size'] = test_size
    config['target_col'] = target_col
    
    st.session_state.preprocess_config = config
    
    # Apply Button
    st.markdown("<br>", unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        if st.button("🔄 Reset", width='stretch'):
            st.session_state.df_clean = None
            st.session_state.preprocessing_log = []
            st.session_state.preprocess_config = {}
            st.rerun()
    
    with c3:
        if st.button("✅ Apply Preprocessing", width='stretch', type="primary"):
            with st.spinner("Processing..."):
                try:
                    df_proc = df.copy()
                    log = []
                    
                    if config.get('outlier_columns') and config.get('outlier_strategy'):
                        df_proc, ol_log = handle_outliers(df_proc, config['outlier_columns'], config['outlier_strategy'])
                        log.extend(ol_log)
                    
                    df_proc, p_log = apply_preprocessing(df_proc, config)
                    log.extend(p_log)
                    log.append(f"Train/Test split: {int((1-test_size)*100)}% / {int(test_size*100)}%")
                    
                    st.session_state.df_clean = df_proc
                    st.session_state.preprocessing_log = log
                    
                    render_alert("Preprocessing applied successfully!", "success")
                    
                    with st.expander("📋 Preprocessing Log", expanded=True):
                        for step in log:
                            st.markdown(f"<div style='padding: 0.5rem; color: rgba(255,255,255,0.8);'>• {step}</div>", unsafe_allow_html=True)
                except Exception as e:
                    render_alert(f"Error: {str(e)}", "error")
    
    # Preview & Continue
    if st.session_state.df_clean is not None:
        render_section_header("✨", "Preprocessed Data")
        
        cols = st.columns(3)
        with cols[0]:
            render_metric_card(len(df), "Original Rows", "📊")
        with cols[1]:
            render_metric_card(len(st.session_state.df_clean), "Processed Rows", "✅")
        with cols[2]:
            render_metric_card(len(st.session_state.df_clean.columns), "Features", "🔢")
        
        st.dataframe(st.session_state.df_clean.head(10), width='stretch')
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🎯 Continue to Training", width='stretch', type="primary"):
                st.session_state.current_page = "Training"
                st.rerun()



