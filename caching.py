"""
caching.py - Caching functions for performance optimization
Handles caching of expensive operations for large datasets
"""

import streamlit as st
import pandas as pd
from data_utils import analyze_data, detect_issues


@st.cache_data(show_spinner=False)
def load_csv(uploaded_file):
    """Load CSV with caching."""
    return pd.read_csv(uploaded_file)


@st.cache_data(show_spinner=False)
def cached_analyze_data(df_hash, df):
    """Cached data analysis."""
    return analyze_data(df)


@st.cache_data(show_spinner=False)
def cached_detect_issues(df_hash, df, target_col):
    """Cached issue detection."""
    return detect_issues(df, target_col)


def get_df_hash(df):
    """Create a hash for caching based on shape and sample."""
    return f"{df.shape}_{df.columns.tolist()}_{len(df)}"
