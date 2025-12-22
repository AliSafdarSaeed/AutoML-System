"""
visualizations.py - Data Visualization Functions
Plotly-based visualizations for exploratory data analysis
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


# Constants for visualization limits
MAX_ROWS_FOR_VIZ = 10000
MAX_COLS_FOR_CORR = 30


def plot_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """
    Create an interactive correlation heatmap using Plotly.
    Uses sampling for large datasets.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Plotly Figure object
    """
    # Sample for large datasets
    if len(df) > MAX_ROWS_FOR_VIZ:
        df = df.sample(n=MAX_ROWS_FOR_VIZ, random_state=42)
    
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Limit columns for correlation
    if len(numeric_df.columns) > MAX_COLS_FOR_CORR:
        numeric_df = numeric_df.iloc[:, :MAX_COLS_FOR_CORR]
    
    if numeric_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No numeric columns for correlation", 
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    corr_matrix = numeric_df.corr()
    
    fig = px.imshow(
        corr_matrix,
        text_auto='.2f',
        aspect='auto',
        color_continuous_scale='RdBu_r',
        zmin=-1, zmax=1,
        title='Correlation Matrix' + (f' (sampled {MAX_ROWS_FOR_VIZ:,} rows)' if len(df) == MAX_ROWS_FOR_VIZ else '')
    )
    
    fig.update_layout(
        width=700,
        height=600,
        title_x=0.5
    )
    
    return fig


def plot_distributions(df: pd.DataFrame, max_cols: int = 6) -> list:
    """
    Create distribution plots for numeric columns.
    Uses sampling for large datasets.
    
    Args:
        df: Input DataFrame
        max_cols: Maximum number of columns to plot
        
    Returns:
        List of Plotly Figure objects
    """
    # Sample for large datasets
    if len(df) > MAX_ROWS_FOR_VIZ:
        df = df.sample(n=MAX_ROWS_FOR_VIZ, random_state=42)
    
    figures = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns[:max_cols]
    
    for col in numeric_cols:
        fig = px.histogram(
            df, x=col,
            title=f'Distribution of {col}',
            nbins=30,
            color_discrete_sequence=['#3498db']
        )
        fig.update_layout(
            showlegend=False,
            title_x=0.5,
            height=350
        )
        figures.append((col, fig))
    
    return figures


def plot_categorical_distributions(df: pd.DataFrame, max_cols: int = 8) -> list:
    """
    Create bar plots for categorical columns.
    
    Args:
        df: Input DataFrame
        max_cols: Maximum number of columns to plot
        
    Returns:
        List of Plotly Figure objects
    """
    figures = []
    cat_cols = df.select_dtypes(include=['object', 'category']).columns[:max_cols]
    
    for col in cat_cols:
        value_counts = df[col].value_counts().head(15)  # Top 15 categories
        fig = px.bar(
            x=value_counts.index.astype(str),
            y=value_counts.values,
            title=f'Distribution of {col}',
            color_discrete_sequence=['#2ecc71']
        )
        fig.update_layout(
            showlegend=False,
            title_x=0.5,
            height=350,
            xaxis_title=col,
            yaxis_title='Count'
        )
        figures.append((col, fig))
    
    return figures


def plot_target_distribution(df: pd.DataFrame, target_col: str) -> go.Figure:
    """
    Create a pie chart for target class distribution.
    Optimized to only compute value counts (no sampling needed).
    
    Args:
        df: Input DataFrame
        target_col: Name of target column
        
    Returns:
        Plotly Figure object
    """
    # Value counts is efficient even for large datasets
    value_counts = df[target_col].value_counts()
    
    # Limit to top 20 classes for very high cardinality
    if len(value_counts) > 20:
        value_counts = value_counts.head(20)
    
    fig = px.pie(
        names=value_counts.index.astype(str),
        values=value_counts.values,
        title=f'Target Distribution: {target_col}',
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    
    fig.update_layout(
        title_x=0.5,
        height=400
    )
    
    return fig
