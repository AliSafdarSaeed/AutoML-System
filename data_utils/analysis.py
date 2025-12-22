"""
analysis.py - Data Analysis Functions
Functions for analyzing datasets, detecting issues, and generating metadata
"""

import pandas as pd
import numpy as np


# Constants
MAX_SAMPLE_SIZE_OUTLIERS = 50000
MAX_OUTLIER_COLUMNS = 20


def analyze_data(df: pd.DataFrame) -> dict:
    """
    Analyze dataset and return metadata.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary containing dataset metadata
    """
    return {
        'rows': len(df),
        'columns': len(df.columns),
        'column_names': df.columns.tolist(),
        'dtypes': df.dtypes.astype(str).to_dict(),
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object', 'category']).columns.tolist(),
        'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,  # MB
        'duplicates': df.duplicated().sum(),
        'stats': df.describe().to_dict()
    }


def detect_missing_values(df: pd.DataFrame) -> dict:
    """
    Detect missing values in each column.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with column names as keys and missing info as values
    """
    missing = {}
    for col in df.columns:
        count = df[col].isnull().sum()
        if count > 0:
            missing[col] = {
                'count': int(count),
                'percentage': round((count / len(df)) * 100, 2)
            }
    return missing


def detect_outliers(df: pd.DataFrame, sample_size: int = MAX_SAMPLE_SIZE_OUTLIERS) -> dict:
    """
    Detect outliers using IQR method for numeric columns.
    Uses sampling for large datasets.
    
    Args:
        df: Input DataFrame
        sample_size: Max rows to sample for outlier detection
        
    Returns:
        Dictionary with column names as keys and outlier info as values
    """
    # Sample for large datasets
    if len(df) > sample_size:
        df_sample = df.sample(n=sample_size, random_state=42)
    else:
        df_sample = df
    
    outliers = {}
    numeric_cols = df_sample.select_dtypes(include=[np.number]).columns[:MAX_OUTLIER_COLUMNS]
    
    for col in numeric_cols:
        Q1 = df_sample[col].quantile(0.25)
        Q3 = df_sample[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outlier_mask = (df_sample[col] < lower_bound) | (df_sample[col] > upper_bound)
        count = outlier_mask.sum()
        
        # Estimate for full dataset
        estimated_count = int(count * (len(df) / len(df_sample)))
        
        if count > 0:
            outliers[col] = {
                'count': estimated_count,
                'percentage': round((count / len(df_sample)) * 100, 2),
                'lower_bound': round(lower_bound, 4),
                'upper_bound': round(upper_bound, 4)
            }
    
    return outliers


def detect_class_imbalance(df: pd.DataFrame, target_col: str, threshold: float = 0.8) -> dict:
    """
    Detect class imbalance in target column.
    
    Args:
        df: Input DataFrame
        target_col: Name of target column
        threshold: Imbalance threshold (default 0.8 = 80:20 ratio)
        
    Returns:
        Dictionary with imbalance information
    """
    if target_col not in df.columns:
        return {'error': f'Column {target_col} not found'}
    
    value_counts = df[target_col].value_counts(normalize=True)
    max_ratio = value_counts.max()
    
    return {
        'is_imbalanced': max_ratio > threshold,
        'class_distribution': value_counts.to_dict(),
        'majority_class': value_counts.idxmax(),
        'majority_ratio': round(max_ratio * 100, 2),
        'minority_class': value_counts.idxmin(),
        'minority_ratio': round(value_counts.min() * 100, 2)
    }


def detect_issues(df: pd.DataFrame, target_col: str = None) -> dict:
    """
    Aggregate all issue detections into a single report.
    
    Args:
        df: Input DataFrame
        target_col: Optional target column for imbalance detection
        
    Returns:
        Dictionary containing all detected issues
    """
    issues = {
        'missing_values': detect_missing_values(df),
        'outliers': detect_outliers(df),
        'has_issues': False
    }
    
    if target_col:
        issues['class_imbalance'] = detect_class_imbalance(df, target_col)
        if issues['class_imbalance'].get('is_imbalanced', False):
            issues['has_issues'] = True
    
    if issues['missing_values']:
        issues['has_issues'] = True
    if issues['outliers']:
        issues['has_issues'] = True
    
    return issues
