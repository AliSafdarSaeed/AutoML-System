"""
preprocessing.py - Data Preprocessing Functions
Functions for cleaning, encoding, scaling, and handling outliers
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OrdinalEncoder


def apply_preprocessing(df: pd.DataFrame, config: dict) -> tuple:
    """
    Apply preprocessing based on user configuration.
    
    Args:
        df: Input DataFrame
        config: Dictionary containing preprocessing options
            - missing_value_strategies: {col: 'mean'|'median'|'mode'|'drop'}
            - encoding_strategies: {col: 'onehot'|'ordinal'}
            - scaling_strategy: 'standard'|'minmax'|None
            - target_col: Name of target column
            
    Returns:
        Tuple of (processed_df, preprocessing_log)
    """
    df_processed = df.copy()
    log = []
    
    # Handle missing values
    missing_strategies = config.get('missing_value_strategies', {})
    for col, strategy in missing_strategies.items():
        if col not in df_processed.columns:
            continue
            
        if strategy == 'mean':
            fill_value = df_processed[col].mean()
            df_processed[col].fillna(fill_value, inplace=True)
            log.append(f"Filled missing values in '{col}' with mean ({fill_value:.4f})")
        elif strategy == 'median':
            fill_value = df_processed[col].median()
            df_processed[col].fillna(fill_value, inplace=True)
            log.append(f"Filled missing values in '{col}' with median ({fill_value:.4f})")
        elif strategy == 'mode':
            fill_value = df_processed[col].mode()[0]
            df_processed[col].fillna(fill_value, inplace=True)
            log.append(f"Filled missing values in '{col}' with mode ({fill_value})")
        elif strategy == 'drop':
            initial_rows = len(df_processed)
            df_processed.dropna(subset=[col], inplace=True)
            dropped = initial_rows - len(df_processed)
            log.append(f"Dropped {dropped} rows with missing values in '{col}'")
    
    # Handle encoding
    target_col = config.get('target_col')
    encoding_strategies = config.get('encoding_strategies', {})
    
    for col, strategy in encoding_strategies.items():
        if col not in df_processed.columns or col == target_col:
            continue
            
        if strategy == 'onehot':
            dummies = pd.get_dummies(df_processed[col], prefix=col, drop_first=True)
            df_processed = pd.concat([df_processed.drop(col, axis=1), dummies], axis=1)
            log.append(f"Applied one-hot encoding to '{col}'")
        elif strategy == 'ordinal':
            encoder = OrdinalEncoder()
            df_processed[col] = encoder.fit_transform(df_processed[[col]])
            log.append(f"Applied ordinal encoding to '{col}'")
    
    # Encode target column if categorical
    if target_col and target_col in df_processed.columns:
        if df_processed[target_col].dtype == 'object':
            le = LabelEncoder()
            df_processed[target_col] = le.fit_transform(df_processed[target_col])
            log.append(f"Encoded target column '{target_col}' (classes: {list(le.classes_)})")
    
    # Handle scaling (applied to numeric features only, excluding target)
    scaling_strategy = config.get('scaling_strategy')
    if scaling_strategy:
        feature_cols = [c for c in df_processed.select_dtypes(include=[np.number]).columns 
                       if c != target_col]
        
        if feature_cols:
            if scaling_strategy == 'standard':
                scaler = StandardScaler()
                df_processed[feature_cols] = scaler.fit_transform(df_processed[feature_cols])
                log.append(f"Applied StandardScaler to {len(feature_cols)} numeric features")
            elif scaling_strategy == 'minmax':
                scaler = MinMaxScaler()
                df_processed[feature_cols] = scaler.fit_transform(df_processed[feature_cols])
                log.append(f"Applied MinMaxScaler to {len(feature_cols)} numeric features")
            elif scaling_strategy == 'robust':
                scaler = RobustScaler()
                df_processed[feature_cols] = scaler.fit_transform(df_processed[feature_cols])
                log.append(f"Applied RobustScaler to {len(feature_cols)} numeric features")
    
    return df_processed, log


def handle_outliers(df: pd.DataFrame, columns: list, strategy: str = 'clip') -> tuple:
    """
    Handle outliers in specified columns.
    
    Args:
        df: Input DataFrame
        columns: List of columns to process
        strategy: 'clip' (cap at bounds) or 'remove' (remove rows)
        
    Returns:
        Tuple of (processed_df, log_messages)
    """
    df_processed = df.copy()
    log = []
    
    for col in columns:
        if col not in df_processed.columns:
            continue
            
        Q1 = df_processed[col].quantile(0.25)
        Q3 = df_processed[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        
        if strategy == 'clip':
            df_processed[col] = df_processed[col].clip(lower=lower, upper=upper)
            log.append(f"Clipped outliers in '{col}' to [{lower:.2f}, {upper:.2f}]")
        elif strategy == 'remove':
            initial = len(df_processed)
            mask = (df_processed[col] >= lower) & (df_processed[col] <= upper)
            df_processed = df_processed[mask]
            removed = initial - len(df_processed)
            log.append(f"Removed {removed} outlier rows from '{col}'")
    
    return df_processed, log
