"""
recommendations.py - Smart Recommendation System

Provides intelligent suggestions for data quality fixes based on detected issues.
"""

from typing import Dict, Any, List, Tuple
import pandas as pd


def get_missing_value_recommendation(
    col_name: str,
    dtype: str,
    missing_pct: float,
    unique_ratio: float = 0.0
) -> Tuple[str, str, str]:
    """
    Generate smart recommendation for handling missing values.
    
    Args:
        col_name: Column name
        dtype: Data type of column
        missing_pct: Percentage of missing values
        unique_ratio: Ratio of unique values to total rows
    
    Returns:
        Tuple of (recommendation, reasoning, suggested_method)
    """
    if missing_pct > 50:
        return (
            f"🔴 **Critical**: {missing_pct:.1f}% missing values detected",
            "More than 50% of data is missing. Imputation may introduce significant bias. Consider dropping this column unless it's critical for your analysis.",
            "drop"
        )
    
    if dtype in ['float64', 'int64']:
        if unique_ratio > 0.8:  # High cardinality suggests continuous variable
            return (
                f"💡 **Recommended**: Use **median** imputation for {col_name}",
                "This appears to be a continuous numeric variable. Median is robust to outliers and preserves the central tendency of your data better than mean for skewed distributions.",
                "median"
            )
        else:
            return (
                f"💡 **Recommended**: Use **mean** imputation for {col_name}",
                "This appears to be a discrete numeric variable with repeated values. Mean imputation will preserve the overall distribution while being computationally efficient.",
                "mean"
            )
    else:  # Categorical
        return (
            f"💡 **Recommended**: Use **mode** (most frequent) imputation for {col_name}",
            "For categorical data, using the most frequent value (mode) maintains the distribution and is the statistically sound approach that won't introduce new categories.",
            "mode"
        )


def get_outlier_recommendation(
    col_name: str,
    outlier_count: int,
    outlier_pct: float,
    total_rows: int
) -> Tuple[str, str, str]:
    """
    Generate smart recommendation for handling outliers.
    
    Args:
        col_name: Column name
        outlier_count: Number of outlier data points
        outlier_pct: Percentage of outliers
        total_rows: Total number of rows
    
    Returns:
        Tuple of (recommendation, reasoning, suggested_method)
    """
    if outlier_pct > 20:
        return (
            f"⚠️ **Warning**: {outlier_pct:.1f}% outliers - may be legitimate data variation",
            f"With {outlier_count:,} outliers ({outlier_pct:.1f}% of data), this might represent natural variation rather than errors. Consider using **clip** to cap values at reasonable bounds instead of removing data.",
            "clip"
        )
    elif outlier_pct > 5:
        return (
            f"💡 **Recommended**: **Clip** outliers to bounds for {col_name}",
            f"{outlier_count:,} outliers detected. Clipping (capping values at IQR bounds) preserves data size while reducing extreme value impact on models.",
            "clip"
        )
    else:
        return (
            f"💡 **Recommended**: **Remove** outliers for {col_name}",
            f"Only {outlier_count:,} outliers ({outlier_pct:.1f}%) detected. Safe to remove these extreme values to improve model performance, minimal data loss.",
            "remove"
        )


def get_encoding_recommendation(
    col_name: str,
    unique_count: int,
    total_rows: int
) -> Tuple[str, str, str]:
    """
    Generate smart recommendation for categorical encoding.
    
    Args:
        col_name: Column name
        unique_count: Number of unique categories
        total_rows: Total number of rows
    
    Returns:
        Tuple of (recommendation, reasoning, suggested_method)
    """
    cardinality_ratio = unique_count / total_rows
    
    if unique_count > 20:
        if cardinality_ratio > 0.5:
            return (
                f"⚠️ **High Cardinality**: {unique_count} unique values - consider feature engineering",
                f"This column has very high cardinality ({unique_count} categories). Consider using **ordinal** encoding or creating meaningful groups instead of one-hot encoding which would create {unique_count} new columns.",
                "ordinal"
            )
        else:
            return (
                f"💡 **Recommended**: Use **ordinal** encoding for {col_name}",
                f"With {unique_count} categories, one-hot encoding would create too many columns. Ordinal encoding is more efficient and works well with tree-based models.",
                "ordinal"
            )
    elif unique_count <= 10:
        return (
            f"💡 **Recommended**: Use **one-hot** encoding for {col_name}",
            f"With only {unique_count} categories, one-hot encoding will create interpretable features without dimensional explosion. Best for capturing category relationships.",
            "onehot"
        )
    else:
        return (
            f"🤔 **Choice Needed**: {unique_count} categories detected",
            f"This is a moderate cardinality case. **One-hot** (interpretable but {unique_count} new columns) vs **ordinal** (compact but assumes order). Consider your model type and interpretability needs.",
            "onehot"  # Default to onehot for moderate cases
        )


def get_class_imbalance_recommendation(
    imbalance_ratio: float,
    minority_class: str,
    minority_count: int,
    majority_count: int
) -> Tuple[str, str]:
    """
    Generate recommendation for class imbalance.
    
    Args:
        imbalance_ratio: Ratio of majority to minority class
        minority_class: Label of minority class
        minority_count: Count of minority class samples
        majority_count: Count of majority class samples
    
    Returns:
        Tuple of (recommendation, reasoning)
    """
    if imbalance_ratio > 10:
        return (
            f"🔴 **Severe Imbalance**: {imbalance_ratio:.1f}:1 ratio detected",
            f"Class '{minority_class}' has only {minority_count:,} samples vs {majority_count:,} in majority class. Consider: (1) SMOTE/oversampling, (2) class weights in model, (3) collecting more minority data."
        )
    elif imbalance_ratio > 3:
        return (
            f"⚠️ **Moderate Imbalance**: {imbalance_ratio:.1f}:1 ratio",
            f"Imbalance detected but not severe. Use **class weights** in your models (e.g., class_weight='balanced' in sklearn) to handle this automatically."
        )
    else:
        return (
            f"✅ **Balanced**: Classes are reasonably balanced ({imbalance_ratio:.1f}:1)",
            "Your classes are well-balanced. No special handling needed, proceed with standard training."
        )


def get_scaling_recommendation(
    has_tree_models: bool,
    has_linear_models: bool,
    feature_ranges_vary: bool = True
) -> Tuple[str, str, str]:
    """
    Generate recommendation for feature scaling.
    
    Args:
        has_tree_models: Whether tree-based models are selected
        has_linear_models: Whether linear models are selected
        feature_ranges_vary: Whether features have different scales
    
    Returns:
        Tuple of (recommendation, reasoning, suggested_method)
    """
    if not feature_ranges_vary:
        return (
            "✅ **No Scaling Needed**: Features already on similar scales",
            "Your features have similar ranges. Scaling is optional and won't significantly impact performance.",
            "None"
        )
    
    if has_tree_models and not has_linear_models:
        return (
            "💡 **Optional**: Tree models don't require scaling",
            "Decision trees and Random Forests are scale-invariant. However, if you plan to compare feature importance or add linear models later, consider **StandardScaler**.",
            "None"
        )
    elif has_linear_models or not has_tree_models:
        return (
            "💡 **Recommended**: Use **StandardScaler** (z-score normalization)",
            "Linear models, SVM, and neural networks are sensitive to feature scales. StandardScaler (mean=0, std=1) is the most common choice and works well for normally distributed features.",
            "standard"
        )
    else:
        return (
            "💡 **Recommended**: Use **StandardScaler** for mixed models",
            "You're using both tree and linear models. StandardScaler won't hurt tree models and will help linear models perform better.",
            "standard"
        )
