"""
model_recommendations.py - Smart Model Selection Assistant

Recommends best ML models based on dataset characteristics.
"""

from typing import List, Tuple, Dict, Any
import pandas as pd
import numpy as np


def analyze_dataset_characteristics(df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
    """
    Analyze dataset characteristics for model recommendation.
    
    Returns:
        Dictionary with dataset analysis results
    """
    # Basic stats
    num_samples = len(df)
    num_features = len(df.columns) - 1  # Exclude target
    
    # Target analysis
    target_unique = df[target_col].nunique()
    is_binary = target_unique == 2
    is_multiclass = target_unique > 2
    
    # Class balance
    value_counts = df[target_col].value_counts()
    max_count = value_counts.max()
    min_count = value_counts.min()
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    
    # Feature types
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if target_col in cat_cols:
        cat_cols.remove(target_col)
    
    # Feature characteristics
    high_cardinality_cats = sum(1 for c in cat_cols if df[c].nunique() > 10)
    
    # Check for missing values
    missing_ratio = df.isnull().sum().sum() / (num_samples * (num_features + 1))
    
    # Feature variance (for numeric columns)
    if numeric_cols:
        feature_variance = df[numeric_cols].var().mean()
        feature_correlation = df[numeric_cols].corr().abs().mean().mean() if len(numeric_cols) > 1 else 0
    else:
        feature_variance = 0
        feature_correlation = 0
    
    return {
        'num_samples': num_samples,
        'num_features': num_features,
        'target_unique': target_unique,
        'is_binary': is_binary,
        'is_multiclass': is_multiclass,
        'imbalance_ratio': imbalance_ratio,
        'num_numeric': len(numeric_cols),
        'num_categorical': len(cat_cols),
        'high_cardinality_cats': high_cardinality_cats,
        'missing_ratio': missing_ratio,
        'feature_variance': feature_variance,
        'feature_correlation': feature_correlation,
        'feature_ratio': num_features / num_samples if num_samples > 0 else 0
    }


def get_model_recommendations(
    df: pd.DataFrame,
    target_col: str,
    num_features: int,
    num_samples: int
) -> Tuple[List[str], str]:
    """
    Recommend best models based on dataset characteristics.
    
    Args:
        df: DataFrame with features and target
        target_col: Name of target column
        num_features: Number of features
        num_samples: Number of samples
    
    Returns:
        Tuple of (recommended_models, reasoning)
    """
    # Get comprehensive analysis
    analysis = analyze_dataset_characteristics(df, target_col)
    
    recommendations = []
    reasoning_points = []
    
    # ===== DATASET SIZE ANALYSIS =====
    if analysis['num_samples'] < 500:
        reasoning_points.append(
            f"📊 **Very Small Dataset** ({analysis['num_samples']:,} samples)\n"
            f"   - Prefer simpler models to avoid overfitting\n"
            f"   - Cross-validation is crucial for reliable estimates"
        )
        recommendations.extend(['Logistic Regression', 'Decision Tree'])
        
    elif analysis['num_samples'] < 2000:
        reasoning_points.append(
            f"📊 **Small-Medium Dataset** ({analysis['num_samples']:,} samples)\n"
            f"   - Ensemble methods should work well\n"
            f"   - Regularization recommended for linear models"
        )
        recommendations.extend(['Random Forest', 'Logistic Regression'])
        
    elif analysis['num_samples'] < 10000:
        reasoning_points.append(
            f"📊 **Medium Dataset** ({analysis['num_samples']:,} samples)\n"
            f"   - Good size for most algorithms\n"
            f"   - Gradient Boosting should perform well"
        )
        recommendations.extend(['Random Forest', 'Gradient Boosting'])
        
    else:
        reasoning_points.append(
            f"📊 **Large Dataset** ({analysis['num_samples']:,} samples)\n"
            f"   - Complex models can be utilized\n"
            f"   - Consider training time vs. accuracy trade-off"
        )
        recommendations.extend(['Gradient Boosting', 'Random Forest'])
    
    # ===== CLASS IMBALANCE ANALYSIS =====
    if analysis['imbalance_ratio'] > 5:
        reasoning_points.append(
            f"⚖️ **Significant Class Imbalance** ({analysis['imbalance_ratio']:.1f}:1 ratio)\n"
            f"   - Tree-based models handle imbalance better\n"
            f"   - Consider using class_weight='balanced'"
        )
        if 'Random Forest' not in recommendations:
            recommendations.insert(0, 'Random Forest')
            
    elif analysis['imbalance_ratio'] > 2:
        reasoning_points.append(
            f"⚖️ **Moderate Class Imbalance** ({analysis['imbalance_ratio']:.1f}:1 ratio)\n"
            f"   - Most algorithms will handle this\n"
            f"   - Class weights recommended for best results"
        )
    
    # ===== FEATURE ANALYSIS =====
    if analysis['feature_ratio'] > 0.3:
        reasoning_points.append(
            f"🔢 **High Feature-to-Sample Ratio** ({analysis['num_features']}/{analysis['num_samples']})\n"
            f"   - Risk of overfitting is higher\n"
            f"   - Regularized models strongly recommended"
        )
        if 'Logistic Regression' not in recommendations:
            recommendations.insert(0, 'Logistic Regression')
            
    if analysis['num_categorical'] > analysis['num_numeric']:
        reasoning_points.append(
            f"📝 **Mostly Categorical Features** ({analysis['num_categorical']} cat vs {analysis['num_numeric']} num)\n"
            f"   - Tree-based models excel with categorical data\n"
            f"   - Encoding quality is crucial"
        )
        if 'Random Forest' not in recommendations[:2]:
            recommendations.insert(0, 'Random Forest')
    
    if analysis['feature_correlation'] > 0.5 and analysis['num_numeric'] > 3:
        reasoning_points.append(
            f"🔗 **Highly Correlated Features** (avg correlation: {analysis['feature_correlation']:.2f})\n"
            f"   - Feature selection may improve performance\n"
            f"   - Tree models handle multicollinearity well"
        )
    
    # ===== PROBLEM TYPE ANALYSIS =====
    if analysis['is_binary']:
        reasoning_points.append(
            f"🎯 **Binary Classification**\n"
            f"   - Logistic Regression for interpretability\n"
            f"   - Ensemble methods for best accuracy"
        )
    else:
        reasoning_points.append(
            f"🎯 **Multi-class Classification** ({analysis['target_unique']} classes)\n"
            f"   - Tree-based models handle multi-class natively\n"
            f"   - SVM with proper kernel can work well"
        )
        if analysis['target_unique'] > 5 and 'SVM' in recommendations:
            recommendations.remove('SVM')  # SVM struggles with many classes
    
    # Deduplicate and limit recommendations
    seen = set()
    unique_recommendations = []
    for model in recommendations:
        if model not in seen:
            seen.add(model)
            unique_recommendations.append(model)
    
    # Ensure we have at least 2 recommendations
    default_models = ['Random Forest', 'Logistic Regression', 'Gradient Boosting']
    for model in default_models:
        if len(unique_recommendations) < 2 and model not in unique_recommendations:
            unique_recommendations.append(model)
    
    # Format reasoning
    reasoning = "\n\n".join(reasoning_points)
    
    return unique_recommendations[:3], reasoning  # Max 3 recommendations


def get_model_explanation(model_name: str) -> str:
    """
    Get a brief explanation of why a model might be good for the dataset.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Brief explanation string
    """
    explanations = {
        'Logistic Regression': "Simple, interpretable, fast. Good baseline for binary classification. Works well with linear relationships.",
        'Random Forest': "Robust ensemble of decision trees. Handles non-linear relationships, categorical features, and missing values well.",
        'Gradient Boosting': "Powerful ensemble that builds trees sequentially. Often achieves best accuracy but can overfit on small data.",
        'Decision Tree': "Simple, interpretable, visualizable. Good for understanding feature importance but prone to overfitting.",
        'SVM': "Effective in high-dimensional spaces. Works well with clear margin of separation between classes.",
        'K-Nearest Neighbors': "Simple instance-based learning. Good for smaller datasets with well-defined clusters."
    }
    return explanations.get(model_name, "A machine learning model suitable for classification tasks.")
