"""
model_recommendations.py - Smart Model Selection Assistant

Recommends best ML models based on dataset characteristics.
"""

from typing import List, Tuple, Dict, Any
import pandas as pd
import numpy as np


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
    # Analyze target distribution
    target_unique = df[target_col].nunique()
    is_binary = target_unique == 2
    is_multiclass = target_unique > 2
    
    # Analyze class balance
    value_counts = df[target_col].value_counts()
    max_count = value_counts.max()
    min_count = value_counts.min()
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    
    # Feature to sample ratio
    feature_ratio = num_features / num_samples
    
    recommendations = []
    reasoning_points = []
    
    # Rule-based recommendations
    if is_binary:
        if num_samples < 1000:
            # Small dataset
            recommendations = ['Logistic Regression', 'Random Forest']
            reasoning_points.append(f"📊 **Small dataset** ({num_samples:,} samples): Logistic Regression is simple and less prone to overfitting. Random Forest provides good baseline performance.")
        elif imbalance_ratio > 3:
            # Imbalanced
            recommendations = ['Random Forest', 'Gradient Boosting']
            reasoning_points.append(f"⚖️ **Class imbalance detected** ({imbalance_ratio:.1f}:1 ratio): Tree-based models (Random Forest, Gradient Boosting) handle imbalance better with class weights.")
        else:
            # Standard binary
            recommendations = ['Logistic Regression', 'Random Forest']
            reasoning_points.append(f"✅ **Standard binary classification** with {num_samples:,} samples: Logistic Regression for interpretability, Random Forest for accuracy.")
    
    else:  # Multiclass
        if target_unique > 10:
            # Many classes
            recommendations = ['Random Forest', 'Gradient Boosting']
            reasoning_points.append(f"🎯 **Many classes** ({target_unique} classes): Tree-based models handle multi-class naturally without one-vs-rest strategies.")
        else:
            # Standard multiclass
            recommendations = ['Random Forest', 'SVM']
            reasoning_points.append(f"🎯 **Multiclass** ({target_unique} classes): Random Forest for robustness, SVM for complex decision boundaries.")
    
    # Additional considerations
    if feature_ratio > 0.5:
        reasoning_points.append(f"⚠️ **High feature-to-sample ratio** ({num_features}/{num_samples}): Simpler models preferred to avoid overfitting.")
        if 'Logistic Regression' not in recommendations:
            recommendations.insert(0, 'Logistic Regression')
    
    if num_samples > 10000:
        reasoning_points.append(f"💪 **Large dataset** ({num_samples:,} samples): Can support complex models like Gradient Boosting.")
        if 'Gradient Boosting' not in recommendations and len(recommendations) < 3:
            recommendations.append('Gradient Boosting')
    
    # Ensure we have at least 2 recommendations
    if len(recommendations) < 2:
        if 'Random Forest' not in recommendations:
            recommendations.append('Random Forest')
        if 'Logistic Regression' not in recommendations:
            recommendations.append('Logistic Regression')
    
    reasoning = "\n\n".join(reasoning_points)
    
    return recommendations[:3], reasoning  # Max 3 recommendations
