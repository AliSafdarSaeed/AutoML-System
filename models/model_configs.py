"""
model_configs.py - Model Configurations for AutoML System
Contains all model definitions, parameters, and default configurations
"""

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier


# Available models with their default configurations
MODEL_CONFIGS = {
    'Logistic Regression': {
        'model': LogisticRegression,
        'params': {'C': [0.1, 1, 10], 'max_iter': [1000]},
        'default_params': {'max_iter': 1000, 'random_state': 42}
    },
    'K-Nearest Neighbors': {
        'model': KNeighborsClassifier,
        'params': {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']},
        'default_params': {}
    },
    'Decision Tree': {
        'model': DecisionTreeClassifier,
        'params': {'max_depth': [3, 5, 10, None], 'min_samples_split': [2, 5]},
        'default_params': {'random_state': 42}
    },
    'Naive Bayes': {
        'model': GaussianNB,
        'params': {'var_smoothing': [1e-9, 1e-8, 1e-7]},
        'default_params': {}
    },
    'Random Forest': {
        'model': RandomForestClassifier,
        'params': {'n_estimators': [50, 100], 'max_depth': [5, 10, None]},
        'default_params': {'random_state': 42}
    },
    'Support Vector Machine': {
        'model': SVC,
        'params': {'C': [0.1, 1], 'kernel': ['rbf', 'linear']},
        'default_params': {'probability': True, 'random_state': 42}
    },
    'Rule-Based (Baseline)': {
        'model': DummyClassifier,
        'params': {'strategy': ['most_frequent', 'stratified']},
        'default_params': {'random_state': 42}
    }
}


def get_available_models():
    """Return list of available model names."""
    return list(MODEL_CONFIGS.keys())


def get_model_config(model_name: str):
    """Get configuration for a specific model."""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Model '{model_name}' not found. Available: {get_available_models()}")
    return MODEL_CONFIGS[model_name]
