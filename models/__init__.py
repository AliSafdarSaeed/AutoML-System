"""
models package - Machine Learning Models Module
Contains model configurations, trainer class, and visualizations
"""

from .model_configs import MODEL_CONFIGS, get_available_models, get_model_config
from .trainer import ModelTrainer
from .visualizations import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_model_comparison,
    plot_training_times
)

__all__ = [
    'ModelTrainer',
    'MODEL_CONFIGS',
    'get_available_models',
    'get_model_config',
    'plot_confusion_matrix',
    'plot_roc_curve',
    'plot_model_comparison',
    'plot_training_times'
]
