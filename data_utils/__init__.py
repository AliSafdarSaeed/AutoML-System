"""
data_utils package - Data Utility Functions
Contains analysis, visualization, preprocessing, and reporting modules
"""

from .analysis import (
    analyze_data,
    detect_missing_values,
    detect_outliers,
    detect_class_imbalance,
    detect_issues
)

from .visualizations import (
    plot_correlation_heatmap,
    plot_distributions,
    plot_categorical_distributions,
    plot_target_distribution,
    MAX_ROWS_FOR_VIZ,
    MAX_COLS_FOR_CORR
)

from .preprocessing import (
    apply_preprocessing,
    handle_outliers
)

from .reporting import (
    PDFReport,
    generate_pdf_report
)

__all__ = [
    'analyze_data',
    'detect_missing_values',
    'detect_outliers',
    'detect_class_imbalance',
    'detect_issues',
    'plot_correlation_heatmap',
    'plot_distributions',
    'plot_categorical_distributions',
    'plot_target_distribution',
    'MAX_ROWS_FOR_VIZ',
    'MAX_COLS_FOR_CORR',
    'apply_preprocessing',
    'handle_outliers',
    'PDFReport',
    'generate_pdf_report'
]
