"""
visualizations.py - Model Visualization Functions
Plotly-based visualizations for model evaluation and comparison
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Any

from sklearn.metrics import confusion_matrix, roc_curve, auc


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: List = None) -> go.Figure:
    """
    Create an interactive confusion matrix using Plotly.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Optional list of class labels
        
    Returns:
        Plotly Figure object
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if labels is None:
        labels = [f'Class {i}' for i in range(len(cm))]
    else:
        labels = [str(l) for l in labels]
    
    # Create heatmap
    fig = px.imshow(
        cm,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=labels,
        y=labels,
        text_auto=True,
        color_continuous_scale='Blues',
        aspect='auto'
    )
    
    fig.update_layout(
        title='Confusion Matrix',
        title_x=0.5,
        width=500,
        height=450
    )
    
    return fig


def plot_roc_curve(model: Any, X_test: np.ndarray, y_test: np.ndarray, model_name: str = 'Model') -> go.Figure:
    """
    Create ROC curve using Plotly.
    Handles both binary and multi-class classification.
    
    Args:
        model: Trained sklearn model with predict_proba method
        X_test: Test features
        y_test: Test labels
        model_name: Name of the model for the title
        
    Returns:
        Plotly Figure object
    """
    fig = go.Figure()
    
    # Check if model supports probability predictions
    if not hasattr(model, 'predict_proba'):
        fig.add_annotation(
            text=f"{model_name} does not support probability predictions for ROC curve",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    try:
        y_proba = model.predict_proba(X_test)
        classes = model.classes_
        n_classes = len(classes)
        
        if n_classes == 2:
            # Binary classification
            fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                name=f'ROC (AUC = {roc_auc:.4f})',
                mode='lines',
                line=dict(color='#3498db', width=2)
            ))
        else:
            # Multi-class: One-vs-Rest
            from sklearn.preprocessing import label_binarize
            y_test_bin = label_binarize(y_test, classes=classes)
            
            colors = px.colors.qualitative.Set1
            
            for i, cls in enumerate(classes):
                if y_test_bin.shape[1] > 1:
                    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
                else:
                    fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
                    
                roc_auc = auc(fpr, tpr)
                
                fig.add_trace(go.Scatter(
                    x=fpr, y=tpr,
                    name=f'Class {cls} (AUC = {roc_auc:.4f})',
                    mode='lines',
                    line=dict(color=colors[i % len(colors)], width=2)
                ))
        
        # Add diagonal reference line
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            name='Random Classifier',
            mode='lines',
            line=dict(color='gray', width=1, dash='dash')
        ))
        
        fig.update_layout(
            title=f'ROC Curve - {model_name}',
            title_x=0.5,
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            width=550,
            height=450,
            legend=dict(x=0.6, y=0.1)
        )
        
    except Exception as e:
        fig.add_annotation(
            text=f"Error generating ROC curve: {str(e)}",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
        )
    
    return fig


def plot_model_comparison(results: List[Dict]) -> go.Figure:
    """
    Create a bar chart comparing model performance metrics.
    
    Args:
        results: List of result dictionaries from ModelTrainer
        
    Returns:
        Plotly Figure object
    """
    if not results:
        fig = go.Figure()
        fig.add_annotation(text="No results to display", xref="paper", yref="paper", x=0.5, y=0.5)
        return fig
    
    # Sort by F1-Score
    sorted_results = sorted(results, key=lambda x: x.get('f1_score', 0), reverse=True)
    
    models = [r['model_name'] for r in sorted_results]
    accuracy = [r.get('accuracy', 0) for r in sorted_results]
    precision = [r.get('precision', 0) for r in sorted_results]
    recall = [r.get('recall', 0) for r in sorted_results]
    f1 = [r.get('f1_score', 0) for r in sorted_results]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(name='Accuracy', x=models, y=accuracy, marker_color='#3498db'))
    fig.add_trace(go.Bar(name='Precision', x=models, y=precision, marker_color='#2ecc71'))
    fig.add_trace(go.Bar(name='Recall', x=models, y=recall, marker_color='#e74c3c'))
    fig.add_trace(go.Bar(name='F1-Score', x=models, y=f1, marker_color='#9b59b6'))
    
    fig.update_layout(
        title='Model Performance Comparison',
        title_x=0.5,
        barmode='group',
        xaxis_title='Model',
        yaxis_title='Score',
        yaxis_range=[0, 1.05],
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        height=500
    )
    
    return fig


def plot_training_times(results: List[Dict]) -> go.Figure:
    """
    Create a bar chart showing training times for each model.
    
    Args:
        results: List of result dictionaries from ModelTrainer
        
    Returns:
        Plotly Figure object
    """
    if not results:
        fig = go.Figure()
        fig.add_annotation(text="No results to display", xref="paper", yref="paper", x=0.5, y=0.5)
        return fig
    
    # Sort by training time
    sorted_results = sorted(results, key=lambda x: x.get('training_time', 0))
    
    models = [r['model_name'] for r in sorted_results]
    times = [r.get('training_time', 0) for r in sorted_results]
    
    fig = px.bar(
        x=models, y=times,
        title='Model Training Times',
        color=times,
        color_continuous_scale='Viridis',
        labels={'x': 'Model', 'y': 'Time (seconds)'}
    )
    
    fig.update_layout(
        title_x=0.5,
        height=400,
        showlegend=False
    )
    
    return fig
