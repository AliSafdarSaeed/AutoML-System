"""
trainer.py - ModelTrainer Class
Main training orchestrator for AutoML classification
"""

import numpy as np
import pandas as pd
import time
from typing import Dict, List, Any

from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report
)

from .model_configs import MODEL_CONFIGS, get_available_models


class ModelTrainer:
    """
    AutoML Model Trainer for Classification tasks.
    Supports 7 classifiers with hyperparameter tuning via GridSearchCV.
    """
    
    def __init__(self, cv_folds: int = 3, scoring: str = 'f1_weighted'):
        """
        Initialize the ModelTrainer.
        
        Args:
            cv_folds: Number of cross-validation folds (default: 3)
            scoring: Scoring metric for GridSearchCV (default: 'f1_weighted')
        """
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.trained_models = {}
        self.results = []
        self.MODELS = MODEL_CONFIGS  # For backward compatibility
    
    def get_available_models(self) -> List[str]:
        """Return list of available model names."""
        return get_available_models()
    
    def train_model(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray, 
        model_name: str,
        use_grid_search: bool = True
    ) -> Dict[str, Any]:
        """
        Train a single model with optional GridSearchCV.
        
        Args:
            X_train: Training features
            y_train: Training labels
            model_name: Name of the model to train
            use_grid_search: Whether to use GridSearchCV (default: True)
            
        Returns:
            Dictionary containing trained model and training info
        """
        if model_name not in MODEL_CONFIGS:
            raise ValueError(f"Model '{model_name}' not found. Available: {self.get_available_models()}")
        
        model_config = MODEL_CONFIGS[model_name]
        start_time = time.time()
        
        try:
            if use_grid_search:
                # Initialize base model with default params
                base_model = model_config['model'](**model_config['default_params'])
                
                # GridSearchCV
                grid_search = GridSearchCV(
                    estimator=base_model,
                    param_grid=model_config['params'],
                    cv=self.cv_folds,
                    scoring=self.scoring,
                    n_jobs=-1,
                    error_score='raise'
                )
                
                grid_search.fit(X_train, y_train)
                
                trained_model = grid_search.best_estimator_
                best_params = grid_search.best_params_
                cv_score = grid_search.best_score_
            else:
                # Train with default parameters only
                trained_model = model_config['model'](**model_config['default_params'])
                trained_model.fit(X_train, y_train)
                best_params = model_config['default_params']
                cv_score = cross_val_score(
                    trained_model, X_train, y_train, 
                    cv=self.cv_folds, scoring=self.scoring
                ).mean()
            
            training_time = time.time() - start_time
            
            result = {
                'model_name': model_name,
                'model': trained_model,
                'best_params': best_params,
                'cv_score': cv_score,
                'training_time': training_time,
                'success': True,
                'error': None
            }
            
            self.trained_models[model_name] = trained_model
            
        except Exception as e:
            training_time = time.time() - start_time
            result = {
                'model_name': model_name,
                'model': None,
                'best_params': {},
                'cv_score': 0,
                'training_time': training_time,
                'success': False,
                'error': str(e)
            }
        
        return result
    
    def evaluate_model(
        self, 
        model: Any, 
        X_test: np.ndarray, 
        y_test: np.ndarray,
        model_name: str = None
    ) -> Dict[str, Any]:
        """
        Evaluate a trained model on test data.
        
        Args:
            model: Trained sklearn model
            X_test: Test features
            y_test: Test labels
            model_name: Optional name for the model
            
        Returns:
            Dictionary containing evaluation metrics
        """
        y_pred = model.predict(X_test)
        
        # Calculate metrics (weighted average for multi-class)
        metrics = {
            'model_name': model_name,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'y_pred': y_pred,
            'y_test': y_test
        }
        
        # Get classification report
        metrics['classification_report'] = classification_report(
            y_test, y_pred, zero_division=0, output_dict=True
        )
        
        return metrics
    
    def train_all_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        selected_models: List[str] = None,
        use_grid_search: bool = True,
        progress_callback = None
    ) -> List[Dict[str, Any]]:
        """
        Train and evaluate all selected models.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            selected_models: List of model names to train (default: all)
            use_grid_search: Whether to use GridSearchCV
            progress_callback: Optional callback function for progress updates
            
        Returns:
            List of dictionaries containing results for each model
        """
        if selected_models is None:
            selected_models = self.get_available_models()
        
        self.results = []
        
        for i, model_name in enumerate(selected_models):
            if progress_callback:
                progress_callback(model_name, i + 1, len(selected_models))
            
            # Train
            train_result = self.train_model(X_train, y_train, model_name, use_grid_search)
            
            if train_result['success']:
                # Evaluate
                eval_result = self.evaluate_model(
                    train_result['model'], X_test, y_test, model_name
                )
                
                # Combine results
                combined = {
                    **train_result,
                    **eval_result,
                    'training_time': train_result['training_time']
                }
            else:
                combined = {
                    **train_result,
                    'accuracy': 0,
                    'precision': 0,
                    'recall': 0,
                    'f1_score': 0
                }
            
            self.results.append(combined)
        
        return self.results
    
    def get_best_model(self) -> Dict[str, Any]:
        """
        Get the best performing model based on F1-Score.
        
        Returns:
            Dictionary containing best model information
        """
        if not self.results:
            return None
        
        successful_results = [r for r in self.results if r.get('success', False)]
        
        if not successful_results:
            return None
        
        best = max(successful_results, key=lambda x: x.get('f1_score', 0))
        return best
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """
        Get results as a pandas DataFrame, sorted by F1-Score.
        
        Returns:
            DataFrame with model comparison results
        """
        if not self.results:
            return pd.DataFrame()
        
        df = pd.DataFrame([
            {
                'Model': r['model_name'],
                'Accuracy': r.get('accuracy', 0),
                'Precision': r.get('precision', 0),
                'Recall': r.get('recall', 0),
                'F1-Score': r.get('f1_score', 0),
                'Training Time (s)': r.get('training_time', 0),
                'Status': 'Success' if r.get('success', False) else 'Failed'
            }
            for r in self.results
        ])
        
        return df.sort_values('F1-Score', ascending=False).reset_index(drop=True)
