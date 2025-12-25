"""
training_ui.py - Model Training Page

Real-time leaderboard and training controls.
Preserves existing logic from views/page_training.py.
"""

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import List, Dict, Any

from .components import (
    render_page_header,
    render_section_header,
    render_metric_card,
    render_alert,
    render_best_model_card,
    render_proceed_button
)
from models import (
    ModelTrainer,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_model_comparison,
    plot_training_times
)


def page_training() -> None:
    """
    Render the model training page with leaderboard and visualizations.
    """
    render_page_header(
        "Model Training",
        "Train and evaluate machine learning models"
    )
    
    # Removed Back navigation in guard clause
    if st.session_state.df_clean is None:
        render_alert("Please complete preprocessing first", "warning")
        return
    
    df = st.session_state.df_clean
    target_col = st.session_state.target_col
    config = st.session_state.preprocess_config
    
    # Prepare data
    try:
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        if X.isnull().any().any() or y.isnull().any():
            render_alert("Data still contains NaN values. Please fix in preprocessing.", "error")
            return
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=config.get('test_size', 0.2),
            random_state=42,
            stratify=y
        )
    except Exception as e:
        error_msg = str(e)
        if "The least populated classes in y have only 1 member" in error_msg:
            render_alert(
                "Training Error: Some classes have only 1 member. Stratified splitting requires at least 2 members per class. Please reconsider your dataset selection or target variable.",
                "error"
            )
        else:
            render_alert(f"Error: {error_msg}", "error")
        return
    
    # Data split summary
    render_section_header("Data Split Summary")
    
    cols = st.columns(3)
    with cols[0]:
        render_metric_card(f"{len(X_train):,}", "Training Samples")
    with cols[1]:
        render_metric_card(f"{len(X_test):,}", "Test Samples")
    with cols[2]:
        render_metric_card(str(X.shape[1]), "Features")
    
    # Model Selection with AI Recommendations
    render_section_header("Model Selection")
    
    # Get smart recommendations
    from .model_recommendations import get_model_recommendations
    recommended_models, reasoning = get_model_recommendations(
        df, target_col, X.shape[1], len(df)
    )
    
    # Show AI recommendation
    st.info(f"🤖 **AI Recommendation**\n\n{reasoning}", icon="💡")
    
    trainer = ModelTrainer(cv_folds=3, scoring='f1_weighted')
    models = trainer.get_available_models()
    
    # Use AI-recommended models as default
    default_selection = [m for m in recommended_models if m in models]
    
    col1, col2 = st.columns([3, 1])
    with col1:
        selected = st.multiselect(
            "Choose models to train",
            models,
            default=default_selection,
            help=f"AI suggests: {', '.join(recommended_models)}",
            label_visibility="collapsed"
        )
    with col2:
        use_grid = st.checkbox("Use GridSearchCV", value=True)
    
    st.markdown("""
    <div style="color: var(--text-muted); font-size: 0.875rem; margin-top: 0.5rem;">
        3-Fold Cross-Validation · F1-Weighted Scoring
    </div>
    """, unsafe_allow_html=True)
    
    # Train Button
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("Train Models", use_container_width=True, type="primary", disabled=not selected):
        progress = st.progress(0)
        
        with st.status("Training models...", expanded=True) as status:
            results = []
            for i, name in enumerate(selected):
                st.write(f"Training {name}...")
                progress.progress((i + 1) / len(selected))
                
                train_res = trainer.train_model(X_train, y_train, name, use_grid)
                if train_res['success']:
                    eval_res = trainer.evaluate_model(train_res['model'], X_test, y_test, name)
                    results.append({**train_res, **eval_res})
                else:
                    results.append(train_res)
            
            trainer.results = results
            st.session_state.results = results
            st.session_state.trainer = trainer
            
            status.update(label="Training complete", state="complete")
        
        success_count = len([r for r in results if r.get('success')])
        st.toast(f"{success_count} models trained successfully", icon="✅")
        st.balloons()  # Process completion animation
    
    # Results
    if st.session_state.results:
        results = st.session_state.results
        trainer = st.session_state.trainer
        
        render_section_header("Model Leaderboard")
        
        results_df = trainer.get_results_dataframe()
        st.dataframe(
            results_df.style.format({
                'Accuracy': '{:.4f}',
                'Precision': '{:.4f}',
                'Recall': '{:.4f}',
                'F1-Score': '{:.4f}',
                'Training Time (s)': '{:.3f}'
            }).background_gradient(subset=['F1-Score'], cmap='Blues'),
            use_container_width=True
        )
        
        # Best Model Card
        best = trainer.get_best_model()
        if best:
            st.markdown("<br>", unsafe_allow_html=True)
            render_best_model_card(
                best['model_name'],
                best['f1_score'],
                best['accuracy'],
                best['training_time']
            )
        
        # Visualizations
        render_section_header("Performance Analysis")
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "Comparison",
            "Training Time",
            "Confusion Matrix",
            "ROC Curve"
        ])
        
        with tab1:
            st.plotly_chart(plot_model_comparison(results), width='stretch')
        
        with tab2:
            st.plotly_chart(plot_training_times(results), width='stretch')
        
        with tab3:
            success_models = [r['model_name'] for r in results if r.get('success')]
            if success_models:
                sel = st.selectbox("Select model", success_models, key="cm_sel")
                res = next(r for r in results if r['model_name'] == sel)
                if 'y_pred' in res:
                    st.plotly_chart(
                        plot_confusion_matrix(res['y_test'], res['y_pred'], sorted(y.unique())),
                        width='stretch'
                    )
        
        with tab4:
            if success_models:
                sel = st.selectbox("Select model", success_models, key="roc_sel")
                res = next(r for r in results if r['model_name'] == sel)
                if res.get('model'):
                    st.plotly_chart(
                        plot_roc_curve(res['model'], X_test.values, y_test.values, sel),
                        width='stretch'
                    )
        
        # Proceed to Report button
        render_proceed_button(
            next_page="Report",
            label="Proceed to Generate Report",
            disabled=False
        )
