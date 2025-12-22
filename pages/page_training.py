"""page_training.py - Model Training Page"""
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from ui_components import *
from models import ModelTrainer, plot_confusion_matrix, plot_roc_curve, plot_model_comparison, plot_training_times

def page_training():
    render_header("🎯", "Model Training", "Train and evaluate machine learning models")
    
    if st.session_state.df_clean is None:
        render_alert("Please complete preprocessing first", "warning")
        return
    
    df = st.session_state.df_clean
    target_col = st.session_state.target_col
    config = st.session_state.preprocess_config
    
    try:
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        if X.isnull().any().any() or y.isnull().any():
            render_alert("Data still contains NaN values. Please fix in preprocessing.", "error")
            return
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config.get('test_size', 0.2), random_state=42, stratify=y)
    except Exception as e:
        error_msg = str(e)
        if "The least populated classes in y have only 1 member" in error_msg:
            render_alert("Training Error: Some classes in your target variable have only 1 member. Stratified splitting requires at least 2 members per class.", "error")
            st.info("Try selecting a different target variable or use a dataset with more samples per class.")
            
            if st.button("⬅️ Go Back to Data Upload", key="back_to_upload_err"):
                # Complete reset of dataset-related session state
                keys_to_reset = [
                    'target_col', 'df', 'df_clean', 'results', 
                    'trainer', 'file_name', 'issues', 
                    'preprocess_config', 'preprocessing_log'
                ]
                for key in keys_to_reset:
                    st.session_state[key] = None if key != 'preprocessing_log' else []
                st.session_state.current_page = "Upload & EDA"
                st.rerun()
        else:
            render_alert(f"Error: {error_msg}", "error")
        return
    
    # Data split info
    cols = st.columns(3)
    with cols[0]:
        render_metric_card(len(X_train), "Training Samples", "🎓")
    with cols[1]:
        render_metric_card(len(X_test), "Test Samples", "🧪")
    with cols[2]:
        render_metric_card(X.shape[1], "Features", "📐")
    
    # Model Selection
    render_section_header("🤖", "Select Models")
    
    trainer = ModelTrainer(cv_folds=3, scoring='f1_weighted')
    models = trainer.get_available_models()
    
    selected = st.multiselect("Choose models to train:", models, default=models, help="All models use GridSearchCV for hyperparameter tuning")
    
    col1, col2 = st.columns(2)
    with col1:
        use_grid = st.checkbox("Use GridSearchCV (recommended)", value=True)
    with col2:
        st.markdown("<span style='color: rgba(255,255,255,0.5);'>3-Fold CV | F1-Weighted Scoring</span>", unsafe_allow_html=True)
    
    # Train Button
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("🚀 Train Selected Models", use_container_width=True, type="primary", disabled=not selected):
        progress = st.progress(0)
        status = st.empty()
        
        results = []
        for i, name in enumerate(selected):
            status.markdown(f"<div style='text-align: center; color: rgba(255,255,255,0.7);'>Training {name}... ({i+1}/{len(selected)})</div>", unsafe_allow_html=True)
            progress.progress((i+1)/len(selected))
            
            train_res = trainer.train_model(X_train, y_train, name, use_grid)
            if train_res['success']:
                eval_res = trainer.evaluate_model(train_res['model'], X_test, y_test, name)
                results.append({**train_res, **eval_res})
            else:
                results.append(train_res)
        
        trainer.results = results
        st.session_state.results = results
        st.session_state.trainer = trainer
        
        status.empty()
        render_alert(f"Training complete! {len([r for r in results if r.get('success')])} models trained successfully.", "success")
    
    # Results
    if st.session_state.results:
        results = st.session_state.results
        trainer = st.session_state.trainer
        
        render_section_header("📊", "Results")
        
        results_df = trainer.get_results_dataframe()
        st.dataframe(
            results_df.style.format({
                'Accuracy': '{:.4f}', 'Precision': '{:.4f}',
                'Recall': '{:.4f}', 'F1-Score': '{:.4f}',
                'Training Time (s)': '{:.3f}'
            }).background_gradient(subset=['F1-Score'], cmap='Greens'),
            use_container_width=True
        )
        
        # Best Model Card
        best = trainer.get_best_model()
        if best:
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, rgba(16,185,129,0.15), rgba(6,182,212,0.15));
                border: 2px solid rgba(16,185,129,0.4);
                border-radius: 20px;
                padding: 2rem;
                margin: 2rem 0;
                position: relative;
                overflow: hidden;
            ">
                <div style="position: absolute; top: -30px; right: -30px; font-size: 150px; opacity: 0.1;">🏆</div>
                <div style="font-size: 0.875rem; color: rgba(255,255,255,0.5); text-transform: uppercase; letter-spacing: 0.1em;">Best Model</div>
                <div style="font-size: 1.75rem; font-weight: 800; color: #6ee7b7; margin: 0.5rem 0;">{best['model_name']}</div>
                <div style="display: flex; gap: 2rem; margin-top: 1rem;">
                    <div><span style="color: rgba(255,255,255,0.5);">F1-Score:</span> <span style="color: #fff; font-weight: 600;">{best['f1_score']:.4f}</span></div>
                    <div><span style="color: rgba(255,255,255,0.5);">Accuracy:</span> <span style="color: #fff; font-weight: 600;">{best['accuracy']:.4f}</span></div>
                    <div><span style="color: rgba(255,255,255,0.5);">Time:</span> <span style="color: #fff; font-weight: 600;">{best['training_time']:.3f}s</span></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Visualizations
        render_section_header("📈", "Visualizations")
        
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Comparison", "⏱️ Training Time", "🎯 Confusion Matrix", "📉 ROC Curve"])
        
        with tab1:
            st.plotly_chart(plot_model_comparison(results), use_container_width=True)
        
        with tab2:
            st.plotly_chart(plot_training_times(results), use_container_width=True)
        
        with tab3:
            success_models = [r['model_name'] for r in results if r.get('success')]
            if success_models:
                sel = st.selectbox("Select model:", success_models, key="cm_sel")
                res = next(r for r in results if r['model_name'] == sel)
                if 'y_pred' in res:
                    st.plotly_chart(plot_confusion_matrix(res['y_test'], res['y_pred'], sorted(y.unique())), use_container_width=True)
        
        with tab4:
            if success_models:
                sel = st.selectbox("Select model:", success_models, key="roc_sel")
                res = next(r for r in results if r['model_name'] == sel)
                if res.get('model'):
                    st.plotly_chart(plot_roc_curve(res['model'], X_test.values, y_test.values, sel), use_container_width=True)
        
        # Bottom Navigation
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("📊 Generate Report", use_container_width=True, type="primary"):
                st.session_state.current_page = "Report"
                st.rerun()

if __name__ == "__main__":
    page_training()