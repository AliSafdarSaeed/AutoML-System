"""page_report.py - Download Report Page"""
import streamlit as st
from ui_components import *
from data_utils import generate_pdf_report

def page_report():
    """
    Main function for the Report Page.
    Restored to use the original session_state keys (results, df, etc.)
    """
    render_header("📊", "Download Report", "Generate a comprehensive PDF report of your analysis")
    
    # Validation
    if 'results' not in st.session_state or st.session_state.results is None:
        render_alert("Please train models first in the Training section", "warning")
        return
    
    # Summary Cards
    render_section_header("📋", "Project Summary")
    
    cols = st.columns(3)
    with cols[0]:
        render_metric_card(st.session_state.get('file_name', "N/A"), "Dataset", "📁")
    with cols[1]:
        if 'df' in st.session_state and st.session_state.df is not None:
            shape_str = f"{st.session_state.df.shape[0]:,} × {st.session_state.df.shape[1]}"
        else:
            shape_str = "N/A"
        render_metric_card(shape_str, "Shape", "📐")
    with cols[2]:
        render_metric_card(len(st.session_state.results), "Models Trained", "🤖")
    
    # Preprocessing Steps
    render_section_header("🔧", "Preprocessing Applied")
    
    if st.session_state.get('preprocessing_log'):
        for step in st.session_state.preprocessing_log:
            st.markdown(f"<div style='padding: 0.5rem 1rem; background: rgba(255,255,255,0.03); border-radius: 8px; margin-bottom: 0.5rem; color: rgba(255,255,255,0.8);'>✓ {step}</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='color: rgba(255,255,255,0.5); padding-left: 1rem;'>Standard cleaning and defaults applied</div>", unsafe_allow_html=True)
    
    # Results Table
    render_section_header("🏆", "Model Performance")
    
    trainer = st.session_state.get('trainer')
    if trainer:
        results_df = trainer.get_results_dataframe()
        st.dataframe(results_df, use_container_width=True)
        
        best = trainer.get_best_model()
        if best:
            render_alert(f"Recommended Model: {best['model_name']} (F1: {best['f1_score']:.4f})", "success")
    
    # Download Button
    render_section_header("📥", "Generate Official Report")
    
    if st.button("📄 Generate PDF Report", type="primary", use_container_width=True):
        with st.spinner("Compiling results into PDF..."):
            try:
                # Prepare data for generator
                model_results = []
                for r in st.session_state.results:
                    model_results.append({
                        'model_name': r.get('model_name', 'Unknown'),
                        'accuracy': r.get('accuracy', 0),
                        'precision': r.get('precision', 0),
                        'recall': r.get('recall', 0),
                        'f1_score': r.get('f1_score', 0),
                        'training_time': r.get('training_time', 0),
                        'best_params': r.get('best_params', {})
                    })
                
                best_model = None
                if trainer:
                    best = trainer.get_best_model()
                    if best:
                        best_model = {
                            'model_name': best.get('model_name'),
                            'accuracy': best.get('accuracy', 0),
                            'f1_score': best.get('f1_score', 0)
                        }
                
                # Retrieve detected issues from EDA/Preprocessing stage
                detected_issues = st.session_state.get('detected_issues', {})
                
                # Call the generator
                pdf_bytes = generate_pdf_report(
                    st.session_state.get('file_name', "Unknown"),
                    st.session_state.df.shape if 'df' in st.session_state else (0,0),
                    detected_issues,
                    st.session_state.get('preprocessing_log', []),
                    model_results,
                    best_model
                )
                
                st.download_button(
                    "⬇️ Click to Download PDF",
                    data=pdf_bytes,
                    file_name=f"AutoML_Evaluation_Report.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
                st.balloons()
                
            except Exception as e:
                render_alert(f"Error compiling report: {str(e)}", "error")
    
    # Reset Option
    st.markdown("<br><br>", unsafe_allow_html=True)
    if st.button("🗑️ Reset Application", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

if __name__ == "__main__":
    page_report()