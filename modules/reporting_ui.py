"""
reporting_ui.py - Report Generation Page

Document-style layout with prominent PDF download CTA.
Preserves existing logic from views/page_report.py.
"""

import streamlit as st
from typing import Dict, Any, List, Optional

from .components import (
    render_page_header,
    render_section_header,
    render_metric_card,
    render_alert
)
from data_utils import generate_pdf_report


def page_report() -> None:
    """
    Render the report generation page with PDF download.
    """
    render_page_header(
        "Analysis Report",
        "Generate a comprehensive PDF report of your ML pipeline"
    )
    
    # Removed Back navigation in guard clause
    if 'results' not in st.session_state or st.session_state.results is None:
        render_alert("Please train models first in the Training section", "warning")
        return
    
    # Project Summary
    render_section_header("Project Summary")
    
    cols = st.columns(3)
    with cols[0]:
        file_name = st.session_state.get('file_name', 'N/A')
        render_metric_card(file_name, "Dataset")
    with cols[1]:
        if 'df' in st.session_state and st.session_state.df is not None:
            shape_str = f"{st.session_state.df.shape[0]:,} × {st.session_state.df.shape[1]}"
        else:
            shape_str = "N/A"
        render_metric_card(shape_str, "Shape")
    with cols[2]:
        render_metric_card(str(len(st.session_state.results)), "Models Trained")
    
    # Preprocessing Steps
    render_section_header("Preprocessing Applied")
    
    if st.session_state.get('preprocessing_log'):
        for step in st.session_state.preprocessing_log:
            st.markdown(f"""
            <div style="
                padding: 0.75rem 1rem;
                background: var(--bg-card);
                border-radius: var(--radius-sm);
                margin-bottom: 0.5rem;
                color: var(--text-secondary);
                border-left: 3px solid var(--success);
            ">
                {step}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="color: var(--text-muted); padding-left: 1rem;">
            Standard cleaning and defaults applied
        </div>
        """, unsafe_allow_html=True)
    
    # Results Table
    render_section_header("Model Performance")
    
    trainer = st.session_state.get('trainer')
    if trainer:
        results_df = trainer.get_results_dataframe()
        st.dataframe(results_df, use_container_width=True)
        
        best = trainer.get_best_model()
        if best:
            render_alert(
                f"Recommended Model: {best['model_name']} with F1 Score of {best['f1_score']:.4f}",
                "success"
            )
    
    # Download Section
    render_section_header("Generate Report")
    
    st.markdown("""
    <div style="
        background: var(--bg-card);
        border: 1px solid var(--border-default);
        border-radius: var(--radius-lg);
        padding: 2rem;
        text-align: center;
        margin-bottom: 1.5rem;
    ">
        <div style="font-size: 1.125rem; font-weight: 600; color: var(--text-primary); margin-bottom: 0.5rem;">
            Ready to Export
        </div>
        <div style="color: var(--text-muted); font-size: 0.875rem;">
            Generate a comprehensive PDF report containing all analysis results, visualizations, and recommendations.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("Download PDF", use_container_width=True, type="primary"):
        with st.status("Generating report...", expanded=True) as status:
            try:
                st.write("Compiling results...")
                
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
                
                detected_issues = st.session_state.get('detected_issues', {})
                
                st.write("Building PDF...")
                pdf_bytes = generate_pdf_report(
                    st.session_state.get('file_name', 'Unknown'),
                    st.session_state.df.shape if 'df' in st.session_state else (0, 0),
                    detected_issues,
                    st.session_state.get('preprocessing_log', []),
                    model_results,
                    best_model
                )
                
                status.update(label="Report ready", state="complete")
                
                st.download_button(
                    "Click to Download",
                    data=pdf_bytes,
                    file_name="AutoML_Evaluation_Report.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
                
                st.balloons()
                
            except Exception as e:
                status.update(label="Error", state="error")
                render_alert(f"Error generating report: {str(e)}", "error")
    
    # Reset Option
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style="border-top: 1px solid var(--border-default); padding-top: 2rem;">
    """, unsafe_allow_html=True)
    
    if st.button("Reset All", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
