"""
reporting.py - PDF Report Generation
Functions and classes for generating downloadable PDF reports
"""

from fpdf import FPDF
from datetime import datetime


class PDFReport(FPDF):
    """Custom PDF class for AutoML reports."""
    
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=15)
    
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'AutoML Classification Report', border=False, ln=True, align='C')
        self.ln(5)
    
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', align='C')
    
    def chapter_title(self, title):
        self.set_font('Arial', 'B', 14)
        self.set_fill_color(52, 152, 219)
        self.set_text_color(255, 255, 255)
        self.cell(0, 10, title, border=True, ln=True, fill=True)
        self.set_text_color(0, 0, 0)
        self.ln(2)
    
    def sub_title(self, title):
        self.set_font('Arial', 'B', 11)
        self.cell(0, 8, title, ln=True)
    
    def body_text(self, text):
        self.set_font('Arial', '', 10)
        self.multi_cell(0, 6, text)
        self.ln(2)


def generate_pdf_report(
    dataset_name: str,
    dataset_shape: tuple,
    detected_issues: dict,
    preprocessing_steps: list,
    model_results: list,
    best_model: dict
) -> bytes:
    """
    Generate a PDF report summarizing the AutoML run.
    Covers all 7 project requirements: Overview, EDA, Issues, Preprocessing, 
    Configs, Comparison, and Best Model Summary.
    """
    pdf = PDFReport()
    pdf.add_page()
    
    # Title and date
    pdf.set_font('Arial', 'B', 20)
    pdf.cell(0, 15, 'AutoML Classification Report', ln=True, align='C')
    pdf.set_font('Arial', 'I', 10)
    pdf.cell(0, 8, f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', ln=True, align='C')
    pdf.ln(10)
    
    # 1. Dataset Overview
    pdf.chapter_title('1. Dataset Overview')
    pdf.body_text(f'Dataset Name: {dataset_name}')
    pdf.body_text(f'Number of Rows: {dataset_shape[0]:,}')
    pdf.body_text(f'Number of Columns: {dataset_shape[1]}')
    pdf.ln(2)
    
    # 2 & 3. EDA Findings & Detected Issues
    pdf.chapter_title('2. EDA & Detected Issues')
    if detected_issues:
        for issue_type, count in detected_issues.items():
            pdf.body_text(f"- {str(issue_type).replace('_', ' ').title()}: {count} found")
    else:
        pdf.body_text("Automated EDA performed. No critical issues (missing values or outliers) were flagged.")
    pdf.ln(2)
    
    # 4. Preprocessing Decisions Taken
    pdf.chapter_title('3. Preprocessing Decisions')
    if preprocessing_steps:
        for step in preprocessing_steps:
            pdf.body_text(f" - {step}")
    else:
        pdf.body_text("Standard analysis complete. No specific user-approved preprocessing actions recorded.")
    pdf.ln(2)
    
    # 5 & 6. Model Configurations & Comparison Table
    pdf.chapter_title('4. Model Evaluation & Configurations')
    if model_results:
        # Table header
        pdf.set_font('Arial', 'B', 9)
        col_widths = [55, 25, 25, 25, 25, 30]
        headers = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Time (s)']
        
        for w, h in zip(col_widths, headers):
            pdf.cell(w, 8, h, border=1, align='C')
        pdf.ln()
        
        # Table rows
        pdf.set_font('Arial', '', 9)
        for result in model_results:
            pdf.cell(col_widths[0], 7, str(result.get('model_name', 'N/A'))[:25], border=1)
            pdf.cell(col_widths[1], 7, f"{result.get('accuracy', 0):.4f}", border=1, align='C')
            pdf.cell(col_widths[2], 7, f"{result.get('precision', 0):.4f}", border=1, align='C')
            pdf.cell(col_widths[3], 7, f"{result.get('recall', 0):.4f}", border=1, align='C')
            pdf.cell(col_widths[4], 7, f"{result.get('f1_score', 0):.4f}", border=1, align='C')
            pdf.cell(col_widths[5], 7, f"{result.get('training_time', 0):.3f}", border=1, align='C')
            pdf.ln()
            
        # 5. Hyperparameters details
        pdf.ln(5)
        pdf.sub_title("Model Hyperparameters:")
        for result in model_results:
            params = result.get('best_params', {})
            if params:
                param_str = ", ".join([f"{k}={v}" for k, v in params.items()])
                pdf.body_text(f"{result.get('model_name')}: {param_str}")
    else:
        pdf.body_text('No models were trained.')
    
    pdf.ln(5)
    
    # 7. Best Model Summary + Justification
    pdf.chapter_title('5. Best Model Summary')
    if best_model:
        pdf.sub_title(f"Recommendation: {best_model.get('model_name', 'N/A')}")
        pdf.ln(2)
        pdf.body_text(f"Accuracy: {best_model.get('accuracy', 0):.4f}")
        pdf.body_text(f"F1-Score: {best_model.get('f1_score', 0):.4f}")
        pdf.ln(2)
        pdf.sub_title('Justification:')
        pdf.body_text(
            f"The {best_model.get('model_name', 'selected model')} is recommended for deployment because it "
            f"achieved the optimal balance between classification performance and computational efficiency. "
            f"It reached an F1-Score of {best_model.get('f1_score', 0):.4f}, outperforming other models in its class."
        )
    else:
        pdf.body_text('Final best model selection could not be determined.')
    
    # Return bytes (fixes 'string argument without encoding' error)
    return pdf.output(dest='S').encode('latin-1')