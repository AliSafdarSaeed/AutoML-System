"""
app.py - Main Entry Point for AutoML Classification System
Industrial-Grade UI/UX with Modular Architecture
"""

import streamlit as st
from pathlib import Path

# Import page modules
from pages import page_upload_eda, page_preprocessing, page_training, page_report
from sidebar import render_sidebar


# ============================================================================
# PAGE CONFIG & STYLING
# ============================================================================

st.set_page_config(
    page_title="AutoML Pro | ML Classification",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Load custom CSS
def load_css():
    css_file = Path(__file__).parent / "styles.css"
    if css_file.exists():
        with open(css_file, encoding='utf-8') as f:
            st.markdown(f"<style>{f.read()}</style>",  unsafe_allow_html=True)
    
    # Additional inline styles
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #0c0c1e 0%, #1a1a3e 50%, #0f0f23 100%);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main content area */
    .main .block-container {
        padding: 2rem 3rem 3rem;
        max-width: 1400px;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: rgba(255,255,255,0.05); border-radius: 10px; }
    ::-webkit-scrollbar-thumb { background: linear-gradient(135deg, #667eea, #764ba2); border-radius: 10px; }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0a1a 0%, #151530 100%);
        border-right: 1px solid rgba(255,255,255,0.08);
    }
    section[data-testid="stSidebar"] > div { padding-top: 1rem; }
    
    /* Button styles */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.4) !important;
    }
    .stButton > button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.6) !important;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background: rgba(255,255,255,0.03);
        border: 2px dashed rgba(102, 126, 234, 0.5);
        border-radius: 16px;
        padding: 2rem;
    }
    
    /* All other form elements */
    .stSelectbox > div > div, .stMultiSelect > div {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        border-radius: 10px !important;
    }
    
    .streamlit-expanderHeader {
        background: rgba(255,255,255,0.05) !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255,255,255,0.03);
        border-radius: 12px;
        padding: 8px;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: white !important;
    }
    
    .stProgress > div > div { background: linear-gradient(90deg, #06b6d4, #8b5cf6) !important; }
    </style>
    """, unsafe_allow_html=True)


load_css()


# ============================================================================
# SESSION STATE
# ============================================================================

def init_session_state():
    defaults = {
        'df': None, 'df_clean': None, 'target_col': None, 'file_name': None,
        'issues': None, 'preprocess_config': {}, 'preprocessing_log': [],
        'results': None, 'trainer': None, 'current_page': 'Upload & EDA'
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


# ============================================================================
# RENDER SIDEBAR
# ============================================================================

render_sidebar()


# ============================================================================
# MAIN APP ROUTING
# ============================================================================

def main():
    """Main application entry point."""
    
    # Route to appropriate page
    current_page = st.session_state.current_page
    
    if current_page == "Upload & EDA":
        page_upload_eda()
    elif current_page == "Preprocessing":
        page_preprocessing()
    elif current_page == "Training":
        page_training()
    elif current_page == "Report":
        page_report()
    else:
        page_upload_eda()


if __name__ == "__main__":
    main()
