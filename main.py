"""
main.py - Main Entry Point for AutoML Pro

Interactive sidebar with theme toggle, pipeline steps, and live stats.
"""

import streamlit as st
from pathlib import Path

# Import page modules
from modules import (
    page_ingestion,
    page_eda,
    page_quality,
    page_training,
    page_report
)


# =============================================================================
# PAGE CONFIG & STYLING
# =============================================================================

st.set_page_config(
    page_title="AutoML Pro",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded"
)


def load_css() -> None:
    """Load custom CSS and apply theme class."""
    css_file = Path(__file__).parent / "assets" / "styles.css"
    if css_file.exists():
        with open(css_file, encoding='utf-8') as f:
            css = f.read()
        
        # Apply theme variables directly into a style block
        theme = st.session_state.get('theme', 'dark')
        
        # Inject the custom CSS and the theme-specific overrides
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
        
        if theme == 'light':
            st.markdown("""
            <style>
                :root {
                    --bg-app: #f8fafc;
                    --bg-elevated: #ffffff;
                    --bg-surface: #f1f5f9;
                    --bg-hover: #e2e8f0;
                    --bg-active: #cbd5e1;
                    --border-subtle: rgba(0, 0, 0, 0.04);
                    --border-default: rgba(0, 0, 0, 0.08);
                    --border-strong: rgba(0, 0, 0, 0.12);
                    --text-primary: #0f172a;
                    --text-secondary: #475569;
                    --text-muted: #64748b;
                    --text-disabled: #94a3b8;
                    --accent-primary: #4f46e5;
                    --accent-primary-hover: #6366f1;
                    --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.05);
                    --shadow-md: 0 4px 12px rgba(0, 0, 0, 0.08);
                    --shadow-lg: 0 12px 32px rgba(0, 0, 0, 0.12);
                    --shadow-glow: 0 0 20px rgba(79, 70, 229, 0.1);
                }
            </style>
            """, unsafe_allow_html=True)


load_css()


# =============================================================================
# SESSION STATE
# =============================================================================

def init_session_state() -> None:
    """Initialize session state with default values."""
    defaults = {
        'df': None,
        'df_clean': None,
        'target_col': None,
        'file_name': None,
        'issues': None,
        'preprocess_config': {},
        'preprocessing_log': [],
        'results': None,
        'trainer': None,
        'current_page': 'Upload',
        'theme': 'dark'
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


# =============================================================================
# PIPELINE STEPS
# =============================================================================

PIPELINE_STEPS = [
    {"name": "Upload", "key": "upload", "description": "Import your dataset"},
    {"name": "Explore", "key": "eda", "description": "Analyze patterns"},
    {"name": "Quality", "key": "quality", "description": "Fix data issues"},
    {"name": "Training", "key": "training", "description": "Build models"},
    {"name": "Report", "key": "report", "description": "Export results"},
]


def get_step_status(step_key: str) -> str:
    """Get the status of a pipeline step."""
    if step_key == "upload":
        return "completed" if st.session_state.df is not None else "pending"
    elif step_key == "eda":
        return "completed" if st.session_state.target_col is not None else "pending"
    elif step_key == "quality":
        return "completed" if st.session_state.df_clean is not None else "pending"
    elif step_key == "training":
        return "completed" if st.session_state.results is not None else "pending"
    elif step_key == "report":
        return "pending"
    return "pending"


def get_current_step_index() -> int:
    """Get the current step index based on page."""
    page_map = {"Upload": 0, "Explore": 1, "EDA": 1, "Quality": 2, "Training": 3, "Report": 4}
    return page_map.get(st.session_state.current_page, 0)


# =============================================================================
# SIDEBAR
# =============================================================================

def render_sidebar() -> None:
    """Render interactive sidebar with theme toggle and pipeline."""
    with st.sidebar:
        # ===== BRAND =====
        st.markdown("""
        <div style="text-align: center; padding: var(--space-6) 0;">
            <div style="
                font-size: 20px;
                font-weight: 800;
                color: var(--text-primary);
                letter-spacing: -0.02em;
            ">AutoML Pro</div>
            <div style="
                color: var(--text-muted);
                font-size: 11px;
                margin-top: 4px;
            ">Classification System</div>
        </div>
        """, unsafe_allow_html=True)
        
        # ===== THEME TOGGLE =====
        st.markdown('<div class="sidebar-section-title">Appearance</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Light", key="theme_light", 
                        type="primary" if st.session_state.theme == "light" else "secondary"):
                st.session_state.theme = "light"
                st.rerun()
        with col2:
            if st.button("Dark", key="theme_dark",
                        type="primary" if st.session_state.theme == "dark" else "secondary"):
                st.session_state.theme = "dark"
                st.rerun()
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # ===== PIPELINE STEPS =====
        st.markdown('<div class="sidebar-section-title">Workflow</div>', unsafe_allow_html=True)
        
        current_idx = get_current_step_index()
        
        for idx, step in enumerate(PIPELINE_STEPS):
            status = get_step_status(step["key"])
            is_current = idx == current_idx
            is_disabled = False
            
            # Determine if step should be disabled
            if step["key"] == "eda" and st.session_state.df is None:
                is_disabled = True
            elif step["key"] == "quality" and (st.session_state.df is None or st.session_state.target_col is None):
                is_disabled = True
            elif step["key"] == "training" and st.session_state.df_clean is None:
                is_disabled = True
            elif step["key"] == "report" and st.session_state.results is None:
                is_disabled = True
            
            # Step card classes
            card_class = "nav-card"
            if is_current: card_class += " active"
            if status == "completed": card_class += " completed"
            
            # Checkmark for completed steps
            step_indicator = "✓" if status == "completed" and not is_current else str(idx + 1)
            
            # Combined HTML Wrapper with Invisible Button Overlay
            st.markdown(f"""
            <div class="nav-item-wrapper" style="opacity: {'0.5' if is_disabled else '1'};">
                <div class="{card_class}">
                    <div class="step-number" style="
                        width: 24px;
                        height: 24px;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        background: {'rgba(255,255,255,0.2)' if is_current else 'var(--bg-surface)'};
                        border-radius: 50%;
                        font-size: 11px;
                        font-weight: 700;
                    ">{step_indicator}</div>
                    <div>
                        <div style="font-weight: 600; font-size: 13px;">{step["name"]}</div>
                        <div style="font-size: 11px; opacity: 0.8;">{step["description"]}</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            if not is_disabled:
                if st.button(f"Go to {step['name']}", key=f"nav_{step['key']}"):
                    page_name = "EDA" if step["key"] == "eda" else step["name"]
                    st.session_state.current_page = page_name
                    st.rerun()
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # ===== SYSTEM HEALTH =====
        st.markdown('<div class="sidebar-section-title">System Status</div>', unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="health-monitor">
            <div style="display: flex; align-items: center; margin-bottom: 8px;">
                <span class="health-status-dot"></span>
                <span style="font-size: 13px; font-weight: 600;">System Online</span>
            </div>
            <div style="font-size: 11px; color: var(--text-muted);">
                AutoML Engine V2.1 is processing requests normally. High performance mode active.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # ===== QUICK STATS =====
        if st.session_state.df is not None:
            st.markdown('<div class="sidebar-section-title">Dataset</div>', unsafe_allow_html=True)
            
            df = st.session_state.df
            
            st.markdown(f"""
            <div style="
                background: var(--bg-surface);
                border-radius: 8px;
                padding: 12px;
                border: 1px solid var(--border-default);
            ">
                <div style="font-weight: 600; color: var(--text-primary); font-size: 13px; margin-bottom: 8px;">
                    {st.session_state.file_name}
                </div>
                <div class="quick-stat">
                    <span class="quick-stat-label">Rows</span>
                    <span class="quick-stat-value">{df.shape[0]:,}</span>
                </div>
                <div class="quick-stat">
                    <span class="quick-stat-label">Columns</span>
                    <span class="quick-stat-value">{df.shape[1]}</span>
                </div>
                <div class="quick-stat">
                    <span class="quick-stat-label">Missing</span>
                    <span class="quick-stat-value">{df.isnull().sum().sum():,}</span>
                </div>
                <div class="quick-stat">
                    <span class="quick-stat-label">Target</span>
                    <span class="quick-stat-value">{st.session_state.target_col or '—'}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
        
        # ===== PROGRESS =====
        completed_steps = sum(1 for step in PIPELINE_STEPS if get_step_status(step["key"]) == "completed")
        progress_pct = completed_steps / len(PIPELINE_STEPS)
        
        st.markdown(f"""
        <div style="margin-top: auto;">
            <div class="sidebar-section-title">Progress</div>
            <div style="
                display: flex;
                justify-content: space-between;
                font-size: 12px;
                color: var(--text-muted);
                margin-bottom: 8px;
            ">
                <span>{completed_steps}/{len(PIPELINE_STEPS)} steps</span>
                <span>{int(progress_pct * 100)}%</span>
            </div>
            <div style="
                height: 4px;
                background: var(--bg-surface);
                border-radius: 4px;
                overflow: hidden;
            ">
                <div style="
                    width: {progress_pct * 100}%;
                    height: 100%;
                    background: var(--accent-primary);
                    transition: width 0.3s ease;
                "></div>
            </div>
        </div>
        """, unsafe_allow_html=True)


render_sidebar()


# =============================================================================
# MAIN APP ROUTING
# =============================================================================

def main() -> None:
    """Main application entry point with page routing."""
    current_page = st.session_state.current_page
    
    if current_page == "Upload":
        page_ingestion()
    elif current_page == "EDA" or current_page == "Explore":
        page_eda()
    elif current_page == "Quality":
        page_quality()
    elif current_page == "Training":
        page_training()
    elif current_page == "Report":
        page_report()
    else:
        page_ingestion()


if __name__ == "__main__":
    main()
