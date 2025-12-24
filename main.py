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
    """Load custom CSS and apply theme-specific styles modularly."""
    # Get current theme
    theme = st.session_state.get('theme', 'dark')
    themes_dir = Path(__file__).parent / "assets" / "themes"
    
    # Define theme files in loading order
    theme_files = [
        f"{theme}_theme.css",       # Base theme & typography
        f"{theme}_sidebar.css",     # Sidebar styling
        f"{theme}_buttons.css",     # Buttons & interactions
        f"{theme}_forms.css",       # Forms, inputs, file uploader
        f"{theme}_data.css",        # Tables, tabs, expanders
        f"{theme}_alerts.css",      # Alerts, status, progress
        f"{theme}_components.css",  # Custom cards, metrics, badges
        f"{theme}_charts.css"       # Plotly charts & scrollbar
    ]
    
    # Load modular theme CSS files
    combined_css = ""
    for css_file_name in theme_files:
        css_path = themes_dir / css_file_name
        if css_path.exists():
            with open(css_path, encoding='utf-8') as f:
                combined_css += f"\n/* === {css_file_name} === */\n"
                combined_css += f.read()
                combined_css += "\n"
    
    if combined_css:
        st.markdown(f"<style>{combined_css}</style>", unsafe_allow_html=True)


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
    """Render ChatGPT-inspired sidebar with collapsible navigation."""
    
    theme = st.session_state.get('theme', 'dark')
    is_light = theme == 'light'
    
    # Dynamic colors based on theme
    text_primary = '#0f172a' if is_light else '#f8fafc'
    text_muted = '#64748b' if is_light else '#71717a'
    bg_surface = '#f1f5f9' if is_light else '#1e1e26'
    bg_hover = '#e2e8f0' if is_light else '#252530'
    border_color = 'rgba(0, 0, 0, 0.08)' if is_light else 'rgba(255, 255, 255, 0.08)'
    accent_primary = '#4f46e5' if is_light else '#6366f1'
    
    with st.sidebar:
        # ===== BRAND =====
        st.markdown(f"""
        <div style="text-align: center; padding: 20px 0; border-bottom: 1px solid {border_color}; margin-bottom: 16px;">
            <div style="
                display: inline-flex;
                align-items: center;
                justify-content: center;
                width: 40px;
                height: 40px;
                background: linear-gradient(135deg, {accent_primary}, #8b5cf6);
                border-radius: 10px;
                margin-bottom: 8px;
            ">
                <span style="color: white; font-size: 18px; font-weight: 700;">A</span>
            </div>
            <div style="
                font-size: 18px;
                font-weight: 700;
                color: {text_primary};
                letter-spacing: -0.02em;
            ">AutoML Pro</div>
            <div style="
                color: {text_muted};
                font-size: 11px;
                margin-top: 2px;
            ">Intelligent Classification</div>
        </div>
        """, unsafe_allow_html=True)
        
        # ===== THEME TOGGLE - Sleek toggle button =====
        st.markdown(f'<div style="font-size: 11px; font-weight: 600; color: {text_muted}; text-transform: uppercase; letter-spacing: 0.1em; padding: 0 12px; margin-bottom: 8px;">Theme</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("☀️ Light", key="theme_light", use_container_width=True,
                        type="primary" if is_light else "secondary"):
                st.session_state.theme = "light"
                st.rerun()
        with col2:
            if st.button("🌙 Dark", key="theme_dark", use_container_width=True,
                        type="primary" if not is_light else "secondary"):
                st.session_state.theme = "dark"
                st.rerun()
        
        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
        
        # ===== PIPELINE NAVIGATION - ChatGPT Style =====
        st.markdown(f'<div style="font-size: 11px; font-weight: 600; color: {text_muted}; text-transform: uppercase; letter-spacing: 0.1em; padding: 0 12px; margin-bottom: 8px;">Workflow</div>', unsafe_allow_html=True)
        
        current_idx = get_current_step_index()
        
        for idx, step in enumerate(PIPELINE_STEPS):
            status = get_step_status(step["key"])
            is_current = idx == current_idx
            is_completed = status == "completed"
            
            # Determine if step should be disabled
            is_disabled = False
            if step["key"] == "eda" and st.session_state.df is None:
                is_disabled = True
            elif step["key"] == "quality" and (st.session_state.df is None or st.session_state.target_col is None):
                is_disabled = True
            elif step["key"] == "training" and st.session_state.df_clean is None:
                is_disabled = True
            elif step["key"] == "report" and st.session_state.results is None:
                is_disabled = True
            
            # Step indicator
            if is_completed and not is_current:
                step_indicator = "✓"
                indicator_bg = 'rgba(34, 197, 94, 0.15)' if is_light else 'rgba(34, 197, 94, 0.2)'
                indicator_color = '#16a34a' if is_light else '#22c55e'
            else:
                step_indicator = str(idx + 1)
                if is_current:
                    indicator_bg = 'rgba(255,255,255,0.25)'
                    indicator_color = 'white'
                else:
                    indicator_bg = bg_surface
                    indicator_color = text_muted
            
            # Styling based on state
            if is_current:
                card_bg = accent_primary
                card_border = accent_primary
                card_text = 'white'
                card_desc = 'rgba(255,255,255,0.8)'
                card_shadow = f'0 4px 12px rgba(99, 102, 241, 0.4)'
            elif is_disabled:
                card_bg = 'transparent'
                card_border = 'transparent'
                card_text = text_muted
                card_desc = text_muted
                card_shadow = 'none'
            else:
                card_bg = 'transparent'
                card_border = 'transparent'
                card_text = text_primary
                card_desc = text_muted
                card_shadow = 'none'
            
            # Create clickable navigation item
            button_key = f"nav_{step['key']}"
            
            # Use a container for the navigation item
            if not is_disabled:
                if st.button(
                    f"{step_indicator}  {step['name']}", 
                    key=button_key, 
                    use_container_width=True,
                    type="primary" if is_current else "secondary",
                    disabled=is_disabled
                ):
                    page_name = "EDA" if step["key"] == "eda" else step["name"]
                    st.session_state.current_page = page_name
                    st.rerun()
                
                # Add description below button
                if is_current:
                    st.markdown(f"""
                    <div style="
                        font-size: 10px; 
                        color: {text_muted}; 
                        padding: 0 12px 8px 12px;
                        margin-top: -8px;
                    ">{step['description']}</div>
                    """, unsafe_allow_html=True)
            else:
                # Disabled state - show as text
                st.markdown(f"""
                <div style="
                    display: flex;
                    align-items: center;
                    gap: 12px;
                    padding: 10px 12px;
                    border-radius: 8px;
                    opacity: 0.4;
                    cursor: not-allowed;
                ">
                    <div style="
                        width: 24px;
                        height: 24px;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        background: {bg_surface};
                        border-radius: 50%;
                        font-size: 11px;
                        font-weight: 600;
                        color: {text_muted};
                    ">{step_indicator}</div>
                    <div>
                        <div style="font-weight: 500; font-size: 13px; color: {text_muted};">{step['name']}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
        
        # ===== SYSTEM STATUS - Compact =====
        st.markdown(f'<div style="font-size: 11px; font-weight: 600; color: {text_muted}; text-transform: uppercase; letter-spacing: 0.1em; padding: 0 12px; margin-bottom: 8px;">System</div>', unsafe_allow_html=True)
        
        st.markdown(f"""
        <div style="
            background: {bg_surface};
            border: 1px solid {border_color};
            border-radius: 10px;
            padding: 12px;
            margin: 0 4px;
        ">
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="
                    display: inline-block;
                    width: 8px;
                    height: 8px;
                    border-radius: 50%;
                    background: #22c55e;
                    box-shadow: 0 0 8px #22c55e;
                "></span>
                <span style="font-size: 12px; font-weight: 500; color: {text_primary};">Online</span>
            </div>
            <div style="font-size: 10px; color: {text_muted}; margin-top: 4px;">
                AutoML Engine v2.1
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
        
        # ===== QUICK STATS (if data loaded) =====
        if st.session_state.df is not None:
            st.markdown(f'<div style="font-size: 11px; font-weight: 600; color: {text_muted}; text-transform: uppercase; letter-spacing: 0.1em; padding: 0 12px; margin-bottom: 8px;">Dataset</div>', unsafe_allow_html=True)
            
            df = st.session_state.df
            
            st.markdown(f"""
            <div style="
                background: {bg_surface};
                border-radius: 10px;
                padding: 12px;
                border: 1px solid {border_color};
                margin: 0 4px;
            ">
                <div style="font-weight: 600; color: {text_primary}; font-size: 12px; margin-bottom: 10px; word-break: break-all;">
                    📊 {st.session_state.file_name}
                </div>
                <div style="display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px solid {border_color};">
                    <span style="font-size: 11px; color: {text_muted};">Rows</span>
                    <span style="font-size: 11px; font-weight: 600; color: {text_primary};">{df.shape[0]:,}</span>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px solid {border_color};">
                    <span style="font-size: 11px; color: {text_muted};">Columns</span>
                    <span style="font-size: 11px; font-weight: 600; color: {text_primary};">{df.shape[1]}</span>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 4px 0; border-bottom: 1px solid {border_color};">
                    <span style="font-size: 11px; color: {text_muted};">Missing</span>
                    <span style="font-size: 11px; font-weight: 600; color: {text_primary};">{df.isnull().sum().sum():,}</span>
                </div>
                <div style="display: flex; justify-content: space-between; padding: 4px 0;">
                    <span style="font-size: 11px; color: {text_muted};">Target</span>
                    <span style="font-size: 11px; font-weight: 600; color: {accent_primary};">{st.session_state.target_col or '—'}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
        
        # ===== PROGRESS =====
        completed_steps = sum(1 for step in PIPELINE_STEPS if get_step_status(step["key"]) == "completed")
        progress_pct = completed_steps / len(PIPELINE_STEPS)
        
        st.markdown(f"""
        <div style="margin: 0 4px;">
            <div style="font-size: 11px; font-weight: 600; color: {text_muted}; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 8px;">Progress</div>
            <div style="
                display: flex;
                justify-content: space-between;
                font-size: 11px;
                color: {text_muted};
                margin-bottom: 6px;
            ">
                <span>{completed_steps}/{len(PIPELINE_STEPS)} steps</span>
                <span style="font-weight: 600; color: {text_primary};">{int(progress_pct * 100)}%</span>
            </div>
            <div style="
                height: 6px;
                background: {bg_surface};
                border-radius: 6px;
                overflow: hidden;
            ">
                <div style="
                    width: {progress_pct * 100}%;
                    height: 100%;
                    background: linear-gradient(90deg, {accent_primary}, #8b5cf6);
                    border-radius: 6px;
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
