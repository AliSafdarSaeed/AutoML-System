"""
sidebar.py - Sidebar navigation and status display
Handles the sidebar UI including navigation buttons and project status
"""

import streamlit as st


def render_sidebar():
    """Render the sidebar with navigation and status."""
    with st.sidebar:
        # Logo/Brand
        st.markdown("""
        <div style="text-align: center; padding: 1.5rem 0 2rem;">
            <div style="font-size: 3rem; margin-bottom: 0.5rem;">🚀</div>
            <div style="
                font-size: 1.5rem;
                font-weight: 800;
                background: linear-gradient(135deg, #667eea, #764ba2);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            ">AutoML Pro</div>
            <div style="color: rgba(255,255,255,0.4); font-size: 0.8rem; margin-top: 0.25rem;">
                ML Classification System
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div style="height: 1px; background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent); margin: 0 1rem 1.5rem;"></div>', unsafe_allow_html=True)
        
        # Navigation
        pages = [
            ("📤", "Upload & EDA", "upload", st.session_state.df is not None),
            ("🔧", "Preprocessing", "preprocess", st.session_state.df_clean is not None),
            ("🎯", "Training", "train", st.session_state.results is not None),
            ("📊", "Report", "report", st.session_state.results is not None)
        ]
        
        for icon, name, key, completed in pages:
            is_current = st.session_state.current_page == name
            disabled = (key == "preprocess" and st.session_state.df is None) or \
                      (key == "train" and st.session_state.df_clean is None) or \
                      (key == "report" and st.session_state.results is None)
            
            status_icon = "✓" if completed else ""
            
            if st.button(f"{icon}  {name} {status_icon}", key=f"nav_{key}", disabled=disabled, width='stretch'):
                st.session_state.current_page = name
                st.rerun()
        
        st.markdown('<div style="height: 1px; background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent); margin: 2rem 1rem 1.5rem;"></div>', unsafe_allow_html=True)
        
        # Status Panel
        st.markdown("""
        <div style="color: rgba(255,255,255,0.5); font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.75rem; padding: 0 0.5rem;">
            📊 Project Status
        </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.df is not None:
            st.markdown(f"""
            <div style="
                background: rgba(16,185,129,0.15);
                border: 1px solid rgba(16,185,129,0.3);
                border-radius: 12px;
                padding: 1rem;
                margin-bottom: 0.75rem;
            ">
                <div style="color: #6ee7b7; font-weight: 600; font-size: 0.9rem;">📁 {st.session_state.file_name}</div>
                <div style="color: rgba(255,255,255,0.5); font-size: 0.8rem; margin-top: 0.25rem;">
                    {st.session_state.df.shape[0]:,} rows × {st.session_state.df.shape[1]} cols
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            steps_done = sum([
                st.session_state.df is not None,
                st.session_state.df_clean is not None,
                st.session_state.results is not None
            ])
            
            st.markdown(f"""
            <div style="margin-top: 1rem;">
                <div style="display: flex; justify-content: space-between; color: rgba(255,255,255,0.5); font-size: 0.75rem; margin-bottom: 0.5rem;">
                    <span>Progress</span>
                    <span>{steps_done}/3 complete</span>
                </div>
                <div style="background: rgba(255,255,255,0.1); border-radius: 10px; height: 6px; overflow: hidden;">
                    <div style="background: linear-gradient(90deg, #667eea, #764ba2); width: {steps_done/3*100}%; height: 100%; border-radius: 10px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="
                background: rgba(245,158,11,0.15);
                border: 1px solid rgba(245,158,11,0.3);
                border-radius: 12px;
                padding: 1rem;
                color: #fcd34d;
                font-size: 0.9rem;
            ">
                ⏳ Waiting for data upload...
            </div>
            """, unsafe_allow_html=True)
