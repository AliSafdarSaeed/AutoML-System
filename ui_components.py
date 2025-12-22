"""
ui_components.py - Reusable UI components for the AutoML app
Contains all render functions for headers, cards, alerts, badges, etc.
"""

import streamlit as st


def render_header(icon: str, title: str, subtitle: str = None):
    """Render a premium animated header."""
    st.markdown(f"""
    <div style="text-align: center; margin-bottom: 2rem; animation: fadeIn 0.8s ease-out;">
        <div style="font-size: 4rem; margin-bottom: 0.5rem;">{icon}</div>
        <h1 style="
            font-size: 2.5rem;
            font-weight: 800;
            background: linear-gradient(135deg, #fff, #a5b4fc, #818cf8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0;
        ">{title}</h1>
        {f'<p style="color: rgba(255,255,255,0.5); font-size: 1.1rem; margin-top: 0.5rem;">{subtitle}</p>' if subtitle else ''}
    </div>
    """, unsafe_allow_html=True)


def render_section_header(icon: str, title: str):
    """Render a section header with gradient underline."""
    st.markdown(f"""
    <div style="
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin: 2.5rem 0 1.5rem;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid;
        border-image: linear-gradient(90deg, #667eea, #764ba2, transparent) 1;
    ">
        <span style="font-size: 1.5rem;">{icon}</span>
        <span style="font-size: 1.4rem; font-weight: 700; color: #fff;">{title}</span>
    </div>
    """, unsafe_allow_html=True)


def render_metric_card(value, label, icon="📊"):
    """Render a glassmorphic metric card."""
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(139,92,246,0.15));
        border: 1px solid rgba(99,102,241,0.3);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
        cursor: default;
    " onmouseover="this.style.transform='scale(1.03)'; this.style.boxShadow='0 0 30px rgba(99,102,241,0.3)';"
       onmouseout="this.style.transform='scale(1)'; this.style.boxShadow='none';">
        <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">{icon}</div>
        <div style="
            font-size: 2rem;
            font-weight: 800;
            background: linear-gradient(135deg, #4facfe, #00f2fe);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        ">{value}</div>
        <div style="
            color: rgba(255,255,255,0.6);
            font-size: 0.85rem;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-top: 0.25rem;
        ">{label}</div>
    </div>
    """, unsafe_allow_html=True)


def render_status_badge(text: str, status: str = "success"):
    """Render a status badge."""
    colors = {
        "success": ("rgba(16,185,129,0.2)", "#6ee7b7", "rgba(16,185,129,0.5)"),
        "warning": ("rgba(245,158,11,0.2)", "#fcd34d", "rgba(245,158,11,0.5)"),
        "error": ("rgba(239,68,68,0.2)", "#fca5a5", "rgba(239,68,68,0.5)"),
        "info": ("rgba(99,102,241,0.2)", "#a5b4fc", "rgba(99,102,241,0.5)")
    }
    bg, text_color, border = colors.get(status, colors["info"])
    st.markdown(f"""
    <span style="
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        background: {bg};
        color: {text_color};
        border: 1px solid {border};
        border-radius: 50px;
        font-size: 0.875rem;
        font-weight: 600;
    ">{text}</span>
    """, unsafe_allow_html=True)


def render_alert(message: str, alert_type: str = "info"):
    """Render a styled alert box."""
    configs = {
        "success": ("✅", "rgba(16,185,129,0.15)", "rgba(16,185,129,0.4)", "#6ee7b7"),
        "warning": ("⚠️", "rgba(245,158,11,0.15)", "rgba(245,158,11,0.4)", "#fcd34d"),
        "error": ("❌", "rgba(239,68,68,0.15)", "rgba(239,68,68,0.4)", "#fca5a5"),
        "info": ("💡", "rgba(99,102,241,0.15)", "rgba(99,102,241,0.4)", "#a5b4fc")
    }
    icon, bg, border, color = configs.get(alert_type, configs["info"])
    st.markdown(f"""
    <div style="
        background: {bg};
        border: 1px solid {border};
        border-radius: 12px;
        padding: 1rem 1.5rem;
        color: {color};
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin: 1rem 0;
    ">
        <span style="font-size: 1.2rem;">{icon}</span>
        <span>{message}</span>
    </div>
    """, unsafe_allow_html=True)


def render_glass_card(content: str, padding: str = "1.5rem"):
    """Render content in a glass card."""
    st.markdown(f"""
    <div style="
        background: rgba(255,255,255,0.03);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: {padding};
        margin: 1rem 0;
    ">{content}</div>
    """, unsafe_allow_html=True)
