"""
components.py - Professional Minimalist UI Components

Enterprise-grade component library inspired by Linear and Vercel.
Emphasizes glass morphism, refined typography, and subtle interactions.
"""

import streamlit as st
from typing import Optional, Literal


def render_page_header(title: str, subtitle: Optional[str] = None) -> None:
    """
    Render a minimalist page header with title and optional subtitle.
    
    Args:
        title: Main page title
        subtitle: Optional description text
    """
    st.markdown(f"""
    <div class="page-header animate-fade-in">
        <h1 class="page-header-title">{title}</h1>
        {f'<p class="page-header-subtitle">{subtitle}</p>' if subtitle else ''}
    </div>
    """, unsafe_allow_html=True)


def render_section_header(title: str, count: Optional[int] = None) -> None:
    """
    Render a minimal section header with optional count badge.
    
    Args:
        title: Section title
        count: Optional count to display in badge
    """
    count_badge = f'<span class="count-badge">{count}</span>' if count is not None else ''
    
    st.markdown(f"""
    <div class="section-header">
        <span class="section-header-title">{title}</span>
        {count_badge}
    </div>
    """, unsafe_allow_html=True)


def render_metric_card(value: str, label: str, trend: Optional[str] = None) -> None:
    """
    Render a glassmorphic metric card with optional trend indicator.
    
    Args:
        value: The metric value to display
        label: Label describing the metric
        trend: Optional trend text (e.g., "↑ 12%")
    """
    trend_html = f'<div class="metric-trend">{trend}</div>' if trend else ''
    
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {trend_html}
    </div>
    """, unsafe_allow_html=True)


def render_severity_badge(
    text: str, 
    severity: Literal["critical", "warning", "success", "info"] = "info"
) -> str:
    """
    Return HTML for a refined severity badge.
    
    Args:
        text: Badge text
        severity: One of 'critical', 'warning', 'success', 'info'
    
    Returns:
        HTML string for the badge
    """
    return f'<span class="badge badge-{severity}">{text}</span>'


def render_alert_card(
    issue_name: str,
    detail: str,
    severity: Literal["critical", "warning"] = "warning",
    key_suffix: str = ""
) -> bool:
    """
    Render an issue alert card with severity badge and fix toggle.
    
    Args:
        issue_name: Name of the issue
        detail: Additional detail (e.g., "5 missing values")
        severity: 'critical' (red) or 'warning' (yellow)
        key_suffix: Unique key suffix for the checkbox
    
    Returns:
        Boolean indicating whether fix is toggled on
    """
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        st.markdown(f"""
        <div class="issue-card-left">
            <span class="issue-card-name">{issue_name}</span>
            <span class="issue-card-detail">{detail}</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        badge_html = render_severity_badge(
            "CRITICAL" if severity == "critical" else "WARNING",
            severity
        )
        st.markdown(badge_html, unsafe_allow_html=True)
    
    with col3:
        fix_enabled = st.checkbox("Fix", key=f"fix_{key_suffix}", label_visibility="collapsed")
    
    return fix_enabled


def render_alert(
    message: str, 
    alert_type: Literal["success", "warning", "error", "info"] = "info",
    title: Optional[str] = None
) -> None:
    """
    Render a refined alert message with left border accent.
    
    Args:
        message: Alert message text
        alert_type: One of 'success', 'warning', 'error', 'info'
        title: Optional alert title
    """
    # Map alert types to Streamlit alert functions
    if alert_type == "success":
        st.success(f"**{title}**\n\n{message}" if title else message, icon="✅")
    elif alert_type == "warning":
        st.warning(f"**{title}**\n\n{message}" if title else message, icon="⚠️")
    elif alert_type == "error":
        st.error(f"**{title}**\n\n{message}" if title else message, icon="❌")
    else:  # info
        st.info(f"**{title}**\n\n{message}" if title else message, icon="ℹ️")


def render_glass_card(content: str, padding: str = "1.5rem") -> None:
    """
    Render content inside a glassmorphic card container.
    
    Args:
        content: HTML content to render inside card
        padding: CSS padding value
    """
    st.markdown(f"""
    <div class="glass-card" style="padding: {padding};">
        {content}
    </div>
    """, unsafe_allow_html=True)


def render_best_model_card(
    model_name: str,
    f1_score: float,
    accuracy: float,
    training_time: float
) -> None:
    """
    Render the best model highlight card with refined styling.
    
    Args:
        model_name: Name of the best model
        f1_score: F1 score value
        accuracy: Accuracy value
        training_time: Training time in seconds
    """
    st.markdown(f"""
    <div class="best-model-card animate-fade-in">
        <div class="best-model-label">Best Performing Model</div>
        <div class="best-model-name">{model_name}</div>
        <div style="display: flex; gap: 2rem; margin-top: 1rem; color: var(--text-secondary); font-size: 13px;">
            <div><strong>F1:</strong> {f1_score:.4f}</div>
            <div><strong>Accuracy:</strong> {accuracy:.4f}</div>
            <div><strong>Time:</strong> {training_time:.2f}s</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_approval_gate(approved: bool = False) -> bool:
    """
    Render an approval gate with refined warning styling.
    
    Args:
        approved: Initial approval state
    
    Returns:
        Boolean indicating if user approved
    """
    st.markdown("""
    <div class="approval-gate">
        <div class="approval-gate-title">Review Required</div>
        <div class="approval-gate-text">
            Please review the detected issues above and apply fixes before proceeding to training.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    return st.checkbox("I have reviewed and approved the preprocessing configuration", value=approved)


def render_drop_zone() -> None:
    """
    Render a minimal drop zone for file uploads.
    """
    st.markdown("""
    <div style="
        text-align: center;
        padding: 3rem;
        background: var(--bg-elevated);
        border: 2px dashed var(--border-default);
        border-radius: var(--radius-lg);
        transition: all var(--transition-base);
    ">
        <div style="font-size: 16px; font-weight: 600; color: var(--text-primary); margin-bottom: 0.5rem;">
            Import Dataset
        </div>
        <div style="font-size: 13px; color: var(--text-muted);">
            Drop your CSV file here or click to browse
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_status_dot(completed: bool = False) -> str:
    """
    Return HTML for a status dot indicator.
    
    Args:
        completed: Whether the status is completed
    
    Returns:
        HTML string for the status dot
    """
    css_class = "status-dot" if completed else "status-dot status-dot--pending"
    return f'<span class="{css_class}"></span>'


def render_proceed_button(
    next_page: str,
    label: str = "Proceed to Next Step",
    disabled: bool = False,
    show_icon: bool = True
) -> bool:
    """
    Render a prominent 'Proceed to Next Step' button.
    
    Args:
        next_page: The page name to navigate to
        label: Button label text
        disabled: Whether button is disabled
        show_icon: Whether to show arrow icon
    
    Returns:
        Boolean indicating if button was clicked
    """
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Divider before button
    st.markdown("""
    <div style="
        border-top: 1px solid var(--border-default);
        margin: 1.5rem 0;
        opacity: 0.5;
    "></div>
    """, unsafe_allow_html=True)
    
    # Create columns to center the button
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        button_label = f"{label} →" if show_icon else label
        clicked = st.button(
            button_label,
            key=f"proceed_to_{next_page.lower()}",
            type="primary",
            use_container_width=True,
            disabled=disabled
        )
        
        if clicked and not disabled:
            st.session_state.current_page = next_page
            st.rerun()
    
    return clicked