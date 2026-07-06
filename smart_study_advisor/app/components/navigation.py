"""Modern sidebar navigation and layout shell."""

from __future__ import annotations

import streamlit as st

from app.components.metrics import fmt_percent, load_evaluation_metrics, models_available
from app.components.ui import (
    render_recent_prediction_card,
    render_sidebar_brand,
    render_sidebar_status,
)


def render_sidebar_nav(*, recent_records: list[dict] | None = None) -> None:
    render_sidebar_brand()

    eval_metrics = load_evaluation_metrics()
    f1_display = fmt_percent(eval_metrics.holdout_f1) if eval_metrics.holdout_f1 is not None else "N/A"
    model_name = eval_metrics.best_model_label or "N/A"
    render_sidebar_status(model=model_name, f1=f1_display)

    st.markdown('<p class="sidebar-nav-label">Navigacija</p>', unsafe_allow_html=True)
    st.page_link("streamlit_app.py", label="Početna", icon="🏠")
    st.page_link("pages/1_EDA.py", label="EDA analiza", icon="📊")
    st.page_link("pages/2_Model_Evaluation.py", label="Evaluacija", icon="📈")
    st.page_link("pages/3_About.py", label="O projektu", icon="ℹ️")

    st.divider()
    st.markdown('<p class="sidebar-nav-label">ML model</p>', unsafe_allow_html=True)
    if models_available():
        st.markdown(
            f"""
            <div style="font-size:0.8rem; color:#94A3B8; line-height:1.6;">
            🌲 {model_name}<br>
            📦 preprocessor.pkl<br>
            🏷️ label_encoder.pkl
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.caption("Model artefakti nisu dostupni.")

    if recent_records:
        st.divider()
        st.markdown('<p class="sidebar-nav-label">Nedavne preporuke</p>', unsafe_allow_html=True)
        for row in recent_records:
            render_recent_prediction_card(
                row["predicted_strategy"],
                float(row["confidence"]),
                row["timestamp"],
            )


def init_dashboard_page(
    *,
    page_title: str,
    page_icon: str,
    header_title: str,
    header_subtitle: str,
    header_badge: str = "AI Agent",
) -> None:
    """Call at top of each page: config + styles + header."""
    from app.components.styles import apply_page_styles
    from app.components.ui import render_app_header

    st.set_page_config(
        page_title=page_title,
        page_icon=page_icon,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_page_styles()
    render_app_header(header_title, header_subtitle, badge=header_badge, icon=page_icon)
