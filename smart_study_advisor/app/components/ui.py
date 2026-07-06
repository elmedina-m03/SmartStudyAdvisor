"""Reusable dashboard UI components — cards, alerts, header, footer."""

from __future__ import annotations

import html
import time
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

import streamlit as st

from app.components.constants import (
    AGENT_CYCLE_LABEL,
    PROJECT_NAME,
    AGENT_PHASES,
    PROJECT_AUTHORS,
    PROJECT_COURSE,
    STRATEGY_COLORS,
    STRATEGY_LABELS,
)


def labeled_selectbox(
    label: str,
    values: list[str],
    label_map: dict[str, str],
    *,
    index: int = 0,
    **kwargs: object,
) -> str:
    """Selectbox with Bosnian labels; returns internal English value for the ML model."""
    display_to_value = {label_map[v]: v for v in values}
    display_options = [label_map[v] for v in values]
    selected = st.selectbox(label, display_options, index=index, **kwargs)
    return display_to_value[str(selected)]


def render_app_header(
    title: str,
    subtitle: str,
    *,
    badge: str = "AI Agent",
    icon: str = "🎓",
) -> None:
    st.markdown(
        f"""
        <div class="app-header">
            <div class="app-header-inner">
                <div class="app-header-icon">{icon}</div>
                <div class="app-header-text">
                    <div class="app-header-badge">{badge}</div>
                    <h1 class="app-header-title">{html.escape(title)}</h1>
                    <p class="app-header-subtitle">{html.escape(subtitle)}</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    st.markdown(
        f"""
        <div class="app-footer">
            <div class="app-footer-inner">
                <span class="footer-brand">{PROJECT_NAME}</span>
                <span class="footer-sep">·</span>
                <span>{PROJECT_COURSE}</span>
                <span class="footer-sep">·</span>
                <span>Autori: {PROJECT_AUTHORS}</span>
                <span class="footer-sep">·</span>
                <span>{AGENT_CYCLE_LABEL}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def stat_card(
    label: str,
    value: str,
    *,
    icon: str = "📊",
    delta: str | None = None,
    color: str = "#6366F1",
    help_text: str | None = None,
    multiline: bool = False,
) -> None:
    delta_html = ""
    if delta:
        delta_html = f'<div class="stat-delta">{html.escape(delta)}</div>'
    help_attr = f'title="{html.escape(help_text)}"' if help_text else ""
    value_class = "stat-value stat-value-multiline" if multiline else "stat-value"
    if multiline:
        parts = value.replace("<br>", "\n").split("\n")
        safe_value = "<br>".join(html.escape(part.strip()) for part in parts if part.strip())
    else:
        safe_value = html.escape(value)
    st.markdown(
        f"""
        <div class="stat-card" style="--accent: {color};" {help_attr}>
            <div class="stat-card-top">
                <span class="stat-icon">{icon}</span>
            </div>
            <div class="{value_class}">{safe_value}</div>
            <div class="stat-label">{html.escape(label)}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def stat_card_row(cards: list[dict[str, Any]], *, columns: int | None = None) -> None:
    n = columns or len(cards)
    cols = st.columns(n)
    for col, card in zip(cols, cards, strict=False):
        with col:
            stat_card(**card)


def alert_card(message: str, *, variant: str = "info", title: str | None = None, icon: str | None = None) -> None:
    icons = {
        "info": "ℹ️",
        "success": "✅",
        "warning": "⚠️",
        "error": "❌",
    }
    display_icon = icon or icons.get(variant, "ℹ️")
    title_html = f'<div class="alert-title">{html.escape(title)}</div>' if title else ""
    st.markdown(
        f"""
        <div class="alert-card alert-{variant}">
            <span class="alert-icon">{display_icon}</span>
            <div class="alert-body">
                {title_html}
                <div class="alert-message">{message}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_title(title: str, *, icon: str = "", subtitle: str | None = None) -> None:
    sub = f'<span class="section-subtitle">{html.escape(subtitle)}</span>' if subtitle else ""
    st.markdown(
        f"""
        <div class="section-title-wrap">
            <h2 class="section-title">{icon} {html.escape(title)}</h2>
            {sub}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_confidence_bar(confidence: float, *, label: str = "Pouzdanost modela") -> None:
    pct = int(confidence * 100)
    color = "#22C55E" if confidence >= 0.8 else "#F59E0B" if confidence >= 0.6 else "#EF4444"
    st.markdown(
        f"""
        <div class="confidence-wrap">
            <div class="confidence-header">
                <span>{html.escape(label)}</span>
                <span class="confidence-pct" style="color:{color};">{pct}%</span>
            </div>
            <div class="confidence-track">
                <div class="confidence-fill" style="width:{pct}%; background:{color};"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(confidence)


def render_agent_pipeline(*, active_phase: str | None = None) -> None:
    phase_icons = {"Percepcija": "👁️", "Predikcija": "🧠", "Preporuka": "⚡", "Učenje": "📚"}
    cols = st.columns(len(AGENT_PHASES))
    for col, (name, desc, color) in zip(cols, AGENT_PHASES, strict=True):
        active = name == active_phase
        active_class = " agent-phase-active" if active else ""
        icon = phase_icons.get(name, "•")
        with col:
            st.markdown(
                f"""
                <div class="agent-phase{active_class}" style="--phase-color: {color};">
                    <div class="agent-phase-icon">{icon}</div>
                    <div class="agent-phase-title">{name}</div>
                    <div class="agent-phase-desc">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def strategy_badge_html(strategy: str) -> str:
    color = STRATEGY_COLORS.get(strategy, "#64748B")
    label = STRATEGY_LABELS.get(strategy, strategy)
    return (
        f'<span class="strategy-badge" style="--badge-color:{color};">'
        f"{html.escape(label)}</span>"
    )


def render_strategy_hero(strategy: str, confidence: float) -> None:
    label = STRATEGY_LABELS.get(strategy, strategy)
    color = STRATEGY_COLORS.get(strategy, "#6366F1")
    st.markdown(
        f"""
        <div class="strategy-hero" style="--hero-color: {color};">
            <div class="strategy-hero-icon">🎯</div>
            <div>
                <div class="strategy-hero-label">Preporučena strategija</div>
                <div class="strategy-hero-name">{html.escape(label)}</div>
                <div class="strategy-hero-code">{html.escape(label)} · {confidence:.1%} pouzdanost</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@contextmanager
def loading_state(message: str = "Učitavanje..."):
    with st.spinner(message):
        yield


def run_with_progress(fn: Callable[[], Any], *, steps: list[str] | None = None) -> Any:
    """Run a callable with a stepped progress bar. Returns fn() result."""
    labels = steps or [
        "Percepcija — validacija unosa...",
        "Predikcija — ML model...",
        "Preporuka — generisanje savjeta...",
        "Učenje — pohrana zapisa...",
    ]
    progress = st.progress(0.0, text=labels[0])
    result = None
    total = len(labels)
    for idx, label in enumerate(labels):
        progress.progress(idx / total, text=label)
        if idx == 1:
            result = fn()
        elif idx != 1:
            time.sleep(0.12)
    progress.progress(1.0, text="Završeno ✓")
    time.sleep(0.15)
    progress.empty()
    return result


def render_sidebar_brand() -> None:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-logo">🎓</div>
            <div>
                <div class="sidebar-title">Smart Study</div>
                <div class="sidebar-subtitle">Savjetnik</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_status(*, model: str = "N/A", f1: str = "N/A") -> None:
    st.markdown(
        f"""
        <div class="sidebar-status">
            <div class="status-dot"></div>
            <div>
                <div class="status-label">Model aktivan</div>
                <div class="status-value">{html.escape(model)} · F1 {html.escape(f1)}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_recent_prediction_card(strategy: str, confidence: float, timestamp: str) -> None:
    color = STRATEGY_COLORS.get(strategy, "#64748B")
    label = STRATEGY_LABELS.get(strategy, strategy)
    st.markdown(
        f"""
        <div class="recent-card" style="--card-accent: {color};">
            <div class="recent-strategy">{html.escape(label)}</div>
            <div class="recent-meta">{confidence:.0%} · {html.escape(timestamp[:16])}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
