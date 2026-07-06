"""Modern dashboard CSS and layout helpers."""

from __future__ import annotations

import streamlit as st

from app.components.constants import STRATEGY_COLORS


def apply_page_styles() -> None:
    strategy_css = "\n".join(
        f".strategy-tag-{name} {{ background: {color}18; color: {color}; border: 1px solid {color}40; }}"
        for name, color in STRATEGY_COLORS.items()
    )
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }}

        /* ── Layout ── */
        .block-container {{
            padding-top: 1rem;
            padding-bottom: 0;
            max-width: 1280px;
        }}
        #MainMenu {{ visibility: hidden; }}
        footer {{ visibility: hidden; }}
        header[data-testid="stHeader"] {{
            background: transparent;
        }}

        /* ── Header ── */
        .app-header {{
            background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 50%, #6366F1 100%);
            border-radius: 16px;
            padding: 1.75rem 2rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 10px 40px rgba(99, 102, 241, 0.25);
        }}
        .app-header-inner {{
            display: flex;
            align-items: center;
            gap: 1.25rem;
        }}
        .app-header-icon {{
            font-size: 2.75rem;
            background: rgba(255,255,255,0.15);
            border-radius: 14px;
            width: 64px;
            height: 64px;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
        }}
        .app-header-badge {{
            display: inline-block;
            background: rgba(255,255,255,0.2);
            color: #E0E7FF;
            font-size: 0.7rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            padding: 0.2rem 0.65rem;
            border-radius: 999px;
            margin-bottom: 0.4rem;
        }}
        .app-header-title {{
            color: #FFFFFF;
            font-size: 1.75rem;
            font-weight: 800;
            margin: 0;
            line-height: 1.2;
        }}
        .app-header-subtitle {{
            color: rgba(255,255,255,0.85);
            font-size: 0.95rem;
            margin: 0.35rem 0 0 0;
        }}

        /* ── Footer ── */
        .app-footer {{
            margin-top: 2.5rem;
            padding: 1.25rem 0;
            border-top: 1px solid #E2E8F0;
        }}
        .app-footer-inner {{
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            justify-content: center;
            gap: 0.35rem 0.5rem;
            color: #94A3B8;
            font-size: 0.8rem;
        }}
        .footer-brand {{
            font-weight: 700;
            color: #6366F1;
        }}
        .footer-sep {{ color: #CBD5E1; }}

        /* ── Stat cards ── */
        .stat-card {{
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 14px;
            padding: 1.1rem 1.25rem;
            box-shadow: 0 1px 3px rgba(15,23,42,0.04);
            border-top: 3px solid var(--accent, #6366F1);
            height: 100%;
            transition: box-shadow 0.2s, transform 0.2s;
        }}
        .stat-card:hover {{
            box-shadow: 0 8px 24px rgba(15,23,42,0.08);
            transform: translateY(-2px);
        }}
        .stat-icon {{
            font-size: 1.35rem;
            background: color-mix(in srgb, var(--accent) 12%, white);
            width: 40px;
            height: 40px;
            border-radius: 10px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
        }}
        .stat-value {{
            font-size: 1.65rem;
            font-weight: 800;
            color: #0F172A;
            margin: 0.6rem 0 0.15rem;
            line-height: 1.25;
            word-break: break-word;
        }}
        .stat-value-multiline {{
            font-size: 0.95rem;
            font-weight: 700;
            line-height: 1.45;
            white-space: normal;
        }}
        .stat-label {{
            font-size: 0.8rem;
            color: #64748B;
            font-weight: 500;
            line-height: 1.35;
        }}
        .stat-delta {{
            font-size: 0.75rem;
            color: #22C55E;
            margin-top: 0.35rem;
            font-weight: 600;
            line-height: 1.4;
            word-break: break-word;
        }}

        /* ── Alert cards ── */
        .alert-card {{
            display: flex;
            gap: 0.85rem;
            padding: 1rem 1.15rem;
            border-radius: 12px;
            margin: 0.75rem 0;
            border: 1px solid transparent;
        }}
        .alert-icon {{ font-size: 1.25rem; flex-shrink: 0; }}
        .alert-title {{
            font-weight: 700;
            font-size: 0.9rem;
            margin-bottom: 0.25rem;
        }}
        .alert-message {{
            font-size: 0.9rem;
            line-height: 1.55;
        }}
        .alert-info {{
            background: #EFF6FF;
            border-color: #BFDBFE;
            color: #1E40AF;
        }}
        .alert-success {{
            background: #F0FDF4;
            border-color: #BBF7D0;
            color: #166534;
        }}
        .alert-warning {{
            background: #FFFBEB;
            border-color: #FDE68A;
            color: #92400E;
        }}
        .alert-error {{
            background: #FEF2F2;
            border-color: #FECACA;
            color: #991B1B;
        }}

        /* ── Section titles ── */
        .section-title-wrap {{ margin: 1.25rem 0 1rem; }}
        .section-title {{
            font-size: 1.15rem;
            font-weight: 700;
            color: #0F172A;
            margin: 0;
        }}
        .section-subtitle {{
            display: block;
            font-size: 0.85rem;
            color: #64748B;
            margin-top: 0.25rem;
        }}

        /* ── Agent pipeline ── */
        .agent-phase {{
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: 1rem 0.75rem;
            text-align: center;
            height: 100%;
            transition: all 0.2s;
        }}
        .agent-phase-active {{
            border-color: var(--phase-color);
            box-shadow: 0 0 0 3px color-mix(in srgb, var(--phase-color) 20%, transparent);
            background: color-mix(in srgb, var(--phase-color) 5%, white);
        }}
        .agent-phase-icon {{ font-size: 1.5rem; margin-bottom: 0.35rem; }}
        .agent-phase-title {{
            font-weight: 700;
            font-size: 0.9rem;
            color: var(--phase-color, #6366F1);
        }}
        .agent-phase-desc {{
            font-size: 0.72rem;
            color: #64748B;
            margin-top: 0.2rem;
            line-height: 1.35;
        }}

        /* ── Strategy hero ── */
        .strategy-hero {{
            display: flex;
            align-items: center;
            gap: 1.25rem;
            background: linear-gradient(135deg, color-mix(in srgb, var(--hero-color) 8%, white), white);
            border: 1px solid color-mix(in srgb, var(--hero-color) 25%, #E2E8F0);
            border-left: 5px solid var(--hero-color);
            border-radius: 14px;
            padding: 1.35rem 1.5rem;
            margin-bottom: 1rem;
        }}
        .strategy-hero-icon {{ font-size: 2.25rem; }}
        .strategy-hero-label {{
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: #64748B;
            font-weight: 600;
        }}
        .strategy-hero-name {{
            font-size: 1.65rem;
            font-weight: 800;
            color: #0F172A;
        }}
        .strategy-hero-code {{
            font-size: 0.85rem;
            color: #64748B;
            margin-top: 0.15rem;
        }}
        .strategy-badge {{
            display: inline-block;
            background: color-mix(in srgb, var(--badge-color) 15%, white);
            color: var(--badge-color);
            padding: 0.35rem 0.9rem;
            border-radius: 999px;
            font-weight: 700;
            font-size: 0.85rem;
        }}

        /* ── Confidence bar ── */
        .confidence-wrap {{ margin: 0.5rem 0 0.25rem; }}
        .confidence-header {{
            display: flex;
            justify-content: space-between;
            font-size: 0.85rem;
            font-weight: 600;
            color: #475569;
            margin-bottom: 0.4rem;
        }}
        .confidence-pct {{ font-weight: 800; }}
        .confidence-track {{
            height: 8px;
            background: #E2E8F0;
            border-radius: 999px;
            overflow: hidden;
            margin-bottom: 0.5rem;
        }}
        .confidence-fill {{
            height: 100%;
            border-radius: 999px;
            transition: width 0.6s ease;
        }}

        /* ── Sidebar ── */
        [data-testid="stSidebarNav"] {{
            display: none;
        }}
        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #0F172A 0%, #1E293B 100%);
        }}
        section[data-testid="stSidebar"] .stMarkdown,
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] p {{
            color: #CBD5E1 !important;
        }}
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {{
            color: #F1F5F9 !important;
        }}
        section[data-testid="stSidebar"] a {{
            color: #E2E8F0 !important;
        }}
        section[data-testid="stSidebar"] hr {{
            border-color: #334155 !important;
        }}
        .sidebar-brand {{
            display: flex;
            align-items: center;
            gap: 0.75rem;
            padding: 0.5rem 0 1.25rem;
            border-bottom: 1px solid #334155;
            margin-bottom: 1rem;
        }}
        .sidebar-logo {{
            font-size: 1.75rem;
            background: linear-gradient(135deg, #6366F1, #8B5CF6);
            width: 44px;
            height: 44px;
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        .sidebar-title {{
            font-weight: 800;
            font-size: 1rem;
            color: #F8FAFC !important;
        }}
        .sidebar-subtitle {{
            font-size: 0.75rem;
            color: #94A3B8 !important;
        }}
        .sidebar-nav-label {{
            font-size: 0.65rem;
            font-weight: 700;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            color: #64748B !important;
            margin: 1rem 0 0.5rem;
        }}
        .sidebar-status {{
            display: flex;
            align-items: center;
            gap: 0.65rem;
            background: rgba(34, 197, 94, 0.1);
            border: 1px solid rgba(34, 197, 94, 0.25);
            border-radius: 10px;
            padding: 0.75rem;
            margin: 1rem 0;
        }}
        .status-dot {{
            width: 8px;
            height: 8px;
            background: #22C55E;
            border-radius: 50%;
            box-shadow: 0 0 8px #22C55E;
            flex-shrink: 0;
        }}
        .status-label {{
            font-size: 0.7rem;
            color: #86EFAC !important;
            font-weight: 600;
        }}
        .status-value {{
            font-size: 0.8rem;
            color: #F1F5F9 !important;
            font-weight: 600;
        }}
        .recent-card {{
            background: rgba(255,255,255,0.05);
            border-left: 3px solid var(--card-accent);
            border-radius: 0 8px 8px 0;
            padding: 0.55rem 0.75rem;
            margin-bottom: 0.45rem;
        }}
        .recent-strategy {{
            font-size: 0.8rem;
            font-weight: 600;
            color: #F1F5F9 !important;
        }}
        .recent-meta {{
            font-size: 0.7rem;
            color: #94A3B8 !important;
        }}

        /* ── Tabs ── */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 0.35rem;
            background: #F1F5F9;
            border-radius: 12px;
            padding: 0.35rem;
        }}
        .stTabs [data-baseweb="tab"] {{
            border-radius: 8px;
            font-weight: 600;
            font-size: 0.85rem;
            padding: 0.5rem 1rem;
        }}
        .stTabs [aria-selected="true"] {{
            background: #FFFFFF !important;
            box-shadow: 0 1px 4px rgba(15,23,42,0.08);
        }}

        /* ── Forms & buttons ── */
        div[data-testid="stForm"] {{
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 14px;
            padding: 1.25rem;
            box-shadow: 0 1px 3px rgba(15,23,42,0.04);
        }}
        .stButton > button[kind="primary"] {{
            background: linear-gradient(135deg, #6366F1, #7C3AED) !important;
            border: none !important;
            border-radius: 10px !important;
            font-weight: 700 !important;
            padding: 0.6rem 1.5rem !important;
            box-shadow: 0 4px 14px rgba(99,102,241,0.35) !important;
        }}
        .stButton > button[kind="primary"]:hover {{
            box-shadow: 0 6px 20px rgba(99,102,241,0.45) !important;
        }}

        /* ── Content panel ── */
        .content-panel {{
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 14px;
            padding: 1.25rem;
            margin-bottom: 1rem;
        }}

        /* ── Responsive ── */
        @media (max-width: 768px) {{
            .app-header {{ padding: 1.25rem; }}
            .app-header-title {{ font-size: 1.35rem; }}
            .app-header-icon {{ width: 48px; height: 48px; font-size: 2rem; }}
            .strategy-hero {{ flex-direction: column; text-align: center; }}
            .app-footer-inner {{ flex-direction: column; text-align: center; }}
        }}

        {strategy_css}
        </style>
        """,
        unsafe_allow_html=True,
    )


# Backward-compatible aliases
def render_page_header(title: str, subtitle: str) -> None:
    from app.components.ui import render_app_header

    render_app_header(title, subtitle)


def render_agent_cycle() -> None:
    from app.components.ui import render_agent_pipeline

    render_agent_pipeline()


def strategy_badge(strategy: str) -> str:
    from app.components.ui import strategy_badge_html

    return strategy_badge_html(strategy)
