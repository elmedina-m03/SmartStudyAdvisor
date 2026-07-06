"""About page — methodology, pipeline, limitations."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

APP_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.components.constants import AGENT_CYCLE_LABEL, PROJECT_AUTHORS, PROJECT_COURSE, PROJECT_NAME  # noqa: E402
from app.components.dataset_cards import dataset_records_card  # noqa: E402
from app.components.data_loader import dataset_overview  # noqa: E402
from app.components.metrics import (  # noqa: E402
    ML_PIPELINE_STEPS,
    eda_figure_count,
    fmt_number,
    fmt_percent,
    load_evaluation_metrics,
)
from app.components.navigation import init_dashboard_page, render_sidebar_nav  # noqa: E402
from app.components.paths import FINAL_REPORT  # noqa: E402
from app.components.ui import alert_card, render_agent_pipeline, render_footer, section_title, stat_card_row  # noqa: E402

init_dashboard_page(
    page_title=f"O projektu | {PROJECT_NAME}",
    page_icon="ℹ️",
    header_title="O projektu",
    header_subtitle=f"{PROJECT_NAME} · {PROJECT_COURSE} · {PROJECT_AUTHORS}",
    header_badge="Dokumentacija",
)

overview = dataset_overview()
eval_metrics = load_evaluation_metrics()
figure_count = eda_figure_count()
authors_display = "<br>".join(name.strip() for name in PROJECT_AUTHORS.split(","))

stat_card_row(
    [
        {
            "label": "Autori",
            "value": authors_display,
            "icon": "👩‍🎓",
            "multiline": True,
            "color": "#6366F1",
        },
        dataset_records_card(),
        {
            "label": "Koraka pipeline-a",
            "value": str(ML_PIPELINE_STEPS),
            "icon": "⚙️",
            "color": "#06B6D4",
        },
        {
            "label": "Ciljnih klasa",
            "value": fmt_number(overview.get("classes")),
            "icon": "🎯",
            "color": "#22C55E",
        },
    ]
)

tab_overview, tab_pipeline, tab_limits, tab_docs = st.tabs(
    ["📋 Pregled", "⚙️ ML pipeline", "⚠️ Ograničenja", "📄 Dokumentacija"]
)

with tab_overview:
    section_title("Cilj projekta", icon="🎯")
    alert_card(
        "Inteligentni agent koji studentima preporučuje strategiju učenja "
        "(<em>Rest</em>, <em>BalancedStudy</em>, <em>IntensiveStudy</em>, <em>LongTermPlan</em>) "
        "na osnovu akademskog opterećenja, stresa, umora i sna.",
        variant="info",
        title=PROJECT_NAME,
    )

    section_title("Agent arhitektura", icon="🔄", subtitle=AGENT_CYCLE_LABEL)
    render_agent_pipeline()

    c1, c2 = st.columns(2, gap="large")
    with c1:
        alert_card(
            "<strong>Percepcija</strong> — validacija studentskog unosa<br>"
            "<strong>Predikcija</strong> — ML predikcija strategije<br>"
            "<strong>Preporuka</strong> — objašnjenje i savjeti<br>"
            "<strong>Učenje</strong> — pohrana u CSV i SQLite",
            variant="success",
            title="Komponente agenta",
        )
    with c2:
        f1_text = fmt_percent(eval_metrics.holdout_f1)
        model_text = eval_metrics.best_model_label or "N/A"
        alert_card(
            f"{model_text} · F1 {f1_text}<br>"
            "StandardScaler + OneHotEncoder<br>"
            "Objašnjenje modela po predikciji",
            variant="info",
            title="ML komponente",
        )

with tab_pipeline:
    section_title("ML pipeline", icon="⚙️", subtitle="Offline workflow u ml/ folderu")
    records_text = fmt_number(overview.get("records"))
    figures_text = fmt_number(figure_count)
    pipeline = [
        ("1–2", "📐 Schema i dataset", f"{records_text} zapisa, YAML validacija"),
        ("3", "📊 EDA", f"{figures_text} grafikona, deskriptivna statistika"),
        ("4", "🔧 Preprocessing", "Scaler + OneHot, stratificirani 70/30 split"),
        ("5", "🌲 Trening", f"{fmt_number(eval_metrics.model_count)} modela, najbolji: {eval_metrics.best_model_label or 'N/A'}"),
        ("6", "📈 Važnost feature-a", "Gini analiza"),
        ("7", "🔬 Studija uklanjanja", "Marginalni doprinos feature-a"),
        ("8", "🔄 Unakrsna validacija", "5-fold stratificirana CV"),
    ]
    for step, title, detail in pipeline:
        st.markdown(
            f"""
            <div class="stat-card" style="--accent: #6366F1; margin-bottom: 0.5rem;">
                <div class="stat-label">Korak {step}</div>
                <div class="stat-value" style="font-size:1.1rem;">{title}</div>
                <div class="stat-delta" style="color:#64748B;">{detail}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

with tab_limits:
    section_title("Metodologija i ograničenja", icon="⚠️")
    alert_card(
        "Podaci su generisani rule-based skriptom. Visoki skorovi odražavaju "
        "jasno definisana pravila, a ne nužno ponašanje pravih studenata.",
        variant="warning",
        title="Sintetički dataset",
    )
    record_note = fmt_number(overview.get("records"))
    for item in [
        f"Mali uzorak ({record_note} zapisa iz CSV) — diskretni koraci u metrikama.",
        "Nema podešavanja hiperparametara — sklearn zadane vrijednosti.",
        "Faza učenja pohranjuje povratnu informaciju; automatski ponovni trening nije implementiran.",
        "Objašnjenja koriste SHAP metodu i tekstualne šablone u fazi preporuke.",
    ]:
        st.markdown(f"- {item}")

with tab_docs:
    section_title("Dokumentacija", icon="📄")
    if FINAL_REPORT.exists():
        with st.expander("📖 Finalni ML izvještaj (uvod)", expanded=False):
            content = FINAL_REPORT.read_text(encoding="utf-8")
            st.markdown(content[:4000] + "\n\n*… puni izvještaj: `reports/final_ml_report.md`*")
    else:
        alert_card("Finalni ML izvještaj nije pronađen.", variant="warning")

    st.code(
        "cd smart_study_advisor\npip install -r requirements.txt\nstreamlit run app/streamlit_app.py",
        language="bash",
    )

with st.sidebar:
    render_sidebar_nav()

render_footer()
