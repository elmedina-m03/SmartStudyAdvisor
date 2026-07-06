"""Smart Study Advisor — modern AI dashboard (home page)."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.smart_study_agent import SmartStudyAgent  # noqa: E402
from app.components.constants import (  # noqa: E402
    AGENT_CYCLE_LABEL,
    FATIGUE_LEVELS,
    FEEDBACK_OPTIONS_BS,
    FEEDBACK_TYPES,
    FEEDBACK_TYPE_LABELS_BS,
    LEVEL_LABELS_BS,
    PROJECT_AUTHORS,
    PROJECT_COURSE,
    PROJECT_NAME,
    SLEEP_QUALITY,
    SLEEP_QUALITY_LABELS_BS,
    STRESS_LEVELS,
    STRATEGY_LABELS,
)
from app.components.data_loader import dataset_overview  # noqa: E402
from app.components.metrics import (  # noqa: E402
    feedback_record_count,
    fmt_number,
    fmt_percent,
    load_evaluation_metrics,
)
from app.components.explainability import compute_shap_contributions, shap_available  # noqa: E402
from app.components.navigation import init_dashboard_page, render_sidebar_nav  # noqa: E402
from app.components.ui import (  # noqa: E402
    alert_card,
    labeled_selectbox,
    loading_state,
    render_agent_pipeline,
    render_confidence_bar,
    render_footer,
    render_strategy_hero,
    run_with_progress,
    section_title,
    stat_card_row,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def _init_agent() -> SmartStudyAgent:
    if "agent" not in st.session_state:
        with loading_state("Inicijalizacija agenta..."):
            st.session_state.agent = SmartStudyAgent()
    return st.session_state.agent


def _run_prediction(agent: SmartStudyAgent, form_data: dict):
    return agent.run_from_form(form_data)


def main() -> None:
    init_dashboard_page(
        page_title=PROJECT_NAME,
        page_icon="🎓",
        header_title=PROJECT_NAME,
        header_subtitle=f"{PROJECT_COURSE} · {PROJECT_AUTHORS} · Inteligentna preporuka strategije učenja",
        header_badge="ML agent aktivan",
    )

    agent = _init_agent()
    records = agent.feedback_store.recent_records(limit=5)
    overview = dataset_overview()
    eval_metrics = load_evaluation_metrics()
    feedback_total = feedback_record_count()

    stat_card_row(
        [
            {
                "label": "Model",
                "value": eval_metrics.best_model_label or "N/A",
                "icon": "🌲",
                "color": "#6366F1",
            },
            {
                "label": "F1 (test skup)",
                "value": fmt_percent(eval_metrics.holdout_f1),
                "icon": "🎯",
                "delta": "iz evaluation/model_comparison.csv",
                "color": "#22C55E",
            },
            {
                "label": "Ulaznih značajki",
                "value": fmt_number(overview.get("features")),
                "icon": "📐",
                "color": "#8B5CF6",
            },
            {
                "label": "Pohranjenih preporuka",
                "value": fmt_number(feedback_total),
                "icon": "📋",
                "delta": "raste svakom novom preporukom",
                "help_text": "Zapisi iz aplikacije u data/feedback/feedback.db",
                "color": "#06B6D4",
            },
        ]
    )

    st.markdown("")
    render_agent_pipeline(active_phase="Percepcija" if "last_result" not in st.session_state else "Preporuka")

    tab_input, tab_result, tab_explain, tab_feedback = st.tabs(
        ["📥 Unos podataka", "🎯 Preporuka", "🧠 Objašnjenje modela", "💬 Povratna informacija"]
    )

    # ── Tab 1: Input ──────────────────────────────────────────────
    with tab_input:
        section_title("Studentski kontekst", icon="👁️", subtitle="Faza percepcije — unesite trenutno stanje")

        with st.form("student_input_form", border=False):
            col1, col2 = st.columns(2, gap="large")

            with col1:
                st.markdown("##### 📚 Akademsko opterećenje")
                hours_studied = st.slider("⏱️ Sati učenja (dnevno)", 0.0, 12.0, 4.0, 0.5)
                number_of_subjects = st.slider("📖 Broj predmeta", 1, 10, 5)
                days_until_exam = st.slider("📅 Dana do ispita", 0, 60, 8)

            with col2:
                st.markdown("##### 🧘 Dobrobit i san")
                stress_level = labeled_selectbox("😰 Nivo stresa", STRESS_LEVELS, LEVEL_LABELS_BS, index=1)
                fatigue_level = labeled_selectbox("😴 Nivo umora", FATIGUE_LEVELS, LEVEL_LABELS_BS, index=1)
                sleep_hours = st.slider("🌙 Sati sna (prosjek)", 0.0, 12.0, 7.0, 0.5)
                sleep_quality = labeled_selectbox("✨ Kvalitet sna", SLEEP_QUALITY, SLEEP_QUALITY_LABELS_BS, index=2)
                previous_feedback = labeled_selectbox(
                    "💭 Prethodna povratna informacija",
                    FEEDBACK_TYPES,
                    FEEDBACK_TYPE_LABELS_BS,
                    index=0,
                )

            submitted = st.form_submit_button(
                "🚀 Generiši preporuku",
                type="primary",
                use_container_width=True,
            )

        if submitted:
            form_data = {
                "HoursStudied": hours_studied,
                "NumberOfSubjects": number_of_subjects,
                "DaysUntilExam": days_until_exam,
                "StressLevel": stress_level,
                "FatigueLevel": fatigue_level,
                "SleepHours": sleep_hours,
                "SleepQuality": sleep_quality,
                "PreviousFeedback": previous_feedback,
            }
            try:
                result = run_with_progress(
                    lambda: _run_prediction(agent, form_data),
                )
                st.session_state.last_result = result
                st.toast("Preporuka uspješno generisana!", icon="✅")
                st.rerun()
            except Exception as exc:
                alert_card(str(exc), variant="error", title="Greška pri predikciji")

        if "last_result" not in st.session_state:
            alert_card(
                "Popunite formu i kliknite <strong>Generiši preporuku</strong> da pokrenete "
                f"{AGENT_CYCLE_LABEL} ciklus.",
                variant="info",
                title="Kako koristiti",
            )

    # ── Tab 2: Result ─────────────────────────────────────────────
    with tab_result:
        if "last_result" not in st.session_state:
            alert_card(
                "Još nema generisane preporuke. Idite na tab <strong>Unos podataka</strong>.",
                variant="warning",
                title="Nema rezultata",
            )
        else:
            result = st.session_state.last_result
            rec = result.recommendation

            render_strategy_hero(rec.strategy, rec.confidence)
            render_confidence_bar(rec.confidence)

            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("🏷️ Strategija", STRATEGY_LABELS.get(rec.strategy, rec.strategy))
            with m2:
                st.metric("📊 Pouzdanost", f"{rec.confidence:.1%}")
            with m3:
                rid = result.feedback_record.record_id[:8] + "…" if result.feedback_record else "—"
                st.metric("🔑 ID zapisa", rid)

            col_expl, col_prob = st.columns([1, 1], gap="large")
            with col_expl:
                section_title("Objašnjenje", icon="💡")
                alert_card(rec.explanation, variant="info", title="Zašto ova strategija?")
                alert_card(rec.advice, variant="success", title="Praktični savjet")

            with col_prob:
                section_title("Vjerovatnoće klasa", icon="📊")
                prob_df = {STRATEGY_LABELS.get(k, k): v for k, v in result.prediction.probabilities.items()}
                st.bar_chart(prob_df, height=280)

            with st.expander("🔍 Detalji unosa (JSON)", expanded=False):
                st.json(result.context.to_feature_dict())

    # ── Tab 3: SHAP ───────────────────────────────────────────────
    with tab_explain:
        section_title("Objašnjenje modela", icon="🧠", subtitle="Koji feature-i utiču na ovu predikciju?")

        if "last_result" not in st.session_state:
            alert_card(
                "Generišite preporuku da vidite objašnjenje modela.",
                variant="warning",
                title="Potrebna predikcija",
            )
        elif not shap_available():
            alert_card(
                "Instalirajte paket <code>shap</code> (<code>pip install shap</code>) za objašnjenje modela.",
                variant="warning",
                title="Objašnjenje nije dostupno",
            )
        else:
            result = st.session_state.last_result
            try:
                with loading_state("Računanje doprinosa značajki..."):
                    shap_df = compute_shap_contributions(result.context, result.prediction)

                alert_card(
                    "Pozitivne vrijednosti guraju predikciju ka odabranoj klasi; "
                    "negativne je odmiču. Analiza koristi SHAP metodu i odnosi se na vaš unos.",
                    variant="info",
                    title="Kako čitati grafikon",
                )
                chart_df = shap_df.set_index("Značajka")[["Doprinos"]]
                st.bar_chart(chart_df, height=320)
                st.dataframe(
                    shap_df.style.background_gradient(subset=["Doprinos"], cmap="RdYlGn"),
                    use_container_width=True,
                    hide_index=True,
                )
            except Exception as exc:
                alert_card(str(exc), variant="error", title="Greška pri objašnjenju")

    # ── Tab 4: Feedback ───────────────────────────────────────────
    with tab_feedback:
        section_title("Povratna informacija", icon="📚", subtitle="Faza učenja — pomažete budućem ponovnom treningu")

        if "last_result" not in st.session_state or not st.session_state.last_result.feedback_record:
            alert_card(
                "Povratna informacija je dostupna nakon generisanja preporuke.",
                variant="info",
                title="Faza učenja",
            )
        else:
            result = st.session_state.last_result
            alert_card(
                "Vaša povratna informacija se pohranjuje u CSV i SQLite bazu za buduće poboljšanje modela.",
                variant="info",
                title="Zašto je važno?",
            )

            feedback_choice = st.selectbox(
                "Da li je preporuka bila korisna / tačna?",
                ["— Odaberite —", *FEEDBACK_OPTIONS_BS],
            )
            feedback_notes = st.text_area(
                "📝 Opcionalne napomene",
                placeholder="Koja bi strategija bila bolja i zašto?",
                height=100,
            )

            if st.button("📤 Pošalji povratnu informaciju", type="primary", use_container_width=True):
                if feedback_choice == "— Odaberite —":
                    alert_card("Odaberite jednu od ponuđenih opcija.", variant="warning", title="Nedostaje izbor")
                else:
                    payload = feedback_choice
                    if feedback_notes.strip():
                        payload = f"{feedback_choice}: {feedback_notes.strip()}"
                    with loading_state("Čuvanje povratne informacije..."):
                        ok = agent.submit_feedback(result.feedback_record.record_id, payload)
                    if ok:
                        st.toast("Povratna informacija sačuvana!", icon="✅")
                        alert_card(
                            "Hvala — vaša povratna informacija je sačuvana za budući ponovni trening.",
                            variant="success",
                            title="Uspjeh",
                        )
                    else:
                        alert_card("Zapis nije pronađen u bazi.", variant="error", title="Greška")

    with st.sidebar:
        render_sidebar_nav(recent_records=records)

    render_footer()


if __name__ == "__main__":
    main()
