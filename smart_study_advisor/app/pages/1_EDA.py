"""EDA dashboard — descriptive statistics and visualizations."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

APP_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.components.constants import PROJECT_AUTHORS, PROJECT_COURSE, PROJECT_NAME  # noqa: E402
from app.components.dataset_cards import dataset_records_card  # noqa: E402
from app.components.data_loader import (  # noqa: E402
    categorical_summary,
    class_balance,
    dataset_overview,
    iqr_outlier_count,
    styled_descriptive_statistics,
    styled_strategy_means,
    strategy_means,
)
from app.components.metrics import eda_figure_count, fmt_number  # noqa: E402
from app.components.navigation import init_dashboard_page, render_sidebar_nav  # noqa: E402
from app.components.paths import (  # noqa: E402
    CORRELATION_PAIRS_ENCODED,
    CORRELATION_PAIRS_NUMERIC,
    FIGURES_DIR,
    REDUNDANCY_REPORT,
)
from app.components.ui import alert_card, render_footer, section_title, stat_card_row  # noqa: E402

init_dashboard_page(
    page_title=f"EDA | {PROJECT_NAME}",
    page_icon="📊",
    header_title="Eksploratorna analiza podataka",
    header_subtitle=f"Deskriptivna statistika · vizualizacije · korelacije | {PROJECT_COURSE}",
    header_badge="EDA analiza",
)

overview = dataset_overview()
class_pct = None
if overview.get("records") and overview.get("classes"):
    class_pct = round(100 / overview["classes"], 1)


@st.cache_data(show_spinner=False)
def _load_correlation_pairs(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError, ValueError):
        return None

stat_card_row(
    [
        dataset_records_card(),
        {
            "label": "Ulaznih značajki",
            "value": fmt_number(overview.get("features")),
            "icon": "📐",
            "color": "#8B5CF6",
        },
        {
            "label": "Numeričkih",
            "value": fmt_number(overview.get("numeric")),
            "icon": "🔢",
            "color": "#06B6D4",
        },
        {
            "label": "Nedostajućih",
            "value": fmt_number(overview.get("missing")),
            "icon": "✅",
            "delta": f"{fmt_number(overview.get('duplicates'))} duplikata",
            "color": "#22C55E",
        },
        {
            "label": "Ciljnih klasa",
            "value": fmt_number(overview.get("classes")),
            "icon": "🎯",
            "delta": f"~{class_pct}% po klasi" if class_pct is not None else None,
            "color": "#F59E0B",
        },
    ],
    columns=5,
)

tab_stats, tab_strategy, tab_figures, tab_corr = st.tabs(
    ["📈 Statistika", "🎯 Po strategiji", "🖼️ Vizualizacije", "🔗 Korelacije"]
)

with tab_stats:
    section_title("Deskriptivna statistika", icon="📊", subtitle="Numeričke značajke")
    st.dataframe(styled_descriptive_statistics(), width="stretch")

    section_title("Balans ciljne klase", icon="⚖️")
    c1, c2 = st.columns([1, 1], gap="large")
    with c1:
        st.dataframe(
            class_balance().style.format({"Broj": "{:.0f}", "Postotak (%)": "{:.1f}"}),
            width="stretch",
        )
    with c2:
        st.bar_chart(class_balance()["Broj"], height=300)

    alert_card(
        f"Izvor podataka: <code>{overview.get('source_file') or 'student_study_strategy.csv'}</code> · "
        f"Učitano zapisa: <strong>{fmt_number(overview.get('records'))}</strong> · "
        f"Ciljna varijabla: <strong>Preporučena strategija</strong>",
        variant="info",
        title="Kvalitet dataseta",
    )

with tab_strategy:
    section_title("Prosjek po strategiji", icon="🎯", subtitle="Numeričke značajke po ciljnoj klasi")
    means = strategy_means()
    c1, c2 = st.columns([1, 1], gap="large")
    with c1:
        st.dataframe(styled_strategy_means(), width="stretch")
    with c2:
        st.bar_chart(means, height=320)

    alert_card(
        "<strong>Odmor</strong> — nizak opterećenje, blizak ispit. "
        "<strong>Intenzivno učenje</strong> — visoki sati, malo dana. "
        "<strong>Dugoročni plan</strong> — daleko do ispita, niži stres.",
        variant="success",
        title="Ključni obrasci",
    )

with tab_figures:
    figure_count = eda_figure_count()
    fig_subtitle = (
        f"Generisano pipeline-om · {figure_count} figura"
        if figure_count is not None
        else "Generisano pipeline-om"
    )
    section_title("EDA grafikoni", icon="🖼️", subtitle=fig_subtitle)
    figure_groups = {
        "📋 Distribucija i kvalitet": ["01_missing_values.png", "02_target_distribution.png"],
        "📊 Histogrami": [
            "03_hist_HoursStudied.png",
            "03_hist_NumberOfSubjects.png",
            "03_hist_DaysUntilExam.png",
            "03_hist_SleepHours.png",
        ],
        "📦 Box plotovi": [
            "04_box_HoursStudied.png",
            "04_box_NumberOfSubjects.png",
            "04_box_DaysUntilExam.png",
            "04_box_SleepHours.png",
        ],
        "🏷️ Kategorički": [
            "05_count_StressLevel.png",
            "05_count_FatigueLevel.png",
            "05_count_SleepQuality.png",
            "05_count_PreviousFeedback.png",
            "06_StressLevel_vs_strategy.png",
            "06_FatigueLevel_vs_strategy.png",
            "06_SleepQuality_vs_strategy.png",
            "06_PreviousFeedback_vs_strategy.png",
        ],
        "🔬 Napredna analiza": [
            "14_correlation_matrix_numeric.png",
            "07_correlation_matrix_encoded.png",
            "08_iqr_outlier_counts.png",
            "09_sleep_quality_vs_strategy.png",
            "10_stress_level_vs_strategy.png",
            "11_fatigue_level_vs_strategy.png",
            "12_days_until_exam_vs_strategy.png",
            "13_sleep_hours_vs_strategy.png",
        ],
    }
    fig_tabs = st.tabs(list(figure_groups.keys()))
    for fig_tab, (group_name, files) in zip(fig_tabs, figure_groups.items(), strict=True):
        with fig_tab:
            cols = st.columns(2, gap="medium")
            for idx, filename in enumerate(files):
                path = FIGURES_DIR / filename
                with cols[idx % 2]:
                    if path.exists():
                        st.image(
                            str(path),
                            caption=filename.replace("_", " ").replace(".png", ""),
                            use_container_width=True,
                        )
                    else:
                        alert_card(f"Nedostaje: <code>{filename}</code>", variant="warning")

with tab_corr:
    section_title("Korelacijska matrica — numeričke varijable", icon="🔗")
    num_corr_path = FIGURES_DIR / "14_correlation_matrix_numeric.png"
    c1, c2 = st.columns([1, 1], gap="large")
    with c1:
        if num_corr_path.exists():
            st.image(str(num_corr_path), caption="Pearson korelacija (4 numerička feature-a)", width="stretch")
        else:
            alert_card("Numerička korelacijska matrica nije generisana. Pokrenite <code>python scripts/run_eda.py</code>.", variant="warning")
    with c2:
        num_pairs = _load_correlation_pairs(CORRELATION_PAIRS_NUMERIC)
        if num_pairs is not None:
            st.dataframe(num_pairs, width="stretch", hide_index=True)
        else:
            alert_card("Podaci o korelacijama nisu dostupni.", variant="warning")

    high_pairs = num_pairs[num_pairs["PearsonR"].abs() >= 0.85] if num_pairs is not None else None
    if high_pairs is not None and not high_pairs.empty:
        alert_card(
            "Pronađene su vrlo visoke korelacije (|r| ≥ 0.85) — provjerite redundantnost.",
            variant="warning",
            title="Visoka korelacija",
        )
    else:
        alert_card(
            "Nijedan par numeričkih varijabli ne prelazi prag |r| ≥ 0.85. "
            "Najjača korelacija: DaysUntilExam ↔ SleepHours (r ≈ 0.75).",
            variant="info",
            title="Analiza korelacije",
        )

    section_title("Korelacije svih feature-a (ordinalno kodirano)", icon="📊", subtitle="Eksplorativno — ne koristi se u modelu")
    enc_pairs = _load_correlation_pairs(CORRELATION_PAIRS_ENCODED)
    if enc_pairs is not None:
        top_pairs = enc_pairs.head(6)
        cols = st.columns(2, gap="medium")
        for i, (_, row) in enumerate(top_pairs.iterrows()):
            with cols[i % 2]:
                r_val = float(row["PearsonR"])
                color = "#22C55E" if r_val > 0 else "#EF4444"
                st.markdown(
                    f"""
                    <div class="stat-card" style="--accent: {color};">
                        <div class="stat-label">{row['FeatureA']} ↔ {row['FeatureB']}</div>
                        <div class="stat-value" style="font-size:1.35rem;">{r_val:+.3f}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    section_title("Redundantne varijable", icon="⚖️")
    if REDUNDANCY_REPORT.exists():
        with st.expander("📄 Analiza redundantnosti (iz ML pipeline-a)", expanded=True):
            st.markdown(REDUNDANCY_REPORT.read_text(encoding="utf-8"))
    else:
        alert_card("Izvještaj o redundantnosti nije dostupan.", variant="warning")

    section_title("Kategoričke frekvencije", icon="🏷️")
    st.dataframe(categorical_summary(), width="stretch", hide_index=True)

    outlier_total = iqr_outlier_count()
    outlier_text = (
        f"IQR analiza: <strong>{outlier_total} outliera</strong> na numeričkim feature-ima."
        if outlier_total is not None
        else "IQR analiza nije dostupna."
    )
    outlier_note = (
        " Ekstremna stanja zadržana kao validni studentski unosi."
        if outlier_total == 0
        else " Procijenite da li su ekstremne vrijednosti greške ili validna stanja."
    )
    alert_card(outlier_text + outlier_note, variant="info", title="Outlieri (IQR)")

with st.sidebar:
    render_sidebar_nav()

render_footer()
