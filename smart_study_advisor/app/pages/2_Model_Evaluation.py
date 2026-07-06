"""Model evaluation dashboard — metrics, CV, ablation, feature importance."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

APP_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.components.constants import PROJECT_COURSE, PROJECT_NAME  # noqa: E402
from app.components.formatting import fmt_decimal, style_score_table  # noqa: E402
from app.components.labels import feature_label  # noqa: E402
from app.components.metrics import (  # noqa: E402
    MODEL_LABELS,
    fmt_number,
    fmt_percent,
    load_cv_fold_results,
    load_evaluation_metrics,
)
from app.components.navigation import init_dashboard_page, render_sidebar_nav  # noqa: E402
from app.components.paths import ABLATION_DIR, CV_DIR, EVALUATION_DIR, EXPERIMENTS_DIR, FEATURE_IMPORTANCE_DIR  # noqa: E402
from app.components.ui import alert_card, render_confidence_bar, render_footer, section_title, stat_card_row  # noqa: E402

init_dashboard_page(
    page_title=f"Evaluacija | {PROJECT_NAME}",
    page_icon="📈",
    header_title="Evaluacija modela",
    header_subtitle=f"Usporedba · unakrsna validacija · uklanjanje značajki · važnost | {PROJECT_COURSE}",
    header_badge="Evaluacija ML modela",
)

eval_metrics = load_evaluation_metrics()


@st.cache_data(show_spinner="Učitavanje metrika...")
def load_model_comparison() -> pd.DataFrame | None:
    path = EVALUATION_DIR / "model_comparison.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        df["Model"] = df["Model"].map(lambda x: MODEL_LABELS.get(x, x))
        return df.rename(
            columns={
                "Accuracy": "Točnost",
                "Precision_macro": "Preciznost",
                "Recall_macro": "Odziv",
                "F1_macro": "F1",
            }
        )
    except (OSError, pd.errors.ParserError, ValueError):
        return None


@st.cache_data(show_spinner=False)
def load_cv_results() -> pd.DataFrame | None:
    df = load_cv_fold_results()
    if df is None:
        return None
    return df.rename(
        columns={
            "fold": "Fold",
            "accuracy": "Točnost",
            "precision_macro": "Preciznost",
            "recall_macro": "Odziv",
            "f1_macro": "F1",
        }
    )


@st.cache_data(show_spinner=False)
def load_ablation() -> pd.DataFrame | None:
    path = ABLATION_DIR / "ablation_results.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        if "removed_feature" in df.columns:
            df["removed_feature"] = df["removed_feature"].apply(
                lambda x: feature_label(str(x)) if pd.notna(x) and str(x).strip() else "— (sve značajke)"
            )
        return df.rename(
            columns={
                "experiment": "Eksperiment",
                "removed_feature": "Uklonjena značajka",
                "f1_macro": "F1",
                "delta_f1": "Δ F1",
                "accuracy": "Točnost",
                "precision_macro": "Preciznost",
                "recall_macro": "Odziv",
            }
        )
    except (OSError, pd.errors.ParserError, ValueError):
        return None


@st.cache_data(show_spinner=False)
def load_experiments() -> pd.DataFrame | None:
    path = EXPERIMENTS_DIR / "experiment_comparison.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        return df.rename(
            columns={
                "Accuracy": "Točnost",
                "Precision": "Preciznost",
                "Recall": "Odziv",
                "F1 Score": "F1",
                "Train omjer": "Udio treninga",
                "Test omjer": "Udio testa",
                "Train zapisa": "Zapisa (trening)",
                "Test zapisa": "Zapisa (test)",
                "Broj feature-a": "Broj značajki",
                "Uklonjeni feature-i": "Uklonjene značajke",
            }
        )
    except (OSError, pd.errors.ParserError, ValueError):
        return None


@st.cache_data(show_spinner=False)
def load_feature_importance() -> pd.DataFrame | None:
    path = FEATURE_IMPORTANCE_DIR / "feature_importance.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        return df[["Rank", "ReadableLabel", "Importance", "BaseFeature"]].rename(
            columns={"ReadableLabel": "Značajka", "Importance": "Važnost", "BaseFeature": "Osnovna značajka", "Rank": "Rang"}
        )
    except (OSError, pd.errors.ParserError, ValueError, KeyError):
        return None


cv_std_text = (
    f"σ={fmt_decimal(eval_metrics.cv_std_f1, digits=4)}"
    if eval_metrics.cv_std_f1 is not None
    else None
)

stat_card_row(
    [
        {
            "label": "Najbolji model",
            "value": eval_metrics.best_model_label or "N/A",
            "icon": "🌲",
            "color": "#6366F1",
        },
        {
            "label": "F1 (test skup)",
            "value": fmt_percent(eval_metrics.holdout_f1),
            "icon": "🎯",
            "color": "#22C55E",
        },
        {
            "label": "CV prosjek F1",
            "value": fmt_percent(eval_metrics.cv_mean_f1),
            "icon": "🔄",
            "delta": cv_std_text,
            "color": "#8B5CF6",
        },
        {
            "label": "Testiranih modela",
            "value": fmt_number(eval_metrics.model_count),
            "icon": "⚖️",
            "color": "#06B6D4",
        },
    ]
)

st.markdown("")
if eval_metrics.holdout_f1 is not None:
    render_confidence_bar(
        eval_metrics.holdout_f1,
        label=f"F1 na test skupu ({eval_metrics.best_model_label or 'model'})",
    )
else:
    alert_card("F1 na test skupu nije dostupan — nedostaje model_comparison.csv.", variant="warning")

tab_compare, tab_cm, tab_cv, tab_ablation, tab_fi, tab_exp = st.tabs(
    ["⚖️ Usporedba", "🎯 Matrica konfuzije", "🔄 Unakrsna validacija", "🔬 Uklanjanje feature-a", "📊 Važnost značajki", "📋 Eksperimenti"]
)

with tab_compare:
    section_title("Usporedba modela", icon="⚖️", subtitle="Test skup 30% · odabir po F1")
    comparison = load_model_comparison()
    if comparison is not None:
        st.dataframe(
            style_score_table(comparison).highlight_max(subset=["F1"], color="#DCFCE7"),
            width="stretch",
            hide_index=True,
        )
        best_name = eval_metrics.best_model_label or "N/A"
        best_f1 = fmt_percent(eval_metrics.holdout_f1)
        alert_card(
            f"<strong>{best_name}</strong> ima najviši F1 ({best_f1}) na test skupu.",
            variant="success",
            title="Zaključak",
        )
    else:
        alert_card("Usporedba modela nije dostupna.", variant="warning", title="Nedostaju podaci")

with tab_cm:
    section_title("Matrica konfuzije", icon="🎯", subtitle="Test skup — najbolji model")
    cm_path = EVALUATION_DIR / "confusion_matrix_best_model.png"
    report_path = EVALUATION_DIR / "classification_report_best_model.txt"
    c1, c2 = st.columns([1, 1], gap="large")
    with c1:
        if cm_path.exists():
            st.image(str(cm_path), width="stretch")
        else:
            alert_card("Slika matrice konfuzije nije dostupna.", variant="warning")
    with c2:
        if report_path.exists():
            st.markdown("#### Izvještaj po klasama")
            st.code(report_path.read_text(encoding="utf-8"), language="text")
        else:
            alert_card("Izvještaj klasifikacije nije dostupan.", variant="warning")

with tab_cv:
    section_title("5-fold stratificirana CV", icon="🔄")
    cv_df = load_cv_results()
    c1, c2 = st.columns([1, 1], gap="large")
    with c1:
        if cv_df is not None:
            st.dataframe(style_score_table(cv_df), width="stretch", hide_index=True)
            st.metric("Prosjek F1", fmt_percent(float(cv_df["F1"].mean())))
            st.metric("Std. dev. F1", fmt_decimal(float(cv_df["F1"].std()), digits=4))
        else:
            alert_card("CV rezultati nisu dostupni.", variant="warning")
    with c2:
        boxplot = CV_DIR / "cross_validation_boxplot.png"
        if boxplot.exists():
            st.image(str(boxplot), width="stretch")
        else:
            alert_card("CV grafikon nije dostupan.", variant="warning")
    if eval_metrics.cv_mean_f1 is not None:
        cv_msg = f"CV prosjek F1: <strong>{fmt_percent(eval_metrics.cv_mean_f1)}</strong>"
        if eval_metrics.cv_std_f1 is not None:
            cv_msg += f" (σ={eval_metrics.cv_std_f1:.4f})."
        alert_card(cv_msg, variant="info", title="Generalizacija")

with tab_ablation:
    section_title("Studija uklanjanja značajki", icon="🔬", subtitle="Marginalni doprinos svake značajke")
    ablation = load_ablation()
    c1, c2 = st.columns([1, 1], gap="large")
    with c1:
        if ablation is not None:
            st.dataframe(style_score_table(ablation), width="stretch", hide_index=True)
        else:
            alert_card("Rezultati studije uklanjanja nisu dostupni.", variant="warning")
    with c2:
        chart_path = ABLATION_DIR / "ablation_comparison.png"
        if chart_path.exists():
            st.image(str(chart_path), width="stretch")
        else:
            alert_card("Grafikon studije uklanjanja nije dostupan.", variant="warning")

with tab_fi:
    section_title("Važnost značajki", icon="📊", subtitle="Gini važnost — najbolji model")
    importance = load_feature_importance()
    c1, c2 = st.columns([1, 1], gap="large")
    with c1:
        if importance is not None:
            st.dataframe(
                importance.head(12).style.background_gradient(subset=["Važnost"], cmap="Purples"),
                width="stretch",
                hide_index=True,
            )
        else:
            alert_card("Važnost značajki nije dostupna.", variant="warning")
    with c2:
        fi_chart = FEATURE_IMPORTANCE_DIR / "feature_importance.png"
        if fi_chart.exists():
            st.image(str(fi_chart), width="stretch")
        else:
            alert_card("Grafikon važnosti nije dostupan.", variant="warning")

with tab_exp:
    section_title(
        "Konačna analiza eksperimenata",
        icon="📋",
        subtitle="Originalni model · bez redundantnih feature-a · train/test omjeri · outlieri",
    )
    experiments = load_experiments()
    summary_path = EXPERIMENTS_DIR / "experiment_summary.md"
    chart_path = EXPERIMENTS_DIR / "experiment_f1_comparison.png"

    if experiments is None:
        alert_card(
            "Eksperimenti nisu pokrenuti. Izvršite: <code>python scripts/run_ml_experiments.py</code> u <code>ml/</code> folderu.",
            variant="warning",
            title="Nedostaju podaci",
        )
    else:
        metric_cols = ["Točnost", "Preciznost", "Odziv", "F1"]
        st.dataframe(
            style_score_table(experiments).highlight_max(subset=metric_cols, color="#DCFCE7"),
            width="stretch",
            hide_index=True,
        )
        c1, c2 = st.columns([1, 1], gap="large")
        with c1:
            if chart_path.exists():
                st.image(str(chart_path), width="stretch")
        with c2:
            best_row = experiments.loc[experiments["F1"].idxmax()]
            alert_card(
                f"<strong>{best_row['Opis']}</strong><br>"
                f"Točnost: {best_row['Točnost']:.2%} · Preciznost: {best_row['Preciznost']:.2%}<br>"
                f"Odziv: {best_row['Odziv']:.2%} · F1: <strong>{best_row['F1']:.2%}</strong>",
                variant="success",
                title="Najbolji eksperiment (po F1)",
            )
            alert_card(
                "Za produkciju zadržavamo <strong>70/30</strong> jer daje pouzdaniji test skup (120 zapisa). "
                "Viši omjeri (80/20, 85/15) postižu F1=1.0 na sintetičkom datasetu, "
                "ali test skup je premali za generalizaciju.",
                variant="info",
                title="Preporuka",
            )
        if summary_path.exists():
            with st.expander("📄 Detaljan zaključak (ML izvještaj)", expanded=False):
                st.markdown(summary_path.read_text(encoding="utf-8"))

with st.sidebar:
    render_sidebar_nav()

render_footer()
