"""Comprehensive ML experiments — split ratios, redundant features, outliers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from src.analysis.eda_helpers import detect_iqr_outliers, outlier_rows
from src.data.schema import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    load_dataset,
)

RANDOM_STATE = 42
REDUNDANT_FEATURES = ["SleepHours"]


@dataclass
class ExperimentResult:
    experiment_id: str
    description: str
    category: str
    train_ratio: float
    test_ratio: float
    n_train: int
    n_test: int
    n_features: int
    removed_features: str
    outlier_handling: str
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float


def _feature_columns(removed: list[str]) -> tuple[list[str], list[str], list[str]]:
    features = [col for col in FEATURE_COLUMNS if col not in removed]
    numeric = [col for col in NUMERIC_FEATURES if col in features]
    categorical = [col for col in CATEGORICAL_FEATURES if col in features]
    return features, numeric, categorical


def _build_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    transformers = []
    if numeric_cols:
        transformers.append(("numeric", StandardScaler(), numeric_cols))
    if categorical_cols:
        transformers.append(
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_cols,
            )
        )
    return ColumnTransformer(transformers=transformers, remainder="drop")


def _filter_outliers(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove rows flagged as IQR outliers on any numeric feature."""
    flagged = outlier_rows(df, NUMERIC_FEATURES)
    if flagged.empty:
        return df.copy(), 0
    return df.drop(index=flagged.index).copy(), len(flagged)


def run_experiment(
    df: pd.DataFrame,
    *,
    experiment_id: str,
    description: str,
    category: str,
    test_size: float,
    removed: list[str] | None = None,
    remove_outliers: bool = False,
) -> ExperimentResult:
    removed = removed or []
    work_df, n_outliers = _filter_outliers(df) if remove_outliers else (df.copy(), 0)
    outlier_note = f"uklonjeno {n_outliers} redova (IQR)" if remove_outliers else "zadržano"

    features, numeric_cols, categorical_cols = _feature_columns(removed)
    X = work_df[features].copy()
    y_raw = work_df[TARGET_COLUMN].copy()

    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X,
        y_raw,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y_raw,
    )

    preprocessor = _build_preprocessor(numeric_cols, categorical_cols)
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_raw)
    y_test = label_encoder.transform(y_test_raw)

    model = RandomForestClassifier(random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    train_ratio = 1.0 - test_size
    removed_label = ", ".join(removed) if removed else "—"

    return ExperimentResult(
        experiment_id=experiment_id,
        description=description,
        category=category,
        train_ratio=train_ratio,
        test_ratio=test_size,
        n_train=len(X_train_raw),
        n_test=len(X_test_raw),
        n_features=len(features),
        removed_features=removed_label,
        outlier_handling=outlier_note,
        accuracy=accuracy_score(y_test, y_pred),
        precision_macro=precision_score(y_test, y_pred, average="macro", zero_division=0),
        recall_macro=recall_score(y_test, y_pred, average="macro", zero_division=0),
        f1_macro=f1_score(y_test, y_pred, average="macro", zero_division=0),
    )


def run_all_experiments(df: pd.DataFrame | None = None) -> pd.DataFrame:
    df = df if df is not None else load_dataset()
    results: list[ExperimentResult] = []

    split_configs = [
        ("split_70_30", "Originalni model (70/30)", 0.30),
        ("split_75_25", "Train/test 75/25", 0.25),
        ("split_80_20", "Train/test 80/20", 0.20),
        ("split_85_15", "Train/test 85/15", 0.15),
    ]
    for exp_id, desc, test_size in split_configs:
        results.append(
            run_experiment(
                df,
                experiment_id=exp_id,
                description=desc,
                category="train_test_omjer",
                test_size=test_size,
            )
        )

    results.append(
        run_experiment(
            df,
            experiment_id="no_redundant",
            description="Bez SleepHours (redundantan feature)",
            category="redundantni_feature",
            test_size=0.30,
            removed=REDUNDANT_FEATURES,
        )
    )

    results.append(
        run_experiment(
            df,
            experiment_id="no_outliers",
            description="Nakon uklanjanja IQR outliera",
            category="outlieri",
            test_size=0.30,
            remove_outliers=True,
        )
    )

    return pd.DataFrame([r.__dict__ for r in results])


def build_experiment_report(results_df: pd.DataFrame, outlier_summary: pd.DataFrame) -> str:
    best = results_df.loc[results_df["f1_macro"].idxmax()]
    baseline = results_df[results_df["experiment_id"] == "split_70_30"].iloc[0]
    split_rows = results_df[results_df["category"] == "train_test_omjer"]
    best_split = split_rows.loc[split_rows["f1_macro"].idxmax()]

    outlier_count = int(outlier_summary["OutlierCount"].sum())
    lines = [
        "# Konačna analiza ML eksperimenata — Smart Study Advisor",
        "",
        "Model: **RandomForestClassifier** (sklearn zadane vrijednosti) · stratificirani split · random_state=42",
        "",
        "## Tabela svih eksperimenata",
        "",
        "```",
        results_df.round(4).to_string(index=False),
        "```",
        "",
        "## 1. Train/test omjer",
        "",
        f"- Trenutno korišteni omjer u produkciji: **70/30** (F1 = {baseline['f1_macro']:.4f}).",
        f"- Najbolji omjer u ovom poređenju: **{int(best_split['train_ratio']*100)}/{int(best_split['test_ratio']*100)}** "
        f"(F1 = {best_split['f1_macro']:.4f}).",
        "",
        "## 2. Redundantni feature-i",
        "",
        "- **SleepHours** uklonjen u eksperimentu `no_redundant` (preporuka iz korelacijske i ablation analize).",
        f"- F1 prije: {baseline['f1_macro']:.4f} → poslije: "
        f"{results_df[results_df['experiment_id']=='no_redundant'].iloc[0]['f1_macro']:.4f}.",
        "",
        "## 3. Outlieri (IQR)",
        "",
        f"- Ukupno IQR outliera u datasetu: **{outlier_count}**.",
    ]
    if outlier_count == 0:
        lines.append("- Eksperiment uklanjanja outliera daje **identične rezultate** kao original (nema šta ukloniti).")
    else:
        no_out = results_df[results_df["experiment_id"] == "no_outliers"].iloc[0]
        lines.append(f"- F1 nakon uklanjanja: **{no_out['f1_macro']:.4f}**.")

    lines.extend(
        [
            "",
            "## 4. Zaključak",
            "",
            f"**Najbolja konfiguracija:** `{best['experiment_id']}` — {best['description']}",
            f"- Accuracy: {best['accuracy']:.4f}",
            f"- Precision (macro): {best['precision_macro']:.4f}",
            f"- Recall (macro): {best['recall_macro']:.4f}",
            f"- F1 (macro): **{best['f1_macro']:.4f}**",
            "",
            "Za produkciju zadržavamo **70/30** jer je standard u projektu, razlike između omjera su minimalne "
            "na sintetičkom datasetu od 400 zapisa, a veći test skup daje pouzdaniju hold-out evaluaciju.",
            "",
        ]
    )
    return "\n".join(lines)


def plot_experiment_comparison(results_df: pd.DataFrame, output_path: Path) -> None:
    plot_df = results_df.copy()
    plot_df["Label"] = plot_df["description"].str.replace("Originalni model ", "", regex=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=plot_df, x="Label", y="f1_macro", hue="Label", palette="Set2", legend=False, ax=ax)
    ax.set_title("Macro F1 po eksperimentu")
    ax.set_ylabel("F1 (macro)")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_experiment_outputs(results_df: pd.DataFrame, output_dir: Path, df: pd.DataFrame) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outlier_summary = detect_iqr_outliers(df, NUMERIC_FEATURES)

    display_df = results_df.rename(
        columns={
            "experiment_id": "Eksperiment",
            "description": "Opis",
            "category": "Kategorija",
            "train_ratio": "Train omjer",
            "test_ratio": "Test omjer",
            "n_train": "Train zapisa",
            "n_test": "Test zapisa",
            "n_features": "Broj feature-a",
            "removed_features": "Uklonjeni feature-i",
            "outlier_handling": "Outlieri",
            "accuracy": "Accuracy",
            "precision_macro": "Precision",
            "recall_macro": "Recall",
            "f1_macro": "F1 Score",
        }
    )

    csv_path = output_dir / "experiment_comparison.csv"
    display_df.round(4).to_csv(csv_path, index=False)

    md_path = output_dir / "experiment_summary.md"
    md_path.write_text(build_experiment_report(results_df, outlier_summary), encoding="utf-8")

    png_path = output_dir / "experiment_f1_comparison.png"
    plot_experiment_comparison(results_df, png_path)

    return {"csv": csv_path, "md": md_path, "png": png_path}
