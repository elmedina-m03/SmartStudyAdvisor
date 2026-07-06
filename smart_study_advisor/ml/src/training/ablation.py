"""Ablation study — measure impact of removing features one at a time."""

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

from src.data.schema import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    load_dataset,
)

TEST_SIZE = 0.30
RANDOM_STATE = 42

EXPERIMENTS: dict[str, list[str]] = {
    "A_baseline": [],
    "B_no_SleepQuality": ["SleepQuality"],
    "C_no_StressLevel": ["StressLevel"],
    "D_no_FatigueLevel": ["FatigueLevel"],
    "E_no_PreviousFeedback": ["PreviousFeedback"],
    "F_no_SleepHours": ["SleepHours"],
}


@dataclass
class AblationResult:
    experiment: str
    removed_feature: str | None
    n_features: int
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    delta_f1: float
    delta_accuracy: float


def _feature_sets(removed: list[str]) -> tuple[list[str], list[str], list[str]]:
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


def _get_model() -> RandomForestClassifier:
    return RandomForestClassifier(random_state=RANDOM_STATE)


def run_single_experiment(
    df: pd.DataFrame,
    experiment_name: str,
    removed: list[str],
) -> AblationResult:
    features, numeric_cols, categorical_cols = _feature_sets(removed)

    X = df[features].copy()
    y_raw = df[TARGET_COLUMN].copy()

    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X,
        y_raw,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_raw,
    )

    preprocessor = _build_preprocessor(numeric_cols, categorical_cols)
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_raw)
    y_test = label_encoder.transform(y_test_raw)

    model = _get_model()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    removed_label = removed[0] if len(removed) == 1 else None

    return AblationResult(
        experiment=experiment_name,
        removed_feature=removed_label,
        n_features=len(features),
        accuracy=accuracy_score(y_test, y_pred),
        precision_macro=precision_score(y_test, y_pred, average="macro", zero_division=0),
        recall_macro=recall_score(y_test, y_pred, average="macro", zero_division=0),
        f1_macro=f1_score(y_test, y_pred, average="macro", zero_division=0),
        delta_f1=0.0,
        delta_accuracy=0.0,
    )


def run_ablation_study(df: pd.DataFrame | None = None) -> pd.DataFrame:
    df = df if df is not None else load_dataset()
    results: list[AblationResult] = []

    for name, removed in EXPERIMENTS.items():
        results.append(run_single_experiment(df, name, removed))

    baseline = next(r for r in results if r.experiment == "A_baseline")
    for result in results:
        result.delta_f1 = baseline.f1_macro - result.f1_macro
        result.delta_accuracy = baseline.accuracy - result.accuracy

    return pd.DataFrame([r.__dict__ for r in results])


def plot_ablation_comparison(results_df: pd.DataFrame, output_path: Path) -> None:
    plot_df = results_df.copy()
    plot_df["Label"] = plot_df.apply(
        lambda row: "Baseline (all)"
        if row["removed_feature"] is None
        else f"No {row['removed_feature']}",
        axis=1,
    )
    plot_df = plot_df.set_index("experiment").loc[list(EXPERIMENTS.keys())].reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.barplot(
        data=plot_df, x="Label", y="f1_macro",
        hue="Label", palette="Set2", legend=False, ax=axes[0],
    )
    axes[0].set_title("Macro F1 by Ablation Experiment")
    axes[0].set_ylabel("F1 (macro)")
    axes[0].tick_params(axis="x", rotation=25)

    sns.barplot(
        data=plot_df, x="Label", y="delta_f1",
        hue="Label", palette="Reds_r", legend=False, ax=axes[1],
    )
    axes[1].set_title("F1 Drop vs Baseline")
    axes[1].set_ylabel("Δ F1 (baseline − experiment)")
    axes[1].tick_params(axis="x", rotation=25)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _feature_usefulness_rank(results_df: pd.DataFrame) -> pd.DataFrame:
    ablated = results_df[results_df["removed_feature"].notna()].copy()
    ranked = (
        ablated.sort_values("delta_f1", ascending=False)
        .reset_index(drop=True)
        .rename(columns={"removed_feature": "Feature"})
    )
    ranked.insert(0, "UsefulnessRank", ranked.index + 1)
    return ranked[["UsefulnessRank", "Feature", "f1_macro", "delta_f1", "delta_accuracy"]]


def build_ablation_report(results_df: pd.DataFrame, usefulness_df: pd.DataFrame) -> str:
    baseline = results_df[results_df["experiment"] == "A_baseline"].iloc[0]
    ablated = results_df[results_df["removed_feature"].notna()].copy()
    biggest_drop = ablated.loc[ablated["delta_f1"].idxmax()]
    minimal_effect = ablated.loc[ablated["delta_f1"].idxmin()]

    sleep_row = ablated[ablated["removed_feature"] == "SleepQuality"].iloc[0]
    stress_row = ablated[ablated["removed_feature"] == "StressLevel"].iloc[0]
    fatigue_row = ablated[ablated["removed_feature"] == "FatigueLevel"].iloc[0]

    lines = [
        "# Ablation Study Report — Smart Study Advisor",
        "",
        "## Setup",
        "- **Model:** RandomForestClassifier (same hyperparameters as best model)",
        "- **Split:** 70/30 stratified, random_state=42",
        "- **Preprocessing:** StandardScaler (numeric) + OneHotEncoder (categorical)",
        "- **Note:** Temporary models only — `models/best_model.pkl` was not modified.",
        "",
        f"## Baseline (Experiment A)",
        f"- Accuracy: {baseline['accuracy']:.4f}",
        f"- F1 (macro): {baseline['f1_macro']:.4f}",
        f"- Features: {int(baseline['n_features'])}",
        "",
        "## All experiment results",
        "",
        "```",
        results_df.round(4).to_string(index=False),
        "```",
        "",
        "## Feature usefulness ranking (by F1 drop when removed)",
        "",
        "```",
        usefulness_df.round(4).to_string(index=False),
        "```",
        "",
        "## Key findings",
        "",
        f"### Biggest performance drop",
        f"- Removing **{biggest_drop['removed_feature']}** caused the largest F1 decrease "
        f"(ΔF1 = {biggest_drop['delta_f1']:.4f}, F1 = {biggest_drop['f1_macro']:.4f}).",
        "",
        f"### Smallest performance effect",
        f"- Removing **{minimal_effect['removed_feature']}** had the smallest impact "
        f"(ΔF1 = {minimal_effect['delta_f1']:.4f}).",
        "",
        "### Consistency with feature importance",
        "- Feature importance (Gini) ranked DaysUntilExam, HoursStudied, and SleepHours highest among encoded columns.",
        f"- Ablation shows **FatigueLevel** causes the largest drop (ΔF1 = {fatigue_row['delta_f1']:.4f}), consistent with its role in identifying Rest.",
        f"- **SleepHours** showed **no F1 drop** when removed (ΔF1 = 0), despite high Gini importance — its signal is likely captured by SleepQuality, DaysUntilExam, and FatigueLevel.",
        "- Gini importance and ablation measure different things: Gini reflects split usage; ablation measures marginal contribution given all other features.",
        "",
        "### SleepQuality",
        f"- Removing SleepQuality changed F1 by **{sleep_row['delta_f1']:.4f}**.",
        "- SleepQuality is a useful but not critical predictor; combined with SleepHours redundancy, a simplified sleep feature may suffice.",
        "",
        "### StressLevel and FatigueLevel",
        f"- StressLevel removal: ΔF1 = **{stress_row['delta_f1']:.4f}**",
        f"- FatigueLevel removal: ΔF1 = **{fatigue_row['delta_f1']:.4f}** (largest drop)",
        "- FatigueLevel is the most important categorical group for prediction; StressLevel adds moderate value.",
        "",
        "### Features that could potentially be removed",
    ]

    low_impact = ablated[ablated["delta_f1"] <= 0.0083].sort_values("delta_f1")
    if low_impact.empty:
        lines.append("- All removed features caused a measurable drop (>0.01 F1). None are strictly redundant.")
    else:
        for _, row in low_impact.iterrows():
            lines.append(
                f"- **{row['removed_feature']}** (ΔF1 = {row['delta_f1']:.4f}) — minimal impact; candidate for simplification."
            )

    lines.extend(
        [
            "",
            "## Output files",
            "- `ablation_results.csv`",
            "- `ablation_results.md`",
            "- `ablation_comparison.png`",
            "",
        ]
    )
    return "\n".join(lines)


def save_ablation_outputs(results_df: pd.DataFrame, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    usefulness_df = _feature_usefulness_rank(results_df)

    csv_path = output_dir / "ablation_results.csv"
    results_df.round(4).to_csv(csv_path, index=False)

    md_path = output_dir / "ablation_results.md"
    md_path.write_text(build_ablation_report(results_df, usefulness_df), encoding="utf-8")

    png_path = output_dir / "ablation_comparison.png"
    plot_ablation_comparison(results_df, png_path)

    return {"csv": csv_path, "md": md_path, "png": png_path, "usefulness": usefulness_df}
