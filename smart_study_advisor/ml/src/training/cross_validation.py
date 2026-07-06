"""5-fold cross-validation for the Smart Study Advisor Random Forest model."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from src.data.schema import FEATURE_COLUMNS, TARGET_COLUMN, load_dataset
from src.preprocessing.pipeline import build_preprocessor

RANDOM_STATE = 42
N_SPLITS = 5
HOLDOUT_TEST_F1 = 0.9917  # from Step 5 hold-out evaluation


def build_rf_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", RandomForestClassifier(random_state=RANDOM_STATE)),
        ]
    )


def run_cross_validation(df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df if df is not None else load_dataset()

    X = df[FEATURE_COLUMNS]
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[TARGET_COLUMN])

    pipeline = build_rf_pipeline()
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    scoring = {
        "accuracy": "accuracy",
        "precision_macro": "precision_macro",
        "recall_macro": "recall_macro",
        "f1_macro": "f1_macro",
    }

    results = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1,
    )

    fold_rows = []
    for fold in range(N_SPLITS):
        fold_rows.append(
            {
                "fold": fold + 1,
                "accuracy": results["test_accuracy"][fold],
                "precision_macro": results["test_precision_macro"][fold],
                "recall_macro": results["test_recall_macro"][fold],
                "f1_macro": results["test_f1_macro"][fold],
                "train_accuracy": results["train_accuracy"][fold],
                "train_precision_macro": results["train_precision_macro"][fold],
                "train_recall_macro": results["train_recall_macro"][fold],
                "train_f1_macro": results["train_f1_macro"][fold],
            }
        )

    fold_df = pd.DataFrame(fold_rows)

    summary_rows = []
    for metric in ("accuracy", "precision_macro", "recall_macro", "f1_macro"):
        summary_rows.append(
            {
                "metric": metric,
                "mean": fold_df[metric].mean(),
                "std": fold_df[metric].std(),
            }
        )

    train_summary_rows = []
    for metric in ("train_accuracy", "train_precision_macro", "train_recall_macro", "train_f1_macro"):
        train_summary_rows.append(
            {
                "metric": metric,
                "mean": fold_df[metric].mean(),
                "std": fold_df[metric].std(),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    train_summary_df = pd.DataFrame(train_summary_rows)
    return fold_df, summary_df, train_summary_df


def plot_cv_boxplot(fold_df: pd.DataFrame, output_path: Path) -> None:
    plot_df = fold_df.melt(
        id_vars=["fold"],
        value_vars=["accuracy", "precision_macro", "recall_macro", "f1_macro"],
        var_name="metric",
        value_name="score",
    )
    label_map = {
        "accuracy": "Accuracy",
        "precision_macro": "Precision (macro)",
        "recall_macro": "Recall (macro)",
        "f1_macro": "F1 (macro)",
    }
    plot_df["metric"] = plot_df["metric"].map(label_map)
    plot_df["metric"] = pd.Categorical(
        plot_df["metric"],
        categories=[
            "Accuracy",
            "Precision (macro)",
            "Recall (macro)",
            "F1 (macro)",
        ],
        ordered=True,
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=plot_df, x="metric", y="score", color="#8ecae6", ax=ax, width=0.5)
    sns.stripplot(
        data=plot_df,
        x="metric",
        y="score",
        color="#023047",
        size=8,
        jitter=0.08,
        ax=ax,
    )
    ax.set_title("5-Fold Cross-Validation — Out-of-Fold Test Metrics (Random Forest)")
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.set_ylim(0.94, 1.01)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_cv_summary(
    fold_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    train_summary_df: pd.DataFrame,
) -> str:
    test_f1_mean = summary_df.loc[summary_df["metric"] == "f1_macro", "mean"].iloc[0]
    test_f1_std = summary_df.loc[summary_df["metric"] == "f1_macro", "std"].iloc[0]
    train_f1_mean = train_summary_df.loc[
        train_summary_df["metric"] == "train_f1_macro", "mean"
    ].iloc[0]
    train_test_gap = train_f1_mean - test_f1_mean

    fold_f1_std = fold_df["f1_macro"].std()
    fold_f1_range = fold_df["f1_macro"].max() - fold_df["f1_macro"].min()
    folds_consistent = fold_f1_std < 0.02 and fold_f1_range <= 0.05

    generalizes_well = test_f1_mean >= 0.95 and test_f1_std < 0.02
    overfitting_evidence = train_test_gap > 0.03

    def _metric_row(metric_key: str, label: str) -> str:
        row = summary_df[summary_df["metric"] == metric_key].iloc[0]
        return f"| {label} | {row['mean']:.4f} | {row['std']:.4f} |"

    lines = [
        "# Cross-Validation Summary — Random Forest (5-Fold)",
        "",
        "## Setup",
        "- **Model:** RandomForestClassifier (default hyperparameters, random_state=42)",
        "- **Preprocessing:** `build_preprocessor()` — StandardScaler + OneHotEncoder (fitted per fold)",
        "- **CV:** StratifiedKFold, n_splits=5, shuffle=True, random_state=42",
        "- **Dataset:** Full 400 records (all rows participate in CV)",
        "",
        "## Per-fold results (out-of-fold test)",
        "",
        "```",
        fold_df[["fold", "accuracy", "precision_macro", "recall_macro", "f1_macro"]]
        .round(4)
        .to_string(index=False),
        "```",
        "",
        "## Summary statistics (test)",
        "",
        "| Metric | Mean | Std |",
        "|--------|------|-----|",
        _metric_row("accuracy", "Accuracy"),
        _metric_row("precision_macro", "Precision (macro)"),
        _metric_row("recall_macro", "Recall (macro)"),
        _metric_row("f1_macro", "F1 (macro)"),
        "",
        "## Train vs test (in-fold vs out-of-fold)",
        "",
        f"- Mean train F1 (macro): **{train_f1_mean:.4f}**",
        f"- Mean test F1 (macro): **{test_f1_mean:.4f}** ± {test_f1_std:.4f}",
        f"- Train–test F1 gap: **{train_test_gap:.4f}**",
        "",
        "## Comparison with hold-out test (Step 5)",
        f"- Hold-out test F1 (macro): **{HOLDOUT_TEST_F1:.4f}** (70/30 split)",
        f"- CV mean test F1 (macro): **{test_f1_mean:.4f}** ± {test_f1_std:.4f}",
        f"- Difference: {abs(HOLDOUT_TEST_F1 - test_f1_mean):.4f}",
        "",
        "## Generalization",
        "",
    ]

    if generalizes_well:
        lines.append(
            "The model **generalizes well**. Mean out-of-fold F1 is "
            f"**{test_f1_mean:.4f}** with standard deviation **{test_f1_std:.4f}**, "
            "indicating stable performance across folds and alignment with the hold-out evaluation."
        )
    else:
        lines.append(
            "Generalization is **moderate**. Review fold variance and per-class errors before deployment."
        )

    lines.extend(["", "## Overfitting", ""])

    if overfitting_evidence:
        lines.append(
            f"There is **some evidence of overfitting**: mean train F1 ({train_f1_mean:.4f}) "
            f"exceeds mean test F1 ({test_f1_mean:.4f}) by {train_test_gap:.4f}. "
            "The Random Forest fits training folds perfectly; monitor this gap on real data."
        )
    else:
        lines.append(
            f"**No strong evidence of overfitting.** The train–test F1 gap is **{train_test_gap:.4f}** "
            f"(train {train_f1_mean:.4f} vs test {test_f1_mean:.4f}). "
            "Test performance remains high and consistent across folds."
        )

    lines.extend(["", "## Fold consistency", ""])

    if folds_consistent:
        lines.append(
            f"Fold results are **consistent**. Test F1 ranges from "
            f"**{fold_df['f1_macro'].min():.4f}** to **{fold_df['f1_macro'].max():.4f}** "
            f"(range = {fold_f1_range:.4f}, σ = {fold_f1_std:.4f}). "
            "No single fold dominates or underperforms severely."
        )
    else:
        lines.append(
            f"Fold results show **moderate variation**. Test F1 ranges from "
            f"{fold_df['f1_macro'].min():.4f} to {fold_df['f1_macro'].max():.4f} "
            f"(σ = {fold_f1_std:.4f}). Investigate fold-specific class errors."
        )

    lines.extend(
        [
            "",
            "## Output files",
            "- `cross_validation_results.csv`",
            "- `cross_validation_summary.md`",
            "- `cross_validation_boxplot.png`",
            "",
        ]
    )
    return "\n".join(lines)


def save_cv_outputs(
    fold_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    train_summary_df: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    fold_table = fold_df[
        ["fold", "accuracy", "precision_macro", "recall_macro", "f1_macro"]
    ].round(4)
    summary_table = summary_df.round(4)

    csv_path = output_dir / "cross_validation_results.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("# per_fold\n")
        fold_table.to_csv(f, index=False)
        f.write("\n# summary\n")
        summary_table.to_csv(f, index=False)

    summary_path = output_dir / "cross_validation_summary.md"
    summary_path.write_text(
        build_cv_summary(fold_df, summary_df, train_summary_df),
        encoding="utf-8",
    )

    boxplot_path = output_dir / "cross_validation_boxplot.png"
    plot_cv_boxplot(fold_df, boxplot_path)

    return {"csv": csv_path, "summary": summary_path, "boxplot": boxplot_path}
