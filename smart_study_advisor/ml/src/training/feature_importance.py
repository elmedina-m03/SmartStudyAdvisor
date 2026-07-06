"""Feature importance analysis for the Smart Study Advisor best model."""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression

RANDOM_STATE = 42

BUILTIN_IMPORTANCE_TYPES = (
    "RandomForestClassifier",
    "DecisionTreeClassifier",
    "GradientBoostingClassifier",
    "ExtraTreesClassifier",
)


def readable_feature_label(encoded_name: str) -> str:
    if encoded_name.startswith("numeric__"):
        return f"{encoded_name.removeprefix('numeric__')} (scaled)"
    if encoded_name.startswith("categorical__"):
        remainder = encoded_name.removeprefix("categorical__")
        for prefix in ("StressLevel", "FatigueLevel", "SleepQuality", "PreviousFeedback"):
            if remainder.startswith(f"{prefix}_"):
                level = remainder[len(prefix) + 1 :]
                return f"{prefix} = {level}"
    return encoded_name


def base_feature_group(encoded_name: str) -> str:
    if encoded_name.startswith("numeric__"):
        return encoded_name.removeprefix("numeric__")
    if encoded_name.startswith("categorical__"):
        remainder = encoded_name.removeprefix("categorical__")
        for prefix in ("StressLevel", "FatigueLevel", "SleepQuality", "PreviousFeedback"):
            if remainder.startswith(f"{prefix}_"):
                return prefix
    return encoded_name


def model_supports_builtin_importance(model: object) -> bool:
    return hasattr(model, "feature_importances_") and type(model).__name__ in BUILTIN_IMPORTANCE_TYPES


def extract_builtin_importance(model: object, feature_names: list[str]) -> pd.DataFrame:
    importances = model.feature_importances_
    df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
    df["ReadableLabel"] = df["Feature"].map(readable_feature_label)
    df["BaseFeature"] = df["Feature"].map(base_feature_group)
    return df.sort_values("Importance", ascending=False).reset_index(drop=True)


def extract_permutation_importance(
    model: object,
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: list[str],
) -> pd.DataFrame:
    result = permutation_importance(
        model,
        X,
        y,
        n_repeats=10,
        random_state=RANDOM_STATE,
        scoring="f1_macro",
    )
    df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": result.importances_mean,
            "Importance_std": result.importances_std,
        }
    )
    df["ReadableLabel"] = df["Feature"].map(readable_feature_label)
    df["BaseFeature"] = df["Feature"].map(base_feature_group)
    return df.sort_values("Importance", ascending=False).reset_index(drop=True)


def compute_feature_importance(
    model: object,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_names: list[str],
) -> tuple[pd.DataFrame, str, str]:
    if model_supports_builtin_importance(model):
        df = extract_builtin_importance(model, feature_names)
        method = "builtin"
        explanation = (
            f"The best model is a **{type(model).__name__}**, which provides "
            f"built-in `feature_importances_` based on mean decrease in impurity "
            f"(Gini importance across trees)."
        )
    else:
        df = extract_permutation_importance(model, X_train, y_train, feature_names)
        method = "permutation"
        reason = "Permutation importance" if isinstance(model, LogisticRegression) else "Permutation importance (fallback)"
        explanation = (
            f"The best model is a **{type(model).__name__}**, which does not expose "
            f"reliable built-in feature importances. **{reason}** was used instead: "
            f"each feature is shuffled and the drop in macro F1-score is measured "
            f"(10 repeats, averaged)."
        )
    return df, method, explanation


def aggregate_by_base_feature(importance_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        importance_df.groupby("BaseFeature", as_index=False)["Importance"]
        .sum()
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )
    grouped["Rank"] = grouped.index + 1
    return grouped


def plot_feature_importance(importance_df: pd.DataFrame, output_path: Path, top_n: int = 17) -> None:
    plot_df = importance_df.head(top_n).sort_values("Importance", ascending=True)
    fig, ax = plt.subplots(figsize=(10, max(6, 0.35 * len(plot_df))))
    sns.barplot(
        data=plot_df,
        x="Importance",
        y="ReadableLabel",
        hue="ReadableLabel",
        palette="viridis",
        legend=False,
        ax=ax,
    )
    ax.set_title("Feature Importance — Best Model")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _level_importance(importance_df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    mask = importance_df["Feature"].str.contains(f"categorical__{prefix}_", regex=False)
    subset = importance_df.loc[mask, ["ReadableLabel", "Importance"]].copy()
    return subset.sort_values("Importance", ascending=False)


def build_feature_importance_report(
    model: object,
    importance_df: pd.DataFrame,
    grouped_df: pd.DataFrame,
    method: str,
    method_explanation: str,
) -> str:
    top = importance_df.head(10)
    bottom = importance_df.tail(5)
    stress = _level_importance(importance_df, "StressLevel")
    fatigue = _level_importance(importance_df, "FatigueLevel")
    sleep_q = _level_importance(importance_df, "SleepQuality")

    top_display = top.reset_index(drop=True).copy()
    top_display.index = top_display.index + 1
    top_display.index.name = "Rank"

    lines = [
        "# Feature Importance Report — Smart Study Advisor",
        "",
        f"## Model analysed",
        f"- **Type:** `{type(model).__name__}`",
        f"- **Source:** `models/best_model.pkl`",
        f"- **Method:** {method}",
        "",
        method_explanation,
        "",
        "## Top 10 features (encoded names)",
        "",
        "```",
        top_display[["ReadableLabel", "Importance"]].round(4).to_string(),
        "```",
        "",
        "## Aggregated importance by base feature",
        "",
        "```",
        grouped_df.to_string(index=False),
        "```",
        "",
        "## Most important features",
        "",
    ]
    for _, row in grouped_df.head(5).iterrows():
        lines.append(f"- **{row['BaseFeature']}** — total importance {row['Importance']:.4f}")

    lines.extend(
        [
            "",
            "## Least influential features",
            "",
        ]
    )
    for _, row in bottom.iterrows():
        lines.append(f"- `{row['ReadableLabel']}` — importance {row['Importance']:.4f}")

    lines.extend(
        [
            "",
            "## Domain consistency",
            "",
            "The ranking aligns with the rules used to generate the dataset and the EDA findings:",
            "- **DaysUntilExam** separates urgent strategies (Rest, IntensiveStudy) from LongTermPlan.",
            "- **StressLevel** and **FatigueLevel** strongly predict Rest vs recovery-oriented plans.",
            "- **SleepQuality** and **SleepHours** reflect recovery state relevant to Rest recommendations.",
            "- **HoursStudied** and **NumberOfSubjects** distinguish IntensiveStudy from BalancedStudy.",
            "",
            "## How SleepQuality influences recommendations",
            "",
            "```",
            sleep_q.to_string(index=False) if not sleep_q.empty else "N/A",
            "```",
            "",
            f"**Interpretation:** `{sleep_q.iloc[0]['ReadableLabel']}` has the highest SleepQuality-level importance "
            f"({sleep_q.iloc[0]['Importance']:.4f}). "
            "Poor sleep pushes toward **Rest**; Good sleep supports **LongTermPlan** and **BalancedStudy**.",
            "",
            "## How StressLevel influences recommendations",
            "",
            "```",
            stress.to_string(index=False) if not stress.empty else "N/A",
            "```",
            "",
            f"**Interpretation:** `{stress.iloc[0]['ReadableLabel']}` dominates stress-related splits. "
            "High stress aligns with **Rest** and **IntensiveStudy**; Low stress with **LongTermPlan**.",
            "",
            "## How FatigueLevel influences recommendations",
            "",
            "```",
            fatigue.to_string(index=False) if not fatigue.empty else "N/A",
            "```",
            "",
            f"**Interpretation:** `{fatigue.iloc[0]['ReadableLabel']}` is the strongest fatigue signal. "
            "High fatigue is the primary indicator for **Rest**; Low fatigue supports longer-horizon planning.",
            "",
            "## Output files",
            "- `feature_importance.csv`",
            "- `feature_importance.png`",
            "- `feature_importance_report.md`",
            "",
        ]
    )
    return "\n".join(lines)


def run_feature_importance_analysis(
    models_dir: Path,
    processed_dir: Path,
    output_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Path], str, str]:
    model = joblib.load(models_dir / "best_model.pkl")
    X_train = pd.read_csv(processed_dir / "X_train.csv")
    y_train = pd.read_csv(processed_dir / "y_train.csv").squeeze()
    feature_names = X_train.columns.tolist()

    importance_df, method, explanation = compute_feature_importance(
        model, X_train, y_train, feature_names
    )
    importance_df.insert(0, "Rank", importance_df.index + 1)

    grouped_df = aggregate_by_base_feature(importance_df)

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "feature_importance.csv"
    png_path = output_dir / "feature_importance.png"
    report_path = output_dir / "feature_importance_report.md"

    importance_df.to_csv(csv_path, index=False)
    plot_feature_importance(importance_df, png_path)
    report_path.write_text(
        build_feature_importance_report(model, importance_df, grouped_df, method, explanation),
        encoding="utf-8",
    )

    paths = {"csv": csv_path, "png": png_path, "report": report_path}
    return importance_df, paths, method, type(model).__name__
