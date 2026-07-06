"""Model training and evaluation for the Smart Study Advisor classifier."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42


@dataclass
class ModelResult:
    name: str
    model: object
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    y_pred: np.ndarray
    classification_report_text: str
    confusion_matrix: np.ndarray


def get_model_candidates() -> dict[str, object]:
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "decision_tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(random_state=RANDOM_STATE),
        "gradient_boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }


def load_processed_data(processed_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X_train = pd.read_csv(processed_dir / "X_train.csv")
    X_test = pd.read_csv(processed_dir / "X_test.csv")
    y_train = pd.read_csv(processed_dir / "y_train.csv").squeeze()
    y_test = pd.read_csv(processed_dir / "y_test.csv").squeeze()
    return X_train, X_test, y_train, y_test


def evaluate_model(
    name: str,
    model: object,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    target_names: list[str],
) -> ModelResult:
    y_pred = model.predict(X_test)

    return ModelResult(
        name=name,
        model=model,
        accuracy=accuracy_score(y_test, y_pred),
        precision_macro=precision_score(y_test, y_pred, average="macro", zero_division=0),
        recall_macro=recall_score(y_test, y_pred, average="macro", zero_division=0),
        f1_macro=f1_score(y_test, y_pred, average="macro", zero_division=0),
        y_pred=y_pred,
        classification_report_text=classification_report(
            y_test, y_pred, target_names=target_names, zero_division=0
        ),
        confusion_matrix=confusion_matrix(y_test, y_pred),
    )


def train_and_evaluate_all(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    target_names: list[str],
) -> list[ModelResult]:
    results: list[ModelResult] = []
    for name, model in get_model_candidates().items():
        model.fit(X_train, y_train)
        results.append(evaluate_model(name, model, X_test, y_test, target_names))
    return results


def results_to_dataframe(results: list[ModelResult]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Model": r.name,
                "Accuracy": round(r.accuracy, 4),
                "Precision_macro": round(r.precision_macro, 4),
                "Recall_macro": round(r.recall_macro, 4),
                "F1_macro": round(r.f1_macro, 4),
            }
            for r in results
        ]
    ).sort_values("F1_macro", ascending=False)


def select_best_model(results: list[ModelResult]) -> ModelResult:
    return max(results, key=lambda r: r.f1_macro)


def save_models(results: list[ModelResult], models_dir: Path) -> dict[str, Path]:
    models_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for result in results:
        path = models_dir / f"{result.name}.pkl"
        joblib.dump(result.model, path)
        paths[result.name] = path
    return paths


def save_best_model(best: ModelResult, models_dir: Path) -> Path:
    path = models_dir / "best_model.pkl"
    joblib.dump(best.model, path)
    return path


def plot_confusion_matrix(
    cm: np.ndarray,
    target_names: list[str],
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_evaluation_outputs(
    results: list[ModelResult],
    best: ModelResult,
    target_names: list[str],
    evaluation_dir: Path,
) -> dict[str, Path]:
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    comparison_path = evaluation_dir / "model_comparison.csv"
    results_to_dataframe(results).to_csv(comparison_path, index=False)

    report_path = evaluation_dir / "classification_report_best_model.txt"
    report_path.write_text(
        f"Best model: {best.name}\nMacro F1: {best.f1_macro:.4f}\n\n{best.classification_report_text}",
        encoding="utf-8",
    )

    cm_path = evaluation_dir / "confusion_matrix_best_model.png"
    plot_confusion_matrix(
        best.confusion_matrix,
        target_names,
        f"Confusion Matrix — {best.name}",
        cm_path,
    )

    summary_path = evaluation_dir / "evaluation_summary.md"
    comparison_df = results_to_dataframe(results)
    lines = [
        "# Model Evaluation Summary — Smart Study Advisor",
        "",
        "## Models compared",
        "- Logistic Regression",
        "- Decision Tree",
        "- Random Forest",
        "- Gradient Boosting",
        "",
        "## Results table (sorted by macro F1)",
        "",
        "```",
        comparison_df.to_string(index=False),
        "```",
        "",
        f"## Best model: **{best.name}**",
        f"- **Macro F1:** {best.f1_macro:.4f}",
        f"- **Accuracy:** {best.accuracy:.4f}",
        f"- **Precision (macro):** {best.precision_macro:.4f}",
        f"- **Recall (macro):** {best.recall_macro:.4f}",
        "",
        "## Output files",
        f"- `{comparison_path.name}`",
        f"- `{report_path.name}`",
        f"- `{cm_path.name}`",
        "",
        "## Saved models (`models/`)",
    ]
    for result in results:
        lines.append(f"- `{result.name}.pkl`")
    lines.extend(["- `best_model.pkl`", ""])
    summary_path.write_text("\n".join(lines), encoding="utf-8")

    return {
        "comparison": comparison_path,
        "report": report_path,
        "confusion_matrix": cm_path,
        "summary": summary_path,
    }
