"""Preprocessing pipeline for the Smart Study Advisor classifier."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
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


@dataclass
class PreprocessingResult:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    feature_names: list[str]
    preprocessor: ColumnTransformer
    label_encoder: LabelEncoder
    y_train_labels: pd.Series
    y_test_labels: pd.Series


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


def run_preprocessing(
    df: pd.DataFrame | None = None,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> PreprocessingResult:
    df = df if df is not None else load_dataset()

    X = df[FEATURE_COLUMNS].copy()
    y_raw = df[TARGET_COLUMN].copy()

    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X,
        y_raw,
        test_size=test_size,
        random_state=random_state,
        stratify=y_raw,
    )

    preprocessor = build_preprocessor()
    X_train_arr = preprocessor.fit_transform(X_train_raw)
    X_test_arr = preprocessor.transform(X_test_raw)

    feature_names = preprocessor.get_feature_names_out().tolist()
    X_train = pd.DataFrame(X_train_arr, columns=feature_names, index=X_train_raw.index)
    X_test = pd.DataFrame(X_test_arr, columns=feature_names, index=X_test_raw.index)

    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train_raw)
    y_test_enc = label_encoder.transform(y_test_raw)

    y_train = pd.Series(y_train_enc, name=TARGET_COLUMN, index=X_train.index)
    y_test = pd.Series(y_test_enc, name=TARGET_COLUMN, index=X_test.index)

    return PreprocessingResult(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_names,
        preprocessor=preprocessor,
        label_encoder=label_encoder,
        y_train_labels=y_train_raw.reset_index(drop=True),
        y_test_labels=y_test_raw.reset_index(drop=True),
    )


def save_processed_data(result: PreprocessingResult, processed_dir: Path) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    result.X_train.to_csv(processed_dir / "X_train.csv", index=False)
    result.X_test.to_csv(processed_dir / "X_test.csv", index=False)
    result.y_train.to_csv(processed_dir / "y_train.csv", index=False)
    result.y_test.to_csv(processed_dir / "y_test.csv", index=False)


def save_artifacts(result: PreprocessingResult, models_dir: Path) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(result.preprocessor, models_dir / "preprocessor.pkl")
    joblib.dump(result.label_encoder, models_dir / "label_encoder.pkl")


def build_preprocessing_summary(result: PreprocessingResult, source_path: Path) -> str:
    train_dist = result.y_train_labels.value_counts().sort_index()
    test_dist = result.y_test_labels.value_counts().sort_index()
    train_pct = (train_dist / len(result.y_train_labels) * 100).round(1)
    test_pct = (test_dist / len(result.y_test_labels) * 100).round(1)

    class_mapping = {
        str(cls): int(code)
        for cls, code in zip(result.label_encoder.classes_, range(len(result.label_encoder.classes_)))
    }

    encoded_categorical_cols = [
        name for name in result.feature_names if name.startswith("categorical__")
    ]
    scaled_numeric_cols = [
        name for name in result.feature_names if name.startswith("numeric__")
    ]

    lines = [
        "# Preprocessing Summary — Smart Study Advisor",
        "",
        "## 1. Source dataset",
        f"- **Path:** `{source_path}`",
        f"- **Total records:** {len(result.X_train) + len(result.X_test)}",
        "",
        "## 2. Feature separation",
        f"- **X (inputs):** {len(FEATURE_COLUMNS)} features — {', '.join(FEATURE_COLUMNS)}",
        f"- **y (target):** `{TARGET_COLUMN}`",
        "",
        "## 3. Categorical encoding (One-Hot)",
        "",
        "The following columns were one-hot encoded using `OneHotEncoder`:",
        "",
    ]
    for col in CATEGORICAL_FEATURES:
        lines.append(f"- `{col}`")

    lines.extend(
        [
            "",
            f"**Generated columns ({len(encoded_categorical_cols)}):**",
            "",
        ]
    )
    for name in encoded_categorical_cols:
        lines.append(f"- `{name}`")

    lines.extend(
        [
            "",
            "## 4. Numerical scaling (StandardScaler)",
            "",
            "The following columns were standardised (zero mean, unit variance):",
            "",
        ]
    )
    for col in NUMERIC_FEATURES:
        lines.append(f"- `{col}`")

    lines.extend(
        [
            "",
            f"**Scaled column names in output:**",
            "",
        ]
    )
    for name in scaled_numeric_cols:
        lines.append(f"- `{name}`")

    lines.extend(
        [
            "",
            "## 5. Target encoding (LabelEncoder)",
            "",
            "| Class label | Encoded value |",
            "|-------------|---------------|",
        ]
    )
    for cls, code in class_mapping.items():
        lines.append(f"| `{cls}` | {code} |")

    lines.extend(
        [
            "",
            "## 6. Train / test split",
            "",
            "| Setting | Value |",
            "|---------|-------|",
            f"| Train size | {len(result.X_train)} ({100 - TEST_SIZE * 100:.0f}%) |",
            f"| Test size | {len(result.X_test)} ({TEST_SIZE * 100:.0f}%) |",
            f"| Stratified | Yes |",
            f"| random_state | {RANDOM_STATE} |",
            "",
            "**Why stratified 70/30?**",
            "- Preserves the 25% class balance in both train and test sets.",
            "- 70% provides enough data for training while 30% gives a reliable hold-out evaluation set.",
            "- Stratification prevents a class from being under-represented in either split.",
            "",
            "## 7. Final class distribution",
            "",
            "### Training set",
            "```",
            pd.DataFrame({"Count": train_dist, "Percent": train_pct}).to_string(),
            "```",
            "",
            "### Test set",
            "```",
            pd.DataFrame({"Count": test_dist, "Percent": test_pct}).to_string(),
            "```",
            "",
            "## 8. Output files",
            "",
            "**Processed data** (`ml/data/processed/`):",
            "- `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`",
            "",
            "**Artifacts** (`models/`):",
            "- `preprocessor.pkl` — fitted ColumnTransformer (scaler + one-hot encoder)",
            "- `label_encoder.pkl` — fitted LabelEncoder for the target",
            "",
            "## 9. Next step",
            "Train and evaluate a baseline classifier (Step 5).",
            "",
        ]
    )
    return "\n".join(lines)
