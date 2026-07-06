"""Dataset schema and validation for the study-strategy classifier."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml

ML_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ML_ROOT / "config"

FEATURE_COLUMNS = [
    "HoursStudied",
    "NumberOfSubjects",
    "DaysUntilExam",
    "StressLevel",
    "FatigueLevel",
    "SleepHours",
    "SleepQuality",
    "PreviousFeedback",
]

TARGET_COLUMN = "RecommendedStrategy"
ALL_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]

NUMERIC_FEATURES = [
    "HoursStudied",
    "NumberOfSubjects",
    "DaysUntilExam",
    "SleepHours",
]

CATEGORICAL_FEATURES = [
    "StressLevel",
    "FatigueLevel",
    "SleepQuality",
    "PreviousFeedback",
]

ALLOWED_VALUES: dict[str, list[str]] = {
    "StressLevel": ["Low", "Medium", "High"],
    "FatigueLevel": ["Low", "Medium", "High"],
    "SleepQuality": ["Poor", "Average", "Good"],
    "PreviousFeedback": ["None", "Positive", "Negative", "Mixed"],
    "RecommendedStrategy": [
        "Rest",
        "BalancedStudy",
        "IntensiveStudy",
        "LongTermPlan",
    ],
}

NUMERIC_RANGES: dict[str, tuple[float, float]] = {
    "HoursStudied": (0, 12),
    "NumberOfSubjects": (1, 10),
    "DaysUntilExam": (0, 60),
    "SleepHours": (0, 12),
}


@dataclass(frozen=True)
class DatasetSchema:
    filename: str
    source_path: Path
    raw_path: Path
    feature_columns: list[str]
    target_column: str
    validation: dict

    @property
    def all_columns(self) -> list[str]:
        return self.feature_columns + [self.target_column]


@dataclass
class ValidationResult:
    valid: bool
    row_count: int
    duplicate_rows: int
    missing_cells: int
    target_distribution: dict[str, int]
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def raise_if_invalid(self) -> None:
        if not self.valid:
            raise ValueError("\n".join(self.errors))


def load_schema(config_path: Path | None = None) -> DatasetSchema:
    config_path = config_path or CONFIG_DIR / "dataset_schema.yaml"
    with config_path.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)

    dataset = config["dataset"]
    ml_root = config_path.parent.parent

    return DatasetSchema(
        filename=dataset["filename"],
        source_path=(ml_root / dataset["source_path"]).resolve(),
        raw_path=(ml_root / dataset["raw_path"]).resolve(),
        feature_columns=FEATURE_COLUMNS,
        target_column=TARGET_COLUMN,
        validation=config.get("validation", {}),
    )


def validate_dataset(
    df: pd.DataFrame,
    schema: DatasetSchema | None = None,
    *,
    strict: bool = True,
) -> ValidationResult:
    """Validate dataframe against the dataset contract. Returns a structured result."""
    schema = schema or load_schema()
    errors: list[str] = []
    warnings: list[str] = []

    missing_cols = [col for col in schema.all_columns if col not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")

    extra_cols = [col for col in df.columns if col not in schema.all_columns]
    if extra_cols:
        errors.append(f"Unexpected columns: {extra_cols}")

    missing_cells = int(df[schema.all_columns].isnull().sum().sum()) if not missing_cols else 0
    if missing_cells > 0:
        errors.append(f"Dataset contains {missing_cells} missing value(s) in required columns")

    if not missing_cols:
        for column in NUMERIC_FEATURES:
            if column not in df.columns:
                continue
            if not pd.api.types.is_numeric_dtype(df[column]):
                errors.append(f"Column '{column}' must be numeric")

            low, high = NUMERIC_RANGES[column]
            out_of_range = df[(df[column] < low) | (df[column] > high)]
            if not out_of_range.empty:
                errors.append(
                    f"Column '{column}' has {len(out_of_range)} value(s) outside [{low}, {high}]"
                )

        for column in CATEGORICAL_FEATURES + [schema.target_column]:
            if column not in df.columns:
                continue
            allowed = ALLOWED_VALUES[column]
            invalid = sorted(set(df[column].astype(str)) - set(allowed))
            if invalid:
                errors.append(f"Column '{column}' has invalid values: {invalid}")

        duplicate_rows = int(df.duplicated(subset=schema.all_columns).sum())
        max_dupes = schema.validation.get("max_duplicate_rows", 0)
        if duplicate_rows > max_dupes:
            errors.append(
                f"Found {duplicate_rows} duplicate row(s) (max allowed: {max_dupes})"
            )
    else:
        duplicate_rows = 0

    target_distribution: dict[str, int] = {}
    if schema.target_column in df.columns:
        counts = df[schema.target_column].value_counts()
        target_distribution = {str(k): int(v) for k, v in counts.items()}

        expected_classes = ALLOWED_VALUES[schema.target_column]
        missing_classes = [cls for cls in expected_classes if cls not in target_distribution]
        if missing_classes:
            errors.append(f"Target missing classes: {missing_classes}")

        row_count = len(df)
        min_rows = schema.validation.get("min_rows", 0)
        max_rows = schema.validation.get("max_rows", 10_000)
        if row_count < min_rows or row_count > max_rows:
            warnings.append(
                f"Row count {row_count} is outside recommended range [{min_rows}, {max_rows}]"
            )

        if row_count > 0 and target_distribution:
            expected = row_count / len(expected_classes)
            tolerance = schema.validation.get("target_balance_tolerance_percent", 15) / 100
            for cls in expected_classes:
                count = target_distribution.get(cls, 0)
                if abs(count - expected) > expected * tolerance:
                    warnings.append(
                        f"Target class '{cls}' count {count} deviates from balanced "
                        f"expectation ~{expected:.0f} (tolerance {tolerance:.0%})"
                    )

    result = ValidationResult(
        valid=len(errors) == 0,
        row_count=len(df),
        duplicate_rows=duplicate_rows if not missing_cols else int(df.duplicated().sum()),
        missing_cells=missing_cells,
        target_distribution=target_distribution,
        errors=errors,
        warnings=warnings,
    )

    if strict and not result.valid:
        result.raise_if_invalid()

    return result


def load_dataset(path: Path | None = None) -> pd.DataFrame:
    """Load and validate the canonical dataset."""
    schema = load_schema()
    path = path or schema.source_path

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    # keep_default_na=False preserves the literal "None" feedback label
    df = pd.read_csv(path, keep_default_na=False)
    validate_dataset(df, schema)
    return df
