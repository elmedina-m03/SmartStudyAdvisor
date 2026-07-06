"""Validate the Smart Study Advisor dataset against the schema contract."""

from __future__ import annotations

import sys
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_ROOT))

from src.data.schema import (  # noqa: E402
    ALLOWED_VALUES,
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    load_dataset,
    load_schema,
    validate_dataset,
)


def print_section(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def main() -> int:
    schema = load_schema()
    path = schema.source_path

    if not path.exists():
        print(f"FAIL: Dataset not found at {path}")
        return 1

    import pandas as pd

    df = pd.read_csv(path, keep_default_na=False)
    result = validate_dataset(df, schema, strict=False)

    print_section("Dataset Validation Report")
    print(f"Path:   {path}")
    print(f"Status: {'PASS' if result.valid else 'FAIL'}")

    print_section("Shape")
    print(f"Rows:     {result.row_count}")
    print(f"Columns:  {len(df.columns)} (expected {len(schema.all_columns)})")
    print(f"Features: {len(FEATURE_COLUMNS)}")
    print(f"Target:   {TARGET_COLUMN}")

    print_section("Required Columns")
    for col in schema.all_columns:
        present = col in df.columns
        print(f"  [{'OK' if present else 'MISSING'}] {col}")

    print_section("Missing Values")
    if result.missing_cells == 0:
        print("  No missing values in required columns.")
    else:
        print(f"  Total missing cells: {result.missing_cells}")
        print(df[schema.all_columns].isnull().sum().to_string())

    print_section("Duplicate Rows")
    print(f"  Full-row duplicates: {result.duplicate_rows}")

    print_section("Numeric Ranges")
    for col in NUMERIC_FEATURES:
        print(
            f"  {col}: min={df[col].min()}, max={df[col].max()}, "
            f"mean={df[col].mean():.2f}"
        )

    print_section("Categorical Allowed Values")
    for col in CATEGORICAL_FEATURES + [TARGET_COLUMN]:
        allowed = ALLOWED_VALUES[col]
        actual = sorted(df[col].astype(str).unique())
        invalid = sorted(set(actual) - set(allowed))
        status = "OK" if not invalid else "INVALID"
        print(f"  [{status}] {col}: {actual}")
        if invalid:
            print(f"         Invalid: {invalid}")

    print_section("Target Class Distribution")
    expected = result.row_count / len(ALLOWED_VALUES[TARGET_COLUMN])
    for cls in ALLOWED_VALUES[TARGET_COLUMN]:
        count = result.target_distribution.get(cls, 0)
        pct = count / result.row_count * 100 if result.row_count else 0
        print(f"  {cls:<16} {count:>4}  ({pct:5.1f}%)  [expected ~{expected:.0f}]")

    if result.warnings:
        print_section("Warnings")
        for warning in result.warnings:
            print(f"  - {warning}")

    if result.errors:
        print_section("Errors")
        for error in result.errors:
            print(f"  - {error}")
        return 1

    print_section("Strict Re-validation")
    validate_dataset(df, schema, strict=True)
    print("  All checks passed.")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        print()
        print(f"FAIL: {exc}")
        raise SystemExit(1) from exc
