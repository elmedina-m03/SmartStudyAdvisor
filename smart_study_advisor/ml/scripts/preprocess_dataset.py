"""Preprocess the Smart Study Advisor dataset for classification."""

from __future__ import annotations

import sys
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ML_ROOT.parent
sys.path.insert(0, str(ML_ROOT))

from src.data.schema import TARGET_COLUMN, load_dataset, load_schema
from src.preprocessing.pipeline import (
    build_preprocessing_summary,
    run_preprocessing,
    save_artifacts,
    save_processed_data,
)


def main() -> None:
    schema = load_schema()
    df = load_dataset()

    processed_dir = ML_ROOT / "data" / "processed"
    models_dir = PROJECT_ROOT / "models"
    output_dir = ML_ROOT / "output"

    result = run_preprocessing(df)
    save_processed_data(result, processed_dir)
    save_artifacts(result, models_dir)

    summary = build_preprocessing_summary(result, schema.source_path)
    report_path = output_dir / "preprocessing_summary.md"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text(summary, encoding="utf-8")

    print("Preprocessing complete")
    print(f"  Processed data: {processed_dir}")
    print(f"  Artifacts:      {models_dir}")
    print(f"  Report:         {report_path}")
    print()
    print(f"X_train shape: {result.X_train.shape}")
    print(f"X_test shape:  {result.X_test.shape}")
    print()
    print("y_train distribution (encoded):")
    print(result.y_train.value_counts().sort_index().to_string())
    print()
    print("y_test distribution (encoded):")
    print(result.y_test.value_counts().sort_index().to_string())
    print()
    print("y_train distribution (labels):")
    print(result.y_train_labels.value_counts().sort_index().to_string())
    print()
    print("y_test distribution (labels):")
    print(result.y_test_labels.value_counts().sort_index().to_string())
    print()
    print(f"Encoded feature names ({len(result.feature_names)}):")
    for name in result.feature_names:
        print(f"  - {name}")
    print()
    print(f"Target classes ({TARGET_COLUMN}):")
    for i, cls in enumerate(result.label_encoder.classes_):
        print(f"  {cls}: {i}")


if __name__ == "__main__":
    main()
