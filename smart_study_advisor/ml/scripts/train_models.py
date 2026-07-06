"""Train and evaluate Smart Study Advisor classification models."""

from __future__ import annotations

import sys
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ML_ROOT.parent
sys.path.insert(0, str(ML_ROOT))

import joblib

from src.training.train import (
    load_processed_data,
    results_to_dataframe,
    save_best_model,
    save_evaluation_outputs,
    save_models,
    select_best_model,
    train_and_evaluate_all,
)


def main() -> None:
    processed_dir = ML_ROOT / "data" / "processed"
    models_dir = PROJECT_ROOT / "models"
    evaluation_dir = ML_ROOT / "output" / "evaluation"

    X_train, X_test, y_train, y_test = load_processed_data(processed_dir)

    label_encoder = joblib.load(models_dir / "label_encoder.pkl")
    target_names = list(label_encoder.classes_)

    print("Training models on processed data")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test:  {X_test.shape}")
    print()

    results = train_and_evaluate_all(X_train, X_test, y_train, y_test, target_names)

    print("=" * 60)
    for result in results:
        print(f"Model: {result.name}")
        print(f"  Accuracy:         {result.accuracy:.4f}")
        print(f"  Precision (macro): {result.precision_macro:.4f}")
        print(f"  Recall (macro):    {result.recall_macro:.4f}")
        print(f"  F1 (macro):        {result.f1_macro:.4f}")
        print()
        print("  Classification report:")
        for line in result.classification_report_text.splitlines():
            print(f"    {line}")
        print()
        print(f"  Confusion matrix:\n{result.confusion_matrix}")
        print("=" * 60)

    comparison_df = results_to_dataframe(results)
    print("\nModel comparison (sorted by macro F1):")
    print(comparison_df.to_string(index=False))
    print()

    best = select_best_model(results)
    print(f"Best model: {best.name}")
    print(f"Best macro F1: {best.f1_macro:.4f}")
    print()

    model_paths = save_models(results, models_dir)
    best_path = save_best_model(best, models_dir)
    eval_paths = save_evaluation_outputs(results, best, target_names, evaluation_dir)

    print("Saved models:")
    for name, path in model_paths.items():
        print(f"  - {path}")
    print(f"  - {best_path}  (best)")
    print()
    print("Saved evaluation outputs:")
    for path in eval_paths.values():
        print(f"  - {path}")


if __name__ == "__main__":
    main()
