"""Analyse feature importance for the Smart Study Advisor best model."""

from __future__ import annotations

import sys
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ML_ROOT.parent
sys.path.insert(0, str(ML_ROOT))

from src.training.feature_importance import run_feature_importance_analysis


def main() -> None:
    models_dir = PROJECT_ROOT / "models"
    processed_dir = ML_ROOT / "data" / "processed"
    output_dir = ML_ROOT / "output" / "feature_importance"

    importance_df, paths, method, model_type = run_feature_importance_analysis(
        models_dir, processed_dir, output_dir
    )

    print("Feature importance analysis complete")
    print(f"  Best model type: {model_type}")
    print(f"  Importance method: {method}")
    print()
    print("Top 10 most important features:")
    top10 = importance_df.head(10)[["Rank", "ReadableLabel", "Importance"]]
    for _, row in top10.iterrows():
        print(f"  {int(row['Rank']):>2}. {row['ReadableLabel']:<35} {row['Importance']:.4f}")
    print()
    print("Saved files:")
    for path in paths.values():
        print(f"  - {path}")


if __name__ == "__main__":
    main()
