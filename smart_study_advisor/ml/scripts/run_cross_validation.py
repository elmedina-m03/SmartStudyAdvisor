"""Run 5-fold cross-validation on the Random Forest model."""

from __future__ import annotations

import sys
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_ROOT))

from src.training.cross_validation import run_cross_validation, save_cv_outputs


def main() -> None:
    output_dir = ML_ROOT / "output" / "cross_validation"

    print("Running 5-fold stratified cross-validation (Random Forest)...")
    fold_df, summary_df, train_summary_df = run_cross_validation()
    paths = save_cv_outputs(fold_df, summary_df, train_summary_df, output_dir)

    print()
    print("Per-fold results (out-of-fold test):")
    print(
        fold_df[["fold", "accuracy", "precision_macro", "recall_macro", "f1_macro"]]
        .round(4)
        .to_string(index=False)
    )
    print()
    print("Summary (mean ± std):")
    for _, row in summary_df.iterrows():
        print(f"  {row['metric']:<18} mean={row['mean']:.4f}  std={row['std']:.4f}")
    print()
    print("Saved files:")
    for path in paths.values():
        print(f"  - {path}")


if __name__ == "__main__":
    main()
