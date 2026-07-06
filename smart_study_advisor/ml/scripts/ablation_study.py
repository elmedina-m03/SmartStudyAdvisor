"""Run ablation study for the Smart Study Advisor classifier."""

from __future__ import annotations

import sys
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_ROOT))

from src.training.ablation import run_ablation_study, save_ablation_outputs, _feature_usefulness_rank


def main() -> None:
    output_dir = ML_ROOT / "output" / "ablation"

    print("Running ablation study (temporary models only)...")
    results_df = run_ablation_study()
    paths = save_ablation_outputs(results_df, output_dir)
    usefulness = _feature_usefulness_rank(results_df)

    baseline = results_df[results_df["experiment"] == "A_baseline"].iloc[0]
    print(f"\nBaseline F1 (macro): {baseline['f1_macro']:.4f}")
    print("\nExperiment results:")
    print(results_df[["experiment", "removed_feature", "f1_macro", "delta_f1"]].round(4).to_string(index=False))
    print("\nFeature usefulness ranking (by F1 drop):")
    print(usefulness.round(4).to_string(index=False))
    print("\nSaved files:")
    for key, path in paths.items():
        if key != "usefulness":
            print(f"  - {path}")


if __name__ == "__main__":
    main()
