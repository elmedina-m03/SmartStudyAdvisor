"""Run comprehensive ML experiments (split ratios, redundancy, outliers)."""

from __future__ import annotations

import sys
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_ROOT))

from src.data.schema import load_dataset
from src.training.ml_experiments import run_all_experiments, save_experiment_outputs


def main() -> None:
    df = load_dataset()
    results = run_all_experiments(df)
    output_dir = ML_ROOT / "output" / "experiments"
    paths = save_experiment_outputs(results, output_dir, df)

    print("ML experiments complete")
    print(f"  CSV: {paths['csv']}")
    print(f"  Report: {paths['md']}")
    print(f"  Chart: {paths['png']}")
    print()
    print(results[["experiment_id", "f1_macro", "accuracy"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
