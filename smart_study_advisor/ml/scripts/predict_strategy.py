"""Runtime prediction using saved sklearn artifacts (no retraining)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ML_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ML_ROOT.parent
sys.path.insert(0, str(ML_ROOT))

from src.data.schema import FEATURE_COLUMNS  # noqa: E402

REQUIRED_ARTIFACTS = ("preprocessor.pkl", "label_encoder.pkl", "best_model.pkl")


def _models_dir(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).resolve()
    return (REPO_ROOT / "models").resolve()


def _validate_artifacts(models_dir: Path) -> None:
    missing = [name for name in REQUIRED_ARTIFACTS if not (models_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing model artifact(s) in {models_dir}: {', '.join(missing)}"
        )


def _load_artifacts(models_dir: Path) -> tuple[object, object, object]:
    _validate_artifacts(models_dir)
    preprocessor = joblib.load(models_dir / "preprocessor.pkl")
    label_encoder = joblib.load(models_dir / "label_encoder.pkl")
    model = joblib.load(models_dir / "best_model.pkl")
    return preprocessor, label_encoder, model


def predict_record(record: dict, models_dir: Path) -> dict:
    preprocessor, label_encoder, model = _load_artifacts(models_dir)

    row = {column: record[column] for column in FEATURE_COLUMNS}
    frame = pd.DataFrame([row], columns=FEATURE_COLUMNS)

    features = preprocessor.transform(frame)
    probabilities = model.predict_proba(features)[0]
    predicted_index = int(np.argmax(probabilities))
    strategy = str(label_encoder.inverse_transform([predicted_index])[0])
    confidence = float(probabilities[predicted_index])

    class_labels = [str(label) for label in label_encoder.classes_]
    probability_map = {
        label: float(prob)
        for label, prob in zip(class_labels, probabilities, strict=True)
    }

    return {
        "recommended_strategy": strategy,
        "confidence": round(confidence, 4),
        "probabilities": probability_map,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Predict study strategy from student context.")
    parser.add_argument(
        "--models-dir",
        default=None,
        help="Directory containing preprocessor.pkl, label_encoder.pkl, best_model.pkl",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Path to JSON file with feature values (default: read JSON from stdin)",
    )
    args = parser.parse_args()

    models_dir = _models_dir(args.models_dir)

    try:
        if args.input:
            payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
        else:
            payload = json.load(sys.stdin)

        result = predict_record(payload, models_dir)
        json.dump(result, sys.stdout)
        return 0
    except FileNotFoundError as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 2
    except (KeyError, TypeError, ValueError) as exc:
        print(json.dumps({"error": f"Invalid input: {exc}"}), file=sys.stderr)
        return 1
    except Exception as exc:  # pragma: no cover - runtime guard
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
