"""Think phase — load saved ML artifacts and predict RecommendedStrategy."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from agent.types import PredictionResult, StudentContext

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ML_ROOT = PROJECT_ROOT / "ml"
sys.path.insert(0, str(ML_ROOT))

from src.data.schema import FEATURE_COLUMNS  # noqa: E402

REQUIRED_ARTIFACTS = ("preprocessor.pkl", "label_encoder.pkl", "best_model.pkl")


class PredictionService:
    """Loads trained sklearn artifacts and runs inference (no retraining)."""

    def __init__(self, models_dir: Path | None = None) -> None:
        self.models_dir = (models_dir or PROJECT_ROOT / "models").resolve()
        self._preprocessor = None
        self._label_encoder = None
        self._model = None
        self._validate_artifact_paths()

    def _validate_artifact_paths(self) -> None:
        missing = [name for name in REQUIRED_ARTIFACTS if not (self.models_dir / name).exists()]
        if missing:
            msg = f"Missing model artifact(s) in {self.models_dir}: {', '.join(missing)}"
            logger.error(msg)
            raise FileNotFoundError(msg)
        logger.info("Think: model artifacts found in %s", self.models_dir)

    def _load_artifacts(self) -> None:
        if self._model is not None:
            return
        logger.info("Think: loading preprocessor, label encoder, and best model")
        self._preprocessor = joblib.load(self.models_dir / "preprocessor.pkl")
        self._label_encoder = joblib.load(self.models_dir / "label_encoder.pkl")
        self._model = joblib.load(self.models_dir / "best_model.pkl")

    def predict(self, context: StudentContext) -> PredictionResult:
        """Apply the saved preprocessing pipeline and predict strategy with confidence."""
        self._load_artifacts()
        assert self._preprocessor is not None
        assert self._label_encoder is not None
        assert self._model is not None

        record = context.to_feature_dict()
        frame = pd.DataFrame([record], columns=FEATURE_COLUMNS)
        features = self._preprocessor.transform(frame)
        feature_names = self._preprocessor.get_feature_names_out()
        features_df = pd.DataFrame(features, columns=feature_names)
        probabilities = self._model.predict_proba(features_df)[0]
        predicted_index = int(np.argmax(probabilities))
        strategy = str(self._label_encoder.inverse_transform([predicted_index])[0])
        confidence = float(probabilities[predicted_index])

        class_labels = [str(label) for label in self._label_encoder.classes_]
        probability_map = {
            label: float(prob)
            for label, prob in zip(class_labels, probabilities, strict=True)
        }

        logger.info(
            "Think: predicted %s (confidence=%.4f)",
            strategy,
            confidence,
        )
        return PredictionResult(
            recommended_strategy=strategy,
            confidence=round(confidence, 4),
            probabilities=probability_map,
        )
