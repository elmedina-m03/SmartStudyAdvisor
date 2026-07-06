"""SHAP-based prediction explainability for the Random Forest model."""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from agent.types import PredictionResult, StudentContext
from app.components.paths import MODELS_DIR, ML_ROOT

if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

from src.data.schema import FEATURE_COLUMNS  # noqa: E402

READABLE_LABELS: dict[str, str] = {
    "numeric__HoursStudied": "Sati učenja",
    "numeric__NumberOfSubjects": "Broj predmeta",
    "numeric__DaysUntilExam": "Dana do ispita",
    "numeric__SleepHours": "Sati sna",
    "categorical__StressLevel_Low": "Stres: nizak",
    "categorical__StressLevel_Medium": "Stres: srednji",
    "categorical__StressLevel_High": "Stres: visok",
    "categorical__FatigueLevel_Low": "Umor: nizak",
    "categorical__FatigueLevel_Medium": "Umor: srednji",
    "categorical__FatigueLevel_High": "Umor: visok",
    "categorical__SleepQuality_Poor": "Kvalitet sna: loš",
    "categorical__SleepQuality_Average": "Kvalitet sna: prosječan",
    "categorical__SleepQuality_Good": "Kvalitet sna: dobar",
    "categorical__PreviousFeedback_None": "Prethodna povratna informacija: nema",
    "categorical__PreviousFeedback_Positive": "Prethodna povratna informacija: pozitivna",
    "categorical__PreviousFeedback_Negative": "Prethodna povratna informacija: negativna",
    "categorical__PreviousFeedback_Mixed": "Prethodna povratna informacija: miješana",
}


@lru_cache(maxsize=1)
def _load_artifacts() -> tuple[object, object, object]:
    preprocessor = joblib.load(MODELS_DIR / "preprocessor.pkl")
    label_encoder = joblib.load(MODELS_DIR / "label_encoder.pkl")
    model = joblib.load(MODELS_DIR / "best_model.pkl")
    return preprocessor, label_encoder, model


def _readable_name(feature_name: str) -> str:
    return READABLE_LABELS.get(feature_name, feature_name.replace("numeric__", "").replace("categorical__", ""))


@lru_cache(maxsize=1)
def _load_shap_explainer():
    import shap

    _, _, model = _load_artifacts()
    return shap.TreeExplainer(model)


def compute_shap_contributions(
    context: StudentContext,
    prediction: PredictionResult,
    *,
    top_n: int = 8,
) -> pd.DataFrame:
    """
    Return top SHAP contributions for the predicted class.

    Uses TreeExplainer on the trained Random Forest with preprocessed features.
    """
    import shap

    preprocessor, label_encoder, model = _load_artifacts()
    frame = pd.DataFrame([context.to_feature_dict()], columns=FEATURE_COLUMNS)
    features_arr = preprocessor.transform(frame)
    feature_names = list(preprocessor.get_feature_names_out())
    features_df = pd.DataFrame(features_arr, columns=feature_names)

    classes = list(label_encoder.classes_)
    predicted_index = classes.index(prediction.recommended_strategy)

    explainer = _load_shap_explainer()
    shap_values = explainer.shap_values(features_df)

    shap_array = np.asarray(shap_values)
    if shap_array.ndim == 3:
        class_shap = shap_array[0, :, predicted_index].flatten()
    elif isinstance(shap_values, list):
        class_shap = np.asarray(shap_values[predicted_index][0]).flatten()
    else:
        class_shap = np.asarray(shap_values[0]).flatten()

    rows = []
    for name, value in zip(feature_names, class_shap, strict=True):
        shap_val = float(value)
        rows.append(
            {
                "Feature": _readable_name(name),
                "Značajka": _readable_name(name),
                "Doprinos": shap_val,
                "SHAP": shap_val,
                "Utjecaj": "Povećava" if shap_val > 0 else "Smanjuje",
                "Apsolutno": abs(shap_val),
            }
        )

    df = pd.DataFrame(rows).sort_values("Apsolutno", ascending=False).head(top_n)
    return df.drop(columns=["Apsolutno", "Feature", "SHAP"])

def shap_available() -> bool:
    try:
        import shap  # noqa: F401

        return (MODELS_DIR / "best_model.pkl").exists()
    except ImportError:
        return False
