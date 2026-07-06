# Student Strategy ML Pipeline

Offline machine learning pipeline for training and evaluating the study-strategy classifier.

**Authors:** Elmedina Marić, Aldina Kurtović

## Target

Predict `RecommendedStrategy` from student context features:

- HoursStudied
- NumberOfSubjects
- DaysUntilExam
- StressLevel
- FatigueLevel
- SleepHours
- SleepQuality
- PreviousFeedback

## Layout

```
ml/
├── config/          # Feature lists, dataset schema
├── data/
│   ├── raw/         # Original dataset files
│   └── processed/   # Cleaned train/test splits
├── notebooks/       # EDA notebooks
├── src/             # Preprocessing, training, evaluation
├── scripts/         # Runnable pipeline entry points
└── output/          # Figures, reports, metrics
```

Trained artifacts are exported to `../models/`.

## Dataset (Step 2)

| Location | Purpose |
|----------|---------|
| `../datasets/student_study_strategy.csv` | Canonical labeled dataset |
| `data/raw/student_study_strategy.csv` | Copy used by the ML pipeline |
| `config/dataset_schema.yaml` | Column contract (types, allowed values) |
| `config/features.yaml` | Feature ↔ C# domain mapping |

Validate the dataset:

```bash
cd ml
pip install -r requirements.txt
python scripts/validate_dataset.py
```

## EDA (Step 3)

| Artifact | Purpose |
|----------|---------|
| `notebooks/smart_study_eda.ipynb` | Professional EDA notebook |
| `scripts/run_eda.py` | Generates all figures + summary report |
| `output/figures/` | Exported PNG figures |
| `output/eda_summary.md` | Written summary of findings |

Run EDA:

```bash
cd ml
python scripts/run_eda.py
```

## Preprocessing (Step 4)

```bash
cd ml
python scripts/preprocess_dataset.py
```

| Output | Location |
|--------|----------|
| `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv` | `data/processed/` |
| `preprocessor.pkl`, `label_encoder.pkl` | `../models/` |
| `preprocessing_summary.md` | `output/` |

## Model training (Step 5)

```bash
cd ml
python scripts/train_models.py
```

| Output | Location |
|--------|----------|
| Trained models | `../models/*.pkl`, `../models/best_model.pkl` |
| `model_comparison.csv` | `output/evaluation/` |
| `classification_report_best_model.txt` | `output/evaluation/` |
| `confusion_matrix_best_model.png` | `output/evaluation/` |
| `evaluation_summary.md` | `output/evaluation/` |

## Feature importance (Step 6)

```bash
cd ml
python scripts/feature_importance.py
```

| Output | Location |
|--------|----------|
| `feature_importance.csv` | `output/feature_importance/` |
| `feature_importance.png` | `output/feature_importance/` |
| `feature_importance_report.md` | `output/feature_importance/` |

## Ablation study (Step 7)

```bash
cd ml
python scripts/ablation_study.py
```

| Output | Location |
|--------|----------|
| `ablation_results.csv` | `output/ablation/` |
| `ablation_results.md` | `output/ablation/` |
| `ablation_comparison.png` | `output/ablation/` |

## Cross-validation

```bash
cd ml
python scripts/run_cross_validation.py
```

| Output | Location |
|--------|----------|
| `cross_validation_results.csv` | `output/cross_validation/` |
| `cross_validation_summary.md` | `output/cross_validation/` |
| `cross_validation_boxplot.png` | `output/cross_validation/` |
