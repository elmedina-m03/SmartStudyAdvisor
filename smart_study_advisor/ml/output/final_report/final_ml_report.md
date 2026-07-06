# Smart Study Advisor — Final Machine Learning Report

**Project:** SleepQualityAgent → Smart Study Advisor ML Pipeline  
**Task:** Multi-class classification of study strategy recommendations  
**Date:** July 2026  
**Pipeline version:** Steps 1–8 (dataset → cross-validation)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Problem Definition](#2-problem-definition)
3. [Dataset Description](#3-dataset-description)
4. [Feature Descriptions](#4-feature-descriptions)
5. [Dataset Validation](#5-dataset-validation)
6. [Exploratory Data Analysis](#6-exploratory-data-analysis)
7. [Correlation Analysis](#7-correlation-analysis)
8. [Outlier Analysis](#8-outlier-analysis)
9. [Preprocessing Pipeline](#9-preprocessing-pipeline)
10. [Model Comparison](#10-model-comparison)
11. [Best Model Selection](#11-best-model-selection)
12. [Feature Importance Analysis](#12-feature-importance-analysis)
13. [Ablation Study](#13-ablation-study)
14. [Cross-Validation Results](#14-cross-validation-results)
15. [Discussion of Strengths and Limitations](#15-discussion-of-strengths-and-limitations)
16. [Future Improvements](#16-future-improvements)
17. [Final Conclusion](#17-final-conclusion)

---

## 1. Project Overview

This report summarises the complete machine learning pipeline developed for the **Smart Study Advisor**, a component of the broader agent-based software project (SleepQualityAgent / StudentAgent). The goal is to recommend an appropriate study strategy—*Rest*, *BalancedStudy*, *IntensiveStudy*, or *LongTermPlan*—based on a student's current academic and wellbeing context.

The pipeline was implemented in Python under `ml/` and follows a standard supervised learning workflow:

| Step | Activity | Primary outputs |
|------|----------|-----------------|
| 1–2 | Domain schema & dataset creation | `datasets/student_study_strategy.csv`, `ml/config/` |
| 3 | Exploratory Data Analysis (EDA) | `ml/output/figures/`, `ml/output/eda_summary.md` |
| 4 | Preprocessing | `ml/data/processed/`, `models/preprocessor.pkl`, `models/label_encoder.pkl` |
| 5 | Model training & evaluation | `models/best_model.pkl`, `ml/output/evaluation/` |
| 6 | Feature importance | `ml/output/feature_importance/` |
| 7 | Ablation study | `ml/output/ablation/` |
| 8 | 5-fold cross-validation | `ml/output/cross_validation/` |

The trained **Random Forest** classifier achieves macro F1 ≈ 0.99 on hold-out data and mean out-of-fold F1 ≈ 0.98 in cross-validation, confirming stable performance on the current dataset. The model is intended for eventual integration into a C# agent via ONNX or a prediction service (not covered in this report).

---

## 2. Problem Definition

### 2.1 Task formulation

- **Type:** Supervised multi-class classification  
- **Input:** Eight features describing a student's study load, exam proximity, stress, fatigue, sleep, and prior feedback  
- **Output:** One of four study strategies (`RecommendedStrategy`)

### 2.2 Strategy classes

| Class | Description |
|-------|-------------|
| **Rest** | Recovery-focused; recommended when fatigue/stress is high and exam is imminent |
| **BalancedStudy** | Moderate, sustainable study pace across several subjects |
| **IntensiveStudy** | High-intensity cramming when the exam is very close |
| **LongTermPlan** | Structured planning when the exam is far away and stress is low |

### 2.3 Evaluation criteria

Models were compared using **macro-averaged** precision, recall, and F1-score to treat all four classes equally despite balanced sampling. Accuracy is reported for completeness. The primary selection metric was **macro F1**.

### 2.4 Design constraints

- Feature names and categorical values align with the C# domain model (`Student.Domain/`) for future deployment.
- Preprocessing artifacts and the best model are serialised to `models/` for reproducibility.
- No hyperparameter tuning was performed; sklearn defaults were used consistently across training, ablation, and cross-validation.

---

## 3. Dataset Description

### 3.1 Source and size

| Property | Value |
|----------|-------|
| Canonical path | `datasets/student_study_strategy.csv` |
| ML pipeline copy | `ml/data/raw/student_study_strategy.csv` |
| Records | **400** |
| Features | **8** (4 numeric, 4 categorical) |
| Target | `RecommendedStrategy` |
| Generation method | Rule-based synthetic generation (`ml/scripts/generate_dataset.py`, seed = 42) |
| Class balance | **100 samples per class** (25% each) |

Each strategy class was generated from class-specific feature distributions (e.g., Rest samples have high stress/fatigue, low sleep hours, and exams within 0–3 days). See Section 15 for implications of synthetic data.

### 3.2 Descriptive statistics (numeric features)

| Feature | Mean | Median | Std | Min | Max |
|---------|------|--------|-----|-----|-----|
| HoursStudied | 4.26 | 4.0 | 2.70 | 0.0 | 10.0 |
| NumberOfSubjects | 4.62 | 5.0 | 2.05 | 1 | 9 |
| DaysUntilExam | 8.52 | 4.0 | 8.75 | 0 | 30 |
| SleepHours | 6.44 | 6.5 | 1.39 | 3.5 | 8.9 |

**Reference:** `ml/output/eda_summary.md`

### 3.3 Mean numeric features by strategy

| Strategy | HoursStudied | NumberOfSubjects | DaysUntilExam | SleepHours |
|----------|-------------|------------------|---------------|------------|
| Rest | 1.27 | 2.80 | 1.45 | 4.58 |
| BalancedStudy | 4.88 | 4.96 | 8.13 | 6.96 |
| IntensiveStudy | 7.91 | 6.71 | 2.60 | 6.22 |
| LongTermPlan | 2.98 | 3.99 | 21.91 | 8.00 |

These group means confirm that the generation rules produce clearly separable class profiles in feature space.

---

## 4. Feature Descriptions

All features map directly to `StudentContext` properties in the C# domain layer. Definitions are stored in `ml/config/features.yaml` and `ml/config/dataset_schema.yaml`.

| Feature | Type | Allowed values / range | Role |
|---------|------|------------------------|------|
| **HoursStudied** | Numeric (float) | 0–12 | Daily study hours |
| **NumberOfSubjects** | Numeric (int) | 1–10 | Active subject count |
| **DaysUntilExam** | Numeric (int) | 0–60 | Days remaining until next exam |
| **StressLevel** | Categorical | Low, Medium, High | Self-reported stress |
| **FatigueLevel** | Categorical | Low, Medium, High | Self-reported fatigue |
| **SleepHours** | Numeric (float) | 0–12 | Average nightly sleep |
| **SleepQuality** | Categorical | Poor, Average, Good | Subjective sleep quality |
| **PreviousFeedback** | Categorical | None, Positive, Negative, Mixed | Feedback on prior strategy |

**Target:** `RecommendedStrategy` ∈ {Rest, BalancedStudy, IntensiveStudy, LongTermPlan}

---

## 5. Dataset Validation

Before any analysis, the dataset was validated against the schema contract using `ml/scripts/validate_dataset.py` and `src/data/schema.py`.

### 5.1 Validation checks

| Check | Result |
|-------|--------|
| Required columns present | Pass (9/9) |
| Row count within bounds (350–450) | Pass (400) |
| Missing values | **0** |
| Duplicate rows | **0** |
| Invalid categorical values | **0** |
| Numeric range violations | Pass |
| Target class balance (±15%) | Pass (25% per class) |

### 5.2 Data quality summary

The dataset passed all validation rules defined in `ml/config/dataset_schema.yaml`. The loader uses `keep_default_na=False` to prevent the string `"None"` in `PreviousFeedback` from being interpreted as a missing value.

**Figures:**  
- Missing values: [`../figures/01_missing_values.png`](../figures/01_missing_values.png)  
- Target distribution: [`../figures/02_target_distribution.png`](../figures/02_target_distribution.png)

**Scripts & config:**  
- `ml/scripts/validate_dataset.py`  
- `ml/config/dataset_schema.yaml`

---

## 6. Exploratory Data Analysis

EDA was performed via `ml/scripts/run_eda.py` and documented in `ml/notebooks/smart_study_eda.ipynb`. A summary report was exported to `ml/output/eda_summary.md`.

### 6.1 Key findings

1. **Perfect data quality** — no missing values, duplicates, or invalid categories.
2. **Balanced target** — exactly 100 samples per strategy class.
3. **Clear class separation** — numeric features show distinct means per strategy (see Section 3.3).
4. **Categorical predictors** — high stress/fatigue strongly associates with Rest; low stress/fatigue with LongTermPlan.
5. **DaysUntilExam** — strongest numeric separator between urgent strategies (Rest, IntensiveStudy) and LongTermPlan.

### 6.2 Distribution figures (numeric)

| Figure | Description |
|--------|-------------|
| [`../figures/03_hist_HoursStudied.png`](../figures/03_hist_HoursStudied.png) | Histogram — HoursStudied |
| [`../figures/03_hist_NumberOfSubjects.png`](../figures/03_hist_NumberOfSubjects.png) | Histogram — NumberOfSubjects |
| [`../figures/03_hist_DaysUntilExam.png`](../figures/03_hist_DaysUntilExam.png) | Histogram — DaysUntilExam |
| [`../figures/03_hist_SleepHours.png`](../figures/03_hist_SleepHours.png) | Histogram — SleepHours |

### 6.3 Box plots by strategy (numeric)

| Figure | Description |
|--------|-------------|
| [`../figures/04_box_HoursStudied.png`](../figures/04_box_HoursStudied.png) | Box plot — HoursStudied vs strategy |
| [`../figures/04_box_NumberOfSubjects.png`](../figures/04_box_NumberOfSubjects.png) | Box plot — NumberOfSubjects vs strategy |
| [`../figures/04_box_DaysUntilExam.png`](../figures/04_box_DaysUntilExam.png) | Box plot — DaysUntilExam vs strategy |
| [`../figures/04_box_SleepHours.png`](../figures/04_box_SleepHours.png) | Box plot — SleepHours vs strategy |

### 6.4 Categorical frequency counts

| Figure | Description |
|--------|-------------|
| [`../figures/05_count_StressLevel.png`](../figures/05_count_StressLevel.png) | StressLevel distribution |
| [`../figures/05_count_FatigueLevel.png`](../figures/05_count_FatigueLevel.png) | FatigueLevel distribution |
| [`../figures/05_count_SleepQuality.png`](../figures/05_count_SleepQuality.png) | SleepQuality distribution |
| [`../figures/05_count_PreviousFeedback.png`](../figures/05_count_PreviousFeedback.png) | PreviousFeedback distribution |

### 6.5 Categorical features vs target

| Figure | Description |
|--------|-------------|
| [`../figures/06_StressLevel_vs_strategy.png`](../figures/06_StressLevel_vs_strategy.png) | StressLevel × RecommendedStrategy |
| [`../figures/06_FatigueLevel_vs_strategy.png`](../figures/06_FatigueLevel_vs_strategy.png) | FatigueLevel × RecommendedStrategy |
| [`../figures/06_SleepQuality_vs_strategy.png`](../figures/06_SleepQuality_vs_strategy.png) | SleepQuality × RecommendedStrategy |
| [`../figures/06_PreviousFeedback_vs_strategy.png`](../figures/06_PreviousFeedback_vs_strategy.png) | PreviousFeedback × RecommendedStrategy |
| [`../figures/09_sleep_quality_vs_strategy.png`](../figures/09_sleep_quality_vs_strategy.png) | Sleep quality detail vs strategy |
| [`../figures/10_stress_level_vs_strategy.png`](../figures/10_stress_level_vs_strategy.png) | Stress level detail vs strategy |
| [`../figures/11_fatigue_level_vs_strategy.png`](../figures/11_fatigue_level_vs_strategy.png) | Fatigue level detail vs strategy |
| [`../figures/12_days_until_exam_vs_strategy.png`](../figures/12_days_until_exam_vs_strategy.png) | DaysUntilExam detail vs strategy |
| [`../figures/13_sleep_hours_vs_strategy.png`](../figures/13_sleep_hours_vs_strategy.png) | SleepHours detail vs strategy |

**Full EDA report:** [`../eda_summary.md`](../eda_summary.md)

---

## 7. Correlation Analysis

Pearson correlations were computed on numeric features plus ordinal-encoded categoricals (Low = 0, Medium = 1, High = 2) for exploratory purposes only. One-hot encoding was used in modelling.

**Figure:** [`../figures/07_correlation_matrix_encoded.png`](../figures/07_correlation_matrix_encoded.png)

### 7.1 Strongest positive correlations

| Pair | r |
|------|---|
| DaysUntilExam ↔ SleepHours | **0.748** |
| HoursStudied ↔ NumberOfSubjects | **0.642** |
| SleepHours ↔ SleepQuality | **0.627** |
| DaysUntilExam ↔ SleepQuality | **0.514** |
| StressLevel ↔ FatigueLevel | **0.332** |

### 7.2 Strongest negative correlations

| Pair | r |
|------|---|
| DaysUntilExam ↔ StressLevel | **-0.581** |
| StressLevel ↔ SleepHours | **-0.527** |
| FatigueLevel ↔ SleepHours | **-0.521** |
| DaysUntilExam ↔ FatigueLevel | **-0.432** |
| FatigueLevel ↔ SleepQuality | **-0.368** |

### 7.3 Interpretation

- **Exam proximity** correlates with lower stress/fatigue encoding and higher sleep hours — consistent with LongTermPlan profiles.
- **Study load** (HoursStudied, NumberOfSubjects) moves together, supporting IntensiveStudy and BalancedStudy.
- **SleepHours ↔ SleepQuality** redundancy (r = 0.627) later explains why ablation found SleepHours removable without performance loss (Section 13).

---

## 8. Outlier Analysis

Outliers were detected using the **IQR method** (1.5 × IQR below Q1 or above Q3) on numeric features only.

**Figure:** [`../figures/08_iqr_outlier_counts.png`](../figures/08_iqr_outlier_counts.png)

| Feature | Q1 | Q3 | IQR | Outlier count | Outlier % |
|---------|----|----|-----|---------------|-----------|
| HoursStudied | 1.875 | 6.3 | 4.425 | 0 | 0.0% |
| NumberOfSubjects | 3.0 | 6.0 | 3.0 | 0 | 0.0% |
| DaysUntilExam | 2.0 | 14.0 | 12.0 | 0 | 0.0% |
| SleepHours | 5.5 | 7.5 | 2.0 | 0 | 0.0% |

**Total rows flagged:** 0 (0.0%)

**Decision:** No outlier removal was applied. All values fall within the IQR bounds because the synthetic generator constrains feature ranges. In a real-world dataset, extreme but valid student states should be retained rather than discarded.

---

## 9. Preprocessing Pipeline

Preprocessing was implemented in `ml/src/preprocessing/pipeline.py` and executed via `ml/scripts/preprocess_dataset.py`. A detailed summary is in [`../preprocessing_summary.md`](../preprocessing_summary.md).

### 9.1 Pipeline steps

```
Raw CSV (400 rows)
    │
    ├─ Feature/target separation (8 features → X, RecommendedStrategy → y)
    │
    ├─ Train/test split (70/30, stratified, random_state=42)
    │     Train: 280 rows │ Test: 120 rows
    │
    ├─ Numeric features → StandardScaler (zero mean, unit variance)
    │     HoursStudied, NumberOfSubjects, DaysUntilExam, SleepHours
    │
    ├─ Categorical features → OneHotEncoder (handle_unknown="ignore")
    │     StressLevel, FatigueLevel, SleepQuality, PreviousFeedback
    │     → 13 binary columns
    │
    └─ Target → LabelEncoder
          BalancedStudy=0, IntensiveStudy=1, LongTermPlan=2, Rest=3
```

### 9.2 Output dimensions

| Artifact | Shape / count |
|----------|---------------|
| `X_train.csv` | 280 × 17 |
| `X_test.csv` | 120 × 17 |
| Encoded feature columns | 17 (4 scaled numeric + 13 one-hot) |

### 9.3 Saved artifacts

| File | Location |
|------|----------|
| `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv` | `ml/data/processed/` |
| `preprocessor.pkl` | `models/preprocessor.pkl` |
| `label_encoder.pkl` | `models/label_encoder.pkl` |

### 9.4 Split rationale

A **stratified 70/30 split** preserves the 25% class balance in both partitions. `random_state=42` ensures reproducibility across preprocessing, training, ablation, and reporting.

---

## 10. Model Comparison

Four classifiers were trained and evaluated on the preprocessed hold-out test set using `ml/scripts/train_models.py`.

**Results file:** [`../evaluation/model_comparison.csv`](../evaluation/model_comparison.csv)  
**Summary:** [`../evaluation/evaluation_summary.md`](../evaluation/evaluation_summary.md)

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|-------|----------|-------------------|----------------|------------|
| **Random Forest** | **0.9917** | **0.9919** | **0.9917** | **0.9917** |
| Logistic Regression | 0.9833 | 0.9836 | 0.9833 | 0.9833 |
| Decision Tree | 0.9750 | 0.9773 | 0.9750 | 0.9749 |
| Gradient Boosting | 0.9667 | 0.9706 | 0.9667 | 0.9665 |

All models used sklearn default hyperparameters with `random_state=42` where applicable. No grid search or tuning was performed.

**Confusion matrix (best model):** [`../evaluation/confusion_matrix_best_model.png`](../evaluation/confusion_matrix_best_model.png)  
**Classification report:** [`../evaluation/classification_report_best_model.txt`](../evaluation/classification_report_best_model.txt)

---

## 11. Best Model Selection

### 11.1 Selected model

**Random Forest Classifier** (`RandomForestClassifier`, `random_state=42`, sklearn defaults)

| Hyperparameter | Value |
|----------------|-------|
| n_estimators | 100 |
| max_depth | None |
| max_features | sqrt |
| criterion | gini |

### 11.2 Hold-out test performance

| Metric | Value |
|--------|-------|
| Accuracy | 0.9917 |
| Precision (macro) | 0.9919 |
| Recall (macro) | 0.9917 |
| F1 (macro) | **0.9917** |

### 11.3 Per-class performance

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| BalancedStudy | 0.97 | 1.00 | 0.98 | 30 |
| IntensiveStudy | 1.00 | 0.97 | 0.98 | 30 |
| LongTermPlan | 1.00 | 1.00 | 1.00 | 30 |
| Rest | 1.00 | 1.00 | 1.00 | 30 |

Only one test misclassification occurred (IntensiveStudy confused with BalancedStudy).

### 11.4 Selection rationale

Random Forest achieved the highest macro F1 (0.9917), outperforming Logistic Regression by 0.0084 F1 points. Its ensemble structure handles non-linear interactions between numeric and one-hot features without manual feature engineering. The model was saved as `models/best_model.pkl` alongside per-algorithm pickles in `models/`.

---

## 12. Feature Importance Analysis

Feature importance was computed on the saved best model using **Gini importance** (mean decrease in impurity) via `ml/scripts/feature_importance.py`.

**Outputs:**  
- [`../feature_importance/feature_importance.csv`](../feature_importance/feature_importance.csv)  
- [`../feature_importance/feature_importance.png`](../feature_importance/feature_importance.png)  
- [`../feature_importance/feature_importance_report.md`](../feature_importance/feature_importance_report.md)

### 12.1 Top 10 encoded features

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | DaysUntilExam (scaled) | 0.3390 |
| 2 | HoursStudied (scaled) | 0.2435 |
| 3 | SleepHours (scaled) | 0.1784 |
| 4 | NumberOfSubjects (scaled) | 0.0657 |
| 5 | FatigueLevel = High | 0.0356 |
| 6 | PreviousFeedback = Negative | 0.0343 |
| 7 | StressLevel = High | 0.0172 |
| 8 | StressLevel = Low | 0.0168 |
| 9 | FatigueLevel = Low | 0.0163 |
| 10 | SleepQuality = Poor | 0.0133 |

### 12.2 Aggregated importance by base feature

| Rank | Base feature | Total importance |
|------|-------------|------------------|
| 1 | DaysUntilExam | 0.339 |
| 2 | HoursStudied | 0.244 |
| 3 | SleepHours | 0.178 |
| 4 | NumberOfSubjects | 0.066 |
| 5 | FatigueLevel | 0.055 |
| 6 | PreviousFeedback | 0.050 |
| 7 | StressLevel | 0.039 |
| 8 | SleepQuality | 0.030 |

### 12.3 Domain interpretation

- **DaysUntilExam** separates urgent strategies from long-term planning.
- **HoursStudied / NumberOfSubjects** distinguish intensive from balanced study.
- **FatigueLevel = High** is the strongest categorical signal, aligning with Rest recommendations.
- **SleepQuality = Poor** contributes more than Average or Good levels.
- **StressLevel = High** and **Low** matter more than Medium (which has the lowest split importance).

---

## 13. Ablation Study

An ablation study measured the marginal contribution of each feature group by retraining Random Forest models with one feature removed at a time. All ablation models were **temporary**; `models/best_model.pkl` was not modified.

**Script:** `ml/scripts/ablation_study.py`  
**Split:** 70/30 stratified, `random_state=42` (same as preprocessing)

**Outputs:**  
- [`../ablation/ablation_results.csv`](../ablation/ablation_results.csv)  
- [`../ablation/ablation_results.md`](../ablation/ablation_results.md)  
- [`../ablation/ablation_comparison.png`](../ablation/ablation_comparison.png)

### 13.1 Results

| Experiment | Removed feature | Accuracy | F1 (macro) | ΔF1 from baseline |
|------------|-----------------|----------|------------|-------------------|
| A_baseline | — | 0.9917 | 0.9917 | 0.0000 |
| B_no_SleepQuality | SleepQuality | 0.9833 | 0.9833 | 0.0083 |
| C_no_StressLevel | StressLevel | 0.9833 | 0.9833 | 0.0083 |
| D_no_FatigueLevel | FatigueLevel | 0.9750 | 0.9749 | **0.0167** |
| E_no_PreviousFeedback | PreviousFeedback | 0.9833 | 0.9833 | 0.0083 |
| F_no_SleepHours | SleepHours | 0.9917 | 0.9917 | **0.0000** |

### 13.2 Feature usefulness ranking

| Rank | Feature removed | ΔF1 |
|------|---------------|-----|
| 1 | FatigueLevel | 0.0167 |
| 2 | SleepQuality | 0.0083 |
| 3 | StressLevel | 0.0083 |
| 4 | PreviousFeedback | 0.0083 |
| 5 | SleepHours | 0.0000 |

### 13.3 Key observations

- **FatigueLevel** is the most critical removable feature — its absence causes two additional test errors.
- **SleepHours** is fully redundant given the remaining features (zero F1 drop), despite high Gini importance.
- **SleepQuality**, **StressLevel**, and **PreviousFeedback** each cause one extra error (ΔF1 ≈ 1/120).
- Gini importance and ablation measure different quantities: split frequency vs. marginal predictive contribution.

---

## 14. Cross-Validation Results

Five-fold **stratified** cross-validation was performed on the full 400-row dataset using the same Random Forest pipeline and preprocessing (`build_preprocessor()` fitted per fold).

**Script:** `ml/scripts/run_cross_validation.py`

**Outputs:**  
- [`../cross_validation/cross_validation_results.csv`](../cross_validation/cross_validation_results.csv)  
- [`../cross_validation/cross_validation_summary.md`](../cross_validation/cross_validation_summary.md)  
- [`../cross_validation/cross_validation_boxplot.png`](../cross_validation/cross_validation_boxplot.png)

### 14.1 Per-fold results (out-of-fold test)

| Fold | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|------|----------|-------------------|----------------|------------|
| 1 | 0.9875 | 0.9881 | 0.9875 | 0.9875 |
| 2 | 0.9875 | 0.9881 | 0.9875 | 0.9875 |
| 3 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 4 | 0.9625 | 0.9630 | 0.9625 | 0.9625 |
| 5 | 0.9750 | 0.9773 | 0.9750 | 0.9749 |

### 14.2 Summary statistics

| Metric | Mean | Std |
|--------|------|-----|
| Accuracy | 0.9825 | 0.0143 |
| Precision (macro) | 0.9833 | 0.0139 |
| Recall (macro) | 0.9825 | 0.0143 |
| F1 (macro) | **0.9825** | **0.0143** |

### 14.3 Train vs test (overfitting check)

| Split | Mean F1 (macro) |
|-------|-----------------|
| Train (in-fold) | 1.0000 |
| Test (out-of-fold) | 0.9825 |
| Gap | 0.0175 |

### 14.4 Comparison with hold-out evaluation

| Evaluation | F1 (macro) |
|------------|------------|
| Hold-out test (Step 5) | 0.9917 |
| CV mean ± std | 0.9825 ± 0.0143 |
| Difference | 0.0092 |

Cross-validation confirms that the hold-out result is representative and that fold performance is consistent (F1 range: 0.9625–1.0000).

---

## 15. Discussion of Strengths and Limitations

### 15.1 Strengths

1. **End-to-end reproducibility** — schema validation, fixed random seeds, serialised preprocessors, and scripted pipeline steps.
2. **Domain alignment** — feature names and categorical values match the C# `Student.Domain` model, enabling straightforward integration.
3. **Comprehensive evaluation** — hold-out testing, feature importance, ablation, and cross-validation provide multiple lenses on model behaviour.
4. **Strong class separation in EDA** — feature distributions align with domain expectations and generation rules.
5. **Stable CV results** — low standard deviation (σ ≈ 0.014) across folds.
6. **Interpretability** — Random Forest Gini importance and ablation rankings offer actionable feature insights.

### 15.2 Limitations

1. **Synthetic, rule-based dataset** — all 400 samples were generated programmatically with deterministic class profiles. Near-perfect metrics (F1 ≈ 0.99) reflect recoverable rules, not real-world complexity.
2. **Small sample size** — 400 records (120 test, ~80 per fold) produce discrete metric steps (e.g., ΔF1 = 0.0083 = 1/120).
3. **No hyperparameter tuning** — default Random Forest settings were used; performance might differ with optimised parameters on larger data.
4. **Feature redundancy** — SleepHours and SleepQuality overlap; the model may not generalise if real data breaks these correlations.
5. **No real student validation** — feedback loops, subjective reporting bias, and temporal dynamics are not captured.
6. **Static snapshot** — each row is a single context vector; sequential decision-making over time is not modelled.
7. **Ordinal encoding in EDA only** — correlation analysis used ordinal encoding for categoricals; this is exploratory and differs from the one-hot encoding used in training.

---

## 16. Future Improvements

### 16.1 Data collection

- Collect **real student data** (with consent and anonymisation) to replace or augment the synthetic dataset.
- Introduce **longitudinal records** to model strategy changes over time.
- Expand sample size to reduce metric quantisation and improve statistical power.

### 16.2 Modelling

- Perform **hyperparameter tuning** (e.g., `GridSearchCV` on `n_estimators`, `max_depth`) on a larger dataset.
- Experiment with **class probability calibration** for confidence-aware agent decisions.
- Evaluate **model compression** (ONNX export) for C# inference.
- Consider **explainability tools** (SHAP, LIME) for individual prediction explanations in the agent UI.

### 16.3 Feature engineering

- Merge redundant sleep features (SleepHours + SleepQuality → composite sleep score) based on ablation findings.
- Add interaction terms (e.g., stress × days until exam) if real data shows non-linear boundaries not captured by tree splits.

### 16.4 Integration (next project phase)

- Implement `MlStudyStrategyPolicy` in C# using exported ONNX or a Python prediction service.
- Wire the classifier into the **StudentAgent** Sense → Think → Act → Learn loop.
- Add API endpoints for strategy prediction and feedback collection.
- Implement **online learning** or periodic retraining from agent feedback.

### 16.5 Evaluation on deployment

- Monitor **prediction drift** and retrain when real-world F1 drops below a threshold.
- A/B test agent recommendations against a rule-based baseline on live users.

---

## 17. Final Conclusion

This project successfully designed, validated, and evaluated a complete machine learning pipeline for the Smart Study Advisor. Starting from a schema-aligned synthetic dataset of 400 balanced records, the pipeline progressed through rigorous validation, exploratory analysis, preprocessing, multi-model comparison, and in-depth evaluation.

The **Random Forest Classifier** was selected as the best model, achieving **macro F1 = 0.9917** on the hold-out test set and **mean macro F1 = 0.9825 (σ = 0.0143)** in 5-fold stratified cross-validation. Feature importance and ablation analyses consistently identify **DaysUntilExam**, **HoursStudied**, and **FatigueLevel** as the most influential predictors, while **SleepHours** appears redundant given other features.

The pipeline demonstrates strong internal consistency and reproducibility. However, the near-perfect scores must be interpreted in context: the dataset is synthetic and rule-based, and real student behaviour will introduce noise, missing values, and distribution shift. Before production deployment, the model should be retrained and re-evaluated on authentic data, integrated into the C# agent framework, and monitored for performance degradation over time.

The ML pipeline artefacts, figures, and intermediate reports referenced throughout this document provide a complete audit trail suitable for academic submission and future engineering work.

---

## Appendix A — Complete Output File Index

### Dataset & configuration

| File | Description |
|------|-------------|
| `datasets/student_study_strategy.csv` | Canonical labelled dataset |
| `ml/data/raw/student_study_strategy.csv` | Pipeline copy |
| `ml/data/processed/X_train.csv` | Processed training features |
| `ml/data/processed/X_test.csv` | Processed test features |
| `ml/data/processed/y_train.csv` | Encoded training labels |
| `ml/data/processed/y_test.csv` | Encoded test labels |
| `ml/config/dataset_schema.yaml` | Column contract |
| `ml/config/features.yaml` | Feature ↔ C# mapping |

### Models

| File | Description |
|------|-------------|
| `models/preprocessor.pkl` | Fitted ColumnTransformer |
| `models/label_encoder.pkl` | Fitted LabelEncoder |
| `models/best_model.pkl` | Best Random Forest model |
| `models/random_forest.pkl` | Random Forest (Step 5) |
| `models/logistic_regression.pkl` | Logistic Regression |
| `models/decision_tree.pkl` | Decision Tree |
| `models/gradient_boosting.pkl` | Gradient Boosting |

### Reports & summaries

| File | Description |
|------|-------------|
| `ml/output/eda_summary.md` | EDA summary |
| `ml/output/preprocessing_summary.md` | Preprocessing summary |
| `ml/output/evaluation/evaluation_summary.md` | Model evaluation summary |
| `ml/output/evaluation/model_comparison.csv` | Model comparison table |
| `ml/output/evaluation/classification_report_best_model.txt` | Per-class metrics |
| `ml/output/feature_importance/feature_importance_report.md` | Feature importance report |
| `ml/output/feature_importance/feature_importance.csv` | Importance rankings |
| `ml/output/ablation/ablation_results.md` | Ablation report |
| `ml/output/ablation/ablation_results.csv` | Ablation metrics |
| `ml/output/cross_validation/cross_validation_summary.md` | CV summary |
| `ml/output/cross_validation/cross_validation_results.csv` | CV per-fold metrics |
| `ml/output/final_report/final_ml_report.md` | This report |

### Figures

| File | Description |
|------|-------------|
| `ml/output/figures/01_missing_values.png` | Missing value check |
| `ml/output/figures/02_target_distribution.png` | Class balance |
| `ml/output/figures/03_hist_*.png` | Numeric histograms (4) |
| `ml/output/figures/04_box_*.png` | Box plots by strategy (4) |
| `ml/output/figures/05_count_*.png` | Categorical counts (4) |
| `ml/output/figures/06_*_vs_strategy.png` | Categorical × target (4) |
| `ml/output/figures/07_correlation_matrix_encoded.png` | Correlation heatmap |
| `ml/output/figures/08_iqr_outlier_counts.png` | Outlier counts |
| `ml/output/figures/09–13_*_vs_strategy.png` | Detailed strategy plots (5) |
| `ml/output/evaluation/confusion_matrix_best_model.png` | Confusion matrix |
| `ml/output/feature_importance/feature_importance.png` | Importance bar chart |
| `ml/output/ablation/ablation_comparison.png` | Ablation F1 comparison |
| `ml/output/cross_validation/cross_validation_boxplot.png` | CV metric boxplot |

### Scripts

| Script | Purpose |
|--------|---------|
| `ml/scripts/generate_dataset.py` | Dataset generation |
| `ml/scripts/validate_dataset.py` | Schema validation |
| `ml/scripts/run_eda.py` | Exploratory data analysis |
| `ml/scripts/preprocess_dataset.py` | Preprocessing |
| `ml/scripts/train_models.py` | Model training & evaluation |
| `ml/scripts/feature_importance.py` | Feature importance |
| `ml/scripts/ablation_study.py` | Ablation study |
| `ml/scripts/run_cross_validation.py` | 5-fold cross-validation |

---

*End of report.*
