# Model Evaluation Summary — Smart Study Advisor

## Models compared
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting

## Results table (sorted by macro F1)

```
              Model  Accuracy  Precision_macro  Recall_macro  F1_macro
      random_forest    0.9917           0.9919        0.9917    0.9917
logistic_regression    0.9833           0.9836        0.9833    0.9833
      decision_tree    0.9750           0.9773        0.9750    0.9749
  gradient_boosting    0.9667           0.9706        0.9667    0.9665
```

## Best model: **random_forest**
- **Macro F1:** 0.9917
- **Accuracy:** 0.9917
- **Precision (macro):** 0.9919
- **Recall (macro):** 0.9917

## Output files
- `model_comparison.csv`
- `classification_report_best_model.txt`
- `confusion_matrix_best_model.png`

## Saved models (`models/`)
- `logistic_regression.pkl`
- `decision_tree.pkl`
- `random_forest.pkl`
- `gradient_boosting.pkl`
- `best_model.pkl`
