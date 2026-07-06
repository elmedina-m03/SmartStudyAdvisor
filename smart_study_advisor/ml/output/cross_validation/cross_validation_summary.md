# Cross-Validation Summary — Random Forest (5-Fold)

## Setup
- **Model:** RandomForestClassifier (default hyperparameters, random_state=42)
- **Preprocessing:** `build_preprocessor()` — StandardScaler + OneHotEncoder (fitted per fold)
- **CV:** StratifiedKFold, n_splits=5, shuffle=True, random_state=42
- **Dataset:** Full 400 records (all rows participate in CV)

## Per-fold results (out-of-fold test)

```
 fold  accuracy  precision_macro  recall_macro  f1_macro
    1    0.9875           0.9881        0.9875    0.9875
    2    0.9875           0.9881        0.9875    0.9875
    3    1.0000           1.0000        1.0000    1.0000
    4    0.9625           0.9630        0.9625    0.9625
    5    0.9750           0.9773        0.9750    0.9749
```

## Summary statistics (test)

| Metric | Mean | Std |
|--------|------|-----|
| Accuracy | 0.9825 | 0.0143 |
| Precision (macro) | 0.9833 | 0.0139 |
| Recall (macro) | 0.9825 | 0.0143 |
| F1 (macro) | 0.9825 | 0.0143 |

## Train vs test (in-fold vs out-of-fold)

- Mean train F1 (macro): **1.0000**
- Mean test F1 (macro): **0.9825** ± 0.0143
- Train–test F1 gap: **0.0175**

## Comparison with hold-out test (Step 5)
- Hold-out test F1 (macro): **0.9917** (70/30 split)
- CV mean test F1 (macro): **0.9825** ± 0.0143
- Difference: 0.0092

## Generalization

The model **generalizes well**. Mean out-of-fold F1 is **0.9825** with standard deviation **0.0143**, indicating stable performance across folds and alignment with the hold-out evaluation.

## Overfitting

**No strong evidence of overfitting.** The train–test F1 gap is **0.0175** (train 1.0000 vs test 0.9825). Test performance remains high and consistent across folds.

## Fold consistency

Fold results are **consistent**. Test F1 ranges from **0.9625** to **1.0000** (range = 0.0375, σ = 0.0143). No single fold dominates or underperforms severely.

## Output files
- `cross_validation_results.csv`
- `cross_validation_summary.md`
- `cross_validation_boxplot.png`
