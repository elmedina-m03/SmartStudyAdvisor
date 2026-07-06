# Ablation Study Report — Smart Study Advisor

## Setup
- **Model:** RandomForestClassifier (same hyperparameters as best model)
- **Split:** 70/30 stratified, random_state=42
- **Preprocessing:** StandardScaler (numeric) + OneHotEncoder (categorical)
- **Note:** Temporary models only — `models/best_model.pkl` was not modified.

## Baseline (Experiment A)
- Accuracy: 0.9917
- F1 (macro): 0.9917
- Features: 8

## All experiment results

```
           experiment  removed_feature  n_features  accuracy  precision_macro  recall_macro  f1_macro  delta_f1  delta_accuracy
           A_baseline              NaN           8    0.9917           0.9919        0.9917    0.9917    0.0000          0.0000
    B_no_SleepQuality     SleepQuality           7    0.9833           0.9844        0.9833    0.9833    0.0083          0.0083
     C_no_StressLevel      StressLevel           7    0.9833           0.9844        0.9833    0.9833    0.0083          0.0083
    D_no_FatigueLevel     FatigueLevel           7    0.9750           0.9773        0.9750    0.9749    0.0167          0.0167
E_no_PreviousFeedback PreviousFeedback           7    0.9833           0.9844        0.9833    0.9833    0.0083          0.0083
      F_no_SleepHours       SleepHours           7    0.9917           0.9919        0.9917    0.9917    0.0000          0.0000
```

## Feature usefulness ranking (by F1 drop when removed)

```
 UsefulnessRank          Feature  f1_macro  delta_f1  delta_accuracy
              1     FatigueLevel    0.9749    0.0167          0.0167
              2     SleepQuality    0.9833    0.0083          0.0083
              3      StressLevel    0.9833    0.0083          0.0083
              4 PreviousFeedback    0.9833    0.0083          0.0083
              5       SleepHours    0.9917    0.0000          0.0000
```

## Key findings

### Biggest performance drop
- Removing **FatigueLevel** caused the largest F1 decrease (ΔF1 = 0.0167, F1 = 0.9749).

### Smallest performance effect
- Removing **SleepHours** had the smallest impact (ΔF1 = 0.0000).

### Consistency with feature importance
- Feature importance (Gini) ranked DaysUntilExam, HoursStudied, and SleepHours highest among encoded columns.
- Ablation shows **FatigueLevel** causes the largest drop (ΔF1 = 0.0167), consistent with its role in identifying Rest.
- **SleepHours** showed **no F1 drop** when removed (ΔF1 = 0), despite high Gini importance — its signal is likely captured by SleepQuality, DaysUntilExam, and FatigueLevel.
- Gini importance and ablation measure different things: Gini reflects split usage; ablation measures marginal contribution given all other features.

### SleepQuality
- Removing SleepQuality changed F1 by **0.0083**.
- SleepQuality is a useful but not critical predictor; combined with SleepHours redundancy, a simplified sleep feature may suffice.

### StressLevel and FatigueLevel
- StressLevel removal: ΔF1 = **0.0083**
- FatigueLevel removal: ΔF1 = **0.0167** (largest drop)
- FatigueLevel is the most important categorical group for prediction; StressLevel adds moderate value.

### Features that could potentially be removed
- **SleepHours** (ΔF1 = 0.0000) — minimal impact; candidate for simplification.

## Output files
- `ablation_results.csv`
- `ablation_results.md`
- `ablation_comparison.png`
