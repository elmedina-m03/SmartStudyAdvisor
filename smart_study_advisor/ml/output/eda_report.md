# Exploratory Data Analysis Report

## 1. Overview
- **Records:** 400
- **Features:** 8
- **Target:** `RecommendedStrategy` (4 classes)
- **Source:** `C:\Users\acer\OneDrive - Faculty of Information Technologies\Desktop\4. Godina\SleepQualityAgent\datasets\student_study_strategy.csv`

## 2. Data quality
- Missing values: **0** across all columns
- Duplicate rows: **0**
- Duplicate feature combinations (excluding target): **0**

## 3. Target distribution
RecommendedStrategy
Rest              100
BalancedStudy     100
IntensiveStudy    100
LongTermPlan      100

Classes are balanced (100 samples each), which supports fair classifier evaluation.

## 4. Descriptive statistics (numeric features)

```
                   mean  median    std  min   max   Q1      Q3
HoursStudied      4.268     4.1  2.538  0.0  10.0  2.1   6.125
NumberOfSubjects  4.415     4.0  1.782  1.0   8.0  3.0   6.000
DaysUntilExam     8.808     4.5  8.626  0.0  30.0  2.0  14.000
StressLevel       6.112     6.5  2.474  1.0  10.0  4.0   8.000
FatigueLevel      5.038     5.0  2.413  1.0  10.0  3.0   6.250
SleepHours        6.470     6.6  1.321  3.5   9.0  5.5   7.400
```

## 5. Correlation highlights
Strongest absolute correlations among numeric features:

- `DaysUntilExam` vs `StressLevel`: **-0.81**
- `FatigueLevel` vs `SleepHours`: **-0.76**
- `DaysUntilExam` vs `SleepHours`: **0.73**
- `StressLevel` vs `SleepHours`: **-0.73**
- `DaysUntilExam` vs `FatigueLevel`: **-0.68**

## 6. Mean feature values by strategy

```
                     HoursStudied  NumberOfSubjects  DaysUntilExam  StressLevel  FatigueLevel  SleepHours
RecommendedStrategy                                                                                      
Rest                         1.53              2.97           1.42         8.58          8.40        4.76
BalancedStudy                5.19              4.55           9.49         5.42          4.53        6.98
IntensiveStudy               7.43              6.22           2.56         7.53          4.70        6.24
LongTermPlan                 2.93              3.92          21.76         2.92          2.52        7.89
```

### Interpretation
- **Rest** samples show the highest stress and fatigue and the lowest sleep duration.
- **LongTermPlan** samples have the most days until exam and the lowest stress/fatigue.
- **IntensiveStudy** combines short exam horizons with higher study hours.
- **BalancedStudy** occupies the middle range on most numeric features.

## 7. Outlier analysis (IQR method)

```
         Feature  Q1     Q3    IQR  LowerBound  UpperBound  OutlierCount  OutlierPercent
    HoursStudied 2.1  6.125  4.025      -3.938      12.163             0             0.0
NumberOfSubjects 3.0  6.000  3.000      -1.500      10.500             0             0.0
   DaysUntilExam 2.0 14.000 12.000     -16.000      32.000             0             0.0
     StressLevel 4.0  8.000  4.000      -2.000      14.000             0             0.0
    FatigueLevel 3.0  6.250  3.250      -1.875      11.125             0             0.0
      SleepHours 5.5  7.400  1.900       2.650      10.250             0             0.0
```

Total rows flagged by at least one IQR rule: **0** (0.0% of dataset).

Outliers in this synthetic dataset often reflect extreme but valid student states (e.g., exam today with maximum fatigue) rather than data errors.

## 8. Figures
- `figures/01_missing_values.png`
- `figures/02_target_distribution.png`
- `figures/03_numeric_histograms.png`
- `figures/04_categorical_counts.png`
- `figures/05_correlation_matrix.png`
- `figures/06_boxplot_DaysUntilExam.png`
- `figures/06_boxplot_FatigueLevel.png`
- `figures/06_boxplot_HoursStudied.png`
- `figures/06_boxplot_NumberOfSubjects.png`
- `figures/06_boxplot_SleepHours.png`
- `figures/06_boxplot_StressLevel.png`
- `figures/06_boxplots_by_strategy.png`
- `figures/07_pairplot_by_strategy.png`
- `figures/08_sleep_quality_vs_strategy.png`
- `figures/09_stress_level_vs_strategy.png`
- `figures/10_fatigue_level_vs_strategy.png`
- `figures/11_days_until_exam_vs_strategy.png`
- `figures/12_iqr_outlier_counts.png`

## 9. Implications for modeling
- Numeric features show clear separation by strategy; tree-based models should perform well.
- `DaysUntilExam`, `StressLevel`, and `FatigueLevel` are primary discriminators.
- `SleepHours` and `SleepQuality` are related; ablation studies should test redundancy.
- No missing values or duplicate rows — preprocessing can focus on encoding and scaling.
- Recommended next step: stratified train/validation split and baseline classifier training.
