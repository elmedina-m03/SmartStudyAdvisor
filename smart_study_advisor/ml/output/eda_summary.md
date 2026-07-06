# Smart Study Advisor — EDA Summary

## 1. Dataset overview
- **Source:** `ml/data/raw/student_study_strategy.csv`
- **Records:** 400
- **Features:** 8 (4 numeric, 4 categorical)
- **Target:** `RecommendedStrategy`

## 2. Data quality
- Missing values: **0**
- Duplicate rows: **0**
- Invalid categorical values: **0**

### Class balance
```
                     Count  Percent
RecommendedStrategy                
Rest                   100     25.0
BalancedStudy          100     25.0
IntensiveStudy         100     25.0
LongTermPlan           100     25.0
```

## 3. Descriptive statistics (numeric only)
```
                   mean  median    std  min   max     Q1    Q3
HoursStudied      4.260     4.0  2.702  0.0  10.0  1.875   6.3
NumberOfSubjects  4.615     5.0  2.050  1.0   9.0  3.000   6.0
DaysUntilExam     8.522     4.0  8.754  0.0  30.0  2.000  14.0
SleepHours        6.439     6.5  1.391  3.5   8.9  5.500   7.5
```

## 4. Mean numeric features by strategy
```
                     HoursStudied  NumberOfSubjects  DaysUntilExam  SleepHours
RecommendedStrategy                                                           
Rest                         1.27              2.80           1.45        4.58
BalancedStudy                4.88              4.96           8.13        6.96
IntensiveStudy               7.91              6.71           2.60        6.22
LongTermPlan                 2.98              3.99          21.91        8.00
```

## 5. Correlation analysis (ordinal-encoded categoricals)

**Strongest positive correlations:**
- `DaysUntilExam` ↔ `SleepHours`: **0.748**
- `HoursStudied` ↔ `NumberOfSubjects`: **0.642**
- `SleepHours` ↔ `SleepQuality`: **0.627**
- `DaysUntilExam` ↔ `SleepQuality`: **0.514**
- `StressLevel` ↔ `FatigueLevel`: **0.332**

**Strongest negative correlations:**
- `DaysUntilExam` ↔ `StressLevel`: **-0.581**
- `StressLevel` ↔ `SleepHours`: **-0.527**
- `FatigueLevel` ↔ `SleepHours`: **-0.521**
- `DaysUntilExam` ↔ `FatigueLevel`: **-0.432**
- `FatigueLevel` ↔ `SleepQuality`: **-0.368**

StressLevel/FatigueLevel encoded as Low=0, Medium=1, High=2 (correlation only).

## 6. Outlier analysis (IQR, numeric only)
```
         Feature    Q1   Q3    IQR  LowerBound  UpperBound  OutlierCount  OutlierPercent
    HoursStudied 1.875  6.3  4.425      -4.762      12.938             0             0.0
NumberOfSubjects 3.000  6.0  3.000      -1.500      10.500             0             0.0
   DaysUntilExam 2.000 14.0 12.000     -16.000      32.000             0             0.0
      SleepHours 5.500  7.5  2.000       2.500      10.500             0             0.0
```

Rows flagged: **0** (0.0%).

**Recommendation:** Retain outliers — valid extreme student states.

## 7. Key findings
- StressLevel and FatigueLevel are **categorical** (Low/Medium/High), not numeric.
- High stress/fatigue → Rest; Low stress/fatigue → LongTermPlan.
- DaysUntilExam is the strongest numeric separator.

## 8. Exported figures
- `figures/01_missing_values.png`
- `figures/02_target_distribution.png`
- `figures/03_hist_DaysUntilExam.png`
- `figures/03_hist_HoursStudied.png`
- `figures/03_hist_NumberOfSubjects.png`
- `figures/03_hist_SleepHours.png`
- `figures/04_box_DaysUntilExam.png`
- `figures/04_box_HoursStudied.png`
- `figures/04_box_NumberOfSubjects.png`
- `figures/04_box_SleepHours.png`
- `figures/05_count_FatigueLevel.png`
- `figures/05_count_PreviousFeedback.png`
- `figures/05_count_SleepQuality.png`
- `figures/05_count_StressLevel.png`
- `figures/06_FatigueLevel_vs_strategy.png`
- `figures/06_PreviousFeedback_vs_strategy.png`
- `figures/06_SleepQuality_vs_strategy.png`
- `figures/06_StressLevel_vs_strategy.png`
- `figures/07_correlation_matrix_encoded.png`
- `figures/08_iqr_outlier_counts.png`
- `figures/09_sleep_quality_vs_strategy.png`
- `figures/10_stress_level_vs_strategy.png`
- `figures/11_fatigue_level_vs_strategy.png`
- `figures/12_days_until_exam_vs_strategy.png`
- `figures/13_sleep_hours_vs_strategy.png`
- `figures/14_correlation_matrix_numeric.png`

## 9. Next step
Preprocessing + baseline classifier (Step 4).
