# Feature Importance Report — Smart Study Advisor

## Model analysed
- **Type:** `RandomForestClassifier`
- **Source:** `models/best_model.pkl`
- **Method:** builtin

The best model is a **RandomForestClassifier**, which provides built-in `feature_importances_` based on mean decrease in impurity (Gini importance across trees).

## Top 10 features (encoded names)

```
                    ReadableLabel  Importance
Rank                                         
1          DaysUntilExam (scaled)      0.3390
2           HoursStudied (scaled)      0.2435
3             SleepHours (scaled)      0.1784
4       NumberOfSubjects (scaled)      0.0657
5             FatigueLevel = High      0.0356
6     PreviousFeedback = Negative      0.0343
7              StressLevel = High      0.0172
8               StressLevel = Low      0.0168
9              FatigueLevel = Low      0.0163
10            SleepQuality = Poor      0.0133
```

## Aggregated importance by base feature

```
     BaseFeature  Importance  Rank
   DaysUntilExam    0.338996     1
    HoursStudied    0.243513     2
      SleepHours    0.178397     3
NumberOfSubjects    0.065740     4
    FatigueLevel    0.054645     5
PreviousFeedback    0.050208     6
     StressLevel    0.038689     7
    SleepQuality    0.029812     8
```

## Most important features

- **DaysUntilExam** — total importance 0.3390
- **HoursStudied** — total importance 0.2435
- **SleepHours** — total importance 0.1784
- **NumberOfSubjects** — total importance 0.0657
- **FatigueLevel** — total importance 0.0546

## Least influential features

- `PreviousFeedback = Mixed` — importance 0.0049
- `StressLevel = Medium` — importance 0.0048
- `PreviousFeedback = None` — importance 0.0047
- `SleepQuality = Average` — importance 0.0043
- `FatigueLevel = Medium` — importance 0.0028

## Domain consistency

The ranking aligns with the rules used to generate the dataset and the EDA findings:
- **DaysUntilExam** separates urgent strategies (Rest, IntensiveStudy) from LongTermPlan.
- **StressLevel** and **FatigueLevel** strongly predict Rest vs recovery-oriented plans.
- **SleepQuality** and **SleepHours** reflect recovery state relevant to Rest recommendations.
- **HoursStudied** and **NumberOfSubjects** distinguish IntensiveStudy from BalancedStudy.

## How SleepQuality influences recommendations

```
         ReadableLabel  Importance
   SleepQuality = Poor    0.013339
   SleepQuality = Good    0.012211
SleepQuality = Average    0.004262
```

**Interpretation:** `SleepQuality = Poor` has the highest SleepQuality-level importance (0.0133). Poor sleep pushes toward **Rest**; Good sleep supports **LongTermPlan** and **BalancedStudy**.

## How StressLevel influences recommendations

```
       ReadableLabel  Importance
  StressLevel = High    0.017168
   StressLevel = Low    0.016761
StressLevel = Medium    0.004760
```

**Interpretation:** `StressLevel = High` dominates stress-related splits. High stress aligns with **Rest** and **IntensiveStudy**; Low stress with **LongTermPlan**.

## How FatigueLevel influences recommendations

```
        ReadableLabel  Importance
  FatigueLevel = High    0.035564
   FatigueLevel = Low    0.016302
FatigueLevel = Medium    0.002779
```

**Interpretation:** `FatigueLevel = High` is the strongest fatigue signal. High fatigue is the primary indicator for **Rest**; Low fatigue supports longer-horizon planning.

## Output files
- `feature_importance.csv`
- `feature_importance.png`
- `feature_importance_report.md`
