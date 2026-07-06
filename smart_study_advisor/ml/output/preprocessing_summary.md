# Preprocessing Summary — Smart Study Advisor

## 1. Source dataset
- **Path:** `C:\Users\acer\OneDrive - Faculty of Information Technologies\Desktop\4. Godina\SleepQualityAgent\datasets\student_study_strategy.csv`
- **Total records:** 400

## 2. Feature separation
- **X (inputs):** 8 features — HoursStudied, NumberOfSubjects, DaysUntilExam, StressLevel, FatigueLevel, SleepHours, SleepQuality, PreviousFeedback
- **y (target):** `RecommendedStrategy`

## 3. Categorical encoding (One-Hot)

The following columns were one-hot encoded using `OneHotEncoder`:

- `StressLevel`
- `FatigueLevel`
- `SleepQuality`
- `PreviousFeedback`

**Generated columns (13):**

- `categorical__StressLevel_High`
- `categorical__StressLevel_Low`
- `categorical__StressLevel_Medium`
- `categorical__FatigueLevel_High`
- `categorical__FatigueLevel_Low`
- `categorical__FatigueLevel_Medium`
- `categorical__SleepQuality_Average`
- `categorical__SleepQuality_Good`
- `categorical__SleepQuality_Poor`
- `categorical__PreviousFeedback_Mixed`
- `categorical__PreviousFeedback_Negative`
- `categorical__PreviousFeedback_None`
- `categorical__PreviousFeedback_Positive`

## 4. Numerical scaling (StandardScaler)

The following columns were standardised (zero mean, unit variance):

- `HoursStudied`
- `NumberOfSubjects`
- `DaysUntilExam`
- `SleepHours`

**Scaled column names in output:**

- `numeric__HoursStudied`
- `numeric__NumberOfSubjects`
- `numeric__DaysUntilExam`
- `numeric__SleepHours`

## 5. Target encoding (LabelEncoder)

| Class label | Encoded value |
|-------------|---------------|
| `BalancedStudy` | 0 |
| `IntensiveStudy` | 1 |
| `LongTermPlan` | 2 |
| `Rest` | 3 |

## 6. Train / test split

| Setting | Value |
|---------|-------|
| Train size | 280 (70%) |
| Test size | 120 (30%) |
| Stratified | Yes |
| random_state | 42 |

**Why stratified 70/30?**
- Preserves the 25% class balance in both train and test sets.
- 70% provides enough data for training while 30% gives a reliable hold-out evaluation set.
- Stratification prevents a class from being under-represented in either split.

## 7. Final class distribution

### Training set
```
                     Count  Percent
RecommendedStrategy                
BalancedStudy           70     25.0
IntensiveStudy          70     25.0
LongTermPlan            70     25.0
Rest                    70     25.0
```

### Test set
```
                     Count  Percent
RecommendedStrategy                
BalancedStudy           30     25.0
IntensiveStudy          30     25.0
LongTermPlan            30     25.0
Rest                    30     25.0
```

## 8. Output files

**Processed data** (`ml/data/processed/`):
- `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`

**Artifacts** (`models/`):
- `preprocessor.pkl` — fitted ColumnTransformer (scaler + one-hot encoder)
- `label_encoder.pkl` — fitted LabelEncoder for the target

## 9. Next step
Train and evaluate a baseline classifier (Step 5).
