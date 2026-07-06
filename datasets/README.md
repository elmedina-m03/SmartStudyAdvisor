# Dataset (canonical copy)

The student study strategy dataset is maintained in two locations:

| Path | Purpose |
|------|---------|
| `datasets/student_study_strategy.csv` | Original canonical copy (preserved) |
| `smart_study_advisor/data/raw/student_study_strategy.csv` | Active copy used by the ML pipeline and agent |

Do not delete either copy. The ML pipeline reads from `smart_study_advisor/data/raw/` via `ml/config/dataset_schema.yaml`.
