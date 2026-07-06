namespace Student.Domain;

/// <summary>
/// Column names and allowed values shared between dataset I/O and the ML pipeline.
/// Must stay in sync with ml/config/dataset_schema.yaml.
/// </summary>
public static class StudentDatasetSchema
{
    public const string FileName = "student_study_strategy.csv";

    public static class Columns
    {
        public const string HoursStudied = nameof(HoursStudied);
        public const string NumberOfSubjects = nameof(NumberOfSubjects);
        public const string DaysUntilExam = nameof(DaysUntilExam);
        public const string StressLevel = nameof(StressLevel);
        public const string FatigueLevel = nameof(FatigueLevel);
        public const string SleepHours = nameof(SleepHours);
        public const string SleepQuality = nameof(SleepQuality);
        public const string PreviousFeedback = nameof(PreviousFeedback);
        public const string RecommendedStrategy = nameof(RecommendedStrategy);
    }

    public static readonly string[] FeatureColumns =
    [
        Columns.HoursStudied,
        Columns.NumberOfSubjects,
        Columns.DaysUntilExam,
        Columns.StressLevel,
        Columns.FatigueLevel,
        Columns.SleepHours,
        Columns.SleepQuality,
        Columns.PreviousFeedback
    ];

    public static readonly string[] AllColumns =
    [
        .. FeatureColumns,
        Columns.RecommendedStrategy
    ];

    public static readonly string[] HeaderRow = AllColumns;

    public static readonly string[] StressLevels = ["Low", "Medium", "High"];
    public static readonly string[] FatigueLevels = ["Low", "Medium", "High"];
    public static readonly string[] SleepQualityLevels = ["Poor", "Average", "Good"];
    public static readonly string[] PreviousFeedbackValues = ["None", "Positive", "Negative", "Mixed"];
    public static readonly string[] RecommendedStrategies =
        ["Rest", "BalancedStudy", "IntensiveStudy", "LongTermPlan"];
}
