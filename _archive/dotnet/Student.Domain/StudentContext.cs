namespace Student.Domain;

public readonly record struct StudentContext
{
    public double HoursStudied { get; init; }
    public int NumberOfSubjects { get; init; }
    public int DaysUntilExam { get; init; }
    public StressLevelCategory StressLevel { get; init; }
    public FatigueLevelCategory FatigueLevel { get; init; }
    public double SleepHours { get; init; }
    public SleepQualityLevel SleepQuality { get; init; }
    public PreviousFeedbackType PreviousFeedback { get; init; }

    public StudentContext(
        double hoursStudied,
        int numberOfSubjects,
        int daysUntilExam,
        StressLevelCategory stressLevel,
        FatigueLevelCategory fatigueLevel,
        double sleepHours,
        SleepQualityLevel sleepQuality,
        PreviousFeedbackType previousFeedback)
    {
        HoursStudied = hoursStudied;
        NumberOfSubjects = numberOfSubjects;
        DaysUntilExam = daysUntilExam;
        StressLevel = stressLevel;
        FatigueLevel = fatigueLevel;
        SleepHours = sleepHours;
        SleepQuality = sleepQuality;
        PreviousFeedback = previousFeedback;
    }
}
