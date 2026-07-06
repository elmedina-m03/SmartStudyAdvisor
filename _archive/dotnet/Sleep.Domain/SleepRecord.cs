namespace Sleep.Domain;

public readonly record struct SleepRecord
{
    public int Age { get; init; }
    public string Gender { get; init; }
    public int DailySteps { get; init; }
    public int PhysicalActivity { get; init; }
    public int StressLevel { get; init; }
    public double SleepDuration { get; init; }

    public SleepRecord(int age, string gender, int dailySteps, int physicalActivity, int stressLevel, double sleepDuration)
    {
        Age = age;
        Gender = gender;
        DailySteps = dailySteps;
        PhysicalActivity = physicalActivity;
        StressLevel = stressLevel;
        SleepDuration = sleepDuration;
    }
}

