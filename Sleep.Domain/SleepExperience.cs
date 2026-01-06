namespace Sleep.Domain;

public readonly record struct SleepExperience
{
    public SleepRecord Input { get; init; }
    public SleepQualityClass Predicted { get; init; }
    public SleepQualityClass Actual { get; init; }

    public SleepExperience(SleepRecord input, SleepQualityClass predicted, SleepQualityClass actual)
    {
        Input = input;
        Predicted = predicted;
        Actual = actual;
    }
}

