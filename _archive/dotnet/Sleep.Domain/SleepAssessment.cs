namespace Sleep.Domain;

public readonly record struct SleepAssessment
{
    public SleepQualityClass Quality { get; init; }
    public string Advice { get; init; }

    public SleepAssessment(SleepQualityClass quality, string advice)
    {
        Quality = quality;
        Advice = advice;
    }
}

