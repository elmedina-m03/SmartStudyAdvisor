using Sleep.Domain;

namespace Sleep.Application;

public class SleepScoringTickResult
{
    public string RecordIdentifier { get; init; }
    public SleepQualityClass PredictedQuality { get; init; }
    public string Advice { get; init; }

    public SleepScoringTickResult(string recordIdentifier, SleepQualityClass predictedQuality, string advice)
    {
        RecordIdentifier = recordIdentifier;
        PredictedQuality = predictedQuality;
        Advice = advice;
    }
}

