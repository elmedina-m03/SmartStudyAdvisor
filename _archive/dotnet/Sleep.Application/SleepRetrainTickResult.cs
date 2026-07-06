using Sleep.Domain;

namespace Sleep.Application;

public class SleepRetrainTickResult
{
    public RetrainDecision Decision { get; init; }
    public int ModelVersion { get; init; }
    public string Message { get; init; }

    public SleepRetrainTickResult(RetrainDecision decision, int modelVersion, string message)
    {
        Decision = decision;
        ModelVersion = modelVersion;
        Message = message;
    }
}

