using Sleep.Domain;

namespace Sleep.Application;

public static class RetrainGroundTruthProvider
{
    private const int PreferredThreshold = 5;

    public static RetrainDecision GetGroundTruth(ProcessedRecordsCounter counter)
    {
        return counter.Count >= PreferredThreshold ? RetrainDecision.Retrain : RetrainDecision.Skip;
    }
}
