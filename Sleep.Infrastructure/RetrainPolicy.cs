using SharedCore;
using Sleep.Domain;

namespace Sleep.Infrastructure;

public class RetrainPolicy : IPolicy<ProcessedRecordsCounter, RetrainDecision>
{
    private readonly int _retrainThreshold;

    public RetrainPolicy(int retrainThreshold = 3)
    {
        _retrainThreshold = retrainThreshold;
    }

    public RetrainDecision SelectAction(ProcessedRecordsCounter percept)
    {
        if (percept.Count >= _retrainThreshold)
        {
            return RetrainDecision.Retrain;
        }

        return RetrainDecision.Skip;
    }
}

