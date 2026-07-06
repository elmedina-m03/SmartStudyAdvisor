using SharedCore;
using Sleep.Domain;

namespace Sleep.Infrastructure;

public class RetrainPolicy : IPolicy<ProcessedRecordsCounter, RetrainDecision>
{
    private readonly RetrainPolicyStateStore _stateStore;
    private RetrainPolicyState _state;

    public RetrainPolicy(RetrainPolicyStateStore stateStore)
    {
        _stateStore = stateStore;
        _state = _stateStore.Load();
    }

    public RetrainDecision SelectAction(ProcessedRecordsCounter percept)
    {
        if (percept.Count >= _state.Threshold)
        {
            return RetrainDecision.Retrain;
        }

        return RetrainDecision.Skip;
    }

    public int CurrentThreshold => _state.Threshold;

    public bool UpdateThreshold(int newThreshold)
    {
        var normalized = Math.Max(newThreshold, RetrainPolicyState.MinThreshold);
        if (normalized == _state.Threshold)
        {
            return false;
        }

        _state = _state with { Threshold = normalized };
        _stateStore.Save(_state);
        return true;
    }
}

