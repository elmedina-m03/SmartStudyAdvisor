using SharedCore;
using Sleep.Domain;

namespace Sleep.Application;

public class RetrainResultCapturingActuator : IActuator<RetrainDecision, RetrainResult>
{
    private readonly IActuator<RetrainDecision, RetrainResult> _innerActuator;
    private RetrainResult? _lastResult;

    public RetrainResultCapturingActuator(IActuator<RetrainDecision, RetrainResult> innerActuator)
    {
        _innerActuator = innerActuator;
    }

    public RetrainResult Execute(RetrainDecision action)
    {
        _lastResult = _innerActuator.Execute(action);
        return _lastResult.Value;
    }

    public RetrainResult? GetLastResult()
    {
        return _lastResult;
    }
}

