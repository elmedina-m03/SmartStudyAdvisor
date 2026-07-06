using SharedCore;
using Sleep.Domain;

namespace Sleep.Application;

public class ResultCapturingActuator : IActuator<SleepQualityClass, SleepAssessment>
{
    private readonly IActuator<SleepQualityClass, SleepAssessment> _innerActuator;
    private SleepAssessment? _lastResult;

    public ResultCapturingActuator(IActuator<SleepQualityClass, SleepAssessment> innerActuator)
    {
        _innerActuator = innerActuator;
    }

    public SleepAssessment Execute(SleepQualityClass action)
    {
        _lastResult = _innerActuator.Execute(action);
        return _lastResult.Value;
    }

    public SleepAssessment? GetLastResult()
    {
        return _lastResult;
    }
}

