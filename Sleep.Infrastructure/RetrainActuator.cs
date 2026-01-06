using SharedCore;
using Sleep.Domain;

namespace Sleep.Infrastructure;

public class RetrainActuator : IActuator<RetrainDecision, RetrainResult>
{
    private int _currentVersion = 1;

    public RetrainResult Execute(RetrainDecision action)
    {
        if (action == RetrainDecision.Retrain)
        {
            _currentVersion++;
            return new RetrainResult(
                RetrainDecision.Retrain,
                _currentVersion,
                $"Model retrained. New version: {_currentVersion}"
            );
        }

        return new RetrainResult(
            RetrainDecision.Skip,
            _currentVersion,
            $"Retraining skipped. Current version: {_currentVersion}"
        );
    }
}

