using SharedCore;
using Sleep.Domain;

namespace Sleep.Infrastructure;

public class RetrainLearner : ILearningComponent<RetrainExperience>
{
    private readonly RetrainPolicy _policy;

    public RetrainLearner(RetrainPolicy policy)
    {
        _policy = policy;
    }

    public void Learn(RetrainExperience experience)
    {
        if (experience.Predicted == experience.Actual)
        {
            return;
        }

        if (experience.Actual == RetrainDecision.Retrain)
        {
            _policy.UpdateThreshold(_policy.CurrentThreshold - 1);
        }
        else
        {
            _policy.UpdateThreshold(_policy.CurrentThreshold + 1);
        }
    }
}

