using SharedCore;
using Sleep.Domain;

namespace Sleep.Infrastructure;

public class RetrainLearner : ILearningComponent<RetrainExperience>
{
    public void Learn(RetrainExperience experience)
    {
        // Learning logic - in a real implementation, this would update the policy threshold
        // For educational purposes, this is a placeholder
    }
}

