using SharedCore;
using Sleep.Domain;

namespace Sleep.Infrastructure;

public class SleepLearner : ILearningComponent<SleepExperience>
{
    public void Learn(SleepExperience experience)
    {
        // Learning logic - in a real implementation, this would update the policy
        // For educational purposes, this is a placeholder
    }
}

