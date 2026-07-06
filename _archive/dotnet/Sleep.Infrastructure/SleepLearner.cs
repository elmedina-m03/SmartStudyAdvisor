using SharedCore;
using Sleep.Domain;

namespace Sleep.Infrastructure;

public class SleepLearner : ILearningComponent<SleepExperience>
{
    private readonly SleepQualityPolicy _policy;

    public SleepLearner(SleepQualityPolicy policy)
    {
        _policy = policy;
    }

    public void Learn(SleepExperience experience)
    {
        _policy.ApplyLearning(experience);
    }
}

