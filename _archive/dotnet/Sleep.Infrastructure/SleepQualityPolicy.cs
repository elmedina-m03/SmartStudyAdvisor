using SharedCore;
using Sleep.Domain;

namespace Sleep.Infrastructure;

public class SleepQualityPolicy : IPolicy<SleepRecord, SleepQualityClass>
{
    private readonly SleepPolicyStateStore _stateStore;
    private SleepPolicyState _state;

    public SleepQualityPolicy(SleepPolicyStateStore stateStore)
    {
        _stateStore = stateStore;
        _state = _stateStore.Load();
    }

    public int CalculateScore(SleepRecord percept)
    {
        int score = 0;

        if (percept.SleepDuration >= 7.0 && percept.SleepDuration <= 9.0)
        {
            score += 3;
        }
        else if (percept.SleepDuration >= 6.0 && percept.SleepDuration < 7.0)
        {
            score += 2;
        }
        else if (percept.SleepDuration > 9.0 && percept.SleepDuration <= 10.0)
        {
            score += 2;
        }
        else
        {
            score += 1;
        }

        if (percept.PhysicalActivity >= 30)
        {
            score += 2;
        }
        else if (percept.PhysicalActivity >= 15)
        {
            score += 1;
        }

        if (percept.DailySteps >= 8000)
        {
            score += 2;
        }
        else if (percept.DailySteps >= 5000)
        {
            score += 1;
        }

        if (percept.StressLevel <= 3)
        {
            score += 2;
        }
        else if (percept.StressLevel <= 5)
        {
            score += 1;
        }

        return score;
    }

    public SleepQualityClass SelectAction(SleepRecord percept)
    {
        var score = CalculateScore(percept);

        if (score >= _state.GoodMinScore)
        {
            return SleepQualityClass.Good;
        }
        else if (score >= _state.AverageMinScore)
        {
            return SleepQualityClass.Average;
        }

        return SleepQualityClass.Poor;
    }

    public bool ApplyLearning(SleepExperience experience)
    {
        if (experience.Predicted == experience.Actual)
        {
            return false;
        }

        var updated = _state;

        if (experience.Actual == SleepQualityClass.Good)
        {
            updated = updated with
            {
                GoodMinScore = Math.Max(updated.GoodMinScore - 1, updated.AverageMinScore + 1)
            };
        }
        else if (experience.Actual == SleepQualityClass.Poor)
        {
            updated = updated with
            {
                AverageMinScore = Math.Min(updated.AverageMinScore + 1, updated.GoodMinScore - 1),
                GoodMinScore = Math.Min(updated.GoodMinScore + 1, SleepPolicyState.MaxScore)
            };
        }
        else if (experience.Actual == SleepQualityClass.Average)
        {
            if (experience.Predicted == SleepQualityClass.Good)
            {
                updated = updated with
                {
                    GoodMinScore = Math.Min(updated.GoodMinScore + 1, SleepPolicyState.MaxScore)
                };
            }
            else if (experience.Predicted == SleepQualityClass.Poor)
            {
                updated = updated with
                {
                    AverageMinScore = Math.Max(updated.AverageMinScore - 1, SleepPolicyState.MinScore)
                };
            }
        }

        updated = updated with
        {
            AverageMinScore = Math.Clamp(updated.AverageMinScore, SleepPolicyState.MinScore, SleepPolicyState.MaxScore - 1),
            GoodMinScore = Math.Clamp(updated.GoodMinScore, updated.AverageMinScore + 1, SleepPolicyState.MaxScore)
        };

        if (updated.Equals(_state))
        {
            return false;
        }

        _state = updated;
        _stateStore.Save(_state);
        return true;
    }
}

