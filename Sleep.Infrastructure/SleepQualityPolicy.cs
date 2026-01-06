using SharedCore;
using Sleep.Domain;

namespace Sleep.Infrastructure;

public class SleepQualityPolicy : IPolicy<SleepRecord, SleepQualityClass>
{
    public SleepQualityClass SelectAction(SleepRecord percept)
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

        if (score >= 7)
        {
            return SleepQualityClass.Good;
        }
        else if (score >= 4)
        {
            return SleepQualityClass.Average;
        }
        else
        {
            return SleepQualityClass.Poor;
        }
    }
}

