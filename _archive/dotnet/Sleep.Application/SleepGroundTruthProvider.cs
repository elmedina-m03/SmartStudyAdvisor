using Sleep.Domain;

namespace Sleep.Application;

public static class SleepGroundTruthProvider
{
    public static SleepQualityClass GetGroundTruth(SleepRecord record)
    {
        if (record.SleepDuration >= 7.5 && record.SleepDuration <= 9.0 &&
            record.PhysicalActivity >= 30 && record.StressLevel <= 4)
        {
            return SleepQualityClass.Good;
        }
        else if (record.SleepDuration >= 6.0 && record.SleepDuration < 7.5 &&
                 record.PhysicalActivity >= 15 && record.StressLevel <= 6)
        {
            return SleepQualityClass.Average;
        }
        else
        {
            return SleepQualityClass.Poor;
        }
    }
}

