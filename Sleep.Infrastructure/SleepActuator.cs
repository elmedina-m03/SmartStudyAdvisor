using SharedCore;
using Sleep.Domain;

namespace Sleep.Infrastructure;

public class SleepActuator : IActuator<SleepQualityClass, SleepAssessment>
{
    public SleepAssessment Execute(SleepQualityClass action)
    {
        string advice = action switch
        {
            SleepQualityClass.Good => "Excellent! Maintain your current sleep routine and lifestyle habits.",
            SleepQualityClass.Average => "Your sleep quality is moderate. Consider improving sleep duration, increasing physical activity, or reducing stress.",
            SleepQualityClass.Poor => "Your sleep quality needs improvement. Focus on: getting 7-9 hours of sleep, regular exercise, stress management, and maintaining a consistent sleep schedule.",
            _ => "Unable to provide specific advice."
        };

        return new SleepAssessment(action, advice);
    }
}

