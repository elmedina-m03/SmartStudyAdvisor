namespace Sleep.Web;

public class SleepDataRequest
{
    public int Age { get; set; }
    public string Gender { get; set; } = string.Empty;
    public int DailySteps { get; set; }
    public int PhysicalActivity { get; set; }
    public int StressLevel { get; set; }
    public double SleepDuration { get; set; }
}

