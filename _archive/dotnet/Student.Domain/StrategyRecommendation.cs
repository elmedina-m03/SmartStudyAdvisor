namespace Student.Domain;

public readonly record struct StrategyRecommendation
{
    public StudyStrategy Strategy { get; init; }
    public string Advice { get; init; }

    public StrategyRecommendation(StudyStrategy strategy, string advice)
    {
        Strategy = strategy;
        Advice = advice;
    }
}
