namespace Sleep.Infrastructure;

public record SleepPolicyState
{
    public const int MinScore = 1;
    public const int MaxScore = 10;

    public int GoodMinScore { get; init; } = 7;
    public int AverageMinScore { get; init; } = 4;
}
