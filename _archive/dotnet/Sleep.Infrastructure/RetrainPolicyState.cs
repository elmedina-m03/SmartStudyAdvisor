namespace Sleep.Infrastructure;

public record RetrainPolicyState
{
    public const int MinThreshold = 1;

    public int Threshold { get; init; } = 3;
}
