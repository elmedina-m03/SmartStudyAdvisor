namespace Sleep.Domain;

public readonly record struct RetrainResult
{
    public RetrainDecision Decision { get; init; }
    public int NewVersion { get; init; }
    public string Message { get; init; }

    public RetrainResult(RetrainDecision decision, int newVersion, string message)
    {
        Decision = decision;
        NewVersion = newVersion;
        Message = message;
    }
}

