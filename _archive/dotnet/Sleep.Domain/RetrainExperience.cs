namespace Sleep.Domain;

public readonly record struct RetrainExperience
{
    public ProcessedRecordsCounter Input { get; init; }
    public RetrainDecision Predicted { get; init; }
    public RetrainDecision Actual { get; init; }

    public RetrainExperience(ProcessedRecordsCounter input, RetrainDecision predicted, RetrainDecision actual)
    {
        Input = input;
        Predicted = predicted;
        Actual = actual;
    }
}

