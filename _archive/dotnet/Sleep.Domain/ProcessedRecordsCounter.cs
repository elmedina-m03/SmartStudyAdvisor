namespace Sleep.Domain;

public readonly record struct ProcessedRecordsCounter
{
    public int Count { get; init; }

    public ProcessedRecordsCounter(int count)
    {
        Count = count;
    }
}

