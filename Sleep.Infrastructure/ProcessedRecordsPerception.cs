using SharedCore;
using Sleep.Domain;

namespace Sleep.Infrastructure;

public class ProcessedRecordsPerception : IPerceptionSource<ProcessedRecordsCounter>
{
    private readonly Func<int> _counterProvider;

    public ProcessedRecordsPerception(Func<int> counterProvider)
    {
        _counterProvider = counterProvider;
    }

    public ProcessedRecordsCounter Observe()
    {
        var count = _counterProvider();
        return new ProcessedRecordsCounter(count);
    }
}

