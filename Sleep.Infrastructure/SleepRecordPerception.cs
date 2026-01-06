using SharedCore;
using Sleep.Domain;

namespace Sleep.Infrastructure;

public class SleepRecordPerception : IPerceptionSource<SleepRecord>
{
    private readonly Queue<SleepRecord> _sleepRecords;

    public SleepRecordPerception(Queue<SleepRecord> sleepRecords)
    {
        _sleepRecords = sleepRecords;
    }

    public SleepRecord Observe()
    {
        if (_sleepRecords.Count == 0)
        {
            throw new InvalidOperationException("No more sleep records available.");
        }

        return _sleepRecords.Dequeue();
    }
}

