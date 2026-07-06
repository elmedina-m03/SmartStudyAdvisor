using Sleep.Domain;
using Sleep.Infrastructure;

namespace Sleep.Application;

public class SleepAgentOrchestrator
{
    private readonly ExcelDatasetService _excelService;
    private readonly Queue<SleepRecord> _scoringQueue = new();
    private readonly object _queueLock = new();
    private readonly SleepScoringAgentRunner _scoringRunner;
    private readonly SleepRetrainAgentRunner _retrainRunner;

    public SleepAgentOrchestrator(
        ExcelDatasetService excelService,
        SleepQualityPolicy sleepPolicy,
        SleepLearner sleepLearner,
        RetrainPolicy retrainPolicy,
        RetrainLearner retrainLearner)
    {
        _excelService = excelService;

        var perception = new SleepRecordPerception(_scoringQueue);
        var actuator = new SleepActuator();

        SleepExperience CreateExperience(SleepRecord percept, SleepQualityClass action, SleepAssessment result)
        {
            var actual = SleepGroundTruthProvider.GetGroundTruth(percept);
            return new SleepExperience(percept, action, actual);
        }

        _scoringRunner = new SleepScoringAgentRunner(
            perception,
            sleepPolicy,
            actuator,
            SleepGroundTruthProvider.GetGroundTruth,
            CreateExperience,
            sleepLearner
        );

        var retrainPerception = new ProcessedRecordsPerception(() => _excelService.ProcessedCount);
        var retrainActuator = new RetrainActuator();

        _retrainRunner = new SleepRetrainAgentRunner(
            retrainPerception,
            retrainPolicy,
            retrainActuator,
            RetrainGroundTruthProvider.GetGroundTruth,
            null,
            retrainLearner
        );
    }

    public SleepScoringTickResult? PredictNow(SleepRecord record)
    {
        lock (_queueLock)
        {
            _scoringQueue.Enqueue(record);
            return _scoringRunner.Tick();
        }
    }

    public SleepScoringTickResult? ProcessScoringTick()
    {
        lock (_queueLock)
        {
            return _scoringRunner.Tick();
        }
    }

    public SleepRetrainTickResult? RetrainWithRecord(SleepRecord record, int qualityLabel)
    {
        _excelService.AppendRecord(record, qualityLabel);
        return _retrainRunner.Tick();
    }
}
