using Sleep.Application;

namespace Sleep.Web;

public class SleepAgentBackgroundWorker : BackgroundService
{
    private static readonly TimeSpan IdleDelay = TimeSpan.FromSeconds(2);
    private static readonly TimeSpan ActiveDelay = TimeSpan.FromMilliseconds(150);

    private readonly SleepAgentOrchestrator _orchestrator;
    private readonly ILogger<SleepAgentBackgroundWorker> _logger;

    public SleepAgentBackgroundWorker(
        SleepAgentOrchestrator orchestrator,
        ILogger<SleepAgentBackgroundWorker> logger)
    {
        _orchestrator = orchestrator;
        _logger = logger;
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        _logger.LogInformation("Sleep agent background worker started.");

        while (!stoppingToken.IsCancellationRequested)
        {
            SleepScoringTickResult? result = null;

            try
            {
                result = _orchestrator.ProcessScoringTick();
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error while processing scoring tick");
            }

            var delay = result == null ? IdleDelay : ActiveDelay;

            try
            {
                await Task.Delay(delay, stoppingToken);
            }
            catch (TaskCanceledException)
            {
                break;
            }
        }
    }
}
