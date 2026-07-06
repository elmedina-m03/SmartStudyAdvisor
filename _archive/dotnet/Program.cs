using Sleep.Application;
using Sleep.Domain;
using Sleep.Infrastructure;
using Sleep.Web;

var builder = WebApplication.CreateBuilder(args);
builder.Logging.AddConsole();

builder.Services.AddSingleton<ExcelDatasetService>();
builder.Services.AddSingleton<SleepPolicyStateStore>();
builder.Services.AddSingleton<SleepQualityPolicy>();
builder.Services.AddSingleton<SleepLearner>();
builder.Services.AddSingleton<RetrainPolicyStateStore>();
builder.Services.AddSingleton<RetrainPolicy>();
builder.Services.AddSingleton<RetrainLearner>();
builder.Services.AddSingleton<SleepAgentOrchestrator>();
builder.Services.AddHostedService<SleepAgentBackgroundWorker>();

var app = builder.Build();

app.UseDefaultFiles();
app.UseStaticFiles();

app.MapPost("/predict", (SleepDataRequest request, SleepAgentOrchestrator orchestrator) =>
{
    var sleepRecord = new SleepRecord(
        age: request.Age,
        gender: request.Gender,
        dailySteps: request.DailySteps,
        physicalActivity: request.PhysicalActivity,
        stressLevel: request.StressLevel,
        sleepDuration: request.SleepDuration
    );

    var result = orchestrator.PredictNow(sleepRecord);
    
    if (result == null)
    {
        return Results.BadRequest("Failed to process sleep record");
    }

    return Results.Ok(new PredictResponse
    {
        Quality = result.PredictedQuality.ToString(),
        Advice = result.Advice
    });
});

app.MapPost("/retrain", (RetrainRequest request, SleepAgentOrchestrator orchestrator, ILogger<Program> logger) =>
{
    if (request == null)
    {
        logger.LogWarning("Retrain endpoint called with null request");
        return Results.BadRequest("Request body is required");
    }

    try
    {
        var sleepRecord = new SleepRecord(
            age: request.Age,
            gender: request.Gender,
            dailySteps: request.DailySteps,
            physicalActivity: request.PhysicalActivity,
            stressLevel: request.StressLevel,
            sleepDuration: request.SleepDuration
        );

        var result = orchestrator.RetrainWithRecord(sleepRecord, request.QualityOfSleep);

        // Always return success since the record was appended successfully
        // The Tick() result is informational (whether retraining was triggered)
        var message = result != null && result.Decision == RetrainDecision.Retrain
            ? $"Dataset updated. {result.Message}"
            : "Dataset updated successfully.";

        return Results.Ok(new RetrainResponse
        {
            Message = message
        });
    }
    catch (InvalidOperationException ex) when (ex.InnerException is IOException || ex.Message.Contains("file") || ex.Message.Contains("Excel"))
    {
        logger.LogError(ex, "File access error while processing retrain request");
        var innerMessage = ex.InnerException?.Message ?? ex.Message;
        return Results.BadRequest($"Cannot save to Excel file. Please close the file if it's open: {innerMessage}");
    }
    catch (IOException ex)
    {
        logger.LogError(ex, "IOException while processing retrain request");
        return Results.BadRequest($"Cannot save to Excel file. Please close the file if it's open: {ex.Message}");
    }
    catch (Exception ex)
    {
        logger.LogError(ex, "Unexpected error during retraining");
        return Results.BadRequest($"Error during retraining: {ex.Message}");
    }
});

app.Run();

