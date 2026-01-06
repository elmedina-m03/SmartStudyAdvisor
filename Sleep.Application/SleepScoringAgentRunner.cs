using SharedCore;
using Sleep.Domain;

namespace Sleep.Application;

/// <summary>
/// Agent runner for sleep quality scoring.
/// Each Tick() call executes one complete agent iteration using the universal SoftwareAgent architecture.
/// </summary>
public class SleepScoringAgentRunner
{
    private readonly SoftwareAgent<SleepRecord, SleepQualityClass, SleepAssessment, SleepExperience> _agent;
    private readonly ResultCapturingActuator _resultCapturingActuator;
    private int _tickCounter = 0;

    public SleepScoringAgentRunner(
        IPerceptionSource<SleepRecord> perception,
        IPolicy<SleepRecord, SleepQualityClass> policy,
        IActuator<SleepQualityClass, SleepAssessment> actuator,
        Func<SleepRecord, SleepQualityClass> groundTruthProvider,
        Func<SleepRecord, SleepQualityClass, SleepAssessment, SleepExperience>? experienceBuilder = null,
        ILearningComponent<SleepExperience>? learner = null)
    {
        _resultCapturingActuator = new ResultCapturingActuator(actuator);

        SleepExperience CreateExperience(SleepRecord percept, SleepQualityClass action, SleepAssessment result)
        {
            var actual = groundTruthProvider(percept);
            return new SleepExperience(percept, action, actual);
        }

        _agent = new SoftwareAgent<SleepRecord, SleepQualityClass, SleepAssessment, SleepExperience>(
            perception,
            policy,
            _resultCapturingActuator,
            experienceBuilder ?? CreateExperience,
            learner
        );
    }

    /// <summary>
    /// Executes one agent iteration (Tick = one complete agent cycle).
    /// Idempotent: safe to call multiple times; returns null when no work is available.
    /// </summary>
    /// <returns>SleepScoringTickResult if work was processed, null if no records available.</returns>
    public SleepScoringTickResult? Tick()
    {
        try
        {
            // Execute one complete agent cycle using the universal SoftwareAgent architecture.
            // The agent.Step() method internally performs: Sense → Think → Act → Learn
            _agent.Step();

            // Retrieve the result from the actuator
            var result = _resultCapturingActuator.GetLastResult();
            if (result == null)
            {
                return null;
            }

            _tickCounter++;
            var recordIdentifier = $"Record-{_tickCounter}";

            return new SleepScoringTickResult(recordIdentifier, result.Value.Quality, result.Value.Advice);
        }
        catch (InvalidOperationException)
        {
            // Idempotency: safely return null when no percepts are available (queue empty)
            return null;
        }
    }
}

