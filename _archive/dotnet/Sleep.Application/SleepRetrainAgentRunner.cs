using SharedCore;
using Sleep.Domain;

namespace Sleep.Application;

/// <summary>
/// Agent runner for model retraining decisions.
/// Each Tick() call executes one complete agent iteration using the universal SoftwareAgent architecture.
/// </summary>
public class SleepRetrainAgentRunner
{
    private readonly SoftwareAgent<ProcessedRecordsCounter, RetrainDecision, RetrainResult, RetrainExperience> _agent;
    private readonly RetrainResultCapturingActuator _resultCapturingActuator;

    public SleepRetrainAgentRunner(
        IPerceptionSource<ProcessedRecordsCounter> perception,
        IPolicy<ProcessedRecordsCounter, RetrainDecision> policy,
        IActuator<RetrainDecision, RetrainResult> actuator,
        Func<ProcessedRecordsCounter, RetrainDecision> groundTruthProvider,
        Func<ProcessedRecordsCounter, RetrainDecision, RetrainResult, RetrainExperience>? experienceBuilder = null,
        ILearningComponent<RetrainExperience>? learner = null)
    {
        _resultCapturingActuator = new RetrainResultCapturingActuator(actuator);

        RetrainExperience CreateExperience(ProcessedRecordsCounter percept, RetrainDecision action, RetrainResult result)
        {
            var actual = groundTruthProvider(percept);
            return new RetrainExperience(percept, action, actual);
        }

        _agent = new SoftwareAgent<ProcessedRecordsCounter, RetrainDecision, RetrainResult, RetrainExperience>(
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
    /// <returns>SleepRetrainTickResult if work was processed, null if no work available.</returns>
    public SleepRetrainTickResult? Tick()
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

            return new SleepRetrainTickResult(
                result.Value.Decision,
                result.Value.NewVersion,
                result.Value.Message
            );
        }
        catch (InvalidOperationException)
        {
            // Idempotency: safely return null when no percepts are available
            return null;
        }
    }
}

