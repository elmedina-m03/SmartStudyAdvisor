using SharedCore;
using Sleep.Domain;
using Sleep.Infrastructure;

namespace Sleep.Application;

/// <summary>
/// Agent runner for model retraining decisions.
/// Each Tick() call executes one complete agent iteration using the universal SoftwareAgent architecture.
/// </summary>
public class SleepRetrainAgentRunner
{
    private readonly SoftwareAgent<ProcessedRecordsCounter, RetrainDecision, RetrainResult, RetrainExperience> _agent;
    private readonly RetrainResultCapturingActuator _resultCapturingActuator;
    private readonly Func<int> _counterProvider;

    public SleepRetrainAgentRunner(
        Func<int> counterProvider,
        int retrainThreshold = 3)
    {
        _counterProvider = counterProvider;

        var perception = new ProcessedRecordsPerception(counterProvider);
        var policy = new RetrainPolicy(retrainThreshold);
        var actuator = new RetrainActuator();
        var learner = new RetrainLearner();

        _resultCapturingActuator = new RetrainResultCapturingActuator(actuator);

        RetrainExperience CreateExperience(ProcessedRecordsCounter percept, RetrainDecision action, RetrainResult result)
        {
            // For retraining, actual decision matches predicted (simplified for educational purposes)
            var actual = action;
            return new RetrainExperience(percept, action, actual);
        }

        _agent = new SoftwareAgent<ProcessedRecordsCounter, RetrainDecision, RetrainResult, RetrainExperience>(
            perception,
            policy,
            _resultCapturingActuator,
            CreateExperience,
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

