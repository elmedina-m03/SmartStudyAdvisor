namespace SharedCore;

public class SoftwareAgent<TPercept, TAction, TResult, TExperience>
{
    protected readonly IPerceptionSource<TPercept> _perception;
    protected readonly IPolicy<TPercept, TAction> _policy;
    protected readonly IActuator<TAction, TResult> _actuator;
    protected readonly ILearningComponent<TExperience>? _learner;
    protected readonly Func<TPercept, TAction, TResult, TExperience>? _experienceBuilder;

    public SoftwareAgent(
        IPerceptionSource<TPercept> perception,
        IPolicy<TPercept, TAction> policy,
        IActuator<TAction, TResult> actuator,
        Func<TPercept, TAction, TResult, TExperience>? experienceBuilder = null,
        ILearningComponent<TExperience>? learner = null)
    {
        _perception = perception;
        _policy = policy;
        _actuator = actuator;
        _experienceBuilder = experienceBuilder;
        _learner = learner;
    }

    public virtual void Step()
    {
        // 1. SENSE
        var percept = _perception.Observe();

        // 2. THINK
        var action = _policy.SelectAction(percept);

        // 3. ACT
        var result = _actuator.Execute(action);

        // 4. LEARN (optional)
        if (_learner != null && _experienceBuilder != null)
        {
            var experience = _experienceBuilder(percept, action, result);
            _learner.Learn(experience);
        }
    }
}

