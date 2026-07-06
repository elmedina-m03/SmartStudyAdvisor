namespace Student.Domain;

public readonly record struct StrategyExperience
{
    public StudentContext Input { get; init; }
    public StudyStrategy Predicted { get; init; }
    public StudyStrategy Actual { get; init; }

    public StrategyExperience(StudentContext input, StudyStrategy predicted, StudyStrategy actual)
    {
        Input = input;
        Predicted = predicted;
        Actual = actual;
    }
}
