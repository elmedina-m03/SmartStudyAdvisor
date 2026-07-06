namespace SharedCore;

public interface IPolicy<TPercept, TAction>
{
    TAction SelectAction(TPercept percept);
}

