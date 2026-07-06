namespace SharedCore;

public interface IActuator<TAction, TResult>
{
    TResult Execute(TAction action);
}

