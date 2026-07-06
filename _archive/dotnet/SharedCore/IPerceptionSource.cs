namespace SharedCore;

public interface IPerceptionSource<TPercept>
{
    TPercept Observe();
}

