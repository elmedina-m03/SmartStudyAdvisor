namespace SharedCore;

public interface ILearningComponent<TExperience>
{
    void Learn(TExperience experience);
}

