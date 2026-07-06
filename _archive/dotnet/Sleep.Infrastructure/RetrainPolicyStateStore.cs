using System.Text.Json;

namespace Sleep.Infrastructure;

public class RetrainPolicyStateStore
{
    private readonly string _filePath;
    private readonly object _sync = new();

    public RetrainPolicyStateStore(string? filePath = null)
    {
        _filePath = filePath ?? Path.Combine(AppContext.BaseDirectory, "retrain-policy.json");
    }

    public RetrainPolicyState Load()
    {
        lock (_sync)
        {
            if (!File.Exists(_filePath))
            {
                return new RetrainPolicyState();
            }

            try
            {
                var json = File.ReadAllText(_filePath);
                return JsonSerializer.Deserialize<RetrainPolicyState>(json) ?? new RetrainPolicyState();
            }
            catch
            {
                return new RetrainPolicyState();
            }
        }
    }

    public void Save(RetrainPolicyState state)
    {
        lock (_sync)
        {
            var json = JsonSerializer.Serialize(state, new JsonSerializerOptions
            {
                WriteIndented = true
            });
            File.WriteAllText(_filePath, json);
        }
    }
}
