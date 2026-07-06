using System.Text.Json;

namespace Sleep.Infrastructure;

public class SleepPolicyStateStore
{
    private readonly string _filePath;
    private readonly object _sync = new();

    public SleepPolicyStateStore(string? filePath = null)
    {
        _filePath = filePath ?? Path.Combine(AppContext.BaseDirectory, "sleep-policy.json");
    }

    public SleepPolicyState Load()
    {
        lock (_sync)
        {
            if (!File.Exists(_filePath))
            {
                return new SleepPolicyState();
            }

            try
            {
                var json = File.ReadAllText(_filePath);
                return JsonSerializer.Deserialize<SleepPolicyState>(json) ?? new SleepPolicyState();
            }
            catch
            {
                return new SleepPolicyState();
            }
        }
    }

    public void Save(SleepPolicyState state)
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
