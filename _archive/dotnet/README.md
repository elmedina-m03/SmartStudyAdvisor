# Archived .NET Sleep Agent

This folder contains the legacy C# implementation of the SleepQualityAgent project:

- `Program.cs`, ASP.NET endpoints (`/predict`, `/retrain`)
- `Sleep.Application`, `Sleep.Infrastructure`, `Sleep.Domain`, `Sleep.Web`
- `SharedCore` (Sense–Think–Act–Learn abstractions)
- `Student.Domain` (C# domain types, superseded by Python `agent/types.py`)
- `SleepAgent.Core`, `wwwroot/`, solution and project files

The active **Smart Study Advisor** is now Python-first under `smart_study_advisor/`.

These files are kept for reference only and are not required to run the ML agent.
