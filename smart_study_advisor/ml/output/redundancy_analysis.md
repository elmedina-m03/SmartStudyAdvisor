# Analiza redundantnih varijabli — Smart Study Advisor

Prag visoke korelacije: **|r| ≥ 0.85**

## 1. Numeričke varijable (Pearson)

Nijedan par numeričkih feature-a ne prelazi prag 0.85. Najjača pozitivna korelacija je **DaysUntilExam ↔ SleepHours** (r = 0.748).

## 2. Svi feature-i (ordinalno kodirani, eksplorativno)

Nijedan par ne prelazi prag 0.85. Najjača korelacija: **DaysUntilExam ↔ SleepHours** (r = 0.748).

## 3. Preporuka (bez automatskog brisanja)

**SleepHours** — kandidat za uklanjanje: ablation studija pokazuje ΔF1 = 0 kada se ukloni, a signal sna djelomično preklapa sa **SleepQuality**, **DaysUntilExam** i **FatigueLevel**.

**HoursStudied** i **NumberOfSubjects** (r ≈ 0.64) nose sličnu informaciju o opterećenju, ali oba doprinose predikciji — nije preporučeno uklanjanje bez dodatnog eksperimenta.

**Zaključak:** Nema striktno redundantnih parova (|r| ≥ 0.85). Za eksperiment bez redundantnih feature-a koristimo model **bez SleepHours** (najmanji uticaj na F1 u ablation studiji).
