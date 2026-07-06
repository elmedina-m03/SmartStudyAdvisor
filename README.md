# Smart Study Advisor

**Predmet:** Mašinsko učenje  
**Autori:** Elmedina Marić, Aldina Kurtović  
**Institucija:** Fakultet informacionih tehnologija  

Inteligentni sistem koji studentima preporučuje strategiju učenja pomoću modela mašinskog učenja i agentske arhitekture **Sense → Think → Act → Learn**.

> Seminarski rad iz predmeta Mašinsko učenje.

---

## Pokretanje

Sav aktivni kod nalazi se u folderu [`smart_study_advisor/`](smart_study_advisor/):

```bash
cd smart_study_advisor
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

Puna dokumentacija: **[`smart_study_advisor/README.md`](smart_study_advisor/README.md)**

---

## Sadržaj repozitorija

| Lokacija | Opis |
|----------|------|
| [`smart_study_advisor/`](smart_study_advisor/) | Python agent, Streamlit aplikacija, ML pipeline |
| [`smart_study_advisor/app/`](smart_study_advisor/app/) | Dashboard (preporuka, EDA, evaluacija) |
| [`smart_study_advisor/ml/`](smart_study_advisor/ml/) | EDA, trening, evaluacija, eksperimenti |
| [`smart_study_advisor/models/`](smart_study_advisor/models/) | Trenirani model artefakti |
| [`smart_study_advisor/reports/`](smart_study_advisor/reports/) | Finalni ML izvještaj |
| [`_archive/dotnet/`](_archive/dotnet/) | Arhivirani legacy C# kod |

---

## Autori

**Elmedina Marić** · **Aldina Kurtović**
