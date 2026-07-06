# Smart Study Advisor

**Predmet:** Mašinsko učenje  
**Autori:** Elmedina Marić, Aldina Kurtović  
**Institucija:** Fakultet informacionih tehnologija  

---

## Opis projekta

**Smart Study Advisor** je inteligentni sistem za preporuku strategije učenja studentima. Na osnovu akademskog opterećenja, nivoa stresa i umora, sna i vremena do ispita, aplikacija predlaže jednu od četiri strategije: odmor, uravnoteženo učenje, intenzivno učenje ili dugoročni plan.

Projekat kombinuje **klasični ML pipeline** (EDA, preprocessing, trening, evaluacija) sa **agentskom arhitekturom** u Pythonu i interaktivnim **Streamlit** dashboardom za unos podataka, prikaz rezultata i povratnu informaciju.

> Ovaj repozitorij predstavlja **seminarski rad** iz predmeta Mašinsko učenje. Dataset je sintetički generisan rule-based skriptom radi demonstracije cijelog ML workflow-a i agenta.

---

## Arhitektura agenta (Sense → Think → Act → Learn)

Agent prati ciklus percepcije, predikcije, preporuke i učenja:

| Faza | Modul | Odgovornost |
|------|-------|-------------|
| **Sense (Percepcija)** | `agent/sense.py` | Prikupljanje i validacija studentskog unosa |
| **Think (Predikcija)** | `agent/think.py` | Učitavanje ML artefakata i predikcija ciljne klase |
| **Act (Preporuka)** | `agent/act.py`, `agent/explanation.py` | Strategija, pouzdanost, objašnjenje i praktični savjeti |
| **Learn (Učenje)** | `agent/learn.py` | Pohrana predikcija i povratne informacije (CSV + SQLite) |

```
Korisnik → [Sense] → [Think] → [Act] → [Learn] → feedback baza
                ↑__________________________|
```

---

## Korištene tehnologije

| Kategorija | Tehnologija |
|------------|-------------|
| Jezik | Python 3.10+ |
| ML | scikit-learn, pandas, NumPy |
| Vizualizacija | Matplotlib, Seaborn |
| Objašnjivost | SHAP (TreeExplainer) |
| Aplikacija | Streamlit |
| Pohrana | SQLite, CSV |
| Konfiguracija | YAML |

---

## Dataset

| Svojstvo | Vrijednost |
|----------|------------|
| Fajl | `data/raw/student_study_strategy.csv` |
| Broj zapisa | 400 (čita se dinamički iz CSV-a) |
| Ulazne značajke | 8 (4 numeričke, 4 kategoričke) |
| Ciljna varijabla | `RecommendedStrategy` (4 klase) |
| Tip podataka | Sintetički, balansiran (100 zapisa po klasi) |

**Numeričke značajke:** sati učenja, broj predmeta, dani do ispita, sati sna  

**Kategoričke značajke:** nivo stresa, nivo umora, kvalitet sna, prethodna povratna informacija  

**Izlazne strategije:** Rest, BalancedStudy, IntensiveStudy, LongTermPlan  

---

## Eksploratorna analiza podataka (EDA)

U projektu je implementirana cjelovita EDA analiza:

- **Deskriptivna statistika** — sredina, medijan, standardna devijacija, kvartili za numeričke značajke
- **Distribucije i box plotovi** — po svakoj numeričkoj varijabli i po strategiji
- **Korelacijska matrica** — numeričke varijable i ordinalno kodirane značajke (eksplorativno)
- **Analiza redundantnih varijabli** — pregled visokih korelacija (prag \|r\| ≥ 0,85)
- **Outlier analiza (IQR)** — detekcija ekstremnih vrijednosti; na sintetičkom datasetu nema outliera
- **Balans ciljne klase** — ravnomjerna raspodjela od 25% po klasi

Izlazi: `ml/output/figures/`, `ml/output/eda_summary.md`, `ml/output/redundancy_analysis.md`

---

## Model evaluacija i eksperimenti

### Usporedba modela

Testirana su četiri klasifikatora na stratificiranom test skupu (70/30):

- Nasumična šuma (Random Forest) — **najbolji model**
- Logistička regresija
- Stablo odlučivanja
- Gradijentni boosting

Metrike: točnost, preciznost, odziv (recall), F1 (macro).

### Dodatne analize

| Analiza | Opis |
|---------|------|
| Matrica konfuzije | Hold-out test najboljeg modela |
| Unakrsna validacija | 5-fold stratificirana CV |
| Studija uklanjanja značajki | Marginalni doprinos svake značajke (ablation) |
| Važnost značajki | Gini importance (Random Forest) |
| Poređenje eksperimenata | Train/test omjeri (70/30, 75/25, 80/20, 85/15), model bez redundantnih značajki, uticaj outliera |

Rezultati eksperimenata: `ml/output/experiments/experiment_comparison.csv`

---

## Pokretanje projekta

### Preduvjeti

- Python 3.10 ili noviji
- Git (opcionalno)

### Instalacija i pokretanje

```bash
cd smart_study_advisor
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

Aplikacija se otvara u browseru na adresi `http://localhost:8501`.

### Stranice aplikacije

| Stranica | Sadržaj |
|----------|---------|
| **Početna** | Unos konteksta, preporuka strategije, objašnjenje modela, povratna informacija |
| **EDA analiza** | Deskriptivna statistika, grafikoni, korelacije |
| **Evaluacija** | Usporedba modela, CV, uklanjanje značajki, eksperimenti |
| **O projektu** | Metodologija, pipeline, ograničenja |

---

## Struktura foldera

```
smart_study_advisor/
├── app/
│   ├── streamlit_app.py          # Početna stranica (dashboard)
│   ├── pages/                    # EDA, evaluacija, o projektu
│   └── components/               # UI, putanje, učitavanje podataka, metrike
├── agent/                        # Sense → Think → Act → Learn
│   ├── sense.py
│   ├── think.py
│   ├── act.py
│   ├── learn.py
│   └── smart_study_agent.py
├── data/
│   ├── raw/                      # student_study_strategy.csv
│   ├── processed/                # Train/test splitovi
│   └── feedback/                 # Povratne informacije (CSV, SQLite)
├── models/                       # preprocessor.pkl, label_encoder.pkl, best_model.pkl
├── ml/
│   ├── config/                   # Schema dataseta
│   ├── scripts/                  # EDA, trening, evaluacija
│   ├── output/                   # Figure, izvještaji, metrike
│   └── src/                      # Preprocessing, training, analysis
├── reports/
│   └── final_ml_report.md        # Finalni ML izvještaj
├── requirements.txt
└── README.md
```

---

## ML pipeline (offline, opcionalno)

Za regeneraciju modela i izvještaja:

```bash
cd ml
python scripts/validate_dataset.py
python scripts/run_eda.py
python scripts/preprocess_dataset.py
python scripts/train_models.py
python scripts/feature_importance.py
python scripts/ablation_study.py
python scripts/run_cross_validation.py
python scripts/run_ml_experiments.py
```

> Ne pokrećite pipeline ponovo osim ako namjerno osvježavate modele i izvještaje.

---

## Dokumentacija

| Dokument | Lokacija |
|----------|----------|
| Finalni ML izvještaj | [`reports/final_ml_report.md`](reports/final_ml_report.md) |
| EDA sažetak | [`ml/output/eda_summary.md`](ml/output/eda_summary.md) |
| Analiza redundantnosti | [`ml/output/redundancy_analysis.md`](ml/output/redundancy_analysis.md) |
| Eksperimenti | [`ml/output/experiments/experiment_summary.md`](ml/output/experiments/experiment_summary.md) |
| Dataset schema | [`ml/config/dataset_schema.yaml`](ml/config/dataset_schema.yaml) |

---

## Ograničenja

- Dataset je **sintetički** — visoki skorovi odražavaju jasno definisana pravila generisanja
- **Nema automatskog ponovnog treninga** — Learn faza pohranjuje povratne informacije za budući razvoj
- Model koristi sklearn zadane hiperparametre (bez tuninga)
- Broj zapisa u ML datasetu (400) **ne raste** pri korištenju aplikacije; novi unosi idu u feedback bazu

---

## Buduća unapređenja

- Prikupljanje **stvarnih studentskih podataka** (ankete, wearable uređaji)
- **Automatski ponovni trening** modela na osnovu povratnih informacija
- **Hyperparameter tuning** (GridSearch, Optuna)
- Integracija sa **kalendarom ispita** i notifikacijama
- Višejezična podrška u produkcijskoj verziji
- A/B testiranje strategija preporuke
- Deployment na cloud (Docker, Azure, AWS)

---

## Autori

**Elmedina Marić** · **Aldina Kurtović**  
Fakultet informacionih tehnologija · Mašinsko učenje · 2025/2026

---

*Seminarski rad — Smart Study Advisor*
