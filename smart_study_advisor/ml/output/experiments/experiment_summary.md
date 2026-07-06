# Konačna analiza ML eksperimenata — Smart Study Advisor

Model: **RandomForestClassifier** (sklearn zadane vrijednosti) · stratificirani split · random_state=42

## Tabela svih eksperimenata

```
experiment_id                          description            category  train_ratio  test_ratio  n_train  n_test  n_features removed_features         outlier_handling  accuracy  precision_macro  recall_macro  f1_macro
  split_70_30             Originalni model (70/30)    train_test_omjer         0.70        0.30      280     120           8                —                 zadržano    0.9917           0.9919        0.9917    0.9917
  split_75_25                     Train/test 75/25    train_test_omjer         0.75        0.25      300     100           8                —                 zadržano    0.9900           0.9904        0.9900    0.9900
  split_80_20                     Train/test 80/20    train_test_omjer         0.80        0.20      320      80           8                —                 zadržano    1.0000           1.0000        1.0000    1.0000
  split_85_15                     Train/test 85/15    train_test_omjer         0.85        0.15      340      60           8                —                 zadržano    1.0000           1.0000        1.0000    1.0000
 no_redundant Bez SleepHours (redundantan feature) redundantni_feature         0.70        0.30      280     120           7       SleepHours                 zadržano    0.9917           0.9919        0.9917    0.9917
  no_outliers        Nakon uklanjanja IQR outliera            outlieri         0.70        0.30      280     120           8                — uklonjeno 0 redova (IQR)    0.9917           0.9919        0.9917    0.9917
```

## 1. Train/test omjer

- Trenutno korišteni omjer u produkciji: **70/30** (F1 = 0.9917).
- Najbolji omjer u ovom poređenju: **80/20** (F1 = 1.0000).

## 2. Redundantni feature-i

- **SleepHours** uklonjen u eksperimentu `no_redundant` (preporuka iz korelacijske i ablation analize).
- F1 prije: 0.9917 → poslije: 0.9917.

## 3. Outlieri (IQR)

- Ukupno IQR outliera u datasetu: **0**.
- Eksperiment uklanjanja outliera daje **identične rezultate** kao original (nema šta ukloniti).

## 4. Zaključak

**Najbolja konfiguracija:** `split_80_20` — Train/test 80/20
- Accuracy: 1.0000
- Precision (macro): 1.0000
- Recall (macro): 1.0000
- F1 (macro): **1.0000**

Za produkciju zadržavamo **70/30** jer je standard u projektu, razlike između omjera su minimalne na sintetičkom datasetu od 400 zapisa, a veći test skup daje pouzdaniju hold-out evaluaciju.
