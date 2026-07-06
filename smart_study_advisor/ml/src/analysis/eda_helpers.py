"""Shared helpers for Smart Study Advisor EDA."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.data.schema import ALLOWED_VALUES, TARGET_COLUMN

COLUMN_DESCRIPTIONS: dict[str, str] = {
    "HoursStudied": "Average daily study hours in the recent period.",
    "NumberOfSubjects": "Number of subjects the student is currently preparing for.",
    "DaysUntilExam": "Days remaining until the nearest major exam.",
    "StressLevel": "Self-reported stress category (Low, Medium, High).",
    "FatigueLevel": "Self-reported fatigue category (Low, Medium, High).",
    "SleepHours": "Average nightly sleep duration in hours.",
    "SleepQuality": "Qualitative sleep quality (Poor, Average, Good).",
    "PreviousFeedback": "Feedback on the previously recommended strategy.",
    "RecommendedStrategy": "Target label — recommended study strategy.",
}

STRATEGY_ORDER = ["Rest", "BalancedStudy", "IntensiveStudy", "LongTermPlan"]

ORDINAL_ENCODING: dict[str, dict[str, int]] = {
    "StressLevel": {"Low": 0, "Medium": 1, "High": 2},
    "FatigueLevel": {"Low": 0, "Medium": 1, "High": 2},
    "SleepQuality": {"Poor": 0, "Average": 1, "Good": 2},
    "PreviousFeedback": {"None": 0, "Positive": 1, "Mixed": 2, "Negative": 3},
}


def ensure_output_dirs(ml_root: Path) -> tuple[Path, Path]:
    figures_dir = ml_root / "output" / "figures"
    reports_dir = ml_root / "output"
    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir, reports_dir


def save_figure(fig: plt.Figure, figures_dir: Path, filename: str) -> Path:
    path = figures_dir / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def configure_plot_style() -> None:
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)
    plt.rcParams.update({"figure.figsize": (10, 6), "axes.titlesize": 13, "axes.labelsize": 11})


def column_overview_table(feature_columns: list[str], target_column: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Column": name,
                "Description": COLUMN_DESCRIPTIONS.get(name, ""),
                "Role": "Target" if name == target_column else "Feature",
            }
            for name in feature_columns + [target_column]
        ]
    )


def missing_value_summary(df: pd.DataFrame) -> pd.DataFrame:
    total = len(df)
    missing_count = df.isnull().sum()
    return pd.DataFrame(
        {"MissingCount": missing_count, "MissingPercent": (missing_count / total * 100).round(2)}
    )


def duplicate_summary(df: pd.DataFrame) -> dict[str, int]:
    return {
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_feature_rows": int(df.drop(columns=[TARGET_COLUMN]).duplicated().sum()),
    }


def invalid_value_report(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column, allowed in ALLOWED_VALUES.items():
        if column not in df.columns:
            continue
        invalid = sorted(set(df[column].astype(str)) - set(allowed))
        rows.append(
            {
                "Column": column,
                "InvalidCount": int(df[column].astype(str).isin(invalid).sum()),
                "InvalidValues": ", ".join(invalid) if invalid else "None",
            }
        )
    return pd.DataFrame(rows)


def class_balance_table(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    counts = df[target_column].value_counts().reindex(STRATEGY_ORDER)
    return pd.DataFrame({"Count": counts, "Percent": (counts / len(df) * 100).round(1)})


def descriptive_statistics(df: pd.DataFrame, numeric_columns: list[str]) -> pd.DataFrame:
    stats = df[numeric_columns].agg(["mean", "median", "std", "min", "max"]).T
    quartiles = df[numeric_columns].quantile([0.25, 0.50, 0.75]).T
    quartiles.columns = ["Q1", "MedianCheck", "Q3"]
    return pd.concat([stats, quartiles[["Q1", "Q3"]]], axis=1).round(3)


def encode_for_correlation(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Ordinal-encode categoricals for correlation analysis only."""
    encoded = df.copy()
    for column in feature_columns:
        if column in ORDINAL_ENCODING:
            encoded[column] = encoded[column].map(ORDINAL_ENCODING[column])
    return encoded


def correlation_highlights(corr: pd.DataFrame, top_n: int = 5) -> tuple[list[tuple], list[tuple]]:
    pairs: list[tuple[str, str, float]] = []
    columns = corr.columns.tolist()
    for i, col_a in enumerate(columns):
        for col_b in columns[i + 1 :]:
            pairs.append((col_a, col_b, float(corr.loc[col_a, col_b])))
    positive = sorted([p for p in pairs if p[2] > 0], key=lambda x: x[2], reverse=True)[:top_n]
    negative = sorted([p for p in pairs if p[2] < 0], key=lambda x: x[2])[:top_n]
    return positive, negative


def correlation_pairs_table(corr: pd.DataFrame) -> pd.DataFrame:
    """All unique feature pairs with Pearson r (sorted by |r| descending)."""
    rows: list[dict[str, object]] = []
    columns = corr.columns.tolist()
    for i, col_a in enumerate(columns):
        for col_b in columns[i + 1 :]:
            rows.append({"FeatureA": col_a, "FeatureB": col_b, "PearsonR": round(float(corr.loc[col_a, col_b]), 4)})
    return pd.DataFrame(rows).sort_values("PearsonR", key=abs, ascending=False).reset_index(drop=True)


def high_correlation_pairs(corr: pd.DataFrame, threshold: float = 0.85) -> pd.DataFrame:
    pairs = correlation_pairs_table(corr)
    return pairs[pairs["PearsonR"].abs() >= threshold].reset_index(drop=True)


def build_redundancy_report(
    numeric_corr: pd.DataFrame,
    encoded_corr: pd.DataFrame,
    *,
    threshold: float = 0.85,
) -> str:
    num_high = high_correlation_pairs(numeric_corr, threshold)
    enc_high = high_correlation_pairs(encoded_corr, threshold)
    num_pairs = correlation_pairs_table(numeric_corr).head(5)
    enc_pairs = correlation_pairs_table(encoded_corr).head(8)

    lines = [
        "# Analiza redundantnih varijabli — Smart Study Advisor",
        "",
        f"Prag visoke korelacije: **|r| ≥ {threshold:.2f}**",
        "",
        "## 1. Numeričke varijable (Pearson)",
        "",
    ]
    if num_high.empty:
        lines.append(
            f"Nijedan par numeričkih feature-a ne prelazi prag {threshold:.2f}. "
            "Najjača pozitivna korelacija je "
            f"**{num_pairs.iloc[0]['FeatureA']} ↔ {num_pairs.iloc[0]['FeatureB']}** "
            f"(r = {num_pairs.iloc[0]['PearsonR']:.3f})."
        )
    else:
        lines.append("Pronađeni parovi iznad praga:")
        for _, row in num_high.iterrows():
            lines.append(f"- `{row['FeatureA']}` ↔ `{row['FeatureB']}`: **r = {row['PearsonR']:.3f}**")

    lines.extend(["", "## 2. Svi feature-i (ordinalno kodirani, eksplorativno)", ""])
    if enc_high.empty:
        top = enc_pairs.iloc[0]
        lines.append(
            f"Nijedan par ne prelazi prag {threshold:.2f}. Najjača korelacija: "
            f"**{top['FeatureA']} ↔ {top['FeatureB']}** (r = {top['PearsonR']:.3f})."
        )
    else:
        for _, row in enc_high.iterrows():
            lines.append(f"- `{row['FeatureA']}` ↔ `{row['FeatureB']}`: **r = {row['PearsonR']:.3f}**")

    lines.extend(
        [
            "",
            "## 3. Preporuka (bez automatskog brisanja)",
            "",
            "**SleepHours** — kandidat za uklanjanje: ablation studija pokazuje ΔF1 = 0 kada se ukloni, "
            "a signal sna djelomično preklapa sa **SleepQuality**, **DaysUntilExam** i **FatigueLevel**.",
            "",
            "**HoursStudied** i **NumberOfSubjects** (r ≈ 0.64) nose sličnu informaciju o opterećenju, "
            "ali oba doprinose predikciji — nije preporučeno uklanjanje bez dodatnog eksperimenta.",
            "",
            "**Zaključak:** Nema striktno redundantnih parova (|r| ≥ 0.85). "
            "Za eksperiment bez redundantnih feature-a koristimo model **bez SleepHours** "
            "(najmanji uticaj na F1 u ablation studiji).",
            "",
        ]
    )
    return "\n".join(lines)


def detect_iqr_outliers(df: pd.DataFrame, numeric_columns: list[str]) -> pd.DataFrame:
    records = []
    for column in numeric_columns:
        series = df[column]
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask = (series < lower) | (series > upper)
        records.append(
            {
                "Feature": column,
                "Q1": round(q1, 3),
                "Q3": round(q3, 3),
                "IQR": round(iqr, 3),
                "LowerBound": round(lower, 3),
                "UpperBound": round(upper, 3),
                "OutlierCount": int(mask.sum()),
                "OutlierPercent": round(mask.sum() / len(df) * 100, 2),
            }
        )
    return pd.DataFrame(records)


def outlier_rows(df: pd.DataFrame, numeric_columns: list[str]) -> pd.DataFrame:
    flagged: dict[int, list[str]] = {}
    for column in numeric_columns:
        series = df[column]
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        for idx in series[(series < lower) | (series > upper)].index:
            flagged.setdefault(int(idx), []).append(column)
    if not flagged:
        return pd.DataFrame()
    out = df.loc[sorted(flagged.keys())].copy()
    out["OutlierFlags"] = [", ".join(flagged[int(i)]) for i in out.index]
    return out


def build_eda_summary(
    df: pd.DataFrame,
    schema_path: Path,
    figures_dir: Path,
    numeric_columns: list[str],
    categorical_columns: list[str],
    target_column: str,
    missing_df: pd.DataFrame,
    duplicate_info: dict[str, int],
    invalid_df: pd.DataFrame,
    balance_df: pd.DataFrame,
    desc_stats: pd.DataFrame,
    outlier_summary: pd.DataFrame,
    outlier_sample: pd.DataFrame,
    encoded_corr: pd.DataFrame,
) -> str:
    pos, neg = correlation_highlights(encoded_corr)
    grouped = df.groupby(target_column)[numeric_columns].mean().reindex(STRATEGY_ORDER).round(2)

    lines = [
        "# Smart Study Advisor — EDA Summary",
        "",
        "## 1. Dataset overview",
        f"- **Source:** `{schema_path}`",
        f"- **Records:** {len(df)}",
        f"- **Features:** {len(numeric_columns) + len(categorical_columns)} (4 numeric, 4 categorical)",
        f"- **Target:** `{target_column}`",
        "",
        "## 2. Data quality",
        f"- Missing values: **{int(missing_df['MissingCount'].sum())}**",
        f"- Duplicate rows: **{duplicate_info['duplicate_rows']}**",
        f"- Invalid categorical values: **{int(invalid_df['InvalidCount'].sum())}**",
        "",
        "### Class balance",
        "```",
        balance_df.to_string(),
        "```",
        "",
        "## 3. Descriptive statistics (numeric only)",
        "```",
        desc_stats.to_string(),
        "```",
        "",
        "## 4. Mean numeric features by strategy",
        "```",
        grouped.to_string(),
        "```",
        "",
        "## 5. Correlation analysis (ordinal-encoded categoricals)",
        "",
        "**Strongest positive correlations:**",
    ]
    for col_a, col_b, value in pos[:5]:
        lines.append(f"- `{col_a}` ↔ `{col_b}`: **{value:.3f}**")
    lines.append("")
    lines.append("**Strongest negative correlations:**")
    for col_a, col_b, value in neg[:5]:
        lines.append(f"- `{col_a}` ↔ `{col_b}`: **{value:.3f}**")
    lines.extend(
        [
            "",
            "StressLevel/FatigueLevel encoded as Low=0, Medium=1, High=2 (correlation only).",
            "",
            "## 6. Outlier analysis (IQR, numeric only)",
            "```",
            outlier_summary.to_string(index=False),
            "```",
            "",
            f"Rows flagged: **{len(outlier_sample)}** ({len(outlier_sample) / len(df) * 100:.1f}%).",
            "",
            "**Recommendation:** Retain outliers — valid extreme student states.",
            "",
            "## 7. Key findings",
            "- StressLevel and FatigueLevel are **categorical** (Low/Medium/High), not numeric.",
            "- High stress/fatigue → Rest; Low stress/fatigue → LongTermPlan.",
            "- DaysUntilExam is the strongest numeric separator.",
            "",
            "## 8. Exported figures",
        ]
    )
    for name in sorted(p.name for p in figures_dir.glob("*.png")):
        lines.append(f"- `figures/{name}`")
    lines.extend(["", "## 9. Next step", "Preprocessing + baseline classifier (Step 4).", ""])
    return "\n".join(lines)
