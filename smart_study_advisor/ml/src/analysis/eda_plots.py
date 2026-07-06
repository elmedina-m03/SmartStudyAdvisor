"""Figure generation for Smart Study Advisor EDA."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .eda_helpers import STRATEGY_ORDER, save_figure

LEVEL_ORDER = ["Low", "Medium", "High"]
SLEEP_ORDER = ["Poor", "Average", "Good"]
FEEDBACK_ORDER = ["None", "Positive", "Mixed", "Negative"]

CATEGORICAL_ORDERS: dict[str, list[str]] = {
    "StressLevel": LEVEL_ORDER,
    "FatigueLevel": LEVEL_ORDER,
    "SleepQuality": SLEEP_ORDER,
    "PreviousFeedback": FEEDBACK_ORDER,
}


def plot_missing_values(df: pd.DataFrame, figures_dir: Path) -> Path:
    summary = df.isnull().sum()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=summary.index, y=summary.values, ax=ax, hue=summary.index, legend=False)
    ax.set_title("Missing Values per Column")
    ax.tick_params(axis="x", rotation=35)
    return save_figure(fig, figures_dir, "01_missing_values.png")


def plot_target_distribution(df: pd.DataFrame, target_column: str, figures_dir: Path) -> Path:
    counts = df[target_column].value_counts().reindex(STRATEGY_ORDER)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=counts.index, y=counts.values, ax=ax, hue=counts.index, palette="Set2", legend=False)
    ax.set_title("RecommendedStrategy — Class Distribution")
    ax.set_ylabel("Count")
    for idx, value in enumerate(counts.values):
        ax.text(idx, value + 1, str(value), ha="center", fontsize=10)
    return save_figure(fig, figures_dir, "02_target_distribution.png")


def plot_numeric_histogram(df: pd.DataFrame, column: str, figures_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df[column], kde=True, ax=ax, color="#4C72B0", bins=20)
    ax.set_title(f"Histogram — {column}")
    return save_figure(fig, figures_dir, f"03_hist_{column}.png")


def plot_numeric_boxplot_by_strategy(
    df: pd.DataFrame, column: str, target_column: str, figures_dir: Path
) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(
        data=df, x=target_column, y=column, order=STRATEGY_ORDER,
        hue=target_column, dodge=False, ax=ax, palette="Set3", legend=False,
    )
    ax.set_title(f"Boxplot — {column} by RecommendedStrategy")
    ax.tick_params(axis="x", rotation=15)
    return save_figure(fig, figures_dir, f"04_box_{column}.png")


def plot_categorical_count(df: pd.DataFrame, column: str, figures_dir: Path) -> Path:
    order = CATEGORICAL_ORDERS.get(column, sorted(df[column].unique()))
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.countplot(data=df, x=column, order=order, ax=ax, hue=column, palette="pastel", legend=False)
    ax.set_title(f"Count Plot — {column}")
    return save_figure(fig, figures_dir, f"05_count_{column}.png")


def plot_categorical_vs_strategy(
    df: pd.DataFrame, column: str, target_column: str, figures_dir: Path
) -> Path:
    order = CATEGORICAL_ORDERS.get(column, sorted(df[column].unique()))
    ct = pd.crosstab(df[column], df[target_column]).reindex(index=order, columns=STRATEGY_ORDER, fill_value=0)
    fig, ax = plt.subplots(figsize=(9, 5))
    ct.plot(kind="bar", ax=ax, colormap="Set2")
    ax.set_title(f"{column} vs RecommendedStrategy")
    ax.legend(title="Strategy", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    return save_figure(fig, figures_dir, f"06_{column}_vs_strategy.png")


def plot_encoded_correlation_matrix(encoded_df: pd.DataFrame, columns: list[str], figures_dir: Path) -> Path:
    corr = encoded_df[columns].corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax, square=True)
    ax.set_title("Correlation Matrix (Ordinal-Encoded Features)")
    fig.tight_layout()
    return save_figure(fig, figures_dir, "07_correlation_matrix_encoded.png")


def plot_numeric_correlation_matrix(df: pd.DataFrame, numeric_columns: list[str], figures_dir: Path) -> Path:
    corr = df[numeric_columns].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax, square=True, vmin=-1, vmax=1)
    ax.set_title("Correlation Matrix (Numeric Features Only)")
    fig.tight_layout()
    return save_figure(fig, figures_dir, "14_correlation_matrix_numeric.png")


def plot_outlier_summary(outlier_summary: pd.DataFrame, figures_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(
        data=outlier_summary, x="Feature", y="OutlierCount",
        ax=ax, hue="Feature", palette="Reds_r", legend=False,
    )
    ax.set_title("IQR Outlier Count (Numeric Features Only)")
    ax.tick_params(axis="x", rotation=20)
    return save_figure(fig, figures_dir, "08_iqr_outlier_counts.png")


def plot_feature_vs_strategy(
    df: pd.DataFrame,
    feature: str,
    target_column: str,
    figures_dir: Path,
    filename: str,
    *,
    numeric: bool = False,
) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    if numeric:
        sns.boxplot(
            data=df, x=target_column, y=feature, order=STRATEGY_ORDER,
            hue=target_column, dodge=False, ax=ax, palette="Set2", legend=False,
        )
    else:
        order = CATEGORICAL_ORDERS.get(feature, sorted(df[feature].unique()))
        ct = pd.crosstab(df[target_column], df[feature]).reindex(index=STRATEGY_ORDER, columns=order, fill_value=0)
        ct.plot(kind="bar", ax=ax, colormap="Set2")
        ax.set_xlabel("RecommendedStrategy")
        ax.legend(title=feature, bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.set_title(f"{feature} vs RecommendedStrategy")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    return save_figure(fig, figures_dir, filename)
