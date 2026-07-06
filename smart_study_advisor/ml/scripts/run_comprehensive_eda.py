"""Run the full EDA pipeline: figures + markdown report."""

from __future__ import annotations

import sys
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_ROOT))

import pandas as pd

from src.analysis.eda_helpers import (
    build_markdown_report,
    column_overview_table,
    configure_plot_style,
    descriptive_statistics,
    detect_iqr_outliers,
    duplicate_summary,
    ensure_output_dirs,
    missing_value_summary,
    outlier_rows,
)
from src.analysis.eda_plots import (
    plot_boxplots_by_strategy,
    plot_categorical_counts,
    plot_correlation_matrix,
    plot_individual_boxplots,
    plot_missing_values,
    plot_numeric_histograms,
    plot_numeric_vs_strategy,
    plot_outlier_summary,
    plot_pairplot,
    plot_sleep_quality_vs_strategy,
    plot_target_distribution,
)
from src.data.schema import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    load_dataset,
    load_schema,
)


def main() -> None:
    configure_plot_style()
    figures_dir, reports_dir = ensure_output_dirs(ML_ROOT)
    schema = load_schema()
    df = load_dataset()

    missing_df = missing_value_summary(df)
    duplicate_info = duplicate_summary(df)
    desc_stats = descriptive_statistics(df, NUMERIC_FEATURES)
    outlier_summary = detect_iqr_outliers(df, NUMERIC_FEATURES)
    outlier_sample = outlier_rows(df, NUMERIC_FEATURES)

    plot_missing_values(df, figures_dir)
    plot_target_distribution(df, TARGET_COLUMN, figures_dir)
    plot_numeric_histograms(df, NUMERIC_FEATURES, figures_dir)
    plot_categorical_counts(df, CATEGORICAL_FEATURES, figures_dir)
    plot_correlation_matrix(df, NUMERIC_FEATURES, figures_dir)
    plot_boxplots_by_strategy(df, NUMERIC_FEATURES, TARGET_COLUMN, figures_dir)
    plot_individual_boxplots(df, NUMERIC_FEATURES, TARGET_COLUMN, figures_dir)
    plot_pairplot(df, NUMERIC_FEATURES, TARGET_COLUMN, figures_dir)
    plot_sleep_quality_vs_strategy(df, TARGET_COLUMN, figures_dir)
    plot_numeric_vs_strategy(df, "StressLevel", TARGET_COLUMN, figures_dir, "09_stress_level_vs_strategy.png")
    plot_numeric_vs_strategy(df, "FatigueLevel", TARGET_COLUMN, figures_dir, "10_fatigue_level_vs_strategy.png")
    plot_numeric_vs_strategy(df, "DaysUntilExam", TARGET_COLUMN, figures_dir, "11_days_until_exam_vs_strategy.png")
    plot_outlier_summary(outlier_summary, figures_dir)

    report = build_markdown_report(
        df=df,
        schema_path=schema.source_path,
        figures_dir=figures_dir,
        numeric_columns=NUMERIC_FEATURES,
        categorical_columns=CATEGORICAL_FEATURES,
        target_column=TARGET_COLUMN,
        missing_df=missing_df,
        duplicate_info=duplicate_info,
        desc_stats=desc_stats,
        outlier_summary=outlier_summary,
        outlier_sample=outlier_sample,
    )
    report_path = reports_dir / "eda_report.md"
    report_path.write_text(report, encoding="utf-8")

    print(f"EDA complete — {len(df)} rows analysed")
    print(f"Figures: {figures_dir} ({len(list(figures_dir.glob('*.png')))} files)")
    print(f"Report:  {report_path}")
    print()
    print("Target distribution:")
    print(df[TARGET_COLUMN].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
