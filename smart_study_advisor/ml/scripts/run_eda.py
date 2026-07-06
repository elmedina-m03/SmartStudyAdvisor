"""Run complete Smart Study Advisor EDA and export figures + summary."""

from __future__ import annotations

import sys
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_ROOT))

from src.analysis.eda_helpers import (
    build_eda_summary,
    build_redundancy_report,
    class_balance_table,
    configure_plot_style,
    correlation_pairs_table,
    descriptive_statistics,
    detect_iqr_outliers,
    duplicate_summary,
    encode_for_correlation,
    ensure_output_dirs,
    invalid_value_report,
    missing_value_summary,
    outlier_rows,
)
from src.analysis.eda_plots import (
    plot_categorical_count,
    plot_categorical_vs_strategy,
    plot_encoded_correlation_matrix,
    plot_feature_vs_strategy,
    plot_missing_values,
    plot_numeric_boxplot_by_strategy,
    plot_numeric_correlation_matrix,
    plot_numeric_histogram,
    plot_outlier_summary,
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


def run_eda() -> Path:
    configure_plot_style()
    figures_dir, output_dir = ensure_output_dirs(ML_ROOT)
    schema = load_schema()
    df = load_dataset()

    for old in figures_dir.glob("*.png"):
        old.unlink()

    missing_df = missing_value_summary(df)
    dup_info = duplicate_summary(df)
    invalid_df = invalid_value_report(df)
    balance_df = class_balance_table(df, TARGET_COLUMN)
    desc_stats = descriptive_statistics(df, NUMERIC_FEATURES)
    outlier_summary = detect_iqr_outliers(df, NUMERIC_FEATURES)
    outlier_sample = outlier_rows(df, NUMERIC_FEATURES)

    plot_missing_values(df, figures_dir)
    plot_target_distribution(df, TARGET_COLUMN, figures_dir)

    for col in NUMERIC_FEATURES:
        plot_numeric_histogram(df, col, figures_dir)
        plot_numeric_boxplot_by_strategy(df, col, TARGET_COLUMN, figures_dir)

    for col in CATEGORICAL_FEATURES:
        plot_categorical_count(df, col, figures_dir)
        plot_categorical_vs_strategy(df, col, TARGET_COLUMN, figures_dir)

    encoded = encode_for_correlation(df, FEATURE_COLUMNS)
    numeric_corr = df[NUMERIC_FEATURES].corr()
    encoded_corr = encoded[FEATURE_COLUMNS].corr()
    plot_encoded_correlation_matrix(encoded, FEATURE_COLUMNS, figures_dir)
    plot_numeric_correlation_matrix(df, NUMERIC_FEATURES, figures_dir)
    plot_outlier_summary(outlier_summary, figures_dir)

    correlation_pairs_table(encoded_corr).to_csv(output_dir / "correlation_pairs_encoded.csv", index=False)
    correlation_pairs_table(numeric_corr).to_csv(output_dir / "correlation_pairs_numeric.csv", index=False)
    (output_dir / "redundancy_analysis.md").write_text(
        build_redundancy_report(numeric_corr, encoded_corr), encoding="utf-8"
    )

    plot_feature_vs_strategy(df, "SleepQuality", TARGET_COLUMN, figures_dir, "09_sleep_quality_vs_strategy.png", numeric=False)
    plot_feature_vs_strategy(df, "StressLevel", TARGET_COLUMN, figures_dir, "10_stress_level_vs_strategy.png", numeric=False)
    plot_feature_vs_strategy(df, "FatigueLevel", TARGET_COLUMN, figures_dir, "11_fatigue_level_vs_strategy.png", numeric=False)
    plot_feature_vs_strategy(df, "DaysUntilExam", TARGET_COLUMN, figures_dir, "12_days_until_exam_vs_strategy.png", numeric=True)
    plot_feature_vs_strategy(df, "SleepHours", TARGET_COLUMN, figures_dir, "13_sleep_hours_vs_strategy.png", numeric=True)

    display_path = schema.raw_path.relative_to(ML_ROOT.parent).as_posix()

    summary = build_eda_summary(
        df=df,
        schema_path=display_path,
        figures_dir=figures_dir,
        numeric_columns=NUMERIC_FEATURES,
        categorical_columns=CATEGORICAL_FEATURES,
        target_column=TARGET_COLUMN,
        missing_df=missing_df,
        duplicate_info=dup_info,
        invalid_df=invalid_df,
        balance_df=balance_df,
        desc_stats=desc_stats,
        outlier_summary=outlier_summary,
        outlier_sample=outlier_sample,
        encoded_corr=encoded_corr,
    )

    report_path = output_dir / "eda_summary.md"
    report_path.write_text(summary, encoding="utf-8")
    return report_path


def main() -> None:
    report_path = run_eda()
    figures_dir = ML_ROOT / "output" / "figures"
    figure_count = len(list(figures_dir.glob("*.png")))

    df = load_dataset()
    print("Smart Study Advisor EDA complete")
    print(f"  Figures: {figure_count} saved to {figures_dir}")
    print(f"  Report:  {report_path}")
    print()
    print("Categorical stress/fatigue check:")
    print(f"  StressLevel values:  {sorted(df['StressLevel'].unique())}")
    print(f"  FatigueLevel values: {sorted(df['FatigueLevel'].unique())}")
    print(f"  Numeric features:    {NUMERIC_FEATURES}")
    print()
    print("Target distribution:")
    print(df[TARGET_COLUMN].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
