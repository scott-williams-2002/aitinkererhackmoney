"""
Analyze real Kalshi prediction market data.

This script applies calibration metrics to actual Kalshi data.

Usage:
    python scripts/analyze_kalshi.py path/to/kalshi_data.parquet

Expected data format:
    DataFrame with columns: market_id, outcome_label, predicted_prob, actual_outcome
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from calibration.metrics import (
    brier_score,
    expected_calibration_error,
    maximum_calibration_error,
    longshot_favorite_bias,
    overall_mean_bias,
    samples_per_bin
)
from calibration.visualization import (
    plot_calibration_curve,
    plot_calibration_bars,
    plot_bias_analysis
)


def load_kalshi_data(file_path: str) -> pd.DataFrame:
    """
    Load Kalshi data from file.

    Args:
        file_path: Path to data file (parquet or pickle)

    Returns:
        DataFrame with required columns
    """
    if file_path.endswith('.parquet'):
        df = pd.read_parquet(file_path)
    elif file_path.endswith('.pkl') or file_path.endswith('.pickle'):
        df = pd.read_pickle(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

    # Validate required columns
    required_cols = ['market_id', 'outcome_label', 'predicted_prob', 'actual_outcome']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    return df


def calculate_metrics(df: pd.DataFrame) -> dict:
    """
    Calculate all calibration metrics.

    Args:
        df: DataFrame with predicted_prob and actual_outcome columns

    Returns:
        Dictionary of metrics
    """
    predicted = df['predicted_prob'].values
    actual = df['actual_outcome'].values

    longshot_bias, favorite_bias = longshot_favorite_bias(predicted, actual)

    return {
        'N Predictions': len(predicted),
        'Brier Score': brier_score(predicted, actual),
        'ECE': expected_calibration_error(predicted, actual),
        'MCE': maximum_calibration_error(predicted, actual),
        'Overall Bias': overall_mean_bias(predicted, actual),
        'Longshot Bias (0-10%)': longshot_bias,
        'Favorite Bias (90-100%)': favorite_bias,
        'Mean Predicted': np.mean(predicted),
        'Mean Actual': np.mean(actual)
    }


def print_metrics(metrics: dict) -> None:
    """Print metrics in formatted table."""
    print("\n" + "="*60)
    print("KALSHI CALIBRATION ANALYSIS RESULTS")
    print("="*60)

    for metric_name, value in metrics.items():
        if isinstance(value, (int, np.integer)):
            print(f"{metric_name:<30}: {value}")
        elif isinstance(value, (float, np.floating)):
            if np.isnan(value):
                print(f"{metric_name:<30}: N/A")
            else:
                print(f"{metric_name:<30}: {value:.4f}")

    print("="*60 + "\n")


def generate_plots(df: pd.DataFrame, output_dir: str = 'output') -> None:
    """Generate calibration plots."""
    os.makedirs(output_dir, exist_ok=True)

    predicted = df['predicted_prob'].values
    actual = df['actual_outcome'].values

    print("\nGenerating plots...")

    plot_calibration_curve(
        predicted, actual,
        title="Kalshi Market Calibration Curve",
        save_path=f"{output_dir}/kalshi_calibration_curve.png",
        show=False
    )

    plot_calibration_bars(
        predicted, actual,
        title="Kalshi Market Calibration by Bin",
        save_path=f"{output_dir}/kalshi_calibration_bars.png",
        show=False
    )

    plot_bias_analysis(
        predicted, actual,
        title="Kalshi Market Calibration Error",
        save_path=f"{output_dir}/kalshi_bias_analysis.png",
        show=False
    )

    print(f"Plots saved to {output_dir}/")


def main():
    """Main analysis workflow."""
    parser = argparse.ArgumentParser(
        description="Analyze Kalshi prediction market calibration"
    )
    parser.add_argument(
        'data_file',
        help='Path to Kalshi data file (parquet or pickle)'
    )
    parser.add_argument(
        '--output-dir',
        default='output',
        help='Directory for output plots (default: output)'
    )

    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.data_file}...")
    df = load_kalshi_data(args.data_file)
    print(f"Loaded {len(df)} predictions")

    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = calculate_metrics(df)

    # Print results
    print_metrics(metrics)

    # Generate plots
    generate_plots(df, args.output_dir)

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
