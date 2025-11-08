"""
Verify calibration metrics using dummy data.

Generates all test scenarios and calculates metrics to ensure
implementation is correct before applying to real data.
"""

import sys
import os
import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from calibration.dummy_data import generate_all_scenarios
from calibration.metrics import (
    brier_score,
    expected_calibration_error,
    maximum_calibration_error,
    longshot_favorite_bias,
    overall_mean_bias,
    samples_per_bin
)
from calibration.visualization import plot_all_scenarios


def calculate_all_metrics(predicted: np.ndarray, actual: np.ndarray) -> dict:
    """
    Calculate all metrics for a given dataset.

    Args:
        predicted: Array of predicted probabilities
        actual: Array of actual outcomes

    Returns:
        Dictionary of metric names and values
    """
    longshot_bias, favorite_bias = longshot_favorite_bias(predicted, actual)

    return {
        'Brier Score': brier_score(predicted, actual),
        'ECE': expected_calibration_error(predicted, actual),
        'MCE': maximum_calibration_error(predicted, actual),
        'Overall Bias': overall_mean_bias(predicted, actual),
        'Longshot Bias': longshot_bias,
        'Favorite Bias': favorite_bias,
        'N Samples': len(predicted)
    }


def print_metrics_table(results: dict) -> None:
    """
    Print metrics in a formatted table.

    Args:
        results: Dict with scenario names as keys, metric dicts as values
    """
    print("\n" + "="*80)
    print("CALIBRATION METRICS VERIFICATION")
    print("="*80)

    # Get all metric names from first scenario
    first_scenario = list(results.keys())[0]
    metric_names = list(results[first_scenario].keys())

    # Print header
    header = f"{'Metric':<20}"
    for scenario in results.keys():
        header += f"{scenario:<20}"
    print(header)
    print("-"*80)

    # Print each metric
    for metric in metric_names:
        row = f"{metric:<20}"
        for scenario in results.keys():
            value = results[scenario][metric]
            if isinstance(value, (int, np.integer)):
                row += f"{value:<20}"
            elif isinstance(value, (float, np.floating)):
                if np.isnan(value):
                    row += f"{'N/A':<20}"
                else:
                    row += f"{value:<20.4f}"
            else:
                row += f"{str(value):<20}"
        print(row)

    print("="*80)


def verify_expectations(results: dict) -> None:
    """
    Verify that metrics meet expected criteria for each scenario.

    Args:
        results: Dict with scenario names as keys, metric dicts as values
    """
    print("\n" + "="*80)
    print("VERIFICATION CHECKS")
    print("="*80)

    checks_passed = 0
    checks_total = 0

    # Perfect calibration checks
    if 'perfect' in results:
        print("\nPerfect Calibration Scenario:")

        # ECE should be very small (< 0.05)
        checks_total += 1
        ece = results['perfect']['ECE']
        if ece < 0.05:
            print(f"  ✓ ECE < 0.05: {ece:.4f}")
            checks_passed += 1
        else:
            print(f"  ✗ ECE >= 0.05: {ece:.4f} (expected < 0.05)")

        # Overall bias should be near zero (< 0.05)
        checks_total += 1
        bias = abs(results['perfect']['Overall Bias'])
        if bias < 0.05:
            print(f"  ✓ |Overall Bias| < 0.05: {bias:.4f}")
            checks_passed += 1
        else:
            print(f"  ✗ |Overall Bias| >= 0.05: {bias:.4f} (expected < 0.05)")

    # Biased scenarios checks
    for scenario_name in ['biased_overconfidence', 'biased_longshot']:
        if scenario_name in results:
            print(f"\n{scenario_name.replace('_', ' ').title()} Scenario:")

            # ECE should be significant (> 0.03)
            checks_total += 1
            ece = results[scenario_name]['ECE']
            if ece > 0.03:
                print(f"  ✓ ECE > 0.03: {ece:.4f} (bias detected)")
                checks_passed += 1
            else:
                print(f"  ✗ ECE <= 0.03: {ece:.4f} (expected > 0.03)")

    # Longshot bias check
    if 'biased_longshot' in results:
        checks_total += 1
        longshot = results['biased_longshot']['Longshot Bias']
        if not np.isnan(longshot) and longshot > 0.05:
            print(f"  ✓ Positive longshot bias detected: {longshot:.4f}")
            checks_passed += 1
        else:
            longshot_str = f"{longshot:.4f}" if not np.isnan(longshot) else "N/A"
            print(f"  ✗ No significant longshot bias: {longshot_str}")

    # Realistic scenario check
    if 'realistic' in results:
        print("\nRealistic Noise Scenario:")

        # ECE should be moderate (0.01 - 0.10)
        checks_total += 1
        ece = results['realistic']['ECE']
        if 0.01 < ece < 0.10:
            print(f"  ✓ ECE in reasonable range: {ece:.4f}")
            checks_passed += 1
        else:
            print(f"  ✗ ECE outside expected range: {ece:.4f} (expected 0.01-0.10)")

    print("\n" + "="*80)
    print(f"VERIFICATION SUMMARY: {checks_passed}/{checks_total} checks passed")
    print("="*80 + "\n")

    if checks_passed == checks_total:
        print("✓ All verification checks passed! Metrics are working correctly.")
    else:
        print("✗ Some checks failed. Review implementation.")


def main():
    """Main verification workflow."""
    print("Generating dummy data scenarios...")
    scenarios = generate_all_scenarios(n_samples=1000, random_seed=42)

    print(f"Generated {len(scenarios)} scenarios:")
    for name in scenarios.keys():
        print(f"  - {name}")

    print("\nCalculating metrics for each scenario...")
    results = {}
    for scenario_name, df in scenarios.items():
        predicted = df['predicted_prob'].values
        actual = df['actual_outcome'].values
        results[scenario_name] = calculate_all_metrics(predicted, actual)

    # Print results
    print_metrics_table(results)

    # Verify expectations
    verify_expectations(results)

    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_all_scenarios(scenarios, output_dir='output')

    print("\n" + "="*80)
    print("VERIFICATION COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("  1. Review metrics table above")
    print("  2. Check verification results")
    print("  3. Examine plots in output/ directory")
    print("  4. If all looks good, proceed to real data analysis")
    print("\n")


if __name__ == "__main__":
    main()
