"""
Visualization functions for calibration analysis.

Uses matplotlib and seaborn for static plots.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional
from .metrics import calibration_curve_data


# Set seaborn style
sns.set_style("whitegrid")
sns.set_palette("colorblind")


def plot_calibration_curve(
    predicted: np.ndarray,
    actual: np.ndarray,
    n_bins: int = 10,
    title: str = "Calibration Curve",
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot calibration curve: predicted vs actual probabilities.

    Perfect calibration appears as diagonal line y=x.

    Args:
        predicted: Array of predicted probabilities
        actual: Array of binary outcomes
        n_bins: Number of bins for calibration
        title: Plot title
        save_path: If provided, save figure to this path
        show: If True, display the plot

    Returns:
        Matplotlib figure object
    """
    # Get calibration data
    cal_data = calibration_curve_data(predicted, actual, n_bins)

    # Filter out empty bins
    valid_mask = ~np.isnan(cal_data['mean_predicted'])
    mean_pred = cal_data['mean_predicted'][valid_mask]
    mean_actual = cal_data['mean_actual'][valid_mask]
    counts = cal_data['counts'][valid_mask]

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Perfect calibration')

    # Plot actual calibration curve
    ax.plot(mean_pred, mean_actual, 'o-', linewidth=2, markersize=8,
            label='Observed calibration')

    # Add bin sizes as point sizes (optional visual aid)
    # Normalize counts for point size
    if len(counts) > 0:
        size_scale = 200 * counts / np.max(counts)
        ax.scatter(mean_pred, mean_actual, s=size_scale, alpha=0.3, color='blue')

    # Labels and formatting
    ax.set_xlabel('Predicted Probability', fontsize=12)
    ax.set_ylabel('Actual Frequency', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved calibration curve to {save_path}")

    if show:
        plt.show()

    return fig


def plot_calibration_bars(
    predicted: np.ndarray,
    actual: np.ndarray,
    n_bins: int = 10,
    title: str = "Calibration by Bin",
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot bar chart comparing predicted vs actual probabilities per bin.

    Args:
        predicted: Array of predicted probabilities
        actual: Array of binary outcomes
        n_bins: Number of bins
        title: Plot title
        save_path: If provided, save figure to this path
        show: If True, display the plot

    Returns:
        Matplotlib figure object
    """
    # Get calibration data
    cal_data = calibration_curve_data(predicted, actual, n_bins)

    # Prepare data for plotting
    bin_labels = [f"{int(cal_data['bin_edges'][i]*100)}-{int(cal_data['bin_edges'][i+1]*100)}%"
                  for i in range(n_bins)]

    mean_pred = cal_data['mean_predicted']
    mean_actual = cal_data['mean_actual']
    counts = cal_data['counts']

    # Replace NaN with 0 for plotting (empty bins)
    mean_pred = np.nan_to_num(mean_pred, 0)
    mean_actual = np.nan_to_num(mean_actual, 0)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(n_bins)
    width = 0.35

    # Plot bars
    ax.bar(x - width/2, mean_pred, width, label='Predicted', alpha=0.8)
    ax.bar(x + width/2, mean_actual, width, label='Actual', alpha=0.8)

    # Add sample counts as text
    for i, count in enumerate(counts):
        if count > 0:
            ax.text(i, max(mean_pred[i], mean_actual[i]) + 0.05,
                   f'n={count}', ha='center', fontsize=8, color='gray')

    # Labels and formatting
    ax.set_xlabel('Probability Bin', fontsize=12)
    ax.set_ylabel('Probability', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.set_ylim([0, 1.0])
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved calibration bars to {save_path}")

    if show:
        plt.show()

    return fig


def plot_bias_analysis(
    predicted: np.ndarray,
    actual: np.ndarray,
    n_bins: int = 10,
    title: str = "Calibration Error by Bin",
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot calibration error by bin, highlighting longshot/favorite bins.

    Args:
        predicted: Array of predicted probabilities
        actual: Array of binary outcomes
        n_bins: Number of bins
        title: Plot title
        save_path: If provided, save figure to this path
        show: If True, display the plot

    Returns:
        Matplotlib figure object
    """
    # Get calibration data
    cal_data = calibration_curve_data(predicted, actual, n_bins)

    # Calculate errors per bin
    errors = cal_data['mean_predicted'] - cal_data['mean_actual']
    counts = cal_data['counts']

    # Bin labels
    bin_labels = [f"{int(cal_data['bin_edges'][i]*100)}-{int(cal_data['bin_edges'][i+1]*100)}%"
                  for i in range(n_bins)]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(n_bins)

    # Color bars: red for overestimation, blue for underestimation
    # Highlight extreme bins (0 and 9 for longshot/favorite)
    colors = ['red' if e > 0 else 'blue' for e in errors]
    colors[0] = 'darkred' if not np.isnan(errors[0]) else 'gray'  # Longshot
    colors[-1] = 'darkblue' if not np.isnan(errors[-1]) else 'gray'  # Favorite

    # Plot bars
    bars = ax.bar(x, errors, color=colors, alpha=0.7)

    # Add sample counts
    for i, count in enumerate(counts):
        if count > 0 and not np.isnan(errors[i]):
            ax.text(i, errors[i] + 0.02 * np.sign(errors[i]) if errors[i] != 0 else 0.02,
                   f'n={count}', ha='center', fontsize=8, color='gray')

    # Zero line
    ax.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)

    # Labels and formatting
    ax.set_xlabel('Probability Bin', fontsize=12)
    ax.set_ylabel('Calibration Error (Predicted - Actual)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', alpha=0.7, label='Overestimation'),
        Patch(facecolor='blue', alpha=0.7, label='Underestimation'),
        Patch(facecolor='darkred', alpha=0.7, label='Longshot bias (0-10%)'),
        Patch(facecolor='darkblue', alpha=0.7, label='Favorite bias (90-100%)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved bias analysis to {save_path}")

    if show:
        plt.show()

    return fig


def plot_all_scenarios(
    scenarios: dict,
    output_dir: str = 'output'
) -> None:
    """
    Generate all plots for multiple scenarios.

    Args:
        scenarios: Dict with scenario names as keys, DataFrames as values
                  Each DataFrame must have 'predicted_prob' and 'actual_outcome' columns
        output_dir: Directory to save plots
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    for scenario_name, df in scenarios.items():
        predicted = df['predicted_prob'].values
        actual = df['actual_outcome'].values

        print(f"\nGenerating plots for scenario: {scenario_name}")

        # Calibration curve
        plot_calibration_curve(
            predicted, actual,
            title=f"Calibration Curve - {scenario_name.replace('_', ' ').title()}",
            save_path=f"{output_dir}/curve_{scenario_name}.png",
            show=False
        )

        # Calibration bars
        plot_calibration_bars(
            predicted, actual,
            title=f"Calibration by Bin - {scenario_name.replace('_', ' ').title()}",
            save_path=f"{output_dir}/bars_{scenario_name}.png",
            show=False
        )

        # Bias analysis
        plot_bias_analysis(
            predicted, actual,
            title=f"Calibration Error - {scenario_name.replace('_', ' ').title()}",
            save_path=f"{output_dir}/bias_{scenario_name}.png",
            show=False
        )

    print(f"\nAll plots saved to {output_dir}/")
