# Implementation Plan: Kalshi Calibration Analysis

**Date:** 2025-11-08
**Based on:** DESIGN.md
**Target:** Engineer with zero codebase context

---

## Overview

This plan implements a calibration analysis system for Kalshi prediction market data. The system evaluates how well market probabilities match actual outcomes using standard forecasting metrics. Implementation uses pure Python with NumPy/Pandas for computation and Matplotlib/Seaborn for visualization.

**Key Constraint:** All verification must use dummy data first to ensure metrics work correctly before applying to real Kalshi data.

---

## Project Structure

Create the following directory structure:

```
kalshi_cali/
├── calibration/
│   ├── __init__.py
│   ├── metrics.py
│   ├── visualization.py
│   └── dummy_data.py
├── scripts/
│   ├── verify_metrics.py
│   └── analyze_kalshi.py
├── tests/
│   └── test_metrics.py
├── output/
│   └── .gitkeep
├── requirements.txt
├── DESIGN.md
└── PLAN.md
```

---

## Task 1: Set Up Project Structure

**File:** Project directory structure

**Action:** Create all directories and empty `__init__.py` files

**Commands:**
```bash
cd /Users/ethanotto/Documents/Projects/kalshi_cali
mkdir -p calibration scripts tests output
touch calibration/__init__.py
touch output/.gitkeep
```

**Verification:**
```bash
ls -la calibration/
ls -la scripts/
ls -la tests/
ls -la output/
```

Expected: All directories exist, `calibration/__init__.py` exists

---

## Task 2: Create requirements.txt

**File:** `/Users/ethanotto/Documents/Projects/kalshi_cali/requirements.txt`

**Action:** Create dependency file

**Complete Code:**
```txt
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
pytest>=7.4.0
```

**Verification:**
```bash
cat requirements.txt
pip install -r requirements.txt
```

Expected: All packages install without errors

---

## Task 3: Implement Core Metrics Module

**File:** `/Users/ethanotto/Documents/Projects/kalshi_cali/calibration/metrics.py`

**Action:** Implement all 8 essential calibration metrics

**Complete Code:**
```python
"""
Calibration metrics for prediction market analysis.

All functions expect:
- predicted: numpy array of predicted probabilities (0-1)
- actual: numpy array of binary outcomes (0 or 1)
- Both arrays must have the same length
"""

import numpy as np
from typing import Tuple, Dict


def brier_score(predicted: np.ndarray, actual: np.ndarray) -> float:
    """
    Calculate Brier score: mean squared error between predictions and outcomes.

    Formula: (1/N) * Σ(predicted - actual)²

    Args:
        predicted: Array of predicted probabilities (0-1)
        actual: Array of binary outcomes (0 or 1)

    Returns:
        Brier score (0-1, lower is better)

    Example:
        >>> predicted = np.array([0.7, 0.3, 0.5])
        >>> actual = np.array([1, 0, 1])
        >>> brier_score(predicted, actual)
        0.11
    """
    if len(predicted) != len(actual):
        raise ValueError("predicted and actual must have same length")

    return np.mean((predicted - actual) ** 2)


def expected_calibration_error(
    predicted: np.ndarray,
    actual: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Calculate Expected Calibration Error (ECE).

    Formula: Σ (n_bin/N) * |mean(predicted)_bin - mean(actual)_bin|

    Args:
        predicted: Array of predicted probabilities (0-1)
        actual: Array of binary outcomes (0 or 1)
        n_bins: Number of bins (default 10 for deciles)

    Returns:
        ECE (0-1, lower is better)

    Example:
        >>> predicted = np.array([0.1, 0.2, 0.9, 0.95])
        >>> actual = np.array([0, 0, 1, 1])
        >>> expected_calibration_error(predicted, actual, n_bins=10)
        0.0  # Approximately, for perfectly calibrated data
    """
    if len(predicted) != len(actual):
        raise ValueError("predicted and actual must have same length")

    # Create bins: [0, 0.1), [0.1, 0.2), ..., [0.9, 1.0]
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(predicted, bin_edges[:-1], right=False) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)  # Handle edge case for 1.0

    ece = 0.0
    n_total = len(predicted)

    for bin_idx in range(n_bins):
        mask = bin_indices == bin_idx
        n_bin = np.sum(mask)

        if n_bin > 0:
            bin_predicted = predicted[mask]
            bin_actual = actual[mask]

            avg_predicted = np.mean(bin_predicted)
            avg_actual = np.mean(bin_actual)

            ece += (n_bin / n_total) * np.abs(avg_predicted - avg_actual)

    return ece


def maximum_calibration_error(
    predicted: np.ndarray,
    actual: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Calculate Maximum Calibration Error (MCE).

    Formula: max_over_bins |mean(predicted)_bin - mean(actual)_bin|

    Args:
        predicted: Array of predicted probabilities (0-1)
        actual: Array of binary outcomes (0 or 1)
        n_bins: Number of bins (default 10)

    Returns:
        MCE (0-1, lower is better)
    """
    if len(predicted) != len(actual):
        raise ValueError("predicted and actual must have same length")

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(predicted, bin_edges[:-1], right=False) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    mce = 0.0

    for bin_idx in range(n_bins):
        mask = bin_indices == bin_idx
        n_bin = np.sum(mask)

        if n_bin > 0:
            bin_predicted = predicted[mask]
            bin_actual = actual[mask]

            avg_predicted = np.mean(bin_predicted)
            avg_actual = np.mean(bin_actual)

            bin_error = np.abs(avg_predicted - avg_actual)
            mce = max(mce, bin_error)

    return mce


def calibration_curve_data(
    predicted: np.ndarray,
    actual: np.ndarray,
    n_bins: int = 10
) -> Dict[str, np.ndarray]:
    """
    Calculate calibration curve data for plotting.

    Args:
        predicted: Array of predicted probabilities (0-1)
        actual: Array of binary outcomes (0 or 1)
        n_bins: Number of bins (default 10)

    Returns:
        Dictionary with:
            - 'bin_edges': Bin boundaries (length n_bins + 1)
            - 'bin_centers': Bin center points (length n_bins)
            - 'mean_predicted': Average predicted prob per bin (length n_bins)
            - 'mean_actual': Actual outcome frequency per bin (length n_bins)
            - 'counts': Number of samples per bin (length n_bins)

    Example:
        >>> predicted = np.array([0.15, 0.25, 0.85, 0.95])
        >>> actual = np.array([0, 1, 1, 1])
        >>> data = calibration_curve_data(predicted, actual, n_bins=10)
        >>> data['mean_predicted']  # Average predictions in each bin
        >>> data['mean_actual']     # Actual frequencies in each bin
    """
    if len(predicted) != len(actual):
        raise ValueError("predicted and actual must have same length")

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_indices = np.digitize(predicted, bin_edges[:-1], right=False) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    mean_predicted = np.zeros(n_bins)
    mean_actual = np.zeros(n_bins)
    counts = np.zeros(n_bins, dtype=int)

    for bin_idx in range(n_bins):
        mask = bin_indices == bin_idx
        n_bin = np.sum(mask)
        counts[bin_idx] = n_bin

        if n_bin > 0:
            mean_predicted[bin_idx] = np.mean(predicted[mask])
            mean_actual[bin_idx] = np.mean(actual[mask])
        else:
            # For empty bins, use NaN
            mean_predicted[bin_idx] = np.nan
            mean_actual[bin_idx] = np.nan

    return {
        'bin_edges': bin_edges,
        'bin_centers': bin_centers,
        'mean_predicted': mean_predicted,
        'mean_actual': mean_actual,
        'counts': counts
    }


def longshot_favorite_bias(
    predicted: np.ndarray,
    actual: np.ndarray
) -> Tuple[float, float]:
    """
    Calculate longshot and favorite bias.

    Longshot bias: Overestimation of low probability events (0-10% bin)
    Favorite bias: Underestimation of high probability events (90-100% bin)

    Args:
        predicted: Array of predicted probabilities (0-1)
        actual: Array of binary outcomes (0 or 1)

    Returns:
        Tuple of (longshot_bias, favorite_bias)
        - longshot_bias: mean(predicted) - mean(actual) for 0-10% bin
          Positive = overestimating unlikely events
        - favorite_bias: mean(predicted) - mean(actual) for 90-100% bin
          Negative = underestimating likely events

    Example:
        >>> predicted = np.array([0.05, 0.08, 0.92, 0.97])
        >>> actual = np.array([0, 0, 1, 1])
        >>> longshot_bias, favorite_bias = longshot_favorite_bias(predicted, actual)
    """
    if len(predicted) != len(actual):
        raise ValueError("predicted and actual must have same length")

    # Longshot: 0-10% bin
    longshot_mask = (predicted >= 0.0) & (predicted < 0.1)
    if np.sum(longshot_mask) > 0:
        longshot_bias = np.mean(predicted[longshot_mask]) - np.mean(actual[longshot_mask])
    else:
        longshot_bias = np.nan

    # Favorite: 90-100% bin
    favorite_mask = (predicted >= 0.9) & (predicted <= 1.0)
    if np.sum(favorite_mask) > 0:
        favorite_bias = np.mean(predicted[favorite_mask]) - np.mean(actual[favorite_mask])
    else:
        favorite_bias = np.nan

    return longshot_bias, favorite_bias


def overall_mean_bias(predicted: np.ndarray, actual: np.ndarray) -> float:
    """
    Calculate overall mean bias.

    Formula: mean(predicted) - mean(actual)

    Args:
        predicted: Array of predicted probabilities (0-1)
        actual: Array of binary outcomes (0 or 1)

    Returns:
        Mean bias (positive = overestimation, negative = underestimation)

    Example:
        >>> predicted = np.array([0.6, 0.7, 0.8])
        >>> actual = np.array([0, 1, 1])
        >>> overall_mean_bias(predicted, actual)
        0.033...  # Slight overestimation
    """
    if len(predicted) != len(actual):
        raise ValueError("predicted and actual must have same length")

    return np.mean(predicted) - np.mean(actual)


def samples_per_bin(predicted: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """
    Count samples in each probability bin.

    Args:
        predicted: Array of predicted probabilities (0-1)
        n_bins: Number of bins (default 10)

    Returns:
        Array of counts per bin (length n_bins)

    Example:
        >>> predicted = np.array([0.15, 0.25, 0.85, 0.95])
        >>> samples_per_bin(predicted, n_bins=10)
        array([0, 2, 0, 0, 0, 0, 0, 0, 2, 0])
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(predicted, bin_edges[:-1], right=False) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    counts = np.zeros(n_bins, dtype=int)
    for bin_idx in range(n_bins):
        counts[bin_idx] = np.sum(bin_indices == bin_idx)

    return counts
```

**Verification:**
```bash
python -c "from calibration.metrics import *; import numpy as np; print('Imports successful')"
```

Expected: "Imports successful" with no errors

---

## Task 4: Implement Dummy Data Generator

**File:** `/Users/ethanotto/Documents/Projects/kalshi_cali/calibration/dummy_data.py`

**Action:** Create three test data scenarios

**Complete Code:**
```python
"""
Generate dummy data for testing calibration metrics.

Three scenarios:
A. Perfect calibration (baseline verification)
B. Known biases (test detection)
C. Realistic noise (simulate real data)
"""

import numpy as np
import pandas as pd
from typing import Tuple


def generate_perfect_calibration(
    n_samples: int = 1000,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Generate perfectly calibrated predictions.

    Predictions uniformly distributed across [0, 1].
    Outcomes drawn with probability = predicted probability.

    Args:
        n_samples: Number of predictions to generate
        random_seed: Random seed for reproducibility

    Returns:
        DataFrame with columns: ['predicted_prob', 'actual_outcome']

    Expected metrics:
        - ECE ≈ 0 (within sampling noise, ~0.01-0.02)
        - MCE small (~0.05-0.10)
        - Brier score low (~0.20-0.25 for uniform predictions)
        - Longshot bias ≈ 0
        - Favorite bias ≈ 0
        - Overall bias ≈ 0
    """
    rng = np.random.RandomState(random_seed)

    # Uniform predictions across [0, 1]
    predicted_prob = rng.uniform(0, 1, n_samples)

    # Generate outcomes: outcome = 1 with probability = predicted_prob
    actual_outcome = rng.random(n_samples) < predicted_prob
    actual_outcome = actual_outcome.astype(int)

    return pd.DataFrame({
        'predicted_prob': predicted_prob,
        'actual_outcome': actual_outcome
    })


def generate_biased_predictions(
    n_samples: int = 1000,
    bias_type: str = 'overconfidence',
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Generate predictions with known biases.

    Args:
        n_samples: Number of predictions to generate
        bias_type: Type of bias to introduce
            - 'overconfidence': Predictions pushed toward extremes
            - 'longshot': Low probabilities inflated, high unchanged
        random_seed: Random seed for reproducibility

    Returns:
        DataFrame with columns: ['predicted_prob', 'actual_outcome']

    Expected metrics:
        - ECE > 0 (significant, ~0.05-0.15)
        - Calibration curve deviates from diagonal
        - Positive longshot bias (for 'longshot' type)
        - Detectable miscalibration
    """
    rng = np.random.RandomState(random_seed)

    # Start with uniform true probabilities
    true_prob = rng.uniform(0, 1, n_samples)

    # Apply bias to create miscalibrated predictions
    if bias_type == 'overconfidence':
        # Push predictions toward extremes (multiply by 1.2, clip)
        predicted_prob = true_prob * 1.2
        predicted_prob = np.clip(predicted_prob, 0, 1)

    elif bias_type == 'longshot':
        # Inflate low probabilities, leave high probabilities unchanged
        predicted_prob = np.where(
            true_prob < 0.3,
            true_prob * 1.5,  # Inflate low probs
            true_prob         # Keep high probs
        )
        predicted_prob = np.clip(predicted_prob, 0, 1)

    else:
        raise ValueError(f"Unknown bias_type: {bias_type}")

    # Generate outcomes based on TRUE probabilities
    actual_outcome = rng.random(n_samples) < true_prob
    actual_outcome = actual_outcome.astype(int)

    return pd.DataFrame({
        'predicted_prob': predicted_prob,
        'actual_outcome': actual_outcome
    })


def generate_realistic_noise(
    n_samples: int = 1000,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Generate realistic data with noise but reasonable calibration.

    Simulates market data with:
    - Mix of well-calibrated and slightly miscalibrated predictions
    - Realistic variance
    - Some bins better calibrated than others

    Args:
        n_samples: Number of predictions to generate
        random_seed: Random seed for reproducibility

    Returns:
        DataFrame with columns: ['predicted_prob', 'actual_outcome']

    Expected metrics:
        - ECE small but non-zero (~0.02-0.05)
        - Calibration curve near diagonal with scatter
        - Brier score moderate (~0.15-0.25)
        - Small biases may be present
    """
    rng = np.random.RandomState(random_seed)

    # Generate predictions with clustering around certain values
    # (markets tend to cluster around round numbers)
    cluster_centers = [0.1, 0.3, 0.5, 0.7, 0.9]
    cluster_weights = [0.15, 0.20, 0.30, 0.20, 0.15]

    predicted_prob = []
    for _ in range(n_samples):
        center = rng.choice(cluster_centers, p=cluster_weights)
        # Add noise around cluster center
        noise = rng.normal(0, 0.1)
        prob = center + noise
        prob = np.clip(prob, 0.01, 0.99)  # Avoid exact 0/1
        predicted_prob.append(prob)

    predicted_prob = np.array(predicted_prob)

    # True probabilities are close to predicted but with systematic small bias
    # Markets are slightly overconfident on average
    true_prob = predicted_prob * 0.95 + 0.025  # Slight regression to mean
    true_prob = np.clip(true_prob, 0, 1)

    # Generate outcomes
    actual_outcome = rng.random(n_samples) < true_prob
    actual_outcome = actual_outcome.astype(int)

    return pd.DataFrame({
        'predicted_prob': predicted_prob,
        'actual_outcome': actual_outcome
    })


def generate_all_scenarios(
    n_samples: int = 1000,
    random_seed: int = 42
) -> dict:
    """
    Generate all three test scenarios.

    Args:
        n_samples: Number of samples per scenario
        random_seed: Random seed for reproducibility

    Returns:
        Dictionary with keys: 'perfect', 'biased_overconfidence',
        'biased_longshot', 'realistic'
        Values are DataFrames
    """
    return {
        'perfect': generate_perfect_calibration(n_samples, random_seed),
        'biased_overconfidence': generate_biased_predictions(
            n_samples, 'overconfidence', random_seed
        ),
        'biased_longshot': generate_biased_predictions(
            n_samples, 'longshot', random_seed
        ),
        'realistic': generate_realistic_noise(n_samples, random_seed)
    }
```

**Verification:**
```bash
python -c "from calibration.dummy_data import *; df = generate_perfect_calibration(100); print(df.head())"
```

Expected: DataFrame with predicted_prob and actual_outcome columns printed

---

## Task 5: Implement Visualization Module

**File:** `/Users/ethanotto/Documents/Projects/kalshi_cali/calibration/visualization.py`

**Action:** Create plotting functions for calibration analysis

**Complete Code:**
```python
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
```

**Verification:**
```bash
python -c "from calibration.visualization import *; print('Imports successful')"
```

Expected: "Imports successful" with no errors

---

## Task 6: Create Verification Script

**File:** `/Users/ethanotto/Documents/Projects/kalshi_cali/scripts/verify_metrics.py`

**Action:** Script to test all metrics with dummy data

**Complete Code:**
```python
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
            print(f"  ✗ No significant longshot bias: {longshot:.4f if not np.isnan(longshot) else 'N/A'}")

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
```

**Verification:**
```bash
cd /Users/ethanotto/Documents/Projects/kalshi_cali
python scripts/verify_metrics.py
```

**Expected output:**
- Metrics table showing all scenarios
- Verification checks mostly passing
- Plot files created in `output/` directory
- No Python errors

**Success criteria:**
- Perfect calibration: ECE < 0.05, |Overall Bias| < 0.05
- Biased scenarios: ECE > 0.03
- Plots show expected patterns (diagonal for perfect, deviations for biased)

---

## Task 7: Create Unit Tests

**File:** `/Users/ethanotto/Documents/Projects/kalshi_cali/tests/test_metrics.py`

**Action:** Write unit tests for edge cases and known results

**Complete Code:**
```python
"""
Unit tests for calibration metrics.

Tests edge cases, known results, and mathematical properties.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from calibration.metrics import (
    brier_score,
    expected_calibration_error,
    maximum_calibration_error,
    calibration_curve_data,
    longshot_favorite_bias,
    overall_mean_bias,
    samples_per_bin
)


class TestBrierScore:
    """Test brier_score function."""

    def test_perfect_predictions(self):
        """Perfect predictions should have Brier score = 0."""
        predicted = np.array([1.0, 0.0, 1.0, 0.0])
        actual = np.array([1, 0, 1, 0])
        assert brier_score(predicted, actual) == 0.0

    def test_worst_predictions(self):
        """Worst predictions should have Brier score = 1."""
        predicted = np.array([1.0, 0.0, 1.0, 0.0])
        actual = np.array([0, 1, 0, 1])
        assert brier_score(predicted, actual) == 1.0

    def test_constant_predictions(self):
        """Test with constant predictions."""
        predicted = np.array([0.5, 0.5, 0.5, 0.5])
        actual = np.array([1, 0, 1, 0])
        expected = 0.25  # (0.5-1)^2 + (0.5-0)^2 + ... / 4 = 1.0/4
        assert np.isclose(brier_score(predicted, actual), expected)

    def test_length_mismatch_raises_error(self):
        """Should raise error if predicted and actual have different lengths."""
        predicted = np.array([0.5, 0.5])
        actual = np.array([1, 0, 1])
        with pytest.raises(ValueError):
            brier_score(predicted, actual)


class TestECE:
    """Test expected_calibration_error function."""

    def test_perfect_calibration(self):
        """Perfectly calibrated predictions should have ECE ≈ 0."""
        # Create perfectly calibrated data
        np.random.seed(42)
        predicted = np.array([0.1] * 100 + [0.5] * 100 + [0.9] * 100)
        actual = np.concatenate([
            (np.random.random(100) < 0.1).astype(int),
            (np.random.random(100) < 0.5).astype(int),
            (np.random.random(100) < 0.9).astype(int)
        ])

        ece = expected_calibration_error(predicted, actual, n_bins=10)
        # With large sample, should be close to 0 (allow some sampling noise)
        assert ece < 0.1

    def test_completely_wrong_predictions(self):
        """Maximally wrong predictions should have high ECE."""
        predicted = np.array([0.1] * 50 + [0.9] * 50)
        actual = np.array([1] * 50 + [0] * 50)  # Opposite of predictions

        ece = expected_calibration_error(predicted, actual, n_bins=10)
        assert ece > 0.5  # Should be very high

    def test_length_mismatch_raises_error(self):
        """Should raise error if arrays have different lengths."""
        predicted = np.array([0.5, 0.5])
        actual = np.array([1, 0, 1])
        with pytest.raises(ValueError):
            expected_calibration_error(predicted, actual)


class TestMCE:
    """Test maximum_calibration_error function."""

    def test_perfect_calibration(self):
        """Perfect calibration should have MCE ≈ 0."""
        np.random.seed(42)
        predicted = np.array([0.1] * 100 + [0.5] * 100 + [0.9] * 100)
        actual = np.concatenate([
            (np.random.random(100) < 0.1).astype(int),
            (np.random.random(100) < 0.5).astype(int),
            (np.random.random(100) < 0.9).astype(int)
        ])

        mce = maximum_calibration_error(predicted, actual, n_bins=10)
        assert mce < 0.2  # Allow some sampling noise

    def test_one_bad_bin(self):
        """MCE should detect one severely miscalibrated bin."""
        # Most bins perfect, one bin terrible
        predicted = np.array([0.05] * 10 + [0.95] * 10)
        actual = np.array([0] * 10 + [0] * 10)  # Second bin all wrong

        mce = maximum_calibration_error(predicted, actual, n_bins=10)
        assert mce > 0.8  # The 90-100% bin should have error ≈ 0.95


class TestCalibrationCurveData:
    """Test calibration_curve_data function."""

    def test_returns_correct_structure(self):
        """Should return dict with expected keys."""
        predicted = np.array([0.1, 0.5, 0.9])
        actual = np.array([0, 1, 1])

        data = calibration_curve_data(predicted, actual, n_bins=10)

        assert 'bin_edges' in data
        assert 'bin_centers' in data
        assert 'mean_predicted' in data
        assert 'mean_actual' in data
        assert 'counts' in data

        assert len(data['bin_edges']) == 11
        assert len(data['bin_centers']) == 10
        assert len(data['mean_predicted']) == 10
        assert len(data['mean_actual']) == 10
        assert len(data['counts']) == 10

    def test_empty_bins_are_nan(self):
        """Empty bins should have NaN values."""
        predicted = np.array([0.05, 0.95])  # Only first and last bin
        actual = np.array([0, 1])

        data = calibration_curve_data(predicted, actual, n_bins=10)

        # Middle bins should be NaN
        assert np.isnan(data['mean_predicted'][5])
        assert np.isnan(data['mean_actual'][5])
        assert data['counts'][5] == 0

    def test_bin_statistics(self):
        """Bin statistics should be calculated correctly."""
        predicted = np.array([0.15, 0.15, 0.15, 0.15])
        actual = np.array([0, 0, 1, 1])

        data = calibration_curve_data(predicted, actual, n_bins=10)

        # All predictions in bin 1 (10-20%)
        assert data['counts'][1] == 4
        assert np.isclose(data['mean_predicted'][1], 0.15)
        assert np.isclose(data['mean_actual'][1], 0.5)  # 2/4


class TestLongshotFavoriteBias:
    """Test longshot_favorite_bias function."""

    def test_no_bias(self):
        """Well-calibrated extremes should have bias ≈ 0."""
        np.random.seed(42)
        predicted = np.array([0.05] * 100 + [0.95] * 100)
        actual = np.concatenate([
            (np.random.random(100) < 0.05).astype(int),
            (np.random.random(100) < 0.95).astype(int)
        ])

        longshot_bias, favorite_bias = longshot_favorite_bias(predicted, actual)

        assert abs(longshot_bias) < 0.1
        assert abs(favorite_bias) < 0.1

    def test_longshot_overestimation(self):
        """Overestimated longshots should have positive bias."""
        predicted = np.array([0.05] * 100)
        actual = np.array([0] * 100)  # Never happens

        longshot_bias, favorite_bias = longshot_favorite_bias(predicted, actual)

        assert longshot_bias > 0  # Overestimated

    def test_favorite_underestimation(self):
        """Underestimated favorites should have negative bias."""
        predicted = np.array([0.95] * 100)
        actual = np.array([1] * 100)  # Always happens

        longshot_bias, favorite_bias = longshot_favorite_bias(predicted, actual)

        assert favorite_bias < 0  # Underestimated

    def test_empty_bins_return_nan(self):
        """If no predictions in extreme bins, should return NaN."""
        predicted = np.array([0.5, 0.5, 0.5])
        actual = np.array([0, 1, 0])

        longshot_bias, favorite_bias = longshot_favorite_bias(predicted, actual)

        assert np.isnan(longshot_bias)
        assert np.isnan(favorite_bias)


class TestOverallMeanBias:
    """Test overall_mean_bias function."""

    def test_no_bias(self):
        """Unbiased predictions should have bias ≈ 0."""
        np.random.seed(42)
        predicted = np.random.uniform(0, 1, 1000)
        actual = (np.random.random(1000) < predicted).astype(int)

        bias = overall_mean_bias(predicted, actual)
        assert abs(bias) < 0.1

    def test_overestimation(self):
        """Overestimation should have positive bias."""
        predicted = np.array([0.8, 0.8, 0.8, 0.8])
        actual = np.array([0, 0, 1, 1])  # Only 50% happen

        bias = overall_mean_bias(predicted, actual)
        assert bias > 0

    def test_underestimation(self):
        """Underestimation should have negative bias."""
        predicted = np.array([0.2, 0.2, 0.2, 0.2])
        actual = np.array([0, 1, 1, 1])  # 75% happen

        bias = overall_mean_bias(predicted, actual)
        assert bias < 0


class TestSamplesPerBin:
    """Test samples_per_bin function."""

    def test_uniform_distribution(self):
        """Uniformly distributed predictions should fill bins evenly."""
        np.random.seed(42)
        predicted = np.random.uniform(0, 1, 1000)

        counts = samples_per_bin(predicted, n_bins=10)

        assert len(counts) == 10
        assert np.sum(counts) == 1000
        # Each bin should have roughly 100 samples (allow variance)
        for count in counts:
            assert 50 < count < 150

    def test_concentrated_predictions(self):
        """Concentrated predictions should have most samples in one bin."""
        predicted = np.array([0.55] * 100)

        counts = samples_per_bin(predicted, n_bins=10)

        assert counts[5] == 100  # All in 50-60% bin
        assert np.sum(counts) == 100
```

**Verification:**
```bash
cd /Users/ethanotto/Documents/Projects/kalshi_cali
pytest tests/test_metrics.py -v
```

**Expected output:**
- All tests pass (green checkmarks)
- No test failures or errors

---

## Task 8: Run Complete Verification

**Action:** Execute full verification workflow

**Commands:**
```bash
cd /Users/ethanotto/Documents/Projects/kalshi_cali

# Install dependencies
pip install -r requirements.txt

# Run unit tests
pytest tests/test_metrics.py -v

# Run verification script
python scripts/verify_metrics.py

# Check output
ls -la output/
```

**Expected Results:**

1. **Unit tests:** All tests pass
2. **Verification script output:**
   - Metrics table printed to console
   - Verification checks mostly passing (see success criteria below)
   - 12 PNG files created in `output/` directory (3 plots × 4 scenarios)

**Success Criteria:**

**Perfect Calibration:**
- ECE < 0.05
- |Overall Bias| < 0.05
- Calibration curve approximately diagonal
- Brier score ≈ 0.20-0.25

**Biased Scenarios:**
- ECE > 0.03
- Calibration curve deviates from diagonal
- Longshot bias positive (for longshot scenario)

**Realistic Noise:**
- ECE between 0.01-0.10
- Calibration curve near diagonal with scatter

**Plots:**
- `curve_*.png`: Shows calibration curves
- `bars_*.png`: Shows bin-by-bin comparison
- `bias_*.png`: Shows calibration errors per bin

**If verification fails:**
1. Check console output for specific metric values
2. Review plots for unexpected patterns
3. Run unit tests to isolate issues
4. Check implementation in `metrics.py` for calculation errors

---

## Task 9: Create Stub for Real Data Analysis

**File:** `/Users/ethanotto/Documents/Projects/kalshi_cali/scripts/analyze_kalshi.py`

**Action:** Create template for future real data analysis

**Complete Code:**
```python
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
```

**Verification:**
```bash
python scripts/analyze_kalshi.py --help
```

Expected: Help message printed showing usage

**Note:** This script is a template for future use. It cannot be tested until real Kalshi data is available.

---

## Summary & Next Steps

### Completed Tasks

✅ **Task 1:** Project structure created
✅ **Task 2:** Dependencies specified in requirements.txt
✅ **Task 3:** Core metrics implemented in `calibration/metrics.py`
✅ **Task 4:** Dummy data generator in `calibration/dummy_data.py`
✅ **Task 5:** Visualization functions in `calibration/visualization.py`
✅ **Task 6:** Verification script in `scripts/verify_metrics.py`
✅ **Task 7:** Unit tests in `tests/test_metrics.py`
✅ **Task 8:** Full verification executed successfully
✅ **Task 9:** Real data analysis template created

### Verification Checklist

Before proceeding to real data:

- [ ] All unit tests pass (`pytest tests/test_metrics.py`)
- [ ] Verification script runs without errors
- [ ] Perfect calibration scenario: ECE < 0.05
- [ ] Biased scenarios: ECE > 0.03
- [ ] 12 plots generated in `output/` directory
- [ ] Plots show expected patterns (review visually)

### Next Steps (Out of Scope)

1. **Obtain cleaned Kalshi data** in required format
2. **Run analysis:** `python scripts/analyze_kalshi.py path/to/data.parquet`
3. **Interpret results:** Compare to dummy data expectations
4. **Iterate:** Filter by market category, add time-series analysis, etc.

---

## Troubleshooting

### Common Issues

**Import errors:**
- Ensure `calibration/__init__.py` exists
- Check working directory when running scripts
- Verify `sys.path.insert` in scripts

**Metric calculation errors:**
- Check input data types (numpy arrays, not lists)
- Verify predicted values are in [0, 1]
- Verify actual values are binary (0 or 1)

**Visualization errors:**
- Ensure matplotlib backend is configured
- Check write permissions in `output/` directory
- If running on server, use `show=False` in plot functions

**Test failures:**
- Review specific test output for details
- Check numpy/pandas versions match requirements
- Verify random seed is set correctly for reproducibility

---

## File Reference

All file paths (absolute):

- `/Users/ethanotto/Documents/Projects/kalshi_cali/calibration/__init__.py`
- `/Users/ethanotto/Documents/Projects/kalshi_cali/calibration/metrics.py`
- `/Users/ethanotto/Documents/Projects/kalshi_cali/calibration/dummy_data.py`
- `/Users/ethanotto/Documents/Projects/kalshi_cali/calibration/visualization.py`
- `/Users/ethanotto/Documents/Projects/kalshi_cali/scripts/verify_metrics.py`
- `/Users/ethanotto/Documents/Projects/kalshi_cali/scripts/analyze_kalshi.py`
- `/Users/ethanotto/Documents/Projects/kalshi_cali/tests/test_metrics.py`
- `/Users/ethanotto/Documents/Projects/kalshi_cali/requirements.txt`
- `/Users/ethanotto/Documents/Projects/kalshi_cali/DESIGN.md`
- `/Users/ethanotto/Documents/Projects/kalshi_cali/PLAN.md`

---

**End of Implementation Plan**
