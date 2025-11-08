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
