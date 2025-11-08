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
