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
