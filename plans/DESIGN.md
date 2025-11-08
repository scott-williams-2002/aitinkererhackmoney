# Kalshi Calibration Analysis - Design Document

**Date:** 2025-11-08
**Status:** Design Complete - Ready for Implementation

## Problem Statement

Analyze Kalshi prediction market data to evaluate market calibration by treating the aggregate market as a forecaster. We want to measure how well predicted probabilities match actual outcome frequencies and identify systematic biases (longshot bias, favorite bias).

### Core Goal
Measure calibration error by binning market positions as independent predictions and evaluating them using standard forecasting metrics.

---

## Data Input

### Source
- Cleaned Kalshi prediction market data
- Format: Pandas DataFrame (pickle/parquet)

### Schema
Each row represents one prediction to evaluate:

```python
columns = [
    'market_id',           # Unique market identifier
    'outcome_label',       # Which outcome (e.g., 'YES', 'Democrat', 'Republican')
    'predicted_prob',      # Forecasted probability (0-1)
    'actual_outcome',      # Binary: 1 if outcome occurred, 0 otherwise
    # Optional metadata:
    'market_category',     # e.g., 'Politics', 'Economics' (for future filtering)
    'resolution_date',     # When market resolved
]
```

### Sampling Strategy
- **One prediction per outcome per market**
- Initially: sample at a single point in time per market
- Future enhancement: multiple sample points (announcements, volume spikes, etc.)

---

## Design Decisions

### 1. Multi-Outcome Markets
**Decision:** Treat each outcome as a separate binary prediction

**Rationale:**
- A 3-outcome market (e.g., Democrat/Republican/Other at 45%/50%/5%) becomes 3 independent binary predictions
- Each outcome can be evaluated: "Will Democrat win?" at 45%, "Will Republican win?" at 50%, etc.
- Allows standard binary calibration metrics to apply

**Alternative Considered:** Analyze multi-outcome markets differently
- Rejected: Adds complexity without clear benefit for initial analysis

### 2. Probability Binning
**Decision:** 10% bins [0-10%), [10-20%), ..., [90-100%]

**Rationale:**
- Industry standard for calibration analysis
- Balances resolution (enough bins to see patterns) with statistical power (enough samples per bin)
- Facilitates longshot/favorite bias analysis with clear extreme bins

**Alternatives Considered:**
- 5% bins: More resolution but requires more data
- Adaptive bins: Complex, premature optimization
- Chose 10% for simplicity and standard practice

### 3. Code Structure
**Decision:** Pure Python modules + scripts (no notebooks)

**Rationale:**
- More reproducible and version-control friendly
- Easier to test systematically
- User preference

**Structure:**
```
kalshi_cali/
├── calibration/
│   ├── __init__.py
│   ├── metrics.py              # Core metric calculations
│   ├── visualization.py        # Plotting functions
│   └── dummy_data.py           # Test data generation
├── scripts/
│   ├── verify_metrics.py       # Test with dummy data
│   └── analyze_kalshi.py       # Apply to real data (future)
├── tests/
│   └── test_metrics.py         # Unit tests
└── requirements.txt
```

### 4. Data Schema Design
**Decision:** Single DataFrame with potential metadata redundancy

**Rationale:**
- Simpler to work with for initial analysis
- Easy filtering by metadata (e.g., calibration by market category)
- Can normalize later if redundancy becomes problematic

**Alternative Considered:** Separate prediction and metadata DataFrames
- Rejected: Added complexity without immediate benefit

### 5. Bias Measurement
**Decision:** Compare extreme bins (0-10% vs 90-100%)

**Rationale:**
- Directly measures longshot bias (low prob overestimation) and favorite bias (high prob underestimation)
- Simple to interpret
- Standard approach in prediction market literature

**Alternatives Considered:**
- Regression-based bias detection (will implement as calibration slope/intercept)
- Range-specific analysis (may add later)

### 6. Visualization
**Decision:** Matplotlib + Seaborn for static plots

**Rationale:**
- Standard scientific visualization tools
- Publication-ready output
- User preference for simplicity over interactivity

**Plots to Generate:**
1. Calibration curve (line plot): predicted vs actual by bin
2. Calibration bars (bar plot): side-by-side comparison per bin
3. Bias analysis: highlighting extreme bins

---

## Metrics to Implement

### Essential Metrics (Phase 1)

1. **Brier Score**
   - Formula: `(1/N) * Σ(predicted - actual)²`
   - Range: [0, 1], lower is better
   - Purpose: Overall forecast accuracy combining calibration and sharpness

2. **Expected Calibration Error (ECE)**
   - Formula: `Σ (n_bin/N) * |mean(predicted)_bin - mean(actual)_bin|`
   - Range: [0, 1], lower is better
   - Purpose: Pure calibration quality measurement

3. **Maximum Calibration Error (MCE)**
   - Formula: `max_over_bins |mean(predicted) - mean(actual)|`
   - Range: [0, 1], lower is better
   - Purpose: Identify worst-case bin miscalibration

4. **Longshot Bias**
   - Formula: `mean(predicted) - mean(actual)` for 0-10% bin
   - Positive value = overestimating unlikely events
   - Purpose: Detect systematic overconfidence in low-probability predictions

5. **Favorite Bias**
   - Formula: `mean(predicted) - mean(actual)` for 90-100% bin
   - Negative value = underestimating likely events
   - Purpose: Detect systematic underconfidence in high-probability predictions

6. **Overall Mean Bias**
   - Formula: `mean(predicted) - mean(actual)`
   - Positive = systematic overestimation, negative = underestimation
   - Purpose: Detect directional bias across all predictions

7. **Calibration Curve Data**
   - Per-bin statistics: mean predicted, mean actual, sample count
   - Purpose: Visualization and detailed calibration analysis

8. **Samples Per Bin**
   - Count of predictions in each bin
   - Purpose: Diagnostic - sparse bins indicate unreliable ECE

### Future Metrics (Phase 2+)
- Brier decomposition (uncertainty, resolution, reliability)
- Sharpness (prediction variance)
- Log score (logarithmic loss)
- Calibration regression (slope, intercept)
- Confidence intervals on calibration estimates

---

## Dummy Data Strategy

To verify metric calculations work correctly, we'll generate three test scenarios:

### Scenario A: Perfect Calibration
- **Generation:** N predictions uniformly distributed across [0, 1]; outcomes drawn with prob = predicted_prob
- **Expected Results:**
  - ECE ≈ 0 (within sampling noise)
  - MCE small
  - Brier score low
  - Calibration curve = perfect diagonal
  - Longshot/favorite bias ≈ 0
- **Purpose:** Verify metrics return expected values for known-good data

### Scenario B: Known Biases
- **Generation:**
  - Overconfidence: predictions shifted toward extremes (multiply by 1.2, clip to [0,1])
  - Longshot bias: inflate low probabilities, leave high probabilities unchanged
- **Expected Results:**
  - ECE > 0
  - Calibration curve deviates from diagonal systematically
  - Positive longshot bias
  - Possible favorite bias
- **Purpose:** Test that metrics correctly detect miscalibration

### Scenario C: Realistic Noise
- **Generation:**
  - Well-calibrated with variance: outcomes match predictions on average but with realistic noise
  - Mix of "market types" with different calibration quality
- **Expected Results:**
  - ECE small but non-zero
  - Calibration curve near diagonal with scatter
  - Some bins better calibrated than others
- **Purpose:** Simulate real-world data characteristics

---

## Implementation Plan

### Phase 1: Core Infrastructure + Verification
1. ✅ Create project structure (directories, `__init__.py` files)
2. ✅ Implement `calibration/metrics.py`:
   - `brier_score(predicted, actual)`
   - `expected_calibration_error(predicted, actual, n_bins=10)`
   - `maximum_calibration_error(predicted, actual, n_bins=10)`
   - `calibration_curve_data(predicted, actual, n_bins=10)`
   - `longshot_favorite_bias(predicted, actual)`
   - `overall_mean_bias(predicted, actual)`
   - `samples_per_bin(predicted, n_bins=10)`
3. ✅ Implement `calibration/dummy_data.py`:
   - `generate_perfect_calibration(n_samples)`
   - `generate_biased_predictions(n_samples, bias_type='overconfidence')`
   - `generate_realistic_noise(n_samples)`
4. ✅ Create `scripts/verify_metrics.py`:
   - Load all 3 dummy scenarios
   - Calculate all metrics for each
   - Print results to console
   - Save summary table
5. ✅ Run verification and validate results

### Phase 2: Visualization
6. ✅ Implement `calibration/visualization.py`:
   - `plot_calibration_curve(predicted, actual, title, save_path)`
   - `plot_calibration_bars(predicted, actual, title, save_path)`
   - `plot_bias_analysis(predicted, actual, title, save_path)`
7. ✅ Update `verify_metrics.py` to generate plots for each scenario
8. ✅ Review plots for correctness

### Phase 3: Testing
9. ✅ Implement `tests/test_metrics.py`:
   - Test edge cases (all same predictions, empty bins, etc.)
   - Test known results (perfect calibration → ECE ≈ 0)
   - Test mathematical properties (Brier score bounds, etc.)
10. ✅ Run pytest and fix any issues

### Phase 4: Real Data Analysis (Future)
11. Create `scripts/analyze_kalshi.py`:
    - Load cleaned Kalshi DataFrame
    - Apply all metrics
    - Generate visualizations
    - Save results (metrics table, plots)
12. Iterate based on initial findings

---

## Success Criteria

After Phase 1-3 completion, the system will have:

✅ **Verified Metrics:** Calculations produce expected results on dummy data
- Perfect calibration → ECE ≈ 0, diagonal calibration curve
- Known biases → metrics detect them correctly

✅ **Clear Visualizations:** Plots clearly show calibration quality
- Calibration curves show relationship between predicted and actual
- Bar plots highlight per-bin discrepancies
- Bias plots focus on extreme bins

✅ **Reusable Code:** Module structure allows easy application to real data
- Import `calibration.metrics` in any script
- Consistent API across functions

✅ **Confidence:** Trust that metrics correctly identify miscalibration
- Unit tests pass
- Dummy data scenarios behave as expected
- Ready to apply to real Kalshi data

---

## Open Questions & Future Enhancements

### Immediate Scope (Out of scope for Phase 1-3)
- Applying to real Kalshi data
- Filtering by market category or other metadata
- Time-series analysis (how calibration changes over time)
- Multiple sampling points per market

### Future Enhancements
- Bootstrap confidence intervals on calibration estimates
- Brier score decomposition (resolution, reliability)
- Alternative scoring rules (log score)
- Calibration regression analysis
- Interactive visualizations (plotly)
- Market-level vs aggregate-level calibration
- Subsample analysis (e.g., calibration by market category)

---

## Notes

- This is a forecasting evaluation problem, not a market efficiency analysis
- We're treating the market's probability as "the forecast" and evaluating it
- Sample size per bin matters: sparse bins = unreliable calibration estimates
- Perfect calibration doesn't mean profitable trading (markets can be calibrated but not exploitable)

---

## References & Resources

- **Calibration metrics:** Standard forecasting evaluation literature
- **Prediction markets:** Research on market accuracy and biases
- **Brier score decomposition:** Murphy (1973)
- **ECE/MCE:** Modern ML calibration literature (Guo et al., 2017)

---

## Revision History

- 2025-11-08: Initial design document created
