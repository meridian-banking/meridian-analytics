"""Tests for the statistics and forecasting modules.

TESTING STATISTICAL CODE IS DIFFERENT
You are usually not asserting an exact number — randomness makes that
impossible — but a PROPERTY that must hold: an effect is detected when it is
real, a relationship has the expected sign, a validated error is worse than a
leaky one. Where randomness is involved, seeds are fixed so failures are
reproducible rather than mysterious.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from meridian_analytics.ab_testing import (
    bonferroni_correct,
    compare_distributions,
    compare_means,
    compare_proportions,
)
from meridian_analytics.experiment_design import (
    ExperimentPlan,
    check_randomization_balance,
    required_sample_size_means,
    required_sample_size_proportions,
)
from meridian_analytics.forecasting import (
    demonstrate_leakage,
    evaluate_forecast,
    expanding_window_splits,
    naive_forecast,
    seasonal_naive_forecast,
    walk_forward_validate,
)
from meridian_analytics.regression import bootstrap_ci, fit_with_diagnostics

# --- power analysis ---------------------------------------------------------


def test_smaller_effects_need_larger_samples():
    """The core relationship: n scales with 1/MDE^2."""
    big = required_sample_size_proportions(0.08, 0.04)
    small = required_sample_size_proportions(0.08, 0.02)
    assert small.n_per_group > big.n_per_group


def test_sample_size_scales_quadratically():
    """Halving the MDE should roughly quadruple n."""
    a = required_sample_size_proportions(0.10, 0.04)
    b = required_sample_size_proportions(0.10, 0.02)
    ratio = b.n_per_group / a.n_per_group
    assert 3.5 < ratio < 4.5, f"expected ~4x, got {ratio:.2f}x"


def test_higher_power_needs_more_samples():
    low = required_sample_size_proportions(0.08, 0.02, power=0.80)
    high = required_sample_size_proportions(0.08, 0.02, power=0.95)
    assert high.n_per_group > low.n_per_group


def test_stricter_alpha_needs_more_samples():
    lenient = required_sample_size_proportions(0.08, 0.02, alpha=0.05)
    strict = required_sample_size_proportions(0.08, 0.02, alpha=0.01)
    assert strict.n_per_group > lenient.n_per_group


def test_lower_variance_needs_fewer_samples():
    """Why variance reduction buys power without extra data."""
    noisy = required_sample_size_means(4200, 3100, 150)
    quiet = required_sample_size_means(4200, 1550, 150)
    assert quiet.n_per_group < noisy.n_per_group


def test_proportion_and_mean_results_format_differently():
    """A rate reads as a percentage; a balance does not."""
    prop = required_sample_size_proportions(0.08, 0.02)
    mean = required_sample_size_means(4200, 3100, 150)
    assert "%" in prop.summary()
    assert "4,200.00" in mean.summary()


# --- pre-registration -------------------------------------------------------


def test_experiment_plan_roundtrips(tmp_path):
    """The plan must persist — its committed timestamp is the evidence that
    the decisions preceded the data."""
    plan = ExperimentPlan(
        name="fee_increase",
        hypothesis="A $3 fee increase raises 90-day churn",
        primary_metric="churn_90d",
        secondary_metrics=["avg_balance"],
        minimum_detectable_effect=0.02,
        required_n_per_group=3213,
        planned_duration_days=90,
    )
    path = plan.save(tmp_path / "plan.json")
    loaded = ExperimentPlan.load(path)
    assert loaded.primary_metric == "churn_90d"
    assert loaded.required_n_per_group == 3213


def test_balanced_randomization_passes():
    """Inverted logic: here a HIGH p-value is the good outcome."""
    rng = np.random.default_rng(1)
    a, b = rng.normal(700, 50, 2000), rng.normal(700, 50, 2000)
    assert check_randomization_balance(a, b, "credit_score")["balanced"]


def test_broken_randomization_is_detected():
    """If groups differ on a pre-existing covariate, randomisation failed and
    no downstream analysis can fix it."""
    rng = np.random.default_rng(1)
    a, b = rng.normal(700, 50, 2000), rng.normal(730, 50, 2000)
    assert not check_randomization_balance(a, b, "credit_score")["balanced"]


# --- A/B analysis -----------------------------------------------------------


def test_real_effect_is_detected():
    result = compare_proportions(240, 3000, 330, 3000, "churn")
    assert result.significant
    assert not result.ci_includes_zero


def test_no_effect_is_not_detected():
    result = compare_proportions(240, 3000, 244, 3000, "churn")
    assert not result.significant
    assert result.ci_includes_zero


def test_ci_and_pvalue_always_agree():
    """Two views of the same computation — they must never contradict."""
    for treat in (240, 260, 290, 330, 400):
        r = compare_proportions(240, 3000, treat, 3000, "churn")
        assert r.significant == (not r.ci_includes_zero)


def test_welch_handles_unequal_variance():
    """Welch's is the right default; it costs nothing when variances match."""
    rng = np.random.default_rng(2)
    a = rng.normal(100, 5, 500)
    b = rng.normal(100, 40, 500)  # same mean, very different spread
    result = compare_means(a, b)
    assert not result.significant


def test_mann_whitney_detects_shift_in_skewed_data():
    rng = np.random.default_rng(3)
    a = rng.lognormal(8.0, 1.0, 1500)
    b = rng.lognormal(8.3, 1.0, 1500)
    assert compare_distributions(a, b)["p_value"] < 0.05


def test_multiple_testing_inflates_error():
    """20 tests at 5% gives a ~64% chance of at least one false positive."""
    result = bonferroni_correct([0.04] * 20)
    assert result["family_wise_error_if_uncorrected"] > 0.60
    assert not any(result["significant_after"])
    assert all(result["significant_before"])


# --- regression -------------------------------------------------------------


def test_regression_recovers_known_coefficients():
    rng = np.random.default_rng(4)
    n = 3000
    x1, x2 = rng.normal(0, 1, n), rng.normal(0, 1, n)
    y = 5 + 2.0 * x1 - 3.0 * x2 + rng.normal(0, 0.5, n)
    rep = fit_with_diagnostics(pd.Series(y), pd.DataFrame({"x1": x1, "x2": x2}))
    assert abs(rep.coefficients.loc["x1", "coefficient"] - 2.0) < 0.1
    assert abs(rep.coefficients.loc["x2", "coefficient"] + 3.0) < 0.1


def test_multicollinearity_is_flagged():
    """High VIF: the model cannot tell near-duplicate predictors apart."""
    rng = np.random.default_rng(5)
    n = 1000
    x1 = rng.normal(0, 1, n)
    x2 = x1 + rng.normal(0, 0.05, n)  # almost identical to x1
    y = 3 * x1 + rng.normal(0, 1, n)
    rep = fit_with_diagnostics(pd.Series(y), pd.DataFrame({"x1": x1, "x2": x2}))
    vifs = [d for d in rep.diagnostics if d.name.startswith("VIF")]
    assert any(not d.passed for d in vifs)


def test_heteroscedasticity_is_flagged():
    """Error variance growing with x is the normal case in finance."""
    rng = np.random.default_rng(6)
    n = 2000
    x = rng.uniform(1, 100, n)
    y = 2 * x + rng.normal(0, x * 0.5, n)
    rep = fit_with_diagnostics(pd.Series(y), pd.DataFrame({"x": x}))
    bp = next(d for d in rep.diagnostics if d.name == "Breusch-Pagan")
    assert not bp.passed


def test_robust_errors_change_uncertainty_not_estimates():
    """HC3 leaves coefficients alone and fixes the standard errors."""
    rng = np.random.default_rng(7)
    n = 2000
    x = rng.uniform(1, 100, n)
    y = 2 * x + rng.normal(0, x * 0.5, n)
    X, ys = pd.DataFrame({"x": x}), pd.Series(y)
    plain = fit_with_diagnostics(ys, X, robust=False)
    robust = fit_with_diagnostics(ys, X, robust=True)
    assert (
        abs(
            plain.coefficients.loc["x", "coefficient"] - robust.coefficients.loc["x", "coefficient"]
        )
        < 1e-8
    )
    assert plain.coefficients.loc["x", "std_error"] != robust.coefficients.loc["x", "std_error"]


# --- bootstrap --------------------------------------------------------------


def test_bootstrap_ci_contains_true_mean():
    rng = np.random.default_rng(8)
    data = rng.normal(100, 15, 1000)
    result = bootstrap_ci(data, np.mean, n_resamples=2000)
    lo, hi = result["bootstrap_ci"]
    assert lo < 100 < hi


def test_bootstrap_works_for_statistics_with_no_formula():
    """The median has no simple analytic CI. The bootstrap does not care."""
    rng = np.random.default_rng(9)
    data = rng.lognormal(8.0, 1.2, 800)
    result = bootstrap_ci(data, np.median, n_resamples=2000)
    lo, hi = result["bootstrap_ci"]
    assert lo < np.median(data) < hi


# --- forecasting ------------------------------------------------------------


@pytest.fixture
def deposit_series() -> pd.Series:
    """A realistic deposit series: trend, weekly payday cycle, monthly cycle.

    The multiple overlapping cycles matter for the leakage test below. With
    only a simple weekly pattern, a local model extrapolates almost as well as
    it interpolates and the leakage effect is muted. Richer local structure is
    exactly what a leaky split lets a model cheat on — which is also why real
    series, which have plenty of it, are so vulnerable.
    """
    rng = np.random.default_rng(42)
    t = np.arange(730)
    values = (
        45_000_000
        + 12_000 * t
        + 900_000 * np.sin(2 * np.pi * t / 7)
        + 1_800_000 * np.sin(2 * np.pi * t / 30.4)
    )
    return pd.Series(values + rng.normal(0, 600_000, 730))


def test_expanding_windows_never_train_on_the_future():
    """THE critical property. Every training index must precede every test
    index — otherwise the model is shown the answer."""
    for train_idx, test_idx in expanding_window_splits(200, 100, 10):
        assert train_idx.max() < test_idx.min()


def test_expanding_windows_grow():
    sizes = [len(tr) for tr, _ in expanding_window_splits(200, 100, 10)]
    assert sizes == sorted(sizes)
    assert sizes[-1] > sizes[0]


def test_random_splitting_produces_optimistic_results(deposit_series):
    """Leakage is not theoretical — a leaky backtest genuinely looks better.

    We assert the DIRECTION strongly and the magnitude loosely. The size of the
    optimism depends on how much local structure the series has and how
    flexible the model is; the direction is the invariant, and is what makes
    random splitting on time series wrong in every case rather than most.
    """
    result = demonstrate_leakage(deposit_series, horizon=30)
    assert result["random_split"].mape < result["chronological_split"].mape
    assert (
        result["optimism_ratio"] > 1.5
    ), f"expected the honest split to be clearly worse, got {result['optimism_ratio']:.2f}x"


def test_seasonal_naive_beats_naive_on_seasonal_data():
    """A strong baseline on data with rhythm."""
    t = np.arange(400)
    seasonal = pd.Series(1000 + 300 * np.sin(2 * np.pi * t / 7))
    naive_m = walk_forward_validate(seasonal, naive_forecast, 200, 7, "naive", step=7)
    seasonal_m = walk_forward_validate(
        seasonal, lambda s, h: seasonal_naive_forecast(s, h, 7), 200, 7, "seasonal", step=7
    )
    assert seasonal_m.rmse < naive_m.rmse


def test_rmse_punishes_large_errors_more_than_mae():
    """Why you report both: they disagree in an informative way."""
    actual = np.array([100.0] * 10)
    many_small = np.array([105.0] * 10)
    one_huge = np.array([100.0] * 9 + [150.0])
    m1 = evaluate_forecast(actual, many_small, "many small")
    m2 = evaluate_forecast(actual, one_huge, "one huge")
    assert m1.mae == m2.mae  # identical average error
    assert m2.rmse > m1.rmse  # but RMSE penalises the outlier


def test_forecast_metrics_are_non_negative(deposit_series):
    m = walk_forward_validate(deposit_series, naive_forecast, 300, 14, "naive")
    assert m.mae >= 0 and m.rmse >= 0 and m.mape >= 0
    assert m.rmse >= m.mae  # RMSE >= MAE always, by Jensen's inequality
