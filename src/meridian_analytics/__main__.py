"""CLI demonstrations of the concepts this sprint is really about.

    python -m meridian_analytics power      # why sample size explodes
    python -m meridian_analytics peeking    # what repeated checking costs you
    python -m meridian_analytics leakage    # random splits on time series

Each of these is a claim that is much more convincing when you run it than when
you read it. "Peeking is bad practice" is an opinion; "peeking took my false
positive rate from 4.8% to 20% across 2,000 simulations" is a measurement.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
from scipy import stats

from .experiment_design import required_sample_size_means, required_sample_size_proportions
from .forecasting import (
    demonstrate_leakage,
    linear_trend_forecast,
    moving_average_forecast,
    naive_forecast,
    seasonal_naive_forecast,
    walk_forward_validate,
)


def _deposit_series(days: int = 730, seed: int = 42) -> pd.Series:
    """A realistic deposit series: trend, weekly payday cycle, monthly cycle."""
    rng = np.random.default_rng(seed)
    t = np.arange(days)
    values = (
        45_000_000
        + 12_000 * t
        + 900_000 * np.sin(2 * np.pi * t / 7)
        + 1_800_000 * np.sin(2 * np.pi * t / 30.4)
    )
    return pd.Series(
        values + rng.normal(0, 600_000, days),
        index=pd.date_range("2023-01-01", periods=days, freq="D"),
    )


def cmd_power(_args) -> int:
    print("=" * 72)
    print("POWER ANALYSIS: how sample size responds to the effect you chase")
    print("=" * 72)
    print("\nBaseline churn 8%. How many customers do we need?\n")
    print(f"{'MDE (abs)':<12}{'new rate':<12}{'n per group':>14}{'total':>12}")
    print("-" * 50)
    for mde in (0.04, 0.02, 0.01, 0.005, 0.0025):
        r = required_sample_size_proportions(0.08, mde)
        print(f"{mde:<12.2%}{0.08 + mde:<12.1%}{r.n_per_group:>14,}{r.n_total:>12,}")
    print()
    print("n scales with 1/MDE^2 — halving the effect QUADRUPLES the sample.")
    print("Detecting 0.25pp needs ~256x the sample of detecting 4pp.")
    print()
    print("=" * 72)
    print("Variance matters just as much as effect size")
    print("=" * 72)
    noisy = required_sample_size_means(4200, 3100, 150)
    quiet = required_sample_size_means(4200, 1550, 150)
    print(f"\nDetecting a $150 change in average balance:")
    print(f"  with std $3,100:  {noisy.n_per_group:>7,} per group")
    print(f"  with std $1,550:  {quiet.n_per_group:>7,} per group")
    print(f"  -> {noisy.n_per_group / quiet.n_per_group:.0f}x fewer samples for half the variance")
    print()
    print("This is why variance reduction (CUPED, stratification) is valuable:")
    print("it buys statistical power without collecting a single extra sample.")
    return 0


def cmd_peeking(args) -> int:
    n_sims = args.simulations
    final_n, peek_every = 3000, 250

    print("=" * 74)
    print("THE PEEKING PROBLEM")
    print("=" * 74)
    print(f"\n{n_sims:,} simulated experiments where the treatment has NO effect.")
    print("Both groups drawn from the SAME distribution, so every 'significant'")
    print("result below is by definition a false positive.\n")

    rng = np.random.default_rng(7)
    honest = peeking = 0
    for _ in range(n_sims):
        a = rng.normal(100, 15, final_n)
        b = rng.normal(100, 15, final_n)
        if stats.ttest_ind(a, b, equal_var=False).pvalue < 0.05:
            honest += 1
        for k in range(peek_every, final_n + 1, peek_every):
            if stats.ttest_ind(a[:k], b[:k], equal_var=False).pvalue < 0.05:
                peeking += 1
                break

    print(f"{'strategy':<40}{'false positives':>16}{'rate':>10}")
    print("-" * 66)
    print(f"{'Analyse ONCE at the planned end':<40}{honest:>16,}{honest / n_sims:>10.1%}")
    print(
        f"{f'Peek every {peek_every}, stop if p<0.05':<40}{peeking:>16,}{peeking / n_sims:>10.1%}"
    )
    print()
    print(f"Peeking {final_n // peek_every} times inflates the error rate ~{peeking / max(honest, 1):.1f}x.")
    print()
    print("WHY: every look is a fresh chance to cross the threshold by luck.")
    print("A random walk crosses any fixed line eventually if you keep watching.")
    print()
    print("FIX: fix the stopping rule in advance, or use a sequential method")
    print("designed for continuous monitoring.")
    return 0


def cmd_leakage(_args) -> int:
    series = _deposit_series()

    print("=" * 76)
    print("TIME-SERIES LEAKAGE: random splitting vs chronological splitting")
    print("=" * 76)
    print()
    result = demonstrate_leakage(series, horizon=30)
    print(result["random_split"].summary())
    print(result["chronological_split"].summary())
    print()
    print(f"Honest error is {result['optimism_ratio']:.1f}x WORSE than the leaky backtest.")
    print("Same data, same model — only the split differs.")
    print()
    print("The random split lets the model see points on BOTH SIDES of each test")
    print("point, so it interpolates instead of forecasting. Ship that and your")
    print("production error is several times what the backtest promised.")
    print()
    print("=" * 76)
    print("WALK-FORWARD VALIDATION: baselines, 30-day horizon")
    print("=" * 76)
    print()
    models = [
        (naive_forecast, "naive (last value)"),
        (lambda s, h: seasonal_naive_forecast(s, h, 7), "seasonal naive (7d)"),
        (lambda s, h: moving_average_forecast(s, h, 7), "moving average (7d)"),
        (lambda s, h: moving_average_forecast(s, h, 30), "moving average (30d)"),
        (linear_trend_forecast, "linear trend"),
    ]
    results = []
    for fn, name in models:
        m = walk_forward_validate(series, fn, 365, 30, name, step=30)
        results.append(m)
        print(m.summary())

    best = min(results, key=lambda r: r.rmse)
    base = next(r for r in results if r.model == "naive (last value)")
    print()
    print(f"BEST by RMSE: {best.model} — beats naive by {1 - best.rmse / base.rmse:.1%}")
    print()
    print("Always report against a naive baseline. A model that cannot beat")
    print("'tomorrow equals today' is decoration, not a model.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="meridian_analytics")
    sub = parser.add_subparsers(dest="command", required=True)

    p_power = sub.add_parser("power", help="power analysis demonstration")
    p_power.set_defaults(func=cmd_power)

    p_peek = sub.add_parser("peeking", help="how peeking inflates false positives")
    p_peek.add_argument("--simulations", type=int, default=2000)
    p_peek.set_defaults(func=cmd_peeking)

    p_leak = sub.add_parser("leakage", help="time-series leakage from random splits")
    p_leak.set_defaults(func=cmd_leakage)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
