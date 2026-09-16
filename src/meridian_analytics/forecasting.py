"""Time-series forecasting, and the validation discipline it demands.

THE SINGLE MOST IMPORTANT RULE IN THIS FILE
You cannot use random cross-validation on time series. Ever.

Random k-fold shuffles rows into folds. On time-ordered data that means some
training rows come from AFTER the test rows — the model is shown the future and
asked to predict the past. It will do brilliantly, because interpolating
between two known points is trivial compared to extrapolating past the last one.

The result is a backtest that looks excellent and a live model that fails.
Nothing errors. The metrics are real numbers. They are simply measuring a task
nobody will ever ask the model to perform. We demonstrate this below with
actual numbers rather than asserting it.

THE CORRECT APPROACH: expanding-window (walk-forward) validation.
  train [------]                    test [--]
  train [---------]                      test [--]
  train [------------]                        test [--]
Training data always precedes test data, and the split mimics how the model
will really be used: everything known up to now, predict what comes next.

ALWAYS COMPARE AGAINST A NAIVE BASELINE.
The naive forecast — "tomorrow equals today", or "this month equals last
year's same month" — is often hard to beat on financial series. A sophisticated
model that cannot beat it is not sophisticated, it is decorative. Reporting
MAPE without a baseline hides this completely.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ForecastMetrics:
    """Accuracy of a forecast against actuals."""

    model: str
    mae: float
    rmse: float
    mape: float
    n_predictions: int

    def summary(self) -> str:
        return (
            f"  {self.model:<24} MAE={self.mae:>10,.1f}  RMSE={self.rmse:>10,.1f}  "
            f"MAPE={self.mape:>6.2f}%  (n={self.n_predictions})"
        )


def evaluate_forecast(actual: np.ndarray, predicted: np.ndarray, model: str) -> ForecastMetrics:
    """Standard forecast error metrics, and what each is for.

    MAE  — mean absolute error. Same units as the data, easy to explain to a
           stakeholder ("off by about $40k on average"). Treats all errors
           proportionally.
    RMSE — root mean squared error. Squares errors first, so it PUNISHES LARGE
           MISSES disproportionately. Prefer it when one big error is much
           worse than several small ones, which is usually true for liquidity.
    MAPE — mean absolute percentage error. Unit-free, so it compares across
           series of different scales. Breaks down badly near zero (division by
           a tiny number explodes) and penalises over-forecasting more than
           under-forecasting.

    Report at least two. A model can win on MAE and lose on RMSE, which tells
    you it makes many small errors rather than a few large ones — a genuinely
    useful distinction that either metric alone would hide.
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    errors = actual - predicted
    return ForecastMetrics(
        model=model,
        mae=float(np.mean(np.abs(errors))),
        rmse=float(np.sqrt(np.mean(errors**2))),
        mape=float(np.mean(np.abs(errors / actual)) * 100),
        n_predictions=len(actual),
    )


# --- baselines --------------------------------------------------------------


def naive_forecast(series: pd.Series, horizon: int) -> np.ndarray:
    """ "Tomorrow equals today." The baseline every model must beat.

    Sounds trivially weak, and on a random walk it is provably OPTIMAL — no
    model can do better, because the next value genuinely is the current value
    plus unpredictable noise. Many financial series are close to random walks,
    which is exactly why this baseline embarrasses so many sophisticated
    models.
    """
    return np.repeat(series.iloc[-1], horizon)


def seasonal_naive_forecast(series: pd.Series, horizon: int, season_length: int) -> np.ndarray:
    """ "This period equals the same period last season."

    A much stronger baseline on anything with weekly or annual rhythm — and
    banking deposits have both (payday cycles, tax season, holiday spending).
    """
    last_season = series.iloc[-season_length:].values
    reps = int(np.ceil(horizon / season_length))
    return np.tile(last_season, reps)[:horizon]


def moving_average_forecast(series: pd.Series, horizon: int, window: int = 7) -> np.ndarray:
    """Flat forecast at the recent mean. Smooths noise, ignores trend."""
    return np.repeat(series.iloc[-window:].mean(), horizon)


def linear_trend_forecast(series: pd.Series, horizon: int) -> np.ndarray:
    """Fit a straight line and extend it.

    Captures trend, ignores seasonality, and extrapolates confidently forever —
    which is its danger. A linear trend fitted to a boom happily forecasts the
    boom continuing indefinitely.
    """
    x = np.arange(len(series))
    slope, intercept = np.polyfit(x, series.values, 1)
    future_x = np.arange(len(series), len(series) + horizon)
    return intercept + slope * future_x


# --- validation -------------------------------------------------------------


def expanding_window_splits(n: int, initial_train: int, horizon: int, step: int | None = None):
    """Generate (train_idx, test_idx) pairs for walk-forward validation.

    Each fold trains on everything up to a point and tests on what comes next.
    Training data ALWAYS precedes test data, which is the whole point.
    """
    step = step or horizon
    start = initial_train
    while start + horizon <= n:
        yield np.arange(0, start), np.arange(start, start + horizon)
        start += step


def walk_forward_validate(
    series: pd.Series,
    forecast_fn,
    initial_train: int,
    horizon: int,
    model_name: str,
    step: int | None = None,
) -> ForecastMetrics:
    """Validate a forecast function the way it will actually be used.

    Aggregates errors across every fold, so the metric reflects performance
    over many different starting points rather than one lucky window. A single
    train/test split on time series is a sample of size one.
    """
    all_actual, all_pred = [], []
    for train_idx, test_idx in expanding_window_splits(len(series), initial_train, horizon, step):
        train = series.iloc[train_idx]
        actual = series.iloc[test_idx].values
        pred = forecast_fn(train, len(test_idx))
        all_actual.extend(actual)
        all_pred.extend(pred)
    return evaluate_forecast(np.array(all_actual), np.array(all_pred), model_name)


def demonstrate_leakage(series: pd.Series, horizon: int = 30, seed: int = 42) -> dict:
    """Show, with numbers, what random splitting does to a time series.

    We use a NEAREST-NEIGHBOUR style model (predict each point from the values
    closest to it in time) rather than a global linear fit. That choice is
    deliberate and is itself the lesson:

      A global linear trend is almost immune to leakage, because a straight
      line fitted to any subset of trending data is roughly the same line —
      seeing the future adds nothing.

      A flexible model that learns LOCAL structure is devastated by leakage.
      Under a random split it can look up the neighbouring days, which sit on
      both sides of the target. Under a chronological split it must
      extrapolate past the end of everything it has seen.

    Most real models — trees, boosting, neural nets, anything with enough
    capacity to fit local patterns — behave like the second case. That is why
    the leakage warning matters in practice and not just in theory.
    """
    rng = np.random.default_rng(seed)
    y = series.values.astype(float)
    n = len(y)
    idx = np.arange(n)

    def knn_predict(train_idx: np.ndarray, test_idx: np.ndarray, k: int = 5) -> np.ndarray:
        """Average the k training points nearest in TIME to each test point."""
        preds = []
        for t in test_idx:
            distances = np.abs(train_idx - t)
            nearest = train_idx[np.argsort(distances)[:k]]
            preds.append(y[nearest].mean())
        return np.array(preds)

    # --- random split: the wrong way. Test points are surrounded by training
    # points on BOTH sides, so the model interpolates rather than forecasts.
    perm = rng.permutation(n)
    cut = n - horizon
    rand_train, rand_test = np.sort(perm[:cut]), np.sort(perm[cut:])
    random_metrics = evaluate_forecast(
        y[rand_test], knn_predict(rand_train, rand_test), "random split (LEAKY)"
    )

    # --- chronological split: the honest way. Every test point lies beyond the
    # end of the training data, so the model must genuinely extrapolate.
    chrono_train, chrono_test = idx[:cut], idx[cut:]
    chrono_metrics = evaluate_forecast(
        y[chrono_test], knn_predict(chrono_train, chrono_test), "chronological split"
    )

    return {
        "random_split": random_metrics,
        "chronological_split": chrono_metrics,
        "optimism_ratio": (
            chrono_metrics.mape / random_metrics.mape if random_metrics.mape else float("nan")
        ),
    }


def decompose(series: pd.Series, period: int) -> pd.DataFrame:
    """Split a series into trend, seasonal, and residual components.

    Worth doing BEFORE modelling: it tells you whether there is seasonality to
    capture, whether the trend is linear, and how much of the variation is
    simply noise that no model will ever explain.
    """
    from statsmodels.tsa.seasonal import seasonal_decompose

    result = seasonal_decompose(series, model="additive", period=period)
    return pd.DataFrame(
        {
            "observed": result.observed,
            "trend": result.trend,
            "seasonal": result.seasonal,
            "residual": result.resid,
        }
    )
