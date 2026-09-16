# meridian-analytics

Statistics, experimentation, and forecasting for the Meridian platform. Where the previous six repos built pipes and schemas, this one starts **asking questions of the data** — and, more importantly, asking them in a way that doesn't fool you.

## What's here

| Module | Purpose |
|---|---|
| `experiment_design` | Power analysis, pre-registration, randomisation balance checks |
| `ab_testing` | Proportion and mean comparisons, non-parametric cross-checks, multiple-testing correction |
| `regression` | OLS with a full diagnostic battery, robust standard errors, bootstrap CIs |
| `forecasting` | Baselines, walk-forward validation, and a leakage demonstration |

## Run the demonstrations

Three claims in this repo are far more convincing when you run them than when you read them:

```bash
python -m meridian_analytics power      # why sample size explodes
python -m meridian_analytics peeking    # what repeated checking costs you
python -m meridian_analytics leakage    # random splits on time series
```

**Measured results** (reproducible, seeded):

- **Power:** detecting a 0.25pp churn change needs ~256× the sample of a 4pp change. `n` scales with `1/MDE²`.
- **Peeking:** analysing once gives a 4.8% false-positive rate, as designed. Peeking 12 times inflates it to **~20–23%**, across 2,000 simulations where the treatment had *literally no effect*.
- **Leakage:** a random train/test split on a time series reported 1.23% MAPE. The honest chronological split reported 3.26% — **2.6× worse**. Same data, same model, only the split differs.

## The ideas that matter

**Power analysis happens *before* you see data.** A test too small to detect the effect you care about will report "no difference" from a study that never could have found one. That's a non-result, not a null result, and reporting it as evidence of no effect is wrong.

**Pre-registration is evidence, not honour.** [`experiments/fee_increase_2024q2.json`](experiments/) is committed to git — its timestamp proves the primary metric was chosen before the data existed. Without that, you'll unconsciously gravitate to whichever of your twelve metrics happened to look good, and with twelve metrics at 5% you expect roughly one false positive by chance.

**A p-value answers one narrow question.** It doesn't tell you the probability the null is true, whether the result will replicate, how big the effect is, or whether it matters. With enough data, trivial differences become "significant." Always report the effect size and confidence interval alongside — that's what the business actually asked.

**Fitting is easy; trusting the fit is the work.** `sm.OLS(...).fit()` returns a confident-looking table whether or not its assumptions hold. Nothing errors. If they're violated, coefficients may be fine while every standard error and p-value is wrong — you end up certain about something you should be unsure of. The diagnostic that fires most often in finance is heteroscedasticity (error scales with size); the fix is `cov_type='HC3'`, which leaves estimates alone and makes the uncertainty honest. In our test data, naive standard errors understated the income coefficient's uncertainty by **46%**.

**You cannot use random cross-validation on time series.** Shuffling puts future rows in the training set, so the model interpolates between known points instead of extrapolating past the last one. A global linear trend is nearly immune to this; *any model that learns local structure* — trees, boosting, neural nets — is devastated by it. That's why the warning matters in practice and not just in theory.

**Always compare against a naive baseline.** "Tomorrow equals today" is provably optimal on a random walk, and many financial series are close to random walks. A model that can't beat it is decoration. Our walk-forward comparison: the 30-day moving average beat naive by 26.5% on RMSE; the 7-day seasonal naive *lost* to it.

## Quick start

```bash
pip install -e ".[dev]"
make test    # 27 tests, no database or infrastructure required
make lint
```

Statistical code here is pure — functions over arrays and DataFrames — which is why the tests need no infrastructure at all.

## Testing statistical code

You're usually not asserting an exact number (randomness makes that impossible) but a **property that must hold**: an effect is detected when real, a relationship has the expected sign, a validated error exceeds a leaky one, RMSE ≥ MAE always. Seeds are fixed so failures are reproducible rather than mysterious.

See [`docs/adr/`](docs/adr/) for the decisions behind pre-registration and validation strategy.

Part of the 8-repository Meridian platform.


_Verified locally: peeking inflated the false-positive rate from 4.8% to 20.0% across 2,000 null simulations, and a leaky time-series split understated forecast error by 2.5x versus the honest chronological split._