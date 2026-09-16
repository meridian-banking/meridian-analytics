"""Regression with diagnostics: fitting is easy, trusting the fit is the work.

WHY DIAGNOSTICS MATTER
`sm.OLS(y, X).fit()` always returns coefficients, standard errors, p-values and
an R-squared. It returns them whether or not the assumptions behind them hold.
Nothing errors. Nothing warns. You get a confident-looking table either way.

If the assumptions are violated, the COEFFICIENTS may still be roughly fine —
but the STANDARD ERRORS, and therefore every p-value and confidence interval,
can be badly wrong. You end up certain about something you should be unsure of,
which is worse than having no model at all.

THE FOUR ASSUMPTIONS WORTH CHECKING, and what each failure does:

  LINEARITY — the relationship really is a straight line. If it curves, the
    model is systematically wrong in a pattern you can see in the residuals.

  INDEPENDENCE — observations do not influence each other. Time series violate
    this constantly (today resembles yesterday), which shrinks standard errors
    and makes everything look more significant than it is.

  HOMOSCEDASTICITY — residual variance is constant across fitted values. In
    finance this is almost always violated: prediction error on a $1M balance
    is naturally larger than on a $1k balance. Fix with robust standard errors.

  NO PERFECT MULTICOLLINEARITY — predictors are not near-duplicates. When they
    are, the model cannot tell which one deserves the credit, so individual
    coefficients become unstable and their p-values meaningless — even while
    overall predictions stay fine.

The last one has a distinctive symptom worth memorising: a high overall
R-squared with no individually significant predictors. The model clearly knows
something; it just cannot attribute it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson


@dataclass
class DiagnosticResult:
    name: str
    statistic: float
    p_value: float | None
    passed: bool
    interpretation: str

    def summary(self) -> str:
        status = "OK  " if self.passed else "FAIL"
        p = f"p={self.p_value:.4f}" if self.p_value is not None else ""
        return (
            f"  [{status}] {self.name:<26} stat={self.statistic:>8.3f} "
            f"{p:<14} {self.interpretation}"
        )


@dataclass
class RegressionReport:
    n_obs: int
    r_squared: float
    adj_r_squared: float
    f_pvalue: float
    coefficients: pd.DataFrame
    diagnostics: list[DiagnosticResult] = field(default_factory=list)

    @property
    def all_diagnostics_passed(self) -> bool:
        return all(d.passed for d in self.diagnostics)

    def summary(self) -> str:
        lines = [
            f"OLS regression  n={self.n_obs:,}",
            f"  R-squared {self.r_squared:.4f}   adjusted {self.adj_r_squared:.4f}"
            f"   F-test p={self.f_pvalue:.3g}",
            "",
            "COEFFICIENTS",
            self.coefficients.to_string(),
            "",
            "DIAGNOSTICS",
        ]
        lines.extend(d.summary() for d in self.diagnostics)
        if not self.all_diagnostics_passed:
            lines.append("")
            lines.append(
                "  NOTE: a failed diagnostic does not invalidate the coefficients, but it "
                "does mean\n        the standard errors — and so every p-value above — "
                "should not be trusted as printed."
            )
        return "\n".join(lines)


def check_multicollinearity(X: pd.DataFrame, threshold: float = 5.0) -> list[DiagnosticResult]:
    """Variance Inflation Factor per predictor.

    VIF answers: how much is this coefficient's variance inflated by its
    correlation with the OTHER predictors? VIF = 1 means uncorrelated. The
    usual rules of thumb are >5 concerning, >10 serious.

    THE INTUITION: VIF_j = 1 / (1 - R²_j), where R²_j comes from regressing
    predictor j on all the others. If the other predictors explain 90% of
    predictor j, then R²_j = 0.9 and VIF = 10 — the model genuinely cannot
    separate j's contribution from theirs.
    """
    results = []
    X_const = sm.add_constant(X)
    for i, col in enumerate(X_const.columns):
        if col == "const":
            continue
        vif = variance_inflation_factor(X_const.values, i)
        results.append(
            DiagnosticResult(
                name=f"VIF: {col}",
                statistic=float(vif),
                p_value=None,
                passed=vif < threshold,
                interpretation=(
                    "independent enough"
                    if vif < threshold
                    else f"correlated with other predictors (>{threshold})"
                ),
            )
        )
    return results


def check_homoscedasticity(model) -> DiagnosticResult:
    """Breusch-Pagan test for constant residual variance.

    H0: variance is constant. A LOW p-value means heteroscedasticity.

    Very common in financial data, because error naturally scales with size.
    The fix is usually not to abandon the model but to use heteroscedasticity-
    robust standard errors: `model.fit(cov_type='HC3')`. Coefficients stay the
    same; the uncertainty around them gets honest.
    """
    lm, lm_p, _, _ = het_breuschpagan(model.resid, model.model.exog)
    return DiagnosticResult(
        name="Breusch-Pagan",
        statistic=float(lm),
        p_value=float(lm_p),
        passed=lm_p > 0.05,
        interpretation=(
            "constant variance" if lm_p > 0.05 else "heteroscedastic — use robust SEs (HC3)"
        ),
    )


def check_normality_of_residuals(model) -> DiagnosticResult:
    """Jarque-Bera test on residuals.

    IMPORTANT CONTEXT, because this one is over-weighted by beginners:
    OLS does NOT require normally distributed residuals for the coefficients to
    be unbiased. Normality matters for exact small-sample inference. With a few
    thousand observations the Central Limit Theorem does the work regardless,
    and this test becomes so sensitive that it rejects on trivial deviations.

    Treat a failure here as "look at a residual plot", not "the model is
    invalid". At large n it is closer to a curiosity than a verdict.
    """
    jb, jb_p, skew, kurtosis = sm.stats.stattools.jarque_bera(model.resid)
    return DiagnosticResult(
        name="Jarque-Bera (normality)",
        statistic=float(jb),
        p_value=float(jb_p),
        passed=jb_p > 0.05,
        interpretation=(
            "residuals ~ normal"
            if jb_p > 0.05
            else f"non-normal (skew={skew:.2f}, kurt={kurtosis:.2f}); usually fine at large n"
        ),
    )


def check_autocorrelation(model) -> DiagnosticResult:
    """Durbin-Watson: are residuals correlated with their neighbours?

    Ranges 0-4; about 2 means no autocorrelation. Below ~1.5 suggests positive
    autocorrelation, which is the norm in time series and badly deflates
    standard errors — making everything look more significant than it is.

    Only meaningful when rows have a natural order. On a cross-section of
    customers it is not interpretable.
    """
    dw = durbin_watson(model.resid)
    ok = 1.5 < dw < 2.5
    return DiagnosticResult(
        name="Durbin-Watson",
        statistic=float(dw),
        p_value=None,
        passed=bool(ok),
        interpretation=(
            "no autocorrelation"
            if ok
            else ("positive autocorrelation" if dw <= 1.5 else "negative autocorrelation")
        ),
    )


def fit_with_diagnostics(
    y: pd.Series,
    X: pd.DataFrame,
    robust: bool = False,
    check_autocorr: bool = False,
) -> RegressionReport:
    """Fit OLS and run the full diagnostic battery.

    `robust=True` uses HC3 heteroscedasticity-robust standard errors — the
    standard response to a failed Breusch-Pagan test.
    """
    X_const = sm.add_constant(X)
    model = sm.OLS(y, X_const).fit(cov_type="HC3" if robust else "nonrobust")

    coefs = pd.DataFrame(
        {
            "coefficient": model.params,
            "std_error": model.bse,
            "t_stat": model.tvalues,
            "p_value": model.pvalues,
            "ci_lower": model.conf_int()[0],
            "ci_upper": model.conf_int()[1],
        }
    ).round(4)

    diagnostics = check_multicollinearity(X)
    diagnostics.append(check_homoscedasticity(model))
    diagnostics.append(check_normality_of_residuals(model))
    if check_autocorr:
        diagnostics.append(check_autocorrelation(model))

    return RegressionReport(
        n_obs=int(model.nobs),
        r_squared=float(model.rsquared),
        adj_r_squared=float(model.rsquared_adj),
        f_pvalue=float(model.f_pvalue),
        coefficients=coefs,
        diagnostics=diagnostics,
    )


def bootstrap_ci(
    data: np.ndarray,
    statistic=np.mean,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict:
    """Confidence interval by resampling, making almost no assumptions.

    HOW IT WORKS, and why it feels like cheating but is not:
    Draw a sample of the same size from your data WITH replacement, compute the
    statistic, repeat thousands of times. The spread of those values estimates
    the sampling distribution of the statistic. Take the 2.5th and 97.5th
    percentiles.

    WHY IT MATTERS HERE
    The analytic CI formula assumes a known sampling distribution — usually
    normal. Financial data is often log-normal, and for statistics like the
    MEDIAN or a percentile there may be no clean formula at all. The bootstrap
    works for any statistic you can compute.

    THE COST: it needs enough data to represent the population. Bootstrapping
    from 8 observations resamples the same 8 values forever and produces a
    confident interval built on nothing.
    """
    rng = np.random.default_rng(seed)
    data = np.asarray(data)
    boot = np.array(
        [statistic(rng.choice(data, size=len(data), replace=True)) for _ in range(n_resamples)]
    )
    lo = (1 - confidence) / 2 * 100
    hi = (1 + confidence) / 2 * 100
    point = float(statistic(data))

    # Analytic comparison, valid only for the mean and only under normality.
    se = float(np.std(data, ddof=1) / np.sqrt(len(data)))
    z = stats.norm.ppf((1 + confidence) / 2)

    return {
        "point_estimate": point,
        "bootstrap_ci": (float(np.percentile(boot, lo)), float(np.percentile(boot, hi))),
        "analytic_ci_for_mean": (point - z * se, point + z * se),
        "n_resamples": n_resamples,
        "confidence": confidence,
    }
