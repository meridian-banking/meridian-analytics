"""Analysing an A/B test, after the pre-registered plan has been executed.

WHAT THIS MODULE IS CAREFUL ABOUT

A p-value answers exactly one narrow question: "if there were truly no effect,
how surprising would data this extreme be?" It does NOT tell you:
  - the probability the null hypothesis is true
  - the probability your result will replicate
  - how big the effect is
  - whether the effect matters

That last one is the important one commercially. With a large enough sample,
trivially small differences become statistically significant. "Significant" is
a statement about EVIDENCE, not about IMPORTANCE. Always report the effect size
and a confidence interval alongside any p-value, because those answer "how big,
and how sure are we?" — which is what the business actually asked.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class TestResult:
    """Result of a two-group comparison."""

    metric: str
    test_name: str
    control_n: int
    treatment_n: int
    control_value: float
    treatment_value: float
    absolute_effect: float
    relative_effect: float
    ci_lower: float
    ci_upper: float
    p_value: float
    alpha: float
    is_proportion: bool = False

    @property
    def significant(self) -> bool:
        return self.p_value < self.alpha

    @property
    def ci_includes_zero(self) -> bool:
        """A CI containing zero means 'no effect' remains a plausible value.

        This is always consistent with the p-value at the same alpha — they are
        two views of the same computation — but the interval is far more
        useful, because it shows the RANGE of effects the data is compatible
        with rather than collapsing everything to one binary verdict.
        """
        return self.ci_lower <= 0 <= self.ci_upper

    def summary(self) -> str:
        fmt = (lambda x: f"{x:.3%}") if self.is_proportion else (lambda x: f"{x:,.2f}")
        verdict = "SIGNIFICANT" if self.significant else "not significant"
        lines = [
            f"{self.metric}  ({self.test_name})",
            f"  control    n={self.control_n:>7,}  value={fmt(self.control_value)}",
            f"  treatment  n={self.treatment_n:>7,}  value={fmt(self.treatment_value)}",
            f"  effect     {fmt(self.absolute_effect)} absolute "
            f"({self.relative_effect:+.1%} relative)",
            f"  95% CI     [{fmt(self.ci_lower)}, {fmt(self.ci_upper)}]",
            f"  p-value    {self.p_value:.4f}  -> {verdict} at alpha={self.alpha}",
        ]
        return "\n".join(lines)


def compare_proportions(
    control_successes: int,
    control_n: int,
    treatment_successes: int,
    treatment_n: int,
    metric: str = "rate",
    alpha: float = 0.05,
) -> TestResult:
    """Two-proportion z-test — the standard test for rate metrics like churn."""
    p1 = control_successes / control_n
    p2 = treatment_successes / treatment_n

    # Pooled proportion under the null (both groups the same).
    p_pool = (control_successes + treatment_successes) / (control_n + treatment_n)
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / control_n + 1 / treatment_n))
    z = (p2 - p1) / se_pool if se_pool > 0 else 0.0
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    # The CI uses the UNPOOLED standard error, unlike the test statistic.
    # Subtle but correct: the test assumes the null is true (so pool), while
    # the interval estimates the actual difference (so do not pool).
    se_unpooled = np.sqrt(p1 * (1 - p1) / control_n + p2 * (1 - p2) / treatment_n)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    diff = p2 - p1

    return TestResult(
        metric=metric,
        test_name="two-proportion z-test",
        control_n=control_n,
        treatment_n=treatment_n,
        control_value=p1,
        treatment_value=p2,
        absolute_effect=diff,
        relative_effect=(p2 - p1) / p1 if p1 else 0.0,
        ci_lower=diff - z_crit * se_unpooled,
        ci_upper=diff + z_crit * se_unpooled,
        p_value=float(p_value),
        alpha=alpha,
        is_proportion=True,
    )


def compare_means(
    control: np.ndarray,
    treatment: np.ndarray,
    metric: str = "mean",
    alpha: float = 0.05,
) -> TestResult:
    """Welch's t-test for comparing two means.

    WHY WELCH'S AND NOT STUDENT'S (equal_var=False):
    Student's t-test assumes both groups have the same variance. Welch's does
    not. In practice the equal-variance assumption is rarely checked and often
    false — and Welch's costs almost nothing when variances ARE equal. There is
    essentially no scenario where Student's is the better default, which is why
    `equal_var=False` should be your habit.
    """
    t_stat, p_value = stats.ttest_ind(treatment, control, equal_var=False)

    m1, m2 = float(np.mean(control)), float(np.mean(treatment))
    se = np.sqrt(
        np.var(control, ddof=1) / len(control) + np.var(treatment, ddof=1) / len(treatment)
    )

    # Welch-Satterthwaite degrees of freedom — typically non-integer, which is
    # the giveaway that you are looking at a Welch test rather than a Student's.
    var_c = np.var(control, ddof=1) / len(control)
    var_t = np.var(treatment, ddof=1) / len(treatment)
    df = se**4 / (var_c**2 / (len(control) - 1) + var_t**2 / (len(treatment) - 1))
    t_crit = stats.t.ppf(1 - alpha / 2, df)
    diff = m2 - m1

    return TestResult(
        metric=metric,
        test_name="Welch's t-test",
        control_n=len(control),
        treatment_n=len(treatment),
        control_value=m1,
        treatment_value=m2,
        absolute_effect=diff,
        relative_effect=diff / m1 if m1 else 0.0,
        ci_lower=diff - t_crit * se,
        ci_upper=diff + t_crit * se,
        p_value=float(p_value),
        alpha=alpha,
    )


def compare_distributions(
    control: np.ndarray, treatment: np.ndarray, metric: str = "distribution"
) -> dict:
    """Mann-Whitney U: a non-parametric alternative to the t-test.

    WHEN TO REACH FOR THIS
    The t-test compares MEANS and relies on the Central Limit Theorem, which
    needs a reasonable sample size to rescue you from non-normal data. With
    heavily skewed data — and financial amounts are almost always log-normal —
    the mean is also a poor summary: it is dragged around by a few large values
    and describes almost nobody.

    Mann-Whitney asks a different question: "if I draw one value from each
    group, how often is the treatment value larger?" It compares the whole
    distribution rather than one moment of it.

    BEST PRACTICE: run both. If they agree, you have a robust finding. If they
    disagree, that disagreement is itself informative — usually it means a few
    extreme values are driving the mean, and you should look at the
    distribution before believing either result.
    """
    u_stat, p_value = stats.mannwhitneyu(treatment, control, alternative="two-sided")

    # Rank-biserial correlation: an effect size on a 0-1 scale, interpretable
    # as "probability a random treatment value exceeds a random control value".
    n1, n2 = len(control), len(treatment)
    prob_superior = u_stat / (n1 * n2)

    return {
        "metric": metric,
        "test": "Mann-Whitney U",
        "control_median": float(np.median(control)),
        "treatment_median": float(np.median(treatment)),
        "prob_treatment_larger": float(prob_superior),
        "p_value": float(p_value),
    }


def bonferroni_correct(p_values: list[float], alpha: float = 0.05) -> dict:
    """Adjust for testing many hypotheses at once.

    THE PROBLEM: each test at alpha=0.05 has a 5% false-positive rate. Run 20
    independent tests on data with no real effects anywhere and you expect one
    "significant" result purely by chance. Report only that one and you have
    manufactured a finding from noise — usually without meaning to.

    Bonferroni divides alpha by the number of tests. It is the most
    conservative correction and is often criticised for being too strict (it
    raises the false-NEGATIVE rate). Benjamini-Hochberg, which controls the
    false discovery rate, is usually a better choice for exploratory work.

    BUT THE REAL DEFENCE IS PRE-REGISTRATION. Corrections manage the damage of
    multiple testing; naming ONE primary metric in advance avoids it.
    """
    n = len(p_values)
    adjusted_alpha = alpha / n
    return {
        "n_tests": n,
        "original_alpha": alpha,
        "adjusted_alpha": adjusted_alpha,
        "significant_before": [p < alpha for p in p_values],
        "significant_after": [p < adjusted_alpha for p in p_values],
        "family_wise_error_if_uncorrected": 1 - (1 - alpha) ** n,
    }
