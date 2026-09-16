"""Experiment design: the decisions you must make BEFORE seeing any data.

THE CENTRAL IDEA OF THIS MODULE
Everything here happens before the experiment runs. That ordering is not
bureaucratic — it is the only thing standing between "we ran a test" and "we
convinced ourselves of whatever we already believed".

Once you have seen the results, you cannot un-see them. Every subsequent
decision — which metric to report, which segment to highlight, when to stop
collecting — becomes contaminated by knowing which choice produces the answer
you want. That contamination is usually unconscious, which is exactly why it
needs a procedural fix rather than good intentions.

THE THREE THINGS THIS MODULE ENFORCES

1. POWER ANALYSIS. How many customers do we need for this test to be capable
   of detecting the effect we care about? Run a test that is too small and a
   real effect will be invisible — you will conclude "no difference" from a
   test that never could have found one. That is not a null result; it is a
   non-result, and reporting it as evidence of no effect is wrong.

2. PRE-REGISTRATION. Which metric decides this, stated in writing, before the
   data arrives. Without it you will (honestly, unconsciously) gravitate to
   whichever of your twelve metrics happened to look good. With twelve metrics
   at the 5% level, you expect roughly one false positive by chance alone.

3. A STOPPING RULE. When we stop collecting. Peeking repeatedly and stopping
   when significance appears turns a 5% false-positive rate into something far
   worse — around 30% with daily checks over a few weeks, because you get a
   fresh chance to cross the threshold every single time you look.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
from scipy import stats


@dataclass
class PowerResult:
    """Required sample size for a two-sample test.

    `is_proportion` controls formatting only: a churn RATE reads naturally as a
    percentage, while an average BALANCE does not. Getting this wrong produces
    output like "a 15000% change from a 420000% baseline", which is the kind of
    nonsense that makes a reader distrust everything else on the page.
    """

    baseline_rate: float
    minimum_detectable_effect: float
    alpha: float
    power: float
    n_per_group: int
    n_total: int
    is_proportion: bool = True

    def summary(self) -> str:
        if self.is_proportion:
            effect = f"{self.minimum_detectable_effect:.2%} absolute"
            baseline = f"{self.baseline_rate:.1%}"
        else:
            effect = f"{self.minimum_detectable_effect:,.2f}"
            baseline = f"{self.baseline_rate:,.2f}"
        return (
            f"To detect a change of {effect} from a baseline of {baseline}\n"
            f"  at alpha={self.alpha} and power={self.power:.0%}:\n"
            f"  {self.n_per_group:,} per group ({self.n_total:,} total)"
        )


def required_sample_size_proportions(
    baseline_rate: float,
    minimum_detectable_effect: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> PowerResult:
    """Sample size for comparing two proportions (e.g. churn rate).

    THE FOUR QUANTITIES AND WHAT THEY MEAN, because interviewers ask:

      alpha (0.05) — the false positive rate you accept. The probability of
        declaring an effect when none exists. 5% is convention, not law; for an
        irreversible decision you might want 1%.

      power (0.80) — the probability of DETECTING an effect that is really
        there. 80% is the usual floor, and note what it implies: even a
        correctly-powered test misses a real effect one time in five.

      minimum detectable effect (MDE) — the smallest change you care about.
        This is a BUSINESS decision, not a statistical one. A 0.1% churn
        reduction may be real and still not worth the engineering effort.

      baseline rate — where you are starting from. Detecting a change from a
        50% baseline needs far fewer samples than from a 2% baseline, because
        rare events carry proportionally more noise.

    THE KEY RELATIONSHIP: n scales with 1/MDE². Halving the effect you want to
    detect QUADRUPLES the sample needed. This is why "let's also check if it
    helps a tiny bit" is so expensive, and why tiny effects usually cannot be
    measured at all at realistic sample sizes.
    """
    p1 = baseline_rate
    p2 = baseline_rate + minimum_detectable_effect
    p_pooled = (p1 + p2) / 2

    # Two-sided test, so alpha is split across both tails.
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_power = stats.norm.ppf(power)

    numerator = (
        z_alpha * np.sqrt(2 * p_pooled * (1 - p_pooled))
        + z_power * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    ) ** 2
    n_per_group = int(np.ceil(numerator / (minimum_detectable_effect**2)))

    return PowerResult(
        baseline_rate=baseline_rate,
        minimum_detectable_effect=minimum_detectable_effect,
        alpha=alpha,
        power=power,
        n_per_group=n_per_group,
        n_total=n_per_group * 2,
    )


def required_sample_size_means(
    baseline_mean: float,
    baseline_std: float,
    minimum_detectable_effect: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> PowerResult:
    """Sample size for comparing two means (e.g. average balance).

    Driven by EFFECT SIZE = MDE / standard deviation, not by the raw MDE. A
    $50 change is enormous if balances vary by $100 and invisible if they vary
    by $10,000. This is why you cannot answer "how many samples do I need?"
    without knowing the variance.
    """
    effect_size = minimum_detectable_effect / baseline_std
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_power = stats.norm.ppf(power)
    n_per_group = int(np.ceil(2 * ((z_alpha + z_power) / effect_size) ** 2))

    return PowerResult(
        baseline_rate=baseline_mean,
        minimum_detectable_effect=minimum_detectable_effect,
        alpha=alpha,
        power=power,
        n_per_group=n_per_group,
        n_total=n_per_group * 2,
        is_proportion=False,
    )


@dataclass
class ExperimentPlan:
    """A pre-registered experiment plan.

    WHY WRITE THIS DOWN AND COMMIT IT TO GIT?
    Because the file's timestamp proves the decisions were made before the
    data existed. That is the entire mechanism: not honour, but evidence.
    Clinical trials have required this for decades, for exactly the reason it
    matters here — humans are extremely good at retrospectively justifying the
    analysis that produced the answer they wanted.
    """

    name: str
    hypothesis: str
    primary_metric: str
    secondary_metrics: list[str] = field(default_factory=list)
    randomization_unit: str = "customer"
    minimum_detectable_effect: float = 0.0
    alpha: float = 0.05
    power: float = 0.80
    required_n_per_group: int = 0
    planned_duration_days: int = 0
    stopping_rule: str = "Analyse ONCE at the planned end. No interim peeking."
    guardrail_metrics: list[str] = field(default_factory=list)
    registered_on: str = field(default_factory=lambda: date.today().isoformat())

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))
        return path

    @classmethod
    def load(cls, path: str | Path) -> ExperimentPlan:
        return cls(**json.loads(Path(path).read_text()))

    def summary(self) -> str:
        lines = [
            f"PRE-REGISTERED PLAN: {self.name}",
            f"  registered      {self.registered_on}",
            f"  hypothesis      {self.hypothesis}",
            f"  PRIMARY metric  {self.primary_metric}   <- this alone decides the outcome",
            f"  secondary       {', '.join(self.secondary_metrics) or '(none)'}",
            f"  guardrails      {', '.join(self.guardrail_metrics) or '(none)'}",
            f"  randomise by    {self.randomization_unit}",
            f"  MDE             {self.minimum_detectable_effect}",
            f"  alpha / power   {self.alpha} / {self.power:.0%}",
            f"  required n      {self.required_n_per_group:,} per group",
            f"  duration        {self.planned_duration_days} days",
            f"  stopping rule   {self.stopping_rule}",
        ]
        return "\n".join(lines)


def check_randomization_balance(
    control: np.ndarray, treatment: np.ndarray, name: str, alpha: float = 0.05
) -> dict:
    """Verify the two groups look alike on a pre-existing covariate.

    WHY THIS CHECK EXISTS: randomisation is supposed to balance everything,
    including things you did not measure. But randomisation can fail — a buggy
    assignment function, a hash that correlates with signup date, a filter
    applied after assignment. If the groups differ on a variable that was fixed
    BEFORE the experiment started, the randomisation is broken and no amount of
    clever analysis fixes it.

    NOTE THE INVERTED LOGIC: here a HIGH p-value is the good outcome. We WANT
    to fail to reject "these groups are the same". This is one of the few
    places in statistics where that is true, and it trips people up.
    """
    t_stat, p_value = stats.ttest_ind(control, treatment, equal_var=False)
    return {
        "covariate": name,
        "control_mean": float(np.mean(control)),
        "treatment_mean": float(np.mean(treatment)),
        "difference": float(np.mean(treatment) - np.mean(control)),
        "p_value": float(p_value),
        "balanced": bool(p_value > alpha),
    }
