# ADR 0011: Expanding-window validation for all time-series models

## Status
Accepted — 2026-07

## Context
Standard k-fold cross-validation shuffles rows into folds. On time-ordered data
that places future observations in the training set for past test points. The
model is shown the answer and asked to predict it.

Nothing errors. The metrics are real numbers. They simply measure a task nobody
will ever ask the model to perform.

## Decision
All time-series evaluation uses expanding-window (walk-forward) validation.
Training data always precedes test data. Random splitting is prohibited, and
`demonstrate_leakage()` exists specifically to show why.

## Rationale
Measured on our own deposit series: a random split reported 1.23% MAPE while
the chronological split reported 3.26% — the honest error is 2.6x worse. Ship
the leaky version and production error is several times what was promised.

An important nuance we discovered while building this: the size of the
distortion depends on the MODEL. A global linear trend is nearly immune,
because a straight line fitted to any subset of trending data is roughly the
same line. Models that learn LOCAL structure — trees, gradient boosting, neural
networks, nearest-neighbour methods — are devastated, because a random split
lets them look up neighbouring points on both sides of each target.

Since almost every model anyone actually deploys falls into the second
category, the rule is absolute rather than conditional.

## Consequences
+ Reported accuracy reflects how the model will really be used.
+ Aggregating across folds means the metric is not one lucky window.
- Fewer effective training examples per fold than random splitting would give,
  and evaluation is slower (many fits rather than k).
- Early folds train on little data and may look poor. That is informative, not
  a flaw: it tells you how much history the model actually needs.

## Related
Sprint 8's credit model requires OUT-OF-TIME validation for the same underlying
reason, with an extra one on top: credit populations drift, so a model
validated on a shuffled sample of its own era flatters itself twice over.
