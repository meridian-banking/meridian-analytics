# ADR 0010: Pre-register experiment plans as committed artifacts

## Status
Accepted — 2026-07

## Context
Analysing an experiment involves many choices: which metric decides the
outcome, which segments to examine, when to stop collecting, how to handle
outliers. Made AFTER seeing results, each choice is contaminated by knowing
which option produces the answer you want.

This contamination is usually unconscious. Competent, honest analysts p-hack
routinely without intending to, because "let's check the other metric" feels
like diligence rather than fishing.

## Decision
Every experiment gets a plan committed to git BEFORE data collection starts,
recording: hypothesis, ONE primary metric, secondary and guardrail metrics,
randomisation unit, MDE, alpha, power, required sample size, and an explicit
stopping rule.

## Rationale
The mechanism is not honour, it is EVIDENCE. The file's commit timestamp proves
the decisions preceded the data. Clinical trials have required this for decades
for exactly this reason.

Naming one primary metric also removes the multiple-testing problem at the
source rather than correcting for it afterwards. With twelve metrics at
alpha=0.05 you expect roughly one false positive by chance; corrections manage
that damage, pre-registration avoids it.

The stopping rule matters just as much. We measured it: peeking twelve times
takes the false-positive rate from 4.8% to ~20%.

## Consequences
+ The analysis is decided when it can be decided impartially.
+ Secondary metrics remain reportable, but explicitly as EXPLORATORY — findings
  to test properly next time, not conclusions.
+ A reviewer can check the plan against the report.
- Less flexibility. If the primary metric turns out to be poorly chosen, you
  must say so openly rather than quietly switching.
- It requires doing power analysis up front, which some teams resist because it
  sometimes reveals the experiment is not feasible at all.
