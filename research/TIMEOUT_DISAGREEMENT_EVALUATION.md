# Timeout-router disagreement probe (rejected)

The v6 model's structural-stress fold had a conspicuous failure: 36
threshold-64 circuits above 1 MB were routed to the 14,400-second cap while
the continuous regressors predicted under 100 seconds. All 36 were successful
runs, and all belonged to structural fold 2. Replacing the cap for those rows
with a log-space blend of 80% continuous prediction and 20% cap raised the
structural-stress score from **0.75537 to 0.76654**. It changed **zero** rows
on the committed distribution-matched folds, so that split could not validate
the rule.

I then refitted the v6 model on three additional five-fold assignments,
grouping all thresholds of each circuit together. The candidate rule was
fixed before these refits. Results:

| Fold seed | Baseline score | Soft-route score | Changed rows | Real timeouts among changed |
|---|---:|---:|---:|---:|
| 81 | 0.926520 | 0.926520 | 0 | 0 |
| 99 | 0.919700 | 0.919700 | 0 | 0 |
| 113 | 0.924461 | **0.924073** | 2 | 1 |

The third assignment exposes the downside: the soft route demotes a real
timeout and lowers score by 0.00039. The fixed structural-fold gain therefore
does not justify changing the deployed timeout router for a
similar-distribution holdout. The v6 artifact is unchanged.

Reproduce with:

```bash
uv run --locked python research/probe_timeout_disagreement.py
```

The machine-readable scores are in `timeout_disagreement_probe.json`. This is
a model-selection diagnostic on the released labeled circuits, not an
independent hidden-holdout estimate.
