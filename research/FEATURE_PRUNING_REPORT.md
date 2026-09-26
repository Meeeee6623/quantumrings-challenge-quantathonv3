# Feature audit and pruning (selected v6)

The selected runtime predictor now gives its global regressor **120** inputs,
each threshold-specific runtime regressor **200**, and each timeout classifier
**80**. The previous model used 243/480/480 respectively. Across all six
threshold models and the global model, the fitted artifact references **325
distinct columns**, down from 480. Both parsers still contribute useful
columns, so this change simplifies the learned model but does not eliminate a
QASM parsing pass.

![Feature pruning comparison](feature_pruning.png)

## Audit

The 1,497 labeled rows cover 532 circuits. Two columns were constant
(`unsupported_statements` and its log transform), and 22 further columns were
exact duplicates on the released circuits. Examples include duplicate qubit,
byte, threshold, measurement, and reset counts; four copies of the extended
DAG critical-path length; and equivalent interaction-density outputs. The
complete names and retained counterparts are in
`feature_pruning_validation.json`. We remove these without using runtime
labels.

There are also 73 pairs with absolute Pearson correlation above 0.9999 after
excluding missing/constant columns. We did **not** prune by correlation alone:
rare gate counts can coincide on this dataset while representing distinct
operations on unseen circuits, and raw/log transforms can give tree models
different useful split thresholds. Model-specific feature selection handles
the remaining redundancy.

The prior fitted model's mean impurity importance was concentrated but not
uniformly so. Its global top 120 columns carried about 94.6% of importance;
the specialists' top 200 carried about 93.4%; the timeout classifiers' top 80
carried about 95.4%. These percentages are a screening diagnostic, **not** an
independent validation metric; correlated features share importance.

## Selection and validation

For each held-out circuit fold, ranking estimators see only that fold's
training rows. Separate rankings are learned for the global regressor, each
threshold regressor, and each threshold timeout classifier. Fresh estimators
are then trained on the selected columns. All thresholds of a circuit remain
in the same fold. The existing postprocessing (runtime floors, guarded
template analogue, near-basis correction) is applied identically to every
candidate.

| Inputs (global / expert / classifier) | Matched score | Structural-stress score | Matched >10× misses | Structural >10× misses |
|---|---:|---:|---:|---:|
| Full 243 / 480 / 480 | 0.92547 | 0.75030 | 28 | 228 |
| Constant/exact-duplicate removal, 241 / 456 / 456 | 0.92521 | 0.74992 | 29 | 225 |
| **Selected 120 / 200 / 80** | **0.92578** | **0.75537** | **28** | **221** |
| Aggressive 80 / 140 / 60 | 0.92587 | 0.74846 | 28 | 222 |

The selected cut is +0.00031 matched and +0.00507 structural stress against
the full v5 model. The matched circuit-bootstrap 95% interval for this
difference is [-0.00057, +0.00113]; the 12-cluster structural-stress interval
is [+0.00054, +0.01018]. The small matched gain is not distinguishable from
noise. We selected the moderate cut because it reduced model width and
improved the harder structural split; the more aggressive cut reduced that
split's score. These folds have been used repeatedly for development, so
neither interval estimates hidden-holdout performance without selection bias.

The selected union still includes 11 chi-walk features, 19 geometry features
with `graph_` in their names, 14 randomness features, 16 algorithm
fingerprints, 18 extended angle features, 8 extended DAG features, and 20
extended interaction features. The optional sequence-level `motif_` extractor
was already disabled in the submission parser; QAOA and other algorithm
fingerprints remain in the model. Near-zero rotation count and the
near-basis fraction remain selected; the near-π count is still extracted but
was not selected by a final estimator. The near-basis postprocessing uses the
fraction separately from the learned models.

The serialized artifact is 47.10 MB versus 47.30 MB before pruning. Tree
nodes dominate its size, so the large input-width reduction does not imply a
large file-size reduction. Exact duplicates are established on released
circuits; some could diverge on future circuits. The two parsers and the
chi walk remain necessary for selected inputs and for postprocessing.

## End-to-end check

The 31 unit tests pass. Running the supplied harness on all 532 training
circuits produced all 1,596 requested predictions, each positive and finite,
with no 15-second cap violations. Parser median/p95/max were
0.0935/3.2270/14.0297 seconds; prediction median/p95/max were
0.0270/0.0281/0.1365 seconds. The fitted training-label score was 98.79%,
which checks artifact integration but is **not** a generalization estimate.
The complete output is `training_submission_pruned.csv`.

## Reproduce

```bash
uv sync --locked --extra report
uv run --locked python research/probe_feature_pruning.py
uv run --locked python research/train_full_union_model.py
uv run --locked python research/plot_feature_pruning.py
uv run --locked python -m unittest discover -s research -p 'test_*.py' -q
```

The probe writes only `feature_pruning_validation.json`, not the deployed
artifact. The trainer writes the selected v6 artifact, its fixed-fold OOF CSV,
and `full_union_model_validation.json` with the exact selected column names.
