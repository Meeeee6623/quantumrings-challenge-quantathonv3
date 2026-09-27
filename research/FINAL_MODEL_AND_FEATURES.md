# Final feature extractor and runtime predictor

Historical v8 model description. The current 120-input production v9 schema
and refit command are in the [README](../README.md).

The v8 submission was `categorical_setting_threshold_experts_v8`, SHA-256 `3a53a37acd93e3e84a39cf73e252b00b8f8dc448ed937600f6f246855f79819b`. It predicted positive wall-clock **seconds** from QASM text and a simulator setting (16, 64, or 512). It did not use scrambled filenames, source collections, inferred external family labels, or training-error metadata at inference. Its exact **241 fitted feature names** are preserved in [the v8 validation schema](categorical_model_validation.json). The current [feature catalog](final_feature_catalog.csv) describes production v9. The [one-by-one historical audit](feature_audit/FINAL_FEATURE_DECISIONS.md) gives the v7-to-v8 keep/drop decisions, fitted importance, and threshold-grouped runtime plots.

## What the extractor measures

The harness makes **one `featurize` call per circuit**, then three `predict` calls. The primary bounded scanner in [`model.py`](../quantathon-harness/model.py) recognizes common QASM 2 and 3 declarations, operations, parameters, custom gate definitions and calls, measurements, resets, barriers, and conditionals. A second independent scanner in [`extended_features.py`](../quantathon-harness/extended_features.py) measures additional DAG, interaction, angle, and source-shape properties. The optional third pass is the angle-aware [`chi_walk.py`](../quantathon-harness/chi_walk.py). These are static source analyses, not full QASM execution or the simulator's actual compilation trace.

| Fitted feature view | Distinct selected columns | What is represented |
|---|---:|---|
| Primary QASM fields | 40 | Width and active qubits; gate/arity counts; depth and local simplification; Clifford/diagonal mix; interactions, reordered cuts, time windows, liveness, heuristic family fingerprints, χ bounds, diversity, and dense-vector reference sizes. |
| `log1p(max(0,x))` primary transforms | 79 | Compressed versions of selected counts and nonnegative structural quantities. Redundant raw/log pairs were screened. |
| Simulator setting | 3 | One-hot category fields `setting_16`, `setting_64`, and `setting_512`; they also route specialists and timeout classifiers. |
| Angle-aware χ walk | 10 | Threshold-specific work proxy, potentially entangling fraction, peak bond-rank upper bound, saturation timing/fraction, capped links, availability, and numeric rotations near 0 or π. |
| Secondary QASM scanner | 109 | Angle distributions, custom-gate expansion proxies, source shape, gate mix, interaction-graph metrics, cut crossings, DAG/layer width, min-degree/min-fill graph-width proxies, and SupermarQ-style summaries. |
| **Union referenced by fitted models** | **241** | 120 columns feed the global regressor; each threshold expert uses 200 and each timeout classifier uses 80. Membership overlaps. |

The v8 names are in [the validation schema](categorical_model_validation.json), rather than suggesting that every emitted parser field entered every fitted model. Its selected χ-walk columns were `chi_walk_cost_overhead`, `chi_walk_entangling_frac`, `chi_walk_max_logchi`, `chi_walk_sat_frac`, `chi_walk_sat_start`, `chi_walk_links_at_cap`, `chi_walk_available`, `chi_walk_rot_near_zero`, `chi_walk_rot_near_frac`, and `chi_walk_rot_mixing_mean`. The near-π count and extrapolation flag were still extracted but no v8 tree selected them; the near-basis **fraction** remained an input and a special-rule guard.

The χ geometry uses, for cut `k`, a **potential rank bound** of the form `log2 χ_upper(k) = min(k, n−k, sum of crossing-gate log-rank allowances)`. It is not a measured Schmidt rank. A separate diversity proxy combines gate, adjacent-gate, angle, interacting-pair, and temporal-window entropies; its heuristic effective-rank feature approaches the upper bound as diversity approaches one. The χ walk tracks possible bond growth over the gate stream, with computational-basis-state filtering. A numeric `RX`, `RY`, `U`, or `U3` rotation within **0.3 radians of an integer multiple of π** preserves the walk's basis-state flag; the gate is still charged as work. The walk computes summaries under caps associated with 16, 64, and 512, but the challenge does **not** establish that its threshold equals the simulator's literal MPS bond dimension. See the [χ/randomness ablation](CHI_RANDOMNESS_EVALUATION.md), [walk ablation](CHI_WALK_EVALUATION.md), and [rotation experiment](ROTATION_FILTER_EVALUATION.md).

The selected `chi_walk_cost_overhead` is `log10(1 + Σ_j t1[j](10 + χ_j²) + Σ_j t2[j](100 + χ_j^2.5))`, where `χ_j=2^j`, `t1` counts single-qubit gates at that local rank level, and `t2` weights two-qubit gates by crossed-link distance. It is a learned-model input, **not** a runtime equation for Quantum Rings.

The primary graph fields compare register order with reverse Cuthill–McKee order; the secondary fields add weighted interactions, cut crossings, and approximate treewidth. These are **qubit interaction-graph proxies**, not exact tensor-network contraction width. Soft QFT, QAOA, arithmetic, Grover, GHZ, graph-state, random-grid, and variational fingerprints are numeric QASM pattern scores, not asserted algorithm identities. The dense complex64 state-vector byte estimates are reference scales, not claims about Quantum Rings allocation. The feature choices are motivated by entanglement-sensitive simulation [Vidal](https://arxiv.org/abs/quant-ph/0301063), contraction-width analysis [Markov and Shi](https://arxiv.org/abs/quant-ph/0511069), structural benchmark vectors [SupermarQ](https://arxiv.org/abs/2202.11045) and [QASMBench](https://arxiv.org/abs/2005.13018), and the distinction between source and transformed circuits in the [11 September 2026 runtime study](https://arxiv.org/abs/2609.12980). Their usefulness **here** comes from the linked local ablations, not from transferring those papers' scores.

The extractor uses a coarse fast path above **40 million decoded QASM characters**. It preserves width, source work, selected whole-text gate counts, and fast secondary-scanner fields; it marks unavailable detailed topology and χ-walk fields rather than pretending to have computed them. Four training circuits take that path. On ordinary inputs, the walk has a 3-second budget and is shortened if earlier parsing has consumed most of a 14-second working budget. This is a deliberate accuracy/speed tradeoff to stay within the harness's separate 15-second parse and prediction limits. The [large-circuit audit](BIG_CIRCUITS_EVALUATION.md) and [parser timing report](PARSER_OPTIMIZATION.md) describe what is lost.

## Chosen model

All 1,497 released labeled runs were used for the fitted artifact after circuit-grouped validation. The target is `log10(seconds)`; timeouts use the observed four-hour cap of 14,400 seconds as the target. The global regressor is an `ExtraTreesRegressor` with 240 trees, minimum leaf size 2, and `max_features=0.9`. It receives three one-hot setting indicators, never an ordinal threshold or its logarithm. Each known threshold has a 600-tree, leaf-size-1 runtime expert. The prediction before rules is the **equal-weight average of global and specialist log10 outputs**, equivalent to the geometric mean of their runtime estimates. The per-threshold `ExtraTreesClassifier` has 300 balanced trees, minimum leaf size 2, and `max_features=0.8`; probability at least **0.35** emits the timeout cap. For an unknown threshold, prediction falls back to the global model with all three setting indicators zero; that path is uncalibrated because the challenge guarantees only the three known settings.

The earlier 480-column union yielded 325 v7 selected inputs. The v8 audit removed 84 of those inputs, including both ordinal setting encodings, and added three one-hot categories. Its **244 candidate columns** were then ranked inside each training fold and separately for the global model, each runtime expert, and each timeout classifier. Final fits use 120, 200, and 80 columns respectively, yielding 241 distinct columns across components. Three audit-kept candidates were not selected by any final component. The selection uses impurity importance and threshold-grouped plots; correlated features can divide importance and flat marginal curves can hide interactions. The [final per-feature audit](feature_audit/FINAL_FEATURE_DECISIONS.md) and [earlier pruning report](FEATURE_PRUNING_REPORT.md) give the evidence and limitations. The v8 screening used all released labels before the fixed-fold ablation, so that score improvement may be optimistic.

| Final fixed-fold stage | Distribution-matched score ↑ | Structural-cluster stress score ↑ |
|---|---:|---:|
| Global regressor only | 0.915347 | 0.739767 |
| Global + threshold experts + timeout router | 0.922546 | 0.749836 |
| Plus reset-heavy search floor | 0.924362 | 0.751791 |
| Plus very-large-work floor | 0.924668 | 0.755010 |
| Plus guarded count-signature analogue | 0.925473 | 0.755010 |
| Plus high-threshold near-basis calibration: previous v7 | 0.926153 | 0.755369 |
| Categorical setting + feature audit pruning: **selected v8** | **0.926914** | **0.760192** |

![Fold-local feature-selection tradeoff](feature_pruning.png)

These are out-of-fold results on released labels, **not hidden-holdout accuracy**. Matched folds group every threshold of a circuit and stratify label-free structural clusters; structural stress folds withhold whole clusters. The two scores ask different questions. Matched v8 median factor error is 1.062, its 95th-percentile factor error is 5.06, and it has 25 tenfold misses (v7 had 28). The matched timeout subscore declined from 0.9639 to 0.9426; structural timeout performance was essentially unchanged. The second scanner and threshold experts were retained because the merged model improved matched validation over the earlier single-parser global model. The separate [family-aware neural runtime experiment](PAPER_RUNTIME_REPLICATION.md) scored 0.900924 matched and 0.748393 structural and showed no stable family-conditioning gain across seeds, so it did not replace the tree model.

## Special prediction rules and reasons

| Rule | Exact trigger/action | Why retained, and limitation |
|---|---|---|
| Timeout routing | At a known threshold, classifier probability `≥0.35` returns **14,400 s** immediately. | The challenge scores timeouts at this cap; the threshold-specific router improved the merged predictor's timeout performance. Only 33 training labels are timeouts, so probability calibration is fragile. An attempted softening for classifier/regressor disagreement was rejected on alternate folds ([probe](TIMEOUT_DISAGREEMENT_EVALUATION.md)). |
| Reset-heavy search floor | `resets≥10`, `multi_q≥10`, and `fingerprint_grover≥0.9`: find the nearest same-threshold training reference in log operation count, scale its capped runtime by the operation ratio, cap at 14,400 s, and use this **only as a lower bound**. | Seven released circuits have this signature; earlier Grover-like underpredictions were extreme. Fixed-fold banks contain training folds only. This is a narrow observed-regime guard, not a proof of algorithm identity ([evaluation](EDGE_CASE_UPDATE.md)). |
| Very large work floor | `effective_ops≥500,000` ⇒ at least **10 s**; `≥1,000,000` ⇒ at least **100 s**. | The smallest observed runtimes in those regimes were 16.12 and 188.08 s. This prevents severe tree extrapolation without making QASM byte size the runtime target ([evaluation](EDGE_CASE_UPDATE.md)). |
| Exact count-signature analogue | For the same threshold and exact `(n_qubits, ops, two_q, multi_q)`, use no analogue with 0–1 references; with 2, take a 50/50 **log-space** blend; with ≥3, use the median training log-runtime directly. | Repeated structural templates are common. Single-reference copying failed; support-aware weighting improved four grouped assignments. The bank stores counts and runtimes, never filenames ([v7 test](TEMPLATE_WEIGHT_EVALUATION.md)). |
| Near-basis correction | At threshold **512**, if `χ_walk_rot_near_frac≥0.9`, `ops≥1,000`, and `1≤predicted_s<14,400`, multiply by **0.5**. | Corrected a small set of overpredicted near-0/π circuits; the analogous threshold-16 correction hurt. It is guarded to preserve subsecond and explicit timeout predictions ([probe](EDGE_CASE_UPDATE.md)). |
| Output domain | Continuous log prediction is bounded to **10⁻⁴–10⁷ s** before floors. Successful runs above 14,400 s are **not** globally clamped. | The scorer caps predictions only when the *actual* row timed out; one successful training run lasted 42,993.98 s. Threshold response is not forced monotone: 270 of 532 circuits show measured inversions across available settings, and isotonic projection reduced matched score ([headroom analysis](REMAINING_HEADROOM.md)). |

`predict_interval` also exposes an optional multiplicative risk band using a matched-fold absolute-log-residual radius of 0.47789 (about **3.01×** in either direction). It is an inspection aid, not an official submission field or a distribution-shift guarantee.

## Verification and use

The supplied v8 harness produced **1,596/1,596** positive predictions on the training circuit directory, covering all **1,497 labeled pairs**, with no 15-second violations; parse and prediction maxima were **13.5479 s** and **0.1530 s**. All 38 focused tests passed. The training-label score is intentionally excluded as a generalization claim.

```bash
uv sync --locked
uv run --locked python research/write_final_feature_catalog.py
uv run --locked python quantathon-harness/run.py --team 'Your Team' \
  --circuits /path/to/holdout --out submission.csv
uv run --locked python research/validate_submission.py \
  --circuits /path/to/holdout --submission submission.csv
```

The [holdout guide](HOLDOUT_READINESS.md) covers plain/compressed QASM and the geometry/nearest-training-circuit inspector. Exact v8 validation metrics and selected schemas are in [the model validation JSON](categorical_model_validation.json); the per-row grouped predictions are in [the OOF CSV](categorical_model_oof.csv). The v7 [validation](full_union_model_validation.json) and [OOF predictions](full_union_model_oof.csv) remain for comparison.
