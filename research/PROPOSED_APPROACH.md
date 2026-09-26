# Quantum Rings runtime prediction: research and proposed approach

Prepared September 26, 2026. This document records the research and initial plan. The later implementation, measured validation, and reproduction commands are in [IMPLEMENTATION.md](IMPLEMENTATION.md). No quantum simulations were run.

## Recommendation

Build a fast, semantic OpenQASM feature extractor and a small gradient-boosted tree model predicting log runtime. Extract two parallel views: the raw circuit and a bounded, locally simplified circuit that approximates some of the transformations a simulator might perform. The distinguishing features should describe how interactions accumulate through the circuit, how much of the gate sequence is Clifford or near identity, and where measurement and classical control occur. Validate on entirely unseen circuits and, more conservatively, unseen structural families. Use the supplied scorer for model selection.

There are only 532 independent circuit examples. A compact, explainable model with careful features and leakage-resistant evaluation is a better initial investment than a neural architecture. The competition assigns two-thirds of the final score to explanation, novelty, process, presentation, and handling difficulties: a reproducible ablation study is therefore a central deliverable.

## What the local challenge actually requires

The input is QASM text and one threshold in {16, 64, 512}. Output positive finite wall-clock seconds. Hardware, backend, precision, and shots are fixed. The parser and each prediction have separate 15-second limits. `run.py` decompresses before starting the parser timer, but decompressed text still creates a memory burden. Its timing limits produce warnings rather than forcibly interrupting slow calls; judging may impose penalties.

The actual local layout differs from the README examples: circuits are in `training_circuits/` and labels in the challenge root. From `quantathon-harness/`, use `--circuits ../training_circuits` and score with `--labels ../runtime-data.csv`.

The baseline is intentionally weak. Its register regex does not recognize `qubit[256] q;`, an OpenQASM 3 declaration present in a sampled circuit. Counting lines or gate-name occurrences also mistakes gate definitions, braces, and comments for executable work.

### Audit of all labels

Source: `runtime-data.csv`, reproduced by `audit_labels.py`, with results in `label_audit.json`.

| Threshold | Labeled runs | Successes | Timeouts | Median successful runtime |
|---|---:|---:|---:|---:|
| 16 | 532 | 523 | 9 | 1.53 s |
| 64 | 530 | 526 | 4 | 4.16 s |
| 512 | 435 | 415 | 20 | 91.75 s |

There are 1,497 rows, 532 circuits, 33 explicit timeouts, no duplicate circuit/threshold keys, no status/duration inconsistencies, and complete label-to-file coverage. All rows specify 1,024 shots, single precision, and `scarlet_quantum_rings`.

Important qualifications:

* **99 missing combinations:** 97 circuits lack threshold 512 and two lack threshold 64. Missing measurements are not timeout labels. Their selection mechanism is unknown; unpaired medians cannot isolate the causal effect of threshold.
* **Nonmonotonic runtimes:** among 523 circuits successful at both 16 and 64, 169 are faster at 64, and 121 are over 20% faster. Among 415 successful at both 64 and 512, 109 are faster at 512, and 87 are over 20% faster. Do not impose hard threshold monotonicity without further investigation. These observations do not establish why the reversals occur.
* **One cap contradiction:** `2afe3d2e.qasm` at threshold 64 is labeled success at 42,993.9847111 s, about 11.94 hours. Preserve the raw label; compare validation with and without this flagged observation and request organizer clarification before treating it as an error.
* **Metric contradiction:** the code implements `max(0, 1 - abs(log10(pred/actual))/2)`. A 10x error earns 0.5; a 100x error earns zero. The README's statement that 10x earns zero is wrong relative to the code. For timeout rows only, the scorer caps predictions at 14,400 s. Successful observations are not capped, including the anomalous 11.94-hour run.

These are documented facts from this checkout, not assumptions about a future revised dataset or scorer.

## Local reading and related literature

The local repository includes teaching material rather than a dedicated runtime-prediction paper. I read relevant sections of the editable TeX sources, avoiding a need to extract the corresponding PDFs:

* `../../bootcamp/day-4/src/notes.tex`, lines 259–274: product states, Schmidt decomposition, and entanglement. This motivates measuring interaction structure rather than relying on qubit count alone.
* The same file, lines 373–381: GHZ and exponential full-state-vector memory. Its 16 bytes per complex amplitude example is double precision; the challenge uses single precision, conventionally eight bytes per complex amplitude for a dense representation. Neither number establishes the memory representation of this backend.
* `../../bootcamp/day-7/src/notes.tex`, lines 63–68: product-state overlaps factor into independent per-qubit terms. Useful intuition for separability, not evidence that this challenge needs quantum machine learning.

Primary research and official documentation supply the additional basis:

| Source | Relevant result or idea | Proposed use and limitation |
|---|---|---|
| [Azizov, Vela-Tambo, and Guo, *Transpilation-Aware Runtime Prediction for Noisy Quantum Circuit Simulation* (submitted 11 September 2026)](https://arxiv.org/html/2609.12980v1) | Compare source-circuit graphs, source graphs with post-transpilation features, and transpiled graphs for predicting Qiskit Aer runtime. Post-transpilation representation performs best overall; conventional regressors remain competitive in selected settings. | Make raw-versus-simplified feature ablation a priority. Do not transfer their Qiskit Aer performance numbers or assume their backend compilation applies here. |
| [Vidal, Efficient classical simulation of slightly entangled quantum computations (2003)](https://arxiv.org/abs/quant-ph/0301063) | Restricted entanglement permits efficient classical simulation. | Track cut-crossing and interaction-growth proxies. Gate counts do not measure Schmidt rank or entanglement entropy. |
| [Markov and Shi, Simulating quantum computation by contracting tensor networks (2008; preprint 2005)](https://arxiv.org/abs/quant-ph/0511069) | Simulation bounds depend exponentially on an appropriate circuit graph's treewidth. | Add inexpensive separator/frontier proxies. Treewidth of the static qubit interaction graph is not automatically the theorem's circuit/tensor-network width. |
| [Aaronson and Gottesman, Improved Simulation of Stabilizer Circuits (2004)](https://arxiv.org/abs/quant-ph/0406196) | Stabilizer circuits admit efficient classical simulation. | Count Clifford and non-Clifford operations, resolving rotation angles. This does not prove Quantum Rings automatically dispatches to a stabilizer algorithm. |
| [SupermarQ, Tomesh et al. (2022)](https://arxiv.org/abs/2202.11045) | Structural feature vectors characterize diverse workloads. | Starting vocabulary: communication, parallelism, liveness, two-qubit-gate ratio, and critical depth. These characterize workload structure, not calibrated simulator seconds. |
| [QASMBench, Li et al. (2020/2022)](https://arxiv.org/abs/2005.13018) | Width/depth plus density, lifespan, measurement density, and entanglement-variance metrics. | Add scheduling and per-qubit workload imbalance. Treat named entanglement metrics as structural proxies. |
| [MQT Predictor, official project](https://github.com/munich-quantum-toolkit/predictor) | Supervised learning predicts a suitable device without performing compilation. | Precedent for learning from circuit structure; its target and pretrained models are not this runtime task. |
| [Quantum Rings run settings](https://www.quantumrings.com/doc/usage/run_settings.html) and [0.12.2 release notes](https://www.quantumrings.com/news/quantum-rings-sdk-0122-release-notes) | Threshold controls a custom performance mode; tensor sizes may grow and shrink during execution. | Supports learning threshold/structure interactions and temporal features. The reviewed documentation does not establish `threshold = MPS bond dimension`, nor identify the exact challenge SDK version. |
| [OpenQASM specification: types](https://openqasm.com/language/types.html) | QASM 3 register declaration syntax differs from QASM 2. | Implement both syntaxes and semantic handling of measurement/control. |

The mechanistic feature suggestions below are hypotheses derived from these sources. None of the papers establishes their predictive accuracy on this dataset.

### What the September 11 paper changes

This is the closest located study to the task and should be discussed prominently in the presentation. Its experiments use 1,402 unique circuits across 22 families, two Qiskit fake-backend configurations, four transpiler optimization levels, and Qiskit Aer with device noise. The authors create 41 source-level global features and 13 additional post-transpilation features, then compare graph models with linear, ridge, SVR, random forest, and XGBoost regressors. Post-transpilation graph models lead their overall comparisons, but the backend-specific results sometimes favor ordinary regressors. Their evaluation uses RMSE and R² on runs that completed within a 900-second limit; this challenge uses a logarithmic score and deliberately includes 14,400-second timeouts. [Paper: methods and evaluation](https://arxiv.org/html/2609.12980v1#S4)

The useful transferable idea is that source QASM and the operations actually processed can differ. Here we do not know the Quantum Rings optimizer, native decomposition, or timing boundary. Qiskit transpiling to a fake device could produce misleading structure and may exceed the 15-second feature limit on very large files. Therefore, compute a **lightweight canonicalized view** with exactly specified, safe rewrites, such as folding adjacent single-qubit rotations when they share an axis, recognizing identity angles, and eliminating immediate inverse pairs when no intervening dependency exists. Retain raw counts because preprocessing may itself contribute to measured wall time. Compare raw-only, canonicalized-only, and combined features by grouped validation; discard the second view if it does not improve the official score or parser latency.

The paper's graph-construction cost is also instructive: its transpiled graphs average 34.9 times the source graph size and mean construction time about 8 seconds, though the median is much smaller. Its transpiled-graph dataset also excludes circuits with more than 300,000 graph nodes. A 15-second per-circuit limit and inputs as large as roughly 200 MB make full graph construction a risky first implementation here. Temporal summary features retain some sequence information without materializing a node for every gate. [Paper: dataset and inference cost](https://arxiv.org/html/2609.12980v1#S5)

## Proposed implementation

### 1. A bounded, semantic parser

Use a single-pass tokenizer/statement scanner with bounded auxiliary memory. Avoid constructing a full SDK circuit object or multiple full-file line lists: the challenge advertises files around 200 MB.

Support QASM 2/3 register declarations, multiple registers, gate parameters, custom gate definitions and calls, broadcast operands, measurement syntax, resets, barriers, and classical conditions. Summarize custom gates and bounded loops without eagerly expanding massive sequences. Definitions contribute only when called. If control flow cannot be resolved statically, emit conservative structural summaries and an uncertainty feature rather than silently assuming one execution.

Sampled inputs already include QASM 3 mid-circuit measurements, conditional phase operations, zero-angle gates, empty conditional blocks, and many numeric rotations. Distinguish syntactic work from effective quantum work: even a removable identity may cost the backend parsing/optimization time. Retain both raw and simplified counts.

Use a restricted arithmetic expression evaluator for angle constants (`pi`, numbers, arithmetic), with caching for repeated expressions. Do not evaluate arbitrary QASM text as Python. A single-qubit `rx`, `ry`, or `rz` at an integer multiple of pi/2 is Clifford; use gate-specific rules rather than applying this to all controlled gates. From the parsed stream, produce both raw and conservatively canonicalized operation summaries. Count discarded/folded operations as features; the actual backend may pay for or simplify them differently.

Ignore filenames, comments identifying origins, paths, compression sizes, and source labels as model features. Circuit names may join labels and group validation splits, never become predictors. Unknown syntax must trigger a recorded fallback and valid prediction rather than a missing submission row.

### 2. Features in order of priority

**First tier — size and workload:** declared/active qubits; operation counts by arity and gate type; log counts; one- and two-qubit depth; measurement/reset/control counts; per-qubit operation distribution; depth/width and two-qubit ratios. Compute these for raw and canonicalized views, plus their differences. Obtain ASAP depth by updating the most recent occupied layer on each operand, while respecting barriers and classical dependencies where modeled.

**Second tier — algebraic structure:** Clifford fraction; non-Clifford count and approximate depth; diagonal-gate fraction; zero/near-zero rotations; rotation-angle summaries; repeated local gate blocks; easily detected adjacent inverse pairs. Keep approximate-near-Clifford indicators separate from exact Clifford classification. Small angles may matter to approximate simulation, but do not prove the simulator discards them.

**Third tier — connectivity:** weighted interaction graph, unique pair count, degree distribution, connected-component sizes, edge reuse, interaction distance, and inexpensive cut summaries under original and a deterministic graph-derived qubit ordering. Order-dependent features are candidates, not physical invariants. Measure whether relabeling affects the model, and prefer robust summaries where practical.

**Novelty candidate — temporal interaction pressure:** divide the gate stream into a small fixed number of windows. Track first appearance and reuse of interaction edges, growth of connected components, cut-crossing operation counts, and concurrent live-qubit/frontier summaries. Record peak, average, and cumulative pressure. Compare circuits with the same totals but different temporal structure.

A cut-crossing counter can use range updates along a fixed qubit order instead of touching every cut for every gate. This provides inexpensive aggregate cut counts; exact per-time cut profiles require additional work and must be budgeted separately. Do not call any of these values exact bond dimension, contraction cost, or entanglement. Static union-find components also cannot detect later disentanglement.

Include threshold both as a category and `log2(threshold)`; evaluate whether explicit interactions with depth, non-Clifford count, and temporal pressure improve results. The model should learn the backend's response rather than hard-code cubic or monotonic scaling.

### 3. Model and scoring objective

Start with training-fold per-threshold log-median baselines, then a regularized linear model and a shallow gradient-boosted tree model (for example CatBoost or a suitable scikit-learn implementation, subject to competition library rules). Use roughly 40–80 interpretable features before expanding the feature set. A second tree model can be blended in log space only if it improves held-out performance consistently.

For the initial regressor set `y = log10(duration)` for successful runs and `y = log10(14400)` for timeouts. Log absolute error is a convenient training surrogate. It is not exactly the official objective: official loss saturates at a two-decade error and is one-sided above the cap on timeout rows. Tune and select on the exact scorer, not training RMSE or runtime R-squared. Do not automatically apply `log1p`: it changes how sub-second examples are weighted.

If the simple model misses many long runs, compare a separate timeout classifier plus successful-runtime regressor, or a censored-time objective. There are only 33 timeout rows, so classifier calibration will be fragile. Calibrate any decision to emit 14,400 s on out-of-fold scores; a default probability threshold of 0.5 is not guaranteed optimal. Avoid extrapolating unknown timeout durations as if they were observed.

Do not blindly clip every output to 14,400 while a successful above-cap label remains in the official data. Keep positive finite outputs, investigate that observation, and test sensitivity. An elaborate survival model is optional because the challenge rewards capped outcomes rather than identifying the unknown tail.

### 4. Validation that reflects hidden circuits

1. Five-fold grouped validation: all threshold rows for a circuit belong to one fold. Any scaling, feature selection, fitting, and calibration use training folds only.
2. Group exact duplicates and near-identical structural variants together, using circuit content rather than filename metadata. Report this as a more conservative validation setting.
3. Stress-test structural-cluster holdouts and large-circuit extrapolation. Structural clusters are not claimed to be known benchmark source families. No source recovery from scrambled filenames.
4. Report official mean score, log absolute error, fractions within 2x/10x, threshold-specific performance, timeout errors, and the worst examples. Bootstrap by circuit rather than by row when estimating uncertainty.
5. Ablate: size only → raw gate and scheduling features → canonicalized features → connectivity → temporal pressure → optional timeout handling. Each addition must earn its complexity on held-out data and fit the timing budget. The raw/canonicalized comparison directly tests the September 11 paper's central idea in the available setting.

The same training circuit at another threshold must not serve as a validation shortcut. At inference all three thresholds may be predicted jointly from structure, but no measured runtime for the hidden circuit is available.

### 5. Runtime and submission checks

Benchmark parsing on the largest decompressed circuits and each observed syntax class early. Aim for substantial margin under 15 seconds; model inference should normally be milliseconds. Record maximum and tail parser latency, not only the average. Until measured, there is no claim this parser design meets the cap.

Cache extracted features for training experiments. On fresh holdout inference, compute them from QASM. Verify one finite positive prediction per requested circuit/threshold and keep a cheap robust fallback. Do not change `run.py` or `score.py` to make a result look better. Package loading relative to `model.py`, with approved dependencies/artifacts or embedded compact parameters as permitted by organizers.

## Suggested 24-hour allocation

| Time budget | Outcome |
|---|---|
| 0–3 hours | Clarify scoring/cap anomalies, inventory syntax, parser and label audit |
| 3–7 hours | Core features, grouped folds, baseline and first tree model |
| 7–12 hours | Canonicalized and algebraic/connectivity features, largest-file timing |
| 12–17 hours | Temporal features, ablations, structural holdout checks |
| 17–20 hours | Error analysis, calibration, select and freeze model |
| 20–24 hours | Holdout predictions, coverage verification, write-up and presentation |

Organizer questions worth resolving: Is the supplied scorer authoritative? Is the 42,994-second success valid? Why are thresholds missing? What SDK version and exact timing boundary generated labels? Which libraries and artifact packaging are allowed? These questions do not prevent parser development or baseline validation against the supplied files.

The presentation should explain one concrete result from each stage: a parser pitfall, an observed data contradiction, a physically motivated feature, a measured ablation improvement or failure, and an honest generalization limit. Proposed improvements must remain labeled as hypotheses until evaluated.
