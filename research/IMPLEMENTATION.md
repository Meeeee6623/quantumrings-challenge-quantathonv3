# Runtime predictor implementation

This implementation is based on the [Quantum Rings Quantathon V3 challenge repository](https://github.com/Quantum-Rings/quantumrings-challenge-quantathonv3). The training circuits, labels, and harness scripts in that repository were verified to match the earlier local challenge copy.

The submission entry point is `quantathon-harness/model.py`. It parses QASM 2/3, extracts raw size, gate, depth, connectivity, and coarse temporal interaction features, and records an approximate simplified operation count from zero rotations and adjacent cancellations. It also calls the supplied `quantathon-harness/chi_walk.py` to estimate threshold-specific MPS bond-dimension pressure. The walk filters single-qubit rotations near integer multiples of π and supplies separate near-0/π counts to the model. A separate bounded fast path handles inputs above 40 MB; it uses whole-file counts, so its graph and depth features are approximate. The trained estimator is in `quantathon-harness/artifacts/runtime_model.joblib`.

This implementation tests the feature idea in the [September 11 runtime-prediction paper](https://arxiv.org/html/2609.12980v1): features derived from a closer approximation to executable work may help beyond raw QASM features. It does not claim to reproduce Quantum Rings compilation or the paper's transpilation process.

## Reproduce with uv

From the challenge root:

```sh
uv sync --locked
uv run --locked python -m unittest research/test_model.py
uv run --locked python research/train_runtime.py --extract-only
uv run --locked python research/evaluate_chi_randomness.py
uv run --locked python research/evaluate_geometry_families.py
uv run --locked python research/fit_geometry_families.py
uv run --locked --extra report python research/plot_geometry_families.py
uv run --locked python research/chi_walk_probe.py --extract
uv run --locked python research/chi_walk_probe.py --evaluate
uv run --locked python research/fit_chi_walk.py
uv run --locked --extra report python research/plot_chi_walk.py
uv run --locked python research/evaluate_rotation_filter.py --extract
uv run --locked python research/evaluate_rotation_filter.py --evaluate
uv run --locked python research/fit_rotation_filter.py
uv run --locked --extra report python research/plot_rotation_filter.py
uv run --locked python quantathon-harness/run.py --team "Your Team Name" --circuits training_circuits --out research/training_submission_rotation_filter.csv
uv run --locked python quantathon-harness/score.py --pred research/training_submission_rotation_filter.csv --labels runtime-data.csv
```

`train_runtime.py --extract-only` caches circuit features in `research/features.json`. The selected fitting script writes the current model artifact. The original model-selection study remains in `research/validation.json`; rerunning `train_runtime.py` without `--extract-only` would overwrite the selected model. The studies use five folds, keeping all thresholds of each circuit together, and a stricter exploratory holdout of structural clusters. Model selection uses the exact score formula in `score.py`; the label for each timeout is 14,400 seconds. The one successful label above the stated cap remains unchanged.

The initial five-fold circuit-grouped comparison selected ExtraTrees:

| Feature set | Official score | Timeout-row score |
|---|---:|---:|
| Basic counts and depth | 0.8742 | 0.5931 |
| Raw structure | 0.8876 | 0.6541 |
| Raw plus simplified and temporal features | **0.8911** | **0.6821** |

The more severe structural-cluster holdout scored 0.6902. This reflects extrapolation risk when the hidden circuits differ substantially from the training structures. A 10% ridge blend raised that stress score to 0.7043 but lowered ordinary grouped validation to 0.8815. The log-median multiplicative error on the grouped folds was 1.123×; this coexists with a few large misses, listed in `validation.json`.

The subsequent [χ and circuit-diversity evaluation](CHI_RANDOMNESS_EVALUATION.md) added cut-rank upper-bound and entropy-style diversity features. The corrected effective-entanglement proxy is `χ_est = χ_upper ^ randomness_proxy`, so it approaches the χ upper bound as randomness increases. Its earlier fitted artifact blended 75% of the χ + diversity model with 25% of the effective-χ model; its circuit-grouped score was **0.89210** and structural-cluster holdout score was **0.70050**. The diversity–runtime association was uneven, so no monotonic runtime constraint was imposed.

The previous fitted artifact added time-resolved cut pressure, reordered graph geometry, soft circuit-family fingerprints, liveness, and dense-vector-reference features. The paired [geometry and family evaluation](GEOMETRY_FAMILY_EVALUATION.md) reported **0.89702** circuit-grouped and **0.70705** structural-cluster scores. The supplied χ-walk shape and cost features then achieved **0.91047** on a distribution-matched holdout with circuit grouping. The current artifact adds an [angle-aware χ-walk filter and explicit rotation counts](ROTATION_FILTER_EVALUATION.md), improving that matched score to **0.91459** while lowering the structural-cluster stress score from **0.73520 to 0.72787**.

For the hidden circuits, put them in a directory and run the harness with `--circuits` pointing to it. Ensure the model artifact and uv environment are present alongside `model.py`; every requested circuit and threshold should yield one positive finite prediction. The harness accepts raw `.qasm` and compressed `.qasm.zst` files.

## Limits

The parser is a bounded structural scanner, not a full OpenQASM interpreter. It counts custom gate bodies when they are called and resolves common angle constants. Unrecognized syntax is tallied as a feature; complex loops and recursive custom gates are not fully expanded. The large-file path trades detailed topology for speed. Both paths were trained on the provided circuit files. The validation report measures predictive accuracy on these files, not on the hidden holdout.

All 532 supplied circuits were extracted and none had unrecognized statements after parser fixes. Four circuits used the fast path and cannot receive detailed χ-walk features. The χ walk uses a 3-second per-circuit budget, with extrapolation on ten training circuits in the angle-aware cache. In the latest full harness run, maximum parser time was 12.8083 seconds and maximum prediction time was 0.0648 seconds; the walk budget is reduced when a new circuit's baseline scan uses most of the 14-second working budget. Timings on a different machine may differ.

The unmodified harness produced 1,596 positive finite predictions for all 532 training circuits at three thresholds. All 1,497 labeled combinations were covered, with no cap violations. The fitted training-set score was 98.19%; this is a pipeline check, not a generalization estimate. Use the 0.91459 matched validation score for model assessment.
