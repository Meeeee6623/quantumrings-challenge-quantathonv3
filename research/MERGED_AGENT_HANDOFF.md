# Handoff: merged Quantum Rings runtime predictor

This note is for an agent continuing from the parallel research flow.  The
original methodology snapshot remains in `COMPARISON_HANDOFF.md`; the complete
cross-pipeline analysis is in `MERGED_COMPARISON.md`.

## Current selected submission

The checked-in `quantathon-harness/artifacts/runtime_model.joblib` is now the
**full-union threshold-expert model**, artifact version
`full_union_threshold_experts_v1`.  It replaces the previous single global
ExtraTrees artifact.

The model uses only QASM-derived quantities and the requested threshold.  It
does not use filename, benchmark source, provided/inferred family labels, or
other adapter metadata.

Prediction path:

```text
QASM
 ├─ existing bounded geometry parser + angle-aware chi walk → 243 columns
 └─ independent structural/DAG/graph/angle scanner          → 237 columns

global ExtraTrees(243 columns)
threshold-specific ExtraTrees[16|64|512](480-column union)
point log10(runtime) = 0.5 global + 0.5 threshold expert

per-threshold timeout classifier(480 columns)
 └─ probability >= 0.35 → predict 14,400 seconds
```

Unknown future thresholds fall back safely to the global model.  The challenge
holdout is expected to use the known thresholds 16, 64, and 512.

## Apples-to-apples fixed-fold results

All rows below use the committed `comparison_folds.csv`, 1,497 labeled runs,
532 circuits, the exact challenge score, and scikit-learn 1.9.1.

| Model | Matched score | Structural stress | Matched timeout score | Matched >10× misses |
|---|---:|---:|---:|---:|
| Previous parallel global model, refitted | 0.91520 | 0.72854 | 0.73148 | 46 |
| Compact threshold/timeout merge | 0.91967 | **0.75712** | 0.91799 | 38 |
| **Selected full union** | **0.92266** | 0.74611 | **0.94468** | **30** |

Full union minus parallel global on the matched folds is **+0.00746**.  A
paired bootstrap over whole circuits gives a 95% interval of
**[+0.00402, +0.01109]**.  Its structural-stress gain is +0.01756, but the
12-cluster interval crosses zero.  These remain training-data model-selection
estimates, not hidden-holdout results.

Matched scores by threshold for the selected model:

- threshold 16: 0.92924
- threshold 64: 0.92017
- threshold 512: 0.91764

Median factor error is 1.078× and p95 factor error is 5.21×.  The calibrated
90% conformal-style interval has log10 radius 0.4991, or approximately 3.16×,
with 89.98% matched-OOF coverage.

## Large-circuit update

The full union improves every decoded-size bin from the prior large-circuit
analysis.  For the four >40-million-character circuits, matched score rises
from 0.7500 to 0.7859 and >10× misses fall from one to zero.  This does not make
file size a causal complexity metric; the second coarse structural view simply
retains useful source/gate information when the original parser uses its huge
fast path.

## Important files

| File | Purpose |
|---|---|
| `quantathon-harness/model.py` | Runtime integration, blending, timeout routing, intervals |
| `quantathon-harness/extended_features.py` | Embedded second source-QASM scanner |
| `quantathon-harness/extended_threshold.py` | Threshold transforms used by that scanner |
| `quantathon-harness/artifacts/runtime_model.joblib` | Selected fitted artifact |
| `research/train_full_union_model.py` | Refit/evaluate the selected model |
| `research/train_merged_model.py` | Refit the compact 243-feature fallback |
| `research/extract_extended_features.py` | Rebuild the second-pass training cache |
| `research/extended_features.json` | Cached 532-circuit second-pass features |
| `research/full_union_model_validation.json` | Machine-readable fixed-fold metrics |
| `research/full_union_model_oof.csv` | Every fixed-fold prediction |
| `research/full_union_feature_columns.json` | Exact ordered input schemas |
| `research/MERGED_COMPARISON.md` | Full comparison, caveats, and rationale |
| `research/merged_model_comparison.png` | Presentation comparison figure |

The 236 circuit-only values produced by the embedded scanner were checked
against the independent pipeline cache for all 532 circuits with zero
mismatches.

## Reproduce

```bash
uv sync --locked --extra report
uv run --locked python research/extract_extended_features.py
uv run --locked python research/train_full_union_model.py
uv run --locked python research/plot_merged_comparison.py
uv run --locked python -m unittest discover -s research -p 'test_*.py'
```

Run a holdout set:

```bash
uv run --locked python quantathon-harness/run.py \
  --team "Your Team" --circuits path/to/holdout --out submission.csv
```

The full training-library harness completed 532 circuits / 1,596 requested
predictions with no failures, no missing rows, and all positive predictions.
Same-circuit score was 0.9901; that number is only a pipeline check.  On this
development host, combined parse median/p95/max were 0.31/8.79/38.32 seconds,
with nine circuits over 15 seconds.  The final target machine is expected to be
substantially faster, and accuracy was explicitly prioritized over this local
timing limit.  Inference max in the full run was 0.321 seconds; the final
single-thread-per-row setting benchmarks faster than the version loaded by that
run.

## Continuation warnings

1. Do **not** run `fit_rotation_filter.py` as the final build step: it recreates
   the previous single-global artifact and overwrites the selected union model.
2. Use `train_full_union_model.py` for the primary artifact.  Use
   `train_merged_model.py` only when intentionally selecting the compact
   structural-shift fallback.
3. Keep all threshold rows of a circuit in one fold.  Do not compare scores from
   different split definitions as if they were paired.
4. Preserve the `extended__` namespace.  Some concepts intentionally appear in
   both representations; collapsing them changes the learned model.
5. Do not present the 0.9901 fitted score as validation.  Use 0.92266 matched OOF
   and disclose that model choices were made on the available training data.
6. Artifact SHA-256:
   `6c1e44915436ad291610c54fe4dfe7b423cbafd44f0c11eb25ed509ad12780e9`.

## Recommended challenge-time sequence

1. Put only the holdout QASM/QASM.ZST files in a separate directory.
2. Run the harness once with all three supplied thresholds.
3. Confirm expected row count, positive finite predictions, and no failures.
4. Inspect parse/predict timing warnings, but do not refit based on hidden labels.
5. Preserve the generated CSV before trying any experimental alternative.
6. If there is time, generate a compact-model CSV as a fallback, but use the
   full union as the default based on the matched fixed-fold evidence.
