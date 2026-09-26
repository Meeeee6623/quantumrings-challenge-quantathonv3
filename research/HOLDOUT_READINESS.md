# Holdout submission and circuit inspection

The selected holdout model is **v7**
(`pruned_union_threshold_experts_v7`), with artifact SHA-256
`3f0f8a04f99dfa03aadcc2179c62a1093fefb46e97198df89c8dd6ee1d183958`.
It uses QASM-derived features and the requested threshold. Filename and
training-error metadata are absent from the predictor. The screened
structural-neighbor correction remains research-only because its gains were
not stable across new grouped assignments
([probe](STRUCTURAL_NEIGHBOR_PROBE.md)).

Use a directory containing **only holdout `.qasm` or `.qasm.zst` files**.
The harness reads compressed files directly. From the repository root:

```bash
uv sync --locked --extra report
uv run --locked python quantathon-harness/run.py \
  --team 'Quantum Rings Research' --circuits /path/to/holdout \
  --out holdout_submission.csv
uv run --locked python research/validate_submission.py \
  --circuits /path/to/holdout --submission holdout_submission.csv
```

The validator exits nonzero if any requested `(circuit, threshold)` row is
missing or duplicated, a prediction is nonpositive/nonfinite, the team field
is inconsistent, or a parse/predict time exceeds 15 seconds. Keep predictions
above 14,400 seconds if the model emits them: the released data include
successful runs longer than the usual timeout cap, so unconditional clamping
would be wrong.

To inspect geometry and training analogues without changing predictions:

```bash
uv run --locked python research/inspect_holdout.py \
  --circuits /path/to/holdout --out-dir holdout_inspection --plot
uv run --locked python research/inspect_holdout.py \
  --circuits /path/to/holdout --name example.qasm \
  --out-dir holdout_inspection_example --plot
```

The inspection CLI writes:

- `inspection.md`: the 50 highest-priority unusual circuits, with geometry,
  feature-range flags, nearest training circuits, and grouped-OOF neighbor
  errors;
- `inspection.csv`: one row per circuit/threshold, suited to sorting by
  nearest-neighbor distance, unusual-feature count, or prediction;
- `inspection.json`: all circuits, top neighbors in two geometry views,
  nearby **missed** training examples (≥3× error), nearby **close** examples
  (≤1.25× error), and exact count-signature matches;
- `geometry_overview.png` with `--plot`: distances versus circuit operation
  count, with training nearest-neighbor 95th-percentile references.

These distances are heuristic out-of-training-range flags, **not** calibrated
error probabilities. Training neighbor errors are from circuit-grouped
out-of-fold predictions. The inspector never reads holdout runtime labels and
does not modify the submission CSV or artifact. If an inspected filename also
exists in the training set, that same file is excluded as its own neighbor.

Validation completed before holdout release: **34 tests** passed; the
full harness generated **1,596/1,596** training-library predictions with no
failures or timing-cap violations (parser max **13.3033 s**, prediction max
**0.0935 s**). The fixed circuit-grouped matched score is **0.926153** and
structural-stress score **0.755369**. A two-file plain/compressed QASM fixture
also passed the harness, validator, and inspector; the largest 218 MB
training QASM was inspected successfully. These checks do not establish
holdout accuracy before its circuits and labels arrive. A copy of the
`quantathon-harness` directory also generated and validated predictions from
outside the repository, confirming that the submission path does not need
research caches.
