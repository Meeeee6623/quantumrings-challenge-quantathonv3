# Holdout submission and circuit inspection

The selected holdout model is **v8**
(`categorical_setting_threshold_experts_v8`), with artifact SHA-256
`3a53a37acd93e3e84a39cf73e252b00b8f8dc448ed937600f6f246855f79819b`.
It uses QASM-derived features and a categorical simulator setting. Filename and
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
error probabilities. Training neighbor errors use the v8 circuit-grouped
out-of-fold predictions. The inspector never reads holdout runtime labels and
does not modify the submission CSV or artifact. If an inspected filename also
exists in the training set, that same file is excluded as its own neighbor.

Validation completed before holdout release: **38 tests** passed; the
v8 full harness generated **1,596/1,596** training-library predictions with no
failures or timing-cap violations (parser max **13.5479 s**, prediction max
**0.1530 s**). The fixed circuit-grouped matched score is **0.926914** and
structural-stress score **0.760192**. Its matched timeout subscore is lower
than v7 (0.9426 versus 0.9639); prescreening used released labels, so the
small score gains may be optimistic. A two-file plain/compressed QASM fixture
also passed the harness, validator, and inspector; the largest 218 MB
training QASM was inspected successfully. These checks do not establish
holdout accuracy before its circuits and labels arrive. A copy of the
`quantathon-harness` directory also generated and validated predictions from
outside the repository, confirming that the submission path does not need
research caches.
