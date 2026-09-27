# Holdout submission and circuit inspection

The selected holdout model is **v9**
(`categorical_setting_threshold_experts_v9`), with artifact SHA-256
`fcb440014d75153cd41db6cfa5595a79ca0081e4b5aeaf5c599353f0bfc9fc59`.
It uses QASM-derived features and a categorical simulator setting. Filename and
training-error metadata are absent from the predictor. Its 120 input columns
are frozen in [`quantathon-harness/production_features.json`](../quantathon-harness/production_features.json).
The screened structural-neighbor correction remains research-only because its gains were
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
error probabilities. Training neighbor errors use the v9 circuit-grouped
out-of-fold predictions. The inspector never reads holdout runtime labels and
does not modify the submission CSV or artifact. If an inspected filename also
exists in the training set, that same file is excluded as its own neighbor.

The v9 full harness generated **1,596/1,596** training-library predictions with
no failures or timing-cap violations (parser max **13.1351 s**, prediction max
**0.1372 s**). A final-code check of four large QASM files generated all 12
rows, with parser max **10.8796 s** and prediction max **0.0324 s**. The fixed
circuit-grouped matched score is **0.927739** and structural-stress score
**0.766378**. Feature selection reused released labels, so these checks do
not establish hidden-holdout accuracy.
