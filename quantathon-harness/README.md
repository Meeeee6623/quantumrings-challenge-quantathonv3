# Quantathon — Submission Harness

Predict how long each circuit takes to simulate, straight from the `.qasm` file.
This harness produces the **one file you send back to us**.

## Setup (once, from the repository root)

```bash
uv sync --locked
```

The repository also has a pinned `requirements.txt` for environments that
cannot use `uv`, but the locked `uv` environment is the tested path. The fitted
artifact and supporting Python modules are checked in; no training step is
required for inference.

## Harness interface

The harness calls these methods in `model.py`:

| Method | Job | Called |
|---|---|---|
| `RuntimeModel()` | load your trained model | once |
| `featurize(qasm_text)` | parse a circuit into features | once per circuit |
| `predict(features, threshold)` | return **predicted seconds** (one number) | once per (circuit, threshold) |

There is no timeout flag — if you think a run will time out, just predict a duration
**≥ the 4-hour cap (14400 s)**.

This fork already includes the final predictor. It uses `chi_walk.py`,
`extended_features.py`, `extended_threshold.py`, `runtime_floors.py`,
`template_analogues.py`, `rotation_calibration.py`, and
`artifacts/runtime_model.joblib` alongside `model.py`. Keep these files together
when copying the harness to another location.

**Caps:** `featurize` and `predict` must each run in **≤ 15 s per circuit**. The
harness times you and warns on anything over.

## What's included

| Path | Contents |
|---|---|
| `../training_circuits/` | the training circuits, one `<id>.qasm.zst` per circuit |
| `../holdout-circuits/` | the 59 released circuits, with no runtime labels |
| `../runtime-data.csv` | training labels, one row per labeled `(circuit, threshold)` run |

The circuits are `zstd`-compressed. `run.py` decompresses them for you, so you
don't need raw files to run the harness. To get raw `.qasm` files for exploring
and training, see
[Getting the raw `.qasm` files](../README.md#getting-the-raw-qasm-files) in the
main README. Do not put a raw copy beside its compressed file in the same
input folder, or the harness will predict it twice.

## Run it

1. From the repository root, generate a CSV from the released holdout folder:

```bash
uv run --locked python quantathon-harness/run.py \
  --team "Your Registered Team Name" --circuits holdout-circuits \
  --out submission.csv
```

This writes **`submission.csv`**:

```
team,filename,threshold,pred_duration_s,parse_s,predict_s
```

2. Validate all 177 predictions and the 15-second limits:

```bash
uv run --locked python research/validate_submission.py \
  --circuits holdout-circuits --submission submission.csv
```

3. **DM `submission.csv` back to the organizers.** The command does not send it.

## Training pipeline check (optional)

You have the training labels (`../runtime-data.csv`). Run the harness on
`../training_circuits/` and use the scorer for an in-sample pipeline check:

```bash
uv run --locked python quantathon-harness/run.py \
  --team "Your Registered Team Name" --circuits training_circuits \
  --out training_submission.csv
uv run --locked python quantathon-harness/score.py \
  --pred training_submission.csv --labels runtime-data.csv
```

Because the shipped artifact was fitted on all released training labels, this
score is not a holdout estimate. The circuit-grouped validation results are in
the root README and research handoff.

## How the automated third is scored

Your total score is **1/3 automated + 2/3 subjective** (presentation, novelty,
process, etc. — judged live, five equally-weighted categories). The automated third
is **pure duration accuracy**, per `(circuit, threshold)`:

```
score = max(0, 1 − |log10(pred / actual)| / 2)      # exact=1, 10× off=.5, 100× off=0
```

Runtimes span seconds to hours, so accuracy is measured in **log scale** — being
2× off costs the same whether the run is 1 second or 1 hour. Timeout circuits are
scored against the 4-hour cap, so predicting ≥ cap earns full credit on them and
predicting seconds for one is penalized like any other large miss.

Your score is the average over **every** hold-out run. A run missing from your
`submission.csv` scores 0, so make sure every circuit gets a prediction.

## Rules

- Build your own parser; no starter feature list is provided.
- You must be able to explain every part of your submission.
- Don't try to recover the source algorithm from filenames/metadata.
