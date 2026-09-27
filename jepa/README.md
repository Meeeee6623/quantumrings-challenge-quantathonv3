# JEPA circuit-representation experiment

## Outcome

This experiment tested whether a JEPA-style learned circuit embedding improves
the Quantum Rings runtime predictor. It was implemented as an additive branch:
the existing QASM parsers, threshold specialists, timeout classifiers, floors,
analogue correction, and submission format remain unchanged.

It did **not** improve validation, so no JEPA weights or feature columns were
promoted. The table is the recorded v7-era benchmark; the branch has since
advanced to the separately validated v8 categorical-setting artifact, which
also does not load JEPA weights or features.

| Variant | Matched OOF | Change vs. v7 | Structural OOF | Change vs. v7 |
|---|---:|---:|---:|---:|
| v7 baseline | 0.926153 | — | 0.755369 | — |
| JEPA in experts/classifiers | 0.924610 | -0.001542 | 0.753391 | -0.001977 |
| JEPA in global model + experts | 0.924141 | -0.002011 | 0.746728 | -0.008641 |

The v7 promotion gate required structural-stress gain of at least `+0.005` and
matched-score decline no worse than `-0.002`. Neither route qualified. The
experiment has not been re-run against v8 because its rejection already made
it unsuitable for the active submission path.

## Motivation and design

The established predictor already contains gate, depth, topology, temporal-cut,
effective-entanglement, angle, DAG, and soft-family features. JEPA was tested
to learn local gate order, interaction layout, and repeated parameter blocks
directly from QASM without using runtime labels to train the representation.

`quantathon-harness/jepa_tokens.py` collects a deterministic, bounded token
stream inside the existing `Stats.add()` parser path—no second QASM scan. It
keeps at most 64 tokens from each of eight source-order windows (512 maximum).
Each token includes gate family, arity, coarse angle class, source window,
normalized endpoint qubits, qubit span, and an entangling-operation flag.

`quantathon-harness/jepa_encoder.py` defines a one-layer, four-head
Transformer (hidden width 64) with a 32-dimensional output. A context encoder
sees one masked contiguous span; an EMA target encoder embeds the unmasked
sequence. Training uses normalized latent MSE plus a small variance penalty.
Runtime, timeout, filename, and source family never enter that objective.

## Runtime and validation safeguards

Files above 20 MB intentionally receive zero JEPA values and
`jepa_available=0` to protect the 15-second parser limit. Nine of 532 released
circuits used that fallback. Bounded token collection reached 7.842 seconds at
maximum, excluding the existing independent auxiliary parser passes.

Validation is fold-isolated. For every committed matched and structural-stress
fold, the encoder trains only on the training-fold QASM. It then embeds both
tree-training and held-out circuits for that fold, preserving a shared
coordinate system without fitting on held-out QASM. Each encoder trained for
six epochs; the existing feature selection and postprocessing otherwise match
v7 exactly.

The JEPA objective converged, but its representation did not capture simulator
cost better than the existing explicit features. With 532 circuits and no
broader permitted unlabeled corpus, it appears to add noise or unstable family
variation, especially under structural transfer. The implementation remains a
bounded, reproducible base for future representation work.

## Reproduce

```bash
uv sync --locked --extra paper
uv run --locked --extra paper python research/extract_jepa_tokens.py
uv run --locked --extra paper python research/probe_jepa.py
uv run --locked --extra paper python -m unittest research/test_model.py
```

The complete result and encoder logs are in `jepa/results.json`; row-level OOF
predictions are in `jepa/results_oof.csv`. PyTorch comes from the existing
optional `paper` extra because JEPA is not part of the active artifact.
