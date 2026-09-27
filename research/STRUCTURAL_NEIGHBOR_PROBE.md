# Structural-neighbor analogue screen

We tested whether training circuits with similar QASM-derived gate counts and
geometry could improve predictions when an exact count signature had fewer
than two references. For each circuit-grouped fold, the neighbors and their
runtime labels came only from that fold's training circuits at the same
threshold. Label-free distance scaling used the full released QASM corpus,
including the held-out fold's circuit features. The selected v7
exact-template rule was left intact.

An initial screen covered three feature views, neighbor counts 1/3/5,
distance gates, neighbor-runtime-spread gates, and blend weights. Count-only
neighbors failed to improve matched validation. Two candidates that helped
both fixed splits were then tested on three *new* grouped assignments:

| View and fixed rule | Fixed matched change | Fixed structural change | Seed 181 matched | Seed 197 matched | Seed 211 matched |
|---|---:|---:|---:|---:|---:|
| Gate mix, 3 neighbors, 25% blend | +0.000512 | +0.010667 | −0.000063 | +0.001508 | +0.000072 |
| Shape, 3 neighbors, 25% blend | +0.000599 | +0.010115 | −0.000609 | +0.001083 | +0.001406 |

The larger fixed structural gains did not provide a convincing reason to add
another label-based postprocessor. The matched gain is small and one new
assignment reverses each candidate; the screen also searched many settings
on released labels. **The holdout model remains v7.** Gate-mix and shape
distances are useful as inspection diagnostics without affecting predictions.

The [probe script](probe_structural_neighbors.py) writes all screened results
to `structural_neighbor_probe.json`. Reproduce with:

```bash
uv run --locked python research/probe_structural_neighbors.py
```
