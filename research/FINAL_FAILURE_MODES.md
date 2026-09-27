# Circuits the final model still misses

This is an audit of the **selected v7 model's circuit-grouped out-of-fold predictions on released training labels**. It does not identify failures on the unseen holdout. The exact scoring rule caps a prediction at 14,400 seconds **only for a timed-out actual run**; successful runs retain their measured duration. Regenerate [the 28-row severe-error table](final_failure_audit.csv) and [segment counts](final_failure_summary.json) with [`analyze_final_failures.py`](analyze_final_failures.py). The script never changes the fitted artifact.

The matched-fold score is **0.926153** over 1,497 runs. **28 rows from 25 circuits exceed 10× error**: 18 underpredictions, 10 overpredictions, 26 successful runs, and 2 timeouts. Their errors are distributed: the worst 10 circuits account for only **11.8%** of all score loss, and perfect hindsight correction of every run on those 10 would add at most **0.00870** score. Subsecond runs are 501 rows but **37.6%** of remaining score loss; 33 timeout rows contribute only **1.1%**. The separate structural-cluster holdout scores **0.755369** and has **221** >10× rows, indicating that unseen structure is a much harder transfer setting than the expected similar-distribution holdout. [The headroom report](REMAINING_HEADROOM.md) gives the loss accounting.

![Remaining out-of-fold score loss](remaining_headroom.png)

## Tentative algorithm-like types

The names below are **overlapping QASM-pattern tags**, never verified source algorithms. QAOA uses the independently tested ordered cost/mixer motif; the other tags use transparent soft fingerprints and fixed cutoffs in the audit script. A circuit may appear in several rows. Mean scores are for *all* runs carrying the tag; severe counts refer to the 28 rows above 10×. These segments were not used to train a hard algorithm classifier.

| Structural pattern tag | Circuits / runs | Mean matched score | >10× runs | What the remaining error suggests |
|---|---:|---:|---:|---|
| QAOA-like ordered cost/mixer motif | 33 / 93 | 0.8712 | 5 | Strongly nonmonotone threshold response and parameter-sensitive near-basis behavior remain after scalar angle summaries. |
| QFT/phase-like fingerprint | 26 / 76 | 0.8458 | 5 | Dense controlled-phase structure can have very different realized work from a cut-rank possibility; near-identical coarse summaries can diverge. |
| Arithmetic-like fingerprint | 56 / 144 | 0.8972 | 6 | Width, multi-control operations, and expanded custom work are difficult to price from counts and static graph proxies. |
| Graph-state-like fingerprint | 8 / 23 | 0.8101 | 2 | Small gate counts do not rule out costly threshold-512 execution; the exact entangling pattern matters. |
| Random-grid-like fingerprint | 60 / 173 | 0.8737 | 8 | Interaction order and parameter patterns matter beyond pair counts; this tag overlaps QAOA/variational. |
| Variational-like fingerprint | 105 / 275 | 0.8954 | 8 | Rotation distributions are coarser than exact parameter sequences; this tag also overlaps QAOA. |
| >40-million-character fast parser | 4 / 12 | 0.7623 | 1 | The safety path omits detailed graph and χ-walk signals on the largest inputs. |
| Reset-heavy | 7 / 14 | 0.9112 | **0** | The narrow training-analogue floor corrected the earlier spectacular Grover-like misses; no current >10× row remains in this tag. |

For reference, the external eight-family MQT geometry classifier placed **23 of 28 severe rows** outside its generated-reference distance range. But it marks **443 of all 532 circuits** out of range, so that number alone is weak evidence about failure cause. Its forced `qaoa`, `qft`, or `randomcircuit` nearest-class guesses are **not actual challenge labels**. The ordered QAOA motif is a stronger within-generator structural test (154/162 generated positives, zero among 190 tested negatives), but it too does not prove algorithmic intent or transfer across source collections; see [the motif probe](SEQUENCE_MOTIF_PROBE.md), [geometry classifier probe](ALGORITHM_GEOMETRY_PROBE.md), and [Shor/QPE probe](SHOR_FAMILY_PROBE.md). No released challenge circuit cleared the optional Shor-like sequence threshold, so **none of these misses can be called a verified Shor circuit**.

## Concrete misses and hypotheses

| Circuit, threshold | Pattern evidence | Actual → predicted | Plausible explanation to test |
|---|---|---:|---|
| `917fa943.qasm`, 512 | 56-qubit QAOA-like motif score 1.0; all 392 parsed rotations near 0/π | **166.31 → 3.76 s** (44× low) | The binary near-basis χ flag and one fraction may miss the **ordering** of cost/mixer angles or a backend threshold branch. The same circuit is much faster at 16 and 64. |
| `5c176f37.qasm`, 64 | 56-qubit QAOA-like motif score 1.0; none of 504 parsed rotations near 0/π | **0.18 → 7.51 s** (42× high) | Some parameter/threshold combinations may simplify or truncate sharply; a smooth runtime model cannot assume monotonic work. This is a hypothesis about simulator behavior, not an observed optimization trace. |
| `957c248b.qasm` versus `db12c1f0.qasm`, 16 | Both 32 qubits, 574 operations, 511 two-qubit gates, QFT-like score ≈0.8 | **0.175 → 4.14 s** versus **4.705 → 0.20 s** | Controlled-phase angle details, exact `pi` syntax, bit reversal, or execution variability may distinguish the pair. Extra normalized `pi`-syntax features failed to improve matched validation ([probe](PI_SYNTAX_PROBE.md)); the mechanism remains unproven. |
| `bd32080b.qasm`, 64 | 256 qubits, 278,875 operations, 20,176 multi-qubit operations; mixed arithmetic/GHZ/graph soft fingerprints | **6,223 → 240 s** (26× low) | Custom/multi-control expansion and actual contraction order may cost more than our bounded structural summaries imply. Its 512 run times out, which the router catches. |
| `6533abd6.qasm`, 16 | 256 qubits, 97,920 two-qubit operations; no strong listed soft family score | **9,723 → 580 s** (17× low) | A wide, repeated entangling workload can escape the small-work floors despite obvious gate volume; a circuit-level contraction-width or temporal workload representation might help. |
| `f5d72dae.qasm`, 512 | 32-qubit graph-state-like fingerprint 1.0; only 64 operations; external graph-state template guess is within its reference range | **34.96 → 3.12 s** (11× low) | A compact graph preparation may trigger costly entanglement or threshold behavior that gate count misses. The tag is still a structural match, not a verified source name. |
| `0534bf07.qasm`, 16 | 51.7-million-character fast-path file, 256 qubits; actual timeout | **≥14,400 → 1,387 s** (10× low) | Fast parsing preserves work scale but not detailed connectivity, custom expansion, or χ trajectory. It is one of four giant files and cannot support a broad size-specific correction. |

The QFT-related hypothesis has a specific theoretical caution: the **core** QFT can have low entangling power while bit reversal changes operator entanglement, so controlled-phase count alone is not a realized-entanglement measurement ([Chen, Stoudenmire, and White](https://arxiv.org/abs/2210.08468)). The QAOA motif follows the cost/mixer structure introduced by [Farhi, Goldstone, and Gutmann](https://arxiv.org/abs/1411.4028). Both papers motivate what to inspect; neither diagnoses Quantum Rings' hidden optimizer or confirms a source family for a scrambled file.

## Why simple patches were not promoted

The remaining loss is heterogeneous. On released labels, isotonic enforcement of nondecreasing runtime across thresholds cut matched score from **0.92615 to 0.89738**; **270 of 532 circuits** have measured inversions across their available threshold settings. A more aggressive timeout-router override improved one structural fold but hurt a real timeout on a fresh grouped assignment ([disagreement probe](TIMEOUT_DISAGREEMENT_EVALUATION.md)). Soft structural-neighbor blends showed small, unstable gains on new folds ([neighbor probe](STRUCTURAL_NEIGHBOR_PROBE.md)). Extra source-`pi` features and optional QAOA/Shor sequence features likewise failed to improve both relevant holdouts consistently. None of these negative findings proves the associated physics idea false; they show that the tested, cheap proxy did not justify changing v7.

Two useful follow-up diagnostics remain **hypotheses**, not validated fixes: compare ordered parameter blocks and local gate neighborhoods for the QAOA/QFT near-duplicate pairs; and inspect a reduced gate-time tensor-network contraction width for the wide arithmetic/graph outliers. Any new feature needs several fresh circuit-grouped assignments, a parser-time check on >40 MB inputs, and a per-segment audit before replacing the selected artifact.
