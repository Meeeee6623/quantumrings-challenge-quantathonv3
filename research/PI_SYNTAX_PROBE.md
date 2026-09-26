# Probe: source-level π syntax in near-identical circuits

Two 32-qubit QASM3 circuits (`957c248b.qasm` and `db12c1f0.qasm`) have the
same coarse geometry and 574 operations. At threshold 16 they took 0.175 and
4.705 seconds. Their source text contains 170 and 169 `pi` tokens,
respectively, while many controlled-phase angles differ by small numerical
amounts. This suggests exact angle representation might affect simulator
simplification, but the pair does not establish causality.

The selected v5 model already receives `extended__source_pi_token_count`.
We tested two additional circuit-only features for the threshold experts:
the count divided by the number of angle expressions, and the fraction of
`pi` tokens in the final 20% of source text. The global model, folds, runtime
floors, analogue blend, and near-basis calibration were unchanged. The fixed
folds keep every threshold of a circuit together.

| Extra features | Matched OOF score | Structural-stress score | Matched >10× misses |
|---|---:|---:|---:|
| Selected v5, no extras | **0.92547** | 0.75030 | 28 |
| Normalized `pi` count | 0.92507 | **0.75668** | 27 |
| Normalized count + tail fraction | 0.92524 | 0.75288 | **26** |

A 25% log-space blend of v5 with the normalized-count model scored 0.92539
matched and 0.75118 structural stress. Its paired matched change was −0.00009;
the 95% circuit-bootstrap interval crosses zero. The standalone model's
structural improvement also has a 12-cluster bootstrap interval crossing zero.
Because holdout circuits are expected to resemble the training distribution,
we kept v5 as the selected artifact.

`probe_pi_syntax.py` reproduces the experiment; `pi_syntax_features.json`,
`pi_syntax_oof.csv`, and `pi_syntax_probe.json` preserve the inputs and
predictions. The script never overwrites the selected model.
