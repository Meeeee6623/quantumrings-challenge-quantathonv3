# JEV runtime-bin experiment

**Status:** partial live result. Vercel AI Gateway accepted the account and
produced two evaluations for one held-out circuit, then its TypeSafe upstream
repeatedly returned `429 rate_limit_exceeded` / “high demand.” The results below
are useful as a probe, not an accuracy claim.

## Question

Can TypeSafe JEV classify a Quantum Rings circuit into log-runtime
representatives, and does showing it structurally similar labelled circuits
improve the result?

The challenge scores predictions as:

```
score = max(0, 1 - abs(log10(predicted_s / actual_s)) / 2)
```

A classifier should therefore be evaluated by its decoded seconds and this
score, not only by class accuracy.

## Fixed evaluation design

- **Evaluation split:** existing matched grouped fold 0: 107 circuits / 301
  labelled runs.
- **References:** the other folds, 425 circuits. Every threshold of a circuit
  remains in the same split.
- **Inputs to JEV:** QASM-derived structural summaries, gate counts, temporal
  two-qubit-operation windows, and the fixed simulator and hardware
  configuration.
- **Not sent:** raw QASM, filenames, source/family labels, or target labels.
- **Zero-shot condition:** only target circuit summary.
- **Example-assisted condition:** eight nearest training circuits by
  standardized log1p structural features; labels were not used to choose
  neighbors. JEV then receives their measured threshold/runtime pairs.
- **Categories tested:** `[1, 10, 100, 1000]` seconds, and a wider set
  `[0.1, 1, 10, 100, 1000, 10000, 14400, 100000]` seconds. JEV returns a
  probability for every category.

The wide set includes a 14,400-second timeout representative. Successful runs
can exceed 14,400 seconds, so it also includes 100,000 seconds.

## Live JEV probe

The completed circuit had actual runtimes of 86.85 s, 144.50 s, and 5,763.22 s.

| Threshold | Actual | Zero-shot choice | Zero-shot score | Eight-example choice | Eight-example score |
|---:|---:|---:|---:|---:|---:|
| 16 | 86.85 s | 1,000 s | 46.9% | 100 s | 96.9% |
| 64 | 144.50 s | 1,000 s | 58.0% | 100 s | 92.0% |
| 512 | 5,763.22 s | 1,000 s | 62.0% | 1,000 s | 62.0% |

For the wide-bin version at threshold 512, both conditions chose 14,400 s and
scored 80.1%; the example-assisted call assigned only 39% probability to that
category.

The probe suggests that retrieved, measured analogues can make JEV's low- and
medium-threshold choice materially better. It also shows that a coarse bin
remains lossy for high runtimes. It is only one circuit and must not be
presented as a benchmark result.

## Offline binning reference

These scores use all 301 runs in the fixed fold. They are **not JEV results**.

| Method | Challenge score |
|---|---:|
| Current predictor, continuous stored OOF predictions | 93.25% |
| Current predictor rounded to 1 / 10 / 100 / 1,000 s | 78.65% |
| Oracle restricted to 1 / 10 / 100 / 1,000 s | 80.86% |
| Current predictor rounded to wider bins | 84.22% |
| Oracle restricted to wider bins | 87.62% |

The current predictor identifies the nearest four-bin class on 92.03% of rows,
but its binned score is only 78.65%. Correct coarse classification and accurate
log-runtime prediction are therefore different objectives. A four-class JEV
model cannot exceed 80.86% on this fold even with perfect class choices.

## Interpretation

JEV looks most interesting here as a retrieval-aware decision layer: provide a
target summary plus selected measured analogues, then use its probability
distribution as one signal beside the existing runtime predictor. It should not
replace the continuous model with four coarse outputs.

To test it properly, complete the cached 107-circuit evaluation, report both
zero-shot and reference-assisted scores with circuit-grouped confidence
intervals, and compare with a plain nearest-neighbour baseline that uses the
same eight references. The Vercel TypeSafe provider currently needs to recover
from its temporary capacity errors before that run can finish.

## Reproduction

The local experiment runner was prepared as `research/jev_benchmark.py` in the
development workspace. It uses Vercel AI Gateway's evaluation endpoint and
model `typesafe-ai/jev`; credentials are read at runtime and are not committed.

References:

- [JEV on Vercel AI Gateway](https://vercel.com/ai-gateway/models/jev)
- [Vercel evaluation API](https://vercel.com/docs/ai-gateway/modalities/evaluation)
