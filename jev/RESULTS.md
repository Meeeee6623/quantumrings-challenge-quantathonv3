# JEV runtime-bin experiment

**Status:** complete evaluation through OpenRouter on JEV latest. The experiment made 214 successful JEV calls for 107 held-out circuits / 301 measured circuit-threshold runs. Reported cost was **$0.06080**, median request latency **0.334 s**, p95 **0.422 s**.

## Question

Can JEV classify a Quantum Rings circuit into log-runtime categories? Does providing measured, structurally similar circuits improve its prediction?

The challenge score is:

```
score = max(0, 1 - abs(log10(predicted_s / actual_s)) / 2)
```

We report decoded-seconds challenge score and nearest-category accuracy separately.

## Evaluation design

- Test split: the pre-existing matched grouped fold 0, 107 circuits / 301 runs.
- Reference pool: remaining folds, 425 circuits. All threshold rows for one circuit remain together.
- Circuit input: QASM-derived structural summaries, gate counts, temporal two-qubit windows, fixed simulator setting, and hardware configuration.
- Excluded: raw QASM, filenames, source/family labels, and target labels.
- Zero-shot: target circuit summary only.
- Retrieval-assisted: eight nearest training circuits using standardized log1p structural features. Measured runtimes were not used to select references; they were shown to JEV after selection.
- JEV model: `~typesafe/jev-latest` through OpenRouter's Decisions API.
- Categories:
  - four-bin: `[1, 10, 100, 1000]` seconds;
  - wide-bin: `[0.1, 1, 10, 100, 1000, 10000, 14400, 100000]` seconds.

The 14,400-second category represents a timeout; successful runs may exceed it, hence the 100,000-second category.

## Results

| Method | Categories | Class accuracy | Challenge score | 95% circuit bootstrap interval |
|---|---|---:|---:|---:|
| JEV zero-shot | Four | 19.9% | 24.5% | 20.6%–28.5% |
| JEV zero-shot | Wide | 6.3% | 16.0% | 11.9%–20.7% |
| JEV + eight structural references | Four | 85.7% | 73.9% | 70.1%–77.3% |
| JEV + eight structural references | Wide | 76.4% | **78.7%** | 75.3%–82.0% |
| Eight-reference nearest-neighbour baseline | Continuous runtime | — | **84.4%** | — |
| Current predictor, stored grouped OOF predictions | Continuous runtime | — | **93.2%** | — |

For context, the maximum possible score if an oracle is restricted to four representative outputs is 80.9%; with the wide bins it is 87.6%. Bins themselves leave substantial precision on the table.

JEV's probability-weighted expected-score decoder did not materially improve it: 74.5% for retrieved four-bin and 78.1% for retrieved wide-bin, versus 73.9% and 78.7% from its selected categories.

## What happened

Zero-shot JEV systematically predicted slow or timeout-like bins. It is not usable as a standalone runtime classifier on these structural summaries.

Retrieval changes the result dramatically: it reaches 85.7% four-bin category accuracy and 78.7% challenge score with wide bins. But the plain nearest-neighbour baseline using the exact same eight reference circuits scores 84.4% continuously, so most of the value is in structural retrieval and known reference runtimes, rather than JEV adding predictive power.

JEV can be an explainable coarse routing or uncertainty signal, but it should not replace the local continuous predictor. A possible later ablation is to add JEV's category probabilities to a cross-fitted blend and retain it only if grouped OOF score improves beyond the existing 93.2% baseline.

## Reproduction

The experiment uses the OpenRouter Decisions endpoint, model `~typesafe/jev-latest`, and a server-side API key. No credential is committed. The request cache records all prompts, response probabilities, and split metadata; it must remain outside the public repository because it contains circuit-derived summaries and training reference runtimes.

References:

- [JEV latest on OpenRouter](https://openrouter.ai/~typesafe/jev-latest)
- [OpenRouter Decisions API](https://openrouter.ai/docs/api/api-reference/alphadecisions/submit-a-decisions-request)
