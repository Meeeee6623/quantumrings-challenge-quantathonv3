# Near-0/π rotation filter

The χ walk now keeps a computational-basis flag for numeric `RX`, `RY`, `U`, and `U3` rotations whose first angle is within **0.3 radians of an integer multiple of π**. Such an ideal rotation at exactly `kπ` maps a basis state to a basis state; the tolerance is a heuristic for near-basis behavior. Symbolic or unparsed angles remain conservative. The walk still charges the gate's single-qubit cost. It also counts rotations near 0 and near π separately, and records the fraction of parsed rotations near either point and mean `|sin(angle)|`. Those four values are additional model inputs, separate from the filtered χ trajectory.

The primary test uses the **same five distribution-matched, circuit-grouped folds** as the prior evaluation. Thresholds 16, 64, and 512 appear together in every fold; no threshold is held out as a group. All 1,497 labeled rows get out-of-fold predictions. The previous 231 features, estimator settings, and score formula are unchanged.

| Added view | Matched score | Gain over original walk | >10× misses | Structural stress score |
|---|---:|---:|---:|---:|
| Original χ walk | 0.91047 | — | 54 | 0.73520 |
| Original walk + angle counts | 0.91364 | +0.00317 | 47 | 0.73509 |
| Filtered walk | 0.91398 | +0.00350 | **42** | 0.73124 |
| **Filtered walk + angle counts** | **0.91459** | **+0.00412** | 45 | 0.72787 |

![Paired rotation-filter ablation](rotation_filter_ablation.png)

The selected combined view has a paired circuit-bootstrap 95% interval of **+0.00072 to +0.00783** score over the original walk. This interval does not account for selecting among the four tested views. The gain is strongest at χ=64 (**0.91089→0.91499**) and χ=512 (**0.89626→0.90687**); χ=16 falls slightly (**0.92169→0.92050**). The harder structural-cluster stress score falls by 0.00734. Because the stated target is a similar-distribution holdout, the combined view was selected for the fitted submission; extrapolation to unseen circuit families remains a limitation.

The change directly addresses the pair found in the [error analysis](CHI_WALK_ERROR_ANALYSIS.md). `70d5d461.qasm` has repeated `RX` angles close to π and 2π; the filtered walk treats them as near-basis and no longer grows χ for its diagonal `RZZ` pattern. `8bd576fc.qasm` has the same gate skeleton but mixing `RX` angles, so its χ trajectory still reaches the cap. At χ=512, their out-of-fold predictions change from **1,488→9.81 s** (actual 8.85 s) and **9.06→1,574 s** (actual 1,465 s), respectively. This pair is evidence of predictive value, not proof that the simulator uses this exact physical shortcut. At χ=64, the fast circuit remains overpredicted by roughly 19×.

![χ=512 out-of-fold predictions before and after angle filter](rotation_filter_predictions.png)

The filter changes χ tables for 71 of the 528 circuits with walk features. It does not solve other miss types such as reset-heavy Grover-like circuits. The model still uses a 3-second walk budget and the existing 40 MB fast path. The rotation counts and filtered walk are generated in the same pass, so they do not require a second QASM parse.

The full unmodified harness produced **1,596 positive finite predictions** for all 532 circuits × three thresholds, covering **all 1,497 labeled rows**. There were no parser or inference failures and no 15-second cap violations. Maximum parser time was **12.8083 s** and maximum per-row prediction time was **0.0648 s**. All **21 tests** passed. The official scorer returned **98.19%** on those same training circuits. That number verifies the end-to-end pipeline; the **0.91459 out-of-fold score** is the generalization estimate for a similar-distribution holdout.

## Reproduce

```sh
uv run --locked python research/evaluate_rotation_filter.py --extract
uv run --locked python research/evaluate_rotation_filter.py --evaluate
uv run --locked python research/fit_rotation_filter.py
uv run --locked --extra report python research/plot_rotation_filter.py
uv run --locked python -m unittest discover -s research -p 'test_*.py'
uv run --locked python quantathon-harness/run.py --team 'Quantum Rings Research' --circuits training_circuits --out research/training_submission_rotation_filter.csv
uv run --locked python quantathon-harness/score.py --pred research/training_submission_rotation_filter.csv --labels runtime-data.csv
```

The cached walks, fold scores, and out-of-fold predictions are in `chi_walk_angle_cache.json`, `rotation_filter_probe.json`, and `rotation_filter_oof.csv`.
