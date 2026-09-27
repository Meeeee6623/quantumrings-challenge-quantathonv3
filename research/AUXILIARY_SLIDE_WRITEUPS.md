# Auxiliary slide writeups: predicting simulator runtime from QASM

These are slide-ready words for the detailed backup section of the presentation. "Score" means the challenge's mean duration score on the fixed, matched circuit-grouped folds. A percentage-point change is a subtraction of two scores expressed as percentages. The first seven points on the [continuous progression chart](presentation_figures/00_feature_progression.png) come from the same four-model sweep with one cumulative feature pack at a time. The later points come from successive implementations on the same folds, so their differences describe the path taken rather than an isolated causal effect of one component.

## Slide 1 — Evaluation design

**On-slide copy**

- 532 circuits, 1,497 labeled circuit-setting runs
- Five matched folds keep every setting of a circuit together
- The score penalizes multiplicative runtime error: a 10× miss earns 0.5

**Speaker notes**

The predictor receives a QASM circuit and a simulator setting, then returns seconds. Settings 16, 64, and 512 are mixed within each validation fold; they are not used to define separate holdouts. Circuit grouping prevents the model from seeing one setting of a test circuit during training and another during validation. The folds also balance twelve structural profiles derived without runtime labels, making the primary estimate resemble a holdout from the supplied distribution. We score all 1,497 labeled runs out of fold. For a timeout, the target is 14,400 seconds; successful runs keep their measured duration even if it exceeds that value. The score is logarithmic: a tenfold underprediction and a tenfold overprediction receive the same per-row score of 0.5. We selected features using these released labels, so the hidden holdout is the decisive test of the small final gains. A separate structural-cluster split probes a harder distribution shift.

**Suggested visual:** the [model sweep heatmap](presentation_figures/01_initial_model_sweep.png) or a cropped view of the fold protocol. **Evidence:** [fold mapping](comparison_folds.csv), [evaluation contract](COMPARISON_HANDOFF.md).

## Slide 2 — Counts become a workload description

**On-slide copy**

- Basic circuit size and depth: **87.66%**
- Gate mix, timing, and local simplification: **89.27%** (**+1.61 points**)
- ExtraTrees leads the same four-model comparison at both stages

**Speaker notes**

The baseline knows how large the source circuit is: qubit count, active qubits, operation counts, depth, measurements, resets, and QASM size. That is already useful, but circuits with similar dimensions can require very different work. The next feature pack distinguishes one-qubit from two-qubit work, counts named gate types, measures Clifford and non-Clifford mix, and records where operations occur. It also estimates a small amount of simplification, such as zero-angle rotations, adjacent cancellations, and folded rotations. These are source-level proxies, not a trace of Quantum Rings' compiler. The 1.61-point gain is the largest step before the χ walk: describing the *kind* and sequence of work matters more than adding another estimator to basic counts. For each pack, Ridge, histogram boosting, random forest, and ExtraTrees used identical grouped folds and fixed configurations; ExtraTrees won.

**Suggested visual:** the first two points of the [progression chart](presentation_figures/00_feature_progression.png). **Evidence:** [sweep results](presentation_figures/model_sweep.json), [feature matrix definition](sweep_feature_model_tree.py).

## Slide 3 — Static entanglement, geometry, and soft patterns

**On-slide copy**

- χ upper bound and structural diversity: **89.43%** (**+0.15 points**)
- Interaction graph and cut geometry: **89.70%** (**+0.27 points**)
- Soft circuit-pattern scores: **89.98%** (**+0.28 points**)

**Speaker notes**

The χ features describe how much rank a circuit *could* develop across qubit cuts, subject to geometric limits and the requested setting. Diversity summaries measure how evenly gate types, angles, qubit pairs, and time windows are used. Neither quantity measures actual entanglement or proves that a circuit is random. Geometry then adds connected components, pair reuse, span, graph ordering, cut pressure through time, and qubit liveness. Two circuits can have the same two-qubit gate count but connect different register regions or accumulate pressure at different times. The final static pack uses continuous QFT-like, QAOA-like, arithmetic-like, and other pattern scores. These fingerprints let the regressor use partial resemblance without forcing a single algorithm name. The gains are modest because all three packs summarize the circuit; they do not yet follow how potential simulation cost changes gate by gate.

**Suggested visual:** [static-feature ablations](presentation_figures/02_static_feature_ablations.png). **Evidence:** [sweep results](presentation_figures/model_sweep.json), [geometry study](GEOMETRY_FAMILY_EVALUATION.md), [χ/diversity study](CHI_RANDOMNESS_EVALUATION.md).

## Slide 4 — The χ walk captures the path through the circuit

**On-slide copy**

- Static features: **89.98%**; add the χ walk: **91.12%** (**+1.14 points**)
- Stream gates and track potential bond growth at each cut
- Combine the walk's shape with a χ-dependent work estimate

**Speaker notes**

A static peak cannot tell us whether a circuit reaches that peak early and stays there or reaches it only near the end. The χ walk processes operations in order. It maintains a conservative upper path for local log₂ χ across register cuts, records when the path first reaches the requested cap, and notes how much two-qubit work occurs near that cap. A cost proxy increases the price of one- and two-qubit gates as local χ grows; two-qubit terms also account for the distance they cross. The model receives eight walk summaries: cost, potentially entangling fraction, peak, saturation fraction and onset, links at cap, extrapolation, and availability. In an earlier paired ExtraTrees ablation, the selected shape-plus-cost view gained 1.13 points over its own static baseline, consistent with the 1.14-point gain in the staged four-model sweep. The walk is a runtime feature, not a calculation of Schmidt rank or a claim about the simulator's internal algorithm. It has a three-second budget; very large files use a bounded fast path.

**Suggested visual:** [χ-walk ablation](presentation_figures/03_chi_walk_ablation.png). **Evidence:** [χ-walk implementation](../quantathon-harness/chi_walk.py), [paired study](CHI_WALK_EVALUATION.md), [sweep results](presentation_figures/model_sweep.json).

## Slide 5 — Rotation angles change the walk's interpretation

**On-slide copy**

- Original walk: **91.12%**; angle-aware walk: **91.55%** (**+0.44 points**)
- Rotations within 0.3 radians of an integer multiple of π keep a basis-state flag
- Near-0/π counts and mean angle mixing enter the model directly

**Speaker notes**

The original classical-state tracker treated a numeric X/Y-like rotation as mixing even when its angle was near zero or π. That can make the subsequent χ trajectory look more entangling than the circuit's parameters suggest. The revised rule applies to numeric RX, RY, U, and U3 rotations whose first angle lies within 0.3 radians of an integer multiple of π. The gate still contributes to the work estimate; the rule only changes the walk's basis-state flag. Four additional inputs report the counts near zero and π, the fraction among parsed rotations, and mean |sin(angle)|. Symbolic or unparsed angles remain conservative, and this rule never turns a previously nonclassical qubit back into a classical one. The matched-fold gain is 0.44 points in the staged sweep. In the historical paired test, the same choice reduced structural-cluster stress score from 73.52% to 72.79%, so it is an empirical choice for the expected similar-distribution holdout rather than a universal physics correction.

**Suggested visual:** [angle-gating ablation](presentation_figures/04_angle_gating.png). **Evidence:** [paired rotation study](ROTATION_FILTER_EVALUATION.md), [walk implementation](../quantathon-harness/chi_walk.py).

## Slide 6 — Soft patterns transfer better than hard algorithm labels

**On-slide copy**

- Soft circuit fingerprints added **+0.28 points** in the staged sweep
- External family probabilities changed **91.55% to 91.50%**
- Algorithm names remain hypotheses unless source labels verify them

**Speaker notes**

We tested whether geometry could identify the underlying algorithm and thereby predict runtime. Continuous fingerprints helped the runtime regressor: a circuit can be partly QFT-like or QAOA-like while also sharing other structural traits. We also trained an external classifier on generated MQT reference families. It reached 96.9% family accuracy on that reference task, but 443 of the 532 challenge circuits lay beyond the reference distance range. Adding its family probabilities to the staged runtime model slightly lowered the matched score, and the earlier paired test also lowered both matched and structural-stress scores. Ordered QAOA and Shor-like motifs worked on controlled generator examples but did not transfer reliably enough to enter the final model. The challenge files do not provide verified algorithm identities, so the presentation should call these QASM-pattern matches, not source-algorithm labels.

**Suggested visual:** [algorithm-signal comparison](presentation_figures/05_algorithm_signals.png). **Evidence:** [external classifier probe](ALGORITHM_GEOMETRY_PROBE.md), [ordered motif probe](SEQUENCE_MOTIF_PROBE.md), [sweep results](presentation_figures/model_sweep.json).

## Slide 7 — Model selection and setting-specific experts

**On-slide copy**

- Four regressors tested for each of eight feature packs; ExtraTrees won **8/8**
- Global feature model: **91.55%**
- Setting experts and timeout router: **91.97%** (**+0.41 points** on the chart)

**Speaker notes**

Feature engineering drove more of the progress than estimator changes. Every staged pack used the same five circuit-grouped folds and the same four fixed model configurations: Ridge, histogram gradient boosting, random forest, and ExtraTrees. ExtraTrees led all eight rows of the sweep, including the rejected external-family pack. After the feature-rich global model, we added one ExtraTrees regressor for each simulator setting, 16, 64, and 512. Predictions blend the global and setting-specific estimates halfway in log-runtime space. Separate setting-specific classifiers identify likely timeouts and set the prediction to the timeout target when their probability crosses the chosen cutoff. The chart rises from 91.55% to 91.97%. This transition comes from a later implementation with some refitting differences; the within-experiment paired comparison measured a 0.45-point gain over its own global baseline. The expert system handles setting interactions and timeout behavior without imposing a false monotonic runtime rule.

**Suggested visual:** [complete model sweep](presentation_figures/01_initial_model_sweep.png) alongside the [model architecture tree](presentation_figures/08_model_architecture_tree.png). **Evidence:** [sweep results](presentation_figures/model_sweep.json), [expert validation](merged_model_validation.json), [training code](train_merged_model.py).

## Slide 8 — Second parser and corrections for residual misses

**On-slide copy**

- Secondary DAG/angle scanner: **92.25%** (**+0.29 points**)
- Runtime floors, template analogues, near-basis calibration: **92.62%** (**+0.36 points**)
- Each correction targets a specific error pattern

**Speaker notes**

The second parser supplies another view of operation dependencies and angle geometry. The two parsers produced 480 candidate columns before deduplication and model-specific selection. Specialists and timeout classifiers can use the union, while the global regressor uses a smaller primary view. The chart's 0.29-point step is the difference between successive implementations, so it should not be attributed solely to the parser. The next 0.36 points come from sequential corrections tested within the same full-union evaluation: a reset-family minimum adds about 0.18 points, a large-work minimum about 0.03, blending with training-fold template analogues about 0.08, and near-basis calibration about 0.07. Each analogue bank or floor reference is built from the training folds when scoring held-out rows. These rules address errors that a smooth regressor struggled to price, but they remain narrow rather than imposing a universal runtime floor.

**Suggested visual:** the later segment of the [continuous progression](presentation_figures/00_feature_progression.png) or the [architecture tree](presentation_figures/08_model_architecture_tree.png). **Evidence:** [dual-parser and rule validation](full_union_model_validation.json), [training code](train_full_union_model.py).

## Slide 9 — Feature audit and the final model

**On-slide copy**

- Distinct fitted inputs: **325 to 241 to 120**
- Matched score: **92.62% to 92.69% to 92.77%**
- Structural stress score also rose: **75.54% to 76.02% to 76.64%**

**Speaker notes**

The simulator setting is categorical in the final fit: separate indicators represent 16, 64, and 512 instead of forcing the model to interpret those numbers as a smooth numeric scale. Categorical encoding by itself was essentially neutral and slightly lowered the matched score. The feature audit removed redundant or weakly useful inputs; together, the categorical and audited view reached 92.69%. The final fixed schema retains 120 distinct input columns across the global regressor, setting experts, and timeout classifiers. It scores 92.77% on the matched grouped folds, an additional 0.08 points while using far fewer inputs than the 325-column earlier fit. The small final gains deserve caution because we selected features on the released labels. The 76.64% structural-stress score is a useful warning about unfamiliar circuit structure, not the target distribution used to choose the model.

**Suggested visual:** [final pruning](presentation_figures/06_final_pruning.png). **Evidence:** [categorical test](categorical_setting_probe.json), [audit test](feature_audit_prune_probe.json), [final-model out-of-fold predictions](production_model_oof.csv), [fixed input schema](../quantathon-harness/production_features.json).

## Slide 10 — Limits to state during Q&A

**On-slide copy**

- χ and algorithm features are structural proxies, not observed simulator internals
- Four circuits above 40 MB use a bounded parser path without detailed walk features
- Hidden-circuit accuracy remains unmeasured

**Speaker notes**

The χ walk tracks a possible upper trajectory; it does not recover the actual entanglement spectrum, numerical truncation, or internal contraction order. Soft algorithm fingerprints describe QASM resemblance, not provenance. The parser has a strict time budget, and the four largest training circuits take a fast path that omits detailed graph and walk measurements. This keeps inference within the challenge limit but weakens information on unusual giant inputs. The final matched-fold estimate is 92.77%, whereas holding out whole structural clusters scores 76.64%. The difference demonstrates a real transfer risk if unseen circuits come from new structures. Finally, repeated feature choices and pruning used released labels. We should present the grouped scores as model-selection evidence and use the hidden holdout to judge generalization.

**Suggested visual:** [continuous progression](presentation_figures/00_feature_progression.png) with the stress series from [final pruning](presentation_figures/06_final_pruning.png). **Evidence:** [holdout readiness](HOLDOUT_READINESS.md), [large-circuit evaluation](BIG_CIRCUITS_EVALUATION.md), [final-model out-of-fold predictions](production_model_oof.csv).
