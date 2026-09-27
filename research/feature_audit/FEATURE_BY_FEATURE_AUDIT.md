# Feature-by-feature audit

Fitted artifact: `pruned_union_threshold_experts_v7`. 325 inputs; 241 preliminary keeps and 84 preliminary drops. The target is observed log10 runtime in seconds (timeouts capped by the challenge). Every curve bins a feature within each of the three simulator settings and shows the median with interquartile shading. The simulator setting is treated as categorical in the plots. Impurity importances are weighted across the global and setting experts; correlated inputs divide importance. A flat marginal plot does not rule out an interaction. These are proposed decisions until grouped ablation is validated.

Runtime importance weights: 50% global and 50% setting experts, weighted by setting frequency. Timeout importance averages the three timeout classifiers. Spearman correlations are separately computed inside each setting.

![Observed runtime by categorical simulator setting](setting_vs_runtime.png)

![Grouped curves page 1](grouped_curves_01.png)

![Grouped curves page 2](grouped_curves_02.png)

![Grouped curves page 3](grouped_curves_03.png)

![Grouped curves page 4](grouped_curves_04.png)

![Grouped curves page 5](grouped_curves_05.png)

![Grouped curves page 6](grouped_curves_06.png)

![Grouped curves page 7](grouped_curves_07.png)

![Grouped curves page 8](grouped_curves_08.png)

![Grouped curves page 9](grouped_curves_09.png)

![Grouped curves page 10](grouped_curves_10.png)

![Grouped curves page 11](grouped_curves_11.png)

![Grouped curves page 12](grouped_curves_12.png)

![Grouped curves page 13](grouped_curves_13.png)

![Grouped curves page 14](grouped_curves_14.png)

![Grouped curves page 15](grouped_curves_15.png)

![Grouped curves page 16](grouped_curves_16.png)

![Grouped curves page 17](grouped_curves_17.png)

![Grouped curves page 18](grouped_curves_18.png)

![Grouped curves page 19](grouped_curves_19.png)

![Grouped curves page 20](grouped_curves_20.png)

![Grouped curves page 21](grouped_curves_21.png)

| # | Feature | Decision | Runtime imp. | Timeout imp. | ρ 16 / 64 / 512 | Reason |
|---:|---|---|---:|---:|---|---|
| 1 | `active_qubits` | [drop](./grouped_curves_01.png) | 0.00069 | 0.00000 | 0.44/0.53/0.36 | Near-perfect rank duplicate of log_dense_log2_bytes (|rho|=1.000) with lower fitted importance. |
| 2 | `barriers` | [drop](./grouped_curves_01.png) | 0.00078 | 0.00011 | -0.29/-0.21/-0.51 | Near-perfect rank duplicate of log_barriers (|rho|=1.000) with lower fitted importance. |
| 3 | `cancel_pairs` | [keep](./grouped_curves_01.png) | 0.00233 | 0.00001 | 0.42/0.08/0.19 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 4 | `chi_mid_budget` | [drop](./grouped_curves_01.png) | 0.00097 | 0.00093 | 0.66/0.49/0.55 | Near-perfect rank duplicate of log_chi_mid_budget (|rho|=1.000) with lower fitted importance. |
| 5 | `chi_timeline_mid_auc` | [drop](./grouped_curves_01.png) | 0.00114 | 0.00000 | 0.55/0.47/0.58 | Near-perfect rank duplicate of log_chi_timeline_mid_auc (|rho|=1.000) with lower fitted importance. |
| 6 | `chi_timeline_mid_t50` | [keep](./grouped_curves_01.png) | 0.00000 | 0.00088 | -0.41/-0.36/-0.45 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 7 | `chi_timeline_mid_t90` | [keep](./grouped_curves_01.png) | 0.00028 | 0.00121 | -0.46/-0.39/-0.48 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 8 | `chi_timeline_peak_auc` | [drop](./grouped_curves_01.png) | 0.00100 | 0.00000 | 0.56/0.47/0.57 | Near-perfect rank duplicate of log_chi_timeline_peak_auc (|rho|=1.000) with lower fitted importance. |
| 9 | `chi_timeline_post_mid_sat_twoq` | [keep](./grouped_curves_01.png) | 0.00028 | 0.00000 | 0.42/0.33/0.46 | Low fitted importance but a visible within-setting runtime trend; retain pending grouped ablation. |
| 10 | `chi_upper_mean` | [drop](./grouped_curves_01.png) | 0.00005 | 0.00000 | 0.57/0.47/0.57 | Near-perfect rank duplicate of log_chi_upper_peak (|rho|=0.998) with lower fitted importance. |
| 11 | `chi_upper_mid` | [drop](./grouped_curves_01.png) | 0.00004 | 0.00000 | 0.56/0.45/0.57 | Near-perfect rank duplicate of log_chi_upper_mid (|rho|=1.000) with lower fitted importance. |
| 12 | `chi_upper_peak` | [drop](./grouped_curves_01.png) | 0.00047 | 0.00000 | 0.58/0.46/0.57 | Near-perfect rank duplicate of log_chi_upper_peak (|rho|=1.000) with lower fitted importance. |
| 13 | `clifford` | [keep](./grouped_curves_01.png) | 0.00243 | 0.00126 | 0.66/0.68/0.70 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 14 | `component_dense_over_128gb` | [keep](./grouped_curves_01.png) | 0.00197 | 0.00000 | 0.45/0.53/0.56 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 15 | `conditional` | [drop](./grouped_curves_01.png) | 0.00169 | 0.00000 | 0.15/0.08/-0.07 | Near-perfect rank duplicate of log_conditional (|rho|=1.000) with lower fitted importance. |
| 16 | `dense_log2_bytes` | [drop](./grouped_curves_01.png) | 0.00085 | 0.00000 | 0.44/0.53/0.36 | Near-perfect rank duplicate of log_dense_log2_bytes (|rho|=1.000) with lower fitted importance. |
| 17 | `dense_over_128gb` | [keep](./grouped_curves_02.png) | 0.00516 | 0.00000 | 0.46/0.58/0.52 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 18 | `depth` | [drop](./grouped_curves_02.png) | 0.00069 | 0.00000 | 0.60/0.50/0.57 | Near-perfect rank duplicate of log_depth (|rho|=1.000) with lower fitted importance. |
| 19 | `depth_per_qubit` | [keep](./grouped_curves_02.png) | 0.00019 | 0.00000 | 0.41/0.23/0.33 | Low fitted importance but a visible within-setting runtime trend; retain pending grouped ablation. |
| 20 | `diagonal` | [drop](./grouped_curves_02.png) | 0.00110 | 0.00000 | 0.75/0.59/0.72 | Near-perfect rank duplicate of log_diagonal (|rho|=1.000) with lower fitted importance. |
| 21 | `diagonal_run_frac` | [keep](./grouped_curves_02.png) | 0.00395 | 0.00136 | 0.03/-0.20/-0.02 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 22 | `diagonal_run_max` | [keep](./grouped_curves_02.png) | 0.00220 | 0.00000 | 0.54/0.30/0.56 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 23 | `effective_ops` | [keep](./grouped_curves_02.png) | 0.00215 | 0.00163 | 0.72/0.71/0.79 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 24 | `fingerprint_arithmetic` | [drop](./grouped_curves_02.png) | 0.00124 | 0.00008 | -0.17/-0.16/-0.28 | Near-perfect rank duplicate of log_fingerprint_arithmetic (|rho|=1.000) with lower fitted importance. |
| 25 | `fingerprint_ghz` | [drop](./grouped_curves_02.png) | 0.00016 | 0.00000 | 0.08/-0.16/-0.17 | Near-perfect rank duplicate of log_fingerprint_ghz (|rho|=1.000) with lower fitted importance. |
| 26 | `fingerprint_grover` | [keep](./grouped_curves_02.png) | 0.00180 | 0.00116 | -0.08/-0.09/-0.15 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 27 | `fingerprint_phase_dyadic` | [keep](./grouped_curves_02.png) | 0.00258 | 0.00000 | 0.02/-0.18/-0.24 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 28 | `fingerprint_qaoa` | [keep](./grouped_curves_02.png) | 0.00019 | 0.00000 | 0.16/-0.01/-0.04 | Low fitted importance but a visible within-setting runtime trend; retain pending grouped ablation. |
| 29 | `fingerprint_qft` | [keep](./grouped_curves_02.png) | 0.00361 | 0.00216 | 0.03/-0.15/-0.21 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 30 | `fingerprint_random_grid` | [keep](./grouped_curves_02.png) | 0.00336 | 0.00879 | -0.21/-0.02/0.07 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 31 | `fingerprint_variational` | [keep](./grouped_curves_02.png) | 0.00575 | 0.00334 | 0.17/0.13/0.23 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 32 | `gate_ccx` | [drop](./grouped_curves_02.png) | 0.00000 | 0.00000 | -0.11/-0.13/-0.25 | Near-perfect rank duplicate of log_gate_ccx (|rho|=1.000) with lower fitted importance. |
| 33 | `gate_cp` | [keep](./grouped_curves_03.png) | 0.00009 | 0.00000 | 0.09/-0.06/-0.15 | Low fitted importance but a visible within-setting runtime trend; retain pending grouped ablation. |
| 34 | `gate_cx` | [keep](./grouped_curves_03.png) | 0.00200 | 0.00133 | 0.58/0.60/0.65 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 35 | `gate_h` | [drop](./grouped_curves_03.png) | 0.00101 | 0.00004 | 0.30/-0.01/-0.13 | Near-perfect rank duplicate of log_gate_h (|rho|=1.000) with lower fitted importance. |
| 36 | `gate_rx` | [drop](./grouped_curves_03.png) | 0.00069 | 0.00000 | 0.48/0.43/0.68 | Near-perfect rank duplicate of log_gate_rx (|rho|=1.000) with lower fitted importance. |
| 37 | `gate_ry` | [keep](./grouped_curves_03.png) | 0.00028 | 0.00000 | 0.08/0.42/0.48 | Low fitted importance but a visible within-setting runtime trend; retain pending grouped ablation. |
| 38 | `gate_rz` | [drop](./grouped_curves_03.png) | 0.00000 | 0.00011 | 0.54/0.49/0.69 | Near-perfect rank duplicate of log_gate_rz (|rho|=1.000) with lower fitted importance. |
| 39 | `gate_rzz` | [drop](./grouped_curves_03.png) | 0.00046 | 0.00000 | 0.27/0.17/0.28 | Near-perfect rank duplicate of log_gate_rzz (|rho|=1.000) with lower fitted importance. |
| 40 | `gate_swap` | [drop](./grouped_curves_03.png) | 0.00006 | 0.00000 | 0.06/-0.09/-0.10 | Tiny fitted importance and no clear within-setting runtime trend in the grouped curves. |
| 41 | `gate_sx` | [drop](./grouped_curves_03.png) | 0.00017 | 0.00000 | 0.15/0.44/0.53 | Near-perfect rank duplicate of extended__gate_count__sxdg (|rho|=0.995) with lower fitted importance. |
| 42 | `gate_t` | [drop](./grouped_curves_03.png) | 0.00000 | 0.00086 | 0.07/0.02/-0.00 | Near-perfect rank duplicate of log_gate_t (|rho|=1.000) with lower fitted importance. |
| 43 | `gate_u3` | [drop](./grouped_curves_03.png) | 0.00019 | 0.00000 | 0.09/0.42/0.49 | Near-perfect rank duplicate of log_gate_u3 (|rho|=1.000) with lower fitted importance. |
| 44 | `gate_x` | [keep](./grouped_curves_03.png) | 0.00217 | 0.00281 | -0.01/-0.13/-0.22 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 45 | `graph_cutwidth_original` | [drop](./grouped_curves_03.png) | 0.00146 | 0.00000 | 0.69/0.44/0.56 | Near-perfect rank duplicate of log_graph_cutwidth_original (|rho|=1.000) with lower fitted importance. |
| 46 | `graph_cutwidth_rcm` | [drop](./grouped_curves_03.png) | 0.00104 | 0.00068 | 0.69/0.48/0.60 | Near-perfect rank duplicate of log_graph_cutwidth_rcm (|rho|=1.000) with lower fitted importance. |
| 47 | `graph_degree_entropy` | [keep](./grouped_curves_03.png) | 0.00545 | 0.00003 | 0.05/0.20/0.07 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 48 | `graph_density` | [keep](./grouped_curves_03.png) | 0.00117 | 0.00000 | 0.10/-0.24/-0.07 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 49 | `graph_mean_cut_original` | [drop](./grouped_curves_04.png) | 0.00086 | 0.00004 | 0.73/0.47/0.64 | Near-perfect rank duplicate of log_graph_mean_cut_original (|rho|=1.000) with lower fitted importance. |
| 50 | `graph_mean_cut_rcm` | [drop](./grouped_curves_04.png) | 0.00013 | 0.00000 | 0.71/0.51/0.64 | Near-perfect rank duplicate of log_graph_mean_cut_rcm (|rho|=1.000) with lower fitted importance. |
| 51 | `graph_mindegree_width` | [keep](./grouped_curves_04.png) | 0.00028 | 0.00000 | 0.29/0.12/0.39 | Low fitted importance but a visible within-setting runtime trend; retain pending grouped ablation. |
| 52 | `graph_rcm_gain` | [keep](./grouped_curves_04.png) | 0.00256 | 0.00009 | -0.08/-0.42/-0.50 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 53 | `graph_span_rcm` | [drop](./grouped_curves_04.png) | 0.00000 | 0.00020 | 0.51/0.29/0.43 | Near-perfect rank duplicate of log_graph_span_rcm (|rho|=1.000) with lower fitted importance. |
| 54 | `largest_component_frac` | [keep](./grouped_curves_04.png) | 0.00011 | 0.00001 | 0.16/0.17/0.32 | Low fitted importance but a visible within-setting runtime trend; retain pending grouped ablation. |
| 55 | `late_new_pair_ratio` | [keep](./grouped_curves_04.png) | 0.00134 | 0.00179 | 0.01/-0.24/-0.20 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 56 | `liveness_mean_span` | [drop](./grouped_curves_04.png) | 0.00010 | 0.00003 | 0.35/0.46/0.54 | Near-perfect rank duplicate of log_liveness_mean_span (|rho|=1.000) with lower fitted importance. |
| 57 | `liveness_mean_window` | [keep](./grouped_curves_04.png) | 0.00422 | 0.00000 | 0.50/0.63/0.52 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 58 | `liveness_peak_window` | [drop](./grouped_curves_04.png) | 0.00109 | 0.00000 | 0.48/0.52/0.39 | Near-perfect rank duplicate of log_liveness_peak_window (|rho|=1.000) with lower fitted importance. |
| 59 | `max_degree` | [drop](./grouped_curves_04.png) | 0.00000 | 0.00000 | 0.28/0.01/-0.03 | Near-perfect rank duplicate of log_max_degree (|rho|=1.000) with lower fitted importance. |
| 60 | `mean_degree` | [drop](./grouped_curves_04.png) | 0.00000 | 0.00001 | 0.45/0.18/0.27 | Near-perfect rank duplicate of log_mean_degree (|rho|=1.000) with lower fitted importance. |
| 61 | `measurement_early_frac` | [keep](./grouped_curves_04.png) | 0.00248 | 0.00000 | -0.06/0.01/-0.16 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 62 | `measurements` | [drop](./grouped_curves_04.png) | 0.00195 | 0.00000 | -0.28/-0.19/-0.51 | Near-perfect rank duplicate of log_measurements (|rho|=1.000) with lower fitted importance. |
| 63 | `multi_q` | [drop](./grouped_curves_04.png) | 0.00000 | 0.00047 | -0.14/-0.13/-0.25 | Near-perfect rank duplicate of log_multi_q (|rho|=1.000) with lower fitted importance. |
| 64 | `n_qubits` | [drop](./grouped_curves_04.png) | 0.00082 | 0.00000 | 0.44/0.53/0.36 | Near-perfect rank duplicate of log_dense_log2_bytes (|rho|=1.000) with lower fitted importance. |
| 65 | `nonclifford` | [drop](./grouped_curves_05.png) | 0.00192 | 0.00000 | 0.73/0.68/0.74 | Near-perfect rank duplicate of log_nonclifford (|rho|=1.000) with lower fitted importance. |
| 66 | `nonclifford_ratio` | [keep](./grouped_curves_05.png) | 0.00225 | 0.00058 | -0.03/-0.01/-0.02 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 67 | `one_q` | [keep](./grouped_curves_05.png) | 0.00209 | 0.00181 | 0.69/0.70/0.75 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 68 | `ops` | [keep](./grouped_curves_05.png) | 0.00248 | 0.00221 | 0.77/0.73/0.79 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 69 | `pair_reuse` | [drop](./grouped_curves_05.png) | 0.00014 | 0.00046 | 0.44/0.44/0.45 | Near-perfect rank duplicate of log_pair_reuse (|rho|=1.000) with lower fitted importance. |
| 70 | `peak_window_twoq` | [keep](./grouped_curves_05.png) | 0.00000 | 0.00060 | -0.51/-0.49/-0.62 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 71 | `qasm_bytes` | [keep](./grouped_curves_05.png) | 0.00315 | 0.00058 | 0.79/0.71/0.79 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 72 | `random_angle_entropy` | [drop](./grouped_curves_05.png) | 0.00019 | 0.00077 | 0.16/0.16/0.29 | Near-perfect rank duplicate of log_random_angle_entropy (|rho|=1.000) with lower fitted importance. |
| 73 | `random_bigram_entropy` | [drop](./grouped_curves_05.png) | 0.00113 | 0.00000 | -0.20/-0.04/0.01 | Near-perfect rank duplicate of log_random_bigram_entropy (|rho|=1.000) with lower fitted importance. |
| 74 | `random_gate_entropy` | [drop](./grouped_curves_05.png) | 0.00007 | 0.00006 | -0.12/0.08/0.13 | Near-perfect rank duplicate of log_random_gate_entropy (|rho|=1.000) with lower fitted importance. |
| 75 | `random_pair_entropy` | [drop](./grouped_curves_05.png) | 0.00014 | 0.00000 | -0.11/0.11/0.10 | Near-perfect rank duplicate of log_random_pair_entropy (|rho|=0.998) with lower fitted importance. |
| 76 | `random_window_entropy` | [keep](./grouped_curves_05.png) | 0.00042 | 0.00000 | 0.31/0.39/0.50 | Low fitted importance but a visible within-setting runtime trend; retain pending grouped ablation. |
| 77 | `randomness_proxy` | [drop](./grouped_curves_05.png) | 0.00041 | 0.00000 | 0.07/0.26/0.35 | Near-perfect rank duplicate of log_randomness_proxy (|rho|=1.000) with lower fitted importance. |
| 78 | `reset_early_frac` | [keep](./grouped_curves_05.png) | 0.00277 | 0.07046 | 0.19/0.19/— | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 79 | `resets` | [keep](./grouped_curves_05.png) | 0.00041 | 0.01295 | 0.20/0.19/— | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 80 | `rotation_folds` | [drop](./grouped_curves_05.png) | 0.00021 | 0.00000 | 0.40/0.05/0.21 | Near-perfect rank duplicate of log_rotation_folds (|rho|=1.000) with lower fitted importance. |
| 81 | `small_angles` | [drop](./grouped_curves_06.png) | 0.00039 | 0.00002 | 0.56/0.40/0.52 | Near-perfect rank duplicate of log_small_angles (|rho|=1.000) with lower fitted importance. |
| 82 | `two_depth` | [drop](./grouped_curves_06.png) | 0.00101 | 0.00000 | 0.54/0.44/0.43 | Near-perfect rank duplicate of log_two_depth (|rho|=1.000) with lower fitted importance. |
| 83 | `two_q` | [drop](./grouped_curves_06.png) | 0.00187 | 0.00000 | 0.78/0.66/0.77 | Near-perfect rank duplicate of log_two_q (|rho|=1.000) with lower fitted importance. |
| 84 | `two_q_ratio` | [keep](./grouped_curves_06.png) | 0.00086 | 0.00004 | -0.03/-0.21/-0.23 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 85 | `twoq_window_2` | [keep](./grouped_curves_06.png) | 0.00014 | 0.00000 | -0.01/-0.11/0.03 | Low fitted importance but a visible within-setting runtime trend; retain pending grouped ablation. |
| 86 | `twoq_window_3` | [drop](./grouped_curves_06.png) | 0.00000 | 0.00000 | -0.04/-0.24/-0.11 | Near-perfect rank duplicate of log_twoq_window_3 (|rho|=1.000) with lower fitted importance. |
| 87 | `twoq_window_4` | [drop](./grouped_curves_06.png) | 0.00007 | 0.00000 | 0.12/0.00/0.20 | Near-perfect rank duplicate of log_twoq_window_4 (|rho|=1.000) with lower fitted importance. |
| 88 | `twoq_window_7` | [keep](./grouped_curves_06.png) | 0.00214 | 0.00000 | 0.44/0.43/0.55 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 89 | `unique_pairs` | [drop](./grouped_curves_06.png) | 0.00013 | 0.00000 | 0.56/0.41/0.45 | Near-perfect rank duplicate of log_unique_pairs (|rho|=1.000) with lower fitted importance. |
| 90 | `zero_angles` | [drop](./grouped_curves_06.png) | 0.00005 | 0.00040 | 0.39/-0.04/0.01 | Near-perfect rank duplicate of log_zero_angles (|rho|=1.000) with lower fitted importance. |
| 91 | `log_active_qubits` | [drop](./grouped_curves_06.png) | 0.00102 | 0.00000 | 0.44/0.53/0.36 | Near-perfect rank duplicate of log_dense_log2_bytes (|rho|=1.000) with lower fitted importance. |
| 92 | `log_barriers` | [keep](./grouped_curves_06.png) | 0.00145 | 0.00007 | -0.29/-0.21/-0.51 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 93 | `log_cancel_pairs` | [keep](./grouped_curves_06.png) | 0.01168 | 0.00011 | 0.42/0.08/0.19 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 94 | `log_chi_mid_budget` | [keep](./grouped_curves_06.png) | 0.01934 | 0.04375 | 0.66/0.49/0.55 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 95 | `log_chi_timeline_mid_auc` | [keep](./grouped_curves_06.png) | 0.00150 | 0.00000 | 0.55/0.47/0.58 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 96 | `log_chi_timeline_mid_t50` | [drop](./grouped_curves_06.png) | 0.00000 | 0.00063 | -0.41/-0.36/-0.45 | Near-perfect rank duplicate of chi_timeline_mid_t50 (|rho|=1.000) with lower fitted importance. |
| 97 | `log_chi_timeline_mid_t90` | [drop](./grouped_curves_07.png) | 0.00025 | 0.00102 | -0.46/-0.39/-0.48 | Near-perfect rank duplicate of chi_timeline_mid_t90 (|rho|=1.000) with lower fitted importance. |
| 98 | `log_chi_timeline_peak_auc` | [keep](./grouped_curves_07.png) | 0.00157 | 0.00000 | 0.56/0.47/0.57 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 99 | `log_chi_timeline_post_mid_sat_twoq` | [drop](./grouped_curves_07.png) | 0.00024 | 0.00000 | 0.42/0.33/0.46 | Near-perfect rank duplicate of chi_timeline_post_mid_sat_twoq (|rho|=1.000) with lower fitted importance. |
| 100 | `log_chi_upper_mean` | [drop](./grouped_curves_07.png) | 0.00138 | 0.00000 | 0.57/0.47/0.57 | Near-perfect rank duplicate of log_chi_upper_peak (|rho|=0.998) with lower fitted importance. |
| 101 | `log_chi_upper_mid` | [keep](./grouped_curves_07.png) | 0.00134 | 0.00000 | 0.56/0.45/0.57 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 102 | `log_chi_upper_peak` | [keep](./grouped_curves_07.png) | 0.00158 | 0.00000 | 0.58/0.46/0.57 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 103 | `log_clifford` | [keep](./grouped_curves_07.png) | 0.03520 | 0.02357 | 0.66/0.68/0.70 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 104 | `log_component_dense_log2_bytes` | [drop](./grouped_curves_07.png) | 0.00034 | 0.00004 | 0.45/0.46/0.46 | Near-perfect rank duplicate of log_largest_component (|rho|=1.000) with lower fitted importance. |
| 105 | `log_component_dense_over_128gb` | [drop](./grouped_curves_07.png) | 0.00154 | 0.00000 | 0.45/0.53/0.56 | Near-perfect rank duplicate of component_dense_over_128gb (|rho|=1.000) with lower fitted importance. |
| 106 | `log_conditional` | [keep](./grouped_curves_07.png) | 0.01592 | 0.00000 | 0.15/0.08/-0.07 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 107 | `log_custom_calls` | [keep](./grouped_curves_07.png) | 0.00000 | 0.00071 | -0.12/-0.15/-0.23 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 108 | `log_custom_definitions` | [keep](./grouped_curves_07.png) | 0.00000 | 0.00095 | -0.11/-0.16/-0.24 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 109 | `log_dense_log2_bytes` | [keep](./grouped_curves_07.png) | 0.00241 | 0.00000 | 0.44/0.53/0.36 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 110 | `log_dense_over_128gb` | [keep](./grouped_curves_07.png) | 0.00599 | 0.00000 | 0.46/0.58/0.52 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 111 | `log_depth` | [keep](./grouped_curves_07.png) | 0.00086 | 0.00000 | 0.60/0.50/0.57 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 112 | `log_diagonal` | [keep](./grouped_curves_07.png) | 0.01105 | 0.00299 | 0.75/0.59/0.72 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 113 | `log_diagonal_run_frac` | [keep](./grouped_curves_08.png) | 0.00404 | 0.00144 | 0.03/-0.20/-0.02 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 114 | `log_diagonal_run_max` | [keep](./grouped_curves_08.png) | 0.00513 | 0.00183 | 0.54/0.30/0.56 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 115 | `log_effective_ops` | [keep](./grouped_curves_08.png) | 0.03226 | 0.02501 | 0.72/0.71/0.79 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 116 | `log_fingerprint_arithmetic` | [keep](./grouped_curves_08.png) | 0.00135 | 0.00005 | -0.17/-0.16/-0.28 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 117 | `log_fingerprint_ghz` | [keep](./grouped_curves_08.png) | 0.00110 | 0.00004 | 0.08/-0.16/-0.17 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 118 | `log_fingerprint_grover` | [keep](./grouped_curves_08.png) | 0.00160 | 0.00116 | -0.08/-0.09/-0.15 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 119 | `log_fingerprint_phase_dyadic` | [keep](./grouped_curves_08.png) | 0.00271 | 0.00000 | 0.02/-0.18/-0.24 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 120 | `log_fingerprint_qaoa` | [drop](./grouped_curves_08.png) | 0.00017 | 0.00000 | 0.16/-0.01/-0.04 | Near-perfect rank duplicate of fingerprint_qaoa (|rho|=1.000) with lower fitted importance. |
| 121 | `log_fingerprint_qft` | [keep](./grouped_curves_08.png) | 0.00427 | 0.00205 | 0.03/-0.15/-0.21 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 122 | `log_fingerprint_random_grid` | [keep](./grouped_curves_08.png) | 0.00380 | 0.00760 | -0.21/-0.02/0.07 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 123 | `log_fingerprint_variational` | [keep](./grouped_curves_08.png) | 0.00551 | 0.00338 | 0.17/0.13/0.23 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 124 | `log_gate_ccx` | [keep](./grouped_curves_08.png) | 0.00022 | 0.00327 | -0.11/-0.13/-0.25 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 125 | `log_gate_cp` | [drop](./grouped_curves_08.png) | 0.00005 | 0.00000 | 0.09/-0.06/-0.15 | Near-perfect rank duplicate of gate_cp (|rho|=1.000) with lower fitted importance. |
| 126 | `log_gate_cx` | [keep](./grouped_curves_08.png) | 0.02585 | 0.04209 | 0.58/0.60/0.65 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 127 | `log_gate_cz` | [keep](./grouped_curves_08.png) | 0.00005 | 0.00000 | -0.13/-0.05/-0.12 | Low fitted importance but a visible within-setting runtime trend; retain pending grouped ablation. |
| 128 | `log_gate_h` | [keep](./grouped_curves_08.png) | 0.01149 | 0.04988 | 0.30/-0.01/-0.13 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 129 | `log_gate_p` | [keep](./grouped_curves_09.png) | 0.00008 | 0.00000 | 0.14/0.04/0.01 | Low fitted importance but a visible within-setting runtime trend; retain pending grouped ablation. |
| 130 | `log_gate_rx` | [keep](./grouped_curves_09.png) | 0.00012 | 0.00098 | 0.48/0.43/0.68 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 131 | `log_gate_rz` | [keep](./grouped_curves_09.png) | 0.00221 | 0.00008 | 0.54/0.49/0.69 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 132 | `log_gate_rzz` | [keep](./grouped_curves_09.png) | 0.00158 | 0.00000 | 0.27/0.17/0.28 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 133 | `log_gate_s` | [keep](./grouped_curves_09.png) | 0.00032 | 0.00001 | 0.32/0.55/0.58 | Low fitted importance but a visible within-setting runtime trend; retain pending grouped ablation. |
| 134 | `log_gate_t` | [keep](./grouped_curves_09.png) | 0.00096 | 0.00008 | 0.07/0.02/-0.00 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 135 | `log_gate_u1` | [keep](./grouped_curves_09.png) | 0.00000 | 0.00135 | 0.04/-0.01/-0.06 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 136 | `log_gate_u2` | [keep](./grouped_curves_09.png) | 0.00025 | 0.00000 | 0.16/0.41/0.48 | Low fitted importance but a visible within-setting runtime trend; retain pending grouped ablation. |
| 137 | `log_gate_u3` | [keep](./grouped_curves_09.png) | 0.00043 | 0.00000 | 0.09/0.42/0.49 | Low fitted importance but a visible within-setting runtime trend; retain pending grouped ablation. |
| 138 | `log_gate_x` | [keep](./grouped_curves_09.png) | 0.00512 | 0.07462 | -0.01/-0.13/-0.22 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 139 | `log_graph_cutwidth_original` | [keep](./grouped_curves_09.png) | 0.00857 | 0.00555 | 0.69/0.44/0.56 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 140 | `log_graph_cutwidth_rcm` | [keep](./grouped_curves_09.png) | 0.00466 | 0.00471 | 0.69/0.48/0.60 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 141 | `log_graph_degree_entropy` | [keep](./grouped_curves_09.png) | 0.00503 | 0.00000 | 0.05/0.20/0.07 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 142 | `log_graph_density` | [drop](./grouped_curves_09.png) | 0.00095 | 0.00000 | 0.10/-0.24/-0.07 | Near-perfect rank duplicate of graph_density (|rho|=1.000) with lower fitted importance. |
| 143 | `log_graph_mean_cut_original` | [keep](./grouped_curves_09.png) | 0.00432 | 0.00003 | 0.73/0.47/0.64 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 144 | `log_graph_mean_cut_rcm` | [keep](./grouped_curves_09.png) | 0.00257 | 0.00000 | 0.71/0.51/0.64 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 145 | `log_graph_mindegree_available` | [keep](./grouped_curves_10.png) | 0.00024 | 0.00000 | -0.12/-0.16/0.05 | Low fitted importance but a visible within-setting runtime trend; retain pending grouped ablation. |
| 146 | `log_graph_mindegree_width` | [drop](./grouped_curves_10.png) | 0.00023 | 0.00000 | 0.29/0.12/0.39 | Near-perfect rank duplicate of graph_mindegree_width (|rho|=1.000) with lower fitted importance. |
| 147 | `log_graph_rcm_gain` | [keep](./grouped_curves_10.png) | 0.00239 | 0.00004 | 0.08/-0.28/-0.23 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 148 | `log_graph_span_rcm` | [keep](./grouped_curves_10.png) | 0.00098 | 0.00099 | 0.51/0.29/0.43 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 149 | `log_largest_component` | [keep](./grouped_curves_10.png) | 0.00083 | 0.00002 | 0.45/0.46/0.46 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 150 | `log_late_new_pair_ratio` | [keep](./grouped_curves_10.png) | 0.00043 | 0.00240 | 0.01/-0.24/-0.20 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 151 | `log_liveness_mean_span` | [keep](./grouped_curves_10.png) | 0.00014 | 0.00013 | 0.35/0.46/0.54 | Low fitted importance but a visible within-setting runtime trend; retain pending grouped ablation. |
| 152 | `log_liveness_mean_window` | [keep](./grouped_curves_10.png) | 0.00891 | 0.00000 | 0.50/0.63/0.52 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 153 | `log_liveness_peak_window` | [keep](./grouped_curves_10.png) | 0.00168 | 0.00000 | 0.48/0.52/0.39 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 154 | `log_max_degree` | [keep](./grouped_curves_10.png) | 0.00194 | 0.00000 | 0.28/0.01/-0.03 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 155 | `log_max_span` | [keep](./grouped_curves_10.png) | 0.00037 | 0.00000 | 0.50/0.22/0.30 | Low fitted importance but a visible within-setting runtime trend; retain pending grouped ablation. |
| 156 | `log_mean_degree` | [keep](./grouped_curves_10.png) | 0.00188 | 0.00000 | 0.45/0.18/0.27 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 157 | `log_measurement_early_frac` | [keep](./grouped_curves_10.png) | 0.00252 | 0.00000 | -0.06/0.01/-0.16 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 158 | `log_measurements` | [keep](./grouped_curves_10.png) | 0.00368 | 0.00000 | -0.28/-0.19/-0.51 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 159 | `log_multi_q` | [keep](./grouped_curves_10.png) | 0.00022 | 0.00324 | -0.14/-0.13/-0.25 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 160 | `log_n_qubits` | [keep](./grouped_curves_10.png) | 0.00228 | 0.00000 | 0.44/0.53/0.36 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 161 | `log_nonclifford` | [keep](./grouped_curves_11.png) | 0.02396 | 0.00825 | 0.73/0.68/0.74 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 162 | `log_one_q` | [keep](./grouped_curves_11.png) | 0.01882 | 0.01265 | 0.69/0.70/0.75 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 163 | `log_ops` | [keep](./grouped_curves_11.png) | 0.04927 | 0.03982 | 0.77/0.73/0.79 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 164 | `log_pair_reuse` | [keep](./grouped_curves_11.png) | 0.00044 | 0.00989 | 0.44/0.44/0.45 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 165 | `log_peak_window_twoq` | [drop](./grouped_curves_11.png) | 0.00016 | 0.00000 | -0.51/-0.49/-0.62 | Near-perfect rank duplicate of peak_window_twoq (|rho|=1.000) with lower fitted importance. |
| 166 | `log_qasm_bytes` | [keep](./grouped_curves_11.png) | 0.03336 | 0.01627 | 0.79/0.71/0.79 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 167 | `log_random_angle_entropy` | [keep](./grouped_curves_11.png) | 0.00102 | 0.00050 | 0.16/0.16/0.29 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 168 | `log_random_bigram_entropy` | [keep](./grouped_curves_11.png) | 0.00121 | 0.00000 | -0.20/-0.04/0.01 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 169 | `log_random_gate_entropy` | [keep](./grouped_curves_11.png) | 0.00012 | 0.00008 | -0.12/0.08/0.13 | Low fitted importance but a visible within-setting runtime trend; retain pending grouped ablation. |
| 170 | `log_random_pair_entropy` | [keep](./grouped_curves_11.png) | 0.00034 | 0.00000 | -0.11/0.12/0.10 | Low fitted importance but a visible within-setting runtime trend; retain pending grouped ablation. |
| 171 | `log_random_window_entropy` | [drop](./grouped_curves_11.png) | 0.00029 | 0.00000 | 0.31/0.39/0.50 | Near-perfect rank duplicate of random_window_entropy (|rho|=1.000) with lower fitted importance. |
| 172 | `log_randomness_proxy` | [keep](./grouped_curves_11.png) | 0.00048 | 0.00003 | 0.07/0.26/0.35 | Low fitted importance but a visible within-setting runtime trend; retain pending grouped ablation. |
| 173 | `log_reset_early_frac` | [keep](./grouped_curves_11.png) | 0.00241 | 0.06160 | 0.19/0.19/— | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 174 | `log_resets` | [keep](./grouped_curves_11.png) | 0.00206 | 0.15278 | 0.20/0.19/— | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 175 | `log_rotation_folds` | [keep](./grouped_curves_11.png) | 0.00167 | 0.00000 | 0.40/0.05/0.21 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 176 | `log_small_angles` | [keep](./grouped_curves_11.png) | 0.00154 | 0.00000 | 0.56/0.40/0.52 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 177 | `log_two_depth` | [keep](./grouped_curves_12.png) | 0.00164 | 0.00000 | 0.54/0.44/0.43 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 178 | `log_two_q` | [keep](./grouped_curves_12.png) | 0.03245 | 0.03143 | 0.78/0.66/0.77 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 179 | `log_twoq_window_2` | [drop](./grouped_curves_12.png) | 0.00011 | 0.00000 | -0.01/-0.11/0.03 | Near-perfect rank duplicate of twoq_window_2 (|rho|=1.000) with lower fitted importance. |
| 180 | `log_twoq_window_3` | [keep](./grouped_curves_12.png) | 0.00029 | 0.00000 | -0.04/-0.24/-0.11 | Low fitted importance but a visible within-setting runtime trend; retain pending grouped ablation. |
| 181 | `log_twoq_window_4` | [keep](./grouped_curves_12.png) | 0.00007 | 0.00000 | 0.12/0.00/0.20 | Low fitted importance but a visible within-setting runtime trend; retain pending grouped ablation. |
| 182 | `log_twoq_window_6` | [keep](./grouped_curves_12.png) | 0.00000 | 0.00019 | 0.34/0.19/0.31 | Low fitted importance but a visible within-setting runtime trend; retain pending grouped ablation. |
| 183 | `log_twoq_window_7` | [drop](./grouped_curves_12.png) | 0.00182 | 0.00000 | 0.44/0.43/0.55 | Near-perfect rank duplicate of twoq_window_7 (|rho|=1.000) with lower fitted importance. |
| 184 | `log_unique_pairs` | [keep](./grouped_curves_12.png) | 0.00144 | 0.00188 | 0.56/0.41/0.45 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 185 | `log_zero_angles` | [keep](./grouped_curves_12.png) | 0.00502 | 0.00080 | 0.39/-0.04/0.01 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 186 | `threshold` | [drop](./grouped_curves_12.png) | 0.03342 | 0.00000 | —/—/— | Replaced by categorical one-hot setting; the ordinal encoding adds an unjustified distance. |
| 187 | `log_threshold` | [drop](./grouped_curves_12.png) | 0.01989 | 0.00000 | —/—/— | Replaced by categorical one-hot setting; the ordinal encoding adds an unjustified distance. |
| 188 | `chi_walk_cost_overhead` | [keep](./grouped_curves_12.png) | 0.07584 | 0.00421 | 0.75/0.60/0.73 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 189 | `chi_walk_entangling_frac` | [keep](./grouped_curves_12.png) | 0.00502 | 0.00000 | 0.08/0.34/0.45 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 190 | `chi_walk_max_logchi` | [keep](./grouped_curves_12.png) | 0.03627 | 0.00000 | 0.42/0.41/0.66 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 191 | `chi_walk_sat_frac` | [keep](./grouped_curves_12.png) | 0.01031 | 0.00000 | 0.72/0.54/0.72 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 192 | `chi_walk_sat_start` | [keep](./grouped_curves_12.png) | 0.13678 | 0.00944 | -0.73/-0.71/-0.85 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 193 | `chi_walk_links_at_cap` | [keep](./grouped_curves_13.png) | 0.00421 | 0.00000 | 0.60/0.53/0.68 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 194 | `chi_walk_extrapolated` | [keep](./grouped_curves_13.png) | 0.00000 | 0.00001 | 0.23/0.21/0.21 | Low fitted importance but a visible within-setting runtime trend; retain pending grouped ablation. |
| 195 | `chi_walk_available` | [keep](./grouped_curves_13.png) | 0.00000 | 0.00018 | -0.14/-0.14/-0.16 | Low fitted importance but a visible within-setting runtime trend; retain pending grouped ablation. |
| 196 | `chi_walk_rot_near_zero` | [keep](./grouped_curves_13.png) | 0.00054 | 0.00023 | 0.43/0.42/0.61 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 197 | `chi_walk_rot_near_frac` | [keep](./grouped_curves_13.png) | 0.00468 | 0.00000 | 0.20/0.19/0.30 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 198 | `chi_walk_rot_mixing_mean` | [keep](./grouped_curves_13.png) | 0.00404 | 0.00316 | 0.14/0.32/0.36 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 199 | `extended__angle_abs_over_pi_max` | [keep](./grouped_curves_13.png) | 0.00012 | 0.00009 | 0.33/0.35/0.47 | Low fitted importance but a visible within-setting runtime trend; retain pending grouped ablation. |
| 200 | `extended__angle_abs_over_pi_mean` | [keep](./grouped_curves_13.png) | 0.00035 | 0.00000 | 0.31/0.19/0.33 | Low fitted importance but a visible within-setting runtime trend; retain pending grouped ablation. |
| 201 | `extended__angle_abs_over_pi_std` | [keep](./grouped_curves_13.png) | 0.00007 | 0.00000 | 0.27/0.28/0.38 | Low fitted importance but a visible within-setting runtime trend; retain pending grouped ablation. |
| 202 | `extended__angle_clifford_fraction` | [keep](./grouped_curves_13.png) | 0.00396 | 0.00030 | 0.29/0.25/0.30 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 203 | `extended__angle_distinct_count` | [keep](./grouped_curves_13.png) | 0.00126 | 0.01012 | 0.42/0.55/0.61 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 204 | `extended__angle_distinct_overflow` | [keep](./grouped_curves_13.png) | 0.00019 | 0.00594 | 0.14/0.16/0.17 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 205 | `extended__angle_expression_count` | [keep](./grouped_curves_13.png) | 0.00152 | 0.00019 | 0.61/0.60/0.71 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 206 | `extended__angle_generic_fraction` | [keep](./grouped_curves_13.png) | 0.00302 | 0.00388 | 0.18/0.09/0.15 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 207 | `extended__angle_mod_two_pi_distance_mean` | [keep](./grouped_curves_13.png) | 0.00063 | 0.00005 | 0.12/0.28/0.24 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 208 | `extended__angle_negative_fraction` | [keep](./grouped_curves_13.png) | 0.00081 | 0.00001 | -0.14/-0.18/-0.36 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 209 | `extended__angle_numeric_count` | [drop](./grouped_curves_14.png) | 0.00111 | 0.00006 | 0.62/0.60/0.72 | Near-perfect rank duplicate of extended__angle_expression_count (|rho|=1.000) with lower fitted importance. |
| 210 | `extended__angle_numeric_fraction` | [keep](./grouped_curves_14.png) | 0.00074 | 0.00003 | 0.27/-0.00/0.03 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 211 | `extended__angle_quarter_turn_fraction` | [keep](./grouped_curves_14.png) | 0.00790 | 0.00133 | 0.21/0.20/0.22 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 212 | `extended__angle_symbolic_count` | [keep](./grouped_curves_14.png) | 0.00019 | 0.00000 | 0.20/0.51/0.60 | Low fitted importance but a visible within-setting runtime trend; retain pending grouped ablation. |
| 213 | `extended__angle_t_like_fraction` | [keep](./grouped_curves_14.png) | 0.00063 | 0.00064 | 0.10/0.14/0.18 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 214 | `extended__angle_value_entropy` | [keep](./grouped_curves_14.png) | 0.00062 | 0.00211 | 0.27/0.39/0.40 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 215 | `extended__angle_value_entropy_normalized` | [keep](./grouped_curves_14.png) | 0.00070 | 0.00004 | -0.08/-0.15/-0.21 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 216 | `extended__angle_zero_mod_two_pi_fraction` | [keep](./grouped_curves_14.png) | 0.00044 | 0.00159 | 0.33/0.37/0.35 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 217 | `extended__approx_treewidth` | [keep](./grouped_curves_14.png) | 0.00045 | 0.00038 | 0.42/0.18/0.40 | Low fitted importance but a visible within-setting runtime trend; retain pending grouped ablation. |
| 218 | `extended__barrier_count` | [drop](./grouped_curves_14.png) | 0.00025 | 0.00020 | -0.29/-0.21/-0.51 | Near-perfect rank duplicate of log_barriers (|rho|=1.000) with lower fitted importance. |
| 219 | `extended__circuit_depth` | [keep](./grouped_curves_14.png) | 0.00020 | 0.00207 | 0.65/0.54/0.61 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 220 | `extended__classical_register_count` | [keep](./grouped_curves_14.png) | 0.00036 | 0.00013 | -0.37/-0.33/-0.60 | Low fitted importance but a visible within-setting runtime trend; retain pending grouped ablation. |
| 221 | `extended__coarse_unclassified_semicolon_count` | [keep](./grouped_curves_14.png) | 0.00000 | 0.00007 | 0.34/0.30/0.26 | Low fitted importance but a visible within-setting runtime trend; retain pending grouped ablation. |
| 222 | `extended__custom_expanded_gate_count_proxy` | [keep](./grouped_curves_14.png) | 0.00169 | 0.00452 | 0.77/0.73/0.79 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 223 | `extended__custom_expanded_parameter_count_proxy` | [keep](./grouped_curves_14.png) | 0.00057 | 0.00004 | 0.64/0.62/0.74 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 224 | `extended__custom_expanded_two_qubit_fraction` | [keep](./grouped_curves_14.png) | 0.00039 | 0.00150 | -0.03/-0.19/-0.21 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 225 | `extended__custom_expanded_two_qubit_gate_count_proxy` | [keep](./grouped_curves_15.png) | 0.00119 | 0.00161 | 0.74/0.65/0.74 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 226 | `extended__custom_gate_invocation_count` | [keep](./grouped_curves_15.png) | 0.00066 | 0.00460 | -0.13/-0.17/-0.25 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 227 | `extended__cut_crossings_active_fraction` | [keep](./grouped_curves_15.png) | 0.00014 | 0.00000 | -0.03/0.07/0.13 | Low fitted importance but a visible within-setting runtime trend; retain pending grouped ablation. |
| 228 | `extended__cut_crossings_max` | [keep](./grouped_curves_15.png) | 0.00045 | 0.00000 | 0.50/0.29/0.51 | Low fitted importance but a visible within-setting runtime trend; retain pending grouped ablation. |
| 229 | `extended__cut_crossings_max_fraction` | [keep](./grouped_curves_15.png) | 0.00193 | 0.00000 | -0.09/-0.43/-0.31 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 230 | `extended__cut_crossings_mean` | [keep](./grouped_curves_15.png) | 0.00049 | 0.00000 | 0.51/0.32/0.52 | Low fitted importance but a visible within-setting runtime trend; retain pending grouped ablation. |
| 231 | `extended__cut_crossings_std` | [keep](./grouped_curves_15.png) | 0.00090 | 0.00000 | 0.52/0.30/0.52 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 232 | `extended__dag_density` | [keep](./grouped_curves_15.png) | 0.00122 | 0.00000 | -0.72/-0.74/-0.77 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 233 | `extended__dag_dependency_edge_count` | [keep](./grouped_curves_15.png) | 0.00171 | 0.00018 | 0.53/0.51/0.64 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 234 | `extended__dag_layer_utilization` | [keep](./grouped_curves_15.png) | 0.00161 | 0.00118 | 0.31/0.41/0.66 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 235 | `extended__dag_max_layer_width` | [keep](./grouped_curves_15.png) | 0.00101 | 0.00004 | 0.36/0.38/0.42 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 236 | `extended__dag_mean_layer_width` | [keep](./grouped_curves_15.png) | 0.00233 | 0.00017 | 0.22/0.46/0.57 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 237 | `extended__dag_node_count` | [keep](./grouped_curves_15.png) | 0.00097 | 0.00156 | 0.75/0.73/0.79 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 238 | `extended__dag_parallelizable_gate_fraction` | [keep](./grouped_curves_15.png) | 0.00698 | 0.00000 | 0.50/0.67/0.75 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 239 | `extended__dag_width_depth_ratio` | [keep](./grouped_curves_15.png) | 0.00000 | 0.00036 | -0.32/-0.18/-0.17 | Low fitted importance but a visible within-setting runtime trend; retain pending grouped ablation. |
| 240 | `extended__depth_per_gate` | [keep](./grouped_curves_15.png) | 0.00457 | 0.00000 | -0.51/-0.68/-0.75 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 241 | `extended__fast_detail_mode` | [drop](./grouped_curves_16.png) | 0.00000 | 0.00001 | -0.34/-0.30/-0.26 | Near-perfect rank duplicate of extended__coarse_unclassified_semicolon_count (|rho|=1.000) with lower fitted importance. |
| 242 | `extended__fast_detailed_work_units` | [keep](./grouped_curves_16.png) | 0.00129 | 0.00000 | 0.65/0.74/0.58 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 243 | `extended__fast_sampled_operation_count` | [keep](./grouped_curves_16.png) | 0.00069 | 0.00008 | 0.51/0.52/0.62 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 244 | `extended__gate_count__ccx` | [keep](./grouped_curves_16.png) | 0.00039 | 0.01197 | 0.02/-0.03/-0.08 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 245 | `extended__gate_count__cswap` | [drop](./grouped_curves_16.png) | 0.00009 | 0.00000 | -0.06/0.01/0.04 | Tiny fitted importance and no clear within-setting runtime trend in the grouped curves. |
| 246 | `extended__gate_count__cu1` | [keep](./grouped_curves_16.png) | 0.00025 | 0.00000 | 0.07/0.02/0.10 | Small nonzero fitted importance; retain pending grouped ablation because interactions can be marginally flat. |
| 247 | `extended__gate_count__cx` | [keep](./grouped_curves_16.png) | 0.00025 | 0.00000 | 0.49/0.52/0.58 | Low fitted importance but a visible within-setting runtime trend; retain pending grouped ablation. |
| 248 | `extended__gate_count__h` | [keep](./grouped_curves_16.png) | 0.00116 | 0.00000 | 0.35/0.01/-0.04 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 249 | `extended__gate_count__other` | [keep](./grouped_curves_16.png) | 0.00000 | 0.00000 | -0.14/-0.23/-0.26 | Low fitted importance but a visible within-setting runtime trend; retain pending grouped ablation. |
| 250 | `extended__gate_count__ry` | [drop](./grouped_curves_16.png) | 0.00026 | 0.00000 | 0.08/0.42/0.47 | Near-perfect rank duplicate of gate_ry (|rho|=1.000) with lower fitted importance. |
| 251 | `extended__gate_count__rz` | [keep](./grouped_curves_16.png) | 0.00000 | 0.00008 | 0.48/0.44/0.64 | Low fitted importance but a visible within-setting runtime trend; retain pending grouped ablation. |
| 252 | `extended__gate_count__rzz` | [keep](./grouped_curves_16.png) | 0.00072 | 0.00000 | 0.28/0.19/0.30 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 253 | `extended__gate_count__s` | [keep](./grouped_curves_16.png) | 0.00035 | 0.00000 | 0.33/0.57/0.62 | Low fitted importance but a visible within-setting runtime trend; retain pending grouped ablation. |
| 254 | `extended__gate_count__sxdg` | [keep](./grouped_curves_16.png) | 0.00020 | 0.00000 | 0.15/0.44/0.53 | Low fitted importance but a visible within-setting runtime trend; retain pending grouped ablation. |
| 255 | `extended__gate_count__t` | [keep](./grouped_curves_16.png) | 0.00000 | 0.00483 | 0.13/0.09/0.09 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 256 | `extended__gate_count__tdg` | [keep](./grouped_curves_16.png) | 0.00000 | 0.00512 | 0.10/0.06/-0.01 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 257 | `extended__gate_count__u2` | [keep](./grouped_curves_17.png) | 0.00000 | 0.00118 | 0.18/0.43/0.50 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 258 | `extended__gate_count__x` | [keep](./grouped_curves_17.png) | 0.00221 | 0.00142 | 0.03/-0.10/-0.18 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 259 | `extended__gate_count__z` | [keep](./grouped_curves_17.png) | 0.00076 | 0.00000 | 0.11/0.09/0.04 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 260 | `extended__gate_freq__ccx` | [drop](./grouped_curves_17.png) | 0.00186 | 0.00000 | 0.00/-0.05/-0.09 | Near-perfect rank duplicate of extended__gate_count__ccx (|rho|=0.998) with lower fitted importance. |
| 261 | `extended__gate_freq__cp` | [keep](./grouped_curves_17.png) | 0.00012 | 0.00000 | 0.08/-0.07/-0.13 | Low fitted importance but a visible within-setting runtime trend; retain pending grouped ablation. |
| 262 | `extended__gate_freq__cx` | [keep](./grouped_curves_17.png) | 0.00032 | 0.00002 | 0.06/0.08/0.09 | Low fitted importance but a visible within-setting runtime trend; retain pending grouped ablation. |
| 263 | `extended__gate_freq__h` | [keep](./grouped_curves_17.png) | 0.00095 | 0.00008 | 0.10/-0.15/-0.15 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 264 | `extended__gate_freq__p` | [drop](./grouped_curves_17.png) | 0.00011 | 0.00000 | 0.10/0.04/-0.02 | Tiny fitted importance and no clear within-setting runtime trend in the grouped curves. |
| 265 | `extended__gate_freq__rccx` | [drop](./grouped_curves_17.png) | 0.00000 | 0.00013 | 0.01/0.02/-0.00 | Tiny fitted importance and no clear within-setting runtime trend in the grouped curves. |
| 266 | `extended__gate_freq__rx` | [keep](./grouped_curves_17.png) | 0.00031 | 0.00000 | 0.41/0.30/0.63 | Low fitted importance but a visible within-setting runtime trend; retain pending grouped ablation. |
| 267 | `extended__gate_freq__rz` | [keep](./grouped_curves_17.png) | 0.00083 | 0.00000 | 0.33/0.24/0.48 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 268 | `extended__gate_freq__rzz` | [drop](./grouped_curves_17.png) | 0.00005 | 0.00000 | 0.26/0.17/0.29 | Near-perfect rank duplicate of extended__gate_count__rzz (|rho|=0.997) with lower fitted importance. |
| 269 | `extended__gate_freq__s` | [keep](./grouped_curves_17.png) | 0.00000 | 0.00007 | 0.26/0.49/0.54 | Low fitted importance but a visible within-setting runtime trend; retain pending grouped ablation. |
| 270 | `extended__gate_freq__sdg` | [keep](./grouped_curves_17.png) | 0.00057 | 0.00008 | 0.39/0.28/0.30 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 271 | `extended__gate_freq__swap` | [keep](./grouped_curves_17.png) | 0.00021 | 0.00000 | 0.02/-0.12/-0.13 | Low fitted importance but a visible within-setting runtime trend; retain pending grouped ablation. |
| 272 | `extended__gate_freq__t` | [drop](./grouped_curves_17.png) | 0.00000 | 0.00004 | 0.11/0.08/0.08 | Near-perfect rank duplicate of extended__gate_count__t (|rho|=0.998) with lower fitted importance. |
| 273 | `extended__gate_freq__tdg` | [drop](./grouped_curves_18.png) | 0.00000 | 0.00001 | 0.09/0.05/-0.01 | Near-perfect rank duplicate of extended__gate_count__tdg (|rho|=0.999) with lower fitted importance. |
| 274 | `extended__gate_freq__x` | [keep](./grouped_curves_18.png) | 0.00206 | 0.00716 | 0.00/-0.12/-0.20 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 275 | `extended__gate_freq__z` | [drop](./grouped_curves_18.png) | 0.00051 | 0.00004 | 0.11/0.08/0.03 | Near-perfect rank duplicate of extended__gate_count__z (|rho|=1.000) with lower fitted importance. |
| 276 | `extended__gate_type_distinct_count` | [keep](./grouped_curves_18.png) | 0.00000 | 0.00004 | 0.24/0.33/0.34 | Low fitted importance but a visible within-setting runtime trend; retain pending grouped ablation. |
| 277 | `extended__gates_per_qubit` | [keep](./grouped_curves_18.png) | 0.00043 | 0.00000 | 0.66/0.60/0.73 | Low fitted importance but a visible within-setting runtime trend; retain pending grouped ablation. |
| 278 | `extended__interaction_assortativity` | [keep](./grouped_curves_18.png) | 0.00099 | 0.00018 | 0.02/0.38/0.27 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 279 | `extended__interaction_clustering_coefficient` | [keep](./grouped_curves_18.png) | 0.00059 | 0.00000 | 0.20/-0.18/-0.12 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 280 | `extended__interaction_connected_components` | [keep](./grouped_curves_18.png) | 0.00012 | 0.00332 | 0.00/-0.07/-0.19 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 281 | `extended__interaction_cycle_rank_fraction` | [keep](./grouped_curves_18.png) | 0.00093 | 0.00283 | 0.44/0.18/0.37 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 282 | `extended__interaction_diameter` | [keep](./grouped_curves_18.png) | 0.00113 | 0.00002 | -0.10/0.26/0.21 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 283 | `extended__interaction_distinct_pair_count` | [keep](./grouped_curves_18.png) | 0.00048 | 0.00004 | 0.39/0.30/0.36 | Low fitted importance but a visible within-setting runtime trend; retain pending grouped ablation. |
| 284 | `extended__interaction_distinct_pair_fraction` | [keep](./grouped_curves_18.png) | 0.00024 | 0.00000 | -0.01/-0.28/-0.12 | Low fitted importance but a visible within-setting runtime trend; retain pending grouped ablation. |
| 285 | `extended__interaction_edge_weight_max` | [keep](./grouped_curves_18.png) | 0.00000 | 0.00085 | 0.31/0.18/0.39 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 286 | `extended__interaction_edge_weight_mean` | [keep](./grouped_curves_18.png) | 0.00026 | 0.00023 | 0.28/0.30/0.44 | Low fitted importance but a visible within-setting runtime trend; retain pending grouped ablation. |
| 287 | `extended__interaction_edge_weight_std` | [keep](./grouped_curves_18.png) | 0.00022 | 0.00000 | 0.29/0.13/0.27 | Low fitted importance but a visible within-setting runtime trend; retain pending grouped ablation. |
| 288 | `extended__interaction_entropy` | [keep](./grouped_curves_18.png) | 0.00082 | 0.00035 | 0.35/0.39/0.37 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 289 | `extended__interaction_entropy_normalized` | [keep](./grouped_curves_19.png) | 0.00047 | 0.00000 | -0.15/0.10/0.07 | Low fitted importance but a visible within-setting runtime trend; retain pending grouped ablation. |
| 290 | `extended__interaction_largest_component_fraction` | [keep](./grouped_curves_19.png) | 0.00027 | 0.03293 | -0.01/0.07/0.18 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 291 | `extended__interaction_max_edge_weight_fraction` | [keep](./grouped_curves_19.png) | 0.00168 | 0.00213 | -0.43/-0.53/-0.46 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 292 | `extended__interaction_max_weighted_degree` | [keep](./grouped_curves_19.png) | 0.00007 | 0.00000 | 0.42/0.26/0.43 | Low fitted importance but a visible within-setting runtime trend; retain pending grouped ablation. |
| 293 | `extended__interaction_mean_degree` | [keep](./grouped_curves_19.png) | 0.00024 | 0.00000 | 0.39/0.18/0.35 | Low fitted importance but a visible within-setting runtime trend; retain pending grouped ablation. |
| 294 | `extended__interaction_mean_weighted_degree` | [keep](./grouped_curves_19.png) | 0.00019 | 0.00000 | 0.45/0.39/0.55 | Low fitted importance but a visible within-setting runtime trend; retain pending grouped ablation. |
| 295 | `extended__interaction_nontrivial_components` | [keep](./grouped_curves_19.png) | 0.00015 | 0.00001 | -0.10/-0.08/-0.09 | Low fitted importance but a visible within-setting runtime trend; retain pending grouped ablation. |
| 296 | `extended__interaction_spectral_radius` | [keep](./grouped_curves_19.png) | 0.00018 | 0.00000 | 0.47/0.31/0.53 | Low fitted importance but a visible within-setting runtime trend; retain pending grouped ablation. |
| 297 | `extended__interaction_total_count` | [keep](./grouped_curves_19.png) | 0.00123 | 0.00000 | 0.51/0.44/0.60 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 298 | `extended__multi_qubit_gate_count` | [keep](./grouped_curves_19.png) | 0.00000 | 0.00694 | -0.13/-0.19/-0.28 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 299 | `extended__multi_qubit_gate_fraction` | [keep](./grouped_curves_19.png) | 0.00012 | 0.04641 | -0.17/-0.22/-0.31 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 300 | `extended__num_clbits` | [drop](./grouped_curves_19.png) | 0.00079 | 0.00000 | -0.28/-0.19/-0.52 | Near-perfect rank duplicate of log_measurements (|rho|=0.999) with lower fitted importance. |
| 301 | `extended__parameter_count` | [keep](./grouped_curves_19.png) | 0.00089 | 0.00007 | 0.62/0.62/0.73 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 302 | `extended__parameterized_gate_count` | [keep](./grouped_curves_19.png) | 0.00092 | 0.00028 | 0.65/0.61/0.74 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 303 | `extended__qasm_version_3` | [keep](./grouped_curves_19.png) | 0.00006 | 0.00000 | -0.19/-0.18/-0.50 | Low fitted importance but a visible within-setting runtime trend; retain pending grouped ablation. |
| 304 | `extended__quantum_register_count` | [keep](./grouped_curves_19.png) | 0.00130 | 0.00072 | 0.25/-0.18/-0.10 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 305 | `extended__single_qubit_gate_count` | [keep](./grouped_curves_20.png) | 0.00038 | 0.00101 | 0.62/0.65/0.71 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 306 | `extended__source_chars_per_line` | [keep](./grouped_curves_20.png) | 0.00153 | 0.00000 | 0.13/-0.00/0.02 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 307 | `extended__source_conditional_count` | [drop](./grouped_curves_20.png) | 0.00057 | 0.00000 | 0.15/0.08/-0.07 | Near-perfect rank duplicate of log_conditional (|rho|=1.000) with lower fitted importance. |
| 308 | `extended__source_lines` | [keep](./grouped_curves_20.png) | 0.00120 | 0.00000 | 0.77/0.74/0.78 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 309 | `extended__source_pi_token_count` | [keep](./grouped_curves_20.png) | 0.00023 | 0.00000 | 0.55/0.52/0.69 | Low fitted importance but a visible within-setting runtime trend; retain pending grouped ablation. |
| 310 | `extended__source_semicolon_count` | [drop](./grouped_curves_20.png) | 0.00084 | 0.00006 | 0.76/0.73/0.79 | Near-perfect rank duplicate of extended__source_lines (|rho|=0.995) with lower fitted importance. |
| 311 | `extended__structural_gate_entropy_normalized` | [keep](./grouped_curves_20.png) | 0.00015 | 0.00005 | -0.06/0.13/0.17 | Low fitted importance but a visible within-setting runtime trend; retain pending grouped ablation. |
| 312 | `extended__structural_mean_gate_width` | [keep](./grouped_curves_20.png) | 0.00020 | 0.00061 | -0.06/-0.19/-0.20 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 313 | `extended__supermarq_critical_depth` | [keep](./grouped_curves_20.png) | 0.00340 | 0.00250 | -0.48/-0.55/-0.67 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 314 | `extended__supermarq_entanglement_ratio` | [keep](./grouped_curves_20.png) | 0.00000 | 0.00053 | 0.06/-0.08/-0.04 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 315 | `extended__supermarq_liveness` | [keep](./grouped_curves_20.png) | 0.00038 | 0.00000 | 0.28/0.27/0.49 | Low fitted importance but a visible within-setting runtime trend; retain pending grouped ablation. |
| 316 | `extended__supermarq_measurement` | [keep](./grouped_curves_20.png) | 0.00091 | 0.03604 | 0.19/0.19/— | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 317 | `extended__supermarq_parallelism` | [keep](./grouped_curves_20.png) | 0.00090 | 0.00000 | 0.32/0.49/0.67 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 318 | `extended__total_gate_count` | [drop](./grouped_curves_20.png) | 0.00118 | 0.00034 | 0.74/0.73/0.79 | Near-perfect rank duplicate of extended__total_operation_count (|rho|=0.998) with lower fitted importance. |
| 319 | `extended__total_operation_count` | [keep](./grouped_curves_20.png) | 0.00129 | 0.00161 | 0.75/0.73/0.79 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 320 | `extended__treewidth_min_degree` | [drop](./grouped_curves_20.png) | 0.00029 | 0.00000 | 0.42/0.17/0.40 | Near-perfect rank duplicate of extended__approx_treewidth (|rho|=0.999) with lower fitted importance. |
| 321 | `extended__treewidth_min_fill` | [keep](./grouped_curves_21.png) | 0.00110 | 0.00000 | 0.44/0.24/0.39 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 322 | `extended__two_qubit_angle_abs_over_pi_mean` | [keep](./grouped_curves_21.png) | 0.00108 | 0.00000 | 0.17/-0.01/-0.02 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
| 323 | `extended__two_qubit_angle_clifford_fraction` | [keep](./grouped_curves_21.png) | 0.00009 | 0.00033 | 0.12/-0.08/-0.15 | Low fitted importance but a visible within-setting runtime trend; retain pending grouped ablation. |
| 324 | `extended__two_qubit_angle_numeric_count` | [keep](./grouped_curves_21.png) | 0.00042 | 0.00000 | 0.22/0.01/0.02 | Low fitted importance but a visible within-setting runtime trend; retain pending grouped ablation. |
| 325 | `extended__two_qubit_gate_count` | [keep](./grouped_curves_21.png) | 0.00139 | 0.00063 | 0.72/0.63/0.75 | Material fitted runtime or timeout importance; grouped curve supplies marginal context. |
