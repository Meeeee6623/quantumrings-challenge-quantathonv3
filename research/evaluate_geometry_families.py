"""Paired ablation of temporal, graph, family, and ancillary QASM features."""
import csv
import hashlib
import json
import math

import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from evaluate_chi_randomness import NEW as PREVIOUS_ABLATION, boot_delta
from train_runtime import ROOT, CAP, candidates, columns_for, matrix, score


TIMELINE = {"chi_timeline_peak_auc", "chi_timeline_capacity_auc",
            "chi_timeline_mid_auc", "chi_timeline_mid_t50", "chi_timeline_mid_t90",
            "chi_timeline_post_mid_sat_twoq"}
GEOMETRY = {"graph_cutwidth_original", "graph_cutwidth_rcm", "graph_mean_cut_original",
            "graph_mean_cut_rcm", "graph_span_rcm", "graph_rcm_gain",
            "graph_degree_entropy", "graph_density", "graph_mindegree_width",
            "graph_mindegree_available"}
FAMILY = {"fingerprint_qft", "fingerprint_random_grid", "fingerprint_qaoa",
          "fingerprint_arithmetic", "fingerprint_variational", "fingerprint_ghz",
          "fingerprint_graph_state", "fingerprint_grover", "fingerprint_phase_dyadic"}
MISC = {"dense_log2_bytes", "component_dense_log2_bytes", "dense_over_128gb",
        "component_dense_over_128gb", "diagonal_run_max", "diagonal_run_frac",
        "liveness_peak_window", "liveness_mean_window", "liveness_mean_span",
        "measurement_early_frac", "reset_early_frac"}
ADDED = TIMELINE | GEOMETRY | FAMILY | MISC
VIEWS = {"current_single":set(), "timeline":TIMELINE, "geometry":GEOMETRY,
         "family":FAMILY, "misc":MISC,
         "timeline_geometry":TIMELINE | GEOMETRY,
         "all_new":ADDED,
         "all_new_effective_peak":ADDED}


def main():
    feats = json.loads((ROOT/'research/features.json').read_text())
    assert len(feats) == 532
    assert all(ADDED <= set(f) or f.get('huge_fast_path') for f in feats.values())
    with (ROOT/'runtime-data.csv').open(newline='') as file:
        rows = [r for r in csv.DictReader(file) if r['filename'] in feats]
    names = np.array([r['filename'] for r in rows])
    y = np.array([math.log10(CAP if r['status']=='timeout' else float(r['duration_s'])) for r in rows])
    timeout = np.array([r['status']=='timeout' for r in rows])
    # Match the prior χ evaluation's folds exactly for a paired comparison.
    signatures = {name:hashlib.sha1(json.dumps(
        {k:v for k,v in f.items() if k not in ADDED | PREVIOUS_ABLATION},
        sort_keys=True).encode()).hexdigest() for name,f in feats.items()}
    groups = np.array([signatures[name] for name in names])
    normal = list(GroupKFold(n_splits=5).split(np.zeros(len(rows)),y,groups))
    circuit_names = sorted(feats)
    cluster_keys = ('n_qubits','ops','two_q_ratio','nonclifford_ratio',
                    'conditional','custom_definitions','qasm_bytes')
    C = np.array([[math.log1p(max(0,float(feats[name].get(k,0))))
                   for k in cluster_keys] for name in circuit_names])
    cluster_ids = KMeans(n_clusters=12,n_init=10,random_state=17).fit_predict(
        StandardScaler().fit_transform(C))
    stress_lookup = dict(zip(circuit_names,cluster_ids))
    stress = list(GroupKFold(n_splits=5).split(np.zeros(len(rows)),y,
                 np.array([stress_lookup[name] for name in names])))

    all_columns = columns_for(feats,'all')
    blocked_base = {"chi_random_peak", "chi_random_pressure", "chi_est_log2_peak",
                    "chi_est_log2_mid", "chi_est_log2_mean", "chi_est_capacity_fraction"}
    results = []
    predictions = {}
    for view,extra in VIEWS.items():
        retained_effective = ({'chi_est_log2_peak','chi_est_capacity_fraction'}
                              if view == 'all_new_effective_peak' else set())
        excluded = (ADDED-extra) | (blocked_base-retained_effective)
        cols = [c for c in all_columns if (c[4:] if c.startswith('log_') and c != 'log_threshold'
                                       else c) not in excluded]
        X = matrix(rows,feats,cols)
        scored = {}
        predictions[view] = {}
        for split_name,splits in (('circuit',normal),('structural',stress)):
            pred = np.zeros(len(rows))
            for train_idx,test_idx in splits:
                estimator = candidates()['extra_trees']()
                estimator.fit(X[train_idx],y[train_idx])
                pred[test_idx] = estimator.predict(X[test_idx])
            predictions[view][split_name] = pred
            s = score(y,pred,timeout)
            scored[split_name] = {'score':float(s.mean()),
                                  'timeout_score':float(s[timeout].mean()),
                                  'median_factor_error':float(10**np.median(abs(pred-y)))}
        row = {'view':view,'features':sorted(extra),'columns':len(cols),**scored}
        results.append(row)
        print(json.dumps(row),flush=True)

    # Existing fitted blend is a stronger reference than its single component.
    old = {}
    with (ROOT/'research/chi_randomness_oof.csv').open(newline='') as file:
        for r in csv.DictReader(file):
            old[(r['filename'],r['threshold'])] = r
    reference = {}
    for split_name in ('circuit','structural'):
        reference[split_name] = np.array([
            .75*math.log10(float(old[(r['filename'],r['threshold'])][f'chi_plus_random_{split_name}_pred_s']))
            +.25*math.log10(float(old[(r['filename'],r['threshold'])][f'chi_plus_effective_peak_{split_name}_pred_s']))
            for r in rows])
    comparison = []
    for row in results:
        view = row['view']
        item = {'view':view}
        for split_name in ('circuit','structural'):
            base_score = score(y,reference[split_name],timeout)
            variant_score = score(y,predictions[view][split_name],timeout)
            item[split_name+'_delta_vs_current_blend'] = float(np.mean(variant_score-base_score))
            item[split_name+'_bootstrap_95'] = boot_delta(base_score,variant_score,names)
        comparison.append(item)
        print(json.dumps(item),flush=True)
    report = {'rows':len(rows),'circuits':len(feats),'model':'extra_trees',
              'results':results,'vs_current_blend':comparison,
              'reference':'research/chi_randomness_oof.csv',
              'scope':'Circuit-grouped and structural-cluster training-data holdouts; no hidden-circuit claim.'}
    (ROOT/'research/geometry_family_ablation.json').write_text(json.dumps(report,indent=2)+'\n')
    with (ROOT/'research/geometry_family_oof.csv').open('w',newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['filename','threshold','status','actual_s']
                        +[f'{view}_{split}_pred_s' for split in ('circuit','structural')
                                                    for view in VIEWS])
        for i,r in enumerate(rows):
            writer.writerow([r['filename'],r['threshold'],r['status'],10**y[i]]
                            +[10**predictions[view][split][i]
                              for split in ('circuit','structural') for view in VIEWS])


if __name__=='__main__':
    main()
