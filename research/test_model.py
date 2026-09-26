"""Focused parser and prediction checks for syntax present in the challenge."""
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'quantathon-harness'))
from model import RuntimeModel, estimated_log_chi
from extended_features import _coarse_prefix_counts
from runtime_floors import apply_floor, eligible, large_work_floor, runtime_floor
from template_analogues import blend_with_analogues, signature
from rotation_calibration import calibrate_near_basis_runtime


class FeatureParserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = RuntimeModel()

    def test_coarse_gate_prefixes_distinguish_bodies_and_longer_names(self):
        top, definitions = _coarse_prefix_counts('''U(0,0,0) q[0];
s q[0];
sdg q[0];
swap q[0],q[1];
mcphase(pi) q[0],q[1],q[2];
pragma foo;
gate pair a,b {
  h a;
  cx a,b;
}
''')
        self.assertEqual(top['u'],1)
        self.assertEqual((top['s'],top['sdg'],top['swap']), (1,1,1))
        self.assertEqual((top['mcp'],top['mcphase'],top['p']), (0,1,0))
        self.assertEqual((definitions['h'],definitions['cx']), (1,1))

    def test_qasm2_measurement_and_entangling_gate(self):
        f = self.model.featurize('''OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0],q[1];
measure q[0] -> c[0];
''')
        self.assertEqual((f['n_qubits'],f['ops'],f['two_q'],f['measurements']), (2,2,1,1))
        self.assertEqual(f['unsupported_statements'],0)

    def test_qasm3_multiple_registers_and_spaced_parameters(self):
        f = self.model.featurize('''OPENQASM 3.0;
include "stdgates.inc";
qubit[2] node;
qubit[1] coin;
bit[1] meas;
U(pi/2, 0, 0) coin[0];
u2(0, 0) node[0];
cx coin[0], node[1];
meas[0] = measure node[1];
if (meas[0]) {
  p(pi/4) node[0];
}
''')
        self.assertEqual((f['n_qubits'],f['ops'],f['two_q'],f['measurements']), (3,4,1,1))
        self.assertEqual(f['conditional'],1)
        self.assertEqual(f['unsupported_statements'],0)

    def test_custom_definition_only_contributes_when_called(self):
        f = self.model.featurize('''OPENQASM 3.0;
gate pair a,b {
  h a;
  cx a,b;
}
qubit[2] q;
pair q[0],q[1];
''')
        self.assertEqual(f['custom_definitions'],1)
        self.assertEqual(f['custom_calls'],1)
        self.assertEqual((f['ops'],f['two_q'],f['gate_h']), (2,1,1))

    def test_nested_custom_gates_and_mixed_case_calls(self):
        f = self.model.featurize('''OPENQASM 3.0;
gate phase a,b {
  cp(pi/4) a,b;
}
gate cMAJ a,b {
  phase a,b;
  cx a,b;
}
qubit[2] q;
cMAJ q[0],q[1];
''')
        self.assertEqual((f['ops'],f['two_q'],f['gate_cp'],f['gate_cx']), (2,2,1,1))
        self.assertEqual(f['unsupported_statements'],0)

    def test_predict_is_finite_and_positive(self):
        f = self.model.featurize('OPENQASM 2.0;\nqreg q[1];\nh q[0];')
        for threshold in (16,64,512):
            p = self.model.predict(f,threshold)
            self.assertTrue(math.isfinite(p) and p > 0)

    def test_submission_artifact_uses_threshold_experts(self):
        self.assertEqual(self.model.model['artifact_version'],
                         'pruned_union_threshold_experts_v7')
        self.assertEqual(set(self.model.model['threshold_specialists']),{16,64,512})
        self.assertEqual(set(self.model.model['timeout_classifiers']),{16,64,512})
        self.assertAlmostEqual(self.model.model['threshold_specialist_weight'],.5)
        self.assertAlmostEqual(self.model.model['timeout_probability_cutoff'],.35)
        self.assertEqual(len(self.model.model['global_columns']),120)
        self.assertEqual({k:len(v) for k,v in
                          self.model.model['specialist_columns_by_threshold'].items()},
                         {16:200,64:200,512:200})
        self.assertEqual({k:len(v) for k,v in
                          self.model.model['classifier_columns_by_threshold'].items()},
                         {16:80,64:80,512:80})
        self.assertTrue(any(column.startswith('extended__') for columns in
                            self.model.model['specialist_columns_by_threshold'].values()
                            for column in columns))
        self.assertEqual(len(self.model.model['reset_family_references']),14)
        self.assertTrue(self.model.model['template_analogue_bank'])

    def test_template_blend_needs_two_same_threshold_analogues(self):
        features={'n_qubits':4,'ops':100,'two_q':40,'multi_q':0}
        key=(16,*signature(features))
        self.assertEqual(blend_with_analogues(4.0,features,16,{key:(2.0,)}),4.0)
        self.assertEqual(blend_with_analogues(4.0,features,64,{key:(2.0,2.0)}),4.0)
        self.assertEqual(blend_with_analogues(4.0,features,16,{key:(2.0,2.0)}),20.0)
        self.assertEqual(blend_with_analogues(4.0,features,16,{key:(2.0,2.0,2.0)}),100.0)

    def test_near_basis_calibration_has_threshold_and_work_guards(self):
        features={'chi_walk_rot_near_frac':1.0,'ops':2000}
        self.assertEqual(calibrate_near_basis_runtime(10.0,features,512),5.0)
        self.assertEqual(calibrate_near_basis_runtime(10.0,features,16),10.0)
        self.assertEqual(calibrate_near_basis_runtime(.5,features,512),.5)
        self.assertEqual(calibrate_near_basis_runtime(14400.0,features,512),14400.0)
        features['ops']=999
        self.assertEqual(calibrate_near_basis_runtime(10.0,features,512),10.0)

    def test_reset_family_floor_is_narrow_and_threshold_specific(self):
        features = {'resets':65,'multi_q':130,'fingerprint_grover':1.0,'ops':500}
        references = [(16,1000,100.0),(64,1000,10.0)]
        self.assertTrue(eligible(features))
        self.assertEqual(runtime_floor(features,16,references),50.0)
        self.assertEqual(runtime_floor(features,64,references),5.0)
        self.assertEqual(apply_floor(60.0,features,16,references),60.0)
        self.assertEqual(apply_floor(1.0,features,16,references),50.0)
        self.assertEqual(runtime_floor(features,512,references),0.0)
        features['resets'] = 0
        self.assertFalse(eligible(features))
        self.assertEqual(runtime_floor(features,16,references),0.0)

    def test_large_work_floor_requires_half_million_effective_operations(self):
        self.assertEqual(large_work_floor({'effective_ops':499_999}),0.0)
        self.assertEqual(large_work_floor({'effective_ops':500_000}),10.0)
        self.assertEqual(large_work_floor({'effective_ops':999_999}),10.0)
        self.assertEqual(large_work_floor({'effective_ops':1_000_000}),100.0)
        fallback = RuntimeModel()
        fallback.model = None
        self.assertEqual(fallback.predict({'n_qubits':1,'effective_ops':1_000_000},16),100.0)

    def test_unknown_threshold_falls_back_to_global_model(self):
        f = self.model.featurize('OPENQASM 2.0;\nqreg q[1];\nh q[0];')
        prediction = self.model.predict(f,32)
        self.assertTrue(math.isfinite(prediction) and prediction > 0)

    def test_prediction_interval_contains_point(self):
        f = self.model.featurize('OPENQASM 2.0;\nqreg q[2];\nh q[0];\ncx q[0],q[1];')
        point,lower,upper = self.model.predict_interval(f,64)
        self.assertLessEqual(lower,point)
        self.assertLessEqual(point,upper)
        self.assertGreater(lower,0)

    def test_union_exposes_graph_dag_and_angle_features(self):
        f = self.model.featurize('''OPENQASM 2.0;
qreg q[3];
h q[0];
rx(pi/7) q[1];
cx q[0],q[1];
cz q[1],q[2];
''')
        self.assertEqual(f['extended__interaction_edge_count'],2)
        self.assertGreaterEqual(f['extended__dag_critical_path_length'],3)
        self.assertGreater(f['extended__angle_generic_fraction'],0)
        self.assertEqual(f['extended__fast_detail_mode'],1)

    def test_union_qasm3_registers_and_operations(self):
        f = self.model.featurize('''OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
bit[2] c;
h q[0];
cx q[0], q[1];
c[0] = measure q[0];
''')
        self.assertEqual(f['extended__qasm_version_3'],1)
        self.assertEqual(f['extended__num_qubits'],2)
        self.assertEqual(f['extended__num_clbits'],2)
        self.assertEqual(f['extended__measurement_count'],1)

    def test_chi_cut_bound_distinguishes_controlled_and_generic_gates(self):
        controlled = self.model.featurize('OPENQASM 2.0;\nqreg q[4];\ncx q[1],q[2];')
        generic = self.model.featurize('OPENQASM 2.0;\nqreg q[4];\nswap q[1],q[2];')
        self.assertEqual((controlled['chi_upper_mid'], generic['chi_upper_mid']), (1, 2))
        self.assertEqual(controlled['chi_upper_peak'], 1)

    def test_chi_bound_includes_custom_gate_at_called_operands(self):
        f = self.model.featurize('''OPENQASM 2.0;
gate pair a,b {
  cx a,b;
}
qreg q[4];
pair q[1],q[2];
''')
        self.assertEqual(f['chi_upper_mid'], 1)

    def test_randomness_proxy_responds_to_gate_and_angle_variety(self):
        regular = self.model.featurize('''OPENQASM 2.0;
qreg q[3];
rx(0) q[0];
rx(0) q[1];
cx q[0],q[1];
cx q[0],q[1];
''')
        varied = self.model.featurize('''OPENQASM 2.0;
qreg q[3];
rx(pi/7) q[0];
ry(pi/5) q[1];
cx q[0],q[1];
cz q[1],q[2];
''')
        self.assertGreater(varied['randomness_proxy'], regular['randomness_proxy'])
        self.assertGreater(varied['chi_random_pressure'], regular['chi_random_pressure'])

    def test_effective_chi_approaches_the_upper_bound_with_randomness(self):
        upper_log2 = 12
        estimates = [estimated_log_chi(upper_log2,r) for r in (0,.25,.5,.75,1)]
        self.assertEqual(estimates, [0,3,6,9,12])
        self.assertEqual([2**x for x in estimates], [1,8,64,512,4096])
        f = self.model.featurize('OPENQASM 2.0;\nqreg q[4];\ncx q[1],q[2];\nh q[0];')
        self.assertAlmostEqual(f['chi_est_log2_peak'],
                               f['chi_upper_peak']*f['randomness_proxy'])
        self.assertLessEqual(f['chi_est_log2_peak'],f['chi_upper_peak'])

    def test_cut_timeline_distinguishes_early_and_late_growth(self):
        prefix = 'OPENQASM 2.0;\nqreg q[4];\n'
        filler = 'x q[0];\n'*80
        early = self.model.featurize(prefix+'cx q[1],q[2];\n'+filler)
        late = self.model.featurize(prefix+filler+'cx q[1],q[2];\n')
        self.assertEqual(early['chi_upper_mid'],late['chi_upper_mid'])
        self.assertGreater(early['chi_timeline_mid_auc'],late['chi_timeline_mid_auc'])
        self.assertLess(early['chi_timeline_mid_t50'],late['chi_timeline_mid_t50'])

    def test_reordered_geometry_recovers_scrambled_chain(self):
        prefix = 'OPENQASM 2.0;\nqreg q[6];\n'
        edges = ((0,5),(5,1),(1,4),(4,2),(2,3))
        f = self.model.featurize(prefix+''.join(f'cx q[{a}],q[{b}];\n' for a,b in edges))
        self.assertGreater(f['graph_cutwidth_original'],f['graph_cutwidth_rcm'])
        self.assertEqual(f['graph_cutwidth_rcm'],1)
        self.assertEqual(f['graph_mindegree_width'],1)

    def test_algorithm_fingerprints_and_dense_size(self):
        qft = self.model.featurize('''OPENQASM 2.0;
qreg q[3];
h q[0];
cp(pi/2) q[1],q[0];
cp(pi/4) q[2],q[0];
h q[1];
cp(pi/2) q[2],q[1];
h q[2];
''')
        ghz = self.model.featurize('''OPENQASM 2.0;
qreg q[3];
h q[0];
cx q[0],q[1];
cx q[0],q[2];
''')
        self.assertGreater(qft['fingerprint_qft'],ghz['fingerprint_qft'])
        self.assertGreater(ghz['fingerprint_ghz'],qft['fingerprint_ghz'])
        dense = self.model.featurize('OPENQASM 2.0;\nqreg q[34];\nh q[0];')
        self.assertEqual(dense['dense_log2_bytes'],37)
        self.assertEqual(dense['dense_over_128gb'],1)
        self.assertEqual(dense['component_dense_over_128gb'],0)

    def test_ordered_qaoa_motif_survives_gate_decomposition(self):
        native = '''OPENQASM 2.0;
qreg q[2];
h q[0]; h q[1];
rzz(pi/3) q[0],q[1];
rx(pi/4) q[0]; rx(pi/4) q[1];
rzz(pi/5) q[0],q[1];
rx(pi/6) q[0]; rx(pi/6) q[1];
'''.replace('; ', ';\n')
        compiled = '''OPENQASM 2.0;
qreg q[2];
u(pi/2,0,pi) q[0];
u(pi/2,0,pi) q[1];
cx q[0],q[1]; u(0,0,pi/3) q[1]; cx q[0],q[1];
u(pi/4,-pi/2,pi/2) q[0]; u(pi/4,-pi/2,pi/2) q[1];
cx q[0],q[1]; u(0,0,pi/5) q[1]; cx q[0],q[1];
u(pi/6,-pi/2,pi/2) q[0]; u(pi/6,-pi/2,pi/2) q[1];
'''.replace('; ', ';\n')
        for qasm in (native,compiled):
            f=self.model.featurize(qasm,include_sequence=True)
            self.assertEqual(f['motif_qaoa_score'],1.0)
            self.assertEqual(f['unsupported_statements'],0)
        plain=self.model.featurize(native)
        self.assertNotIn('motif_qaoa_score',plain)


if __name__ == '__main__':
    unittest.main()
