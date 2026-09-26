"""Focused parser and prediction checks for syntax present in the challenge."""
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'quantathon-harness'))
from model import RuntimeModel, estimated_log_chi


class FeatureParserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = RuntimeModel()

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
                         'full_union_threshold_experts_v1')
        self.assertEqual(set(self.model.model['threshold_specialists']),{16,64,512})
        self.assertEqual(set(self.model.model['timeout_classifiers']),{16,64,512})
        self.assertAlmostEqual(self.model.model['threshold_specialist_weight'],.5)
        self.assertAlmostEqual(self.model.model['timeout_probability_cutoff'],.35)
        self.assertTrue(any(column.startswith('extended__')
                            for column in self.model.model['specialist_columns']))

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
