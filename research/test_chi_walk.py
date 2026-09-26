"""Physics and parser checks for the user-supplied chi-walk proxy."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'quantathon-harness'))
from chi_walk import MODEL_FEATURE_NAMES, model_features, price, select_model_features, walk
from model import RuntimeModel


class ChiWalkTests(unittest.TestCase):
    def test_classical_control_and_diagonal_gate_do_not_entangle(self):
        prefix='OPENQASM 2.0; qreg q[2]; '
        classical=walk(prefix+'x q[0]; cx q[0],q[1];')
        one_quantum=walk(prefix+'h q[0]; cz q[0],q[1];')
        both_quantum=walk(prefix+'h q[0]; h q[1]; cz q[0],q[1];')
        self.assertEqual(classical['n_2q_entangling'],0)
        self.assertEqual(one_quantum['n_2q_entangling'],0)
        self.assertEqual(both_quantum['n_2q_entangling'],1)
        self.assertEqual(both_quantum['tables']['16']['max_logchi'],1)
        self.assertGreater(price(both_quantum,10,100,2.5)['16'],
                           price(one_quantum,10,100,2.5)['16'])

    def test_custom_gate_matches_its_expansion(self):
        expanded='OPENQASM 2.0; qreg q[2]; h q[0]; cx q[0],q[1];'
        custom='OPENQASM 2.0; gate pair a,b { h a; cx a,b; } qreg q[2]; pair q[0],q[1];'
        self.assertEqual(walk(expanded)['tables'],walk(custom)['tables'])

    def test_near_zero_and_pi_rotations_preserve_basis_flag(self):
        prefix='OPENQASM 2.0; qreg q[2]; '
        for rotation in ('rx(0)', 'rx(pi)', 'rx(2*pi)', 'ry(-pi)',
                         'rx(3)', 'rx(6)', 'u3(pi,0.2,0.3)'):
            with self.subTest(rotation=rotation):
                w=walk(prefix+f'{rotation} q[0]; h q[1]; cz q[0],q[1];')
                self.assertEqual(w['n_2q_entangling'],0)
                self.assertEqual(w['tables']['16']['max_logchi'],0)
                self.assertEqual(w['n_rotation_near_zero']+w['n_rotation_near_pi'],1)
        for rotation in ('rx(pi/2)', 'ry(0.31)', 'rx(theta)'):
            with self.subTest(rotation=rotation):
                w=walk(prefix+f'{rotation} q[0]; h q[1]; cz q[0],q[1];')
                self.assertEqual(w['n_2q_entangling'],1)

    def test_rotation_filter_retains_quantum_state_and_custom_angles(self):
        prefix='OPENQASM 2.0; qreg q[2]; '
        already_quantum=walk(prefix+'h q[0]; rx(pi) q[0]; h q[1]; cz q[0],q[1];')
        self.assertEqual(already_quantum['n_2q_entangling'],1)
        expanded=walk(prefix+'rx(pi) q[0]; h q[1]; cz q[0],q[1];')
        custom=walk('OPENQASM 2.0; gate near a { rx(pi) a; } qreg q[2]; '
                    'near q[0]; h q[1]; cz q[0],q[1];')
        self.assertEqual(expanded['tables'],custom['tables'])
        self.assertEqual(expanded['n_rotation_near_pi'],custom['n_rotation_near_pi'])

    def test_model_exposes_near_zero_and_pi_counts(self):
        model=RuntimeModel()
        qasm='OPENQASM 2.0;\nqreg q[2];\nrx(0) q[0];\nry(pi) q[1];\n'
        out=model.featurize(qasm)
        self.assertEqual(out['chi_walk_rot_near_zero'],1)
        self.assertEqual(out['chi_walk_rot_near_pi'],1)
        self.assertEqual(out['chi_walk_rot_near_frac'],1)
        self.assertGreater(model.predict(out,64),0)

    def test_cost_non_decreasing_with_threshold(self):
        qasm='OPENQASM 2.0; qreg q[4]; h q[0]; h q[1]; h q[2]; h q[3]; '
        qasm+='; '.join('cx q[0],q[3]' for _ in range(8))+';'
        w=walk(qasm)
        values=price(w,10,100,2.5)
        self.assertLessEqual(values['16'],values['64'])
        self.assertLessEqual(values['64'],values['512'])
        self.assertLessEqual(values['512'],values['uncapped'])

    def test_model_features_select_correct_threshold(self):
        qasm='OPENQASM 2.0; qreg q[4]; h q[0]; h q[1]; cx q[0],q[1];'
        w=walk(qasm)
        flat=model_features(w)
        for threshold in (16,64,512):
            selected=select_model_features(flat,threshold)
            self.assertEqual(set(selected),set(MODEL_FEATURE_NAMES))
            self.assertEqual(selected['chi_walk_max_logchi'],
                             w['tables'][str(threshold)]['max_logchi'])
            self.assertAlmostEqual(selected['chi_walk_cost_overhead'],
                                   price(w,10,100,2.5)[str(threshold)])

    def test_model_parser_adds_chi_walk_for_trained_artifact(self):
        qasm='OPENQASM 2.0;\nqreg q[2];\nh q[0];\ncx q[0],q[1];\n'
        model=RuntimeModel()
        model.use_chi_walk=True
        extracted=model.featurize(qasm)
        expected=model_features(walk(qasm,classical_tracking=True,budget_s=3.0))
        for name,value in expected.items():
            self.assertAlmostEqual(extracted[name],value)


if __name__=='__main__':
    unittest.main()
