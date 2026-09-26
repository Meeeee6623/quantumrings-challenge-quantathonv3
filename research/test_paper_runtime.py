"""Checks for the runtime-only paper architecture and grouped fit inputs."""
import importlib.util
import unittest

import numpy as np


@unittest.skipUnless(importlib.util.find_spec('torch'),
                     'install the paper extra to test neural architecture')
class PaperRuntimeTests(unittest.TestCase):
    def test_feature_shape_and_threshold_input(self):
        from paper_runtime_replication import paper_features
        base = {'n_qubits': 4, 'depth': 10, 'ops': 20, 'two_q': 5,
                'fingerprint_qft': .8}
        low = paper_features(base, {'gate_count__h': 3}, 16)
        high = paper_features(base, {'gate_count__h': 3}, 512)
        self.assertEqual(low.shape, (33,))
        self.assertEqual(low[32], 4)
        self.assertEqual(high[32], 9)
        self.assertAlmostEqual(float(low[3]), float(np.log1p(3)))
        self.assertEqual(low[25], .8)

    def test_family_residual_begins_as_shared_backbone(self):
        import torch
        from paper_runtime_replication import PaperRuntimeNet
        torch.manual_seed(17)
        plain = PaperRuntimeNet(33, 8, 'plain').eval()
        torch.manual_seed(17)
        family = PaperRuntimeNet(33, 8, 'family').eval()
        x = torch.randn(4, 33)
        labels = torch.tensor([0, 1, 2, 7])
        with torch.no_grad():
            np.testing.assert_allclose(plain(x, labels).numpy(),
                                       family(x, labels).numpy(),
                                       rtol=1e-6, atol=1e-6)
        self.assertTrue(torch.all(family.family_residual.weight == 0))
        self.assertTrue(torch.all(family.gamma.weight == 0))

    def test_inner_validation_keeps_circuits_together(self):
        from paper_runtime_replication import inner_split
        names = np.asarray(['a', 'a', 'b', 'b', 'c', 'c', 'd', 'd',
                            'e', 'e', 'f', 'f', 'g', 'g', 'h', 'h'])
        fit, valid = inner_split(np.arange(len(names)), names, 17)
        self.assertFalse(set(names[fit]) & set(names[valid]))
        self.assertEqual(set(fit) | set(valid), set(range(len(names))))

    def test_timeout_diagnostics_use_observed_cap(self):
        from paper_runtime_replication import evaluate
        actual = np.log10([14400.0, 100.0])
        result = evaluate(actual, np.asarray([1e7, 100.0]),
                          np.asarray([True, False]), np.asarray([16, 16]))
        self.assertAlmostEqual(result['r2_seconds'], 1.0)
        self.assertAlmostEqual(result['r2_log10'], 1.0)


if __name__ == '__main__':
    unittest.main()
