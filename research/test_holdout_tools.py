"""Coverage and inspection safeguards for holdout preparation tools."""
import csv
from pathlib import Path
import tempfile
import unittest

from inspect_holdout import circuit_files, qasm_name, training_outcome
from validate_submission import validate


class HoldoutToolTests(unittest.TestCase):
    def test_circuit_discovery_and_duplicate_basename_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'a.qasm').write_text('OPENQASM 2.0;\n')
            (root / 'b.qasm.zst').write_bytes(b'fixture')
            (root / 'notes.txt').write_text('ignored')
            self.assertEqual([qasm_name(path) for path in circuit_files(root)],
                             ['a.qasm', 'b.qasm'])
            (root / 'a.qasm.zst').write_bytes(b'fixture')
            with self.assertRaisesRegex(ValueError, 'Duplicate QASM basenames'):
                circuit_files(root)

    def test_validator_requires_every_pair_and_valid_numbers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'a.qasm').write_text('OPENQASM 2.0;\n')
            submission = root / 'submission.csv'
            fields = ['team', 'filename', 'threshold', 'pred_duration_s',
                      'parse_s', 'predict_s']
            with submission.open('w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=fields)
                writer.writeheader()
                writer.writerow(dict(team='T', filename='a.qasm', threshold=16,
                                     pred_duration_s=2, parse_s=.1, predict_s=.1))
                writer.writerow(dict(team='T', filename='a.qasm', threshold=64,
                                     pred_duration_s=20000, parse_s=.1, predict_s=.1))
            result = validate(root, submission, [16, 64])
            self.assertTrue(result['valid'], result)
            self.assertEqual(result['expected_rows'], 2)
            with submission.open('a', newline='') as file:
                csv.writer(file).writerow(['T', 'a.qasm', 64, 'nan', .1, 16])
            result = validate(root, submission, [16, 64])
            self.assertFalse(result['valid'])
            self.assertTrue(any('Invalid numeric value' in issue
                                for issue in result['issues']))
            with submission.open('w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=fields)
                writer.writeheader()
                writer.writerow(dict(team='T', filename='a.qasm', threshold=16,
                                     pred_duration_s=2, parse_s=.1, predict_s=.1))
            result = validate(root, submission, [16, 64])
            self.assertFalse(result['valid'])
            self.assertTrue(any('Missing predictions (1)' in issue
                                for issue in result['issues']))

    def test_neighbor_labels_use_grouped_oof_factor(self):
        row = {'actual_s': '10', 'matched_pred_s': '40',
               'status': 'success'}
        outcome = training_outcome({('x.qasm', 16): row}, 'x.qasm', 16)
        self.assertEqual(outcome['label'], 'missed')
        self.assertEqual(outcome['oof_factor_error'], 4)
        self.assertIsNone(training_outcome({}, 'x.qasm', 64))
        legacy = {'actual_s': '10', 'matched_basis_pred_s': '40',
                  'status': 'success'}
        self.assertEqual(training_outcome({('x.qasm', 16): legacy},
                                          'x.qasm', 16)['oof_factor_error'], 4)


if __name__ == '__main__':
    unittest.main()
