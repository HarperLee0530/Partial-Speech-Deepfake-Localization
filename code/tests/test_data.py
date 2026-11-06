import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import unittest
import numpy as np
from tools.data_tools import strata_for, choose
from tools.reporting import compare_pair


class FixedSubsetTests(unittest.TestCase):
    def test_string_labels_are_stratified_as_numeric_labels(self):
        numeric = {'real': np.ones(10), 'fake': np.zeros(10), 'partial': np.array([1, 0, 1])}
        text = {key: value.astype(int).astype(str) for key, value in numeric.items()}
        protocol = dict.fromkeys(numeric, 'attack')
        self.assertEqual(strata_for(numeric, protocol), strata_for(text, protocol))
        self.assertEqual(len(strata_for(text, protocol)), 3)

    def test_sampling_is_fixed_exact_and_independent_of_input_order(self):
        groups = {'a': [f'a_{i:03}' for i in range(47)], 'b': [f'b_{i:03}' for i in range(56)]}
        first, allocation = choose(groups, 'train')
        second, _ = choose({k: list(reversed(v)) for k, v in groups.items()}, 'train')
        self.assertEqual(first, second)
        self.assertEqual(len(first), 10)
        self.assertEqual(len(first), len(set(first)))
        self.assertEqual(sum(allocation.values()), 10)

    def test_comparison_refuses_full_and_subset_even_if_other_fields_match(self):
        row = dict(status='complete', data_profile='fixed10')
        self.assertFalse(compare_pair(row, dict(row, data_profile='full'))['comparable'])


if __name__ == '__main__':
    unittest.main()
