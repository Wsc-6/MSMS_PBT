"""Regression checks for outer-test isolation in probability calibration."""
import unittest
import numpy as np
import pandas as pd
from pbt_msms.dreams_heads import _outer_training_calibration_scores


class NestedCalibrationIsolationTests(unittest.TestCase):
    def test_outer_test_changes_cannot_change_calibration_scores(self):
        rng = np.random.default_rng(61)
        features = rng.normal(size=(50, 6))
        folds = np.repeat(np.arange(5), 10)
        labels = np.tile([0, 1], 25)
        index = pd.DataFrame({
            'inchikey_2d': [f'compound_{i}' for i in range(50)],
            'cv_group': [f'group_{i}' for i in range(50)],
            'cv_fold': folds, 'label__P_screen_not_ready_binary': labels,
        })
        arrays = {'all_spectra': features.copy(), 'single_representative': features.copy()}
        original, audit = _outer_training_calibration_scores(arrays, index, 'linear', {'C': .3}, 2, 'all_spectra')
        changed_index = index.copy()
        changed_index.loc[folds == 2, 'label__P_screen_not_ready_binary'] = 1 - labels[folds == 2]
        changed_arrays = {key: value.copy() for key, value in arrays.items()}
        for value in changed_arrays.values():
            value[folds == 2] = rng.normal(1000, 100, size=(10, 6))
        changed, changed_audit = _outer_training_calibration_scores(changed_arrays, changed_index, 'linear', {'C': .3}, 2, 'all_spectra')
        for key in arrays:
            np.testing.assert_array_equal(original[key], changed[key])
            self.assertTrue(np.isnan(original[key][folds == 2]).all())
            self.assertTrue(np.isfinite(original[key][folds != 2]).all())
        self.assertEqual(audit, changed_audit)
        self.assertEqual(len(audit), 4)
        self.assertTrue(all(row['outer_test_overlap_n'] == 0 and row['inner_group_overlap_n'] == 0 for row in audit))

    def test_reused_structure_group_is_rejected(self):
        rng = np.random.default_rng(3)
        index = pd.DataFrame({'inchikey_2d': [f'c{i}' for i in range(30)],
                              'cv_group': ['same_group'] * 30,
                              'cv_fold': np.repeat(np.arange(5), 6),
                              'label__P_screen_not_ready_binary': np.tile([0, 1], 15)})
        with self.assertRaisesRegex(ValueError, 'structure group'):
            _outer_training_calibration_scores({'all_spectra': rng.normal(size=(30, 3))}, index,
                                               'linear', {'C': .3}, 0, 'all_spectra')


if __name__ == '__main__':
    unittest.main()
