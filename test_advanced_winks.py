import unittest

import numpy as np

from advanced_winks import FS, _features, _robust_scale


class AdvancedWinkTests(unittest.TestCase):
    def test_features_are_finite_and_fixed_length(self):
        rng = np.random.default_rng(1)
        epoch = rng.normal(size=(2, int(0.7 * FS)))
        feature = _features(epoch, np.ones(2))
        self.assertEqual(feature.shape, (33,))
        self.assertTrue(np.all(np.isfinite(feature)))

    def test_robust_scale_is_positive(self):
        baseline = np.zeros((2, FS * 3))
        scale = _robust_scale(baseline)
        self.assertTrue(np.all(scale > 0))


if __name__ == "__main__":
    unittest.main()
