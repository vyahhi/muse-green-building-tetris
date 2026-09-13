import unittest

import numpy as np

from head_orientation import HeadOrientation, RollColumnMapper


class HeadOrientationTests(unittest.TestCase):
    def test_stationary_settles_at_center(self):
        pose = HeadOrientation(settle_frames=10)
        for _ in range(20):
            roll = pose.update(np.array([0.0, 0.0, 1.0]), np.zeros(3))
        self.assertTrue(pose.ready)
        self.assertAlmostEqual(roll, 0.0, delta=0.5)

    def test_fused_roll_tracks_physical_roll(self):
        pose = HeadOrientation(settle_frames=10)
        for _ in range(15):
            pose.update(np.array([0.0, 0.0, 1.0]), np.zeros(3))
        for angle in np.linspace(0, 20, 53):
            radians = np.radians(angle)
            roll = pose.update(
                np.array([0.0, np.sin(radians), np.cos(radians)]),
                np.array([20.0, 0.0, 0.0]),
            )
        self.assertGreater(abs(roll), 8.0)

    def test_returns_to_center_without_euler_wrap(self):
        pose = HeadOrientation(settle_frames=10)
        for _ in range(15):
            pose.update(np.array([0.0, 0.0, 1.0]), np.zeros(3))
        angle = np.radians(15)
        for _ in range(100):
            pose.update(
                np.array([0.0, np.sin(angle), np.cos(angle)]), np.zeros(3)
            )
        for _ in range(160):
            roll = pose.update(np.array([0.0, 0.0, 1.0]), np.zeros(3))
        self.assertAlmostEqual(roll, 0.0, delta=2.0)

    def test_personalized_endpoints_map_to_columns(self):
        mapper = RollColumnMapper(left_roll=-14, right_roll=18)
        self.assertEqual(mapper.target(-14, 9), 0)
        self.assertEqual(mapper.target(18, 9), 8)
        self.assertIn(mapper.target(2, 9), (3, 4, 5))

    def test_rejects_tiny_calibration_range(self):
        with self.assertRaises(ValueError):
            RollColumnMapper(-2, 3)


if __name__ == "__main__":
    unittest.main()
