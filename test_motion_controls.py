import unittest

import numpy as np

from motion_controls import (
    AccelNodDetector, HeadGestureDetector, NodRotateDetector,
    PitchNodDetector, YawRotateDetector,
)


class HeadGestureDetectorTests(unittest.TestCase):
    def detector(self) -> HeadGestureDetector:
        detector = HeadGestureDetector(
            threshold_g=0.16,
            neutral_g=0.075,
            calibration_samples=9,
            smooth_samples=3,
        )
        detector.add(np.tile([0.02, -0.01, 0.99], (9, 1)))
        return detector

    def test_right_tilt_fires_once_until_neutral(self):
        detector = self.detector()
        detector.add(np.tile([0.02, 0.21, 0.98], (3, 1)))
        self.assertEqual(detector.detect()[0], "right")
        self.assertIsNone(detector.detect()[0])
        detector.add(np.tile([0.02, -0.01, 0.99], (3, 1)))
        self.assertIsNone(detector.detect()[0])
        detector.add(np.tile([0.02, 0.21, 0.98], (3, 1)))
        self.assertEqual(detector.detect()[0], "right")

    def test_left_tilt(self):
        detector = self.detector()
        detector.add(np.tile([0.02, -0.22, 0.97], (3, 1)))
        self.assertEqual(detector.detect()[0], "left")

    def test_nod_does_not_move(self):
        detector = self.detector()
        detector.add(np.tile([0.25, -0.01, 0.97], (3, 1)))
        self.assertIsNone(detector.detect()[0])

    def test_small_movements_do_not_fire(self):
        detector = self.detector()
        detector.add(np.tile([0.08, 0.08, 0.99], (3, 1)))
        self.assertIsNone(detector.detect()[0])

    def test_yaw_fires_once_and_masks_return(self):
        detector = YawRotateDetector(threshold_dps=50, cooldown_samples=6)
        detector.add(np.array([[0, 0, 65], [0, 0, 10], [0, 0, -70]]))
        self.assertTrue(detector.detect()[0])
        self.assertFalse(detector.detect()[0])
        detector.add(np.tile([0, 0, 0], (6, 1)))
        detector.add(np.array([[0, 0, -60]]))
        fired, peak = detector.detect()
        self.assertTrue(fired)
        self.assertEqual(peak, -60)

    def test_nod_requires_down_and_back(self):
        detector = NodRotateDetector(
            threshold_dps=50, return_threshold_dps=25,
            max_gap_samples=8, cooldown_samples=6,
        )
        detector.add(np.array([[0, 60, 4], [0, 20, 2]]))
        self.assertFalse(detector.detect()[0])
        detector.add(np.array([[0, -32, 3]]))
        fired, peak = detector.detect()
        self.assertTrue(fired)
        self.assertEqual(peak, 60)
        self.assertFalse(detector.detect()[0])

    def test_nod_ignores_roll_and_one_way_motion(self):
        detector = NodRotateDetector(
            threshold_dps=50, return_threshold_dps=25,
            max_gap_samples=3, cooldown_samples=2,
        )
        detector.add(np.array([[80, 55, 0], [0, 70, 0], [0, 5, 0]]))
        self.assertFalse(detector.detect()[0])
        detector.add(np.array([[0, 2, 0], [0, 1, 0]]))
        self.assertFalse(detector.detect()[0])

    def test_pitch_nod_requires_held_pose_and_return(self):
        detector = PitchNodDetector(
            calibrated_pitch=-24, hold_samples=2, neutral_samples=2,
            cooldown_samples=2,
        )
        detector.add(-15)
        self.assertFalse(detector.detect()[0])
        detector.add(-16)
        self.assertFalse(detector.detect()[0])
        detector.add(-2)
        detector.add(-1)
        self.assertTrue(detector.detect()[0])
        self.assertFalse(detector.detect()[0])

    def test_pitch_nod_rejects_spike_and_wrong_direction(self):
        detector = PitchNodDetector(
            calibrated_pitch=-24, hold_samples=3, neutral_samples=2,
        )
        for pitch in (-30, 0, 30, 30, 30, 0, 0):
            detector.add(pitch)
        self.assertFalse(detector.detect()[0])

    def test_accel_nod_requires_hold_and_center_return(self):
        detector = AccelNodDetector(
            center_forward=0.0, down_forward=0.5,
            hold_samples=2, neutral_samples=2, cooldown_samples=2,
        )
        for value in (
            0, 0, 0, 0, 0.40, 0.42, 0.44, 0.45, 0.46,
            0.02, 0.01, 0, 0, 0,
        ):
            detector.add(value)
        self.assertTrue(detector.detect()[0])
        self.assertFalse(detector.detect()[0])

    def test_accel_nod_rejects_short_spike(self):
        detector = AccelNodDetector(
            center_forward=0.0, down_forward=-0.5,
            hold_samples=3, neutral_samples=2,
        )
        for value in (0, 0, 0, 0, -0.6, 0, 0, 0, 0):
            detector.add(value)
        self.assertFalse(detector.detect()[0])


if __name__ == "__main__":
    unittest.main()
