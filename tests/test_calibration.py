import unittest

import cv2
import numpy as np

from sensing import CameraCalibration, LidarCalibration
from sensing import calibrate_camera, calibrate_lidar


def _checkerboard_images():
    columns, rows = 7, 6
    square_px = 55
    board = np.full((rows * square_px, columns * square_px), 255, dtype=np.uint8)
    for row in range(rows):
        for column in range(columns):
            if (row + column) % 2 == 0:
                board[
                    row * square_px : (row + 1) * square_px,
                    column * square_px : (column + 1) * square_px,
                ] = 0

    source = np.float32(
        [
            [0, 0],
            [board.shape[1] - 1, 0],
            [board.shape[1] - 1, board.shape[0] - 1],
            [0, board.shape[0] - 1],
        ]
    )
    destinations = (
        [[80, 60], [470, 48], [490, 380], [72, 395]],
        [[120, 45], [530, 80], [500, 410], [100, 370]],
        [[55, 85], [450, 55], [465, 390], [70, 420]],
        [[130, 75], [510, 45], [540, 385], [105, 400]],
    )
    images = []
    for destination in destinations:
        transform = cv2.getPerspectiveTransform(source, np.float32(destination))
        images.append(
            cv2.warpPerspective(
                board,
                transform,
                (640, 480),
                borderValue=255,
            )
        )
    return images


class CameraCalibrationTests(unittest.TestCase):
    def test_calibrates_intrinsics_from_multiple_checkerboard_views(self):
        calibration = calibrate_camera(
            _checkerboard_images(),
            checkerboard_size=(6, 5),
            square_size_m=0.024,
        )

        self.assertIsInstance(calibration, CameraCalibration)
        self.assertEqual(calibration.camera_matrix.shape, (3, 3))
        self.assertEqual(calibration.image_size, (640, 480))
        self.assertEqual(calibration.successful_views, 4)
        self.assertTrue(np.all(np.isfinite(calibration.dist_coeffs)))
        self.assertGreater(calibration.reprojection_error_px, 0)

    def test_rejects_insufficient_checkerboard_views(self):
        with self.assertRaisesRegex(ValueError, "at least 3 views"):
            calibrate_camera(
                [np.full((100, 100), 255, dtype=np.uint8)],
                checkerboard_size=(6, 5),
                square_size_m=0.024,
            )

    def test_rejects_mixed_image_resolutions(self):
        images = _checkerboard_images()[:3]
        images[1] = images[1][:, :-1]
        with self.assertRaisesRegex(ValueError, "same resolution"):
            calibrate_camera(
                images,
                checkerboard_size=(6, 5),
                square_size_m=0.024,
            )


class LidarCalibrationTests(unittest.TestCase):
    def test_fits_range_correction_and_wrapped_angle_offset(self):
        measured_ranges = np.array([0.5, 1.0, 2.0, 3.5])
        expected_ranges = 1.02 * measured_ranges - 0.03
        measured_angles = np.array([3.10, -3.12, -1.0, 0.5])
        expected_angles = np.arctan2(
            np.sin(measured_angles + 0.08),
            np.cos(measured_angles + 0.08),
        )

        calibration = calibrate_lidar(
            measured_ranges,
            expected_ranges,
            measured_angles,
            expected_angles,
        )

        self.assertIsInstance(calibration, LidarCalibration)
        self.assertAlmostEqual(calibration.range_scale, 1.02)
        self.assertAlmostEqual(calibration.range_offset_m, -0.03)
        self.assertAlmostEqual(calibration.range_rmse_m, 0.0, places=12)
        self.assertAlmostEqual(calibration.angle_offset_rad, 0.08)
        self.assertAlmostEqual(calibration.angle_rmse_rad, 0.0, places=12)
        np.testing.assert_allclose(
            calibration.apply_ranges(measured_ranges),
            expected_ranges,
        )
        np.testing.assert_allclose(
            calibration.apply_angles(measured_angles),
            expected_angles,
        )

    def test_range_only_calibration_does_not_claim_angle_correction(self):
        calibration = calibrate_lidar(
            np.array([1.0, 2.0]),
            np.array([1.1, 2.1]),
        )
        self.assertIsNone(calibration.angle_offset_rad)
        with self.assertRaisesRegex(RuntimeError, "not included"):
            calibration.apply_angles(np.array([0.0]))

    def test_rejects_unidentifiable_or_unpaired_observations(self):
        with self.assertRaisesRegex(ValueError, "distinct values"):
            calibrate_lidar(np.array([1.0, 1.0]), np.array([1.1, 1.2]))
        with self.assertRaisesRegex(ValueError, "provided together"):
            calibrate_lidar(
                np.array([1.0, 2.0]),
                np.array([1.1, 2.1]),
                measured_angles=np.array([0.0]),
            )
        with self.assertRaisesRegex(ValueError, "unique offset"):
            calibrate_lidar(
                np.array([1.0, 2.0]),
                np.array([1.1, 2.1]),
                measured_angles=np.array([0.0, 0.0]),
                reference_angles=np.array([0.0, np.pi]),
            )


if __name__ == "__main__":
    unittest.main()
