import contextlib
import io
import time
import unittest

import cv2
import mujoco
import numpy as np

from pathlib import Path
import sys
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sensing import aruco_camera, lidar


def _make_sensor_scene():
    marker_id = 7
    marker_pixels = cv2.aruco.generateImageMarker(
        cv2.aruco.getPredefinedDictionary(aruco_camera.ARUCO_DICTIONARY),
        marker_id,
        240,
    )
    module_count = 6
    module_size = 0.36 / module_count
    target_x = 2.0
    target_z = 1.0
    panel_half_thickness = 0.005

    marker_geometries = []
    for row in range(module_count):
        for column in range(module_count):
            module_pixels = marker_pixels[
                row * 40 : (row + 1) * 40,
                column * 40 : (column + 1) * 40,
            ]
            if float(np.mean(module_pixels)) >= 128:
                continue
            y = -(column - (module_count - 1) / 2) * module_size
            z = ((module_count - 1) / 2 - row) * module_size
            marker_geometries.append(
                f'<geom type="box" pos="{-panel_half_thickness - 0.004} {y} {z}" '
                f'size="0.002 {module_size / 2} {module_size / 2}" '
                'rgba="0 0 0 1" contype="0" conaffinity="0"/>'
            )

    xml = f"""<mujoco>
      <option timestep="0.002"/>
      <visual><global offwidth="640" offheight="480"/></visual>
      <worldbody>
        <light pos="0 0 4" directional="true" diffuse="1 1 1"/>
        <geom name="floor" type="plane" size="5 5 0.1"
              rgba="0.7 0.7 0.7 1"/>
        <body name="marker_target" pos="{target_x} 0 {target_z}">
          <geom name="marker_panel" type="box"
                size="{panel_half_thickness} 0.18 0.18"
                rgba="1 1 1 1" contype="0" conaffinity="0"/>
          {''.join(marker_geometries)}
        </body>
        <body name="box_obstacle" pos="3 0 1">
          <geom name="obstacle_box" type="box" size="0.1 0.25 0.25"
                rgba="0.8 0.2 0.2 1"/>
        </body>
        <body name="sphere_obstacle" pos="2 0.8 1">
          <geom name="obstacle_sphere" type="sphere" size="0.2"
                rgba="0.2 0.3 0.9 1"/>
        </body>
        <body name="camera_mount" pos="0 0 {target_z}">
          <camera name="test_camera" xyaxes="0 -1 0 0 0 1" fovy="60"/>
        </body>
        <site name="lidar_site" pos="0 0 1"
              quat="0.7071068 0 0.7071068 0" size="0.01"/>
      </worldbody>
      <sensor><rangefinder name="test_lidar" site="lidar_site"/></sensor>
    </mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data, marker_id


class MuJoCoSensorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.data, cls.marker_id = _make_sensor_scene()

    def tearDown(self):
        aruco_camera.close_camera_renderer()


class ArucoCameraTests(MuJoCoSensorTests):
    def test_detect_marker_from_grayscale_and_rgb_images(self):
        marker = cv2.aruco.generateImageMarker(
            cv2.aruco.getPredefinedDictionary(aruco_camera.ARUCO_DICTIONARY),
            self.marker_id,
            160,
        )
        image = np.full((220, 220), 255, dtype=np.uint8)
        image[30:190, 30:190] = marker

        grayscale_detection = aruco_camera.detect_aruco_marker(image)
        rgb_detection = aruco_camera.detect_aruco_marker(
            cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        )

        self.assertIsNotNone(grayscale_detection)
        self.assertIsNotNone(rgb_detection)
        self.assertEqual(grayscale_detection[0], self.marker_id)
        self.assertEqual(rgb_detection[0], self.marker_id)
        self.assertEqual(grayscale_detection[1].shape, (4, 2))
        with self.assertRaises(ValueError):
            aruco_camera.detect_aruco_marker(np.zeros((10, 10, 4), dtype=np.uint8))

    def test_measurement_from_rendered_mujoco_camera(self):
        aruco_camera.configure_marker_size_from_geom(
            self.model,
            "marker_panel",
            (1, 2),
        )

        measurement = aruco_camera.get_camera_measurement(
            self.model,
            self.data,
            "test_camera",
        )

        self.assertIsNotNone(measurement)
        self.assertEqual(measurement.shape, (2, 1))
        self.assertAlmostEqual(float(measurement[0, 0]), 2.0, delta=0.25)
        self.assertAlmostEqual(float(measurement[1, 0]), 0.0, delta=0.12)

    def test_camera_measurement_rejects_unknown_camera(self):
        with self.assertRaisesRegex(ValueError, "was not found"):
            aruco_camera.get_camera_measurement(
                self.model,
                self.data,
                "missing_camera",
            )

    def test_marker_size_configuration_validates_inputs(self):
        aruco_camera.configure_marker_size(0.2)
        with self.assertRaises(ValueError):
            aruco_camera.configure_marker_size(0)
        with self.assertRaises(ValueError):
            aruco_camera.configure_marker_size_from_geom(
                self.model,
                "marker_panel",
                (0, 1),
            )

    def test_marker_range_and_bearing(self):
        measurement = aruco_camera.get_range_and_bearing(
            np.array([3.0, 100.0, 4.0])
        )
        np.testing.assert_allclose(measurement, [[5.0], [-np.arctan2(3.0, 4.0)]])
        with self.assertRaises(ValueError):
            aruco_camera.get_range_and_bearing(np.array([1.0, 0.0, 0.0]))


class LidarTests(MuJoCoSensorTests):
    def test_reads_named_mujoco_sensor_and_returns_a_copy(self):
        readings = lidar.get_raw_lidar_data(
            self.model,
            self.data,
            "test_lidar",
        )

        self.assertEqual(readings.shape, (1,))
        self.assertAlmostEqual(readings[0], 1.995, places=5)
        readings[0] = 0
        self.assertAlmostEqual(self.data.sensordata[0], 1.995, places=5)

    def test_filters_invalid_and_out_of_bounds_ranges(self):
        filtered = lidar.filter_lidar_ranges(
            np.array([0.1, 0.12, 0.5, 3.5, np.inf, np.nan])
        )
        np.testing.assert_allclose(
            filtered,
            [np.nan, 0.12, 0.5, np.nan, np.nan, np.nan],
            equal_nan=True,
        )
        with self.assertRaises(ValueError):
            lidar.filter_lidar_ranges(np.ones((2, 2)))

    def test_converts_valid_ranges_to_robot_frame_cartesian_points(self):
        points = lidar.convert_to_cartesian(
            np.array([2.0, np.nan, 2.0]),
            np.pi / 2,
            angle_min=-np.pi / 2,
        )
        np.testing.assert_allclose(points[0], [0.0, -2.0], atol=1e-12)
        self.assertTrue(np.all(np.isnan(points[1])))
        np.testing.assert_allclose(points[2], [0.0, 2.0], atol=1e-12)

    def test_finds_closest_valid_obstacle_and_safety_state(self):
        distance, angle, critical = lidar.get_closest_obstacle(
            np.array([np.inf, 1.2, 0.4]),
            angle_increment=0.25,
            critical_threshold=0.5,
        )
        self.assertEqual(distance, 0.4)
        self.assertEqual(angle, 0.5)
        self.assertTrue(critical)

    def test_empty_valid_scan_has_no_critical_alert(self):
        distance, angle, critical = lidar.get_closest_obstacle(
            np.array([np.nan, 0.1, 3.5]),
            angle_increment=0.1,
        )
        self.assertTrue(np.isnan(distance))
        self.assertTrue(np.isnan(angle))
        self.assertFalse(critical)

    def test_point_cloud_helpers(self):
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        points = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.5]])
        figure = plt.figure()
        try:
            axis = lidar.show_cloud(
                points,
                colours=np.array(["red", "blue"]),
                ax=figure.add_subplot(projection="3d"),
            )
            self.assertEqual(axis.get_xlabel(), "x fwd [m]")
            self.assertEqual(axis.get_ylabel(), "y left [m]")
            with contextlib.redirect_stdout(io.StringIO()) as output:
                lidar.cloud_stats("test cloud", type("Cloud", (), {
                    "points": points,
                })())
            self.assertIn("2 points", output.getvalue())
        finally:
            plt.close(figure)


def _run_tests_and_show_scene():
    suite = unittest.defaultTestLoader.loadTestsFromModule(
        sys.modules[__name__]
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        return 1

    import mujoco.viewer

    model, data, _ = _make_sensor_scene()
    print("MuJoCo sensor scene is open. Close the viewer window to exit.")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            viewer.sync()
            time.sleep(1 / 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(_run_tests_and_show_scene())
