"""MuJoCo camera and LiDAR calibration demo.

Run ``python tests/test_sensor_calibration.py`` to verify sensor measurements
against known scene geometry and open a comparison plot. Add ``--save PATH``
to save the plot, or ``--no-show`` to run without opening a plot window.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import unittest

import cv2
import matplotlib.pyplot as plt
import mujoco
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sensing import aruco_camera, lidar

MARKER_ID = 7
MARKER_SIZE_M = 0.36
CAMERA_NAME = "calibration_camera"
LIDAR_RAY_COUNT = 72
LIDAR_WALL_DISTANCE_M = 2.0
CAMERA_TARGET_X_M = 1.5
CAMERA_TARGET_Y_M = 0.3
CAMERA_HEIGHT_M = 1.5
LIDAR_HEIGHT_M = 0.8


def _marker_geometries() -> str:
    marker = cv2.aruco.generateImageMarker(
        cv2.aruco.getPredefinedDictionary(aruco_camera.ARUCO_DICTIONARY),
        MARKER_ID,
        240,
    )
    module_count = 6
    module_size = MARKER_SIZE_M / module_count
    panel_half_thickness = 0.005
    geometries = []
    for row in range(module_count):
        for column in range(module_count):
            module_pixels = marker[
                row * 40 : (row + 1) * 40,
                column * 40 : (column + 1) * 40,
            ]
            if float(np.mean(module_pixels)) >= 128:
                continue
            y = -(column - (module_count - 1) / 2) * module_size
            z = ((module_count - 1) / 2 - row) * module_size
            geometries.append(
                f'<geom type="box" pos="{-panel_half_thickness - 0.004} {y} {z}" '
                f'size="0.002 {module_size / 2} {module_size / 2}" '
                'rgba="0 0 0 1" contype="0" conaffinity="0"/>'
            )
    return "".join(geometries)


def _lidar_sites_and_sensors() -> tuple[str, str, np.ndarray]:
    angles = np.linspace(-np.pi, np.pi, LIDAR_RAY_COUNT, endpoint=False)
    sites = []
    sensors = []
    for index, angle in enumerate(angles):
        half_yaw = angle / 2
        half_sqrt_two = np.sqrt(0.5)
        quaternion = (
            np.cos(half_yaw) * half_sqrt_two,
            -np.sin(half_yaw) * half_sqrt_two,
            np.cos(half_yaw) * half_sqrt_two,
            np.sin(half_yaw) * half_sqrt_two,
        )
        quat = " ".join(f"{component:.12g}" for component in quaternion)
        sites.append(
            f'<site name="lidar_ray_{index}" pos="0 0 {LIDAR_HEIGHT_M}" '
            f'quat="{quat}" size="0.01" rgba="0.2 0.2 0.2 0"/>'
        )
        sensors.append(
            f'<rangefinder name="lidar_range_{index}" site="lidar_ray_{index}"/>'
        )
    return "".join(sites), "".join(sensors), angles


def _make_calibration_scene():
    sites, sensors, angles = _lidar_sites_and_sensors()
    xml = f"""<mujoco model="sensor calibration scene">
      <option timestep="0.002"/>
      <visual><global offwidth="640" offheight="480"/></visual>
      <worldbody>
        <light pos="0 0 4" directional="true" diffuse="1 1 1"/>
        <geom name="floor" type="plane" size="5 5 0.1"
              rgba="0.7 0.7 0.7 1"/>
        <body name="marker_target"
              pos="{CAMERA_TARGET_X_M} {CAMERA_TARGET_Y_M} {CAMERA_HEIGHT_M}">
          <geom name="marker_panel" type="box" size="0.005 0.18 0.18"
                rgba="1 1 1 1" contype="0" conaffinity="0"/>
          {_marker_geometries()}
        </body>
        <body name="camera_mount" pos="0 0 {CAMERA_HEIGHT_M}">
          <camera name="{CAMERA_NAME}" xyaxes="0 -1 0 0 0 1" fovy="60"/>
        </body>
        <body name="calibration_walls">
          <geom name="wall_front" type="box" pos="2.05 0 1.4"
                size="0.05 2.05 1.4" rgba="0.55 0.55 0.55 1"/>
          <geom name="wall_back" type="box" pos="-2.05 0 1.4"
                size="0.05 2.05 1.4" rgba="0.55 0.55 0.55 1"/>
          <geom name="wall_left" type="box" pos="0 2.05 1.4"
                size="2.0 0.05 1.4" rgba="0.55 0.55 0.55 1"/>
          <geom name="wall_right" type="box" pos="0 -2.05 1.4"
                size="2.0 0.05 1.4" rgba="0.55 0.55 0.55 1"/>
        </body>
        {sites}
      </worldbody>
      <sensor>{sensors}</sensor>
    </mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data, angles


def _camera_calibration(model, data):
    aruco_camera.configure_marker_size_from_geom(
        model,
        "marker_panel",
        (1, 2),
    )
    measurement = aruco_camera.get_camera_measurement(model, data, CAMERA_NAME)
    if measurement is None:
        raise AssertionError("Calibration marker was not visible to the camera")

    width = int(model.vis.global_.offwidth)
    height = int(model.vis.global_.offheight)
    camera_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_CAMERA,
        CAMERA_NAME,
    )
    vertical_fov = np.deg2rad(float(model.cam_fovy[camera_id]))
    focal_length = height / (2 * np.tan(vertical_fov / 2))
    camera_matrix = np.array(
        [
            [focal_length, 0, (width - 1) / 2],
            [0, focal_length, (height - 1) / 2],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )

    renderer = mujoco.Renderer(model, height=height, width=width)
    try:
        scene_option = mujoco.MjvOption()
        scene_option.sitegroup[0] = 0
        renderer.update_scene(
            data,
            camera=CAMERA_NAME,
            scene_option=scene_option,
        )
        image = renderer.render()
    finally:
        renderer.close()

    detection = aruco_camera.detect_aruco_marker(image)
    if detection is None:
        raise AssertionError("Calibration marker was not detected in the camera image")
    marker_id, observed_corners = detection
    tvec, rvec = aruco_camera.estimate_marker_pose(
        observed_corners,
        camera_matrix,
        np.zeros(5, dtype=np.float64),
    )
    half_size = MARKER_SIZE_M / 2
    object_corners = np.array(
        [
            [-half_size, half_size, 0],
            [half_size, half_size, 0],
            [half_size, -half_size, 0],
            [-half_size, -half_size, 0],
        ],
        dtype=np.float32,
    )
    projected_corners, _ = cv2.projectPoints(
        object_corners,
        rvec,
        tvec,
        camera_matrix,
        np.zeros(5, dtype=np.float64),
    )
    projected_corners = projected_corners.reshape(4, 2)
    reprojection_error_px = float(
        np.sqrt(np.mean(np.sum((projected_corners - observed_corners) ** 2, axis=1)))
    )
    truth = np.array(
        [
            np.hypot(CAMERA_TARGET_X_M, CAMERA_TARGET_Y_M),
            np.arctan2(CAMERA_TARGET_Y_M, CAMERA_TARGET_X_M),
        ]
    )
    return {
        "image": image,
        "marker_id": marker_id,
        "observed_corners": observed_corners,
        "projected_corners": projected_corners,
        "measurement": measurement.reshape(2),
        "truth": truth,
        "reprojection_error_px": reprojection_error_px,
    }


def _lidar_calibration(model, data, angles):
    ranges = np.array(
        [
            lidar.get_raw_lidar_data(model, data, f"lidar_range_{index}")[0]
            for index in range(LIDAR_RAY_COUNT)
        ],
        dtype=np.float64,
    )
    filtered_ranges = lidar.filter_lidar_ranges(ranges, max_range=3.5)
    expected_ranges = LIDAR_WALL_DISTANCE_M / np.maximum(
        np.abs(np.cos(angles)),
        np.abs(np.sin(angles)),
    )
    points = lidar.convert_to_cartesian(
        filtered_ranges,
        angles_increment=2 * np.pi / LIDAR_RAY_COUNT,
        angle_min=angles[0],
    )
    expected_points = np.column_stack(
        (expected_ranges * np.cos(angles), expected_ranges * np.sin(angles))
    )
    errors = ranges - expected_ranges
    return {
        "ranges": ranges,
        "expected_ranges": expected_ranges,
        "points": points,
        "expected_points": expected_points,
        "errors": errors,
        "rmse_m": float(np.sqrt(np.mean(errors**2))),
        "max_error_m": float(np.max(np.abs(errors))),
    }


class SensorCalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.data, cls.angles = _make_calibration_scene()

    @classmethod
    def tearDownClass(cls):
        aruco_camera.close_camera_renderer()

    def test_camera_range_bearing_and_marker_reprojection(self):
        result = _camera_calibration(self.model, self.data)
        range_error = abs(result["measurement"][0] - result["truth"][0])
        bearing_error = abs(result["measurement"][1] - result["truth"][1])

        self.assertEqual(result["marker_id"], MARKER_ID)
        self.assertLess(range_error, 0.15, "Camera range error should be below 15 cm")
        self.assertLess(
            bearing_error,
            0.08,
            "Camera bearing error should be below 0.08 radians",
        )
        self.assertLess(
            result["reprojection_error_px"],
            5.0,
            "Projected marker corners should match the detected corners",
        )

    def test_lidar_scan_matches_known_square_enclosure(self):
        result = _lidar_calibration(self.model, self.data, self.angles)

        self.assertTrue(np.all(np.isfinite(result["ranges"])))
        self.assertLess(result["rmse_m"], 0.01, "LiDAR scan RMSE should be below 1 cm")
        self.assertLess(
            result["max_error_m"],
            0.02,
            "Maximum LiDAR range error should be below 2 cm",
        )
        np.testing.assert_allclose(
            result["points"],
            result["expected_points"],
            atol=0.02,
        )


def _show_calibration():
    model, data, angles = _make_calibration_scene()
    try:
        camera = _camera_calibration(model, data)
        scan = _lidar_calibration(model, data, angles)
    finally:
        aruco_camera.close_camera_renderer()

    range_error = camera["measurement"][0] - camera["truth"][0]
    bearing_error = camera["measurement"][1] - camera["truth"][1]
    print(
        "Camera: "
        f"range {camera['measurement'][0]:.3f} m "
        f"(truth {camera['truth'][0]:.3f} m, error {range_error:+.3f} m); "
        f"bearing {camera['measurement'][1]:+.3f} rad "
        f"(truth {camera['truth'][1]:+.3f} rad, error {bearing_error:+.3f} rad); "
        f"corner reprojection RMSE {camera['reprojection_error_px']:.2f} px"
    )
    print(
        f"LiDAR: {LIDAR_RAY_COUNT} rays, range RMSE {scan['rmse_m']:.4f} m, "
        f"maximum absolute error {scan['max_error_m']:.4f} m"
    )

    figure, (camera_axis, lidar_axis) = plt.subplots(1, 2, figsize=(13, 5))
    camera_axis.imshow(camera["image"])
    camera_axis.scatter(
        camera["observed_corners"][:, 0],
        camera["observed_corners"][:, 1],
        marker="o",
        facecolors="none",
        edgecolors="lime",
        label="Detected corners",
    )
    camera_axis.scatter(
        camera["projected_corners"][:, 0],
        camera["projected_corners"][:, 1],
        marker="x",
        color="red",
        label="Reprojected corners",
    )
    camera_axis.set_title(
        f"Camera marker {camera['marker_id']}: "
        f"{camera['reprojection_error_px']:.2f} px corner RMSE"
    )
    camera_axis.axis("off")
    camera_axis.legend(loc="lower right")

    lidar_axis.plot(
        [-2, 2, 2, -2, -2],
        [-2, -2, 2, 2, -2],
        "k--",
        label="Known enclosure",
    )
    for index in range(0, LIDAR_RAY_COUNT, 6):
        point = scan["points"][index]
        lidar_axis.plot([0, point[0]], [0, point[1]], color="0.8", linewidth=0.7)
    lidar_axis.scatter(
        scan["expected_points"][:, 0],
        scan["expected_points"][:, 1],
        marker="x",
        color="tab:blue",
        label="Expected hit points",
    )
    lidar_axis.scatter(
        scan["points"][:, 0],
        scan["points"][:, 1],
        s=15,
        color="tab:orange",
        label="Measured hit points",
    )
    lidar_axis.set(
        title=f"LiDAR {LIDAR_RAY_COUNT}-ray scan: {scan['rmse_m']:.4f} m RMSE",
        xlabel="x forward [m]",
        ylabel="y left [m]",
        xlim=(-2.3, 2.3),
        ylim=(-2.3, 2.3),
        aspect="equal",
    )
    lidar_axis.grid(True, alpha=0.3)
    lidar_axis.legend(loc="upper right")
    figure.tight_layout()
    return figure


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", type=Path, help="save the calibration comparison plot")
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="run the tests and print errors without opening the plot window",
    )
    args = parser.parse_args()

    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        return 1

    figure = _show_calibration()
    try:
        if args.save is not None:
            figure.savefig(args.save, dpi=150, bbox_inches="tight")
            print(f"Calibration plot saved to {args.save}")
        if not args.no_show:
            plt.show()
    finally:
        plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
