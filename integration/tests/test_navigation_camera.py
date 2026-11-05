from pathlib import Path

import mujoco

import cv2
import numpy as np

from sensing.aruco_camera import (
    configure_marker_size_from_geom,
    get_camera_measurement,
    close_camera_renderer,
)


def _associated_camera_observation(result, marker_positions):
    """Validate the camera-origin API; reject absent/unmapped IDs without fusing.

    Extrinsics remain in the EKF observation model; no body offset is applied here.
    """
    if result is None:
        return None
    if not isinstance(result, tuple) or len(result) != 2:
        raise ValueError("Expected (marker_id, measurement)")
    marker_id, measurement = result
    if isinstance(marker_id, (bool, np.bool_)) or not isinstance(marker_id, (int, np.integer)):
        raise ValueError("Expected integer marker ID")
    measurement = np.asarray(measurement, dtype=float)
    if (measurement.shape != (2, 1) or not np.isfinite(measurement).all()
            or measurement[0, 0] <= 0):
        raise ValueError("Expected finite (2,1) range/bearing with positive range")
    if marker_id not in marker_positions:
        return None
    landmark = np.asarray(marker_positions[marker_id], dtype=float)
    if landmark.shape != (2,) or not np.isfinite(landmark).all():
        raise ValueError("Expected finite mapped world landmark (2,)")
    return int(marker_id), measurement.copy(), landmark.copy()


# ========================== MODEL PATH ========================== #

ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    ROOT
    / "integration"
    / "scenes"
    / "scene_turtlebot3_navigation.xml"
)


# ========================== LOAD MODEL ========================== #

model = mujoco.MjModel.from_xml_path(
    str(MODEL_PATH)
)

data = mujoco.MjData(model)

mujoco.mj_forward(
    model,
    data
)


# ========================== CAMERA DEBUG IMAGE ========================== #

width = int(model.vis.global_.offwidth)
height = int(model.vis.global_.offheight)

renderer = mujoco.Renderer(
    model,
    height=height,
    width=width
)

renderer.update_scene(
    data,
    camera="front_camera"
)

image = renderer.render()

cv2.imwrite(
    "integration/tests/camera_view.png",
    cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
)

renderer.close()

print("Camera image saved to integration/tests/camera_view.png")


# ========================== MARKER SETUP ========================== #

configure_marker_size_from_geom(
    model,
    "aruco_marker",
    (1, 2)
)


# ========================== CAMERA TEST ========================== #

camera_result = get_camera_measurement(
    model,
    data,
    "front_camera"
)

# Explicit known-ID map from fixed scene geometry; not robot ground-truth feedback.
marker_positions = {
    0: data.geom_xpos[model.geom("aruco_marker").id, :2].copy(),
    1: data.geom_xpos[model.geom("aruco_marker_1").id, :2].copy(),
}
observation = _associated_camera_observation(camera_result, marker_positions)
print("\nCamera measurement:")

if observation is None:

    print("No ArUco marker detected.")

else:

    marker_id, measurement, marker_position = observation
    print(f"Marker ID: {marker_id}; mapped world landmark: {marker_position}")
    print(
        f"Range: {measurement[0, 0]:.3f} m"
    )

    print(
        f"Bearing: {measurement[1, 0]:.3f} rad"
    )


# ========================== CLEANUP ========================== #

close_camera_renderer()