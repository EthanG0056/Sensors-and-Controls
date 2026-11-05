from pathlib import Path

import mujoco

import cv2

from sensing.aruco_camera import (
    configure_marker_size_from_geom,
    get_camera_measurement,
    close_camera_renderer,
)


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

measurement = get_camera_measurement(
    model,
    data,
    "front_camera"
)

print("\nCamera measurement:")

if measurement is None:

    print("No ArUco marker detected.")

else:

    print(
        f"Range: {measurement[0, 0]:.3f} m"
    )

    print(
        f"Bearing: {measurement[1, 0]:.3f} rad"
    )


# ========================== CLEANUP ========================== #

close_camera_renderer()