from pathlib import Path

import mujoco
import numpy as np


# ============================================================ #
#                         PARAMETERS                           #
# ============================================================ #

NUM_RAYS = 36
ANGLE_INCREMENT = 2.0 * np.pi / NUM_RAYS


# ============================================================ #
#                         MODEL PATH                           #
# ============================================================ #

ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    ROOT
    / "integration"
    / "scenes"
    / "scene_turtlebot3_navigation.xml"
)


# ============================================================ #
#                       READ LIDAR DATA                        #
# ============================================================ #

def get_lidar_ranges(model, data):

    ranges = np.full(
        NUM_RAYS,
        np.inf
    )

    for i in range(NUM_RAYS):

        sensor_name = (
            f"lidar_{i:02d}_sensor"
        )

        sensor_id = model.sensor(
            sensor_name
        ).id

        sensor_address = model.sensor_adr[
            sensor_id
        ]

        distance = data.sensordata[
            sensor_address
        ]

        # MuJoCo returns -1 when the ray hits nothing.
        if distance >= 0.0:
            ranges[i] = distance

    return ranges


# ============================================================ #
#                         LOAD MODEL                           #
# ============================================================ #

model = mujoco.MjModel.from_xml_path(
    str(MODEL_PATH)
)

data = mujoco.MjData(
    model
)

mujoco.mj_forward(
    model,
    data
)


# ============================================================ #
#                       READ LIDAR SCAN                        #
# ============================================================ #

ranges = get_lidar_ranges(
    model,
    data
)


# ============================================================ #
#                           ANGLES                             #
# ============================================================ #

angles = np.arange(
    NUM_RAYS
) * ANGLE_INCREMENT

# Convert angles above pi into the range [-pi, pi).
angles = (
    angles + np.pi
) % (
    2.0 * np.pi
) - np.pi


# ============================================================ #
#                           RESULTS                            #
# ============================================================ #

valid = np.isfinite(
    ranges
)

print(
    "\n================ LIDAR TEST ================"
)

print(
    f"Number of rays: {NUM_RAYS}"
)

print(
    f"Valid obstacle hits: "
    f"{np.sum(valid)}/{NUM_RAYS}"
)


if np.any(valid):

    valid_indices = np.where(
        valid
    )[0]

    closest_index = valid_indices[
        np.argmin(
            ranges[valid_indices]
        )
    ]

    closest_distance = ranges[
        closest_index
    ]

    closest_angle = angles[
        closest_index
    ]

    print(
        f"Closest obstacle: "
        f"{closest_distance:.3f} m"
    )

    print(
        f"Closest ray: "
        f"{closest_index}"
    )

    print(
        f"Closest angle: "
        f"{np.rad2deg(closest_angle):.1f} deg"
    )


print(
    "\nIndividual ray readings:"
)

for i in range(NUM_RAYS):

    angle_deg = np.rad2deg(
        angles[i]
    )

    if np.isfinite(
        ranges[i]
    ):

        distance_text = (
            f"{ranges[i]:.3f} m"
        )

    else:

        distance_text = (
            "no hit"
        )

    print(
        f"Ray {i:02d}: "
        f"{angle_deg:7.1f} deg  "
        f"{distance_text}"
    )