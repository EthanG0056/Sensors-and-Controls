from pathlib import Path

import mujoco
import numpy as np

from estimation.ekf import (
    TurtleBotEKF,
    wheel_odometry,
    wrap,
)

from sensing.aruco_camera import (
    configure_marker_size_from_geom,
    get_camera_measurement,
    close_camera_renderer,
)


# ============================================================ #
#                         PARAMETERS                           #
# ============================================================ #

DT = 0.05
STEPS = 250

WHEEL_RADIUS = 0.033
WHEEL_SEPARATION = 0.160

CAMERA_OFFSET = (
    0.05,
    0.0
)

CAMERA_YAW_OFFSET = 0.0

CAMERA_DROPOUT_START = 100
CAMERA_DROPOUT_END = 150


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
#                    MUJOCO HELPER FUNCTIONS                   #
# ============================================================ #

def get_wheel_speeds(model, data):

    left_joint = model.joint("wheel_left").id
    right_joint = model.joint("wheel_right").id

    left_dof = model.jnt_dofadr[left_joint]
    right_dof = model.jnt_dofadr[right_joint]

    left_speed = data.qvel[left_dof]
    right_speed = data.qvel[right_dof]

    return left_speed, right_speed


def get_true_pose(model, data):

    # Ground truth is used ONLY for benchmark evaluation.

    base_joint = model.joint("base_joint").id
    address = model.jnt_qposadr[base_joint]

    x = data.qpos[address]
    y = data.qpos[address + 1]

    # MuJoCo free-joint quaternion is [w, x, y, z]
    qw = data.qpos[address + 3]
    qx = data.qpos[address + 4]
    qy = data.qpos[address + 5]
    qz = data.qpos[address + 6]

    yaw = np.arctan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy**2 + qz**2),
    )

    return np.array([
        x,
        y,
        wrap(yaw),
    ])


# ============================================================ #
#                         LOAD MODEL                           #
# ============================================================ #

model = mujoco.MjModel.from_xml_path(
    str(MODEL_PATH)
)

data = mujoco.MjData(model)

mujoco.mj_forward(
    model,
    data
)


# ============================================================ #
#                      SIMULATION TIMING                       #
# ============================================================ #

simulation_dt = model.opt.timestep

substeps = round(
    DT / simulation_dt
)

if not np.isclose(
    substeps * simulation_dt,
    DT,
):
    raise ValueError(
        "DT must be an integer multiple of the MuJoCo timestep"
    )


# ============================================================ #
#                          ACTUATORS                           #
# ============================================================ #

left_actuator = model.actuator("wheel_left").id
right_actuator = model.actuator("wheel_right").id


# ============================================================ #
#                       ARUCO MARKER                           #
# ============================================================ #

configure_marker_size_from_geom(
    model,
    "aruco_marker",
    (1, 2),
)

# The marker is fixed in the world, so its known world position
# forms part of the localisation map.
marker_id = model.geom("aruco_marker").id

MARKER_POSITION = data.geom_xpos[
    marker_id,
    :2
].copy()

print(
    "Known marker position:",
    MARKER_POSITION
)


# ============================================================ #
#                          EKF SETUP                           #
# ============================================================ #

initial_state = np.array([
    0.0,
    0.0,
    0.0,
])

P0 = np.diag([
    0.05**2,
    0.05**2,
    np.deg2rad(5.0)**2,
])

Q = np.diag([
    0.005**2,
    0.005**2,
    np.deg2rad(0.5)**2,
])

R = np.diag([
    0.03**2,
    np.deg2rad(1.5)**2,
])

ekf = TurtleBotEKF(
    initial_state,
    P0,
    Q,
    R,
)


# ============================================================ #
#                           HISTORY                            #
# ============================================================ #

true_history = []
estimate_history = []
covariance_history = []
camera_visible_history = []


# ============================================================ #
#                       SIMULATION LOOP                        #
# ============================================================ #

for step in range(STEPS):

    # Gentle curved motion so both position and heading are tested.
    v_command = 0.06
    omega_command = 0.03

    # Convert body command to wheel angular velocities.
    left_command = (
        v_command
        - 0.5 * WHEEL_SEPARATION * omega_command
    ) / WHEEL_RADIUS

    right_command = (
        v_command
        + 0.5 * WHEEL_SEPARATION * omega_command
    ) / WHEEL_RADIUS

    data.ctrl[left_actuator] = left_command
    data.ctrl[right_actuator] = right_command

    # Advance MuJoCo by one EKF sample interval.
    for _ in range(substeps):

        mujoco.mj_step(
            model,
            data
        )


    # -------------------------------------------------------- #
    # EKF PREDICTION - MEASURED WHEEL VELOCITIES
    # -------------------------------------------------------- #

    left_measured, right_measured = get_wheel_speeds(
        model,
        data
    )

    control = wheel_odometry(
        left_measured,
        right_measured
    )

    ekf.predict(
        control,
        DT
    )


    # -------------------------------------------------------- #
    # EKF UPDATE - REAL RENDERED ARUCO MEASUREMENT
    # -------------------------------------------------------- #

    camera_allowed = not (
        CAMERA_DROPOUT_START
        <= step
        < CAMERA_DROPOUT_END
    )

    measurement = None

    if camera_allowed:
        measurement = get_camera_measurement(
            model,
            data,
            "front_camera",
        )

    camera_visible = measurement is not None

    if camera_visible:
        ekf.update_camera(
            measurement,
            MARKER_POSITION,
            CAMERA_OFFSET,
            CAMERA_YAW_OFFSET,
        )


    # -------------------------------------------------------- #
    # GROUND TRUTH - EVALUATION ONLY
    # -------------------------------------------------------- #

    true_pose = get_true_pose(
        model,
        data
    )


    # -------------------------------------------------------- #
    # LOGGING
    # -------------------------------------------------------- #

    true_history.append(
        true_pose.copy()
    )

    estimate_history.append(
        ekf.state.copy()
    )

    covariance_history.append(
        ekf.P.copy()
    )

    camera_visible_history.append(
        camera_visible
    )


# ============================================================ #
#                           CLEANUP                            #
# ============================================================ #

close_camera_renderer()


# ============================================================ #
#                         CONVERT LOGS                         #
# ============================================================ #

true_history = np.array(
    true_history
)

estimate_history = np.array(
    estimate_history
)

covariance_history = np.array(
    covariance_history
)

camera_visible_history = np.array(
    camera_visible_history
)


# ============================================================ #
#                           ERRORS                             #
# ============================================================ #

error = true_history - estimate_history

error[:, 2] = np.array([
    wrap(angle)
    for angle in error[:, 2]
])


position_rmse = np.sqrt(
    np.mean(
        error[:, 0]**2
        + error[:, 1]**2
    )
)

heading_rmse = np.sqrt(
    np.mean(
        error[:, 2]**2
    )
)


final_position_error = np.linalg.norm(
    error[-1, :2]
)

final_heading_error = error[
    -1,
    2
]


# ============================================================ #
#                           RESULTS                            #
# ============================================================ #

camera_detections = np.sum(
    camera_visible_history
)

camera_detection_rate = (
    camera_detections
    / STEPS
)


print("\n================ LOCALISATION RESULTS ================")

print(
    f"Position RMSE: "
    f"{position_rmse:.4f} m"
)

print(
    f"Heading RMSE: "
    f"{np.rad2deg(heading_rmse):.2f} deg"
)

print(
    f"Final position error: "
    f"{final_position_error:.4f} m"
)

print(
    f"Final heading error: "
    f"{np.rad2deg(final_heading_error):.2f} deg"
)

print(
    f"Camera detections: "
    f"{camera_detections}/{STEPS}"
)

print(
    f"Camera detection rate: "
    f"{100 * camera_detection_rate:.1f}%"
)

print("\nFinal true pose:")
print(
    true_history[-1]
)

print("\nFinal EKF estimate:")
print(
    estimate_history[-1]
)

sigma_x = np.sqrt(
    covariance_history[:, 0, 0]
)

sigma_y = np.sqrt(
    covariance_history[:, 1, 1]
)

sigma_theta = np.sqrt(
    covariance_history[:, 2, 2]
)

print("\n================ DROPOUT UNCERTAINTY ================")

print(
    "Before dropout:",
    f"x = {sigma_x[99]:.4f} m,",
    f"y = {sigma_y[99]:.4f} m,",
    f"theta = {np.rad2deg(sigma_theta[99]):.2f} deg"
)

print(
    "End of dropout:",
    f"x = {sigma_x[149]:.4f} m,",
    f"y = {sigma_y[149]:.4f} m,",
    f"theta = {np.rad2deg(sigma_theta[149]):.2f} deg"
)

print(
    "After recovery:",
    f"x = {sigma_x[180]:.4f} m,",
    f"y = {sigma_y[180]:.4f} m,",
    f"theta = {np.rad2deg(sigma_theta[180]):.2f} deg"
)