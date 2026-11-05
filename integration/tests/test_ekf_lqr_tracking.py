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

from control.lqr_controller import (
    LQRTracker,
    wheel_commands,
)

from control.trajectory import reference


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


# ============================================================ #
#                         PARAMETERS                           #
# ============================================================ #

DT = 0.05

TRAJECTORY_NAME = "straight"
DURATION = 12.0
STEPS = int(DURATION / DT)

CAMERA_OFFSET = (
    0.05,
    0.0
)

CAMERA_YAW_OFFSET = 0.0


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

    # Ground truth is used ONLY for evaluation.

    base_joint = model.joint("base_joint").id
    address = model.jnt_qposadr[base_joint]

    x = data.qpos[address]
    y = data.qpos[address + 1]

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
    DT
):
    raise ValueError(
        "DT must be an integer multiple of the MuJoCo timestep"
    )


# ============================================================ #
#                          ACTUATORS                           #
# ============================================================ #

left_actuator = model.actuator("wheel_left").id
right_actuator = model.actuator("wheel_right").id

# Start stationary.
data.ctrl[left_actuator] = 0.0
data.ctrl[right_actuator] = 0.0


# ============================================================ #
#                       ARUCO MARKER                           #
# ============================================================ #

configure_marker_size_from_geom(
    model,
    "aruco_marker",
    (1, 2),
)

marker_geom_id = model.geom("aruco_marker").id

MARKER_POSITION = data.geom_xpos[
    marker_geom_id,
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
#                        LQR CONTROLLER                        #
# ============================================================ #

controller = LQRTracker(
    dt=DT
)


# ============================================================ #
#                           HISTORY                            #
# ============================================================ #

true_history = []
estimate_history = []
reference_history = []
wheel_command_history = []
camera_visible_history = []


# ============================================================ #
#                       CLOSED LOOP TEST                       #
# ============================================================ #

for step in range(STEPS):

    # -------------------------------------------------------- #
    # 1. ADVANCE MUJOCO USING PREVIOUS COMMAND
    # -------------------------------------------------------- #

    for _ in range(substeps):

        mujoco.mj_step(
            model,
            data
        )


    # Current model time represented by this state.
    current_time = (step + 1) * DT


    # -------------------------------------------------------- #
    # 2. READ MEASURED WHEEL VELOCITIES
    # -------------------------------------------------------- #

    left_measured, right_measured = get_wheel_speeds(
        model,
        data
    )

    body_motion = wheel_odometry(
        left_measured,
        right_measured
    )


    # -------------------------------------------------------- #
    # 3. EKF PREDICTION
    # -------------------------------------------------------- #

    ekf.predict(
        body_motion,
        DT
    )


    # -------------------------------------------------------- #
    # 4. EKF CAMERA UPDATE
    # -------------------------------------------------------- #

    camera_result = get_camera_measurement(
        model,
        data,
        "front_camera"
    )

    # This diagnostic has only marker 0 mapped. Never fuse marker 1 as marker 0.
    observation = _associated_camera_observation(camera_result, {0: MARKER_POSITION})
    camera_visible = observation is not None

    if camera_visible:
        marker_id, measurement, marker_position = observation

        ekf.update_camera(
            measurement,
            marker_position,
            CAMERA_OFFSET,
            CAMERA_YAW_OFFSET,
        )


    # -------------------------------------------------------- #
    # 5. GET TRAJECTORY REFERENCE
    # -------------------------------------------------------- #

    current_reference = reference(
        TRAJECTORY_NAME,
        current_time
    )


    # -------------------------------------------------------- #
    # 6. LQR USES EKF STATE ONLY
    # -------------------------------------------------------- #

    estimated_pose = ekf.state.copy()

    body_command, controller_mode = controller.command(
        estimated_pose,
        current_reference
    )


    # -------------------------------------------------------- #
    # 7. CONVERT BODY COMMAND TO WHEEL COMMANDS
    # -------------------------------------------------------- #

    wheels, feasible_body, saturated = wheel_commands(
        body_command[0],
        body_command[1]
    )

    data.ctrl[left_actuator] = wheels[0]
    data.ctrl[right_actuator] = wheels[1]


    # -------------------------------------------------------- #
    # 8. GROUND TRUTH - EVALUATION ONLY
    # -------------------------------------------------------- #

    true_pose = get_true_pose(
        model,
        data
    )


    # -------------------------------------------------------- #
    # 9. LOGGING
    # -------------------------------------------------------- #

    true_history.append(
        true_pose.copy()
    )

    estimate_history.append(
        estimated_pose.copy()
    )

    reference_history.append(
        current_reference.pose.copy()
    )

    wheel_command_history.append(
        wheels.copy()
    )

    camera_visible_history.append(
        camera_visible
    )


# ============================================================ #
#                           STOP                               #
# ============================================================ #

data.ctrl[left_actuator] = 0.0
data.ctrl[right_actuator] = 0.0

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

reference_history = np.array(
    reference_history
)

wheel_command_history = np.array(
    wheel_command_history
)

camera_visible_history = np.array(
    camera_visible_history
)


# ============================================================ #
#                    LOCALISATION ERRORS                       #
# ============================================================ #

localisation_error = (
    true_history
    - estimate_history
)

localisation_error[:, 2] = np.array([
    wrap(angle)
    for angle in localisation_error[:, 2]
])

localisation_position_rmse = np.sqrt(
    np.mean(
        localisation_error[:, 0]**2
        + localisation_error[:, 1]**2
    )
)

localisation_heading_rmse = np.sqrt(
    np.mean(
        localisation_error[:, 2]**2
    )
)


# ============================================================ #
#                      TRACKING ERRORS                         #
# ============================================================ #

tracking_error = (
    reference_history
    - true_history
)

tracking_error[:, 2] = np.array([
    wrap(angle)
    for angle in tracking_error[:, 2]
])

tracking_position_rmse = np.sqrt(
    np.mean(
        tracking_error[:, 0]**2
        + tracking_error[:, 1]**2
    )
)

tracking_heading_rmse = np.sqrt(
    np.mean(
        tracking_error[:, 2]**2
    )
)

final_tracking_position_error = np.linalg.norm(
    tracking_error[-1, :2]
)

final_tracking_heading_error = (
    tracking_error[-1, 2]
)


# ============================================================ #
#                           RESULTS                            #
# ============================================================ #

camera_detections = np.sum(
    camera_visible_history
)

print(
    "\n================ EKF + LQR RESULTS ================"
)

print("\nLocalisation:")

print(
    f"Position RMSE: "
    f"{localisation_position_rmse:.4f} m"
)

print(
    f"Heading RMSE: "
    f"{np.rad2deg(localisation_heading_rmse):.2f} deg"
)


print("\nTrajectory tracking:")

print(
    f"Position RMSE: "
    f"{tracking_position_rmse:.4f} m"
)

print(
    f"Heading RMSE: "
    f"{np.rad2deg(tracking_heading_rmse):.2f} deg"
)

print(
    f"Final position error: "
    f"{final_tracking_position_error:.4f} m"
)

print(
    f"Final heading error: "
    f"{np.rad2deg(final_tracking_heading_error):.2f} deg"
)


print("\nCamera:")

print(
    f"Detections: "
    f"{camera_detections}/{STEPS}"
)

print(
    f"Detection rate: "
    f"{100 * camera_detections / STEPS:.1f}%"
)


print("\nFinal reference pose:")
print(
    reference_history[-1]
)

print("\nFinal true pose:")
print(
    true_history[-1]
)

print("\nFinal EKF pose:")
print(
    estimate_history[-1]
)

print("\nMaximum commanded wheel speed:")

print(
    f"{np.max(np.abs(wheel_command_history)):.3f} rad/s"
)