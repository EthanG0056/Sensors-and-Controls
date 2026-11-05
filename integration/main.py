from pathlib import Path
import sys
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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

from control.command_gateway import CommandGateway

from control.estimated_pose_interface import (
    PoseEstimate,
    InterfacePolicy,
    EstimatedPoseInterface,
)

from control.terminal_controller import (
    TrajectoryTerminalController,
    TerminalParameters,
)

from control.trajectory import (
    reference,
    DURATIONS,
)


# ============================================================ #
#                         PARAMETERS                           #
# ============================================================ #

DT = 0.05

TRAJECTORY_NAME = "straight"

DURATION = DURATIONS[
    TRAJECTORY_NAME
]

STEPS = int(
    DURATION / DT
)

CAMERA_OFFSET = (
    0.05,
    0.0,
)

CAMERA_YAW_OFFSET = 0.0

ESTIMATOR_EPOCH = "integration_run_1"

WORLD_FRAME = "map"

BODY_FRAME = "base"

TASK_ID = "straight_tracking_test"


# ============================================================ #
#                       INTEGRATION CLOCK                      #
# ============================================================ #

class IntegrationClock:

    def __init__(self):
        self.time = 0.0

    def __call__(self):
        return self.time

    def advance(self, dt):
        self.time += dt


integration_clock = IntegrationClock()


# ============================================================ #
#                         MODEL PATH                           #
# ============================================================ #

ROOT = Path(__file__).resolve().parents[1]

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

    left_joint = model.joint(
        "wheel_left"
    ).id

    right_joint = model.joint(
        "wheel_right"
    ).id

    left_dof = model.jnt_dofadr[
        left_joint
    ]

    right_dof = model.jnt_dofadr[
        right_joint
    ]

    left_speed = data.qvel[
        left_dof
    ]

    right_speed = data.qvel[
        right_dof
    ]

    return (
        left_speed,
        right_speed,
    )


def get_true_pose(model, data):

    # Ground truth is ONLY used for evaluation.

    base_joint = model.joint(
        "base_joint"
    ).id

    address = model.jnt_qposadr[
        base_joint
    ]

    x = data.qpos[
        address
    ]

    y = data.qpos[
        address + 1
    ]

    qw = data.qpos[
        address + 3
    ]

    qx = data.qpos[
        address + 4
    ]

    qy = data.qpos[
        address + 5
    ]

    qz = data.qpos[
        address + 6
    ]

    yaw = np.arctan2(
        2.0 * (
            qw * qz
            + qx * qy
        ),
        1.0 - 2.0 * (
            qy**2
            + qz**2
        ),
    )

    return np.array([
        x,
        y,
        wrap(yaw),
    ])


# ============================================================ #
#                         LIDAR HELPERS                         #
# ============================================================ #

NUM_LIDAR_RAYS = 36

LIDAR_ANGLES = np.deg2rad(
    np.arange(
        0,
        360,
        10,
    )
)

# Wrap angles into [-pi, pi)
LIDAR_ANGLES = (
    LIDAR_ANGLES + np.pi
) % (
    2.0 * np.pi
) - np.pi


def get_lidar_ranges(
    model,
    data,
):

    ranges = np.full(
        NUM_LIDAR_RAYS,
        np.inf,
    )

    for i in range(
        NUM_LIDAR_RAYS
    ):

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

        if distance >= 0.0:

            ranges[i] = distance

    return ranges


def get_front_obstacle(
    ranges,
    half_angle_deg=40.0,
):

    half_angle = np.deg2rad(
        half_angle_deg
    )

    front_mask = (
        np.abs(
            LIDAR_ANGLES
        )
        <= half_angle
    )

    front_ranges = np.where(
        front_mask,
        ranges,
        np.inf,
    )

    closest_index = np.argmin(
        front_ranges
    )

    closest_distance = front_ranges[
        closest_index
    ]

    if not np.isfinite(
        closest_distance
    ):

        return None

    return (
        closest_distance,
        LIDAR_ANGLES[
            closest_index
        ],
    )


# ============================================================ #
#                  LOCALISATION QUALITY CHECK                  #
# ============================================================ #

def localisation_quality(
    previous,
    current,
):

    # Diagnostic integration limits.
    #
    # These are deliberately conservative compared with the
    # localisation performance already demonstrated in the
    # MuJoCo benchmarks.

    std = np.sqrt(
        np.diag(
            current.covariance
        )
    )

    # Reject very uncertain localisation.
    if std[0] > 0.15:
        return False

    if std[1] > 0.15:
        return False

    if std[2] > np.deg2rad(15.0):
        return False

    # On the first sample there is no previous accepted estimate.
    if previous is None:
        return True

    position_jump = np.linalg.norm(
        current.pose[:2]
        - previous.pose[:2]
    )

    heading_jump = abs(
        wrap(
            current.pose[2]
            - previous.pose[2]
        )
    )

    # Reject implausibly large estimate jumps between 20 Hz samples.
    if position_jump > 0.05:
        return False

    if heading_jump > np.deg2rad(20.0):
        return False

    return True


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
#                      SIMULATION TIMING                       #
# ============================================================ #

simulation_dt = model.opt.timestep

substeps = round(
    DT
    / simulation_dt
)

if not np.isclose(
    substeps * simulation_dt,
    DT,
):
    raise ValueError(
        "DT must be an integer multiple "
        "of the MuJoCo timestep"
    )


# ============================================================ #
#                          ACTUATORS                           #
# ============================================================ #

left_actuator = model.actuator(
    "wheel_left"
).id

right_actuator = model.actuator(
    "wheel_right"
).id


def write_wheels(wheels):

    # This function is now the ONLY place in this integration
    # program allowed to write actuator commands.

    left, right = wheels

    data.ctrl[
        left_actuator
    ] = left

    data.ctrl[
        right_actuator
    ] = right


# ============================================================ #
#                       ARUCO MARKER                           #
# ============================================================ #

configure_marker_size_from_geom(
    model,
    "aruco_marker",
    (1, 2),
)

marker_geom_id = model.geom(
    "aruco_marker"
).id

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
#                     CONTROLLER SETUP                         #
# ============================================================ #

parameters = TerminalParameters(
    sample_time=DT
)

goal = reference(
    TRAJECTORY_NAME,
    DURATION,
).pose.copy()


def supervisor_factory():

    return TrajectoryTerminalController(
        goal=goal,
        end_time=DURATION,
        parameters=parameters,
    )


# ============================================================ #
#                     COMMAND GATEWAY                          #
# ============================================================ #

gateway = CommandGateway(
    write_wheels,
    max_lease_s=0.25,
    clock=integration_clock,
)


# ============================================================ #
#                   ESTIMATED POSE POLICY                      #
# ============================================================ #

policy = InterfacePolicy(
    frame_id=WORLD_FRAME,
    body_frame_id=BODY_FRAME,

    # Wall-clock freshness limits for this integration test.
    max_age_s=0.50,

    # Wheel commands expire quickly if updates stop.
    command_ttl_s=0.20,

    # State time must remain on the 20 Hz controller grid.
    clock_grid_tolerance_s=1e-6,

    sample_time=DT,

    quality_check=localisation_quality,
)


# ============================================================ #
#                   ESTIMATED POSE INTERFACE                   #
# ============================================================ #

interface = EstimatedPoseInterface(
    gateway,
    supervisor_factory,
    policy,
    goal=goal,
    task_id=TASK_ID,
    terminal_start_s=DURATION,
    parameters=parameters,
    clock=integration_clock,
)


# ============================================================ #
#                         ARM SYSTEM                           #
# ============================================================ #

# The first EKF estimate is produced after the first 0.05 s
# simulation interval, so the state-time origin is DT.

arm_result = interface.arm(
    epoch=ESTIMATOR_EPOCH,
    state_time_origin=DT,
)

if not arm_result.accepted:

    raise RuntimeError(
        "Could not arm interface: "
        + arm_result.reason
    )

print(
    "Interface armed successfully."
)


# ============================================================ #
#                           HISTORY                            #
# ============================================================ #

true_history = []

estimate_history = []

reference_history = []

camera_visible_history = []

controller_mode_history = []

obstacle_distance_history = []

obstacle_angle_history = []

avoidance_history = []


# ============================================================ #
#                       INTEGRATION LOOP                       #
# ============================================================ #

try:

    for step in range(STEPS):


        # ---------------------------------------------------- #
        # 1. ADVANCE MUJOCO USING PREVIOUS GATEWAY COMMAND
        # ---------------------------------------------------- #

        for _ in range(substeps):

            mujoco.mj_step(
                model,
                data
            )

        integration_clock.advance(
            DT
        )
        
        state_time = (
            step + 1
        ) * DT


        # ---------------------------------------------------- #
        # 2. READ MEASURED WHEEL VELOCITIES
        # ---------------------------------------------------- #

        left_measured, right_measured = (
            get_wheel_speeds(
                model,
                data
            )
        )


        # ---------------------------------------------------- #
        # 3. EKF PREDICTION
        # ---------------------------------------------------- #

        body_motion = wheel_odometry(
            left_measured,
            right_measured
        )

        ekf.predict(
            body_motion,
            DT
        )


        # ---------------------------------------------------- #
        # 4. EKF CAMERA UPDATE
        # ---------------------------------------------------- #

        measurement = get_camera_measurement(
            model,
            data,
            "front_camera",
        )

        camera_visible = (
            measurement is not None
        )

        if camera_visible:

            ekf.update_camera(
                measurement,
                MARKER_POSITION,
                CAMERA_OFFSET,
                CAMERA_YAW_OFFSET,
            )


        # ---------------------------------------------------- #
        # 5. LIDAR OBSTACLE DETECTION
        # ---------------------------------------------------- #

        lidar_ranges = get_lidar_ranges(
            model,
            data,
        )

        front_obstacle = get_front_obstacle(
            lidar_ranges
        )

        if front_obstacle is not None:

            obstacle_distance, obstacle_angle = (
                front_obstacle
            )

        else:

            obstacle_distance = np.inf
            obstacle_angle = 0.0


        # ---------------------------------------------------- #
        # 6. SIMPLE OBSTACLE AVOIDANCE OVERRIDE
        # ---------------------------------------------------- #

        AVOID_DISTANCE = 0.4

        avoidance_active = (
            obstacle_distance < AVOID_DISTANCE
        )


        # ---------------------------------------------------- #
        # 7. CREATE ESTIMATED-STATE SNAPSHOT
        # ---------------------------------------------------- #

        snapshot = PoseEstimate(
            pose=ekf.state.copy(),

            covariance=ekf.P.copy(),

            state_time=state_time,

            sample_monotonic=integration_clock(),

            sequence=step,

            epoch=ESTIMATOR_EPOCH,

            frame_id=WORLD_FRAME,

            body_frame_id=BODY_FRAME,

            localisation_valid=True,

            validity_reason="EKF operating normally",
        )

        
        # ---------------------------------------------------- #
        # 8. SEND ESTIMATE THROUGH REAL CONTROL INTERFACE
        # ---------------------------------------------------- #

        def reference_provider(t):

            normal_reference = reference(
                TRAJECTORY_NAME,
                t
            )

            if not avoidance_active:
                return normal_reference

            # Current estimated robot pose.
            x, y, theta = ekf.state

            # Unit vectors in the robot's forward and left directions.
            forward = np.array([
                np.cos(theta),
                np.sin(theta),
            ])

            left = np.array([
                -np.sin(theta),
                np.cos(theta),
            ])

            # Move forward while stepping sideways away from the obstacle.
            forward_offset = 0.25
            lateral_offset = 0.20

            # Obstacle on left -> detour right.
            if obstacle_angle > 0.0:

                target_xy = (
                    ekf.state[:2]
                    + forward_offset * forward
                    - lateral_offset * left
                )

                target_heading = wrap(
                    theta
                    - np.deg2rad(20.0)
                )

            # Obstacle on right -> detour left.
            else:

                target_xy = (
                    ekf.state[:2]
                    + forward_offset * forward
                    + lateral_offset * left
                )

                target_heading = wrap(
                    theta
                    + np.deg2rad(20.0)
                )

            avoidance_pose = np.array([
                target_xy[0],
                target_xy[1],
                target_heading,
            ])

            return type(normal_reference)(
                pose=avoidance_pose,
                input=np.array([
                    0.05,
                    0.0,
                ]),
            )


        result = interface.step(
            snapshot,
            reference_provider,
        )


        if not result.accepted:

            print(
                "\nInterface rejected command:"
            )

            print(
                result.reason
            )

            break

        # if avoidance_active:

        #     wheel_radius = 0.033
        #     track_width = 0.160

        #     left_wheel = (
        #         avoid_v
        #         - 0.5 * track_width * avoid_w
        #     ) / wheel_radius

        #     right_wheel = (
        #         avoid_v
        #         + 0.5 * track_width * avoid_w
        #     ) / wheel_radius

        #     wheels = np.array([
        #         left_wheel,
        #         right_wheel,
        #     ])

        #     wheels = np.clip(
        #         wheels,
        #         -6.67,
        #         6.67,
        #     )

        #     write_wheels(
        #         wheels
        #     )


        # ---------------------------------------------------- #
        # 9. GROUND TRUTH - EVALUATION ONLY
        # ---------------------------------------------------- #

        true_pose = get_true_pose(
            model,
            data
        )

        current_reference = reference(
            TRAJECTORY_NAME,
            state_time,
        )


        # ---------------------------------------------------- #
        # 10. LOGGING
        # ---------------------------------------------------- #

        true_history.append(
            true_pose.copy()
        )

        estimate_history.append(
            ekf.state.copy()
        )

        reference_history.append(
            current_reference.pose.copy()
        )

        camera_visible_history.append(
            camera_visible
        )

        controller_mode_history.append(
            result.controller_mode
        )

        obstacle_distance_history.append(
            obstacle_distance
        )

        obstacle_angle_history.append(
            obstacle_angle
        )

        avoidance_history.append(
            avoidance_active
        )


finally:

    stop_result = interface.disarm(
        "integration_complete"
    )

    close_camera_renderer()

    print(
        "\nSystem disarmed:"
    )

    print(
        stop_result.reason
    )


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

camera_visible_history = np.array(
    camera_visible_history
)

obstacle_distance_history = np.array(
    obstacle_distance_history
)

obstacle_angle_history = np.array(
    obstacle_angle_history
)

avoidance_history = np.array(
    avoidance_history
)

# ============================================================ #
#                         CHECK RUN                            #
# ============================================================ #

if len(true_history) == 0:

    raise RuntimeError(
        "No control samples completed."
    )


# ============================================================ #
#                    LOCALISATION ERROR                        #
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
#                      TRACKING ERROR                          #
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

final_position_error = np.linalg.norm(
    tracking_error[-1, :2]
)


# ============================================================ #
#                           RESULTS                            #
# ============================================================ #

print(
    "\n================ FULL INTERFACE RESULTS ================"
)

print(
    f"Completed samples: "
    f"{len(true_history)}/{STEPS}"
)

print(
    "\nLocalisation:"
)

print(
    f"Position RMSE: "
    f"{localisation_position_rmse:.4f} m"
)

print(
    f"Heading RMSE: "
    f"{np.rad2deg(localisation_heading_rmse):.2f} deg"
)

print(
    "\nTracking:"
)

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
    f"{final_position_error:.4f} m"
)

print(
    "\nCamera:"
)

print(
    f"Detections: "
    f"{np.sum(camera_visible_history)}"
    f"/{len(camera_visible_history)}"
)

print(
    "\nFinal controller mode:"
)

print(
    controller_mode_history[-1]
)

print(
    "\nFinal reference pose:"
)

print(
    reference_history[-1]
)

print(
    "\nFinal true pose:"
)

print(
    true_history[-1]
)

print(
    "\nFinal EKF pose:"
)

print(
    estimate_history[-1]
)


#============ Closest Obstacle =============#

print(
    "\nLiDAR:"
)

finite_mask = np.isfinite(
    obstacle_distance_history
)

if np.any(finite_mask):

    valid_indices = np.where(
        finite_mask
    )[0]

    closest_index = valid_indices[
        np.argmin(
            obstacle_distance_history[
                valid_indices
            ]
        )
    ]

    print(
        f"Closest front obstacle: "
        f"{obstacle_distance_history[closest_index]:.3f} m"
    )

    print(
        f"Obstacle angle at closest point: "
        f"{np.rad2deg(obstacle_angle_history[closest_index]):.1f} deg"
    )

else:

    print(
        "No front obstacles detected."
    )

print(
    "\nObstacle avoidance:"
)

print(
    f"Active samples: "
    f"{np.sum(avoidance_history)}"
)