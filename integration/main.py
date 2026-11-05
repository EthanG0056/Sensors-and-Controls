from pathlib import Path
import sys
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pathlib import Path
import mujoco
import mujoco.viewer
import numpy as np
import time

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

from sensing.lidar import (
    get_lidar_ranges,
    get_front_obstacle,
)

from control.command_gateway import CommandGateway

from control.terminal_controller import (
    TerminalParameters,
)

from control.staged_terminal_controller import (
    StagedTrajectoryTerminalController,
)

from control.estimated_pose_interface import (
    PoseEstimate,
    InterfacePolicy,
    EstimatedPoseInterface,
    ClearanceGrant,
)

from control.trajectory import (
    reference,
    DURATIONS,
)


# ============================================================ #
#                         PARAMETERS                           #
# ============================================================ #

DT = 0.05
SHOW_VIEWER = True
RUN_REALTIME = True

TRAJECTORY_NAME = "straight"

DURATION = DURATIONS[
    TRAJECTORY_NAME
]

TERMINAL_TIME = 45.0

TOTAL_DURATION = (
    DURATION
    + TERMINAL_TIME
)

STEPS = int(
    TOTAL_DURATION / DT
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


NAVIGATION_SPEED = 0.08
NAVIGATION_TARGET_X = 1.85


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
        print("QUALITY FAIL: x std =", std[0])
        return False

    if std[1] > 0.15:
        print("QUALITY FAIL: y std =", std[1])
        return False

    if std[2] > np.deg2rad(15.0):
        print(
            "QUALITY FAIL: heading std =",
            np.rad2deg(std[2]),
            "deg",
        )
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

    if position_jump > 0.15:
        print(
            "QUALITY FAIL: position jump =",
            position_jump,
            "m",
        )
        print("previous =", previous.pose)
        print("current  =", current.pose)
        return False

    if heading_jump > np.deg2rad(20.0):
        print(
            "QUALITY FAIL: heading jump =",
            np.rad2deg(heading_jump),
            "deg",
        )
        print("previous =", previous.pose)
        print("current  =", current.pose)
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
#                       MUJOCO VIEWER                          #
# ============================================================ #

viewer = None

if SHOW_VIEWER:

    viewer = mujoco.viewer.launch_passive(
        model,
        data,
    )

    # Turn off LiDAR lines
    viewer.opt.flags[
        mujoco.mjtVisFlag.mjVIS_RANGEFINDER
    ] = False

    # Initial camera position for the navigation scene.
    viewer.cam.lookat[:] = [
        1.0,
        0.0,
        0.0,
    ]

    viewer.cam.distance = 2.8
    viewer.cam.azimuth = 120
    viewer.cam.elevation = -35


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

# Marker 0 - charging station

marker_0_geom_id = model.geom(
    "aruco_marker"
).id

MARKER_POSITION = data.geom_xpos[
    marker_0_geom_id,
    :2
].copy()


# Marker 1 - room landmark

marker_1_geom_id = model.geom(
    "aruco_marker_1"
).id

MARKER_1_POSITION = data.geom_xpos[
    marker_1_geom_id,
    :2
].copy()


# Known landmark map

MARKER_POSITIONS = {
    0: MARKER_POSITION,
    1: MARKER_1_POSITION,
}


print(
    "Marker 0 position:",
    MARKER_POSITION
)

print(
    "Marker 1 position:",
    MARKER_1_POSITION
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

DOCKING_DISTANCE = 0.1

goal = np.array([
    MARKER_POSITION[0] - DOCKING_DISTANCE,
    MARKER_POSITION[1],
    0.0,
])

print(
    "Docking goal:",
    goal
)


# ============================================================ #
#                 OBSTACLE AVOIDANCE SUPERVISOR                #
# ============================================================ #

class ObstacleAvoidanceSupervisor:

    def __init__(
        self,
        goal,
        duration,
        parameters,
    ):

        self.normal_controller = (
            StagedTrajectoryTerminalController(
                goal,
                duration,
                parameters=parameters,
            )
        )

        # Expose the normal controller contract expected
        # by EstimatedPoseInterface.
        self.end_time = self.normal_controller.end_time
        self.tracker = self.normal_controller.tracker
        self.terminal = self.normal_controller.terminal

    def command(
        self,
        estimated_pose,
        reference_sample,
        reference_time,
    ):

        global avoidance_active
        global avoidance_phase
        global avoidance_direction
        global docking_active
        global navigation_time
        global avoidance_phase_start


        # ---------------------------------------- #
        # DOCKING
        # ---------------------------------------- #

        if docking_active:

            terminal_time = (
                self.normal_controller.end_time
                + DT
            )

            return self.normal_controller.command(
                estimated_pose,
                reference_sample,
                terminal_time,
            )


        # ---------------------------------------- #
        # OBSTACLE AVOIDANCE
        # ---------------------------------------- #

        if avoidance_active:

            wheel_radius = 0.033
            wheel_track = 0.160

            # Turn away from obstacle.
            if avoidance_phase == "turn_out":

                v = 0.0

                omega = (
                    avoidance_direction
                    * 0.35
                )

            # Drive diagonally past obstacle.
            elif avoidance_phase == "drive_past":

                v = 0.08
                omega = 0.0

            # Gradually curve back towards original path.
            elif avoidance_phase == "rejoin":

                x, y, theta = estimated_pose

                v = 0.07

                omega = (
                    -0.6 * y
                    -0.8 * theta
                )

                omega = np.clip(
                    omega,
                    -0.18,
                    0.18,
                )

            else:

                v = 0.0
                omega = 0.0

            left = (
                v
                - omega * wheel_track / 2.0
            ) / wheel_radius

            right = (
                v
                + omega * wheel_track / 2.0
            ) / wheel_radius

            wheels = np.array([
                left,
                right,
            ])

            body = np.array([
                v,
                omega,
            ])

            return (
                body,
                wheels,
                "tracking_avoidance",
                False,
            )


        # ---------------------------------------- #
        # NORMAL LQR / TERMINAL CONTROL
        # ---------------------------------------- #

        controller_time = min(
            navigation_time,
            DURATION - DT,
        )

        return self.normal_controller.command(
            estimated_pose,
            reference_sample,
            controller_time,
        )

    
def supervisor_factory():

    return ObstacleAvoidanceSupervisor(
        goal,
        DURATION,
        parameters,
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


marker_detection_counts = {
    0: 0,
    1: 0,
}


#================== Obstacle avoidance state ==================#

AVOID_ENTER_DISTANCE = 0.20

avoidance_active = False
avoidance_phase = None
avoidance_direction = 0.0
avoidance_phase_start = 0.0

TURN_TIME = 2
DRIVE_TIME = 5
REJOIN_TIME = 3.0

DOCKING_TRIGGER_DISTANCE = 0.90
docking_active = False

navigation_time = 0.0

# ============================================================ #
#                       INTEGRATION LOOP                       #
# ============================================================ #

try:

    for step in range(STEPS):

        loop_start_time = time.perf_counter()

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

        # Update the graphical MuJoCo viewer.
        if viewer is not None:

            if not viewer.is_running():
                break

            viewer.sync()

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

        camera_result = get_camera_measurement(
            model,
            data,
            "front_camera",
        )

        camera_visible = (
            camera_result is not None
        )

        if camera_visible:

            marker_id, measurement = (
                camera_result
            )

            if marker_id in MARKER_POSITIONS:

                marker_detection_counts[
                    marker_id
                ] += 1

                marker_position = (
                    MARKER_POSITIONS[
                        marker_id
                    ]
                )

                ekf.update_camera(
                    measurement,
                    marker_position,
                    CAMERA_OFFSET,
                    CAMERA_YAW_OFFSET,
                )


        # ---------------------------------------------------- #
        # 5. DOCKING ACTIVATION
        # ---------------------------------------------------- #

        distance_to_charger = np.linalg.norm(
            ekf.state[:2]
            - MARKER_POSITION
        )

        if (
            not docking_active
            and distance_to_charger
            < DOCKING_TRIGGER_DISTANCE
        ):

            docking_active = True
            avoidance_active = False

            print(
                f"\nDocking activated at "
                f"{distance_to_charger:.2f} m "
                f"from marker 0"
            )


        # ---------------------------------------------------- #
        # 6. LIDAR OBSTACLE DETECTION
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
        # 7. OBSTACLE AVOIDANCE STATE
        # ---------------------------------------------------- #

        theta = ekf.state[2]

        # Start avoidance.
        if (
            not docking_active
            and not avoidance_active
            and obstacle_distance < AVOID_ENTER_DISTANCE
        ):

            avoidance_active = True
            avoidance_phase = "turn_out"
            avoidance_phase_start = state_time

            # Obstacle on left -> go right.
            if obstacle_angle > 0.0:
                avoidance_direction = -1.0

            # Obstacle on right -> go left.
            else:
                avoidance_direction = 1.0


        # ---------------------------------------------------- #
        # ADVANCE AVOIDANCE PHASE
        # ---------------------------------------------------- #

        if avoidance_active:

            phase_time = (
                state_time
                - avoidance_phase_start
            )

            # Finished turning away.
            if (
                avoidance_phase == "turn_out"
                and phase_time >= TURN_TIME
            ):

                avoidance_phase = "drive_past"
                avoidance_phase_start = state_time

            # Finished driving past obstacle.
            elif (
                avoidance_phase == "drive_past"
                and phase_time >= DRIVE_TIME
            ):

                avoidance_phase = "rejoin"
                avoidance_phase_start = state_time

            # Finished gradually rejoining path.
            elif (
                avoidance_phase == "rejoin"
                and phase_time >= REJOIN_TIME
            ):

                avoidance_active = False
                avoidance_phase = None

                # Resynchronise navigation reference.
                navigation_time = max(
                    navigation_time,
                    ekf.state[0] / NAVIGATION_SPEED,
                )

                navigation_time = min(
                    navigation_time,
                    NAVIGATION_TARGET_X / NAVIGATION_SPEED,
                )

        # Only advance trajectory time while normal LQR
        # navigation is actually active.
        if (
            not avoidance_active
            and not docking_active
        ):

            navigation_time += DT


        # ---------------------------------------------------- #
        # 8. CREATE ESTIMATED-STATE SNAPSHOT
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
        # 9. SEND ESTIMATE THROUGH REAL CONTROL INTERFACE
        # ---------------------------------------------------- #
        
        def reference_provider(t):

            base_reference = reference(
                TRAJECTORY_NAME,
                min(
                    navigation_time,
                    DURATION,
                ),
            )

            x_reference = min(
                NAVIGATION_SPEED * navigation_time,
                NAVIGATION_TARGET_X,
            )

            if x_reference < NAVIGATION_TARGET_X:

                reference_speed = NAVIGATION_SPEED

            else:

                reference_speed = 0.0

            return type(base_reference)(
                pose=np.array([
                    x_reference,
                    0.0,
                    0.0,
                ]),
                input=np.array([
                    reference_speed,
                    0.0,
                ]),
            )


        clearance = None

        if docking_active:

            clearance = ClearanceGrant(
                grant_id="terminal_test_clearance",
                task_id=TASK_ID,
                frame_id=WORLD_FRAME,
                estimator_epoch=ESTIMATOR_EPOCH,
                valid_from=0.0,
                expires_at=TOTAL_DURATION + 5.0,
                approved=True,
                covers_terminal_manoeuvre=True,
                revoked=False,
            )


        result = interface.step(
            snapshot,
            reference_provider,
            clearance=clearance,
            terminal_active=docking_active,
        )


        if not result.accepted:

            print(
                "\nInterface rejected command:"
            )

            print(
                result.reason
            )

            break


        # ---------------------------------------------------- #
        # 10. GROUND TRUTH - EVALUATION ONLY
        # ---------------------------------------------------- #

        true_pose = get_true_pose(
            model,
            data
        )

        current_reference = reference_provider(
            state_time,
        )


        # ---------------------------------------------------- #
        # 11. LOGGING
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


        # ---------------------------------------------------- #
        # 12. CHECK FOR COMPLETED DOCKING
        # ---------------------------------------------------- #

        if result.estimated_stop_qualified:

            print(
                "\nDocking stop qualified."
            )

            break


        # ---------------------------------------------------- #
        # 13. RUN SIM IN REAL TIME
        # ---------------------------------------------------- #

        if RUN_REALTIME:

            elapsed = (
                time.perf_counter()
                - loop_start_time
            )

            remaining = (
                DT
                - elapsed
            )

            if remaining > 0.0:

                time.sleep(
                    remaining
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
#                    NAVIGATION METRICS                        #
# ============================================================ #

tracking_samples = int(
    DURATION / DT
)

tracking_true = true_history[
    :tracking_samples
]

tracking_reference = reference_history[
    :tracking_samples
]

tracking_error = (
    tracking_reference
    - tracking_true
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


# ============================================================ #
#                       DOCKING METRICS                        #
# ============================================================ #

true_docking_error = (
    goal
    - true_history[-1]
)

true_docking_error[2] = wrap(
    true_docking_error[2]
)

estimated_docking_error = (
    goal
    - estimate_history[-1]
)

estimated_docking_error[2] = wrap(
    estimated_docking_error[2]
)

true_docking_position_error = np.linalg.norm(
    true_docking_error[:2]
)

true_docking_heading_error = abs(
    true_docking_error[2]
)

estimated_docking_position_error = np.linalg.norm(
    estimated_docking_error[:2]
)

estimated_docking_heading_error = abs(
    estimated_docking_error[2]
)

docking_samples = max(
    0,
    len(true_history)
    - tracking_samples
)

docking_duration = (
    docking_samples
    * DT
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


# ============================================================ #
#                     LOCALISATION RESULTS                     #
# ============================================================ #

print(
    "\n---------------- LOCALISATION ----------------"
)

print(
    f"Position RMSE: "
    f"{localisation_position_rmse:.4f} m"
)

print(
    f"Heading RMSE: "
    f"{np.rad2deg(localisation_heading_rmse):.2f} deg"
)


# ============================================================ #
#                 NAVIGATION TRACKING RESULTS                  #
# ============================================================ #

print(
    "\n------------- NAVIGATION TRACKING -------------"
)

print(
    f"Position RMSE: "
    f"{tracking_position_rmse:.4f} m"
)

print(
    f"Heading RMSE: "
    f"{np.rad2deg(tracking_heading_rmse):.2f} deg"
)


# ============================================================ #
#                       DOCKING RESULTS                        #
# ============================================================ #

print(
    "\n------------------ DOCKING -------------------"
)

print(
    f"Goal pose: "
    f"{goal}"
)

print(
    f"True final position error: "
    f"{true_docking_position_error:.4f} m"
)

print(
    f"True final heading error: "
    f"{np.rad2deg(true_docking_heading_error):.2f} deg"
)

print(
    f"Estimated final position error: "
    f"{estimated_docking_position_error:.4f} m"
)

print(
    f"Estimated final heading error: "
    f"{np.rad2deg(estimated_docking_heading_error):.2f} deg"
)

print(
    f"Docking duration: "
    f"{docking_duration:.2f} s"
)


# ============================================================ #
#                    SENSOR RESULTS                            #
# ============================================================ #

print(
    "\n------------------- CAMERA -------------------"
)

print(
    f"Detections: "
    f"{np.sum(camera_visible_history)}"
    f"/{len(camera_visible_history)}"
)

print(
    f"Marker 0 detections: "
    f"{marker_detection_counts[0]}"
)

print(
    f"Marker 1 detections: "
    f"{marker_detection_counts[1]}"
)

print(
    "\n-------------------- LIDAR -------------------"
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
    f"Obstacle avoidance active samples: "
    f"{np.sum(avoidance_history)}"
)


# ============================================================ #
#                    FINAL SYSTEM STATE                        #
# ============================================================ #

print(
    "\n--------------- FINAL SYSTEM STATE ---------------"
)

print(
    "Final navigation reference:"
)

print(
    reference_history[-1]
)

print(
    "Docking goal:"
)

print(
    goal
)

print(
    "Final true pose:"
)

print(
    true_history[-1]
)

print(
    "Final EKF pose:"
)

print(
    estimate_history[-1]
)


# ============================================================ #
#                    HOLD FINAL VIEWER                         #
# ============================================================ #

if viewer is not None:

    print(
        "\nSimulation complete."
        "\nClose the MuJoCo window to exit."
    )

    while viewer.is_running():

        viewer.sync()

        time.sleep(
            0.05
        )