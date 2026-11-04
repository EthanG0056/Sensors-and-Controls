from pathlib import Path

import mujoco
import numpy as np

from estimation.ekf import (
    TurtleBotEKF,
    wheel_odometry,
    wrap
)

from simulation.turtlebot import (
    get_wheel_speeds,
    get_true_pose
)


# ========================== SETUP ========================== #

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    ROOT
    / "simulation"
    / "robotis_tb3"
    / "scene_turtlebot3_burger_gain_test.xml"
)

model = mujoco.MjModel.from_xml_path(
    str(MODEL_PATH)
)

data = mujoco.MjData(model)

DT = 0.05

substeps = round(
    DT / model.opt.timestep
)

assert np.isclose(
    substeps * model.opt.timestep,
    DT
)


# ========================== ROBOT ========================== #

left_actuator = model.actuator("wheel_left").id
right_actuator = model.actuator("wheel_right").id


# ========================== EKF ========================== #

initial_state = np.array([
    0.0,
    0.0,
    0.0
])

P0 = np.diag([
    0.05**2,
    0.05**2,
    np.deg2rad(5.0)**2
])

Q = np.diag([
    0.005**2,
    0.005**2,
    np.deg2rad(0.5)**2
])

R = np.diag([
    0.03**2,
    np.deg2rad(1.5)**2
])

ekf = TurtleBotEKF(
    initial_state,
    P0,
    Q,
    R
)


# ========================== HISTORY ========================== #

true_history = []
estimate_history = []


# ========================== SIMULATION ========================== #

steps = 400

for step in range(steps):

    # Simple curved motion for initial test
    v_command = 0.08
    omega_command = 0.15

    left_command = (
        v_command
        - 0.5 * 0.160 * omega_command
    ) / 0.033

    right_command = (
        v_command
        + 0.5 * 0.160 * omega_command
    ) / 0.033

    data.ctrl[left_actuator] = left_command
    data.ctrl[right_actuator] = right_command

    # Advance MuJoCo for one EKF sample interval
    for _ in range(substeps):
        mujoco.mj_step(
            model,
            data
        )

    # Actual simulated wheel velocities
    wheel_speeds = get_wheel_speeds(
        model,
        data
    )

    control = wheel_odometry(
        wheel_speeds[0],
        wheel_speeds[1]
    )

    # EKF prediction from wheel odometry only
    ekf.predict(
        control,
        DT
    )

    # Ground truth for evaluation only
    true_pose = get_true_pose(
        model,
        data
    )

    true_history.append(
        true_pose.copy()
    )

    estimate_history.append(
        ekf.state.copy()
    )


# ========================== RESULTS ========================== #

true_history = np.array(
    true_history
)

estimate_history = np.array(
    estimate_history
)

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


print(
    f"Position RMSE: "
    f"{position_rmse:.4f} m"
)

print(
    f"Heading RMSE: "
    f"{np.rad2deg(heading_rmse):.2f} deg"
)

print("\nFinal true pose:")
print(
    true_history[-1]
)

print("\nFinal odometry/EKF pose:")
print(
    estimate_history[-1]
)