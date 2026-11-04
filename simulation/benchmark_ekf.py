# ========================== EKF BENCHMARK ========================== #

import numpy as np

from estimation.ekf import (
    TurtleBotEKF,
    wheel_odometry,
    motion_model,
    camera_model,
    wrap
)


DT = 0.05
STEPS = 400

WHEEL_RADIUS = 0.033
WHEEL_SEPARATION = 0.160

MARKER_POSITIONS = [
    np.array([2.0, 1.5]),
    np.array([-1.0, 2.0])
]


# ========================== FILTER SETUP ========================== #

initial_true_state = np.array([0.0, 0.0, 0.0])

initial_estimate = np.array([
    0.10,
    -0.08,
    0.0
])

P0 = np.diag([
    0.20**2,
    0.20**2,
    np.deg2rad(10.0)**2
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
    initial_estimate,
    P0,
    Q,
    R
)


# ========================== SIMULATION ========================== #

rng = np.random.default_rng(1)

true_state = initial_true_state.copy()

# Log History
true_history = []
estimate_history = []
camera_visible_history = []
covariance_history = []

for step in range(STEPS):

    # Commanded body motion
    v_command = 0.08
    omega_command = 0.15

    # Convert body command to ideal wheel speeds
    left_true = (
        v_command
        - 0.5 * WHEEL_SEPARATION * omega_command
    ) / WHEEL_RADIUS

    right_true = (
        v_command
        + 0.5 * WHEEL_SEPARATION * omega_command
    ) / WHEEL_RADIUS

    # Simulated encoder noise
    left_measured = left_true + rng.normal(0.0, 0.03)
    right_measured = right_true + rng.normal(0.0, 0.03)

    control = wheel_odometry(
        left_measured,
        right_measured
    )

    # True robot movement
    true_state = motion_model(
        true_state,
        np.array([v_command, omega_command]),
        DT
    )

    # EKF prediction
    ekf.predict(
        control,
        DT
    )

    # Camera unavailable between steps 120 and 220
    camera_visible = not (120 <= step < 220)

    if camera_visible:

        for marker_position in MARKER_POSITIONS:

            measurement = camera_model(
                true_state,
                marker_position
            )

            measurement += np.array([
                rng.normal(0.0, 0.03),
                rng.normal(0.0, np.deg2rad(1.5))
            ])

            measurement[1] = wrap(measurement[1])

            ekf.update_camera(
                measurement,
                marker_position
            )

    camera_visible_history.append(camera_visible)

    covariance_history.append(
        ekf.P.copy()
    )
        
    # Save one state per simulation step
    true_history.append(true_state.copy())
    estimate_history.append(ekf.state.copy())

true_history = np.array(true_history)
estimate_history = np.array(estimate_history)
camera_visible_history = np.array(camera_visible_history)
covariance_history = np.array(covariance_history)


# ========================== RESULTS ========================== #

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
    np.mean(error[:, 2]**2)
)


print(f"Position RMSE: {position_rmse:.4f} m")
print(
    f"Heading RMSE: "
    f"{np.rad2deg(heading_rmse):.2f} deg"
)

print("Final true state:")
print(true_history[-1])

print("Final EKF estimate:")
print(estimate_history[-1])


final_position_error = np.linalg.norm(
    true_history[-1, :2] - estimate_history[-1, :2]
)

final_heading_error = wrap(
    true_history[-1, 2] - estimate_history[-1, 2]
)

print(
    f"Final position error: "
    f"{final_position_error:.4f} m"
)

print(
    f"Final heading error: "
    f"{np.rad2deg(final_heading_error):.2f} deg"
)


# ========================== UNCERTAINTY RESULTS ========================== #

sigma_x = np.sqrt(covariance_history[:, 0, 0])
sigma_y = np.sqrt(covariance_history[:, 1, 1])
sigma_theta = np.sqrt(covariance_history[:, 2, 2])

print("\nUncertainty before dropout:")
print(
    f"x = {sigma_x[119]:.4f} m, "
    f"y = {sigma_y[119]:.4f} m, "
    f"theta = {np.rad2deg(sigma_theta[119]):.2f} deg"
)

print("\nUncertainty at end of dropout:")
print(
    f"x = {sigma_x[219]:.4f} m, "
    f"y = {sigma_y[219]:.4f} m, "
    f"theta = {np.rad2deg(sigma_theta[219]):.2f} deg"
)

print("\nUncertainty after camera recovery:")
print(
    f"x = {sigma_x[250]:.4f} m, "
    f"y = {sigma_y[250]:.4f} m, "
    f"theta = {np.rad2deg(sigma_theta[250]):.2f} deg"
)


# ========================== CONSISTENCY METRICS ========================== #

two_sigma_x = 2 * sigma_x
two_sigma_y = 2 * sigma_y
two_sigma_theta = 2 * sigma_theta

inside_x = np.abs(error[:, 0]) <= two_sigma_x
inside_y = np.abs(error[:, 1]) <= two_sigma_y
inside_theta = np.abs(error[:, 2]) <= two_sigma_theta

print("\nFraction inside 2 sigma:")
print(f"x = {np.mean(inside_x):.3f}")
print(f"y = {np.mean(inside_y):.3f}")
print(f"theta = {np.mean(inside_theta):.3f}")


nees_history = []

for i in range(STEPS):

    e = error[i]

    nees = e.T @ np.linalg.solve(
        covariance_history[i],
        e
    )

    nees_history.append(nees)

nees_history = np.array(nees_history)

print("\nNEES results:")
print(f"Mean NEES = {np.mean(nees_history):.3f}")
print(f"Max NEES = {np.max(nees_history):.3f}")