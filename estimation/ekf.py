# ====================================================================== #
#                         EXTENDED KALMAN FILTER                         #
# ====================================================================== #

import numpy as np

# ====================================================================== #
#                            HELPER FUNCTIONS                            #
# ====================================================================== #

# ========================== WRAPPER FUNCTION ========================== #

def wrap(angle):

    # Wrap angle to [-pi, pi)
    return (angle + np.pi) % (2 * np.pi) - np.pi


# ========================== WHEEL ODOMETRY ========================== #

def wheel_odometry(left_speed, right_speed,
                   wheel_radius=0.033, wheel_separation=0.160):

    # Convert measured wheel speeds into robot linear and angular velocity

    v = wheel_radius * (right_speed + left_speed) / 2

    omega = wheel_radius * (right_speed - left_speed) / wheel_separation

    return np.array([v, omega])


# ====================================================================== #
#                             MAIN FUNCTIONS                             #
# ====================================================================== #

# ========================== MOTION MODEL ========================== #

def motion_model(state, control, dt):

    # Predict robot pose from linear and angular velocity:

    # state   = [x, y, theta]
    # control = [v, omega]
    
    x, y, theta = state
    v, omega = control

    if abs(omega) < 1e-6:
        x_new = x + v * dt * np.cos(theta)
        y_new = y + v * dt * np.sin(theta)
        theta_new = theta

    else:
        theta_new = theta + omega * dt

        x_new = x + (v / omega) * (
            np.sin(theta_new) - np.sin(theta)
        )

        y_new = y + (v / omega) * (
            np.cos(theta) - np.cos(theta_new)
        )

    return np.array([
        x_new,
        y_new,
        wrap(theta_new)
    ])


def motion_jacobian(state, control, dt):

    # Jacobian of the motion model with respect to the state

    _, _, theta = state
    v, omega = control

    F = np.eye(3)

    if abs(omega) < 1e-6:
        F[0, 2] = -v * dt * np.sin(theta)
        F[1, 2] =  v * dt * np.cos(theta)

    else:
        theta_new = theta + omega * dt

        F[0, 2] = (v / omega) * (
            np.cos(theta_new) - np.cos(theta)
        )

        F[1, 2] = (v / omega) * (
            np.sin(theta_new) - np.sin(theta)
        )

    return F


# ========================== CAMERA MODEL ========================== #

def camera_model(state, marker_position):

    # Predicted range and bearing from robot to ArUco marker.

    x, y, theta = state
    marker_x, marker_y = marker_position

    dx = marker_x - x
    dy = marker_y - y

    marker_range = np.hypot(dx, dy)

    world_angle = np.arctan2(dy, dx)

    marker_bearing = wrap(world_angle - theta)

    return np.array([
        marker_range,
        marker_bearing
    ])


def camera_jacobian(state, marker_position):

    # Jacobian of camera range/bearing measurement.

    x, y, _ = state
    marker_x, marker_y = marker_position

    dx = marker_x - x
    dy = marker_y - y

    range_sq = dx**2 + dy**2
    marker_range = np.sqrt(range_sq)

    # Prevent divide by 0
    if marker_range < 1e-9:
        raise ValueError("Marker is too close to calculate a valid Jacobian")

    H = np.array([
        [-dx / marker_range, -dy / marker_range,  0.0],
        [ dy / range_sq,     -dx / range_sq,     -1.0]
    ])

    return H


# ========================== EKF ========================== #

class TurtleBotEKF:

    def __init__(self, initial_state, initial_covariance, Q, R):

        self.state = np.array(initial_state, dtype=float)
        self.P = np.array(initial_covariance, dtype=float)

        self.Q = np.array(Q, dtype=float)
        self.R = np.array(R, dtype=float)


    def predict(self, control, dt):

        F = motion_jacobian(self.state, control, dt)

        self.state = motion_model(self.state, control, dt)

        self.P = F @ self.P @ F.T + self.Q


    def update_camera(self, measurement, marker_position):

        z = np.asarray(measurement, dtype=float).reshape(2)

        expected = camera_model(
            self.state,
            marker_position
        )

        H = camera_jacobian(
            self.state,
            marker_position
        )

        innovation = z - expected
        innovation[1] = wrap(innovation[1])

        S = H @ self.P @ H.T + self.R

        K = np.linalg.solve(
            S,
            H @ self.P
        ).T

        self.state = self.state + K @ innovation
        self.state[2] = wrap(self.state[2])

        I = np.eye(3)

        self.P = (
            (I - K @ H) @ self.P @ (I - K @ H).T
            + K @ self.R @ K.T
        )

        # Keep covariance symmetric
        self.P = 0.5 * (self.P + self.P.T)