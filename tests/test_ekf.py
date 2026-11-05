import numpy as np

from estimation.ekf import (
    wrap,
    wheel_odometry,
    motion_model,
    motion_jacobian,
    camera_model,
    camera_jacobian
)


# ========================== BASIC MODEL TESTS ========================== #

def test_straight_motion():

    state = np.array([0.0, 0.0, 0.0])
    control = np.array([1.0, 0.0])

    new_state = motion_model(
        state,
        control,
        1.0
    )

    expected = np.array([1.0, 0.0, 0.0])

    assert np.allclose(new_state, expected)

    print("Straight motion test passed")


def test_camera_measurement():

    state = np.array([0.0, 0.0, 0.0])
    marker = np.array([2.0, 2.0])

    measurement = camera_model(
        state,
        marker
    )

    expected = np.array([
        np.sqrt(8),
        np.deg2rad(45)
    ])

    assert np.allclose(measurement, expected)

    print("Camera model test passed")


def test_wheel_odometry():

    # Both wheels moving at the same speed should give straight motion
    control = wheel_odometry(
        3.0,
        3.0
    )

    expected_v = 0.033 * 3.0

    assert np.isclose(control[0], expected_v)
    assert np.isclose(control[1], 0.0)

    print("Wheel odometry test passed")


# ========================== JACOBIAN TESTS ========================== #

def test_motion_jacobian():

    state = np.array([1.0, 0.5, 0.4])
    control = np.array([0.3, 0.2])
    dt = 0.05

    F = motion_jacobian(
        state,
        control,
        dt
    )

    eps = 1e-6
    F_num = np.zeros((3, 3))

    for i in range(3):

        change = np.zeros(3)
        change[i] = eps

        plus = motion_model(
            state + change,
            control,
            dt
        )

        minus = motion_model(
            state - change,
            control,
            dt
        )

        difference = plus - minus
        difference[2] = wrap(difference[2])

        F_num[:, i] = difference / (2 * eps)

    assert np.allclose(
        F,
        F_num,
        atol=1e-5
    )

    print("Motion Jacobian test passed")


def test_camera_jacobian():

    state = np.array([1.0, 0.5, 0.4])
    marker = np.array([3.0, 2.0])

    H = camera_jacobian(
        state,
        marker
    )

    eps = 1e-6
    H_num = np.zeros((2, 3))

    for i in range(3):

        change = np.zeros(3)
        change[i] = eps

        plus = camera_model(
            state + change,
            marker
        )

        minus = camera_model(
            state - change,
            marker
        )

        difference = plus - minus
        difference[1] = wrap(difference[1])

        H_num[:, i] = difference / (2 * eps)

    assert np.allclose(
        H,
        H_num,
        atol=1e-5
    )

    print("Camera Jacobian test passed")


# ========================== RUN TESTS ========================== #

test_straight_motion()
test_camera_measurement()
test_wheel_odometry()
test_motion_jacobian()
test_camera_jacobian()