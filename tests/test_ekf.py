import numpy as np

from estimation.ekf import (
    motion_model,
    camera_model
)


def test_straight_motion():

    state = np.array([0.0, 0.0, 0.0])
    control = np.array([1.0, 0.0])

    new_state = motion_model(
        state,
        control,
        1.0
    )

    print(new_state)


def test_camera_measurement():

    state = np.array([0.0, 0.0, 0.0])

    marker = np.array([
        2.0,
        2.0
    ])

    measurement = camera_model(
        state,
        marker
    )

    print(measurement)


test_straight_motion()
test_camera_measurement()