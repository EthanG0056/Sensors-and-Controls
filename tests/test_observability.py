"""Synthetic structural-observability checks, separate from benchmark evidence.

Uses the actual EKF motion/observation functions. No EKF, sensor or controller
implementation is changed, and no benchmark files are created or regenerated.
"""
import unittest

import numpy as np
from numpy.testing import assert_allclose

from estimation.ekf import (
    camera_jacobian, camera_model, motion_jacobian, motion_model, wrap,
)

# Absolute and relative singular-value cutoffs, not NumPy's implicit rank default.
RANK_ATOL = 1e-10
RANK_RTOL = 1e-9
FD_STEP = 1e-6
FD_ATOL = 2e-8
FD_RTOL = 2e-7
ALGEBRA_ATOL = 2e-11
DT = .05
MARKER = np.array([2., 1.2])
INITIALS = (
    np.array([.1, -.2, .3]),
    np.array([1.1, .4, -2.4]),
    np.array([-.7, -.6, 2.9]),
)
TRAJECTORIES = {
    'stationary': lambda k: np.array([0., 0.]),
    'rotation': lambda k: np.array([0., .55]),
    'forward': lambda k: np.array([.12, 0.]),
    'reverse': lambda k: np.array([-.09, 0.]),
    'left_arc': lambda k: np.array([.11, .35]),
    'right_arc': lambda k: np.array([.10, -.4]),
    'varying': lambda k: np.array([.12*np.cos(.07*k), .45*np.sin(.09*k)]),
}


def numerical_rank(matrix):
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    threshold = max(RANK_ATOL, RANK_RTOL*singular_values[0])
    return int(np.sum(singular_values > threshold)), singular_values, threshold


def gauge(state, marker):
    dx, dy = np.asarray(marker)-np.asarray(state)[:2]
    return np.array([dy, -dx, 1.])


def rotate_about_marker(state, marker, angle):
    rotation = np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])
    return np.r_[marker + rotation @ (state[:2]-marker), wrap(state[2]+angle)]


def circular_difference(a, b, angle_index):
    difference = np.asarray(a)-np.asarray(b)
    difference[angle_index] = wrap(difference[angle_index])
    return difference


def finite_jacobian(function, state, angle_index):
    columns = []
    for perturbation in FD_STEP*np.eye(3):
        columns.append(circular_difference(function(state+perturbation),
                                            function(state-perturbation),
                                            angle_index)/(2*FD_STEP))
    return np.column_stack(columns)


def continuous_a(state, control):
    theta, v = state[2], control[0]
    return np.array([[0., 0., -v*np.sin(theta)],
                     [0., 0., v*np.cos(theta)], [0., 0., 0.]])


class ObservabilityTests(unittest.TestCase):
    def assert_rank(self, matrix, expected):
        rank, values, threshold = numerical_rank(matrix)
        self.assertEqual(rank, expected,
                         f'singular values={values}, cutoff={threshold}')
        return values, threshold

    def test_observation_definition_matches_ekf(self):
        for state in INITIALS:
            delta = MARKER-state[:2]
            expected = np.array([np.linalg.norm(delta),
                                 wrap(np.arctan2(delta[1], delta[0])-state[2])])
            assert_allclose(camera_model(state, MARKER), expected, atol=ALGEBRA_ATOL)

    def test_observation_jacobians_against_actual_model(self):
        # Include an observation exactly at the bearing branch cut; difference
        # must be circular, not the subtraction of two wrapped angle numbers.
        cases = [(s, MARKER) for s in INITIALS]
        cases += [(np.array([0., 0., 0.]), np.array([-2., 0.])),
                  (np.array([.2, -.1, np.pi-.01]), np.array([.5, -.1]))]
        for state, marker in cases:
            dx, dy = marker-state[:2]; rho = np.hypot(dx, dy)
            analytical = np.array([[-dx/rho, -dy/rho, 0.],
                                   [dy/rho**2, -dx/rho**2, -1.]])
            assert_allclose(camera_jacobian(state, marker), analytical, atol=ALGEBRA_ATOL)
            numerical = finite_jacobian(lambda x: camera_model(x, marker), state, 1)
            assert_allclose(analytical, numerical, atol=FD_ATOL, rtol=FD_RTOL)

    def test_motion_jacobians_against_actual_model(self):
        controls = [(0., 0.), (.12, 0.), (-.09, 0.), (0., .55),
                    (.11, .35), (.1, -.4), (.12, 5e-7), (.12, -5e-7)]
        for state in (*INITIALS, np.array([0., 0., np.pi])):
            for control in controls:
                for dt in (.01, DT, .2):
                    with self.subTest(state=state, control=control, dt=dt):
                        after = motion_model(state, control, dt)
                        dx, dy = after[:2]-state[:2]
                        analytical = np.array([[1., 0., -dy],
                                               [0., 1., dx], [0., 0., 1.]])
                        actual = motion_jacobian(state, control, dt)
                        numerical = finite_jacobian(lambda x: motion_model(x, control, dt), state, 2)
                        assert_allclose(actual, analytical, atol=ALGEBRA_ATOL)
                        assert_allclose(actual, numerical, atol=FD_ATOL, rtol=FD_RTOL)

    def test_single_landmark_rank_and_gauge(self):
        for state in INITIALS:
            h = camera_jacobian(state, MARKER)
            self.assert_rank(h, 2)
            assert_allclose(h @ gauge(state, MARKER), 0., atol=ALGEBRA_ATOL)

    def test_frozen_rank_does_not_establish_nonlinear_observability(self):
        for state in INITIALS:
            h = camera_jacobian(state, MARKER)
            for control, expected in [((0., 0.), 2), ((0., .55), 2),
                                      ((.12, 0.), 3), ((-.09, 0.), 3),
                                      ((.11, .35), 3)]:
                a = continuous_a(state, control)
                self.assert_rank(np.vstack([h, h @ a, h @ a @ a]), expected)

    def test_continuous_output_derivative_retains_nullspace(self):
        for state in INITIALS:
            for control in ((.12, 0.), (.11, .35), (0., .55), (-.09, -.2)):
                v, omega = control; theta = state[2]
                f = np.array([v*np.cos(theta), v*np.sin(theta), omega])
                h = camera_jacobian(state, MARKER); a = continuous_a(state, control)
                hdot = (camera_jacobian(state+FD_STEP*f, MARKER)
                        - camera_jacobian(state-FD_STEP*f, MARKER))/(2*FD_STEP)
                rho, beta = camera_model(state, MARKER)
                output_flow_jacobian = np.array([[0., v*np.sin(beta)],
                    [-v*np.sin(beta)/rho**2, v*np.cos(beta)/rho]])
                assert_allclose(hdot+h@a, output_flow_jacobian@h,
                                atol=FD_ATOL, rtol=FD_RTOL)
                assert_allclose((hdot+h@a)@gauge(state, MARKER), 0., atol=FD_ATOL)
                gdot = (gauge(state+FD_STEP*f, MARKER)
                        - gauge(state-FD_STEP*f, MARKER))/(2*FD_STEP)
                assert_allclose(gdot, a@gauge(state, MARKER), atol=FD_ATOL)

    def test_trajectory_observability_uses_actual_discrete_jacobians(self):
        for name, control_at in TRAJECTORIES.items():
            for initial in INITIALS:
                with self.subTest(trajectory=name, initial=initial):
                    state = initial.copy(); phi = np.eye(3); rows = []
                    g0 = gauge(initial, MARKER)
                    for k in range(81):
                        self.assertGreater(np.linalg.norm(MARKER-state[:2]), .2)
                        h = camera_jacobian(state, MARKER)
                        rows.append(h @ phi)
                        assert_allclose(phi @ g0, gauge(state, MARKER), atol=ALGEBRA_ATOL)
                        if k < 80:
                            control = control_at(k)
                            phi = motion_jacobian(state, control, DT) @ phi
                            state = motion_model(state, control, DT)
                    observation_matrix = np.vstack(rows)
                    values, threshold = self.assert_rank(observation_matrix, 2)
                    # Verify both the missing direction and a clear numerical gap.
                    assert_allclose(observation_matrix @ g0, 0., atol=ALGEBRA_ATOL)
                    self.assertGreater(values[1], 1000*threshold)
                    self.assertLess(values[2], threshold/100)

    def test_finite_nonlinear_symmetry_on_several_trajectories(self):
        for name, control_at in TRAJECTORIES.items():
            for initial in INITIALS:
                for angle in (-2.2, -.4, .7, 2.8):
                    with self.subTest(trajectory=name, initial=initial, rotation=angle):
                        state = initial.copy()
                        rotated = rotate_about_marker(state, MARKER, angle)
                        self.assertGreater(np.linalg.norm(rotated[:2]-state[:2]), .1)
                        for k in range(81):
                            residual = circular_difference(camera_model(rotated, MARKER),
                                                           camera_model(state, MARKER), 1)
                            assert_allclose(residual, 0., atol=ALGEBRA_ATOL)
                            expected = rotate_about_marker(state, MARKER, angle)
                            assert_allclose(circular_difference(rotated, expected, 2),
                                            0., atol=ALGEBRA_ATOL)
                            if k < 80:
                                control = control_at(k)
                                state = motion_model(state, control, DT)
                                rotated = motion_model(rotated, control, DT)

    def test_two_distinct_known_landmarks_and_duplicate_degeneracy(self):
        for state in INITIALS:
            first = camera_jacobian(state, MARKER)
            second = camera_jacobian(state, np.array([-1.3, 2.4]))
            self.assert_rank(np.vstack([first, second]), 3)
            self.assert_rank(np.vstack([first, first]), 2)
        # Collinearity alone is not a degeneracy for identified range+bearing
        # observations from two distinct points (unlike some range-only cases).
        state = np.zeros(3)
        self.assert_rank(np.vstack([camera_jacobian(state, [2., 0.]),
                                    camera_jacobian(state, [4., 0.])]), 3)

    def test_additional_absolute_measurements_are_proposals(self):
        # Synthetic augmented outputs, NOT measurements currently fused by EKF.
        for state in INITIALS:
            h = camera_jacobian(state, MARKER)
            self.assert_rank(np.vstack([h, [0., 0., 1.]]), 3)  # known world heading
            self.assert_rank(np.vstack([h, [0., 0., -1.]]), 3)  # calibrated oriented marker
            self.assert_rank(np.vstack([h, [[1., 0., 0.], [0., 1., 0.]]]), 3)

    def test_informative_prior_is_not_measurement_information(self):
        initial = INITIALS[0]; rotated = rotate_about_marker(initial, MARKER, .7)
        assert_allclose(circular_difference(camera_model(rotated, MARKER),
                                           camera_model(initial, MARKER), 1),
                        0., atol=ALGEBRA_ATOL)
        # An illustrative prior penalises an indistinguishable state. This
        # explains the source of information; covariance rank is not a proof.
        delta = circular_difference(rotated, initial, 2)
        prior_scales = np.array([.2, .2, .1])
        self.assertGreater(float(np.sum((delta/prior_scales)**2)), 1.)

    def test_zero_range_is_excluded_and_branch_difference_is_circular(self):
        state = np.r_[MARKER, .3]
        self.assertEqual(camera_model(state, MARKER)[0], 0.)
        # The incoming EKF explicitly rejects range < 1e-9 m instead of
        # returning nonfinite entries. This changes the guard, not observability.
        with self.assertRaises(ValueError):
            camera_jacobian(state, MARKER)
        # Use an origin landmark to avoid subtractive cancellation and points
        # safely either side of the guard, rather than exact threshold equality.
        marker = np.zeros(2)
        with self.assertRaises(ValueError):
            camera_jacobian(np.array([-0.9e-9, 0., .3]), marker)
        distance = 1.1e-9
        h = camera_jacobian(np.array([-distance, 0., .3]), marker)
        self.assertTrue(np.isfinite(h).all())
        assert_allclose(h, [[-1., 0., 0.], [0., -1/distance, -1.]],
                        rtol=1e-12, atol=1e-12)
        a = np.array([1., np.pi-1e-7]); b = np.array([1., -np.pi+1e-7])
        self.assertGreater(abs(a[1]-b[1]), 6.)
        assert_allclose(circular_difference(a, b, 1), [0., -2e-7], atol=ALGEBRA_ATOL)

    def test_explicit_rank_cutoff(self):
        self.assert_rank(np.diag([1., .5*RANK_RTOL, 2*RANK_RTOL]), 2)


if __name__ == '__main__':
    unittest.main()
