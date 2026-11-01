"""Pose-input discrete LQR. No simulator, sensor or estimator dependency.

Error: reference minus actual pose, expressed in actual robot body axes.
Inputs are increments about reference [v, yaw_rate]. See PART_D_CONTROL.md.
"""
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.linalg import expm, solve_discrete_are


def wrap(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def tracking_error(pose, reference_pose):
    x, y, theta = np.asarray(pose, dtype=float)
    dx, dy, dt = np.asarray(reference_pose, dtype=float) - [x, y, theta]
    return np.array([np.cos(theta)*dx + np.sin(theta)*dy,
                     -np.sin(theta)*dx + np.cos(theta)*dy, wrap(dt)])


def error_dynamics(error, increment, reference_input):
    ex, ey, et = error
    vr, wr = reference_input
    v, w = np.asarray(reference_input) + increment
    return np.array([w*ey + vr*np.cos(et)-v,
                     -w*ex + vr*np.sin(et), wr-w])


def linearise(v, omega):
    return np.array([[0., omega, 0.], [-omega, 0., v], [0., 0., 0.]]), np.array([[-1., 0.], [0., 0.], [0., -1.]])


def discretise(a, b, dt):
    if dt <= 0 or not np.isfinite(dt):
        raise ValueError('dt must be finite and positive')
    n, m = b.shape
    block = np.zeros((n+m, n+m)); block[:n, :n] = a; block[:n, n:] = b
    result = expm(dt*block)
    return result[:n, :n], result[:n, n:]


def controllability(a, b):
    matrix = np.concatenate([np.linalg.matrix_power(a, k) @ b for k in range(len(a))], axis=1)
    return int(np.linalg.matrix_rank(matrix)), np.linalg.svd(matrix, compute_uv=False)


@lru_cache(maxsize=8192)
def _gain(v, omega, dt, q, r):
    a, b = linearise(v, omega)
    ad, bd = discretise(a, b, dt)
    if controllability(ad, bd)[0] < 3:
        raise ValueError('Frozen reference linearisation is not controllable')
    qmat, rmat = np.diag(q), np.diag(r)
    p = solve_discrete_are(ad, bd, qmat, rmat)
    k = np.linalg.solve(rmat + bd.T @ p @ bd, bd.T @ p @ ad)
    if max(abs(np.linalg.eigvals(ad-bd@k))) >= 1:
        raise RuntimeError('DARE solution does not stabilise the frozen model')
    return k, p, ad, bd


def lqr_gain(v, omega, dt, q, r):
    return _gain(float(v), float(omega), float(dt), tuple(q), tuple(r))


def wheel_commands(v, omega, radius=.033, track=.160, limit=6.67):
    if min(radius, track, limit) <= 0:
        raise ValueError('Geometry and limit must be positive')
    wheels = np.array([(v-track*omega/2)/radius, (v+track*omega/2)/radius])
    scale = max(1., float(max(abs(wheels)))/limit)
    return wheels/scale, np.array([v, omega])/scale, scale > 1.


@dataclass(frozen=True)
class Weights:
    q: tuple = (156.25, 156.25, 16.)
    r: tuple = (69.44444444444444, 1.5625)

    def __post_init__(self):
        if len(self.q)!=3 or len(self.r)!=2 or not np.isfinite(self.q+self.r).all() or min(self.q+self.r)<=0:
            raise ValueError('Positive finite diagonal Q(3) and R(2) required')


CONFIGS = {'balanced': Weights(),
           'tracking': Weights(tuple(4*x for x in Weights().q), Weights().r),
           'effort': Weights(Weights().q, tuple(4*x for x in Weights().r))}


class LQRTracker:
    def __init__(self, dt=.05, weights=Weights()):
        if dt <= 0 or not np.isfinite(dt):
            raise ValueError('dt must be finite and positive')
        self.dt, self.weights = dt, weights

    def command(self, estimated_pose, reference):
        """Reference has pose and input attributes. EKF pose can be passed directly.

        Stationary references deliberately request zero: this is not a docking law.
        Returns unsaturated body input and explicit mode for benchmark logging.
        """
        error = tracking_error(estimated_pose, reference.pose)
        if not np.isfinite(error).all() or not np.isfinite(reference.input).all():
            raise ValueError('Pose and reference must be finite')
        v, omega = reference.input
        if abs(v)+abs(omega) < 1e-8:
            return np.zeros(2), 'stationary_stop'
        k, _, _, _ = lqr_gain(v, omega, self.dt, self.weights.q, self.weights.r)
        return np.asarray(reference.input)-k@error, 'lqr'
