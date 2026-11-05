"""Measured-wheel, prediction-only adapter; NO actuator writes or camera updates.

Owner integration: construct WheelReader(model) in the single MuJoCo owner and
call read(data, ...) after completed steps. Send only WheelSample copies across
IPC. The estimator owns WheelPredictionAdapter; no model/data crosses that seam.
The reader enforces process/thread affinity (not a security boundary).

Seed with a measured sample at the explicitly supplied prior's simulation time.
Subsequent samples must advance by prediction_dt; no queued-slot replay, missing
interval extrapolation, or automatic reset. Simulation time drives propagation;
acquisition time is the original owner's receiver-compatible monotonic timestamp,
never receipt time. A pause cannot manufacture another fresh prediction. Resets
require a new epoch, explicit prior, and a new seed. This module supplies no 20 Hz
scheduler: an eventual publisher must select actual .05 s completed state slots.

For each interval use mean endpoint wheel rates (trapezoidal quadrature), convert
v=r*(right+left)/2, yaw_rate=r*(right-left)/track, and call the UNCHANGED EKF
predict(control, elapsed_sim_time). The constant-twist interval approximation is
not exact for time-varying curvature. Rates are rad/s; geometry is m; yaw is CCW.
Use explicitly declared physical/nominal rolling geometry, NOT command-response
fit coefficients. Joint qvel is a simulated encoder source, without synthetic
noise, hardware calibration or a guarantee of no slip.

Q is a REQUIRED 3x3 discrete pose process covariance PER PREDICTION CALL at the
configured interval, not a noise density automatically multiplied by dt. Changing
frequency requires a separately justified Q. Covariance validation is numerical,
not evidence of calibrated uncertainty, observability or convergence. The EKF's
required camera R is an unused identity placeholder; corrections are not exposed.

Snapshot publication and EKF mutation share a lock. Immutable PoseEstimate
subclasses include provenance and ALWAYS localisation_valid=False. No quality
policy, controller enabling, gateway access, ground-truth feedback or evidence
file generation is provided. Inputs rejected before prediction leave the previous
snapshot unchanged; prediction failure latches until explicit new-epoch reset.
"""
from dataclasses import dataclass, field
import math
import os
import threading

import mujoco
import numpy as np

from control.estimated_pose_interface import PoseEstimate
from estimation.ekf import TurtleBotEKF, wrap


def _finite(value, name):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError(name)
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(name)
    return value


def _name(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('Nonempty frame/epoch required')
    return value


def _covariance(value):
    a = np.array(value, dtype=float, copy=True)
    if a.shape != (3, 3) or not np.isfinite(a).all():
        raise ValueError('Finite 3x3 covariance required')
    tolerance = 64*np.finfo(float).eps*max(1., float(np.linalg.norm(a, 2)))
    if np.max(abs(a-a.T)) > tolerance or np.linalg.eigvalsh((a+a.T)/2).min() < -tolerance:
        raise ValueError('Symmetric positive semidefinite covariance required')
    return a  # Do not repair or tune the supplied covariance.


@dataclass(frozen=True)
class WheelSample:
    left_rad_s: float
    right_rad_s: float
    state_time: float
    sample_monotonic: float
    sequence: int
    epoch: str

    def __post_init__(self):
        for name in ('left_rad_s', 'right_rad_s', 'state_time', 'sample_monotonic'):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        if self.state_time < 0 or self.sample_monotonic < 0:
            raise ValueError('Nonnegative clocks required')
        if type(self.sequence) is not int or self.sequence < 0:
            raise ValueError('Nonnegative integer sequence required')
        _name(self.epoch)


class WheelReader:
    """Owner-only named hinge/transmission resolution; never reads ctrl or qpos."""
    def __init__(self, model):
        self._owner = (os.getpid(), threading.get_ident())
        joints = [model.joint(n).id for n in ('wheel_left', 'wheel_right')]
        actuators = [model.actuator(n).id for n in ('wheel_left', 'wheel_right')]
        self.dofs = tuple(int(model.jnt_dofadr[j]) for j in joints)
        if (len(set(joints)) != 2 or len(set(self.dofs)) != 2
                or any(model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE for j in joints)
                or any(not 0 <= d < model.nv for d in self.dofs)
                or any(model.actuator_trntype[a] != mujoco.mjtTrn.mjTRN_JOINT
                       or model.actuator_trnid[a, 0] != j for a, j in zip(actuators, joints))):
            raise ValueError('Invalid named wheel hinge/DOF/transmission mapping')

    def read(self, data, *, sample_monotonic, sequence, epoch):
        if (os.getpid(), threading.get_ident()) != self._owner:
            raise RuntimeError('MuJoCo wheel access outside owner process/thread')
        left, right = (float(data.qvel[d]) for d in self.dofs)
        return WheelSample(left, right, float(data.time), sample_monotonic, sequence, epoch)


def body_velocity(sample, *, radius_m, track_m):
    radius, track = _finite(radius_m, 'radius'), _finite(track_m, 'track')
    if radius <= 0 or track <= 0 or not isinstance(sample, WheelSample):
        raise ValueError('Positive geometry and WheelSample required')
    result = np.array([radius*(sample.right_rad_s+sample.left_rad_s)/2,
                       radius*(sample.right_rad_s-sample.left_rad_s)/track])
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite body velocity')
    return result


@dataclass(frozen=True)
class PredictionSnapshot(PoseEstimate):
    provenance: str = field(default='prediction_only_simulated_joint_encoders', init=False)

    def __reduce__(self):
        # Re-run PoseEstimate's bytes-backed copy construction after IPC unpickling.
        return (type(self), (self.pose, self.covariance, self.state_time,
                            self.sample_monotonic, self.sequence, self.epoch,
                            self.frame_id, self.body_frame_id, self.localisation_valid,
                            self.validity_reason))


class WheelPredictionAdapter:
    def __init__(self, *, radius_m, track_m, prediction_dt, initial_pose,
                 initial_covariance, process_noise_per_call, seed, frame_id,
                 body_frame_id, time_tolerance_s=1e-9):
        self._lock = threading.RLock()
        self._epochs = set()
        self.radius = _finite(radius_m, 'radius')
        self.track = _finite(track_m, 'track')
        self.dt = _finite(prediction_dt, 'prediction_dt')
        self.tolerance = _finite(time_tolerance_s, 'time_tolerance_s')
        if min(self.radius, self.track, self.dt, self.tolerance) <= 0 or self.tolerance >= self.dt/2:
            raise ValueError('Positive geometry/interval and distinguishing clock tolerance required')
        self.frame_id, self.body_frame_id = _name(frame_id), _name(body_frame_id)
        self._q = _covariance(process_noise_per_call)
        self.reset(initial_pose=initial_pose, initial_covariance=initial_covariance, seed=seed)

    def reset(self, *, initial_pose, initial_covariance, seed):
        with self._lock:
            if not isinstance(seed, WheelSample) or seed.epoch in self._epochs:
                raise ValueError('Reset requires a seed in a never-used new epoch')
            pose = np.array(initial_pose, dtype=float, copy=True)
            if pose.shape != (3,) or not np.isfinite(pose).all():
                raise ValueError('Explicit finite initial pose required')
            pose[2] = wrap(pose[2])
            covariance = _covariance(initial_covariance)
            self._ekf = TurtleBotEKF(pose, covariance, self._q.copy(), np.eye(2))
            self._previous = seed
            self._epochs.add(seed.epoch)
            self._fault = False
            self._snapshot = self._publish(seed)
            return self._snapshot

    def _publish(self, sample):
        return PredictionSnapshot(self._ekf.state, self._ekf.P, sample.state_time,
                                  sample.sample_monotonic, sample.sequence, sample.epoch,
                                  self.frame_id, self.body_frame_id, False,
                                  'Prediction only; no justified localisation-quality policy')

    def latest(self):
        with self._lock:
            if self._fault:
                raise RuntimeError('Prediction failed; explicit new-epoch reset required')
            return self._snapshot  # immutable; repeated retrieval is NOT a new sample

    def predict(self, sample):
        with self._lock:
            if self._fault:
                raise RuntimeError('Prediction failed; explicit new-epoch reset required')
            if not isinstance(sample, WheelSample):
                raise ValueError('WheelSample required')
            previous = self._previous
            elapsed = sample.state_time-previous.state_time
            if (sample.epoch != previous.epoch or sample.sequence != previous.sequence+1
                    or sample.sample_monotonic <= previous.sample_monotonic
                    or elapsed <= 0 or abs(elapsed-self.dt) > self.tolerance):
                raise ValueError('Wrong epoch, duplicate/out-of-order or missing interval')
            control = (body_velocity(previous, radius_m=self.radius, track_m=self.track)/2
                       + body_velocity(sample, radius_m=self.radius, track_m=self.track)/2)
            try:
                self._ekf.predict(control, elapsed)
                if not np.isfinite(self._ekf.state).all():
                    raise ValueError('Nonfinite EKF state')
                _covariance(self._ekf.P)
                snapshot = self._publish(sample)
            except Exception:
                self._fault = True
                raise
            self._previous, self._snapshot = sample, snapshot
            return snapshot
