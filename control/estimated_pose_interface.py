"""Fail-closed estimated-state gate around the unchanged Part D supervisors.

No estimator, planner, sensor or actuator implementation is provided. All timing,
quality and clearance assumptions are explicit in ESTIMATED_STATE_INTERFACE.md.
"""
from dataclasses import dataclass
import math
import time
import weakref

import numpy as np

from control.command_gateway import GatewayStatus, WHEEL_LIMIT, WheelCommandLease
from control.lqr_controller import wrap
from control.terminal_controller import TerminalParameters
from control.trajectory import Reference


def _immutable_array(value):
    array = np.asarray(value, dtype=float)
    # bytes-backed arrays cannot have writeability re-enabled, unlike a read-only
    # view of an EKF array. The producer must copy state/P atomically before this.
    return np.frombuffer(array.tobytes(), dtype=float).reshape(array.shape)


@dataclass(frozen=True)
class PoseEstimate:
    pose: np.ndarray
    covariance: np.ndarray
    state_time: float
    sample_monotonic: float
    sequence: int
    epoch: str
    frame_id: str
    body_frame_id: str
    localisation_valid: bool
    validity_reason: str = ''

    def __post_init__(self):
        object.__setattr__(self, 'pose', _immutable_array(self.pose))
        object.__setattr__(self, 'covariance', _immutable_array(self.covariance))


@dataclass(frozen=True)
class InterfacePolicy:
    frame_id: str
    body_frame_id: str
    max_age_s: float
    command_ttl_s: float
    clock_grid_tolerance_s: float
    quality_check: object  # required callable(previous_snapshot_or_None, snapshot)->bool
    sample_time: float = .05  # established controller model interval, not a new tuning
    covariance_scales: tuple = (1., 1., 1.)  # 1 m, 1 m, 1 rad: numerical scaling only
    covariance_tolerance: float = 64*np.finfo(float).eps

    def __post_init__(self):
        times = (self.max_age_s, self.command_ttl_s, self.clock_grid_tolerance_s,
                 self.sample_time, self.covariance_tolerance)
        if not all(math.isfinite(v) and v > 0 for v in times):
            raise ValueError('Positive finite timing/numerical policies required')
        if self.clock_grid_tolerance_s >= self.sample_time/2:
            raise ValueError('Grid tolerance must distinguish adjacent control slots')
        scales = tuple(self.covariance_scales)
        if len(scales) != 3 or not all(math.isfinite(v) and v > 0 for v in scales):
            raise ValueError('Three positive covariance coordinate scales required')
        object.__setattr__(self, 'covariance_scales', scales)
        if (not isinstance(self.frame_id, str) or not self.frame_id
                or not isinstance(self.body_frame_id, str) or not self.body_frame_id
                or not callable(self.quality_check)):
            raise ValueError('Named frames and an explicit quality policy required')


@dataclass(frozen=True)
class ClearanceGrant:
    grant_id: str
    task_id: str
    frame_id: str
    estimator_epoch: str
    valid_from: float  # receiver monotonic clock domain
    expires_at: float
    approved: bool
    covers_terminal_manoeuvre: bool
    revoked: bool = False


@dataclass(frozen=True)
class InterfaceResult:
    accepted: bool
    armed: bool
    reason: str
    controller_mode: str | None = None
    receipt: GatewayStatus | None = None
    estimated_stop_qualified: bool = False

    @property
    def zero_dispatch_confirmed(self):
        return self.receipt is not None and self.receipt.zero_written is True


def validate_estimate(snapshot, policy, *, now, epoch, expected_state_time, previous=None):
    """Raise on invalid input. No covariance repair or physical jump threshold.

    quality_check must supply justified uncertainty/jump/relocalisation decisions.
    A syntactically valid covariance alone does not establish reliable localisation.
    """
    if not isinstance(snapshot, PoseEstimate):
        raise ValueError('malformed_estimate')
    if snapshot.pose.shape != (3,) or snapshot.covariance.shape != (3, 3):
        raise ValueError('malformed_pose_or_covariance')
    if not np.isfinite(snapshot.pose).all() or not np.isfinite(snapshot.covariance).all():
        raise ValueError('nonfinite_estimate')
    if snapshot.frame_id != policy.frame_id or snapshot.body_frame_id != policy.body_frame_id:
        raise ValueError('wrong_frame')
    if snapshot.epoch != epoch:
        raise ValueError('wrong_estimator_epoch')
    if snapshot.localisation_valid is not True:
        raise ValueError('localisation_invalid')
    if type(snapshot.sequence) is not int or snapshot.sequence < 0:
        raise ValueError('invalid_sequence')
    if not all(math.isfinite(v) for v in (now, snapshot.state_time, snapshot.sample_monotonic)):
        raise ValueError('nonfinite_timestamp')
    if snapshot.sample_monotonic > now:
        raise ValueError('future_estimate')
    if now >= snapshot.sample_monotonic+policy.max_age_s:
        raise ValueError('stale_estimate')
    if previous is not None and (snapshot.sequence <= previous.sequence
            or snapshot.state_time <= previous.state_time
            or snapshot.sample_monotonic <= previous.sample_monotonic):
        raise ValueError('duplicate_or_out_of_order')
    if abs(snapshot.state_time-expected_state_time) > policy.clock_grid_tolerance_s:
        raise ValueError('missed_or_wrong_control_slot')
    scale = np.asarray(policy.covariance_scales)
    normalised = snapshot.covariance/np.outer(scale, scale)
    if not np.isfinite(normalised).all():
        raise ValueError('invalid_covariance_scaling')
    norm = np.linalg.norm(normalised, ord=2)
    if not math.isfinite(norm):
        raise ValueError('invalid_covariance_norm')
    tolerance = policy.covariance_tolerance*max(1., norm)
    if np.max(abs(normalised-normalised.T)) > tolerance:
        raise ValueError('asymmetric_covariance')
    eigenvalues = np.linalg.eigvalsh((normalised+normalised.T)/2)
    if not np.isfinite(eigenvalues).all() or eigenvalues.min() < -tolerance:
        raise ValueError('indefinite_covariance')
    quality = policy.quality_check(previous, snapshot)
    if not isinstance(quality, (bool, np.bool_)) or not quality:
        raise ValueError('estimate_quality_rejected')


class EstimatedPoseInterface:
    """One serial controller caller; gateway is the only actuator writer.

    Factory must return a fresh existing trajectory/terminal supervisor on each
    arm. reference_provider(reference_time) is called inside the exception boundary.
    An estimate provider may also be passed to protect snapshot-construction errors.
    """
    def __init__(self, gateway, supervisor_factory, policy, *, goal, task_id,
                 terminal_start_s, parameters=TerminalParameters(), clock=time.monotonic):
        goal = _immutable_array(goal)
        if goal.shape != (3,) or not np.isfinite(goal).all():
            raise ValueError('Finite goal pose required')
        if (not callable(supervisor_factory) or not callable(clock) or not task_id
                or not math.isfinite(terminal_start_s) or terminal_start_s < 0
                or parameters.sample_time != policy.sample_time):
            raise ValueError('Invalid task, factory or sampling contract')
        self.gateway, self.factory, self.policy = gateway, supervisor_factory, policy
        self.goal, self.task_id = goal, task_id
        self.terminal_start_s, self.parameters, self.clock = terminal_start_s, parameters, clock
        self.armed = False
        self.supervisor = None
        self._used_supervisors = weakref.WeakSet()
        self.previous = None
        self.accepted_samples = 0
        self.generation = None
        self.epoch = None
        self.state_origin = None
        self.last_now = -math.inf
        self.completed = False

    def _now(self):
        now = float(self.clock())
        if not math.isfinite(now) or now < self.last_now:
            raise ValueError('invalid_interface_clock')
        self.last_now = now
        return now

    def disarm(self, reason='explicit_disarm'):
        self.armed = False
        self.supervisor = None  # discard accumulated dwell and any partial mutation
        self.previous = None
        self.accepted_samples = 0
        self.completed = False
        self.generation = None
        try:
            receipt = self.gateway.stop(str(reason))
            if not isinstance(receipt, GatewayStatus) or receipt.zero_written is not True:
                return InterfaceResult(False, False, str(reason)+'; stop_unconfirmed', receipt=receipt if isinstance(receipt, GatewayStatus) else None)
            return InterfaceResult(False, False, str(reason), receipt=receipt)
        except Exception as exc:
            return InterfaceResult(False, False, str(reason)+f'; stop_unconfirmed:{type(exc).__name__}')

    def arm(self, *, epoch, state_time_origin):
        stopped = self.disarm('explicit_rearm')
        if not stopped.zero_dispatch_confirmed:
            return stopped
        try:
            self._now()
            if not isinstance(epoch, str) or not epoch or not math.isfinite(state_time_origin):
                raise ValueError('invalid_arm_epoch_or_time')
            supervisor = self.factory()
            if supervisor in self._used_supervisors or not callable(supervisor.command):
                raise ValueError('supervisor_must_be_fresh')
            if (supervisor.end_time != self.terminal_start_s
                    or supervisor.tracker.dt != self.policy.sample_time
                    or supervisor.terminal.p != self.parameters
                    or not np.array_equal(supervisor.terminal.goal, self.goal)):
                raise ValueError('supervisor_contract_mismatch')
            self._used_supervisors.add(supervisor)
            receipt = self.gateway.arm()
            if not isinstance(receipt, GatewayStatus) or not receipt.accepted or not receipt.armed:
                raise ValueError('gateway_arm_failed')
            self.supervisor = supervisor
            self.generation = receipt.generation
            self.epoch, self.state_origin = epoch, float(state_time_origin)
            self.armed = True
            return InterfaceResult(True, True, 'armed', receipt=receipt)
        except Exception as exc:
            return self.disarm(f'arm_failed:{type(exc).__name__}:{exc}')

    def _clearance(self, grant, now):
        if (not isinstance(grant, ClearanceGrant) or not isinstance(grant.grant_id, str)
                or not grant.grant_id or grant.task_id != self.task_id
                or grant.frame_id != self.policy.frame_id or grant.estimator_epoch != self.epoch
                or grant.approved is not True or grant.revoked is not False
                or grant.covers_terminal_manoeuvre is not True
                or not math.isfinite(grant.valid_from) or not math.isfinite(grant.expires_at)
                or grant.valid_from > now or grant.expires_at <= now):
            raise ValueError('terminal_clearance_not_authorised')

    def _at_goal(self, snapshot):
        return (np.linalg.norm(snapshot.pose[:2]-self.goal[:2]) <= self.parameters.position_exit
                and abs(wrap(snapshot.pose[2]-self.goal[2])) <= self.parameters.heading_exit)

    def step(self, estimate, reference_provider, *, clearance=None):
        if not self.armed:
            return self.disarm('interface_not_armed')
        try:
            now = self._now()
            status = self.gateway.poll()  # catch independently expired/disarmed session
            if not status.armed or status.generation != self.generation:
                raise ValueError('gateway_session_lost')
            snapshot = estimate() if callable(estimate) else estimate
            now = self._now()
            expected = self.state_origin+self.accepted_samples*self.policy.sample_time
            validate_estimate(snapshot, self.policy, now=now, epoch=self.epoch,
                              expected_state_time=expected, previous=self.previous)
            relative_time = self.accepted_samples*self.policy.sample_time
            terminal = relative_time >= self.terminal_start_s
            if terminal:
                self._clearance(clearance, now)  # before invoking staging or terminal law
            if self.completed and not self._at_goal(snapshot):
                raise ValueError('disturbance_after_estimated_stop')
            reference = reference_provider(relative_time)
            ref_pose, ref_input = np.asarray(reference.pose, float), np.asarray(reference.input, float)
            if (ref_pose.shape != (3,) or ref_input.shape != (2,)
                    or not np.isfinite(ref_pose).all() or not np.isfinite(ref_input).all()):
                raise ValueError('malformed_reference')
            # Give the controller copies, never mutable access to producer state.
            output = self.supervisor.command(snapshot.pose.copy(), Reference(ref_pose.copy(), ref_input.copy()), relative_time)
            if not isinstance(output, tuple) or len(output) != 4:
                raise ValueError('malformed_controller_output')
            body, wheels, mode, saturated = output
            body, wheels = np.asarray(body, float), np.asarray(wheels, float)
            if (body.shape != (2,) or wheels.shape != (2,) or not np.isfinite(body).all()
                    or not np.isfinite(wheels).all() or max(abs(wheels)) > WHEEL_LIMIT
                    or mode not in (
                            'tracking_lqr',
                            'tracking_stationary_stop',
                            'tracking_avoidance',
                            'bearing',
                            'approach',
                            'heading',
                            'coast',
                            'stage_bearing',
                            'stage_approach',
                            'stop',
                            'fault',
                        )
                    or not isinstance(saturated, (bool, np.bool_))):
                raise ValueError('invalid_controller_output')
            represented = np.array([.033*sum(wheels)/2, .033*(wheels[1]-wheels[0])/.160])
            if not np.allclose(body, represented, rtol=1e-12, atol=1e-12):
                raise ValueError('inconsistent_body_and_wheel_output')
            if 'fault' in mode:
                raise ValueError('controller_fault')
            if terminal == mode.startswith('tracking_'):
                raise ValueError('controller_phase_mismatch')
            if mode in ('stop', 'coast', 'tracking_stationary_stop') and np.any(wheels != 0):
                raise ValueError('nonzero_stop_or_coast')
            if mode == 'stop' and (not terminal or not self._at_goal(snapshot) or np.any(wheels != 0)):
                raise ValueError('invalid_estimated_stop')
            if self.completed and (mode != 'stop' or np.any(wheels != 0)):
                raise ValueError('latched_stop_changed')
            # Recheck after potentially slow providers/controller. The independent
            # owner may already have expired; submit must never re-arm it.
            now = self._now()
            if now >= snapshot.sample_monotonic+self.policy.max_age_s:
                raise ValueError('estimate_expired_during_controller')
            if terminal:
                self._clearance(clearance, now)
            deadline = min(now+self.policy.command_ttl_s,
                           snapshot.sample_monotonic+self.policy.max_age_s,
                           clearance.expires_at if terminal else math.inf)
            lease = WheelCommandLease(self.generation, self.accepted_samples,
                                      tuple(wheels), now, deadline)
            receipt = self.gateway.submit(lease)
            if (not isinstance(receipt, GatewayStatus) or not receipt.accepted or not receipt.armed
                    or receipt.generation != self.generation
                    or receipt.applied_wheels != tuple(float(v) for v in wheels)
                    or (np.all(wheels == 0) and receipt.zero_written is not True)):
                raise ValueError('command_not_accepted')
            self.previous = snapshot
            self.accepted_samples += 1
            self.completed = mode == 'stop'
            return InterfaceResult(True, True, 'command_dispatched', mode, receipt, self.completed)
        except Exception as exc:
            return self.disarm(f'{type(exc).__name__}:{exc}')
