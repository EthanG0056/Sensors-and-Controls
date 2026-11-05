"""Conservative diagnostic validity, never proof of calibrated localisation.

No default uncertainty/jump/dropout limits: autonomous motion is disabled unless
explicit criteria AND an evidence justification are supplied. Structural checks
are necessary but do not establish observability, consistency or reliable pose.
A successful associated correction is required before any prediction-only grace
interval. Missing detection is distinct from an algorithm/numerical failure.
"""
from dataclasses import dataclass
import math

import numpy as np
from control.lqr_controller import wrap


@dataclass(frozen=True)
class QualityCriteria:
    max_std: tuple  # metres, metres, radians, including all three coordinates
    max_position_jump_m: float
    max_heading_jump_rad: float
    max_prediction_only_s: float
    justification: str

    def __post_init__(self):
        object.__setattr__(self, 'max_std', tuple(self.max_std))
        values = (*self.max_std, self.max_position_jump_m, self.max_heading_jump_rad,
                  self.max_prediction_only_s)
        if (len(self.max_std) != 3 or not all(math.isfinite(v) and v > 0 for v in values)
                or not isinstance(self.justification, str) or not self.justification.strip()):
            raise ValueError('Explicit positive diagnostic criteria and evidence justification required')

    def accepts(self, previous, current):
        if np.any(np.sqrt(np.maximum(0., np.diag(current.covariance))) > self.max_std):
            return False
        return previous is None or (
            np.linalg.norm(current.pose[:2]-previous.pose[:2]) <= self.max_position_jump_m
            and abs(wrap(current.pose[2]-previous.pose[2])) <= self.max_heading_jump_rad)


@dataclass(frozen=True)
class Validity:
    valid: bool
    reason: str


@dataclass(frozen=True)
class LocalisationPolicy:
    epoch: str
    frame_id: str
    body_frame_id: str
    sample_time: float
    max_age_s: float
    grid_tolerance_s: float
    criteria: QualityCriteria | None = None

    def __post_init__(self):
        if (not all(isinstance(v, str) and v for v in (self.epoch,self.frame_id,self.body_frame_id))
                or not all(math.isfinite(v) and v > 0 for v in
                           (self.sample_time,self.max_age_s,self.grid_tolerance_s))
                or self.grid_tolerance_s >= self.sample_time/2
                or self.criteria is not None and not isinstance(self.criteria,QualityCriteria)):
            raise ValueError('Explicit frames, epoch and distinguishing timing policy required')

    def assess(self, snapshot, *, previous, expected_time, now, camera_status,
               last_correction_state_time, numerical_health=True):
        """Pure decision: no state mutation, timestamp repair or covariance repair."""
        try:
            if numerical_health is not True:
                return Validity(False, 'estimator_numerical_failure')
            if snapshot.frame_id != self.frame_id or snapshot.body_frame_id != self.body_frame_id:
                return Validity(False, 'wrong_frame')
            if snapshot.epoch != self.epoch:
                return Validity(False, 'wrong_epoch')
            if snapshot.pose.shape != (3,) or snapshot.covariance.shape != (3,3):
                return Validity(False, 'malformed_pose_or_covariance')
            if not np.isfinite(snapshot.pose).all() or not np.isfinite(snapshot.covariance).all():
                return Validity(False, 'nonfinite_estimate')
            if (type(snapshot.sequence) is not int or snapshot.sequence < 0
                    or not all(math.isfinite(v) for v in
                               (snapshot.state_time,snapshot.sample_monotonic,now,expected_time))):
                return Validity(False, 'malformed_timestamp_or_sequence')
            if snapshot.sample_monotonic > now or now >= snapshot.sample_monotonic+self.max_age_s:
                return Validity(False, 'stale_or_future_estimate')
            if abs(snapshot.state_time-expected_time) > self.grid_tolerance_s:
                return Validity(False, 'missed_control_slot')
            if previous is not None and (
                    snapshot.sequence != previous.sequence+1
                    or snapshot.state_time <= previous.state_time
                    or snapshot.sample_monotonic <= previous.sample_monotonic):
                return Validity(False, 'duplicate_or_out_of_order')
            p = snapshot.covariance
            norm = float(np.linalg.norm(p,2))
            if not math.isfinite(norm):
                return Validity(False, 'invalid_covariance_norm')
            tolerance = 64*np.finfo(float).eps*max(1.,norm)
            if np.max(abs(p-p.T)) > tolerance or np.linalg.eigvalsh((p+p.T)/2).min() < -tolerance:
                return Validity(False, 'invalid_covariance')
            if camera_status not in ('corrected','missing'):
                return Validity(False, 'camera_rejected_or_failed')
            if self.criteria is None:
                return Validity(False, 'no_justified_quality_policy; motion_disabled')
            if (last_correction_state_time is None or not math.isfinite(last_correction_state_time)
                    or last_correction_state_time > snapshot.state_time
                    or camera_status == 'corrected'
                    and abs(last_correction_state_time-snapshot.state_time) > self.grid_tolerance_s):
                return Validity(False, 'no_consistent_associated_camera_correction')
            if snapshot.state_time-last_correction_state_time > self.criteria.max_prediction_only_s:
                return Validity(False, 'prediction_only_budget_exceeded')
            if not self.criteria.accepts(previous,snapshot):
                return Validity(False, 'uncertainty_or_jump_rejected')
            return Validity(True, 'diagnostic_quality_accepted; '+self.criteria.justification)
        except (AttributeError, TypeError, ValueError, np.linalg.LinAlgError):
            return Validity(False, 'malformed_estimate')
