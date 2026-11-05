"""Extracted team timed detour heuristic; not collision-free path planning.

Only estimated pose, detached LiDAR and represented simulation time are inputs.
No terminal controller is invoked. Docking requests require an external stop.
"""
import numpy as np
from control.trajectory import Reference

NAVIGATION_SPEED = .08
NAVIGATION_TARGET_X = 1.85
AVOID_ENTER_DISTANCE = .20
TURN_TIME, DRIVE_TIME, REJOIN_TIME = 2., 5., 3.
DOCKING_TRIGGER_DISTANCE = .90


class NavigationSupervisor:
    def __init__(self, normal_controller, marker_position, sample_time=.05):
        self.normal = normal_controller
        self.marker = np.asarray(marker_position, float).copy()
        if self.marker.shape != (2,) or not np.isfinite(self.marker).all():
            raise ValueError('Finite planar mapped marker required')
        if not np.isfinite(sample_time) or sample_time <= 0:
            raise ValueError('Positive sample time required')
        self.dt = sample_time
        self.navigation_time = 0.
        self.phase = None
        self.phase_start = None
        self.direction = 1.
        self.last_time = None
        self.ready = False
        self.stop_reason = None

    def update(self, estimated_pose, obstacle, state_time):
        self.ready = False
        pose = np.asarray(estimated_pose, float)
        if (pose.shape != (3,) or not np.isfinite(pose).all()
                or not np.isfinite(state_time) or state_time < 0
                or self.last_time is not None and not np.isclose(
                    state_time-self.last_time, self.dt, rtol=0, atol=1e-8)):
            raise ValueError('Invalid pose or missed/duplicate navigation interval')
        distance, angle = np.inf, 0.
        if obstacle is not None:
            value = np.asarray(obstacle, float)
            if (value.shape != (2,) or np.isnan(value).any() or value[0] < 0
                    or not np.isfinite(value[1]) or np.isneginf(value[0])):
                raise ValueError('Invalid detached LiDAR obstacle')
            distance, angle = value
        self.last_time = float(state_time)
        self.pose = pose.copy()
        # Preserve the team's distance trigger, but never grant docking clearance.
        if np.linalg.norm(pose[:2]-self.marker) < DOCKING_TRIGGER_DISTANCE:
            self.stop_reason = 'terminal_clearance_unavailable'
        if self.stop_reason is not None:
            return
        if self.phase is None and distance < AVOID_ENTER_DISTANCE:
            self.phase, self.phase_start = 'turn_out', state_time
            self.direction = -1. if angle > 0 else 1.
        if self.phase is not None:
            elapsed = state_time-self.phase_start
            duration = {'turn_out': TURN_TIME, 'drive_past': DRIVE_TIME, 'rejoin': REJOIN_TIME}[self.phase]
            if elapsed >= duration-1e-10:
                self.phase = {'turn_out': 'drive_past', 'drive_past': 'rejoin', 'rejoin': None}[self.phase]
                self.phase_start = state_time
                if self.phase is None:
                    self.navigation_time = min(max(self.navigation_time, pose[0]/NAVIGATION_SPEED),
                                               NAVIGATION_TARGET_X/NAVIGATION_SPEED)
        if self.phase is None:
            self.navigation_time += self.dt
        self.ready = True

    def reference(self, unused_time):
        x = min(NAVIGATION_SPEED*self.navigation_time, NAVIGATION_TARGET_X)
        return Reference(np.array([x, 0., 0.]), np.array([
            NAVIGATION_SPEED if x < NAVIGATION_TARGET_X else 0., 0.]))

    def command(self, estimated_pose, reference_sample, reference_time):
        if not self.ready or self.stop_reason is not None:
            raise ValueError('Navigation update required or clearance unavailable')
        self.ready = False  # one command per represented measurement
        pose = np.asarray(estimated_pose, float)
        if pose.shape != (3,) or not np.array_equal(pose, self.pose):
            raise ValueError('Navigation pose differs from validated update')
        if self.phase is None:
            output = self.normal.command(pose.copy(), reference_sample,
                                       min(self.navigation_time, self.normal.end_time-self.dt))
        else:
            if self.phase == 'turn_out': v, omega = 0., self.direction*.35
            elif self.phase == 'drive_past': v, omega = .08, 0.
            else: v, omega = .07, float(np.clip(-.6*pose[1]-.8*pose[2], -.18, .18))
            wheels = np.array([(v-omega*.160/2)/.033, (v+omega*.160/2)/.033])
            output = np.array([v, omega]), wheels, 'tracking_avoidance', False
        body, wheels, mode, saturated = output
        if (np.asarray(body).shape != (2,) or np.asarray(wheels).shape != (2,)
                or not np.isfinite(body).all() or not np.isfinite(wheels).all()
                or np.max(np.abs(wheels)) > 6.67 or not mode.startswith('tracking_')):
            raise ValueError('Invalid or out-of-phase navigation output')
        return output
