"""Hybrid pose-input terminal control; no simulator feedback dependencies.

Not obstacle avoidance or certified safety. Fixed goal per instance; reset by creating
an instance for a new goal. STOP/FAULT latch. See TERMINAL_CONTROL.md.
"""
from dataclasses import dataclass
import numpy as np
from control.lqr_controller import LQRTracker, wheel_commands, wrap


@dataclass(frozen=True)
class TerminalParameters:
    sample_time: float = .05
    k_distance: float = .8
    k_bearing: float = 2.
    k_heading: float = 2.
    max_v: float = .08
    max_omega: float = .6
    position_enter: float = .012
    position_exit: float = .025
    heading_enter: float = .04
    heading_exit: float = .07
    bearing_enter: float = .15
    bearing_exit: float = .45
    dwell: float = .8
    max_observed_v: float = .008
    max_observed_omega: float = .04
    timeout: float = 45.
    max_reapproaches: int = 3

    def __post_init__(self):
        if not all(np.isfinite(v) and v > 0 for v in vars(self).values()):
            raise ValueError('Parameters must be positive finite values')
        if not (self.position_enter < self.position_exit and self.heading_enter < self.heading_exit
                and self.bearing_enter < self.bearing_exit < np.pi/2):
            raise ValueError('Invalid hysteresis thresholds')
        if self.timeout <= self.dwell or not isinstance(self.max_reapproaches,int):
            raise ValueError('Invalid timeout/retry budget')


class TerminalController:
    def __init__(self, goal, parameters=TerminalParameters()):
        goal=np.asarray(goal,dtype=float)
        if goal.shape!=(3,) or not np.isfinite(goal).all():
            raise ValueError('Finite goal pose of shape (3,) required')
        self.goal=goal.copy(); self.p=parameters
        self.mode='bearing'; self.elapsed=0.; self.quiet=0.; self.previous=None
        self.reapproaches=0

    def command(self, estimated_pose):
        pose=np.asarray(estimated_pose,dtype=float)
        if pose.shape!=(3,) or not np.isfinite(pose).all():
            self.mode='fault'
            return np.zeros(2),self.mode
        if self.mode in ('stop','fault'):
            return np.zeros(2),self.mode
        p=self.p
        if self.elapsed >= p.timeout:
            self.mode='fault'; return np.zeros(2),self.mode
        self.elapsed+=p.sample_time
        if self.previous is None:
            speed,rate=np.inf,np.inf
        else:
            speed=np.linalg.norm(pose[:2]-self.previous[:2])/p.sample_time
            rate=abs(wrap(pose[2]-self.previous[2]))/p.sample_time
        self.previous=pose.copy()
        delta=self.goal[:2]-pose[:2]; rho=np.linalg.norm(delta)
        heading=wrap(self.goal[2]-pose[2])
        if self.mode in ('heading','coast') and rho > p.position_exit:
            self.reapproaches+=1
            if self.reapproaches > p.max_reapproaches:
                self.mode='fault'; return np.zeros(2),self.mode
            self.mode='bearing';self.quiet=0.
        if self.mode in ('bearing','approach') and rho <= p.position_enter:
            self.mode='heading'
        if self.mode=='coast':
            # Zero commands must persist while pose increments also indicate rest.
            if abs(heading)>p.heading_exit:
                self.mode='heading';self.quiet=0.
            elif speed<=p.max_observed_v and rate<=p.max_observed_omega and abs(heading)<=p.heading_exit and rho<=p.position_exit:
                self.quiet+=p.sample_time
                if self.quiet+1e-12>=p.dwell:self.mode='stop'
                return np.zeros(2),self.mode
            else:
                self.quiet=0.
                # Within hysteresis band, allow drift to finish; no threshold chatter.
                return np.zeros(2),self.mode
        if self.mode=='heading':
            if abs(heading)<=p.heading_enter:
                self.mode='coast';self.quiet=0.
                return np.zeros(2),self.mode
            return np.array([0.,np.clip(p.k_heading*heading,-p.max_omega,p.max_omega)]),self.mode
        alpha=wrap(np.arctan2(delta[1],delta[0])-pose[2])
        if self.mode=='bearing' and abs(alpha)<=p.bearing_enter:self.mode='approach'
        elif self.mode=='approach' and abs(alpha)>=p.bearing_exit:self.mode='bearing'
        omega=np.clip(p.k_bearing*alpha,-p.max_omega,p.max_omega)
        v=min(p.max_v,p.k_distance*rho)*max(0.,np.cos(alpha)) if self.mode=='approach' else 0.
        return np.array([v,omega]),self.mode


class TrajectoryTerminalController:
    """One-way scheduled handoff. Caller supplies fixed goal and reference end time.

    Returns feasible body input, wheel commands, mode, and wheel saturation flag.
    sample_time matches LQR dt. No global clock or implicit goal mutation.
    """
    def __init__(self, goal, end_time, parameters=TerminalParameters()):
        if not np.isfinite(end_time) or end_time < 0:raise ValueError('Invalid end time')
        self.end_time=end_time;self.tracker=LQRTracker(parameters.sample_time)
        self.terminal=TerminalController(goal,parameters)
        self.handed_off=False;self.last_time=-np.inf

    def command(self, estimated_pose, reference, time):
        if not np.isfinite(time) or time < self.last_time:raise ValueError('Time must be finite and monotonic')
        self.last_time=time
        if time>=self.end_time:self.handed_off=True
        if self.handed_off:
            raw,mode=self.terminal.command(estimated_pose)
        else:
            raw,mode=self.tracker.command(estimated_pose,reference)
            mode='tracking_'+mode
        wheels,body,saturated=wheel_commands(*raw)
        return body,wheels,mode,saturated
