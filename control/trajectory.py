"""Time-indexed feasible reference poses and body velocities (SI units)."""
from dataclasses import dataclass
import numpy as np
from scipy.interpolate import CubicSpline


@dataclass(frozen=True)
class Reference:
    pose: np.ndarray
    input: np.ndarray


DURATIONS = {'straight':12., 'curve':15., 'waypoint':15., 'low_speed':12., 'rotation':8., 'stationary':6.}
_WAYPOINTS = np.array([[0.,0.],[.4,0.],[.7,.25],[1.1,.25]])
_SPLINE = CubicSpline([0.,5.,10.,15.], _WAYPOINTS, bc_type=((1,[.08,0.]),(1,[.08,0.])))


def reference(name, t):
    if name not in DURATIONS:
        raise ValueError(name)
    # Terminal sample is included, but moving references are not abruptly stopped.
    t = float(np.clip(t, 0, DURATIONS[name]))
    if name in ('straight','low_speed'):
        v = .08 if name=='straight' else .01
        return Reference(np.array([v*t,0.,0.]),np.array([v,0.]))
    if name=='curve':
        v,w = .08,.3
        return Reference(np.array([v/w*np.sin(w*t),v/w*(1-np.cos(w*t)),w*t]), np.array([v,w]))
    if name=='rotation':
        return Reference(np.array([0.,0.,.4*t]),np.array([0.,.4]))
    if name=='stationary':
        return Reference(np.zeros(3),np.zeros(2))
    xy, velocity, acceleration = _SPLINE(t), _SPLINE(t,1), _SPLINE(t,2)
    speed = float(np.linalg.norm(velocity))
    if speed < 1e-6:
        raise ValueError('Waypoint reference has undefined tangent')
    omega = (velocity[0]*acceleration[1]-velocity[1]*acceleration[0])/speed**2
    return Reference(np.r_[xy,np.arctan2(velocity[1],velocity[0])], np.array([speed,omega]))
