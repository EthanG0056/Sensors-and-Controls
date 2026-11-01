# Part D — discrete-time trajectory tracking

This implementation is a pose-input kinematic LQR tracker. The standalone MuJoCo
benchmark intentionally supplies exact simulator pose: **IDEAL-STATE DIAGNOSTIC
ONLY**. No camera, EKF, LiDAR, waypoint detection, final navigation or docking
integration is implemented or claimed. The controller accepts an external
`estimated_pose=[x,y,theta]` in metres/radians, without importing MuJoCo or accessing
sensors. An EKF adapter must provide the same world frame, body origin, units,
heading convention and sample timestamp. Estimation delay/noise remain untested.

## Nonlinear plant and sign convention

The planar no-slip model, with positive x forward at heading zero and positive
heading anticlockwise, is

```math
\dot x=v\cos\theta,\quad\dot y=v\sin\theta,\quad\dot\theta=\omega,
\qquad v=\frac r2(w_R+w_L),\quad\omega=\frac r b(w_R-w_L).
```

This is a nominal kinematic model, not the complete MuJoCo contact/actuator plant.
Keep physical wheel radius r=0.033 m, track b=0.160 m and wheel command limit
6.67 rad/s. Earlier Part A effective coefficients remain empirical and are not
silently substituted into this controller. Their pure-mode static fits have not
been validated as a mixed-motion dynamic model or as an inverse compensation law.

For a feasible reference (x_r,y_r,theta_r,v_r,omega_r), define reference-minus-actual
position in the **actual robot body frame**, and heading error wrapped to [-pi,pi]:

```math
\begin{bmatrix}e_x\\e_y\end{bmatrix}=
\begin{bmatrix}\cos\theta&\sin\theta\\-\sin\theta&\cos\theta\end{bmatrix}
\begin{bmatrix}x_r-x\\y_r-y\end{bmatrix},\quad e_\theta=\mathrm{wrap}(\theta_r-\theta).
```

Thus e_x>0 means the reference is ahead, e_y>0 means left, and e_theta>0 means
its heading is anticlockwise from actual heading. Differentiate the rotating
frame (R^T_dot=-omega J R^T) and substitute both unicycle velocities:

```math
\dot e_x=\omega e_y+v_r\cos e_\theta-v,\quad
\dot e_y=-\omega e_x+v_r\sin e_\theta,\quad
\dot e_\theta=\omega_r-\omega.
```

These equations are exact under the planar kinematic assumptions, locally away
from the heading wrap discontinuity. With input increments
u_delta=[v-v_r, omega-omega_r]^T, linearise at e=0, u_delta=0:

```math
\dot e=Ae+Bu_\delta,\qquad
A=\begin{bmatrix}0&\omega_r&0\\-\omega_r&0&v_r\\0&0&0\end{bmatrix},\quad
B=\begin{bmatrix}-1&0\\0&0\\0&-1\end{bmatrix}.
```

The nonlinear products u_delta,omega e_x/y and the quadratic cosine residual are
neglected. Jacobians are tested by centred finite differences at forward,
reverse, curved, rotation and stationary operating points. A separate world-pose
finite-difference test verifies the exact nonlinear rotating-frame derivation.

## Controllability and local limitations

The continuous controllability matrix is [B AB A²B]. AB contains columns
[0,omega_r,0]^T and [0,-v_r,0]^T; together with B this spans three states if either
reference velocity is nonzero. At zero velocities only rank two is available.

| Reference | Continuous rank | Interpretation |
|---|---:|---|
| Straight, v_r≠0 | 3 | Heading input generates lateral error motion through v_r |
| Curved, v_r≠0, omega_r≠0 | 3 | Both couplings are available |
| Rotation in place, v_r=0, omega_r≠0 | 3 | Rotating body axes couple longitudinal and lateral errors |
| Stationary, v_r=omega_r=0 | 2 | Lateral integrator is uncontrollable and not stabilisable |

Rotation in place is **not** the stationary linearisation. Its local LQR uses
small translational corrections while rotating. At very low speed the system
is mathematically controllable but lateral correction becomes weak/slow and
numerical conditioning worsens. Singular values are saved in benchmark metadata.
The sampled pair is also tested directly: pathological sampling of a rotation
can lose controllability, so continuous rank alone is insufficient.

The nonlinear unicycle can manoeuvre laterally through sequences of forward and
rotational motions (the input vector fields and their Lie bracket span the pose
tangent space). This does not make the stationary first-order linearisation
controllable, nor give a smooth static full-pose LQR parking law. The tracker
therefore explicitly returns zero command with mode `stationary_stop` when both
reference velocities vanish; it makes no attempt to remove stationary pose
error. A separately designed terminal/parking strategy is required later.

## Discretisation and LQR design

Controller interval T_s=0.05 s (20 Hz); MuJoCo physics steps are 0.002 s or 0.001 s,
so commands are held for 25 or 50 steps. Reference-dependent matrices are frozen
at each controller update and exactly zero-order-hold discretised:

```math
\exp\left(T_s\begin{bmatrix}A&B\\0&0\end{bmatrix}\right)
=\begin{bmatrix}A_d&B_d\\0&I\end{bmatrix}.
```

This avoids A inversion and handles singular A. Exactness applies to the frozen
linear model, not the varying nonlinear reference or contact dynamics. A numerical
ODE integration test checks the discretisation independently. 20 Hz is a stated
initial design choice, not a proven optimum; the fast gain=1 servo time scale may
be shorter than this and is stepped by MuJoCo between controller updates.

Minimise the discrete infinite-horizon cost sum(e_k^T Q e_k + u_delta,k^T R
u_delta,k). The implemented DARE is

```math
A_d^TPA_d-P-A_d^TPB_d(R+B_d^TPB_d)^{-1}B_d^TPA_d+Q=0,
\quad K=(R+B_d^TPB_d)^{-1}B_d^TPA_d,\quad
u=[v_r,\omega_r]^T-Ke.
```

The implementation uses SciPy's
[solve_discrete_are](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solve_discrete_are.html)
and [matrix exponential](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.expm.html).
Tests check the DARE residual and frozen closed-loop eigenvalues inside the unit
circle. The controller checks discrete rank and eigenvalues and propagates solver
failure; it does not hide failures by adding artificial reference velocities.
Gains are cached at identical reference values, never interpolated or rounded.

Balanced weights use Bryson-style normalisation of illustrative tolerances:
position 0.08 m, heading 0.25 rad; input increments 0.12 m/s and 0.8 rad/s.
These express design trade-offs, not assessment acceptance thresholds.

| Setting | diag(Q) | diag(R) | Rationale |
|---|---|---|---|
| balanced | 156.25, 156.25, 16 | 69.444444, 1.5625 | Inverse squared tolerances |
| tracking | 625, 625, 64 | 69.444444, 1.5625 | Fourfold error penalty |
| effort | 156.25, 156.25, 16 | 277.777778, 6.25 | Fourfold increment penalty |

Q and R are discrete stage-cost matrices, not an exact conversion of a continuous
cost integral. Positive Q ensures detectability; stabilisability is required.
Re-solving a frozen DARE along a time-varying path is gain scheduling, not a
finite-horizon time-varying Riccati solution. Stable frozen eigenvalues alone do
not prove nonlinear/time-varying global stability. Saturation and model mismatch
further invalidate an unconstrained optimality claim.

## Wheel conversion and saturation

```math
w_L^*=(v-b\omega/2)/r,\quad w_R^*=(v+b\omega/2)/r,\quad
s=\max(1,|w_L^*|/6.67,|w_R^*|/6.67).
```

Send both wheel commands divided by the **same** s; feasible body input is also
[v,omega]/s. This retains turning sign and curvature for nonzero v rather than
independently clipping wheels. There is no integral windup because there is no
integrator. Saturation still changes the closed-loop dynamics. Limits are command
limits, not guarantees of actual wheel speed/torque or traction.

## Benchmark protocol and metrics

Run from repository root:

```sh
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python simulation/benchmark_lqr.py
.venv/bin/python simulation/check_identification.py
```

The final command verifies preserved Part A evidence; it does not regenerate it.
Output is confined to `simulation/lqr_results` and new control/test/document files.
The benchmark hashes and verifies existing models, licence/assets, Part A results,
and movement tests remain unchanged. It records source hashes, versions, exact
weights, initial conditions and controllability singular values in metadata.

Reference paths: straight 0.08 m/s for 12 s; circle arc 0.08 m/s, 0.3 rad/s for
15 s; cubic-spline waypoint path through (0,0), (0.4,0), (0.7,0.25), (1.1,0.25)
at times 0,5,10,15 s, with endpoint derivatives (0.08,0) m/s. Derivatives provide
speed, tangent heading and curvature yaw rate; references are checked numerically
against the nonlinear model and wheel limits. The waypoint path is a predefined
time trajectory, not a visual waypoint selector. It passes through the points
without stopping. No abrupt terminal stop or post-horizon docking is scored.

Each path runs for all three weights, both original kv=0.1 and experimental
kv=1.0, and initial poses (0,0,0), (-0.08,0.06,0.2), (0.08,-0.06,-0.2). Each
fresh model/data settles 2 s at zero command from that pose before time zero.
The measured settled state, not an artificially corrected state, starts tracking.
This gives 54 core trials. Six additional balanced-weight, perturbed-pose trials
cover low-speed translation (0.01 m/s), rotation (0.4 rad/s) and stationary
reference. Six trials repeat balanced/perturbed main paths at half physics timestep:
66 diagnostic trials in total. Primary comparison uses kv=1 because Part A showed
better wheel tracking; original kv=0.1 is retained throughout as a plant baseline.

Every controller sample logs exact simulator pose, reference, signed body errors,
raw body commands, feasible commands, commanded and measured wheel speeds,
saturation and controller mode. Ideal state is deliberately the feedback source
for this diagnostic harness alone; it is not concealed behind an EKF interface.

Position RMSE=sqrt(mean(e_x²+e_y²)); heading RMSE uses wrapped heading errors.
Final position error is Euclidean; final heading is signed and wrapped. RMSE
includes initial errors/transients and the final sample. Saturation frequency is
the fraction of applied hold intervals with s>1. Effort reports sums T_s v²,
T_s omega² and T_s(w_L*²+w_R*²) over actual hold intervals, excluding the unused
terminal command. Wheel effort is a commanded-input proxy, **not mechanical or
electrical energy**. Plots use mean wheel² effort (integral divided by duration)
for fairer comparison of different path durations. Configurations are evaluated
on the same test grid; this is a diagnostic comparison, not an independently
held-out controller tuning study or statistical uncertainty estimate.

See [executed results and plots](../simulation/lqr_results/RESULTS.md) and
[per-trial summary](../simulation/lqr_results/summary.csv). All trials are retained,
including poor tracking, stationarity and saturation. There are zero estimated-state
runs. Neither successful final navigation nor docking can be inferred from these
ideal-state diagnostics.

## Executed findings and design implications

All 66 MuJoCo 3.17.0 trials completed without numerical warnings. Eight automated
unit tests passed, covering nonlinear signs, numerical Jacobians, continuous and
discrete controllability, exact discretisation, Riccati residual/stability,
saturation, feasible reference derivatives, and ideal-kinematic recovery.
The recovery test uses a 40 s straight run; a 12 s horizon does not remove 99%
of the perturbed initial error with balanced weights. A results checker independently
recomputed metrics and wheel conversion/limits across all 66 saved logs, verified
stationary stop behaviour and checked protected/source hashes. The existing
Part A checker also passed without regenerating modelling evidence.

Averaging the nine main-path/initial-condition trials per setting:

| kv | Setting | Position RMSE (m) | Final position error (m) | Mean wheel² effort | Saturated (%) |
|---:|---|---:|---:|---:|---:|
| 0.1 | balanced | 0.054045 | 0.049856 | 43.786633 | 8.907 |
| 0.1 | tracking | 0.037006 | 0.026464 | 45.916150 | 17.037 |
| 0.1 | effort | 0.090150 | 0.095329 | 39.979852 | 0.556 |
| 1 | balanced | 0.031523 | 0.020587 | 23.958773 | 0.843 |
| 1 | tracking | 0.024626 | 0.010900 | 24.628558 | 2.546 |
| 1 | effort | 0.047020 | 0.041154 | 22.494252 | 0.000 |

For kv=1, tracking weights lower mean position RMSE by about 22% relative to
balanced while raising the command-effort proxy about 2.8% and saturation from
0.843% to 2.546%. This is the most accurate setting on this diagnostic grid,
not a proven final choice under estimator noise, delays or other paths. Original
kv=0.1 shows larger lag and much more saturation on curves: tracking weights
saturate up to 47% of intervals in the perturbed curve trial. More aggressive
feedback cannot remove actuator/contact constraints or restore unconstrained
LQR optimality.

The perturbed low-speed run ends with position errors 0.049384 / 0.037764 m and
heading errors approximately 0.1185 rad for kv=0.1 / 1: weak lateral coupling
and actuator friction limit finite-horizon recovery. Rotation-in-place diagnostics
reduce position error to 0.017193 / 0.017970 m but retain heading errors
0.213876 / 0.057176 rad. Stationary references retain approximately 0.100 m
position and -0.200 rad heading error under the explicit zero-command policy.
These are documented limitations, not docking successes.

The six half-timestep pairs change position RMSE by at most 0.000703 m and
final position-error norm by at most 0.003132 m. Differences between scalar
error norms do not bound the full vector difference between trajectories.
Feedback reduces observed sensitivity in this small tested subset compared with
the earlier open-loop study, but neither numerical convergence nor robustness
at all operating points is established. Further timestep/sample-period sweeps
and contact/servo modelling remain necessary.

Nominal kinematics omit actuator lag, dry friction, slip and base/support contact.
Without integral or validated feedforward compensation, persistent tracking bias
is expected and observed even with exact pose feedback. The trajectory linearisation
is local; the small tested pose errors do not establish a global capture region.
Do not extrapolate these results to large heading errors, obstacle avoidance,
physical hardware or perception-based navigation. No noisy/delayed estimated-state
benchmark has been run. The waypoint path has no stop-to-dock segment.

Next: validate an actuator-aware or disturbance-augmented model and a terminal
strategy, quantify sample-period and timestep sensitivity, then pass timestamped
EKF pose into the same controller interface and compare estimated-state against
this labelled ideal diagnostic. Gate any final navigation claim on the separate
camera/EKF/LiDAR integration and corresponding end-to-end tests.

Recheck saved evidence without rerunning the benchmark:

```sh
.venv/bin/python simulation/check_lqr_results.py
```
