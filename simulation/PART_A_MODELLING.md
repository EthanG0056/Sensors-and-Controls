# Part A — TurtleBot3 Burger mathematical modelling

This study separates velocity-actuator tracking from effective rolling response.
It uses deterministic open-loop simulation and ground truth only for identification
and evaluation. It does not implement navigation feedback or alter the ROBOTIS
model, mesh assets, licence, sensing, estimation, or LQR files.

## Reproduce and inspect

From the project root:

```sh
.venv/bin/python simulation/identify_drive.py
.venv/bin/python simulation/check_identification.py
```

Requires NumPy, MuJoCo and Matplotlib in the existing environment. Headless stepping
uses regular Python; `mjpython` is needed for macOS passive viewing, which this
experiment does not use. The earlier `validate_drive.py` and its results remain
unchanged. See [executed tables and plots](identification_results/RESULTS.md),
[all trial summaries](identification_results/summary.csv),
[fine-timestep summaries](identification_results/fine_summary.csv), and
[parameters, versions and source hashes](identification_results/models.json).
Individual trial CSVs retain every sample, including initial rest and transients.
The code asserts finite values, absence of MuJoCo numerical warnings, command
limits and preservation of protected source files.

## Nominal mathematics and physical parameters

Let wheel angular velocities be w_L and w_R, heading theta, wheel radius r,
and wheel-centre separation b. Positive body speed is forward; positive yaw is
anticlockwise viewed from above. Ideal planar no-slip differential drive gives

```math
v = r(w_R+w_L)/2, \qquad \Omega = r(w_R-w_L)/b
```

```math
\dot x=v\cos\theta,\quad \dot y=v\sin\theta,\quad \dot\theta=\Omega
```

```math
w_L^*=(v^*-b\Omega^*/2)/r,\qquad w_R^*=(v^*+b\Omega^*/2)/r.
```

Nominal r=0.033 m is supported by collision geometry, and b=0.160 m by XML body
positions. Commands use these unchanged dimensions and remain within ±6.67 rad/s.
No fitted coefficient is substituted into the XML or command generation.

The velocity actuator is a proportional torque servo, not a wheel-speed constraint:

```math
\tau_i=k_v(u_i-w_i).
```

There is no actuator activation state. A schematic wheel dynamics balance is

```math
J_{\rm eff}\dot w_i=k_v(u_i-w_i)-\tau_{f,i}-\tau_{\rm contact,i}.
```

J_eff includes armature and coupled mechanical effects; it is not identified here.
The unchanged XML has armature 0.01 kg m² and frictionloss 0.1 N m per wheel.
Coulomb/stiction and contact dynamics make a single linear actuator model only a
local approximation. Neither inertia nor a dynamic time constant is estimated
from the static fits below.

## Experimental protocol

For each gain (0.1 and 1.0), use five magnitudes and both signs in each mode:

| Mode | Identification magnitudes | Held-out magnitudes | Units |
|---|---|---|---|
| Straight translation | 0.04, 0.10, 0.16 | 0.07, 0.13 | m/s |
| Pure rotation command | 0.3, 0.7, 1.1 | 0.5, 0.9 | rad/s |

The split is fixed in source before fitting: six identification and four held-out
trials per mode/gain. Held-out points interpolate within the tested range; they
do not validate extrapolation, curved trajectories, or switching commands.

Each of 40 trials creates fresh MjData, including zero time, velocities, controls
and solver history. It steps with zero command for 2 s to permit initial contact
settling, then applies a fixed command for 6 s. Samples are evaluated consistently
at each state using mj_forward. Body velocity is computed with the base-origin
Jacobian: the origin is the wheel-axis midpoint, not the offset centre of mass.
World velocity is projected onto horizontal heading axes. Yaw is continuously
unwrapped; its rate is derived exactly from the rotation matrix and world angular
velocity, retaining roll/pitch effects rather than assuming yaw rate equals the
world angular-velocity z component. Logged actuator/contact forces are evaluated at the same state.

Reported responses average the final 2 s of driving. A trial is labelled settled
only if both (i) the maximum wheel-speed mean difference between the preceding
2 s and final 2 s and (ii) each wheel's final-window standard deviation are below
max(0.02 rad/s, 2% of the larger mean absolute wheel speed). This is an explicit
engineering criterion, not proof of equilibrium. All trials, including failures,
remain in the fit and reported tables; the fits therefore describe finite-window
response, not guaranteed steady state. There is no selective deletion of poor
performance. Full initial-rest logs permit inspection of the fixed settling period.

Every trial is repeated at timestep 0.001 s against nominal 0.002 s, preserving
all physical parameters. This tests numerical sensitivity, not experimental
repeatability or statistical uncertainty. No random noise is introduced.

## Identification, predictions and errors

Define measured late-window wheel mean s=(w_R+w_L)/2 and difference d=w_R-w_L.
Fit separate scalar, zero-intercept empirical rolling models per mode and gain:

```math
\hat v=c_v s,\qquad \hat\Omega=c_\Omega d,\qquad
\hat c=\frac{\sum_{j\in\mathrm{train}}z_j y_j}{\sum_{j\in\mathrm{train}}z_j^2}.
```

Every trial contributes one window mean. Samples within a trial are not treated
as independent replicates. The translation coefficient c_v has units m/rad;
c_Omega maps wheel-speed difference to yaw rate (rad/rad). These lump contact,
slip, geometry, nonplanar motion and operating-point dependence. They are **not**
physical radius or track measurements. In particular, combining coefficients
from different modes to claim a physical track would be unjustified.

For each wheel, gain and motion mode, fit a phenomenological signed dead-zone map:

```math
\hat w_i=a_i\operatorname{sign}(u_i)\max(|u_i|-d_i,0).
```

Search 2001 uniformly spaced dead-zone candidates between zero and the maximum
training command magnitude; for each candidate solve the nonnegative scalar
least-squares slope analytically and retain minimum training MSE. This is an
empirical command map, not a measurement of motor stiction. Separate motion-mode
fits acknowledge different loading; no automatic interpolation between modes is
validated. Fits enforce sign symmetry, so directional asymmetry remains residual.

Held-out predictions are evaluated in four ways:

1. Fitted rolling coefficient with **measured** wheels: isolates rolling-map error.
2. Fitted command-to-wheel map composed with fitted rolling map: fully open-loop
   command prediction, without measured-wheel input.
3. Nominal geometry with measured wheels: tests ideal rolling separately from servo.
4. Ideal requested body speed/yaw rate: tests the original command assumption.

RMSE=sqrt(mean((prediction-observation)²)); bias is mean signed prediction error;
maximum absolute error is also retained in JSON. Errors use physical units,
not percentages near zero. These are response-prediction errors, not evidence of
improved actual motion. Training trial ratio ranges are descriptive spread,
not confidence intervals. There are no physical experiments or independent
stochastic replicates to support statistical confidence claims.

## Findings and limits

The effective translation coefficients are approximately 0.02660 (kv=0.1) and
0.02607 m/rad (kv=1), below the nominal 0.033 m radius. Turn coefficients are
approximately 0.15421 and 0.15673, below nominal r/b=0.20625. Trial ratio spread
shows these constants do not explain every speed and direction.

Held-out command-model RMSE is 0.00136 / 0.00259 m/s for translation and
0.01740 / 0.03271 rad/s for turning (kv=0.1 / 1). These reduced errors describe
better predictions of the same simulated performance; no performance was changed.
The low-gain wheel response has substantial dead-zone behaviour: forward fitted
thresholds are about 1.57–1.60 rad/s versus about 0.158 rad/s at gain=1. Turn
thresholds differ, confirming motion-dependent loading. Near-zero response may
still be creeping rather than exactly zero; the clipped map is an approximation.

Only 34/40 nominal-timestep trials meet the wheel-settling criterion. Specifically,
8/10 low-gain forward and 6/10 low-gain turn trials pass; all gain=1 trials pass.
Do not describe all late-window means as steady-state measurements. Longer runs,
window analysis and possibly a creep model are required before treating the
low-speed, low-gain response as an equilibrium map. The wheel-step figure retains
transients; this study does not fit or validate a dynamic first-order model.

Base-attached floor contacts carry mean summed normal force about 3.46–3.50 N
in late windows. Their contact fractions vary approximately 0.40–1 across trials.
This aggregates the base mesh and rear support sphere; it does not isolate which
geometry causes resistance. Default friction, mesh contact, support loading and
initial settling all remain possible influences. Lateral motion and unintended
yaw are logged. Rolling residuals are evidence against ideal no-slip motion but
are **not** direct contact-point slip measurements. Resolving slip requires
wheel contact-point tangential velocities/forces and a contact-specific study.

Halving timestep changes mean forward response by up to 0.00573 m/s at kv=0.1
and 0.01124 m/s at kv=1. For turn trials, yaw-rate changes reach 0.00331 and
0.00422 rad/s. Forward trials also show unintended-yaw sensitivity up to about
0.099 rad/s. Thus forward response is not demonstrated numerically converged;
this systematic uncertainty can exceed held-out fitting error. Reported coefficients
are provisional for the stated timestep/contact solver, not portable constants.

## Implications for Part D and next evidence

Use kv=1 as the better-tracking candidate for further study, with explicit
actuator dynamics and input limits; this is not yet a final controller choice.
Do not silently invert a dead-zone fit as friction compensation. Its accuracy
near reversals and on curved trajectories has not been validated.

Before final LQR tuning: extend failed settling trials; test timesteps below
0.001 s and solver settings until convergence or quantify a residual envelope;
isolate base-mesh/support/wheel contacts in temporary sensitivity models; test
mixed translation/rotation and command reversals; identify a dynamic actuator
model from steps and validate its transients on held-out inputs. Retain signed
lateral/yaw residuals in the model-error budget.

Derive trajectory-error dynamics about feasible moving references, discretise at
the chosen controller sample period, check stabilisability and actuator saturation,
and evaluate robustness to identified residuals. A stationary unicycle
linearisation cannot control lateral error. Ground truth stays confined to this
validation harness; final assessed feedback comes from the approved sensing and
estimation pipeline. No LQR implementation is added by this study.

## Executed verification

`check_identification.py` passed against the saved study: source SHA256 checks,
a deliberately corrupted held-out-data test showing fitted parameters unchanged,
an exactly reproduced independent held-out trial, 40 complete 4001-sample logs,
the actuator torque law throughout all logs, and agreement of integrated heading
rate with unwrapped orientation within 0.005 rad. These checks verify the evidence
pipeline; they do not establish physical accuracy or numerical convergence.
