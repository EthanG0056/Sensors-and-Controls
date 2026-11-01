# TurtleBot3 open-loop modelling validation

## Loading repair and preserved sources

The experimental robot filename began with U+200B ZERO WIDTH SPACE. The scene
includes the plain ASCII filename, so the operating system could not resolve it.
Renamed that file to `robotis_tb3/turtlebot3_burger_gain_test.xml`; no include or
asset path changes were needed. All four referenced STL assets load successfully.
On inspection **both** robot files had kv=1.0. Restored only the original robot's
velocity default to kv=0.1, matching the requested baseline. The experimental
robot remains kv=1.0. These robot XML files now differ only in kv.

`simulation/turtlebot.py`, the original scene, all mesh assets, and ROBOTIS
`robotis_tb3/LICENSE` are unchanged. The existing interactive script still runs
the gain scene with `mjpython simulation/turtlebot.py`. No sensing, EKF, or control
files were edited. This checkout has no .git directory, so no historical version
comparison was possible. Prior user-provided measurements are retained below.

## Reproduction

From the repository root, using the supplied Python 3.12 environment:

```sh
.venv/bin/python simulation/validate_drive.py
.venv/bin/python simulation/validate_drive.py --joint-friction-scale 0
```

Executed with Python 3.12 and MuJoCo 3.17.0, headless, timestep 0.002 s, default
Euler integrator. No passive viewer was exercised. Each fresh model performs
forward 0.10 m/s for 3 s, immediately rotates at 0.5 rad/s for 2 s, then stops
for 2 s. No settling interval is added, matching the existing movement test.
The rotation therefore includes the forward-to-rotation transient.

`validation_results/*.csv` contain 3500 post-step samples per run with simulation
time, phase time, commanded/actual wheel velocities, x/y, displacement from the
initial pose, unwrapped heading, and phase-relative ideal/actual motion.
Displacement is endpoint Euclidean displacement, not travelled path length.
JSON files contain phase metrics, configuration, versions, and XML/STL SHA256
hashes. Wheel RMSE and bias use all post-step phase samples; mean wheel speed is
not an end-of-phase reading. Measured wheel-angle integrals use trapezoidal
quadrature including the phase-start speed. No ground truth is used as feedback;
commands are fixed open-loop. This runner is solely for validation.

## Executed results

| Quantity | Baseline kv=0.1 | Gain kv=1.0 | Ideal |
|---|---:|---:|---:|
| Forward endpoint distance (m) | 0.116874 | 0.219104 | 0.300000 |
| Forward final L/R speed (rad/s) | 1.437498 / 1.479659 | 2.886527 / 2.887285 | 3.030303 / 3.030303 |
| Forward mean L/R speed (rad/s) | 1.391798 / 1.424199 | 2.866328 / 2.866961 | 3.030303 / 3.030303 |
| Forward wheel RMSE L/R (rad/s) | 1.647815 / 1.616548 | 0.192037 / 0.191504 | 0 / 0 |
| Rotation heading change (rad) | 0.048235 | 0.655650 | 1.000000 |
| Rotation final L/R speed (rad/s) | -0.061813 / 0.061654 | -1.062531 / 1.069897 | -1.212121 / 1.212121 |
| Rotation wheel RMSE L/R (rad/s) | 1.172196 / 1.121056 | 0.237713 / 0.157431 | 0 / 0 |
| Translation during rotation (m) | 0.001999 | 0.014782 | 0 |

The baseline reproduces the supplied earlier forward distance 0.1169 m,
rotation 0.0482 rad, forward final wheel speeds 1.437/1.480 and rotation
-0.062/+0.062 rad/s to their quoted precision. Gain improves tracking, but its
forward distance error is still -0.080896 m and heading error -0.344350 rad.
Full phase trajectory RMSE, signed errors, and stop response are in JSON.

## Interpretation for Part A

For ideal rolling, with r=0.033 m and b=0.160 m:

    v = r (w_R + w_L)/2
    yaw_rate = r (w_R - w_L)/b
    x_dot = v cos(theta), y_dot = v sin(theta), theta_dot = yaw_rate
    w_L* = (v* - b yaw_rate*/2)/r
    w_R* = (v* + b yaw_rate*/2)/r

These are kinematic assumptions, not a complete plant model. XML wheel body
centres have y=+/-0.08 m. Compiled collision-mesh world bounds at the initial
pose give wheel z approximately [0, 0.066] m and x +/-0.033 m, supporting the
nominal geometric radius. This does **not** establish an effective rolling
radius under load/contact.

The velocity actuators have no activation state (dyntype=0), unit default gear,
and torque law tau=kv*(command - wheel speed), not ideal speed enforcement.
Joint armature is 0.01 kg m^2 and dry frictionloss is 0.1 N m per wheel.
The friction-only speed deficit scale is 0.1/kv: 1 rad/s at baseline versus
0.1 rad/s at gain=1. Load torque and contact also matter; this is not a fitted
steady-state solution. The bare armature/kv time scales are 0.1 and 0.01 s;
actual transients include wheel/base inertia and contact. No new torque limit
was introduced.

A controlled in-memory experiment set only wheel joint frictionloss to zero:

| Result | kv=0.1 | kv=1.0 |
|---|---:|---:|
| Forward distance (m) | 0.183738 | 0.225772 |
| Rotation heading (rad) | 0.464459 | 0.718207 |

Thus joint friction materially affects the result, especially at low gain,
but removing it does not recover ideal motion. These are sensitivity runs,
not proposed physical parameter changes; source XML remains unchanged.

Using measured wheel-angle integrals instead of commanded speeds predicts
forward distances 0.139344 / 0.283703 m and rotation angles 0.062562 /
0.874234 rad (baseline / gain). Their differences from ground truth show that
actuator tracking error alone cannot explain the motion. Possible mechanisms
include slip, nonplanar motion, effective radius/track, and mesh contact.
Default wheel/base/floor friction coefficients are [1, 0.005, 0.0001]. The small
rear support sphere uses condim=1 (normal contact only). The initial base
collision mesh extends approximately 0.000123 m below the floor, so base-floor
contact and initial settling deserve explicit investigation; geometry alone
does not establish sustained drag. No contact parameters were changed and no
single contact mechanism has been established causally.

## Next step for Parts A and D

Before LQR, perform independent, settled forward and clockwise/anticlockwise
rotation trials over multiple speeds and both signs, logging wheel motion,
ground-truth body motion, actuator torque and contact forces/slip. Repeat at a
smaller timestep to assess numerical convergence. Separate wheel-servo dynamics
from rolling geometry: fit effective radius/track against measured wheel
integrals on training trials and validate on held-out trials. Do not absorb
slip into geometry without checking speed/direction dependence. Quantify
parameter uncertainty and residual errors; retain the unmodified baseline.

Then choose and document a validated plant and sampling period for LQR. Derive
a trajectory-error linearisation about feasible moving references, discretise,
check controllability/stabilisability and saturation, and only then select
Q/R and evaluate held-out paths. A stationary unicycle linearisation does not
give lateral controllability. The final assessed controller must receive the
approved sensing/estimation outputs; ground truth remains evaluation-only.

## Systematic Part A experiments

The subsequent signed-speed identification and held-out study is documented in
[PART_A_MODELLING.md](PART_A_MODELLING.md), with generated
[results and plots](identification_results/RESULTS.md). It preserves this earlier
baseline experiment and explicitly reports settling and timestep limitations.
