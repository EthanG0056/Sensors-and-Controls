# Part A — observability of the implemented localisation model

This companion section closes the observability-analysis gap in
[Part A modelling](PART_A_MODELLING.md). That earlier document remains byte-for-byte
unchanged because historical benchmark metadata pins its SHA256. This section and
its synthetic tests are new mathematical evidence, not regenerated MuJoCo results.

The [assessment guide](../Sensors_Control_Integrated_Project%20v2.pdf) assigns Part A
20 marks for state-space modelling, observability and controllability. The relevant
contribution here is a justified limit of the actual model, not an unsupported
full-observability claim. No estimator, sensor or controller implementation changes.

## 1. Actual state, inputs and outputs

Use world-frame planar pose and body-frame inputs in SI units:

```math
\mathbf{x}=[x,y,\theta]^T,\quad \mathbf{u}=[v,\omega]^T,\qquad
\dot{\mathbf{x}}=f(\mathbf{x},\mathbf{u})=
[v\cos\theta,\ v\sin\theta,\ \omega]^T.
```

Heading is positive anticlockwise; bearing is positive left from robot forward.
For known wheel radius r and track b, nominal rolling gives
v=r(w_L+w_R)/2 and omega=r(w_R-w_L)/b. Part A's actuator/contact discrepancies
remain relevant: commanded wheel speeds are not automatically measured motion.
This analysis assumes known body inputs and the ideal planar kinematic model;
unknown slip, biases or extra states need their own observability analysis.

For one stationary, correctly associated landmark at known world coordinates
m=(m_x,m_y), let d_x=m_x-x, d_y=m_y-y and rho=sqrt(d_x²+d_y²). The EKF expects

```math
\mathbf{z}_k=h(\mathbf{x}_k;\mathbf{m})+\boldsymbol\nu_k,\qquad
h=\begin{bmatrix}\rho\\\beta\end{bmatrix},\quad
\beta=\operatorname{wrap}(\operatorname{atan2}(d_y,d_x)-\theta).
```

Structural observability is analysed in the noiseless model with known inputs.
Noise tuning and covariance conditioning are not tests of structural observability.
Derivatives below require rho>0 and local angle coordinates; circular differences
handle the equivalent representations at the bearing wrap boundary.

| Equation / operation | Existing implementation |
|---|---|
| Discrete pose propagation | [`estimation/ekf.py`, `motion_model`](../estimation/ekf.py#L18) |
| Discrete state-transition Jacobian | [`motion_jacobian`](../estimation/ekf.py#L51) |
| Landmark range/bearing | [`camera_model`](../estimation/ekf.py#L80) |
| Observation Jacobian | [`camera_jacobian`](../estimation/ekf.py#L102) |
| Wrapped innovation and EKF correction | [`TurtleBotEKF.update_camera`](../estimation/ekf.py#L145) |
| Nominal wheel/body conversion | [`control/lqr_controller.py`, `wheel_commands`](../control/lqr_controller.py#L68) |
| Physical-response identification | [`simulation/identify_drive.py`, `trial` / `fit`](identify_drive.py#L33) |

The world-pose linearisation used here is **not** the LQR body-frame
tracking-error linearisation in `control/lqr_controller.py`.

## 2. Jacobians and instantaneous information

At a nominal pose/input, the continuous world-pose Jacobian is

```math
A=\frac{\partial f}{\partial\mathbf{x}}=
\begin{bmatrix}0&0&-v\sin\theta\\0&0&v\cos\theta\\0&0&0\end{bmatrix},\qquad
H=\frac{\partial h}{\partial\mathbf{x}}=
\begin{bmatrix}
-d_x/\rho&-d_y/\rho&0\\
d_y/\rho^2&-d_x/\rho^2&-1
\end{bmatrix}.
```

This H equals the actual EKF Jacobian. Its two rows are independent for rho>0,
but its nullspace is one-dimensional:

```math
g(\mathbf{x})=\begin{bmatrix}d_y\\-d_x\\1\end{bmatrix},\qquad Hg=0.
```

The observable local quantities are landmark distance and robot-relative bearing.
The missing direction changes position tangentially around the landmark and
heading by the same infinitesimal rotation. It is not simply 'heading alone'.

For the implemented discrete map x_(k+1)=F_map(x_k,u_k), define world displacement
Delta_x, Delta_y. Its state Jacobian, away from the angular coordinate cut, is

```math
F_k=\frac{\partial F_{\mathrm{map}}}{\partial\mathbf{x}_k}=
\begin{bmatrix}1&0&-\Delta_y\\0&1&\Delta_x\\0&0&1\end{bmatrix}.
```

`motion_model` uses the constant-input circular arc for |omega|>=1e-6; below that
threshold it uses straight translation and leaves heading unchanged. The latter
is the implemented small-rate approximation, not an exactly nonzero yaw update.
Tests check the Jacobian against both actual code branches, including reverse
motion, pure rotation and wrap crossings. They do not silently replace this map.

## 3. Frozen linearisation versus trajectory observability

If A and H are frozen at one instant, the LTI observability matrix is

```math
\mathcal O_{\mathrm{frozen}}=
\begin{bmatrix}H\\HA\\HA^2\end{bmatrix},\qquad A^2=0.
```

The only potentially nonzero column of HA is its third:

```math
(HA)_{:,3}=\begin{bmatrix}
v(d_x\sin\theta-d_y\cos\theta)/\rho\\
-v(d_y\sin\theta+d_x\cos\theta)/\rho^2
\end{bmatrix}.
```

For v!=0 and rho>0 these two entries cannot both vanish. Together with rank(H)=2
and g_theta=1, that gives frozen rank three. With v=0, A=0 and rank is two,
including pure rotation (omega may be nonzero). These are results for the frozen
LTI approximation, not a proof of nonlinear full-pose observability during travel.

For a moving trajectory, H changes with position. The correct continuous output
sensitivity is H(t)Phi(t,t_0), where Phi_dot=A(t)Phi and Phi(t_0,t_0)=I. The gauge
satisfies g_dot=A(t)g, hence

```math
H(t)\Phi(t,t_0)g(\mathbf{x}(t_0))=H(t)g(\mathbf{x}(t))=0.
```

Equivalently, the first output-derivative Jacobian is H_dot+HA, not just HA.
Direct differentiation gives

```math
\dot\rho=-v\cos\beta,\qquad
\dot\beta=\frac v\rho\sin\beta-\omega,\qquad
\dot H+HA=MH,\quad
M=\begin{bmatrix}0&v\sin\beta\\-v\sin\beta/\rho^2&v\cos\beta/\rho\end{bmatrix}.
```

All further output derivatives depend only on (rho,beta) and known inputs/their
time derivatives. They cannot create a third independent pose direction.

The discrete trajectory test uses the actual EKF Jacobians:

```math
\Phi_{0,0}=I,\quad\Phi_{k,0}=F_{k-1}\cdots F_0,\qquad
\mathcal O_N=\begin{bmatrix}H_0\\H_1F_0\\H_2F_1F_0\\\vdots\end{bmatrix}.
```

The implemented map preserves F_k g(x_k)=g(x_(k+1)), so O_N g(x_0)=0. A valid
initial range/bearing observation already supplies rank two, and later observations
cannot remove this null direction. Under these assumptions the trajectory rank
remains two for stationary, rotating, straight, curved, reverse and varying inputs.
Motion can change conditioning and noise sensitivity without resolving the gauge.

## 4. Nonlinear symmetry and the role of the initial prior

For any constant angle phi, construct

```math
\mathbf p'=\mathbf m+R(\phi)(\mathbf p-\mathbf m),\qquad
\theta'=\operatorname{wrap}(\theta+\phi).
```

Then m-p'=R(phi)(m-p): range is unchanged and world bearing increases by phi,
which is cancelled by the heading change. Thus h(x';m)=h(x;m). Rotational
invariance of the motion equations also gives the transformed trajectory under
the **same** body inputs. Arbitrarily nearby, different initial poses therefore
have identical output histories. The nonlinear system is not locally fully
observable; the ambiguity is a continuous family, not merely a distant discrete
alternative. This argument is independent of floating-point rank tolerances.

Distinguish three claims:

1. The EKF stores three numbers: a representation, not an observability result.
2. Measurements distinguish states: the single-point likelihood cannot distinguish
   poses related by the rotation above.
3. A prior favours some states: known initial pose/heading can constrain that
   otherwise ambiguous family. This information comes from initialisation, not
   new single-point observations. Finite prior uncertainty need not disappear.

A finite positive-definite covariance does not refute the symmetry. Correlations
can change individual covariance entries, and relinearisation can introduce
spurious information; estimator consistency is a separate, untested question here.

## 5. What additional information would suffice?

These are mathematical proposals, not currently fused outputs:

- **Two distinct, correctly identified landmarks at known world coordinates:**
  stack H_1 and H_2. Each nullspace is span(g_i), with last component one.
  A common nonzero null vector would require g_1=g_2, implying m_1=m_2.
  Distinct points at nonzero ranges therefore give stacked rank three. Collinearity
  alone is not a degeneracy for identified range-plus-bearing measurements.
  Duplicate points add no rank; nearly coincident or distant geometry can be
  poorly conditioned. Unknown map coordinates or wrong associations are different
  problems. Non-simultaneous observations require proper trajectory propagation.
- **Independent world-heading measurement:** adding [0,0,1] removes the gauge
  since its product with g is one. Known absolute 2D position plus bearing also
  determines heading locally at nonzero range.
- **Oriented marker information:** with known marker world yaw psi_m and known
  camera/body yaw offset psi_bc, a valid relative-yaw observation has ideal form
  gamma=wrap(psi_m-theta-psi_bc), with Jacobian [0,0,-1]. Combined with range/bearing
  this can remove the ambiguity. Obtaining this from 3D PnP requires calibrated
  frames, known marker world orientation and resolved pose ambiguities; merely
  detecting a marker or configuring its side length does not establish it.

## 6. Camera/EKF contract and current limitations

[`get_range_and_bearing`](../sensing/aruco_camera.py#L160) returns camera-relative
XZ-plane range and left-positive bearing, using optical X right and Z forward.
The EKF h above instead refers to its robot pose origin and heading. Equality
requires justified co-location/alignment or an explicit extrinsic-aware measurement
model. Camera mounting pitch/height and planar reduction must be handled correctly.
Known rigid camera extrinsics alone do not remove the global rotation symmetry.

[`estimate_marker_pose`](../sensing/aruco_camera.py#L114) computes both translation
and rotation. But [`get_camera_measurement`](../sensing/aruco_camera.py#L242)
discards the marker ID and rotation vector, returning only range/bearing. The
current EKF fuses neither marker orientation nor an absolute heading observation.
Marker coordinates enter through the `marker_position` argument, not an established
map/ID association in the sensing helper. A fixed known point is therefore an
explicit mathematical assumption, not proof of the integrated sensing contract.

Prediction expects body v/omega. [`get_odometry`](turtlebot.py#L1) remains a
placeholder, and Part A demonstrates why requested speed cannot be substituted
for measured speed without accounting for plant error. No teammate code is changed
by this work. The analysis does not exploit marker appearance, visibility events,
wall geometry, collision constraints or LiDAR observations absent from the EKF h.

At zero range H is undefined; the existing implementation returns nonfinite entries
rather than rejecting the case. At the bearing cut, ordinary subtraction is wrong:
use a local angular chart/circular differences. Missed observations cannot supply
information, and noise, outliers or association errors may make practical
localisation substantially worse. None makes the single-point system observable.

## 7. Independent synthetic evidence and reproduction

[`tests/test_observability.py`](../tests/test_observability.py) contains 13 focused
unit tests. It imports the real EKF functions without modifying them or constructing
a new estimator. Tests cover the stated equations, analytical/numerical Jacobians,
frozen-rank contrast, continuous H_dot+HA, discrete gauge transport, finite nonlinear
symmetry, prior information, proposed extra outputs and geometric/angle degeneracies.
The zero-range test explicitly exposes the existing undefined Jacobian; passing
that test does not mean the production EKF safely handles zero range.

For trajectory checks: seven input sequences (stationary, rotation, forward,
reverse, left/right arcs and varying inputs), three initial poses, 80 steps of
0.05 s, and 81 observation instants per history. All test histories remain more
than 0.2 m from the landmark. Finite-symmetry checks additionally use four rotation
angles, giving 84 paired histories. These are deterministic synthetic mathematical
checks, not MuJoCo trials or physical/estimated-state navigation evidence.

Numerical conventions are explicit:

| Check | Tolerance / definition |
|---|---|
| Matrix rank | Count singular values sigma_i > max(1e-10, 1e-9 sigma_max) |
| Trajectory rank separation | Second singular value >1000 times cutoff; third <cutoff/100 |
| Centred state differences | Step 1e-6 in the stated metre/radian coordinates |
| Jacobian comparison | Absolute 2e-8, relative 2e-7 |
| Algebraic/gauge/equivariance residuals | Absolute 2e-11, zero target |

The SVD cutoff is a numerical convention for these SI-coordinate test cases, not
a physical observability threshold. Singular values depend on units/scaling and
geometry; exact rank is justified analytically above. Failing rank tests display
singular values and cutoff. Angular differences are wrapped before differentiation.
No well-conditioned covariance is used as proof, and no new dependencies or
benchmark outputs are introduced.

From the repository root, use the existing development environment (the clone has
no assumed local virtual environment):

```sh
PYTHON="$HOME/Downloads/Sensors-and-Controls-main/.venv/bin/python"
"$PYTHON" -B -m unittest discover -s tests -p test_observability.py -v
"$PYTHON" -B -m unittest discover -s tests -v
"$PYTHON" -B simulation/check_identification.py --transferred-layout
"$PYTHON" -B simulation/check_lqr_results.py --transferred-layout
"$PYTHON" -B simulation/check_terminal_results.py --transferred-layout
```

The historical checkers retain their original scope and saved hashes. The
transferred-layout option only recognises the acknowledged movement-script rename.
These mathematical tests establish structural limits of the implemented model;
they do not establish EKF convergence, covariance consistency, camera calibration,
robust localisation or complete assessed navigation.
