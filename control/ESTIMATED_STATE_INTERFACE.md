# Estimated-state interface and expiring actuator gateway

This extension validates estimated state and dispatches bounded, expiring wheel
commands around the **unchanged** LQR/direct/staged supervisors. It adds no
controller, planner, state estimator, collision algorithm or dependency. It is
not an end-to-end EKF/MuJoCo integration or a verified hardware watchdog.

## Files and data flow

- `estimated_pose_interface.py`: immutable snapshots, explicit policies, terminal
  clearance and a serial exception boundary around the existing supervisor.
- `command_gateway.py`: single-actuator-owner state machine, acknowledged writes,
  session generations, absolute command deadlines and an independent owner loop.
- `../tests/test_estimated_pose_interface.py`: fake clocks, immutable mock estimates,
  real existing supervisors and deliberately failing supervisor/provider variants.
- `../tests/test_command_gateway.py`: gateway logic and independent-process,
  fake-actuator demonstrations. No scientific benchmark outputs are regenerated.

```text
atomic EKF pose/P + timestamps/validity
                 |
     validate frame, freshness, quality and control slot
                 |
     check explicit terminal clearance, if applicable
                 |
       existing trajectory/terminal supervisor
                 |
       validate finite body/wheel output and phase
                 |
         expiring wheel-command lease
                 |
  independently scheduled, SINGLE actuator-owner gateway --> backend write
```

A successful write means the backend callable returned. It does not prove actual
wheel response, braking, traction, zero physical velocity or completed docking.
A writer must raise on failure; a backend that silently drops a write cannot be
verified here. All statuses are observations at receipt time, not permanent claims.

## Pose snapshot contract

`PoseEstimate` contains `pose`, `covariance`, `state_time`, `sample_monotonic`,
`sequence`, `epoch`, `frame_id`, `body_frame_id`, `localisation_valid` and a reason.

- Pose is `[x,y,theta]`, shape `(3,)`, in metres/radians, with anticlockwise-positive
  heading. The agreed world/map frame is shared by references and landmark maps.
- Body origin is the model's `base` origin (planar coordinates at the wheel-axis
  midpoint's ground projection), not the camera or offset centre of mass.
- Covariance is `(3,3)` in those same coordinates, including cross-covariances.
- `state_time` is the MODEL time represented by the estimate, not image receipt
  time. The producer must incorporate measured-wheel prediction to that time.
- `sample_monotonic` identifies that state's sampling time in the RECEIVER'S
  monotonic wall-clock domain. Remote clocks require an explicit trusted mapping.
  Arrival time must not be used to disguise an old state as fresh.
- Sequence is a nonnegative integer increasing for every accepted estimate;
  epoch identifies the estimator/map/time session and changes on reset.
- Validity is an explicit producer assertion, not inferred from finite numbers.

Snapshot arrays are detached bytes-backed read-only copies; writeability cannot
be re-enabled. Snapshot construction does not make a concurrent EKF state/P read
atomic: the producer must copy both under its own lock or publication mechanism.
`step()` can accept an estimate-provider callable so construction/publication
exceptions are caught inside the disarm boundary. Constructing a snapshot outside
that boundary is the producer's responsibility; arbitrary failures before calling
this API cannot be caught magically.

The consumer cannot detect a producer that dishonestly republishes old state with
new timestamps and sequence. Frame labels likewise cannot prove correct extrinsics.
The currently documented single-point-landmark observability limitation remains.
A three-state estimate or well-conditioned covariance is not proof of localisation.

## Explicit policies, not guessed deployment thresholds

`InterfacePolicy` requires named frames, `max_age_s`, `command_ttl_s`, a
`clock_grid_tolerance_s` and a callable `quality_check(previous, current)`.
No physical freshness, uncertainty or jump threshold is provided by default.
The established controller interval is 0.05 s. Producer state times must lie on
`state_time_origin + accepted_slot * sample_time`; tolerance must distinguish
adjacent slots and should represent only justified clock/grid numerical error.
Arrival jitter is handled through freshness, not extra controller calls.

Fresh identical coordinates are allowed. Duplicate sequence, duplicate sample
clock, duplicate/out-of-order state time, missed slots, future/stale estimates,
wrong frame/epoch, malformed/nonfinite arrays or false localisation validity
cause disarm. Commands do not queue old samples or extrapolate missing estimates.

Covariance validation scales coordinates by configurable `(metre,metre,radian)`
reference scales (default `(1,1,1)`), checks finite values, near symmetry and
positive semidefiniteness. Numerical tolerance is `64*machine_epsilon` times
`max(1, spectral_norm(normalised_covariance))`; it is a round-off allowance, not
a permitted localisation uncertainty. Validation never repairs the EKF covariance.

The required quality callback must contain the team's justified uncertainty,
pose-jump and relocalisation decisions; it may inspect previous accepted state.
Its exceptions or false/nonboolean result fail closed. Without a valid policy,
the interface cannot be constructed. It must not be deployed with an unexplained
`always True` substitute. Hardware/task bounds and calibrated estimator evidence
must justify physical thresholds. The tests' variance/jump limits are synthetic
fixtures only, not a calibrated Mahalanobis/chi-square gate. Consecutive EKF states
are correlated: summing their covariances is not automatically a valid innovation
covariance. Angle differences in such policies must wrap through +/-pi.

The two clocks have separate roles. State/model time drives prediction/reference
and the supervisor's fixed-period sample counts; wall monotonic time bounds
command lifetime even if simulation time pauses. Producer timestamps must describe
actual state propagation; the adapter cannot infer this from coordinates alone.
For hardware, the publisher must maintain the agreed real-time sampling contract.

## Arm, step, disarm and completion

`EstimatedPoseInterface` is a single-caller API. Construct it with a gateway/client,
a supervisor factory, policy, immutable goal, task ID, terminal handoff time and
matching `TerminalParameters`. The factory supplies an existing supervisor, not a
new control law. On explicit `arm(epoch=..., state_time_origin=...)`:

1. Attempt and acknowledge zero through the gateway.
2. Construct a fresh supervisor. Reusing a still-referenced previously used
   instance is rejected. Validate goal, handoff time, sample time and parameters.
3. Explicitly arm a new gateway generation with another acknowledged zero write.

Failed arming stays disarmed. A new arm is not authorisation to resume an old path
from an arbitrary state: the caller must agree a valid new reference/time origin
and manoeuvre permission. There is no automatic resume after interruption.

Each `step(estimate_or_provider, reference_provider, clearance=...)`:

1. Checks clock and current gateway generation/armed status.
2. Validates a fresh snapshot for the next model-time slot.
3. Checks terminal clearance BEFORE invoking terminal/staging, when applicable.
4. Generates and validates the reference inside the exception boundary.
5. Calls the unchanged supervisor once, then validates mode/phase, finite shapes,
   wheel limit and body/wheel consistency using the existing 0.033/0.160 geometry.
6. Rechecks freshness/clearance after potentially slow providers/controller.
7. Submits the lease and requires acceptance from the owner.

A rejected submission never silently re-arms the gateway. Any validation/provider/
controller exception, controller fault mode, invalid output or lost gateway session
calls `disarm()`, discards the supervisor, clears completion and attempts an ACTUAL
zero write. Malformed body/wheel output is rejected, not clipped into plausible
performance. The gateway enforces +/-6.67 rad/s; common saturation remains the
existing supervisor's job. No existing law, rate, physical parameter or evidence
manifest is changed.

Disarming discards the instance containing quiet dwell; re-arm creates a fresh
instance. Duplicate samples therefore cannot invoke the controller or advance
its counters. A missed/invalid interval cannot later resume accumulated dwell.
No catch-up burst is used. The model-time grid prevents sample-count drift across
accepted intervals. Supplier clock integrity remains part of the contract.

`estimated_stop_qualified` is true only for an accepted `stop` output at the goal's
existing outer position/heading tolerances, with zero wheels and valid input/
permission. It means the existing controller qualified estimated rest under these
assumptions, not physical docking. Valid new estimates continue to be checked;
leaving the goal bounds after stop revokes completion and holds zero, without
starting recovery motion. Loss of validity also revokes completion on the next
step. The interface's cached armed/completion flags do not override an independently
expired gateway; consumers must not treat an old result as a current safety grant.

If zero dispatch returns failure or transport is unavailable, the result says
`stop_unconfirmed`; `zero_dispatch_confirmed` is false. A successful zero receipt
still confirms only the backend write. Callback absence cannot execute this
exception boundary, which is why the independent gateway is necessary.

## Clearance permission

`ClearanceGrant` binds a grant ID to task, world frame and estimator epoch, with
monotonic validity interval, explicit approval, revocation status and an explicit
`covers_terminal_manoeuvre` attestation. Missing/unknown, expired, revoked,
wrong-task/frame/epoch or incomplete grants prevent terminal invocation and
nonzero dispatch. There is no silent direct-controller fallback.

For this minimal gate ALL terminal manoeuvres require approval, avoiding a second
copy of the staging-trigger logic. The issuer must check the whole swept manoeuvre:
staging, rotations, approaches/re-approaches, robot footprint and uncertainty.
This boolean attestation is not a geometric collision proof performed by this code.
Changing the task, goal/map context or estimator reset requires corresponding
re-authorisation; use a task ID that uniquely binds the fixed goal/map revision.
The current closest-obstacle LiDAR helper cannot supply this permission by itself.

Newly revoked permission supplied at the next step disarms before controller use.
If callbacks are stalled, the authority must signal the owner's `inhibit_event`
(or an equivalent lower-level stop path) directly. The owner checks that event
independently; clearing it never re-arms. A previously copied grant cannot reveal
a revocation never communicated to the consumer. Without that external signal,
only the existing lease/grant expiry bounds stale permission. Tracking-prefix
obstacle protection likewise remains the integration authority's responsibility.

## Wheel leases, ownership and independent expiry

`CommandGateway` is the synchronous single-owner state machine. `WheelCommandLease`
carries generation, increasing command ID, wheel tuple, creation time and absolute
monotonic expiry. The maximum lease duration is required owner configuration.
The interface bounds expiry by the earliest of command TTL, estimate freshness
and terminal-clearance expiry. Delivery never restarts a command's lifetime.

Expiry or disarm invalidates the generation and outstanding nonzero commands,
then attempts zero. Late old-generation packets and superseded/outdated packets
are rejected without renewing a lease or interrupting a newly armed generation.
Malformed current-generation commands disarm and attempt zero. A motion-write
failure also triggers zero; zero failure is explicitly recorded. Both startup
and every explicit arm attempt zero first. Arming alone leaves zero applied.

`GatewayStatus` reports acceptance, generation, armed state, reason, last applied
wheel tuple (unknown after write failure) and whether the last write was a
successful zero. A successful zero lease can remain armed; it is not measured
rest. Receipt truth depends on the backend raising errors reliably.

`run_owner_loop(connection, writer_factory, ...)` constructs the writer INSIDE the
owner and polls expiry while waiting for IPC. It must run independently of EKF/
controller callbacks. `GatewayClient` provides serial request/receipt methods;
request IDs reject mismatched replies. Timeout/transport error closes the channel
permanently so late receipts cannot be mistaken for acknowledged stops. EOF/close
attempt zero in the owner. After communication loss, reconnect/reinitialise
explicitly, never reuse a broken client or re-arm implicitly.

The demonstration uses `multiprocessing` with the `spawn` context and fake writers.
After acknowledging motion, the parent sends no further controller/poll requests;
the child expires the lease and records zero on its own clock. A second process
test signals external inhibit without a controller callback. Generous test waits
are only test timeouts, not real-time latency claims.

A real MuJoCo integration must assign ONE owner to actuator writes and coordinate
ALL access to its `MjData`. Do not pass live `MjData` between processes or create
several writers. Typically the simulation/actuator worker owns model/data and runs
stepping plus the deadline check, while the controller communicates through IPC.
The provided generic loop does not implement that team-specific stepping backend.
It is not safe to add a writer process while another process owns the live data.

Calling `poll()` from the controller callback alone is NOT a watchdog. Independent
Python scheduling is also not hard real time: expiry latency includes scheduling,
poll interval and backend latency. A blocked writer/owner or host death can defeat
host-level stopping; hardware/backend command timeout is then necessary. The
independent worker must remain responsive and backend writes must be bounded.
Process-start failure, closed transport or unresponsive backend does not justify
a physical-stop claim. Emergency inhibit itself is subject to these limitations.

## Team dependencies and smallest integration step

- EKF owner: atomic state/P publication, trustworthy epoch/sequence/timestamps,
  validity/relocalisation metadata and justified uncertainty/jump policy. The current
  estimator has state/P but no such metadata; no estimator changes are made here.
- Odometry owner: actual measured wheel velocities with correct signs/units and
  acquisition time; the current `simulation/turtlebot.py` remains a placeholder.
- Camera owner: known-ID association and world landmark coordinates, camera/body
  extrinsics, timestamps and observation validity. Current helper discards marker
  ID/orientation at its top-level output; the EKF does not fuse marker orientation.
- LiDAR/integration owner: clearance authority covering full swept area and a live
  revoke/inhibit path. An empty scan or `closest_alert=False` is not approval.
- Integration owner: one actuator/simulation owner, clocks, control scheduling,
  bounded IPC/write behaviour, explicit recovery and lower-level stopping policy.

The next small step is a separate integration harness that publishes actual EKF
snapshots from measured-wheel prediction and associated camera updates, and wires
the gateway into the ONE MuJoCo actuator/stepping owner. First verify receipts and
fault/expiry behaviour there; then compare estimated-state tracking against the
retained ideal-state diagnostics in a NEW evidence directory. No simulator-ground-
truth fallback is permitted as assessed control feedback. No navigation, ArUco
docking, obstacle avoidance or production safety success is established here.

## Reproduction and verification scope

From the cloned repository root, using the existing development environment:

```sh
PYTHON="$HOME/Downloads/Sensors-and-Controls-main/.venv/bin/python"
"$PYTHON" -B -m unittest discover -s tests -v
"$PYTHON" -B simulation/check_identification.py --transferred-layout
"$PYTHON" -B simulation/check_lqr_results.py --transferred-layout
"$PYTHON" -B simulation/check_terminal_results.py --transferred-layout
git diff --check
```

The new tests use explicit synthetic limits, mock streams and fake actuators, not
calibrated deployment thresholds. They cover immutable publication, fresh identical
coordinates, duplicate/out-of-order/stale/wrong-slot estimates, frame/epoch/covariance
validation, wrapped heading jumps, quality rejection, provider/controller failure,
partial supervisor mutation, invalid outputs, discarded dwell, explicit re-arm,
failed zero writes, lost transport, permission denial/revocation/expiry, post-stop
disturbance, expiry races, old commands and independent-process expiry/inhibit.
Historical test files, benchmark outputs, manifests and pinned documents remain
unchanged. Existing historical evidence checkers retain their original guarantees;
no benchmark evidence is regenerated by these additions.
