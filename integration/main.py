"""Team navigation wiring with independent expiry and conservative motion gating.

Run `python -B -m integration.main`. Default: no justified quality policy, NO
motion. Rendering and estimation can be diagnosed but are not localisation proof.
Programmatic run(criteria=...) requires experiment-specific justified criteria;
there is no CLI switch that silently substitutes an always-true policy.

MuJoCo access belongs exclusively to OwnerSession's spawned process. Camera/PnP,
EKF and existing supervisor remain in the receiver. No terminal clearance authority
exists: trajectory completion disarms BEFORE terminal invocation. Existing standalone
tracking/docking benchmarks are separate and unchanged. No evidence files are saved.
"""
from dataclasses import replace
from pathlib import Path
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from estimation.ekf import TurtleBotEKF, wheel_odometry, wrap
from sensing.aruco_camera import (configure_marker_size, detect_aruco_marker,
                                 estimate_marker_pose, get_range_and_bearing)
from sensing.lidar import get_front_obstacle
from control.estimated_pose_interface import PoseEstimate, InterfacePolicy, EstimatedPoseInterface
from control.terminal_controller import TerminalParameters
from control.staged_terminal_controller import StagedTrajectoryTerminalController
from control.trajectory import reference, DURATIONS
from integration.mujoco_owner import OwnerConfig, OwnerSession, Measurement
from integration.localisation_policy import LocalisationPolicy

DT = .05
TRAJECTORY_NAME = 'straight'
DURATION = DURATIONS[TRAJECTORY_NAME]
WORLD_FRAME, BODY_FRAME = 'map', 'base'
TASK_ID = 'straight_tracking_test'
# Retained diagnostic timing settings, NOT calibrated hardware safety limits.
MAX_AGE_S, COMMAND_TTL_S, MAX_LEASE_S, GRID_TOLERANCE_S = .50, .20, .25, 1e-6


def camera_observation(packet, metadata):
    """Same team detector/PnP; camera-origin data with explicitly associated ID."""
    if packet.rgb is None:
        return None
    detection = detect_aruco_marker(packet.rgb)
    if detection is None:
        return None
    marker_id, corners = detection
    if marker_id not in metadata['landmarks']:
        raise ValueError('Unknown marker ID')
    configure_marker_size(metadata['marker_sizes'][marker_id])
    translation, _ = estimate_marker_pose(corners, np.asarray(metadata['camera_matrix']), np.zeros(5))
    measurement = get_range_and_bearing(translation)
    if measurement.shape != (2,1) or not np.isfinite(measurement).all() or measurement[0,0] <= 0:
        raise ValueError('Invalid camera measurement')
    return marker_id, measurement, np.asarray(metadata['landmarks'][marker_id])


def reference_for_obstacle(pose, relative_time, obstacle):
    """Preserve the team's diagnostic detour heuristic; NOT swept-path planning."""
    normal = reference(TRAJECTORY_NAME, relative_time)
    if obstacle is None or obstacle[0] >= .4:
        return normal
    theta = pose[2]
    forward = np.array([np.cos(theta), np.sin(theta)])
    left = np.array([-np.sin(theta), np.cos(theta)])
    sign = -1 if obstacle[1] > 0 else 1
    xy = pose[:2]+.25*forward+sign*.20*left
    heading = wrap(theta+sign*np.deg2rad(20))
    return type(normal)(pose=np.r_[xy,heading], input=np.array([.05,0.]))


def dispatch(interface, snapshot, reference_provider, *, relative_time):
    """No grant is manufactured. Stop before the scheduled terminal handoff."""
    if relative_time >= DURATION:
        return interface.disarm('trajectory_complete; terminal_clearance_unavailable')
    if snapshot.localisation_valid is not True:
        return interface.disarm('localisation_invalid: '+snapshot.validity_reason)
    return interface.step(snapshot, reference_provider, clearance=None)


def run(*, criteria=None, render_camera=True, max_samples=None):
    """One explicit arm attempt; every rejected sample terminates this session.

    Q/R/prior below retain team diagnostic settings, not calibrated uncertainty.
    Strict 20 Hz slots fail closed on delay/overflow; no catch-up/relabelled samples.
    """
    if max_samples is None: max_samples = round(DURATION/DT)+1
    if type(max_samples) is not int or max_samples < 1:
        raise ValueError('Positive sample count required')
    epoch = 'navigation-'+str(time.monotonic_ns())
    config = OwnerConfig(sample_time=DT, max_lease_s=MAX_LEASE_S,
                         render_camera=render_camera, epoch=epoch,
                         max_runtime_s=DURATION+10.)
    history, reasons, marker_counts = [], [], {0:0,1:0}
    final_stop = None
    with OwnerSession(config) as owner:
        metadata = owner.metadata
        # Static map metadata is configuration, never an observed robot truth pose.
        goal = np.array([metadata['landmarks'][0][0]-.30, metadata['landmarks'][0][1], 0.])
        parameters = TerminalParameters(sample_time=DT)
        quality = LocalisationPolicy(epoch,WORLD_FRAME,BODY_FRAME,DT,MAX_AGE_S,
                                      GRID_TOLERANCE_S,criteria)
        policy = InterfacePolicy(WORLD_FRAME,BODY_FRAME,MAX_AGE_S,COMMAND_TTL_S,
            GRID_TOLERANCE_S, criteria.accepts if criteria else lambda previous,current:False,
            sample_time=DT)
        interface = EstimatedPoseInterface(owner.client,
            lambda:StagedTrajectoryTerminalController(goal,DURATION,parameters),
            policy,goal=goal,task_id=TASK_ID,terminal_start_s=DURATION,
            parameters=parameters,clock=time.monotonic)
        ekf = TurtleBotEKF([0.,0.,0.], np.diag([.05**2,.05**2,np.deg2rad(5)**2]),
                          np.diag([.005**2,.005**2,np.deg2rad(.5)**2]),
                          np.diag([.03**2,np.deg2rad(1.5)**2]))
        previous = None
        last_correction = None
        # Prior represents simulation time zero, as in the original integration.
        last_state_time = 0.
        arm_attempted = False
        try:
            for index in range(max_samples):
                packet = owner.next_measurement()
                received = time.monotonic()
                if (not isinstance(packet,Measurement) or packet.wheels.shape != (2,)
                        or not np.isfinite(packet.wheels).all()
                        or packet.epoch != epoch or packet.frame_id != WORLD_FRAME
                        or packet.body_frame_id != BODY_FRAME or packet.sequence != index
                        or packet.lidar.ndim != 1 or np.isnan(packet.lidar).any()
                        or not np.isfinite(packet.sample_monotonic)
                        or packet.sample_monotonic > received
                        or not np.isfinite(packet.state_time)
                        or abs(packet.state_time-(index+1)*DT)>GRID_TOLERANCE_S
                        or packet.state_time <= last_state_time
                        or received >= packet.sample_monotonic+MAX_AGE_S):
                    raise ValueError('Invalid/stale owner packet or missed required interval')
                elapsed = packet.state_time-last_state_time
                ekf.predict(wheel_odometry(*packet.wheels),elapsed)
                last_state_time = packet.state_time
                camera_status = 'missing'
                try:
                    observation = camera_observation(packet,metadata)
                    if observation is not None:
                        marker_id,measurement,landmark = observation
                        ekf.update_camera(measurement,landmark,metadata['camera_offset'],metadata['camera_yaw'])
                        last_correction = packet.state_time
                        marker_counts[marker_id] += 1
                        camera_status = 'corrected'
                except Exception:
                    camera_status = 'failed'
                snapshot = PoseEstimate(ekf.state,ekf.P,packet.state_time,packet.sample_monotonic,
                    packet.sequence,packet.epoch,packet.frame_id,packet.body_frame_id,False)
                decision = quality.assess(snapshot,previous=previous,
                    expected_time=(index+1)*DT,now=time.monotonic(),camera_status=camera_status,
                    last_correction_state_time=last_correction)
                snapshot = replace(snapshot,localisation_valid=decision.valid,validity_reason=decision.reason)
                reasons.append(decision.reason)
                # Truth is fetched ONLY for scoring, never passed to policy/EKF/reference.
                evaluation = owner.evaluation(packet.sequence)
                obstacle = get_front_obstacle(packet.lidar)
                relative_time = index*DT
                row = dict(sequence=index,state_time=packet.state_time,
                    acquisition_monotonic=packet.sample_monotonic,estimate=snapshot.pose.copy(),
                    covariance=snapshot.covariance.copy(),valid=decision.valid,
                    camera_status=camera_status,obstacle=obstacle,
                    reference=reference(TRAJECTORY_NAME,relative_time).pose.copy(),
                    truth=None if evaluation is None else evaluation.pose)
                history.append(row)
                if snapshot.localisation_valid and not arm_attempted:
                    arm_attempted = True
                    armed = interface.arm(epoch=epoch,state_time_origin=packet.state_time)
                    if not armed.accepted:
                        reasons.append(armed.reason)
                        break
                result = dispatch(interface,snapshot,
                    lambda t:reference_for_obstacle(snapshot.pose,t,obstacle),relative_time=relative_time)
                row['mode'] = result.controller_mode
                row['reason'] = result.reason
                if not result.accepted:
                    break
                previous = snapshot
        except Exception as error:
            reasons.append(type(error).__name__+': '+str(error))
            interface.disarm(reasons[-1])
        finally:
            final_stop = interface.disarm('integration_complete; no_terminal_clearance')
        report = dict(history=history,reasons=reasons,marker_counts=marker_counts,
                      stop_confirmed=final_stop.zero_dispatch_confirmed,armed_once=arm_attempted,
                      terminal_clearance=False,owner_pid=metadata['owner_pid'])
    if owner.error:
        report['owner_error'] = owner.error
        report['stop_confirmed'] = False
    print('Diagnostic integration; terminal manoeuvres disabled.')
    print('Samples:',len(history),'arm attempted:',arm_attempted,'zero confirmed:',report['stop_confirmed'])
    print('Validity/rejection reasons:',reasons)
    scored = [row for row in history if row['truth'] is not None]
    if arm_attempted and len(scored)>1:
        truth=np.array([row['truth'] for row in scored]);estimate=np.array([row['estimate'] for row in scored])
        refs=np.array([row['reference'] for row in scored])
        print('Diagnostic localisation position RMSE:',float(np.sqrt(np.mean(np.sum((truth[:,:2]-estimate[:,:2])**2,axis=1)))))
        print('Diagnostic tracking position RMSE:',float(np.sqrt(np.mean(np.sum((truth[:,:2]-refs[:,:2])**2,axis=1)))))
    else:
        print('No autonomous tracking performance or docking claim.')
    return report


if __name__ == '__main__':
    run()
