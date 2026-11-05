"""Reproducible localisation ONLY; prescribed leases, NEVER LQR/validity authorisation.

python -B -m integration.tests.validate_localisation --suite --repeat 2
Default is one stationary run. --suite explicitly runs the fixed stationary,
straight, rotation and curve plan below. No files are written. --details prints
all in-memory observations/estimates; default prints metrics and geometry examples.

Plan is fixed before results: 20 Hz, nominal geometry .033/.160 m, .5 s zero
warmup, <=3 s .06 m/s/.15 rad/s prescribed command window, then .7 s expiry/deceleration tail. The last nonzero lease
may persist for up to its .20 wall s lifetime after submissions cease.
The stationary run is 6 s. Leases are .20 wall s, maximum .25 wall s. One explicit
arm per experiment; rejection/overflow/missed sequence aborts, NEVER auto-rearms.
No path planner, controller feedback, calibrated success threshold or terminal
manoeuvre is present. Run only in the supplied simulated scene. The prescribed
commands do not depend on estimated pose OR scoring truth.

Every frame is processed by 14 EKF variants: three declared priors x first/all
observations x dropout/no-dropout, plus correct-prior all-marker face-map variants.
Baseline geom-centre association reproduces production; face association is a
separate labelled static configuration experiment, not a production correction.
Dropout suppresses observations over [1.5,2.5) sim s for moving runs and [2,3) for
stationary runs. Noise/prior are the existing diagnostic settings, NOT calibrated.
Q is added once per measured .05 s interval. No synthetic sensor noise is added.

Detector: unchanged OpenCV dictionary/parameters, retaining ALL detections instead
of the team's first-only wrapper. PnP is the existing team function. The raw-image
first ID remains the first-only baseline; sequential multi-marker update order is
recorded. Deterministic repeats are reproducibility checks, not Monte Carlo.

Truth separation: FilterBank.advance accepts Measurement + measured Observation
only. Scoring accepts Evaluation only AFTER predictions/corrections complete.
Expected translations use full optical R_WC and position from Evaluation. Static
face centres are parsed from scene XML (known map config), not observed robot pose.
Neither expected translation nor errors enter EKF, quality or command generation.

Limitations: only fixed-resolution/current-map tests; visible NEGATIVE-X box faces
are assumed and checked in scoring. Roll/pitch fully retained for camera geometry;
EKF remains the unchanged planar model. Integer corners, white/black texture extent,
range-dependent rasterisation and fixed R can bias results despite tiny reprojection
error. NIS/NEES/2sigma coverage are descriptive for correlated deterministic data.
No hardware localisation, statistical consistency, safe docking or LQR readiness
is certified by this harness.
"""
from dataclasses import dataclass
import argparse
import hashlib
import json
from pathlib import Path
import queue
import subprocess
import time
import xml.etree.ElementTree as ET

import cv2
import numpy as np

from control.command_gateway import WheelCommandLease, WHEEL_LIMIT
from control.lqr_controller import wheel_commands
from estimation.ekf import TurtleBotEKF, wheel_odometry, camera_model, camera_jacobian, wrap
from integration.mujoco_owner import OwnerSession, OwnerConfig, Measurement, Evaluation, SCENE
from sensing.aruco_camera import (ARUCO_DICTIONARY, configure_marker_size,
                                 estimate_marker_pose, get_range_and_bearing)

ROOT = Path(__file__).resolve().parents[2]
DT = .05
PRIORS = ((0.,0.,0.),(.05,-.04,.1),(-.05,.04,-.1))
P0 = np.diag([.05**2,.05**2,np.deg2rad(5.)**2])
Q = np.diag([.005**2,.005**2,np.deg2rad(.5)**2])
R = np.diag([.03**2,np.deg2rad(1.5)**2])


@dataclass(frozen=True)
class Plan:
    name: str
    duration: float
    velocity: float
    yaw_rate: float
    motion_seconds: float

    @property
    def dropout(self): return (2.,3.) if self.name=='stationary' else (1.5,2.5)

    def wheels(self, state_time):
        if not np.isfinite(state_time): raise ValueError('Finite command time required')
        slot = round(state_time/DT)
        active = round(.5/DT) <= slot < round((.5+self.motion_seconds)/DT)
        wheels,_,saturated = wheel_commands(self.velocity if active else 0.,
                                            self.yaw_rate if active else 0.)
        if saturated or max(abs(wheels)) > WHEEL_LIMIT:
            raise ValueError('Prescribed plan exceeds wheel limits')
        return tuple(float(v) for v in wheels)

    @property
    def last_submission(self): return self.duration-.7


PLANS = {
    'stationary': Plan('stationary',6.,0.,0.,0.),
    'straight': Plan('straight',4.2,.06,0.,3.),
    'rotation': Plan('rotation',3.2,0.,.15,2.),
    'curve': Plan('curve',4.2,.06,.15,3.),
}


def rotation(quaternion):
    q=np.asarray(quaternion,dtype=float)
    if q.shape!=(4,) or not np.isfinite(q).all() or np.linalg.norm(q)==0:
        raise ValueError('Finite wxyz quaternion required')
    w,x,y,z=q/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],
                     [2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],
                     [2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])


def marker_faces(scene=SCENE):
    """Known static negative-X face centres; XML only, no authoritative MjData."""
    targets={'aruco_marker':0,'aruco_marker_1':1}
    result={}
    def walk(node,position,orientation):
        for body in node.findall('body'):
            if any(k in body.attrib for k in ('euler','axisangle','xyaxes','zaxis')):
                raise ValueError('Unsupported static body orientation encoding')
            local=np.fromstring(body.get('pos','0 0 0'),sep=' ')
            rot=rotation(np.fromstring(body.get('quat','1 0 0 0'),sep=' '))
            walk(body,position+orientation@local,orientation@rot)
        for geom in node.findall('geom'):
            if geom.get('name') not in targets:continue
            if any(k in geom.attrib for k in ('euler','axisangle','xyaxes','zaxis')):
                raise ValueError('Unsupported marker orientation encoding')
            if geom.get('type')!='box':raise ValueError('Expected box marker')
            p=np.fromstring(geom.get('pos','0 0 0'),sep=' ')
            size=np.fromstring(geom.attrib['size'],sep=' ')
            rot=orientation@rotation(np.fromstring(geom.get('quat','1 0 0 0'),sep=' '))
            centre=position+orientation@p
            if size.shape!=(3,) or not np.all(size>0) or not np.isclose(size[1],size[2]):
                raise ValueError('Expected square YZ marker face')
            result[targets[geom.attrib['name']]]=dict(centre=centre,
                face=centre+rot@[-size[0],0,0],normal=rot@[-1.,0,0],side=2*size[1])
    xml=ET.parse(scene).getroot();walk(xml.find('worldbody'),np.zeros(3),np.eye(3))
    if set(result)!={0,1}:raise ValueError('Both static landmarks required')
    return result


@dataclass(frozen=True)
class Observation:
    marker_id: int
    translation: tuple
    bearing_range: tuple  # range,bearing; explicit name kept separate from truth
    corners: tuple
    reprojection_rmse_px: float
    pixel_side: float
    sequence: int
    state_time: float
    acquisition_monotonic: float
    epoch: str


def detect(packet,metadata):
    """Measured data + configured calibration only; never Evaluation."""
    if not isinstance(packet,Measurement):raise ValueError('Measurement required')
    if packet.rgb is None:return [],[]
    detector=cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(ARUCO_DICTIONARY),
                                   cv2.aruco.DetectorParameters())
    corners,ids,_=detector.detectMarkers(cv2.cvtColor(packet.rgb,cv2.COLOR_RGB2GRAY))
    observations=[];rejected=[]
    if ids is None:return observations,rejected
    if len(set(int(v) for v in ids.ravel())) != len(ids):
        raise ValueError('Ambiguous duplicate marker ID')
    for identity,c in zip(ids.ravel(),corners):
        identity=int(identity)
        if identity not in metadata['landmarks']:
            rejected.append(identity);continue
        size=metadata['marker_sizes'][identity];configure_marker_size(size)
        k=np.asarray(metadata['camera_matrix'])
        tvec,rvec=estimate_marker_pose(c,k,np.zeros(5))
        measured=get_range_and_bearing(tvec).reshape(2)
        if not np.isfinite(measured).all() or measured[0]<=0:
            raise ValueError('Invalid measured range/bearing')
        h=size/2
        obj=np.array([[-h,h,0],[h,h,0],[h,-h,0],[-h,-h,0]],dtype=np.float32)
        projected,_=cv2.projectPoints(obj,rvec,tvec,k,np.zeros(5))
        c=np.asarray(c).reshape(4,2)
        error=float(np.sqrt(np.mean(np.sum((projected.reshape(4,2)-c)**2,axis=1))))
        pixels=float(np.mean(np.linalg.norm(np.roll(c,-1,axis=0)-c,axis=1)))
        observations.append(Observation(identity,tuple(tvec.ravel()),tuple(measured),
            tuple(tuple(float(v) for v in row) for row in c),error,pixels,packet.sequence,
            packet.state_time,packet.sample_monotonic,packet.epoch))
    return observations,rejected


def innovation_diagnostics(ekf, measurement, landmark, camera_offset, camera_yaw):
    """Pre-update diagnostics only; no gating and no filter mutation.

    NIS is undefined for missing, nonfinite, asymmetric or non-positive-definite S.
    A failed diagnostic does not silently repair covariance or reject a correction.
    """
    z=np.asarray(measurement,float).reshape(2)
    expected=camera_model(ekf.state,landmark,camera_offset,camera_yaw)
    h=camera_jacobian(ekf.state,landmark,camera_offset,camera_yaw)
    residual=z-expected;residual[1]=wrap(residual[1])
    covariance=None;nis=None;status='invalid_covariance'
    try:
        covariance=h@np.asarray(ekf.P,float)@h.T+np.asarray(ekf.R,float)
        if (covariance.shape==(2,2) and np.isfinite(covariance).all()
                and np.isfinite(residual).all()
                and np.allclose(covariance,covariance.T,rtol=0,atol=1e-12)):
            chol=np.linalg.cholesky(covariance)
            whitened=np.linalg.solve(chol,residual)
            nis=float(whitened@whitened);status='valid'
    except (ValueError,TypeError,np.linalg.LinAlgError):pass
    return dict(innovation=residual.tolist(),innovation_covariance=None if covariance is None else covariance.tolist(),
                nis=nis,nis_status=status,units=['m','rad'])


class FilterBank:
    """Feedback boundary: no Evaluation parameter or access to owner/scoring channel."""
    def __init__(self,metadata,faces):
        self.metadata=metadata
        self.variants={}
        for i,prior in enumerate(PRIORS):
            for mode in ('first','all'):
                for dropout in (False,True):
                    name=f'prior{i}_{mode}_drop{int(dropout)}_geom'
                    self.variants[name]=(TurtleBotEKF(prior,P0,Q,R),mode,dropout,'geom')
        for dropout in (False,True):
            self.variants[f'prior0_all_drop{int(dropout)}_face']=(
                TurtleBotEKF(PRIORS[0],P0,Q,R),'all',dropout,'face')
        self.maps={'geom':{i:np.asarray(p) for i,p in metadata['landmarks'].items()},
                   'face':{i:np.asarray(g['face'][:2]) for i,g in faces.items()}}
        self.previous=None
        self.control_history=[]

    def advance(self,packet,observations,dropout_window):
        if not isinstance(packet,Measurement):raise ValueError('Measurement required')
        k=0 if self.previous is None else self.previous.sequence+1
        elapsed=packet.state_time-(0. if self.previous is None else self.previous.state_time)
        if (packet.sequence!=k or not np.isfinite(elapsed) or abs(elapsed-DT)>1e-6
                or not np.isfinite(packet.sample_monotonic)
                or packet.frame_id!='map' or packet.body_frame_id!='base'
                or packet.epoch!=self.metadata['epoch']
                or packet.wheels.shape!=(2,) or not np.isfinite(packet.wheels).all()
                or self.previous is not None and packet.sample_monotonic<=self.previous.sample_monotonic):
            raise ValueError('Invalid measured wheel sample/interval/provenance')
        if len({obs.marker_id for obs in observations}) != len(observations):
            raise ValueError('Duplicate marker ID')
        for obs in observations:
            if (not isinstance(obs,Observation) or obs.marker_id not in self.metadata['landmarks']
                    or obs.sequence!=packet.sequence or obs.epoch!=packet.epoch
                    or obs.state_time!=packet.state_time
                    or obs.acquisition_monotonic!=packet.sample_monotonic):
                raise ValueError('Unassociated or mismatched observation')
        control=wheel_odometry(*packet.wheels)
        self.control_history.append(tuple(control))
        result={}
        for name,(ekf,mode,dropout,map_mode) in self.variants.items():
            ekf.predict(control,elapsed)
            predicted=ekf.state.copy()
            selected=observations[:1] if mode=='first' else observations
            if dropout and round(dropout_window[0]/DT)<=packet.sequence+1<round(dropout_window[1]/DT):selected=[]
            nises=[];ids=[];updates=[]
            selected_ids={o.marker_id for o in selected}
            for obs in observations:
                if obs.marker_id not in selected_ids:
                    updates.append(dict(marker_id=obs.marker_id,status="dropout" if dropout and round(dropout_window[0]/DT)<=packet.sequence+1<round(dropout_window[1]/DT) else "not_selected",
                        acquisition_monotonic=obs.acquisition_monotonic,sequence=obs.sequence,epoch=obs.epoch))
            for obs in selected:
                z=np.asarray(obs.bearing_range);landmark=self.maps[map_mode][obs.marker_id]
                diagnostic=innovation_diagnostics(ekf,z,landmark,self.metadata['camera_offset'],self.metadata['camera_yaw'])
                started=time.monotonic()
                try:
                    ekf.update_camera(z,landmark,self.metadata['camera_offset'],self.metadata['camera_yaw'])
                except Exception as exc:
                    updates.append(dict(marker_id=obs.marker_id,status='failed',error=type(exc).__name__,**diagnostic))
                    raise
                updates.append(dict(marker_id=obs.marker_id,status='corrected',
                    acquisition_monotonic=obs.acquisition_monotonic,sequence=obs.sequence,epoch=obs.epoch,
                    processing_started_monotonic=started,processing_finished_monotonic=time.monotonic(),**diagnostic))
                if diagnostic['nis'] is not None:nises.append(diagnostic['nis'])
                ids.append(obs.marker_id)
            if not np.isfinite(ekf.state).all() or not np.isfinite(ekf.P).all():
                raise ValueError('EKF numerical failure')
            result[name]=dict(pose=ekf.state.tolist(),predicted_pose=predicted.tolist(),
                              covariance=ekf.P.tolist(),correction_ids=ids,nis=nises,updates=updates,
                              camera_available=bool(observations))
        self.previous=packet
        return result


def expected_translation(evaluation,world_point):
    """Full scoring transform; optical coordinates x right,y down,z forward."""
    if not isinstance(evaluation,Evaluation):raise ValueError('Evaluation required')
    p=np.asarray(evaluation.camera_world_position,dtype=float)
    r=np.asarray(evaluation.camera_to_world_rotation,dtype=float)
    if (p.shape!=(3,) or r.shape!=(3,3) or not np.isfinite(p).all() or not np.isfinite(r).all()
            or not np.allclose(r.T@r,np.eye(3),rtol=0,atol=1e-10)
            or not np.isclose(np.linalg.det(r),1,rtol=0,atol=1e-10)):
        raise ValueError('Valid full camera transform required')
    return r.T@(np.asarray(world_point)-p)


def score_geometry(observations,evaluation,faces):
    result=[]
    for obs in observations:
        if (obs.sequence!=evaluation.sequence or obs.epoch!=evaluation.epoch
                or obs.state_time!=evaluation.state_time
                or obs.acquisition_monotonic!=evaluation.sample_monotonic):
            raise ValueError('Scoring truth/observation provenance mismatch')
        geometry=faces[obs.marker_id]
        expected=expected_translation(evaluation,geometry['face'])
        if expected[2]<=0 or np.dot(geometry['normal'],np.asarray(evaluation.camera_world_position)-geometry['face'])<=0:
            raise ValueError('Negative-X face assumption not valid for this view')
        centre=expected_translation(evaluation,geometry['centre'])
        er=get_range_and_bearing(expected).reshape(2)
        ec=get_range_and_bearing(centre).reshape(2)
        measured=np.asarray(obs.bearing_range)
        incidence=float(np.rad2deg(np.arccos(np.clip(np.dot(geometry['normal'],
            (np.asarray(evaluation.camera_world_position)-geometry['face'])/
            np.linalg.norm(np.asarray(evaluation.camera_world_position)-geometry['face'])),-1,1))))
        result.append(dict(marker_id=obs.marker_id,sequence=obs.sequence,state_time=obs.state_time,
            epoch=obs.epoch,acquisition_monotonic=obs.acquisition_monotonic,
            translation=list(obs.translation),expected_translation=expected.tolist(),
            corners=obs.corners,reprojection_rmse_px=obs.reprojection_rmse_px,
            pixel_side=obs.pixel_side,incidence_deg=incidence,
            face_world=geometry['face'].tolist(),expected_range=float(er[0]),
            measured_range=float(measured[0]),expected_bearing=float(er[1]),
            measured_bearing=float(measured[1]),range_error_face=float(measured[0]-er[0]),
            range_error_centre=float(measured[0]-ec[0]),
            bearing_error=wrap(float(measured[1]-er[1]))))
    return result


def metric(errors):
    a=np.asarray(errors,dtype=float)
    if a.ndim!=2 or a.shape[1]!=3 or len(a)==0:raise ValueError('Nonempty Nx3 errors required')
    a=a.copy();a[:,2]=wrap(a[:,2])
    return dict(position_rmse_m=float(np.sqrt(np.mean(np.sum(a[:,:2]**2,axis=1)))),
                heading_rmse_deg=float(np.rad2deg(np.sqrt(np.mean(a[:,2]**2)))),
                position_max_m=float(max(np.linalg.norm(a[:,:2],axis=1))),
                heading_max_deg=float(np.rad2deg(max(abs(a[:,2])))))


def summarise(rows,geometry):
    variants={}
    for name in rows[0]['filters'] if rows else []:
        pairs=[(row,row['filters'][name]) for row in rows if row['truth'] is not None]
        if not pairs:continue
        errors=np.array([np.asarray(f['pose'])-row['truth'] for row,f in pairs]);errors[:,2]=wrap(errors[:,2])
        predicted=np.array([np.asarray(f['predicted_pose'])-row['truth'] for row,f in pairs]);predicted[:,2]=wrap(predicted[:,2])
        ps=np.array([f['covariance'] for row,f in pairs]);std=np.sqrt(np.maximum(0.,np.diagonal(ps,axis1=1,axis2=2)))
        nis=[v for row,f in pairs for v in f['nis']]
        item=metric(errors);item.update(predicted=metric(predicted),final_error=errors[-1].tolist(),
            n=len(pairs),corrections=sum(len(f['correction_ids']) for row,f in pairs),
            correction_frames=sum(bool(f['correction_ids']) for row,f in pairs),
            std_min=std.min(axis=0).tolist(),std_max=std.max(axis=0).tolist(),
            mean_nees=float(np.mean([e@np.linalg.solve(p,e) for e,p in zip(errors,ps)])),
            mean_nis=float(np.mean(nis)) if nis else None,
            marginal_2sigma_fraction=np.mean(abs(errors)<=2*std,axis=0).tolist())
        prediction_only=[e for e,(row,f) in zip(errors,pairs) if not f['correction_ids']]
        item['prediction_only_errors']=metric(prediction_only) if len(prediction_only) else None
        variants[name]=item
    markers={}
    for identity in (0,1):
        selected=[g for g in geometry if g['marker_id']==identity]
        if not selected:continue
        markers[str(identity)]=dict(n=len(selected),
            distance_range=[min(g['expected_range'] for g in selected),max(g['expected_range'] for g in selected)],
            pixel_side_range=[min(g['pixel_side'] for g in selected),max(g['pixel_side'] for g in selected)],
            bearing_range_deg=[float(np.rad2deg(min(g['expected_bearing'] for g in selected))),
                               float(np.rad2deg(max(g['expected_bearing'] for g in selected)))],
            incidence_range_deg=[min(g['incidence_deg'] for g in selected),max(g['incidence_deg'] for g in selected)],
            mean_range_error_face_m=float(np.mean([g['range_error_face'] for g in selected])),
            max_abs_range_error_face_m=max(abs(g['range_error_face']) for g in selected),
            mean_range_error_centre_m=float(np.mean([g['range_error_centre'] for g in selected])),
            max_abs_bearing_error_deg=float(np.rad2deg(max(abs(g['bearing_error']) for g in selected))),
            max_reprojection_px=max(g['reprojection_rmse_px'] for g in selected),
            examples=[selected[0],max(selected,key=lambda g:abs(g['range_error_face'])),selected[-1]])
    ages=[row['finish_age'] for row in rows]
    return dict(variants=variants,markers=markers,n_samples=len(rows),scored_samples=sum(r['truth'] is not None for r in rows),
                sequence_misses=sum(rows[i]['sequence']!=rows[i-1]['sequence']+1 for i in range(1,len(rows))),
                finish_age_s={q:float(np.percentile(ages,p)) for q,p in [('median',50),('p95',95),('max',100)]} if ages else {})


def run_plan(plan, frame_sink=None):
    faces=marker_faces();rows=[];geometry=[];error=None;last_lease=None;records=[]
    packet=None;failure_sample=None
    config=OwnerConfig(render_camera=True,max_runtime_s=plan.duration+8.,epoch='validation-'+str(time.monotonic_ns()))
    with OwnerSession(config) as owner:
        bank=FilterBank(owner.metadata,faces)
        generation=None
        try:
            for index in range(round(plan.duration/DT)):
                packet=owner.next_measurement();received=time.monotonic()
                observations,rejected=detect(packet,owner.metadata)
                detected=time.monotonic()
                filters=bank.advance(packet,observations,plan.dropout)
                finished=time.monotonic()
                # Command program depends ONLY on predeclared time/plan, never an estimate/truth.
                if generation is None:
                    armed=owner.client.arm()
                    if not armed.accepted:raise RuntimeError('Explicit arm rejected')
                    generation=armed.generation
                tail_slot = packet.sequence+1 >= round(plan.last_submission/DT)
                if not tail_slot:
                    wheels=plan.wheels(packet.state_time)
                    now=time.monotonic();deadline=now+.20
                    receipt=owner.client.submit(WheelCommandLease(generation,index,wheels,now,deadline))
                    if not receipt.accepted:raise RuntimeError('Lease rejected: '+receipt.reason)
                    last_lease=(deadline,wheels)
                # Separate scoring read is intentionally AFTER estimator and command generation.
                evaluation=None
                end=time.monotonic()+.02
                while time.monotonic()<end:
                    try: candidate=owner.evaluations.get(timeout=max(.0001,end-time.monotonic()))
                    except queue.Empty:break
                    if candidate.sequence==packet.sequence:
                        evaluation=candidate;break
                    if candidate.sequence>packet.sequence:raise ValueError('Evaluation channel skipped required sequence')
                scored=[] if evaluation is None else score_geometry(observations,evaluation,faces)
                geometry.extend(scored)
                # Evaluation errors are attached only AFTER all filter updates and commands.
                if evaluation is not None:
                    for value in filters.values():
                        error_pose=np.asarray(value['pose'])-np.asarray(evaluation.pose)
                        error_pose[2]=wrap(error_pose[2])
                        value['evaluation_error_m_m_rad']=error_pose.tolist()
                if frame_sink is not None:
                    frame_sink(packet,dict(observed_ids=[o.marker_id for o in observations],
                        estimate=filters['prior0_all_drop0_geom']['pose'],motion='stationary zero-only' if plan.name=='stationary' else 'prescribed'))
                rows.append(dict(sequence=packet.sequence,state_time=packet.state_time,
                    acquisition_monotonic=packet.sample_monotonic,receive_age=received-packet.sample_monotonic,
                    finish_age=finished-packet.sample_monotonic,observed_ids=[o.marker_id for o in observations],
                    rejected_ids=rejected,filters=filters,
                    received_monotonic=received,processing_finished_monotonic=finished,
                    detection_duration_s=detected-received,filter_duration_s=finished-detected,
                    rendering_duration_s=None, # owner does not publish render start/end
                    camera_frame_available=packet.rgb is not None,
                    truth=None if evaluation is None else list(evaluation.pose)))
        except Exception as exc:
            error=type(exc).__name__+': '+str(exc)
            if packet is not None:
                failure_sample=dict(sequence=packet.sequence,state_time=packet.state_time,
                    acquisition_monotonic=packet.sample_monotonic,
                    age_at_failure_s=time.monotonic()-packet.sample_monotonic)
        finally:
            try:
                stop=owner.client.stop('prescribed_localisation_validation_complete')
                stop_confirmed=stop.zero_written
            except Exception as exc:
                stop_confirmed=False
                error=(error or '')+'; stop_unconfirmed:'+type(exc).__name__
            records=list(owner.records)
        result=summarise(rows,geometry)
        zero=None if last_lease is None else next((v for kind,v in records if kind=='write'
            and v['success'] and v['requested']==[0.,0.] and v['wall']>=last_lease[0]),None)
        statuses=[v for kind,v in records if kind=='status']
        result.update(plan=plan.__dict__,error=error,stop_confirmed=stop_confirmed,
            completed=error is None and len(rows)==round(plan.duration/DT)
                and all(row['truth'] is not None for row in rows),
            failure_sample=failure_sample,
            overflow=any(kind=='fault' and v['reason']=='measurement_queue_overflow' for kind,v in records),
            expiry_observed=any(v['reason']=='command_expired' for v in statuses),
            expiry_zero_controls=None if zero is None else zero['actual_controls'],
            expiry_lateness_s=None if zero is None else zero['wall']-last_lease[0],
            expired_command=None if last_lease is None else last_lease[1],
            wheels_at_expiry=None if zero is None else zero['wheels'],
            wheels_after_expiry=next((v['wheels'] for kind,v in reversed(records)
                if zero is not None and kind=='status' and v['sim_time']>=zero['sim_time']+.3),None),
            telemetry_max_interval_s=float(max(np.diff([v['wall'] for v in statuses]))) if len(statuses)>1 else None,
            receive_age_max_s=max((r['receive_age'] for r in rows),default=None),
            physics_dt=owner.metadata['physics_dt'],image_size=list(packet.rgb.shape[1::-1]) if rows else None,
            quality_authorised=False,lqr_enabled=False,
            all_recorded_writes_zero=all(v['requested']==[0.,0.] for kind,v in records if kind=='write'),
            successful_zero_writes=sum(kind=='write' and v['success'] and v['requested']==[0.,0.] for kind,v in records),
            rendering_duration_available=False,
            processing_duration_s={name:max((row[name] for row in rows),default=None)
                for name in ('detection_duration_s','filter_duration_s')})
        detail=dict(rows=rows,geometry=geometry,owner_records=records)
    if owner.error:
        result['owner_error']=owner.error;result['stop_confirmed']=False;result['completed']=False
    return result,detail


def file_hashes(root=ROOT):
    paths=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','-z'],cwd=root).decode().split('\0')
    return {p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths if p and (root/p).is_file()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',choices=tuple(PLANS),default='stationary')
    parser.add_argument('--suite',action='store_true')
    parser.add_argument('--repeat',type=int,default=1)
    parser.add_argument('--details',action='store_true')
    args=parser.parse_args()
    if args.repeat<1:parser.error('--repeat must be positive')
    before=file_hashes();failed=False
    for name in PLANS if args.suite else [args.case]:
        for repetition in range(args.repeat):
            summary,detail=run_plan(PLANS[name]);summary['repetition']=repetition+1
            print(json.dumps(summary,sort_keys=True),flush=True)
            if args.details:print(json.dumps(detail,sort_keys=True),flush=True)
            failed |= bool(not summary['completed'] or summary['error'] or not summary['stop_confirmed'] or summary['overflow'])
    after=file_hashes()
    if before!=after:raise RuntimeError('Existing file bytes changed during validation')
    print(json.dumps(dict(preservation_files=len(before),unchanged=True,
                         deterministic_repeats_not_monte_carlo=True)))
    return int(failed)


if __name__=='__main__':raise SystemExit(main())
