"""Focused harness/scoring tests; no calibration claims or evidence regeneration."""
from dataclasses import fields,replace,FrozenInstanceError
import inspect
from types import SimpleNamespace
import pickle
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose

from estimation.ekf import camera_model,TurtleBotEKF
from integration.mujoco_owner import Measurement,Evaluation,OwnerSession,OwnerConfig
from integration.tests.validate_localisation import (
    ROOT,PLANS,Plan,marker_faces,rotation,Observation,FilterBank,
    expected_translation,score_geometry,metric,file_hashes,run_plan,
)


def metadata():
    return dict(epoch='test',camera_offset=[.05,0.],camera_yaw=0.,
                landmarks={0:[1.944,0.],1:[1.05,0.]})


def packet(k=0,wheels=(0.,0.)):
    return Measurement(wheels,[np.inf],None,(k+1)*.05,10.+k*.05,k,'test')


def observation(identity=1,k=0):
    z=camera_model(np.zeros(3),metadata()['landmarks'][identity],[.05,0.],0.)
    return Observation(identity,(0.,-.26,float(z[0])),tuple(z),
                       ((0.,0.),(1.,0.),(1.,1.),(0.,1.)),0.,1.,k,
                       (k+1)*.05,10.+k*.05,'test')


class GeometryAndProvenanceTests(unittest.TestCase):
    def test_static_face_centres_and_physical_sizes(self):
        import mujoco
        from integration.mujoco_owner import SCENE
        # Independent oracle: MuJoCo compiles hierarchy, geom transforms and sizes;
        # marker_faces() instead parses XML using its own quaternion composition.
        model=mujoco.MjModel.from_xml_path(str(SCENE))
        data=mujoco.MjData(model)
        mujoco.mj_forward(model,data)
        faces=marker_faces()
        self.assertEqual(set(faces),{0,1})
        for identity,name in ((0,'aruco_marker'),(1,'aruco_marker_1')):
            gid=model.geom(name).id
            self.assertEqual(model.geom_type[gid],mujoco.mjtGeom.mjGEOM_BOX)
            half=model.geom_size[gid]
            centre=data.geom_xpos[gid]
            axes=data.geom_xmat[gid].reshape(3,3)
            # The printed surface is the negative local-X face, not the box centre.
            face=centre-half[0]*axes[:,0]
            assert_allclose(faces[identity]['centre'],centre,rtol=0,atol=1e-12)
            assert_allclose(faces[identity]['face'],face,rtol=0,atol=1e-12)
            assert_allclose(faces[identity]['normal'],-axes[:,0],rtol=0,atol=1e-12)
            self.assertGreater(half[0],0.)
            assert_allclose(np.linalg.norm(face-centre),half[0],rtol=0,atol=1e-12)
            assert_allclose(half[1],half[2],rtol=0,atol=1e-12)
            assert_allclose(faces[identity]['side'],2*half[1],rtol=0,atol=1e-12)
        # Texture raster dimensions do not enter this physical geometry parser.
        self.assertNotIn('cv2',inspect.getsource(marker_faces))

    def test_optical_axes_and_known_static_translation(self):
        r=np.array([[0,0,1],[-1,0,0],[0,-1,0.]])
        e=Evaluation(0,.05,(0,0,0),(.05,0,.12),r,'test',10.)
        assert_allclose(expected_translation(e,[1.939,0,.15]),[0,-.03,1.889],atol=1e-14)
        self.assertEqual(np.linalg.det(r),1)

    def test_full_roll_pitch_rotation_not_planar_reconstruction(self):
        r=rotation([.9,.1,-.2,.3]);p=np.array([.2,-.3,.4]);optical=np.array([.1,-.2,1.8])
        world=p+r@optical
        e=Evaluation(0,.05,(99,99,2.),p,r,'test',10.)
        assert_allclose(expected_translation(e,world),optical,atol=1e-14)
        self.assertGreater(abs(r[2,0]),.01)

    def test_immutable_evaluation_after_input_mutation_and_pickle(self):
        p=np.array([.05,0,.12]);r=np.eye(3)
        e=Evaluation(0,.05,np.zeros(3),p,r,'test',10.)
        p[:]=99;r[:]=99;copy=pickle.loads(pickle.dumps(e))
        self.assertEqual(copy.camera_world_position,(.05,0.,.12))
        self.assertEqual(copy.camera_to_world_rotation,((1.,0.,0.),(0.,1.,0.),(0.,0.,1.)))
        with self.assertRaises(FrozenInstanceError):copy.epoch='changed'

    def test_provenance_required_before_scoring(self):
        import mujoco
        from integration.mujoco_owner import SCENE
        model=mujoco.MjModel.from_xml_path(str(SCENE))
        data=mujoco.MjData(model)
        mujoco.mj_forward(model,data)
        r=np.array([[0,0,1],[-1,0,0],[0,-1,0.]])
        e=Evaluation(0,.05,(0,0,0),(.05,0,.12),r,'test',10.)
        faces=marker_faces()
        for identity,name in ((0,'aruco_marker'),(1,'aruco_marker_1')):
            gid=model.geom(name).id
            face=data.geom_xpos[gid]-model.geom_size[gid,0]*data.geom_xmat[gid].reshape(3,3)[:,0]
            # Independently project onto the known optical basis (right=-world Y,
            # down=-world Z, forward=world X), without expected_translation().
            delta=face-np.asarray(e.camera_world_position)
            expected=np.array([-delta[1],-delta[2],delta[0]])
            obs=observation(identity)
            score=score_geometry([obs],e,faces)[0]
            assert_allclose(score['expected_translation'],expected,rtol=0,atol=1e-12)
            self.assertEqual(score['marker_id'],identity)
            for changes in (dict(epoch='wrong'),dict(epoch=None),dict(sequence=1),
                            dict(state_time=.10),dict(sample_monotonic=11.),
                            dict(sample_monotonic=None),dict(camera_world_position=None),
                            dict(camera_to_world_rotation=None)):
                with self.subTest(identity=identity,changes=changes):
                    with self.assertRaises(ValueError):
                        score_geometry([obs],replace(e,**changes),faces)
        with self.assertRaises(ValueError):expected_translation(replace(e,camera_to_world_rotation=np.zeros((3,3))),[0,0,0])

    def test_truth_is_not_a_feedback_or_command_argument(self):
        self.assertNotIn('camera_world_position',{f.name for f in fields(Measurement)})
        self.assertNotIn('camera_to_world_rotation',{f.name for f in fields(Measurement)})
        self.assertEqual(list(inspect.signature(FilterBank.advance).parameters),
                         ['self','packet','observations','dropout_window'])
        self.assertEqual(list(inspect.signature(Plan.wheels).parameters),['self','state_time'])
        with self.assertRaises(ValueError):FilterBank(metadata(),marker_faces()).advance(
            Evaluation(0,.05,(0,0,0)),[],(2,3))


class InnovationDiagnosticsTests(unittest.TestCase):
    def test_known_nis_wrapping_and_no_mutation(self):
        from integration.tests.validate_localisation import innovation_diagnostics
        ekf=TurtleBotEKF([0,0,0],np.zeros((3,3)),np.zeros((3,3)),np.diag([.25,.04]))
        before=ekf.state.copy();p=ekf.P.copy()
        expected=camera_model(ekf.state,[1.,0.])
        value=innovation_diagnostics(ekf,expected+[.5,2*np.pi+.2],[1.,0.],[0.,0.],0.)
        assert_allclose(value['innovation'],[.5,.2],atol=1e-14)
        self.assertAlmostEqual(value['nis'],2.)
        self.assertEqual(value['nis_status'],'valid')
        assert_allclose(ekf.state,before);assert_allclose(ekf.P,p)

    def test_missing_singular_and_indefinite_covariance(self):
        from integration.tests.validate_localisation import innovation_diagnostics
        for covariance in (None,np.zeros((3,3)),-np.eye(3)):
            ekf=SimpleNamespace(state=np.zeros(3),P=covariance,R=np.zeros((2,2)))
            value=innovation_diagnostics(ekf,[1.,0.],[1.,0.],[0.,0.],0.)
            self.assertIsNone(value['nis']);self.assertEqual(value['nis_status'],'invalid_covariance')

    def test_update_provenance_and_dropout_status(self):
        bank=FilterBank(metadata(),marker_faces())
        output=bank.advance(packet(),[observation()],(0.,1.))
        corrected=output['prior0_all_drop0_geom']['updates'][0]
        self.assertEqual(corrected['status'],'corrected')
        self.assertEqual(corrected['acquisition_monotonic'],10.)
        self.assertEqual(corrected['sequence'],0);self.assertEqual(corrected['epoch'],'test')
        self.assertGreaterEqual(corrected['processing_finished_monotonic'],corrected['processing_started_monotonic'])
        self.assertEqual(output['prior0_all_drop1_geom']['updates'][0]['status'],'dropout')
        self.assertNotIn('evaluation_error_m_m_rad',output['prior0_all_drop0_geom'])


class FilterAndMetricTests(unittest.TestCase):
    def test_marker_identity_two_landmark_correction_and_missing_data(self):
        bank=FilterBank(metadata(),marker_faces())
        result=bank.advance(packet(),[observation(1),observation(0)],(2,3))
        self.assertEqual(result['prior0_first_drop0_geom']['correction_ids'],[1])
        self.assertEqual(result['prior0_all_drop0_geom']['correction_ids'],[1,0])
        missing=bank.advance(packet(1),[],(2,3))
        self.assertEqual(missing['prior0_all_drop0_geom']['correction_ids'],[])
        self.assertTrue(np.isfinite(missing['prior0_all_drop0_geom']['covariance']).all())

    def test_invalid_association_provenance_or_sequence_rejected(self):
        for obs in [replace(observation(),marker_id=99),replace(observation(),epoch='wrong'),
                    replace(observation(),sequence=1),replace(observation(),acquisition_monotonic=11.)]:
            with self.assertRaises(ValueError):FilterBank(metadata(),marker_faces()).advance(packet(),[obs],(2,3))
        with self.assertRaises(ValueError):FilterBank(metadata(),marker_faces()).advance(packet(1),[],(2,3))
        with self.assertRaises(ValueError):FilterBank(metadata(),marker_faces()).advance(packet(),[observation(),observation()],(2,3))

    def test_dropout_suppresses_correction_not_prediction(self):
        result=FilterBank(metadata(),marker_faces()).advance(packet(),[observation()],(0,.1))
        self.assertEqual(result['prior0_all_drop1_geom']['correction_ids'],[])
        self.assertEqual(result['prior0_all_drop0_geom']['correction_ids'],[1])
        assert_allclose(result['prior0_all_drop1_geom']['covariance'],
                        np.diag([.05**2,.05**2,np.deg2rad(5)**2])+
                        np.diag([.005**2,.005**2,np.deg2rad(.5)**2]))

    def test_prediction_uses_measured_wheels_not_any_target(self):
        calls=[];original=TurtleBotEKF.predict
        def record(ekf,control,dt):calls.append((control.copy(),dt));return original(ekf,control,dt)
        with patch.object(TurtleBotEKF,'predict',record):
            FilterBank(metadata(),marker_faces()).advance(packet(wheels=(1.,3.)),[],(2,3))
        self.assertEqual(len(calls),14)
        for control,dt in calls:assert_allclose(control,[.066,.4125]);self.assertEqual(dt,.05)

    def test_prescribed_commands_are_bounded_signed_and_zero_tailed(self):
        for plan in PLANS.values():
            self.assertEqual(plan.wheels(.1),(0.,0.))
            self.assertEqual(plan.wheels(plan.duration),(0.,0.))
            self.assertLessEqual(max(abs(v) for v in plan.wheels(.6)),6.67)
        left,right=PLANS['rotation'].wheels(.6);self.assertLess(left,0);self.assertGreater(right,0)
        left,right=PLANS['curve'].wheels(.6);self.assertGreater(left,0);self.assertGreater(right,left)
        with self.assertRaises(ValueError):Plan('invalid',3.,1.,0.,2.).wheels(.6)

    def test_command_boundaries_use_sample_slots_not_float_equality(self):
        plan=PLANS['straight']
        self.assertNotEqual(plan.wheels(.5-1e-12),(0.,0.))
        self.assertNotEqual(plan.wheels(3.45-1e-12),(0.,0.))
        self.assertEqual(plan.wheels(3.5-1e-12),(0.,0.))
        self.assertEqual(plan.wheels(3.5+1e-12),(0.,0.))

    def test_metric_definitions_and_wrap_are_reproducible(self):
        errors=[[.03,.04,2*np.pi+.1],[0,0,-.1]]
        result=metric(errors)
        self.assertAlmostEqual(result['position_rmse_m'],.05/np.sqrt(2))
        self.assertAlmostEqual(result['position_max_m'],.05)
        self.assertAlmostEqual(result['heading_rmse_deg'],np.rad2deg(.1))
        self.assertEqual(result,metric(errors))

    def test_expired_lease_reports_incomplete_trial_and_failed_sample(self):
        import time
        class RejectedOwner:
            def __init__(self,config):
                self.metadata=metadata();self.metadata['epoch']=config.epoch;self.metadata['physics_dt']=.002
                self.records=[];self.error=None
                self.client=SimpleNamespace(
                    arm=lambda:SimpleNamespace(accepted=True,generation=1),
                    submit=lambda lease:SimpleNamespace(accepted=False,reason='not_armed; command_expired'),
                    stop=lambda reason:SimpleNamespace(zero_written=True))
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def next_measurement(self):
                return Measurement([0,0],[np.inf],None,.05,time.monotonic(),0,self.metadata['epoch'])
        with patch('integration.tests.validate_localisation.OwnerSession',RejectedOwner), patch(
                'integration.tests.validate_localisation.detect',return_value=([],[])):
            result,_=run_plan(PLANS['stationary'])
        self.assertFalse(result['completed'])
        self.assertEqual(result['n_samples'],0)
        self.assertEqual(result['failure_sample']['sequence'],0)
        self.assertIn('command_expired',result['error'])
        self.assertTrue(result['stop_confirmed'])

    def test_both_historical_runs_preserved(self):
        original=ROOT/'simulation/identification_results';archive=ROOT/'simulation/identification_results_windows_py3146_cb1ea64'
        self.assertEqual(len(list(original.iterdir())),47);self.assertEqual(len(list(archive.iterdir())),47)
        for path in original.iterdir():
            old=subprocess.check_output(['git','show','0550e22:simulation/identification_results/'+path.name],cwd=ROOT)
            self.assertEqual(path.read_bytes(),old)
            windows=subprocess.check_output(['git','show','fa37957:simulation/identification_results/'+path.name],cwd=ROOT)
            self.assertEqual((archive/path.name).read_bytes(),windows)


class ActualOwnerEvaluationTests(unittest.TestCase):
    def test_actual_transform_provenance_expiry_and_zero_without_motion(self):
        import time
        from control.command_gateway import WheelCommandLease
        before=file_hashes()
        with OwnerSession(OwnerConfig(render_camera=False,queue_capacity=64,max_runtime_s=5.,epoch='score-test')) as owner:
            packet=owner.next_measurement();e=owner.evaluations.get(timeout=1.)
            self.assertEqual(e.sequence,packet.sequence);self.assertEqual(e.epoch,packet.epoch)
            self.assertEqual(e.sample_monotonic,packet.sample_monotonic)
            r=np.asarray(e.camera_to_world_rotation)
            assert_allclose(r.T@r,np.eye(3),atol=1e-12)
            self.assertAlmostEqual(np.linalg.det(r),1.)
            t=expected_translation(e,marker_faces()[0]['face']);self.assertGreater(t[2],1.8)
            generation=owner.client.arm().generation;now=time.monotonic();deadline=now+.15
            self.assertTrue(owner.client.submit(WheelCommandLease(generation,0,(0.,0.),now,deadline)).accepted)
            owner.wait(lambda:any(k=='status' and v['reason']=='command_expired' for k,v in owner.records))
            zeros=[v for k,v in owner.records if k=='write' and v['success'] and v['wall']>=deadline]
            self.assertTrue(zeros);self.assertEqual(zeros[0]['actual_controls'],[0.,0.])
            self.assertFalse(owner.client.submit(WheelCommandLease(generation,1,(0.,0.),time.monotonic(),time.monotonic()+.1)).accepted)
            self.assertTrue(owner.client.stop('scoring_test_complete').zero_written)
        self.assertEqual(before,file_hashes())


if __name__=='__main__':unittest.main()
