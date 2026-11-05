"""Navigation safety diagnostics; no historical outputs or deployment calibration.

Quality/timing values below are explicit synthetic fixtures. Real MuJoCo tests
use prescribed leases, NOT an autonomous localisation-validity claim. Rendering
is off in expiry tests to isolate the owner from unavailable GL contexts.
"""
from dataclasses import fields, replace
import os
import pickle
import time
import unittest

import numpy as np

from control.command_gateway import WheelCommandLease
from control.estimated_pose_interface import PoseEstimate, InterfacePolicy, EstimatedPoseInterface
from control.staged_terminal_controller import StagedTrajectoryTerminalController
from control.terminal_controller import TerminalParameters
from control.trajectory import reference
from integration.localisation_policy import QualityCriteria, LocalisationPolicy
from integration.mujoco_owner import OwnerSession, OwnerConfig, Measurement
from integration.main import dispatch, DURATION, run


def criteria():
    return QualityCriteria((.2,.2,.3),.1,.2,.1,'SYNTHETIC TEST ONLY; not calibration')


def pose(k=0, **changes):
    values=dict(pose=np.zeros(3),covariance=np.eye(3)*.001,state_time=.05+k*.05,
                sample_monotonic=10.+k*.05,sequence=k,epoch='fixture',frame_id='map',
                body_frame_id='base',localisation_valid=False)
    values.update(changes)
    return PoseEstimate(**values)


def policy(configured=True):
    return LocalisationPolicy('fixture','map','base',.05,.2,1e-8,
                              criteria() if configured else None)


def assess(snapshot=None, **changes):
    snapshot=pose() if snapshot is None else snapshot
    args=dict(previous=None,expected_time=.05,now=10.01,camera_status='corrected',
              last_correction_state_time=.05)
    args.update(changes)
    return policy().assess(snapshot,**args)


def session(capacity=64):
    return OwnerSession(OwnerConfig(render_camera=False,queue_capacity=capacity,max_runtime_s=10.))


class MarkerFaceMapTests(unittest.TestCase):
    def test_face_map_ids_orientation_and_camera_offset_separation(self):
        import mujoco
        from integration.mujoco_owner import SCENE, marker_face_metadata
        from integration.tests.validate_localisation import marker_faces
        from estimation.ekf import camera_model
        model=mujoco.MjModel.from_xml_path(str(SCENE));data=mujoco.MjData(model)
        mujoco.mj_forward(model,data)
        actual=marker_face_metadata(model,data);expected=marker_faces()
        self.assertEqual(set(actual),{0,1})
        for identity in (0,1):
            np.testing.assert_allclose(actual[identity]['face'],expected[identity]['face'],atol=1e-12)
            np.testing.assert_allclose(actual[identity]['normal'],expected[identity]['normal'],atol=1e-12)
            self.assertGreater(np.linalg.norm(np.asarray(actual[identity]['face'])-actual[identity]['centre']),0)
        camera=model.camera('front_camera').id
        np.testing.assert_allclose(model.cam_pos[camera],[.05,0,.12])
        point=np.asarray(actual[1]['face'][:2])
        expected_range=np.linalg.norm(point-np.array([.05,0.]))
        self.assertAlmostEqual(camera_model(np.zeros(3),point,[.05,0.],0.)[0],expected_range)
        # No camera subtraction in the world landmark: that would apply it twice.
        np.testing.assert_allclose(point,expected[1]['face'][:2])
        gid=model.geom('aruco_marker').id
        angle=.4;axes=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1.]])
        data.geom_xmat[gid]=axes.ravel()
        rotated=marker_face_metadata(model,data)[0]
        np.testing.assert_allclose(rotated['face'],data.geom_xpos[gid]-model.geom_size[gid,0]*axes[:,0])

    def test_live_owner_publishes_face_map_not_box_centres(self):
        with session() as owner:
            self.assertEqual(set(owner.metadata['landmarks']),{0,1})
            for identity,point in owner.metadata['landmarks'].items():
                info=owner.metadata['landmark_faces'][identity]
                np.testing.assert_allclose(point,info['face'][:2])
                self.assertFalse(np.array_equal(point,info['centre'][:2]))


class LocalisationPolicyTests(unittest.TestCase):
    def test_no_policy_never_authorises_motion(self):
        result=policy(False).assess(pose(),previous=None,expected_time=.05,now=10.01,
                                  camera_status='corrected',last_correction_state_time=.05)
        self.assertFalse(result.valid);self.assertIn('no_justified',result.reason)

    def test_fresh_corrected_fixture_accepts_with_explicit_criteria(self):
        self.assertTrue(assess().valid)
        with self.assertRaises(ValueError):
            QualityCriteria((.2,.2,.3),.1,.2,.1,'')

    def test_stale_future_duplicate_skipped_epoch_frame_and_malformed(self):
        for changes in [dict(sample_monotonic=9.),dict(sample_monotonic=11.),
                        dict(epoch='reset'),dict(frame_id='odom'),dict(body_frame_id='camera'),
                        dict(pose=[0,0]),dict(pose=[np.nan,0,0]),
                        dict(covariance=np.eye(2)),dict(covariance=-np.eye(3)),
                        dict(covariance=np.full((3,3),np.inf)),dict(sequence=True)]:
            with self.subTest(changes=changes):self.assertFalse(assess(pose(**changes)).valid)
        self.assertFalse(assess(pose(),previous=pose()).valid)
        self.assertFalse(assess(pose(2),expected_time=.10,now=10.11).valid)

    def test_missing_camera_is_prediction_only_not_numerical_failure(self):
        self.assertTrue(assess(camera_status='missing',last_correction_state_time=0.).valid)
        no_prior=assess(camera_status='missing',last_correction_state_time=None)
        self.assertFalse(no_prior.valid);self.assertNotIn('numerical',no_prior.reason)
        late=assess(pose(3),expected_time=.20,now=10.16,camera_status='missing',
                    last_correction_state_time=.05)
        self.assertFalse(late.valid);self.assertIn('prediction_only_budget',late.reason)
        self.assertFalse(assess(camera_status='failed').valid)
        self.assertFalse(assess(numerical_health=False).valid)

    def test_uncertainty_jump_and_heading_wrap(self):
        self.assertFalse(assess(pose(covariance=np.eye(3))).valid)
        self.assertFalse(assess(pose(1,pose=[1.,0,0]),previous=pose(),expected_time=.1,
                               now=10.06,last_correction_state_time=.1).valid)
        old=pose(pose=[0,0,np.pi-.01])
        new=pose(1,pose=[0,0,-np.pi+.01])
        self.assertTrue(assess(new,previous=old,expected_time=.1,now=10.06,
                              last_correction_state_time=.1).valid)

    def test_processing_delay_does_not_refresh_acquisition_time(self):
        original=pose()
        decision=assess(original,now=10.21)
        self.assertFalse(decision.valid)
        self.assertEqual(original.sample_monotonic,10.)

    def test_measurement_detached_immutable_and_truth_excluded(self):
        wheel=np.array([1.,2.]);rgb=np.zeros((2,2,3),dtype=np.uint8)
        packet=Measurement(wheel,[np.inf],rgb,.05,10.,0,'fixture')
        wheel[:]=99;rgb[:]=255
        copy=pickle.loads(pickle.dumps(packet))
        self.assertEqual(copy.wheels.tolist(),[1.,2.]);self.assertFalse(copy.rgb.any())
        for value in (copy.wheels,copy.lidar,copy.rgb):
            with self.assertRaises(ValueError):value.setflags(write=True)
        self.assertEqual({f.name for f in fields(packet)},
                         {'wheels','lidar','rgb','state_time','sample_monotonic','sequence',
                          'epoch','frame_id','body_frame_id'})


class ActualOwnerExpiryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.owner=session()
        cls.packet=cls.owner.next_measurement()
        first=cls.owner.client.arm();cls.first_generation=first.generation
        now=time.monotonic();cls.deadline=now+.20
        receipt=cls.owner.client.submit(WheelCommandLease(first.generation,0,(3.,3.),now,cls.deadline))
        assert receipt.accepted
        # No RPC/controller callbacks until owner records expiry itself.
        cls.owner.wait(lambda:any(k=='status' and v['reason']=='command_expired'
                                  for k,v in cls.owner.records))
        cls.zero=next(v for k,v in cls.owner.records if k=='write' and v['success']
                      and v['requested']==[0.,0.] and v['wall']>=cls.deadline)
        cls.owner.wait(lambda:any(k=='status' and v['sim_time']>=cls.zero['sim_time']+.4
                                  for k,v in cls.owner.records))

    @classmethod
    def tearDownClass(cls):cls.owner.close()

    def test_actual_zero_then_separate_measured_deceleration(self):
        self.assertEqual(self.zero['actual_controls'],[0.,0.])
        self.assertGreater(min(self.zero['wheels']),.5)
        after=next(v for k,v in self.owner.records if k=='status'
                   and v['sim_time']>=self.zero['sim_time']+.4)
        self.assertLess(max(abs(v) for v in after['wheels']),.1)
        print('\nNavigation owner expiry: wheels at zero',self.zero['wheels'],
              'after .4 sim s',after['wheels'],
              'wall lateness',self.zero['wall']-self.deadline)

    def test_single_owner_and_truth_separate(self):
        pid=self.owner.metadata['owner_pid'];self.assertNotEqual(pid,os.getpid())
        self.assertTrue(all(v['owner_pid']==pid for k,v in self.owner.records))
        self.assertNotIn('pose',vars(self.packet));self.assertNotIn('truth',vars(self.packet))
        self.assertAlmostEqual(self.packet.state_time,.05)
        self.assertLessEqual(self.packet.sample_monotonic,time.monotonic())

    def test_no_stale_revival_and_explicit_rearm(self):
        now=time.monotonic()
        late=self.owner.client.submit(WheelCommandLease(self.first_generation,1,(3.,3.),now,now+.1))
        self.assertFalse(late.accepted);self.assertFalse(late.armed)
        rearm=self.owner.client.arm();self.assertTrue(rearm.armed)
        old=self.owner.client.submit(WheelCommandLease(self.first_generation,2,(3.,3.),now,now+.1))
        self.assertFalse(old.accepted)
        self.assertEqual(old.applied_wheels,(0.,0.))
        self.owner.client.stop('test_complete')


class OwnerFailureAndInterfaceTests(unittest.TestCase):
    def test_default_integration_never_arms_without_quality_policy(self):
        report=run(render_camera=False,max_samples=1)
        self.assertFalse(report['armed_once'])
        self.assertTrue(report['stop_confirmed'])
        self.assertFalse(report['terminal_clearance'])
        self.assertIn('no_justified_quality_policy',report['reasons'][0])

    def test_expiry_while_physics_paused(self):
        with session() as owner:
            owner.next_measurement();owner.pause.set()
            owner.wait(lambda:any(k=='status' for k,v in owner.records))
            g=owner.client.arm().generation;now=time.monotonic();deadline=now+.15
            self.assertTrue(owner.client.submit(WheelCommandLease(g,0,(2.,2.),now,deadline)).accepted)
            owner.wait(lambda:any(k=='write' and v['requested']==[2.,2.] for k,v in owner.records))
            start=next(v for k,v in reversed(owner.records) if k=='write' and v['requested']==[2.,2.])
            owner.wait(lambda:any(k=='status' and v['reason']=='command_expired' for k,v in owner.records))
            zero=next(v for k,v in owner.records if k=='write' and v['success']
                      and v['requested']==[0.,0.] and v['wall']>=deadline)
            self.assertEqual(zero['actual_controls'],[0.,0.])
            self.assertEqual(zero['sim_time'],start['sim_time'])

    def test_overflow_latches_and_requires_new_session(self):
        with session(1) as owner:
            owner.next_measurement()
            generation=owner.client.arm().generation
            now=time.monotonic()
            moving=owner.client.submit(WheelCommandLease(generation,0,(2.,2.),now,now+.25))
            self.assertTrue(moving.accepted)
            self.assertEqual(moving.applied_wheels,(2.,2.))
            owner.wait(lambda:any(k=='fault' and v['reason']=='measurement_queue_overflow'
                                  for k,v in owner.records))
            status=owner.client.poll();self.assertFalse(status.armed)
            self.assertEqual(status.applied_wheels,(0.,0.))
            self.assertFalse(owner.client.arm().armed)

    def test_failed_writer_reports_stop_unconfirmed(self):
        with session() as owner:
            owner.next_measurement();g=owner.client.arm().generation;now=time.monotonic()
            self.assertTrue(owner.client.submit(WheelCommandLease(g,0,(2.,2.),now,now+.2)).accepted)
            owner.fail_write.set()
            failed=owner.client.stop('TEST-ONLY failure')
            self.assertFalse(failed.zero_written)
            self.assertIsNone(failed.applied_wheels)
            owner.wait(lambda:any(k=='write' and not v['success'] for k,v in owner.records))
            event=next(v for k,v in reversed(owner.records) if k=='write' and not v['success'])
            self.assertEqual(event['actual_controls'],[2.,2.])
            owner.fail_write.clear()
            self.assertTrue(owner.client.stop('explicit_recovery_stop').zero_written)

    def make_interface(self,owner,end_time=12.):
        goal=np.array([.5,0,0]);parameters=TerminalParameters()
        p=InterfacePolicy('map','base',.2,.1,1e-8,lambda previous,current:True)
        return EstimatedPoseInterface(owner.client,
            lambda:StagedTrajectoryTerminalController(goal,end_time,parameters),p,
            goal=goal,task_id='fixture',terminal_start_s=end_time,parameters=parameters)

    def test_bad_estimates_and_missing_policy_attempt_real_zero(self):
        with session() as owner:
            owner.next_measurement()
            for changes in [dict(frame_id='wrong'),dict(epoch='wrong'),dict(pose=[0,0]),
                            dict(covariance=-np.eye(3)),dict(sample_monotonic=time.monotonic()-1),
                            dict(localisation_valid=False),dict(sequence=0),dict(state_time=.15)]:
                interface=self.make_interface(owner)
                self.assertTrue(interface.arm(epoch='fixture',state_time_origin=.05).accepted)
                initial=pose(sample_monotonic=time.monotonic(),localisation_valid=True)
                moving=interface.step(initial,lambda t:reference('straight',t))
                self.assertTrue(moving.accepted)
                self.assertGreater(max(abs(v) for v in moving.receipt.applied_wheels),0)
                snap=pose(1,sample_monotonic=time.monotonic(),localisation_valid=True)
                snap=replace(snap,**changes)
                result=interface.step(snap,lambda t:reference('straight',t))
                self.assertFalse(result.accepted);self.assertTrue(result.zero_dispatch_confirmed)
                self.assertFalse(interface.armed)
                self.assertEqual(result.receipt.applied_wheels,(0.,0.))

    def test_navigation_stop_request_does_not_invoke_controller(self):
        with session() as owner:
            owner.next_measurement()
            interface=self.make_interface(owner)
            interface.arm(epoch='fixture',state_time_origin=.05)
            supervisor=interface.supervisor
            result=dispatch(interface,pose(localisation_valid=True),
                lambda t:reference('straight',t),relative_time=0.,
                stop_reason='terminal_clearance_unavailable')
            self.assertTrue(result.zero_dispatch_confirmed)
            self.assertFalse(supervisor.terminal.started)
            self.assertEqual(result.reason,'terminal_clearance_unavailable')

    def test_both_camera_ids_are_associated_from_detached_images(self):
        from unittest.mock import patch
        from integration.main import camera_observation
        packet=Measurement([0.,0.],np.ones(2),np.zeros((4,4,3),np.uint8),.05,1.,0,'fixture')
        metadata=dict(landmarks={0:[2.,.4],1:[1.,-.2]},marker_sizes={0:.2,1:.2},camera_matrix=np.eye(3))
        for marker_id in (0,1):
            with patch('integration.main.detect_aruco_marker',return_value=(marker_id,np.zeros((4,2)))), \
                 patch('integration.main.estimate_marker_pose',return_value=(np.array([0.,0.,1.]),None)):
                identified,measurement,landmark=camera_observation(packet,metadata)
                self.assertEqual(identified,marker_id)
                np.testing.assert_array_equal(landmark,metadata['landmarks'][marker_id])
                self.assertEqual(measurement.shape,(2,1))
        with patch('integration.main.detect_aruco_marker',return_value=(99,np.zeros((4,2)))):
            with self.assertRaises(ValueError): camera_observation(packet,metadata)

    def test_unknown_clearance_and_end_of_trajectory_never_start_staging(self):
        with session() as owner:
            owner.next_measurement()
            interface=self.make_interface(owner,end_time=0.)
            interface.arm(epoch='fixture',state_time_origin=.05)
            supervisor=interface.supervisor
            result=interface.step(pose(sample_monotonic=time.monotonic(),localisation_valid=True),
                                  lambda t:reference('stationary',t),clearance=None)
            self.assertFalse(result.accepted);self.assertTrue(result.zero_dispatch_confirmed)
            self.assertFalse(supervisor.terminal.started)
            interface=self.make_interface(owner)
            interface.arm(epoch='fixture',state_time_origin=.05)
            result=dispatch(interface,pose(localisation_valid=True),lambda t:reference('straight',t),
                            relative_time=DURATION)
            self.assertFalse(result.accepted);self.assertTrue(result.zero_dispatch_confirmed)
            self.assertIn('terminal_clearance_unavailable',result.reason)


if __name__=='__main__':unittest.main()
