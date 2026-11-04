"""Independent prediction-only tests; no historical outputs or navigation claims.

Run: <development-python> -B -m unittest discover -s tests -p test_ekf_wheel_adapter.py -v
Nominal .033/.160 geometry, priors, Q and time tolerances here are explicit test
fixtures, not calibrated localisation/uncertainty policy. Actual-model tests use
prescribed ctrl only inside one test owner, never a controller or gateway.
"""
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
import pickle
import threading
import time
import unittest
from unittest.mock import patch

import mujoco
import numpy as np
from numpy.testing import assert_allclose

from control.estimated_pose_interface import InterfacePolicy, validate_estimate
from estimation.ekf import motion_model
from simulation.ekf_wheel_adapter import (
    WheelReader, WheelSample, WheelPredictionAdapter, body_velocity,
)

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT/'simulation/robotis_tb3/scene_turtlebot3_burger_gain_test.xml'


def sample(t=0., seq=0, left=2., right=2., epoch='test-epoch', wall=None):
    return WheelSample(left, right, t, 10.+t if wall is None else wall, seq, epoch)


def adapter(seed=None, **changes):
    arguments = dict(radius_m=.033, track_m=.160, prediction_dt=.05,
                     initial_pose=[0., 0., 0.], initial_covariance=np.eye(3)*.01,
                     process_noise_per_call=np.eye(3)*1e-6, seed=seed or sample(),
                     frame_id='map', body_frame_id='base')
    arguments.update(changes)
    return WheelPredictionAdapter(**arguments)


class WheelPredictionTests(unittest.TestCase):
    def test_body_signs_and_si_units(self):
        for left, right, expected in [(2,2,[.066,0]), (-2,-2,[-.066,0]),
                                      (-2,2,[0,.825]), (2,-2,[0,-.825]),
                                      (1,3,[.066,.4125])]:
            assert_allclose(body_velocity(sample(left=left,right=right),
                                         radius_m=.033,track_m=.160),expected)

    def test_forward_reverse_rotation_and_curve_match_existing_model(self):
        for left,right in [(2,2),(-2,-2),(-2,2),(2,-2),(1,3)]:
            a = adapter(sample(left=left,right=right))
            control = body_velocity(sample(left=left,right=right),radius_m=.033,track_m=.160)
            result = a.predict(sample(.05,1,left,right))
            assert_allclose(result.pose,motion_model(np.zeros(3),control,.05),atol=1e-14)

    def test_endpoint_trapezoid_not_target_or_latest_rate(self):
        a = adapter(sample(left=0,right=0))
        result = a.predict(sample(.05,1,4,4))
        assert_allclose(result.pose,[.0033,0,0],atol=1e-14)

    def test_exact_prediction_call_and_elapsed_time(self):
        a = adapter()
        with patch.object(a._ekf,'predict',wraps=a._ekf.predict) as predict:
            a.predict(sample(.05,1))
            self.assertEqual(predict.call_count,1)
            assert_allclose(predict.call_args.args[0],[.066,0])
            self.assertAlmostEqual(predict.call_args.args[1],.05)

    def test_heading_wrapped_initial_and_after_prediction(self):
        a = adapter(sample(left=-2,right=2),initial_pose=[0,0,3*np.pi])
        self.assertGreaterEqual(a.latest().pose[2],-np.pi)
        result=a.predict(sample(.05,1,-2,2))
        self.assertTrue(-np.pi <= result.pose[2] < np.pi)
        assert_allclose(result.pose,[0,0,-np.pi+.825*.05],atol=1e-14)

    def test_invalid_sample_values(self):
        for field in ('left_rad_s','right_rad_s','state_time','sample_monotonic'):
            for value in (np.nan,np.inf,-np.inf,True):
                with self.subTest(field=field,value=value), self.assertRaises(ValueError):
                    replace(sample(),**{field:value})
        for fields in (dict(sequence=-1),dict(sequence=True),dict(sequence=1.),
                       dict(epoch=''),dict(state_time=-1),dict(sample_monotonic=-1)):
            with self.assertRaises(ValueError):replace(sample(),**fields)

    def test_duplicate_reversed_missing_time_and_sequence_rejected(self):
        a=adapter(); old=a.latest()
        cases=[sample(),sample(.01,1),sample(.1,1),sample(.05,2),
               sample(.05,1,wall=10.),sample(.05,1,wall=9.),
               sample(.05,1,epoch='other'),None]
        for value in cases:
            with self.subTest(value=value),self.assertRaises(ValueError):a.predict(value)
            self.assertIs(a.latest(),old)
        a.predict(sample(.05,1))
        with self.assertRaises(ValueError):a.predict(sample(.04,2))

    def test_config_and_covariance_validation(self):
        for changes in (dict(radius_m=0),dict(track_m=-1),dict(prediction_dt=0),
                        dict(prediction_dt=np.nan),dict(time_tolerance_s=.025),
                        dict(frame_id=''),dict(initial_pose=[0,0,np.nan]),
                        dict(initial_covariance=-np.eye(3)),
                        dict(process_noise_per_call=np.ones((2,2))),
                        dict(initial_covariance=[[1,1,0],[0,1,0],[0,0,1]])):
            with self.subTest(changes=changes),self.assertRaises(ValueError):adapter(**changes)

    def test_fixed_schedule_and_clock_tolerance(self):
        a=adapter()
        for k in range(1,21):
            snap=a.predict(sample(k*.05,k,wall=10+k*.051))
            self.assertEqual(snap.state_time,k*.05)
            self.assertEqual(snap.sample_monotonic,10+k*.051)
        self.assertAlmostEqual(a.latest().pose[0],.066)
        a=adapter()
        a.predict(sample(.05+1e-10,1))
        with self.assertRaises(ValueError):a.predict(sample(.10+1e-7,2))

    def test_epoch_reset_requires_fresh_prior_and_never_used_epoch(self):
        a=adapter();a.predict(sample(.05,1))
        with self.assertRaises(ValueError):a.reset(initial_pose=[0,0,0],initial_covariance=np.eye(3),seed=sample())
        snap=a.reset(initial_pose=[1,2,.3],initial_covariance=np.eye(3)*.2,
                     seed=sample(0,0,epoch='new-epoch',wall=11))
        assert_allclose(snap.pose,[1,2,.3]);self.assertEqual(snap.epoch,'new-epoch')
        with self.assertRaises(ValueError):a.predict(sample(.1,2))
        a.predict(sample(.05,1,epoch='new-epoch',wall=11.05))
        with self.assertRaises(ValueError):a.reset(initial_pose=[0,0,0],initial_covariance=np.eye(3),seed=sample())

    def test_detached_immutable_snapshot_and_prior_inputs(self):
        pose=np.zeros(3);p=np.eye(3)*.01;q=np.eye(3)*1e-6
        a=adapter(initial_pose=pose,initial_covariance=p,process_noise_per_call=q)
        old=a.latest();pose[:]=99;p[:]=99;q[:]=99
        result=a.predict(sample(.05,1))
        assert_allclose(old.pose,[0,0,0]);assert_allclose(old.covariance,np.eye(3)*.01)
        self.assertFalse(np.shares_memory(result.covariance,a._ekf.P))
        for array in (result.pose,result.covariance):
            with self.assertRaises(ValueError):array.setflags(write=True)
        with self.assertRaises(FrozenInstanceError):result.sequence=99
        self.assertIs(a.latest(),result)  # retrieval never refreshes timestamps/dwell

    def test_snapshot_remains_immutable_after_ipc_serialisation(self):
        snap=adapter().predict(sample(.05,1))
        copy=pickle.loads(pickle.dumps(snap))
        assert_allclose(copy.pose,snap.pose)
        self.assertEqual(copy.provenance,snap.provenance)
        for array in (copy.pose,copy.covariance):
            with self.assertRaises(ValueError):array.setflags(write=True)

    def test_same_completed_prediction_state_and_covariance(self):
        a=adapter()
        def blocked_prediction(control,dt):
            a._ekf.state=np.array([1.,2.,.3]);started.set();release.wait(2)
            a._ekf.P=np.eye(3)*.5
        started=threading.Event();release=threading.Event();observed=[]
        with patch.object(a._ekf,'predict',side_effect=blocked_prediction):
            worker=threading.Thread(target=lambda:a.predict(sample(.05,1)))
            reader=threading.Thread(target=lambda:observed.append(a.latest()))
            worker.start();self.assertTrue(started.wait(2));reader.start()
            release.set();worker.join(2);reader.join(2)
            self.assertFalse(worker.is_alive() or reader.is_alive())
        assert_allclose(observed[0].pose,[1,2,.3]);assert_allclose(observed[0].covariance,np.eye(3)*.5)
        self.assertEqual(observed[0].sequence,1)

    def test_prediction_failure_latches_without_publishing_partial_state(self):
        a=adapter()
        with patch.object(a._ekf,'predict',side_effect=RuntimeError('test-only fault')):
            with self.assertRaises(RuntimeError):a.predict(sample(.05,1))
        with self.assertRaises(RuntimeError):a.latest()
        with self.assertRaises(RuntimeError):a.predict(sample(.05,1))
        a.reset(initial_pose=[0,0,0],initial_covariance=np.eye(3),seed=sample(epoch='reset'))
        self.assertFalse(a.latest().localisation_valid)

    def test_process_noise_is_added_per_call_not_per_second(self):
        for dt,calls in [(.05,20),(.1,10)]:
            a=adapter(sample(left=0,right=0),prediction_dt=dt,initial_covariance=np.zeros((3,3)))
            for k in range(1,calls+1):a.predict(sample(k*dt,k,0,0))
            assert_allclose(a.latest().covariance,calls*np.eye(3)*1e-6,atol=1e-18)

    def test_explicit_frequency_scaled_fixture_Q(self):
        outputs=[]
        for dt,calls in [(.05,20),(.1,10)]:
            a=adapter(sample(left=0,right=0),prediction_dt=dt,
                      initial_covariance=np.zeros((3,3)),process_noise_per_call=np.eye(3)*dt*1e-5)
            for k in range(1,calls+1):a.predict(sample(k*dt,k,0,0))
            outputs.append(a.latest().covariance)
        assert_allclose(outputs[0],outputs[1],atol=1e-18)  # stationary fixture only

    def test_default_invalid_snapshot_cannot_pass_controller_gate(self):
        result=adapter().predict(sample(.05,1))
        self.assertFalse(result.localisation_valid)
        self.assertIn('prediction_only',result.provenance)
        policy=InterfacePolicy('map','base',.2,.1,1e-8,lambda previous,current:True)
        with self.assertRaisesRegex(ValueError,'localisation_invalid'):
            validate_estimate(result,policy,now=10.06,epoch='test-epoch',expected_state_time=.05)


class ActualMuJoCoWheelTests(unittest.TestCase):
    def model(self):
        return mujoco.MjModel.from_xml_path(str(SCENE))

    def test_named_mapping_actual_qvel_not_ctrl_or_qpos(self):
        m=self.model();reader=WheelReader(m);d=mujoco.MjData(m)
        expected=tuple(int(m.jnt_dofadr[m.joint(n).id]) for n in ('wheel_left','wheel_right'))
        self.assertEqual(reader.dofs,expected)
        d.qvel[list(reader.dofs)]=[1,3];d.ctrl[:]=[-6,-6]
        class RestrictedData:
            qvel=d.qvel
            time=d.time
            def __getattr__(self,name):raise AssertionError('Forbidden ground truth/target read: '+name)
        value=reader.read(RestrictedData(),sample_monotonic=time.monotonic(),sequence=0,epoch='owner')
        self.assertEqual((value.left_rad_s,value.right_rad_s),(1,3))
        copy=pickle.loads(pickle.dumps(value));d.qvel[:]=99
        self.assertEqual((copy.left_rad_s,copy.right_rad_s),(1,3))
        self.assertFalse(adapter(copy).latest().localisation_valid)

    def test_invalid_mapping_rejected(self):
        m=self.model();m.actuator_trnid[m.actuator('wheel_left').id,0]=m.joint('wheel_right').id
        with self.assertRaises(ValueError):WheelReader(m)

    def test_wrong_owner_thread_rejected_before_access(self):
        reader=WheelReader(self.model());errors=[]
        def attempt():
            try:reader.read(None,sample_monotonic=10,sequence=0,epoch='test')
            except RuntimeError as error:errors.append(str(error))
        thread=threading.Thread(target=attempt);thread.start();thread.join(2)
        self.assertEqual(len(errors),1)

    def test_wrong_owner_process_rejected_before_access(self):
        reader=WheelReader(self.model())
        with patch('simulation.ekf_wheel_adapter.os.getpid',return_value=-1):
            with self.assertRaises(RuntimeError):
                reader.read(None,sample_monotonic=10,sequence=0,epoch='test')

    def test_actual_reverse_rotation_and_curved_wheel_response(self):
        for targets in [(-3.,-3.),(-2.,2.),(1.,3.)]:
            m=self.model();d=mujoco.MjData(m);reader=WheelReader(m)
            for _ in range(1000):mujoco.mj_step(m,d)
            d.ctrl[[m.actuator(n).id for n in ('wheel_left','wheel_right')]]=targets
            for _ in range(200):mujoco.mj_step(m,d)
            measured=reader.read(d,sample_monotonic=time.monotonic(),sequence=0,epoch='test')
            self.assertEqual(np.sign(measured.left_rad_s),np.sign(targets[0]))
            self.assertEqual(np.sign(measured.right_rad_s),np.sign(targets[1]))
            velocity=body_velocity(measured,radius_m=.033,track_m=.160)
            if targets[0]==targets[1]:self.assertLess(velocity[0],0)
            else:self.assertGreater(velocity[1],0)
            self.assertFalse(any(w.number for w in d.warning))

    def test_prescribed_wheel_prediction_timestep_diagnostic(self):
        results=[]
        for dt in (.002,.001):
            m=self.model();m.opt.timestep=dt;d=mujoco.MjData(m);reader=WheelReader(m)
            for _ in range(round(2/dt)):mujoco.mj_step(m,d)
            seed=reader.read(d,sample_monotonic=time.monotonic(),sequence=0,epoch=str(dt))
            a=adapter(seed,prediction_dt=dt,process_noise_per_call=np.zeros((3,3)))
            ids=[m.actuator(n).id for n in ('wheel_left','wheel_right')]
            d.ctrl[ids]=[3.,3.]  # prescribed test fixture, never autonomous feedback
            for k in range(1,round(.4/dt)+1):
                mujoco.mj_step(m,d)
                snap=a.predict(reader.read(d,sample_monotonic=time.monotonic(),sequence=k,epoch=str(dt)))
            self.assertFalse(any(w.number for w in d.warning))
            self.assertTrue(np.isfinite(snap.pose).all())
            self.assertGreater(snap.pose[0],0)
            self.assertFalse(snap.localisation_valid)
            self.assertAlmostEqual(snap.state_time-seed.state_time,.4)
            results.append(snap.pose.copy())
        print('\nPrediction-only timestep diagnostic (.002/.001 s), poses:',
              [p.tolist() for p in results], 'difference:',(results[1]-results[0]).tolist())
        # Report sensitivity; do not impose an invented accuracy/convergence bound.


if __name__=='__main__':
    unittest.main()
