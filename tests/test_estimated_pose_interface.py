"""Deterministic mocks/synthetic limits; NOT validated deployment settings."""
from dataclasses import replace
import unittest

import numpy as np

from control.command_gateway import CommandGateway
from control.estimated_pose_interface import (
    ClearanceGrant, EstimatedPoseInterface, InterfacePolicy, PoseEstimate,
)
from control.lqr_controller import wrap
from control.staged_terminal_controller import StagedTrajectoryTerminalController
from control.terminal_controller import TerminalParameters
from control.trajectory import Reference


class FakeClock:
    now = 10.
    def __call__(self): return self.now


class FakeActuator:
    def __init__(self): self.writes = []; self.fail_zero = False
    def __call__(self, wheels):
        if self.fail_zero and wheels == (0., 0.): raise OSError('synthetic zero failure')
        self.writes.append(wheels)


class SpySupervisor(StagedTrajectoryTerminalController):
    def __init__(self, goal, end_time):
        super().__init__(goal, end_time)
        self.calls = 0
        self.action = None

    def command(self, pose, reference, time):
        self.calls += 1
        if self.action is not None: return self.action(pose, reference, time)
        return super().command(pose, reference, time)


def fixture_quality(previous, snapshot):
    # Deliberately synthetic: normalised variance/jump limits for exercising the
    # policy boundary, NOT a statistically calibrated innovation gate.
    if max(np.diag(snapshot.covariance)) > .01: return False
    if previous is not None:
        if np.linalg.norm(snapshot.pose[:2]-previous.pose[:2]) > .25: return False
        if abs(wrap(snapshot.pose[2]-previous.pose[2])) > .25: return False
    return True


class EstimatedPoseInterfaceTests(unittest.TestCase):
    def setUp(self): self.reset_interface()

    def reset_interface(self, terminal_start=0.):
        self.clock = FakeClock(); self.writer = FakeActuator(); self.instances = []
        self.gateway = CommandGateway(self.writer, max_lease_s=.2, clock=self.clock)
        self.policy = InterfacePolicy('map', 'base', .2, .15, 1e-8, fixture_quality)
        def factory():
            instance = SpySupervisor(np.zeros(3), terminal_start)
            self.instances.append(instance)
            return instance
        self.interface = EstimatedPoseInterface(self.gateway, factory, self.policy,
            goal=np.zeros(3), task_id='fixture-task', terminal_start_s=terminal_start, clock=self.clock)
        self.assertTrue(self.interface.arm(epoch='fixture-epoch', state_time_origin=0.).accepted)

    def snapshot(self, k=0, pose=(0., 0., 0.), **changes):
        fields = dict(pose=np.asarray(pose), covariance=np.eye(3)*1e-4,
            state_time=k*.05, sample_monotonic=10.+k*.05, sequence=k,
            epoch='fixture-epoch', frame_id='map', body_frame_id='base', localisation_valid=True)
        fields.update(changes)
        return PoseEstimate(**fields)

    def grant(self, **changes):
        fields = dict(grant_id='fixture-grant', task_id='fixture-task', frame_id='map',
            estimator_epoch='fixture-epoch', valid_from=9., expires_at=20.,
            approved=True, covers_terminal_manoeuvre=True)
        fields.update(changes)
        return ClearanceGrant(**fields)

    def reference(self, t): return Reference(np.zeros(3), np.zeros(2))

    def step(self, k=0, **changes):
        self.clock.now = 10.+k*.05
        return self.interface.step(self.snapshot(k, **changes), self.reference, clearance=self.grant())

    def assert_fault(self, result):
        self.assertFalse(result.accepted)
        self.assertFalse(result.armed)
        self.assertTrue(result.zero_dispatch_confirmed)
        self.assertIsNone(self.interface.supervisor)
        self.assertEqual(self.writer.writes[-1], (0., 0.))

    def test_snapshot_is_a_detached_immutable_copy(self):
        pose = np.zeros(3); covariance = np.eye(3)*1e-4
        snapshot = self.snapshot(pose=pose, covariance=covariance)
        pose[0] = 9.; covariance[0, 0] = 9.
        self.assertEqual(snapshot.pose[0], 0.)
        self.assertEqual(snapshot.covariance[0, 0], 1e-4)
        with self.assertRaises(ValueError): snapshot.pose[0] = 1.
        with self.assertRaises(ValueError): snapshot.covariance.setflags(write=True)
        with self.assertRaises(AttributeError): snapshot.sequence = 2

    def test_fresh_identical_coordinates_can_qualify_estimated_stop(self):
        for k in range(22): self.assertTrue(self.step(k).accepted)
        self.assertTrue(self.interface.completed)
        self.assertEqual(self.instances[0].calls, 22)
        self.assertEqual(self.writer.writes[-1], (0., 0.))

    def test_duplicate_cannot_advance_dwell_and_rearm_uses_new_instance(self):
        self.step(0); self.step(1)
        old = self.instances[0]
        quiet = old.terminal.final.quiet
        self.clock.now = 10.1
        result = self.interface.step(self.snapshot(1), self.reference, clearance=self.grant())
        self.assert_fault(result)
        self.assertEqual(old.calls, 2)
        self.assertEqual(old.terminal.final.quiet, quiet)
        self.assertFalse(self.interface.completed)
        self.assertTrue(self.interface.arm(epoch='fixture-epoch', state_time_origin=.1).accepted)
        fresh = self.instances[-1]
        self.assertIsNot(fresh, old)
        result = self.interface.step(self.snapshot(2), self.reference, clearance=self.grant())
        self.assertTrue(result.accepted)
        self.assertEqual(fresh.terminal.final.quiet, 0.)

    def test_out_of_order_and_skipped_slots_disarm(self):
        for changes in (dict(sequence=0), dict(state_time=0.), dict(sample_monotonic=10.),
                        dict(state_time=.1), dict(sequence=True)):
            with self.subTest(changes=changes):
                self.reset_interface(); self.step(0); self.clock.now = 10.05
                self.assert_fault(self.interface.step(self.snapshot(1, **changes), self.reference, clearance=self.grant()))

    def test_stale_future_nonfinite_time_and_wrong_epoch(self):
        cases = [(dict(sample_monotonic=9.7),10.), (dict(sample_monotonic=10.1),10.),
                 (dict(state_time=float('nan')),10.), (dict(epoch='other'),10.),
                 (dict(sample_monotonic=float('inf')),10.)]
        for changes, now in cases:
            with self.subTest(changes=changes):
                self.reset_interface(); self.clock.now = now
                self.assert_fault(self.interface.step(self.snapshot(**changes), self.reference, clearance=self.grant()))

    def test_malformed_nonfinite_pose_covariance_and_frames(self):
        asymmetric = np.eye(3)*1e-4; asymmetric[0, 1] = .01
        cases = [dict(pose=[0.,0.]), dict(pose=[np.nan,0.,0.]),
                 dict(covariance=np.eye(2)), dict(covariance=np.eye(3)*np.nan),
                 dict(covariance=np.diag([-.1,.01,.01])), dict(covariance=asymmetric),
                 dict(frame_id='camera'), dict(body_frame_id='camera'),
                 dict(localisation_valid=False)]
        for changes in cases:
            with self.subTest(changes=changes):
                self.reset_interface()
                self.assert_fault(self.interface.step(self.snapshot(**changes), self.reference, clearance=self.grant()))
        self.reset_interface()
        self.assert_fault(self.interface.step(object(), self.reference, clearance=self.grant()))

    def test_quality_policy_controls_uncertainty_and_pose_jump(self):
        self.assert_fault(self.interface.step(self.snapshot(covariance=np.eye(3)*.1), self.reference, clearance=self.grant()))
        self.reset_interface(); self.step(0)
        self.assert_fault(self.step(1, pose=(.3,0.,0.)))

    def test_heading_wrap_is_not_an_unexpected_jump(self):
        self.assertTrue(self.step(0, pose=(0.,0.,np.pi-.01)).accepted)
        self.assertTrue(self.step(1, pose=(0.,0.,-np.pi+.01)).accepted)

    def test_snapshot_creation_and_reference_errors_are_inside_boundary(self):
        def broken_snapshot(): return self.snapshot(pose=['not-a-number'])
        self.assert_fault(self.interface.step(broken_snapshot, self.reference, clearance=self.grant()))
        for provider in (lambda t: None, lambda t: Reference(np.zeros(2), np.zeros(2)),
                         lambda t: Reference(np.zeros(3), np.array([np.inf,0.]))):
            self.reset_interface()
            self.assert_fault(self.interface.step(self.snapshot(), provider, clearance=self.grant()))
        self.reset_interface()
        def raises(t): raise RuntimeError('synthetic reference failure')
        self.assert_fault(self.interface.step(self.snapshot(), raises, clearance=self.grant()))

    def test_controller_exception_discards_partially_modified_instance(self):
        old = self.instances[0]
        def fails(*args):
            old.partial_mutation = True
            raise RuntimeError('synthetic controller error')
        old.action = fails
        self.assert_fault(self.step())
        self.assertTrue(old.partial_mutation)
        self.interface.factory = lambda: old
        result = self.interface.arm(epoch='fixture-epoch', state_time_origin=0.)
        self.assert_fault(result)
        self.assertIn('supervisor_must_be_fresh', result.reason)

    def test_malformed_controller_outputs_attempt_zero(self):
        bad = [None, (np.zeros(2), np.zeros(2), 'coast'),
            (np.zeros(2), [np.nan,0.], 'coast', False),
            ([0.,0.], [7.,0.], 'approach', False),
            ([1.,0.], [0.,0.], 'approach', False),
            ([0.,0.], [0.,0.], 'unknown', False),
            ([0.,0.], [0.,0.], 'fault', False),
            ([.033,0.], [1.,1.], 'coast', False),
            ([0.,0.], [0.,0.], 'tracking_lqr', False)]
        for output in bad:
            with self.subTest(output=output):
                self.reset_interface()
                self.instances[0].action = lambda *args: output
                self.assert_fault(self.step())

    def test_tracking_prefix_needs_no_terminal_grant(self):
        self.reset_interface(terminal_start=.1)
        self.assertTrue(self.interface.step(self.snapshot(), self.reference).accepted)
        self.clock.now = 10.05
        self.assertTrue(self.interface.step(self.snapshot(1), self.reference).accepted)
        self.clock.now = 10.1
        self.assert_fault(self.interface.step(self.snapshot(2), self.reference))
        self.assertEqual(self.instances[0].calls, 2)  # no terminal invocation

    def test_unknown_expired_revoked_or_wrong_clearance_blocks_invocation(self):
        grants = [None, self.grant(approved=False), self.grant(revoked=True),
                  self.grant(covers_terminal_manoeuvre=False), self.grant(expires_at=10.),
                  self.grant(valid_from=11.), self.grant(task_id='other'),
                  self.grant(frame_id='other'), self.grant(estimator_epoch='other'),
                  self.grant(grant_id='')]
        for grant in grants:
            with self.subTest(grant=grant):
                self.reset_interface()
                self.assert_fault(self.interface.step(self.snapshot(), self.reference, clearance=grant))
                self.assertEqual(self.instances[0].calls, 0)

    def test_revocation_on_next_tick_zeroes_and_does_not_resume(self):
        self.step(0); old = self.instances[0]; self.clock.now = 10.05
        self.assert_fault(self.interface.step(self.snapshot(1), self.reference, clearance=self.grant(revoked=True)))
        self.assertEqual(old.calls, 1)
        self.clock.now = 10.1
        self.assert_fault(self.interface.step(self.snapshot(2), self.reference, clearance=self.grant()))
        self.assertEqual(old.calls, 1)

    def test_clearance_deadline_bounds_command_and_expiry_does_not_rearm(self):
        result = self.interface.step(self.snapshot(), self.reference, clearance=self.grant(expires_at=10.075))
        self.assertTrue(result.accepted)
        self.assertEqual(self.gateway.deadline, 10.075)
        self.clock.now = 10.075; self.gateway.poll()
        self.assert_fault(self.interface.step(self.snapshot(1), self.reference, clearance=self.grant()))

    def test_slow_provider_or_controller_cannot_renew_an_expired_lease(self):
        self.instances[0].action = lambda *args: (np.array([.033,0.]), np.array([1.,1.]), 'approach', False)
        self.assertTrue(self.step(0).accepted)
        self.clock.now = 10.05
        def slow(*args):
            self.clock.now += .11  # old lease expires, current estimate remains fresh
            return np.array([.033,0.]), np.array([1.,1.]), 'approach', False
        self.instances[0].action = slow
        self.assert_fault(self.interface.step(self.snapshot(1), self.reference, clearance=self.grant()))
        self.assertEqual(self.writer.writes.count((1.,1.)), 1)
        self.reset_interface()
        def very_slow(t):
            self.clock.now += .3
            return self.reference(t)
        self.assert_fault(self.interface.step(self.snapshot(), very_slow, clearance=self.grant()))

    def test_failed_zero_dispatch_reports_stop_unconfirmed(self):
        self.step(0); self.writer.fail_zero = True
        result = self.interface.step(None, self.reference, clearance=self.grant())
        self.assertFalse(result.armed)
        self.assertFalse(result.zero_dispatch_confirmed)
        self.assertFalse(result.receipt.zero_written)
        self.assertIn('stop_unconfirmed', result.reason)

    def test_transport_failure_cannot_claim_a_zero_receipt(self):
        class BrokenGateway:
            def poll(self): raise ConnectionError('synthetic transport loss')
            def stop(self, reason): raise ConnectionError('synthetic transport loss')
        self.interface.gateway = BrokenGateway()
        result = self.step()
        self.assertFalse(result.zero_dispatch_confirmed)
        self.assertIn('stop_unconfirmed', result.reason)

    def test_disturbance_after_stop_revokes_completion_without_recovery_motion(self):
        for k in range(22): self.step(k)
        self.assertTrue(self.interface.completed)
        self.assert_fault(self.step(22, pose=(.03,0.,0.)))
        self.assertFalse(self.interface.completed)

    def test_mismatched_supervisor_metadata_cannot_arm(self):
        self.interface.disarm()
        self.interface.factory = lambda: SpySupervisor(np.zeros(3), 1.)
        result = self.interface.arm(epoch='fixture-epoch', state_time_origin=0.)
        self.assert_fault(result)
        self.assertIn('supervisor_contract_mismatch', result.reason)

    def test_nonmonotonic_interface_clock_disarms(self):
        self.clock.now = 9.
        self.assert_fault(self.interface.step(self.snapshot(), self.reference, clearance=self.grant()))

    def test_no_unexplained_default_quality_or_bad_policy(self):
        with self.assertRaises(ValueError):
            replace(self.policy, quality_check=None)
        with self.assertRaises(ValueError):
            replace(self.policy, clock_grid_tolerance_s=.05)


if __name__ == '__main__': unittest.main()
