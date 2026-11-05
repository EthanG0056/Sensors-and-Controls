"""Actual headless MuJoCo integration; all commands/thresholds are test fixtures.

One completed prescribed-command trace is shared for read-only assertions, not
shared mutable simulation state. Separate workers exercise explicit fault/pause
injections. Tests create NO evidence files and never use pose as control feedback.
"""
import os
import time
import unittest

import numpy as np

from control.command_gateway import WHEEL_LIMIT, WheelCommandLease
from simulation.demo_mujoco_command_gateway import FORWARD, TURN, NAMES, OwnerSession, run_diagnostic


class MuJoCoGatewayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = run_diagnostic()

    def test_diagnostic_completed_without_hidden_failure(self):
        self.assertIsNone(self.result['error'])
        self.assertTrue(self.result['summary']['passed'], self.result['summary'])

    def test_named_actuators_and_compiled_transmissions(self):
        metadata = self.result['metadata']
        self.assertEqual(tuple(metadata['actuator_names']), NAMES)
        self.assertEqual(len(set(metadata['actuator_ids'])), 2)
        self.assertEqual(len(set(metadata['velocity_dofs'])), 2)
        np.testing.assert_allclose(metadata['control_limits'], [[-6.67,6.67]]*2)
        np.testing.assert_allclose(metadata['gains'], [1.,1.])
        self.assertEqual(metadata['timestep_s'], .002)

    def test_single_worker_owns_all_model_data_and_writes(self):
        owner = self.result['metadata']['owner_pid']
        self.assertNotEqual(owner, os.getpid())
        self.assertEqual({s['owner_pid'] for s in self.result['samples']}, {owner})
        self.assertEqual({w['owner_pid'] for w in self.result['writes']}, {owner})

    def test_actual_nonzero_control_readback_and_limit_enforcement(self):
        for target in (FORWARD,TURN):
            writes = [w for w in self.result['writes'] if w['success'] and np.allclose(w['requested'], target)]
            self.assertTrue(writes)
            np.testing.assert_array_equal(writes[0]['actual_controls'], target)
        self.assertTrue(all(max(abs(v) for v in s['controls'])<=WHEEL_LIMIT for s in self.result['samples']))

    def test_measured_wheels_respond_and_base_moves(self):
        summary = self.result['summary']
        self.assertTrue(summary['checks']['forward_response'])
        self.assertTrue(summary['checks']['turn_response'])
        self.assertGreater(summary['forward_displacement_m'], .005)  # fixture regression bound
        self.assertTrue(any(s['wheel_rad_s'][0]<-.2 and s['wheel_rad_s'][1]>.2 for s in self.result['samples']))

    def test_expiry_zeros_actual_controls_without_new_callbacks(self):
        summary = self.result['summary']
        self.assertTrue(summary['checks']['actual_zero_on_expiry'])
        self.assertGreaterEqual(summary['first_zero_write_wall_lateness_s'], 0.)
        self.assertGreater(summary['simulation_advance_during_silence_s'], .1)
        forward = next(r for r in self.result['requests'] if r['label']=='forward')
        late = next(r for r in self.result['requests'] if r['label']=='late_after_expiry')
        intervening = [r['label'] for r in self.result['requests'] if
                       forward['wall_monotonic'] < r['wall_monotonic'] < late['wall_monotonic']]
        self.assertEqual(intervening, ['superseded','wrong_generation','outdated'])
        self.assertTrue(any(s['reason']=='command_expired' and s['controls']==[0.,0.] for s in self.result['samples']))

    def test_zero_control_and_physical_wheel_motion_are_distinguished(self):
        summary = self.result['summary']
        self.assertGreater(min(summary['forward_wheel_at_zero_write_rad_s']), .5)
        self.assertLess(max(abs(v) for v in summary['post_expiry_wheel_rad_s']), .1)
        self.assertGreaterEqual(summary['post_expiry_sim_interval_s'], .35)
        self.assertFalse(summary['terminal_physical_stop_claim'])

    def test_old_superseded_commands_do_not_restore_or_replace_motion(self):
        requests = {r['label']:r for r in self.result['requests']}
        for label in ('superseded','wrong_generation','outdated','late_after_expiry','old_generation_after_rearm'):
            self.assertFalse(requests[label]['receipt']['accepted'])
        self.assertEqual(requests['superseded']['receipt']['reason'], 'superseded_command')
        self.assertEqual(requests['outdated']['receipt']['reason'], 'outdated_command')
        np.testing.assert_allclose(requests['old_generation_after_rearm']['receipt']['applied_wheels'], TURN)

    def test_explicit_rearming_changes_generation(self):
        self.assertTrue(self.result['summary']['checks']['explicit_rearm'])
        self.assertTrue(self.result['summary']['checks']['turn_accepted'])


class MuJoCoGatewayFailureTests(unittest.TestCase):
    def test_out_of_limit_command_disarms_actual_actuators(self):
        with OwnerSession() as session:
            generation = session.client.arm().generation
            now = time.monotonic()
            self.assertTrue(session.client.submit(WheelCommandLease(generation,0,(2.,2.),now,now+.8)).accepted)
            session.wait(lambda: any(s['controls']==[2.,2.] for s in session.samples))
            now = time.monotonic()
            result = session.client.submit(WheelCommandLease(generation,1,(WHEEL_LIMIT+.01,0.),now,now+.5))
            self.assertFalse(result.armed)
            self.assertTrue(result.zero_written)
            session.wait(lambda: session.samples[-1]['controls']==[0.,0.])

    def test_injected_writer_failure_reports_unconfirmed_zero_and_retained_control(self):
        with OwnerSession() as session:
            generation = session.client.arm().generation
            now = time.monotonic()
            self.assertTrue(session.client.submit(WheelCommandLease(generation,0,(2.,2.),now,now+.8)).accepted)
            session.wait(lambda: session.samples[-1]['controls']==[2.,2.])
            session.failure_event.set()  # explicitly injected backend failure, NOT a fake actuator
            result = session.client.stop('injected_write_failure')
            self.assertFalse(result.zero_written)
            self.assertFalse(result.armed)
            self.assertIsNone(result.applied_wheels)
            session.wait(lambda: any(not w['success'] for w in session.writes))
            failed = next(w for w in reversed(session.writes) if not w['success'])
            self.assertEqual(failed['actual_controls'], [2.,2.])
            self.assertIn('TEST-ONLY',failed['error'])
            session.failure_event.clear()
            recovered = session.client.stop('restore_writer_and_zero')
            self.assertTrue(recovered.zero_written)
            session.wait(lambda: session.samples[-1]['controls']==[0.,0.])
            self.assertFalse(session.client.poll().armed)  # no automatic re-arm

    def test_physics_pause_cannot_extend_monotonic_lease(self):
        with OwnerSession() as session:
            session.pause_event.set()
            session.wait(lambda: session.samples[-1]['physics_paused'])
            paused_sim = session.samples[-1]['sim_time']
            generation = session.client.arm().generation
            now = time.monotonic(); expiry = now+.25
            self.assertTrue(session.client.submit(WheelCommandLease(generation,0,(2.,2.),now,expiry)).accepted)
            # No calls to gateway.poll from the parent; worker must expire while
            # simulation time is frozen. Frozen velocities are not deceleration.
            session.wait(lambda: session.samples[-1]['reason']=='command_expired')
            sample = session.samples[-1]
            self.assertEqual(sample['sim_time'],paused_sim)
            self.assertEqual(sample['controls'],[0.,0.])
            self.assertGreaterEqual(sample['wall_monotonic'],expiry)
            self.assertFalse(sample['armed'])
            session.pause_event.clear()
            session.wait(lambda: session.samples[-1]['sim_time']>paused_sim)


if __name__ == '__main__': unittest.main()
