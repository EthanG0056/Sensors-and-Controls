"""Unit tests and a real independent-process FAKE-actuator expiry demonstration.
All clock/TTL numbers below are synthetic test fixtures, not deployment settings.
"""
from functools import partial
import multiprocessing
import time
import unittest

from control.command_gateway import (
    CommandGateway, GatewayClient, WheelCommandLease, run_owner_loop,
)


class FakeClock:
    def __init__(self, now=10.):
        self.now = now

    def __call__(self):
        return self.now


class FakeActuator:
    def __init__(self):
        self.writes = []
        self.fail_zero = False
        self.fail_motion = False

    def __call__(self, wheels):
        if (wheels == (0., 0.) and self.fail_zero) or (wheels != (0., 0.) and self.fail_motion):
            raise OSError('synthetic backend failure')
        self.writes.append(wheels)


def process_writer_factory(events):
    # Constructed and invoked inside the independent actuator-owner process.
    def write(wheels):
        events.put((wheels, time.monotonic()))
    return write


class GatewayTests(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.writer = FakeActuator()
        self.gateway = CommandGateway(self.writer, max_lease_s=.2, clock=self.clock)
        self.generation = self.gateway.arm().generation

    def lease(self, command_id=0, wheels=(1., 2.), **changes):
        fields = dict(generation=self.generation, command_id=command_id, wheels=wheels,
                      created_at=self.clock.now, expires_at=self.clock.now+.1)
        fields.update(changes)
        return WheelCommandLease(**fields)

    def test_startup_and_explicit_arm_require_zero_write(self):
        self.assertEqual(self.writer.writes, [(0., 0.), (0., 0.)])
        self.assertTrue(self.gateway.armed)
        result = self.gateway.submit(self.lease(wheels=(6.67, -6.67)))
        self.assertTrue(result.accepted)
        self.assertEqual(result.applied_wheels, (6.67, -6.67))
        self.assertIsNone(result.zero_written)

    def test_exact_expiry_disarms_and_revokes_late_commands(self):
        self.gateway.submit(self.lease())
        deadline = self.gateway.deadline
        self.clock.now = deadline
        result = self.gateway.poll()
        self.assertFalse(result.armed)
        self.assertTrue(result.zero_written)
        self.assertEqual(result.reason, 'command_expired')
        late = self.gateway.submit(self.lease(command_id=1))
        self.assertFalse(late.accepted)
        self.assertEqual(self.writer.writes[-1], (0., 0.))

    def test_superseded_and_outdated_commands_do_not_renew(self):
        self.gateway.submit(self.lease(command_id=2))
        deadline, count = self.gateway.deadline, len(self.writer.writes)
        self.clock.now += .01
        duplicate = self.gateway.submit(self.lease(command_id=2))
        self.assertEqual(duplicate.reason, 'superseded_command')
        old = self.gateway.submit(self.lease(command_id=3, created_at=9.99, expires_at=10.1))
        self.assertEqual(old.reason, 'outdated_command')
        self.assertEqual(self.gateway.deadline, deadline)
        self.assertEqual(len(self.writer.writes), count)

    def test_invalid_commands_attempt_zero(self):
        cases = [dict(wheels=(1.,)), dict(wheels=(float('nan'), 0.)),
                 dict(wheels=(float('inf'), 0.)), dict(wheels=(6.671, 0.)),
                 dict(created_at=10.01), dict(expires_at=10.),
                 dict(expires_at=10.3), dict(command_id=True)]
        for changes in cases:
            with self.subTest(changes=changes):
                self.generation = self.gateway.arm().generation
                self.assertTrue(self.gateway.submit(self.lease()).accepted)
                bad = self.gateway.submit(self.lease(command_id=changes.get('command_id', 1),
                    **{k:v for k,v in changes.items() if k!='command_id'}))
                self.assertFalse(bad.accepted)
                self.assertFalse(bad.armed)
                self.assertTrue(bad.zero_written)
        self.gateway.arm()
        self.assertFalse(self.gateway.submit(object()).armed)

    def test_rearm_rejects_old_generation_without_stopping_new_motion(self):
        old = self.lease(command_id=100)
        self.gateway.stop('test_fault')
        self.generation = self.gateway.arm().generation
        self.assertTrue(self.gateway.submit(self.lease()).accepted)
        count = len(self.writer.writes)
        result = self.gateway.submit(old)
        self.assertFalse(result.accepted)
        self.assertTrue(result.armed)
        self.assertEqual(result.reason, 'wrong_generation')
        self.assertEqual(len(self.writer.writes), count)

    def test_failed_zero_is_reported_and_blocks_arming(self):
        self.gateway.submit(self.lease())
        self.writer.fail_zero = True
        result = self.gateway.stop('test_fault')
        self.assertFalse(result.zero_written)
        self.assertIsNone(result.applied_wheels)
        self.assertFalse(self.gateway.arm().armed)
        self.writer.fail_zero = False
        self.assertTrue(self.gateway.arm().armed)

    def test_motion_write_failure_attempts_zero_and_disarms(self):
        self.writer.fail_motion = True
        result = self.gateway.submit(self.lease())
        self.assertFalse(result.armed)
        self.assertTrue(result.zero_written)
        self.assertIn('command_write_failed', result.reason)

    def test_backward_or_nonfinite_clock_disarms(self):
        self.gateway.submit(self.lease())
        self.clock.now = 9.
        self.assertFalse(self.gateway.poll().armed)
        self.assertEqual(self.writer.writes[-1], (0., 0.))
        self.clock.now = float('nan')
        self.assertFalse(self.gateway.arm().armed)

    def test_zero_command_ack_is_not_a_physical_stop_claim(self):
        result = self.gateway.submit(self.lease(wheels=(0., 0.)))
        self.assertTrue(result.zero_written)
        self.assertTrue(result.armed)  # active zero lease, not measured rest

    def test_rpc_timeout_closes_channel_instead_of_accepting_late_receipt(self):
        class SilentConnection:
            closed = False
            def send(self, value): pass
            def poll(self, timeout): return False
            def close(self): self.closed = True
        connection = SilentConnection()
        client = GatewayClient(connection, response_timeout_s=.1)
        with self.assertRaises(TimeoutError): client.arm()
        self.assertTrue(connection.closed)
        with self.assertRaises(ConnectionError): client.stop()


class IndependentOwnerTests(unittest.TestCase):
    def start_owner(self, max_lease_s, inhibit=None):
        ctx = multiprocessing.get_context('spawn')
        parent, child = ctx.Pipe()
        events = ctx.Queue()
        process = ctx.Process(target=run_owner_loop,
            args=(child, partial(process_writer_factory, events)),
            kwargs=dict(max_lease_s=max_lease_s, poll_interval_s=.01, inhibit_event=inhibit))
        process.start(); child.close()
        client = GatewayClient(parent, response_timeout_s=5.)
        def cleanup():
            if process.is_alive():
                try: client.close()
                except Exception: pass
                process.join(5.)
            if process.is_alive(): process.terminate(); process.join(5.)
            events.close(); events.join_thread()
        self.addCleanup(cleanup)
        return client, events, process

    def wait_for_write(self, events, wanted):
        deadline = time.monotonic()+5.  # generous TEST timeout, not a real-time SLA
        while time.monotonic()<deadline:
            wheels, timestamp = events.get(timeout=max(.01, deadline-time.monotonic()))
            if wheels == wanted: return timestamp
        self.fail(f'No write {wanted}')

    def test_expiry_without_any_further_controller_or_poll_callback(self):
        client, events, process = self.start_owner(.3)
        generation = client.arm().generation
        now = time.monotonic(); expiry = now+.15
        self.assertTrue(client.submit(WheelCommandLease(generation, 0, (1., 2.), now, expiry)).accepted)
        self.wait_for_write(events, (1., 2.))
        # The parent sends NO requests until the child itself writes zero.
        zero_time = self.wait_for_write(events, (0., 0.))
        self.assertGreaterEqual(zero_time, expiry)
        self.assertTrue(process.is_alive())
        result = client.poll()
        self.assertFalse(result.armed)
        self.assertEqual(result.reason, 'command_expired')
        now = time.monotonic()
        self.assertFalse(client.submit(WheelCommandLease(generation, 1, (2., 2.), now, now+.1)).accepted)
        new = client.arm()
        self.assertTrue(new.armed)
        self.assertNotEqual(new.generation, generation)

    def test_external_inhibit_stops_independently_and_needs_rearm(self):
        inhibit = multiprocessing.get_context('spawn').Event()
        client, events, _ = self.start_owner(4., inhibit)
        generation = client.arm().generation
        now = time.monotonic()
        self.assertTrue(client.submit(WheelCommandLease(generation, 0, (1., 1.), now, now+3.)).accepted)
        self.wait_for_write(events, (1., 1.))
        inhibit.set()
        self.wait_for_write(events, (0., 0.))
        self.assertFalse(client.arm().armed)
        inhibit.clear()
        self.assertFalse(client.poll().armed)
        self.assertTrue(client.arm().armed)


if __name__ == '__main__': unittest.main()
