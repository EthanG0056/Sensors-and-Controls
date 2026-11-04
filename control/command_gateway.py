"""Single-owner expiring wheel writes; no MuJoCo or hardware dependency.

CommandGateway is the synchronous state machine, not a watchdog by itself.
run_owner_loop must run independently of controller callbacks. See the contract
in ESTIMATED_STATE_INTERFACE.md for backend ownership and stopping limitations.
"""
from dataclasses import dataclass
import math
import time

import numpy as np

WHEEL_LIMIT = 6.67


@dataclass(frozen=True)
class WheelCommandLease:
    generation: int
    command_id: int
    wheels: tuple
    created_at: float
    expires_at: float

    def __post_init__(self):
        object.__setattr__(self, 'wheels', tuple(self.wheels))


@dataclass(frozen=True)
class GatewayStatus:
    accepted: bool
    armed: bool
    generation: int
    reason: str
    applied_wheels: tuple | None
    zero_written: bool | None


class CommandGateway:
    """Call from ONE actuator owner only. write_wheels receives a (left,right) tuple.

    A successful write means the backend call returned, not that physical motion
    stopped. An actuator writer must raise on failure; silent failures cannot be
    detected here. Configuration times are explicit deployment decisions.
    """
    def __init__(self, write_wheels, *, max_lease_s, clock=time.monotonic):
        if not callable(write_wheels) or not callable(clock):
            raise ValueError('Writer and clock must be callable')
        if not math.isfinite(max_lease_s) or max_lease_s <= 0:
            raise ValueError('Positive finite max_lease_s required')
        self.write_wheels, self.max_lease_s, self.clock = write_wheels, max_lease_s, clock
        self.generation = 0
        self.armed = False
        self.deadline = None
        self.last_id = -1
        self.last_created_at = -math.inf
        self.last_now = -math.inf
        self.applied_wheels = None
        self.zero_written = None
        self.reason = 'startup'
        self._disarm('startup')

    def _status(self, accepted=False, reason=None):
        return GatewayStatus(accepted, self.armed, self.generation,
                             reason or self.reason, self.applied_wheels, self.zero_written)

    def _write(self, wheels):
        try:
            self.write_wheels(wheels)
        except Exception as exc:
            self.applied_wheels = None
            self.zero_written = False if wheels == (0., 0.) else None
            self.reason += f'; write_failed:{type(exc).__name__}'
            return False
        self.applied_wheels = wheels
        self.zero_written = True if wheels == (0., 0.) else None
        return True

    def _disarm(self, reason):
        self.generation += 1  # revoke every outstanding command, including late replies
        self.armed = False
        self.deadline = None
        self.last_id = -1
        self.last_created_at = -math.inf
        self.reason = reason
        self._write((0., 0.))
        return self._status()

    def stop(self, reason='explicit_stop'):
        """Always attempt a real zero write, even if already disarmed."""
        return self._disarm(str(reason))

    def _now(self):
        now = float(self.clock())
        if not math.isfinite(now) or now < self.last_now:
            raise ValueError('Clock is not finite/monotonic')
        self.last_now = now
        return now

    def poll(self):
        try:
            now = self._now()
        except Exception:
            return self._disarm('invalid_clock')
        if self.armed and self.deadline is not None and now >= self.deadline:
            return self._disarm('command_expired')
        return self._status(True)

    def arm(self):
        checked = self.poll()
        if checked.reason.startswith('invalid_clock'):
            return checked
        result = self._disarm('arming')
        if result.zero_written is not True:
            return result
        self.armed = True
        self.reason = 'armed'
        return self._status(True)

    def submit(self, lease):
        checked = self.poll()
        if not self.armed:
            return self._status(False, 'not_armed; '+checked.reason)
        # Old-generation packets are inert: they must not stop a newly armed task.
        if isinstance(lease, WheelCommandLease) and lease.generation != self.generation:
            return self._status(False, 'wrong_generation')
        try:
            if not isinstance(lease, WheelCommandLease):
                raise ValueError('Lease type')
            if (type(lease.generation) is not int or type(lease.command_id) is not int
                    or lease.command_id < 0):
                raise ValueError('Identifiers')
            if lease.command_id <= self.last_id:
                return self._status(False, 'superseded_command')
            now = self.last_now
            created, expires = float(lease.created_at), float(lease.expires_at)
            if (not math.isfinite(created) or not math.isfinite(expires)
                    or created > now or expires <= now or expires <= created
                    or expires > created+self.max_lease_s):
                raise ValueError('Lease times')
            if created < self.last_created_at:
                return self._status(False, 'outdated_command')
            wheels = np.asarray(lease.wheels, dtype=float)
            if wheels.shape != (2,) or not np.isfinite(wheels).all() or np.max(abs(wheels)) > WHEEL_LIMIT:
                raise ValueError('Wheel command')
        except Exception:
            return self._disarm('invalid_command')
        self.reason = 'command_applied'
        if not self._write(tuple(float(v) for v in wheels)):
            return self._disarm('command_write_failed')
        self.last_id, self.last_created_at = lease.command_id, created
        self.deadline = expires
        return self._status(True)


class GatewayClient:
    """Single-caller synchronous RPC to the independent owner.

    After timeout/transport error the channel is closed, not reused: a late reply
    cannot be mistaken for a successful stop receipt. The owner handles EOF or
    lease expiry. Calls must be serialised by the integration layer.
    """
    def __init__(self, connection, *, response_timeout_s):
        if not math.isfinite(response_timeout_s) or response_timeout_s <= 0:
            raise ValueError('Positive response timeout required')
        self.connection = connection
        self.timeout = response_timeout_s
        self.request_id = 0
        self.broken = False

    def _request(self, operation, *args):
        if self.broken:
            raise ConnectionError('Gateway channel is unusable')
        self.request_id += 1
        try:
            self.connection.send((self.request_id, operation, args))
            if not self.connection.poll(self.timeout):
                raise TimeoutError('No actuator-owner receipt')
            request_id, result = self.connection.recv()
            if request_id != self.request_id or not isinstance(result, GatewayStatus):
                raise ConnectionError('Invalid actuator-owner receipt')
            return result
        except Exception:
            self.broken = True
            self.connection.close()
            raise

    def arm(self):
        return self._request('arm')

    def submit(self, lease):
        return self._request('submit', lease)

    def stop(self, reason='explicit_stop'):
        return self._request('stop', reason)

    def poll(self):
        return self._request('poll')

    def close(self):
        try:
            return self._request('close')
        finally:
            self.broken = True
            self.connection.close()


def run_owner_loop(connection, writer_factory, *, max_lease_s, poll_interval_s,
                   clock=time.monotonic, inhibit_event=None):
    """Independent worker entry point; factory constructs the backend IN this owner.

    Use a separate process for the fake-actuator demonstration. A real integration
    must keep this owner responsive and avoid other writers/unsynchronised MjData
    access. An optional external inhibit bypasses controller callbacks; clearing it
    does not re-arm motion. Scheduling/backend delays are not hard-time bounded.
    """
    if not math.isfinite(poll_interval_s) or poll_interval_s <= 0:
        raise ValueError('Positive poll interval required')
    gateway = CommandGateway(writer_factory(), max_lease_s=max_lease_s, clock=clock)
    inhibited = False
    try:
        while True:
            requested = inhibit_event is not None and inhibit_event.is_set()
            if requested and not inhibited:
                gateway.stop('external_inhibit')
            inhibited = requested
            gateway.poll()
            timeout = poll_interval_s
            if gateway.armed and gateway.deadline is not None:
                timeout = min(timeout, max(0., gateway.deadline-float(clock())))
            if not connection.poll(timeout):
                continue
            request_id, operation, args = connection.recv()
            # Recheck an inhibit after waiting and before any write/arm request.
            if inhibit_event is not None and inhibit_event.is_set():
                result = gateway.stop('external_inhibit')
            elif operation == 'arm':
                result = gateway.arm()
            elif operation == 'submit':
                result = gateway.submit(*args)
            elif operation == 'stop':
                result = gateway.stop(*args)
            elif operation == 'poll':
                result = gateway.poll()
            elif operation == 'close':
                result = gateway.stop('owner_close')
            else:
                result = gateway.stop('invalid_request')
            connection.send((request_id, result))
            if operation == 'close':
                break
    except (EOFError, BrokenPipeError, OSError):
        pass
    finally:
        gateway.stop('owner_exit')
        connection.close()
