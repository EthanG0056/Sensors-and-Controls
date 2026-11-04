"""Prescribed-command MuJoCo actuator diagnostic; no EKF/navigation feedback.

Run headlessly with the existing development Python. --no-save prints results only;
default output is a NEW run below simulation/mujoco_gateway_results. Never writes
historical evidence. Command/lease/response thresholds here are diagnostic fixtures,
not calibrated deployment or physical stopping requirements. No passive viewer.
"""
from dataclasses import asdict
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import multiprocessing
import os
import queue
import sys
import threading
import time
import uuid

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from control import command_gateway as gateway_module
from control.command_gateway import CommandGateway, GatewayClient, WheelCommandLease, WHEEL_LIMIT

SCENE = ROOT/'simulation/robotis_tb3/scene_turtlebot3_burger_gain_test.xml'
RESULTS_ROOT = ROOT/'simulation/mujoco_gateway_results'
NAMES = ('wheel_left', 'wheel_right')
FORWARD = (0.10/.033, 0.10/.033)
TURN = (-2., 2.)


def _owner(connection, events, scene, failure_event, pause_event, max_runtime_s):
    """Only this process/thread creates, reads, steps or writes model/data.

    Uses the ORIGINAL gateway state machine. This loop adds MuJoCo stepping to
    its existing RPC protocol; it adds no alternative validation/expiry policy.
    Test-only events inject writer failure or freeze PHYSICS, not the gateway clock.
    """
    gateway = None
    sample = None
    try:
        model = mujoco.MjModel.from_xml_path(str(scene))
        data = mujoco.MjData(model)
        ids = [model.actuator(n).id for n in NAMES]
        joints = [model.joint(n).id for n in NAMES]
        dofs = [int(model.jnt_dofadr[j]) for j in joints]
        if (not np.array_equal(model.actuator_trnid[ids, 0], joints)
                or not np.all(model.actuator_ctrllimited[ids])
                or not np.allclose(model.actuator_ctrlrange[ids], [[-WHEEL_LIMIT, WHEEL_LIMIT]]*2)
                or not np.allclose(model.actuator_gainprm[ids, 0], 1.)):
            raise ValueError('Unexpected named wheel transmission, limits or experimental gain')
        adr = int(model.jnt_qposadr[model.joint('base_joint').id])
        dt = float(model.opt.timestep)
        wall_origin = time.monotonic()
        step_index = 0

        def state():
            w, x, y, z = data.qpos[adr+3:adr+7]
            return dict(owner_pid=os.getpid(), wall_monotonic=time.monotonic(),
                sim_time=float(data.time), step_index=step_index,
                controls=data.ctrl[ids].tolist(), wheel_rad_s=data.qvel[dofs].tolist(),
                base_xy_m=data.qpos[adr:adr+2].tolist(),
                base_yaw_rad=float(np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))))

        def write(wheels):
            # Failure injection is at the REAL backend boundary, explicitly labelled.
            before = state()
            try:
                if failure_event.is_set():
                    raise OSError('TEST-ONLY injected MuJoCo-writer failure')
                data.ctrl[ids] = wheels
                if tuple(data.ctrl[ids]) != tuple(wheels):
                    raise OSError('Actuator control readback mismatch')
            except Exception as exc:
                before['wall_monotonic'] = time.monotonic()
                events.put(('write', dict(**before, requested=list(wheels),
                    success=False, actual_controls=data.ctrl[ids].tolist(), error=str(exc))))
                raise
            before['wall_monotonic'] = time.monotonic()
            events.put(('write', dict(**before, requested=list(wheels), success=True,
                                     actual_controls=data.ctrl[ids].tolist(), error=None)))

        gateway = CommandGateway(write, max_lease_s=1.2)  # diagnostic maximum, not deployment tuning
        # Initial settling is accelerated BEFORE accepting commands; no live lease.
        for _ in range(round(2./dt)):
            mujoco.mj_step(model, data)
            step_index += 1
        wall_origin = time.monotonic()
        files = [scene, scene.parent/'turtlebot3_burger_gain_test.xml', scene.parent/'LICENSE']
        files += [scene.parent/'assets'/f'{n}.stl' for n in ('burger_base','left_tire','right_tire','lds')]
        metadata = dict(owner_pid=os.getpid(), mujoco=mujoco.__version__, numpy=np.__version__,
            python=sys.version.split()[0], scene=str(scene.resolve()), timestep_s=dt,
            actuator_names=NAMES, actuator_ids=ids, joint_ids=joints, velocity_dofs=dofs,
            gains=model.actuator_gainprm[ids, 0].tolist(),
            control_limits=model.actuator_ctrlrange[ids].tolist(),
            state_source='prescribed open-loop wheel commands; no EKF or pose feedback',
            base_pose_use='evaluation only', initial_settle_sim_s=float(data.time),
            wall_origin=wall_origin, max_runtime_wall_s=max_runtime_s, max_lease_s=1.2,
            source_sha256={str(p.relative_to(scene.parent)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
            code_sha256={name: hashlib.sha256(path.read_bytes()).hexdigest() for name,path in
                [('control/command_gateway.py',Path(gateway_module.__file__)),
                 ('simulation/demo_mujoco_command_gateway.py',Path(__file__))]})
        events.put(('ready', metadata))

        def sample():
            status = gateway.poll()  # public API; expiry is never tied to simulation time
            events.put(('sample', dict(**state(), armed=status.armed, generation=status.generation,
                reason=status.reason, zero_written=status.zero_written,
                deadline=gateway.deadline, physics_paused=pause_event.is_set())))

        next_step = time.monotonic()
        sample()
        while time.monotonic()-wall_origin < max_runtime_s:
            gateway.poll()  # monotonic expiry is checked even while physics is paused
            if connection.poll(0):
                request_id, operation, args = connection.recv()
                if operation == 'arm': result = gateway.arm()
                elif operation == 'submit': result = gateway.submit(*args)
                elif operation == 'stop': result = gateway.stop(*args)
                elif operation == 'poll': result = gateway.poll()
                elif operation == 'close': result = gateway.stop('owner_close')
                else: result = gateway.stop('invalid_request')
                connection.send((request_id, result))
                if operation == 'close': break
            now = time.monotonic()
            if now >= next_step:
                if not pause_event.is_set():
                    mujoco.mj_step(model, data)
                    step_index += 1
                if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all() or any(w.number for w in data.warning):
                    raise RuntimeError('MuJoCo nonfinite state or numerical warning')
                sample()
                next_step += dt
            # At most one physics step per iteration, with expiry checked between
            # catch-up steps. Scheduling lag is logged, not hidden by clock resets.
            time.sleep(min(.001, max(0., next_step-time.monotonic())))
        else:
            gateway.stop('diagnostic_runtime_limit')
    except (EOFError, BrokenPipeError):
        pass
    except Exception as exc:
        events.put(('fatal', f'{type(exc).__name__}: {exc}'))
    finally:
        if gateway is not None:
            gateway.stop('owner_exit')
            if sample is not None: sample()
        events.put(('closed', os.getpid()))
        connection.close()


class OwnerSession:
    """Small diagnostic harness. Parent receives copies, never model/data objects."""
    def __init__(self, scene=SCENE, *, max_runtime_s=20.):
        if not math.isfinite(max_runtime_s) or max_runtime_s <= 0:
            raise ValueError('Positive diagnostic runtime required')
        ctx = multiprocessing.get_context('spawn')
        parent, child = ctx.Pipe()
        self.events = ctx.Queue()
        self.failure_event, self.pause_event = ctx.Event(), ctx.Event()
        self.samples, self.writes = [], []
        self.metadata, self.error = None, None
        self.condition = threading.Condition()
        self.finished = False
        self.process = ctx.Process(target=_owner, args=(child, self.events, Path(scene),
            self.failure_event, self.pause_event, max_runtime_s))
        self.process.start(); child.close()
        self.client = GatewayClient(parent, response_timeout_s=5.)
        self.collector = threading.Thread(target=self._collect, daemon=True)
        self.collector.start()
        try: self.wait(lambda: self.metadata is not None and bool(self.samples))
        except Exception:
            self.close()
            raise

    def _collect(self):
        while True:
            try: kind, value = self.events.get(timeout=.1)
            except queue.Empty:
                if not self.process.is_alive(): break
                continue
            with self.condition:
                if kind == 'ready': self.metadata = value
                elif kind == 'sample': self.samples.append(value)
                elif kind == 'write': self.writes.append(value)
                elif kind == 'fatal': self.error = value
                elif kind == 'closed': self.finished = True
                self.condition.notify_all()
                if kind == 'closed': return
        with self.condition:
            self.finished = True
            self.condition.notify_all()

    def wait(self, predicate, timeout=6.):
        deadline = time.monotonic()+timeout  # TEST/harness timeout, not a stopping SLA
        with self.condition:
            while not predicate():
                if self.error: raise RuntimeError(self.error)
                remaining = deadline-time.monotonic()
                if self.finished or remaining <= 0: raise TimeoutError('Diagnostic condition not observed')
                self.condition.wait(min(.1, remaining))

    def close(self):
        # Restore the injected test fault before shutdown; no physical-stop claim
        # is made while an intentionally failing writer cannot apply zero.
        self.failure_event.clear(); self.pause_event.clear()
        if self.process.is_alive():
            try: self.client.close()
            except Exception: pass
            self.process.join(5.)
        if self.process.is_alive():
            self.process.terminate(); self.process.join(5.)
            self.error = self.error or 'Owner required forced termination; stop unconfirmed'
        self.collector.join(3.)
        self.events.close(); self.events.join_thread()

    def __enter__(self): return self
    def __exit__(self, *args): self.close()


def run_diagnostic():
    """Collect actual MuJoCo evidence in memory; no files unless explicitly saved."""
    replies = []
    error = None
    with OwnerSession() as session:
        def record(label, receipt, **fields):
            replies.append(dict(label=label, wall_monotonic=time.monotonic(),
                                receipt=asdict(receipt), **fields))
            return receipt
        try:
            generation = record('first_arm', session.client.arm()).generation
            created = time.monotonic(); expiry = created+.8
            first = WheelCommandLease(generation, 10, FORWARD, created, expiry)
            record('forward', session.client.submit(first), expires_at=expiry, wheels=FORWARD)
            record('superseded', session.client.submit(WheelCommandLease(generation,10,(-2.,-2.),created,expiry)))
            record('wrong_generation', session.client.submit(WheelCommandLease(generation+100,11,(-2.,-2.),created,expiry)))
            record('outdated', session.client.submit(WheelCommandLease(generation,11,(-2.,-2.),created-.01,expiry)))
            silence_sim_time = session.samples[-1]['sim_time']
            # No further RPC/controller callback until owner telemetry proves expiry.
            session.wait(lambda: any(s['reason']=='command_expired' for s in session.samples))
            session.wait(lambda: any(w['success'] and w['requested']==[0.,0.] and w['wall_monotonic']>=expiry for w in session.writes))
            zero = next(w for w in session.writes if w['success'] and w['requested']==[0.,0.] and w['wall_monotonic']>=expiry)
            session.wait(lambda: session.samples[-1]['sim_time'] >= zero['sim_time']+.4)
            now = time.monotonic()
            record('late_after_expiry', session.client.submit(WheelCommandLease(generation,99,FORWARD,now,now+.5)))
            new_generation = record('explicit_rearm',session.client.arm()).generation
            now = time.monotonic(); second_expiry = now+.65
            record('turn',session.client.submit(WheelCommandLease(new_generation,0,TURN,now,second_expiry)), expires_at=second_expiry,wheels=TURN)
            record('old_generation_after_rearm',session.client.submit(WheelCommandLease(generation,100,FORWARD,now,second_expiry)))
            session.wait(lambda: any(s['generation'] != new_generation and s['reason']=='command_expired'
                                    and s['wall_monotonic']>=second_expiry for s in session.samples))
            session.wait(lambda: session.samples[-1]['wall_monotonic'] >= second_expiry+.4)
            record('explicit_stop', session.client.stop('diagnostic_complete'))
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
            silence_sim_time = locals().get('silence_sim_time')
    result = dict(metadata=session.metadata, samples=session.samples, writes=session.writes,
                  requests=replies, error=error or session.error, silence_sim_time=silence_sim_time)
    result['summary'] = summarise(result)
    return result


def summarise(result):
    requests = {r['label']: r for r in result['requests']}
    if result['error'] or not all(k in requests for k in ('forward','turn','explicit_stop')):
        return dict(passed=False, error=result['error'] or 'Incomplete diagnostic')
    first, turn = requests['forward'], requests['turn']
    expiry = first['expires_at']
    zero = next((w for w in result['writes'] if w['success'] and w['requested']==[0.,0.] and w['wall_monotonic']>=expiry), None)
    forward_write = next((w for w in result['writes'] if w['success'] and np.allclose(w['requested'],FORWARD)), None)
    after = [s for s in result['samples'] if zero is not None and s['sim_time']>=zero['sim_time']+.35 and s['wall_monotonic']<turn['wall_monotonic']]
    turn_samples = [s for s in result['samples'] if turn['wall_monotonic']<=s['wall_monotonic']<turn['expires_at']]
    checks = dict(
        forward_accepted=first['receipt']['accepted'], turn_accepted=turn['receipt']['accepted'],
        rejected_stale=all(not requests[k]['receipt']['accepted'] for k in ('superseded','wrong_generation','outdated','late_after_expiry','old_generation_after_rearm')),
        explicit_rearm=requests['explicit_rearm']['receipt']['armed'] and requests['explicit_rearm']['receipt']['generation']!=requests['first_arm']['receipt']['generation'],
        actual_zero_on_expiry=zero is not None and zero['actual_controls']==[0.,0.],
        stepped_during_silence=zero is not None and zero['sim_time']>result['silence_sim_time'],
        forward_response=zero is not None and min(zero['wheel_rad_s'])>.5,
        turn_response=any(s['wheel_rad_s'][0]<-.2 and s['wheel_rad_s'][1]>.2 for s in turn_samples),
        subsequent_deceleration=bool(after) and max(abs(v) for v in after[-1]['wheel_rad_s'])<.1,
        observed_movement=bool(forward_write is not None and zero is not None and np.linalg.norm(np.array(zero['base_xy_m'])-forward_write['base_xy_m'])>.005))
    return dict(passed=all(checks.values()), checks=checks,
        forward_wheel_at_zero_write_rad_s=zero['wheel_rad_s'] if zero else None,
        post_expiry_wheel_rad_s=after[-1]['wheel_rad_s'] if after else None,
        post_expiry_sim_interval_s=after[-1]['sim_time']-zero['sim_time'] if after else None,
        forward_displacement_m=float(np.linalg.norm(np.array(zero['base_xy_m'])-forward_write['base_xy_m'])) if zero and forward_write else None,
        first_zero_write_wall_lateness_s=zero['wall_monotonic']-expiry if zero else None,
        simulation_advance_during_silence_s=zero['sim_time']-result['silence_sim_time'] if zero else None,
        terminal_physical_stop_claim=False)


def save_result(result, output):
    output = Path(output).resolve()
    if output == RESULTS_ROOT.resolve() or RESULTS_ROOT.resolve() not in output.parents:
        raise ValueError('Output must be a NEW subdirectory of simulation/mujoco_gateway_results')
    output.mkdir(parents=True, exist_ok=False)
    with (output/'samples.csv').open('w',newline='') as stream:
        fields = list(result['samples'][0])
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(result['samples'])
    for name in ('metadata','summary','writes','requests'):
        (output/f'{name}.json').write_text(json.dumps(result[name],indent=2)+'\n')
    (output/'REPORT.md').write_text(
        '# Prescribed-command MuJoCo actuator diagnostic\n\n'
        'Actual experimental kv=1 TurtleBot3 actuators; no EKF or navigation feedback.\n\n'
        'The spawned worker owns all model/data access and stepping. Command expiry uses monotonic wall time; simulation time is logged independently. '
        'Initial 2 s simulation settling precedes live commands. No resets occur. '\
        'Zero control is not instantaneous physical rest. Scheduling latency is a measured diagnostic outcome, not a hard-real-time guarantee.\n\n'
        'See summary.json for checks and measured wheel response, writes.json for actual readback/receipts, and samples.csv for the complete trace. '
        'Prescribed forward wheel speed is 0.10/0.033 rad/s; turn commands are (-2,+2) rad/s. '
        'Leases of 0.8/0.65 s and response thresholds are synthetic diagnostic fixtures, not deployment settings.\n\n'
        'Reproduce from repository root with the existing development Python:\n\n'
        '```sh\npython -B simulation/demo_mujoco_command_gateway.py\npython -B -m unittest discover -s tests -p test_mujoco_command_gateway.py -v\n```\n\n'
        'This does not establish EKF navigation, obstacle clearance or hardware stopping. '
        'Tests separately inject a writer failure and a physics pause, explicitly labelled; those fixtures are not part of this successful live-write trace.\n')
    os.environ.setdefault('MPLCONFIGDIR','/tmp/tb3-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    origin = result['metadata']['wall_origin']
    samples = result['samples']; t = [s['wall_monotonic']-origin for s in samples]
    fig, axes = plt.subplots(2,1,figsize=(10,6),sharex=True)
    for index,side in enumerate(('left','right')):
        axes[0].plot(t,[s['controls'][index] for s in samples],label=side+' actual ctrl')
        axes[1].plot(t,[s['wheel_rad_s'][index] for s in samples],label=side+' measured wheel')
    for request in result['requests']:
        if 'expires_at' in request:
            for axis in axes: axis.axvline(request['expires_at']-origin,color='black',linestyle=':',linewidth=1)
    axes[0].set_ylabel('Actuator ctrl (rad/s target)'); axes[1].set_ylabel('Measured wheel speed (rad/s)')
    axes[1].set_xlabel('Monotonic wall elapsed (s)')
    for axis in axes: axis.grid(alpha=.3); axis.legend()
    fig.suptitle('MuJoCo actuator integration diagnostic — prescribed commands, kv=1')
    fig.tight_layout(); fig.savefig(output/'wheel_response.png',dpi=150); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--no-save',action='store_true')
    parser.add_argument('--output',type=Path,default=None)
    args = parser.parse_args()
    result = run_diagnostic()
    print(json.dumps(result['summary'],indent=2))
    if not args.no_save:
        output = args.output or RESULTS_ROOT/('run_'+uuid.uuid4().hex[:10])
        save_result(result,output); print('Saved:',output)
    return 0 if result['summary']['passed'] else 1


if __name__ == '__main__': raise SystemExit(main())
