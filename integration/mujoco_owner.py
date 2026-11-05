"""Single authoritative MuJoCo owner; soft wall-clock expiry, not hard real time.

Only the spawned process accesses model/data/rendering. Detector/PnP and EKF run
in the receiver using copied RGB. Physics/render/native writer calls can still
block this owner: polling occurs around calls, never concurrently with MjData.
No bounded stopping time is claimed. Images retain PRE-render acquisition time.

The bounded measurement queue faults on overflow; samples are never overwritten
or relabelled fresh. Telemetry and evaluation queues are separate and best-effort:
losing either cannot authorise motion. Scoring truth is absent from Measurement.
One serial GatewayClient caller owns RPC. No auto-arm exists in the owner.
"""
from collections import deque
from dataclasses import dataclass
import math
import multiprocessing as mp
import os
from pathlib import Path
import queue
import threading
import time

import mujoco
import numpy as np

from control.command_gateway import CommandGateway, GatewayClient, WHEEL_LIMIT
from sensing.lidar import get_lidar_ranges
from simulation.ekf_wheel_adapter import WheelReader

SCENE = Path(__file__).resolve().parent/'scenes/scene_turtlebot3_navigation.xml'


def frozen_array(value, dtype=float):
    a = np.asarray(value, dtype=dtype)
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


@dataclass(frozen=True)
class Measurement:
    wheels: np.ndarray
    lidar: np.ndarray
    rgb: np.ndarray | None
    state_time: float
    sample_monotonic: float
    sequence: int
    epoch: str
    frame_id: str = 'map'
    body_frame_id: str = 'base'

    def __post_init__(self):
        object.__setattr__(self, 'wheels', frozen_array(self.wheels))
        object.__setattr__(self, 'lidar', frozen_array(self.lidar))
        if self.rgb is not None:
            object.__setattr__(self, 'rgb', frozen_array(self.rgb, np.uint8))

    def __reduce__(self):
        return (type(self), tuple(getattr(self, n) for n in self.__dataclass_fields__))


@dataclass(frozen=True)
class Evaluation:
    sequence: int
    state_time: float
    pose: tuple  # scoring only; never part of Measurement
    # R_WC maps OpenCV optical axes (right, down, forward) to world axes.
    # MuJoCo camera-local axes are right, up, backward: R_WC = R_WM diag(1,-1,-1).
    # These are the actual camera fields used to render this held simulation state,
    # including roll/pitch, not a planar reconstruction of pose.
    camera_world_position: tuple | None = None
    camera_to_world_rotation: tuple | None = None
    epoch: str = ''
    sample_monotonic: float | None = None

    def __post_init__(self):
        object.__setattr__(self, 'pose', tuple(float(v) for v in self.pose))
        if self.camera_world_position is not None:
            object.__setattr__(self, 'camera_world_position',
                               tuple(float(v) for v in np.asarray(self.camera_world_position).reshape(3)))
        if self.camera_to_world_rotation is not None:
            rotation = np.asarray(self.camera_to_world_rotation).reshape(3,3)
            object.__setattr__(self, 'camera_to_world_rotation',
                               tuple(tuple(float(v) for v in row) for row in rotation))


@dataclass(frozen=True)
class OwnerConfig:
    scene: str = str(SCENE)
    sample_time: float = .05
    max_lease_s: float = .25  # diagnostic configuration, not a safety guarantee
    queue_capacity: int = 4
    render_camera: bool = True
    epoch: str = 'navigation-diagnostic'
    max_runtime_s: float = 30.

    def __post_init__(self):
        if (not all(math.isfinite(x) and x > 0 for x in
                    (self.sample_time, self.max_lease_s, self.max_runtime_s))
                or type(self.queue_capacity) is not int or self.queue_capacity < 1
                or not isinstance(self.epoch, str) or not self.epoch):
            raise ValueError('Explicit positive owner timing/capacity and named epoch required')


def _offer(channel, item):
    try:
        channel.put_nowait(item)
        return True
    except queue.Full:
        return False


def marker_face_metadata(model, data):
    """Configured negative-local-X printed faces, transformed fully into world.

    Camera observations are camera-origin range/bearing to these points. No
    camera/body offset belongs in the landmark map; EKF models it separately.
    Call only in the simulation owner (independent fixtures may call in tests).
    """
    result={}
    for identity,name in ((0,'aruco_marker'),(1,'aruco_marker_1')):
        gid=model.geom(name).id
        if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_BOX:
            raise ValueError('Mapped marker must be a box with a negative-X printed face')
        centre=np.asarray(data.geom_xpos[gid],float)
        normal=-np.asarray(data.geom_xmat[gid],float).reshape(3,3)[:,0]
        face=centre+model.geom_size[gid,0]*normal
        if not np.isfinite(face).all() or model.geom_size[gid,0]<=0:
            raise ValueError('Invalid marker face geometry')
        result[identity]=dict(centre=centre.tolist(),face=face.tolist(),normal=normal.tolist())
    return result


def run_owner(connection, measurements, evaluations, events, config, pause, fail_write):
    """Entry point only; test events pause physics or inject a labelled writer fault."""
    gateway = renderer = None
    for channel in (measurements, evaluations, events):
        channel.cancel_join_thread()  # shutdown must not wait for an absent consumer
    try:
        model = mujoco.MjModel.from_xml_path(config.scene)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        reader = WheelReader(model)
        ids = [model.actuator(n).id for n in ('wheel_left', 'wheel_right')]
        if not np.allclose(model.actuator_ctrlrange[ids], [[-WHEEL_LIMIT, WHEEL_LIMIT]]*2):
            raise ValueError('Unexpected actuator limits')
        dt = float(model.opt.timestep)
        count = round(config.sample_time/dt)
        if count < 1 or not np.isclose(count*dt, config.sample_time, rtol=0, atol=1e-12):
            raise ValueError('Controller period must be an integer number of physics steps')
        camera = model.camera('front_camera').id
        width, height = int(model.vis.global_.offwidth), int(model.vis.global_.offheight)
        f = height/(2*np.tan(np.deg2rad(model.cam_fovy[camera])/2))
        k = [[f, 0, (width-1)/2], [0, f, (height-1)/2], [0, 0, 1]]
        rotation = np.zeros(9)
        mujoco.mju_quat2Mat(rotation, model.cam_quat[camera])
        forward = -rotation.reshape(3, 3)[:, 2]
        if abs(forward[2]) > 1e-8:
            raise ValueError('Planar EKF camera model requires a level camera')
        face_metadata = marker_face_metadata(model, data)
        landmarks = {i: value['face'][:2] for i,value in face_metadata.items()}
        sizes = {i: float(2*model.geom_size[model.geom(n).id, 1])
                 for i, n in [(0, 'aruco_marker'), (1, 'aruco_marker_1')]}
        adr = int(model.jnt_qposadr[model.joint('base_joint').id])

        def record(kind, **extra):
            _offer(events, (kind, dict(owner_pid=os.getpid(), wall=time.monotonic(),
                   sim_time=float(data.time), actual_controls=data.ctrl[ids].tolist(),
                   wheels=data.qvel[list(reader.dofs)].tolist(), **extra)))

        def write(wheels):
            try:
                if fail_write.is_set():
                    raise OSError('TEST-ONLY injected writer failure')
                data.ctrl[ids] = wheels
                if tuple(data.ctrl[ids]) != tuple(wheels):
                    raise OSError('MuJoCo actuator readback mismatch')
            except Exception as error:
                record('write', requested=list(wheels), success=False, error=str(error))
                raise
            record('write', requested=list(wheels), success=True, error=None)

        gateway = CommandGateway(write, max_lease_s=config.max_lease_s, clock=time.monotonic)
        if config.render_camera:
            renderer = mujoco.Renderer(model, height=height, width=width)
            renderer._scene_option.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False
        _offer(events, ('ready', dict(owner_pid=os.getpid(), physics_dt=dt,
            sample_time=config.sample_time, landmarks=landmarks, landmark_faces=face_metadata, marker_sizes=sizes,
            camera_matrix=k, camera_offset=model.cam_pos[camera, :2].tolist(),
            camera_yaw=float(np.arctan2(forward[1], forward[0])), epoch=config.epoch)))
        start = next_step = time.monotonic()
        next_telemetry = start
        steps = sequence = 0
        publication_fault = False
        while time.monotonic()-start < config.max_runtime_s:
            gateway.poll()
            if connection.poll(0):
                request_id, operation, args = connection.recv()
                if operation == 'arm' and publication_fault:
                    response = gateway.stop('measurement_overflow_requires_new_session')
                elif operation == 'arm': response = gateway.arm()
                elif operation == 'submit': response = gateway.submit(*args)
                elif operation == 'poll': response = gateway.poll()
                elif operation == 'stop': response = gateway.stop(*args)
                elif operation == 'close': response = gateway.stop('owner_close')
                else: response = gateway.stop('invalid_request')
                connection.send((request_id, response))
                if operation == 'close': break
            now = time.monotonic()
            if now >= next_step:
                if not pause.is_set():
                    mujoco.mj_step(model, data)
                    steps += 1
                    if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all() or any(w.number for w in data.warning):
                        raise RuntimeError('MuJoCo numerical failure')
                    if steps % count == 0 and not publication_fault:
                        acquired, simulation_time = time.monotonic(), float(data.time)
                        wheel = reader.read(data, sample_monotonic=acquired,
                                            sequence=sequence, epoch=config.epoch)
                        lidar = get_lidar_ranges(model, data).copy()
                        rgb = None
                        if renderer is not None:
                            gateway.poll()
                            renderer.update_scene(data, camera='front_camera')
                            rgb = renderer.render().copy()
                            gateway.poll()
                        packet = Measurement([wheel.left_rad_s, wheel.right_rad_s], lidar,
                                             rgb, simulation_time, acquired, sequence, config.epoch)
                        if not _offer(measurements, packet):
                            publication_fault = True
                            gateway.stop('measurement_queue_overflow')
                            record('fault', reason='measurement_queue_overflow')
                        else:
                            # Evaluation uses a distinct channel and cannot alter a packet.
                            w, x, y, z = data.qpos[adr+3:adr+7]
                            yaw = float(np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z)))
                            _offer(evaluations, Evaluation(sequence, simulation_time,
                                   (float(data.qpos[adr]), float(data.qpos[adr+1]), yaw),
                                   camera_world_position=tuple(data.cam_xpos[camera]),
                                   camera_to_world_rotation=data.cam_xmat[camera].reshape(3,3)
                                       @ np.diag([1., -1., -1.]),
                                   epoch=config.epoch, sample_monotonic=acquired))
                            sequence += 1
                next_step += dt
            if now >= next_telemetry:
                status = gateway.poll()
                record('status', reason=status.reason, generation=status.generation,
                       armed=status.armed, zero_written=status.zero_written)
                next_telemetry = now+.02
            time.sleep(min(.001, max(0., next_step-time.monotonic())))
        else:
            gateway.stop('owner_runtime_limit')
    except (EOFError, BrokenPipeError):
        pass
    except Exception as error:
        _offer(events, ('fatal', str(error)))
    finally:
        if gateway is not None:
            status = gateway.stop('owner_exit')
            _offer(events, ('exit_stop', status))
        if renderer is not None:
            renderer.close()
        _offer(events, ('closed', os.getpid()))
        connection.close()


class OwnerSession:
    """One RPC caller; collector consumes telemetry without estimator callbacks."""
    def __init__(self, config=OwnerConfig()):
        self.config = config
        ctx = mp.get_context('spawn')
        parent, child = ctx.Pipe()
        self.measurements = ctx.Queue(config.queue_capacity)
        self.evaluations = ctx.Queue(4)
        self.events = ctx.Queue(128)
        self.pause, self.fail_write = ctx.Event(), ctx.Event()
        self.client = GatewayClient(parent, response_timeout_s=2.)
        self.records = deque(maxlen=512)
        self.metadata = self.error = self.exit_stop = None
        self.closed = False
        self.condition = threading.Condition()
        self.process = ctx.Process(target=run_owner, args=(child, self.measurements,
            self.evaluations, self.events, config, self.pause, self.fail_write))
        self.process.start()
        child.close()
        self.collector = threading.Thread(target=self._collect, daemon=True)
        self.collector.start()
        try:
            self.wait(lambda: self.metadata is not None or self.closed, timeout=10.)
            if self.metadata is None: raise RuntimeError(self.error or 'Owner exited before ready')
        except Exception:
            self.close()
            raise

    def _collect(self):
        while True:
            try: kind, value = self.events.get(timeout=.1)
            except queue.Empty:
                if not self.process.is_alive():
                    with self.condition:
                        self.closed = True
                        self.condition.notify_all()
                    return
                continue
            with self.condition:
                if kind == 'ready': self.metadata = value
                elif kind == 'fatal': self.error = value
                elif kind == 'exit_stop': self.exit_stop = value
                elif kind == 'closed': self.closed = True
                else: self.records.append((kind, value))
                self.condition.notify_all()
            if kind == 'closed': return

    def wait(self, predicate, timeout=5.):
        deadline = time.monotonic()+timeout
        with self.condition:
            while not predicate():
                remaining = deadline-time.monotonic()
                if remaining <= 0: raise TimeoutError('Owner diagnostic timeout')
                self.condition.wait(remaining)

    def next_measurement(self, timeout=2.):
        if self.error: raise RuntimeError(self.error)
        packet = self.measurements.get(timeout=timeout)
        if not isinstance(packet, Measurement): raise ValueError('Malformed owner measurement')
        return packet

    def evaluation(self, sequence):
        # Scoring only; no block and no estimator input dependency.
        while True:
            try:
                value = self.evaluations.get_nowait()
                if value.sequence == sequence: return value
            except queue.Empty: return None

    def close(self):
        self.pause.clear()
        self.fail_write.clear()
        try:
            if self.process.is_alive():
                receipt = self.client.close()
                if receipt.zero_written is not True:
                    self.error = self.error or 'owner_close; stop_unconfirmed'
        except Exception as error:
            self.error = self.error or ('owner_close; stop_unconfirmed:'+type(error).__name__)
        finally:
            self.process.join(3.)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(2.)
                self.error = 'owner_terminated; stop_unconfirmed'
            self.collector.join(1.)
            for channel in (self.measurements, self.evaluations, self.events): channel.close()

    def __enter__(self): return self
    def __exit__(self, *args): self.close()
