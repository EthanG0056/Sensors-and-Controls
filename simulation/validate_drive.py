"""Open-loop modelling validation only. Ground truth must not feed navigation control.

Run: .venv/bin/python simulation/validate_drive.py
Each run starts from XML initial conditions; phases match turtlebot.py (no settling).
Post-step samples represent interval ends; wheel integrals use trapezoidal quadrature.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent
RADIUS = 0.033  # provisional, metres
TRACK = 0.160  # wheel body centre separation, metres
PHASES = [('forward', 0.10, 0.0, 3.0), ('rotation', 0.0, 0.5, 2.0),
          ('stop', 0.0, 0.0, 2.0)]


def pose(model, data):
    address = model.jnt_qposadr[model.joint('base_joint').id]
    q = data.qpos[address:address + 7]
    rotation = np.zeros(9)
    mujoco.mju_quat2Mat(rotation, q[3:7])
    return np.array([q[0], q[1], np.arctan2(rotation[3], rotation[0])])


def run(scene, output, friction_scale=1.0):
    model = mujoco.MjModel.from_xml_path(str(scene))
    ids = [model.actuator(n).id for n in ('wheel_left', 'wheel_right')]
    dofs = [model.jnt_dofadr[model.joint(n).id] for n in ('wheel_left', 'wheel_right')]
    # Optional in-memory sensitivity experiment; never changes XML.
    model.dof_frictionloss[dofs] *= friction_scale
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    initial = previous = pose(model, data)
    yaw = float(initial[2])
    rows, metrics = [], {}
    for name, v, omega, duration in PHASES:
        command = np.array([(v - TRACK * omega / 2) / RADIUS,
                            (v + TRACK * omega / 2) / RADIUS])
        assert np.all(np.abs(command) <= model.actuator_ctrlrange[ids, 1])
        data.ctrl[ids] = command
        start = pose(model, data)
        yaw_start, time_start = yaw, data.time
        wheel_integral = np.zeros(2)
        wheel_previous = data.qvel[dofs].copy()
        phase_rows = []
        steps = round(duration / model.opt.timestep)
        assert np.isclose(steps * model.opt.timestep, duration)
        for _ in range(steps):
            mujoco.mj_step(model, data)
            current = pose(model, data)
            yaw += np.arctan2(np.sin(current[2] - previous[2]),
                              np.cos(current[2] - previous[2]))
            actual = data.qvel[dofs].copy()
            wheel_integral += (wheel_previous + actual) * model.opt.timestep / 2
            elapsed = data.time - time_start
            row = dict(phase=name, time_s=float(data.time), phase_time_s=float(elapsed),
                       command_left_rad_s=float(command[0]), command_right_rad_s=float(command[1]),
                       actual_left_rad_s=float(actual[0]), actual_right_rad_s=float(actual[1]),
                       x_m=float(current[0]), y_m=float(current[1]),
                       displacement_m=float(np.linalg.norm(current[:2] - initial[:2])),
                       heading_unwrapped_rad=yaw,
                       phase_displacement_m=float(np.linalg.norm(current[:2] - start[:2])),
                       ideal_phase_distance_m=abs(v) * elapsed,
                       phase_heading_rad=yaw-yaw_start, ideal_phase_heading_rad=omega*elapsed)
            rows.append(row); phase_rows.append(row)
            previous, wheel_previous = current, actual
        actual_series = np.array([[r['actual_left_rad_s'], r['actual_right_rad_s']] for r in phase_rows])
        errors = actual_series - command
        metrics[name] = dict(command_rad_s=command.tolist(), final_rad_s=actual.tolist(),
            mean_rad_s=actual_series.mean(axis=0).tolist(),
            wheel_rmse_rad_s=np.sqrt(np.mean(errors**2, axis=0)).tolist(),
            wheel_bias_rad_s=errors.mean(axis=0).tolist(),
            distance_m=row['phase_displacement_m'], ideal_distance_m=abs(v)*duration,
            distance_error_m=row['phase_displacement_m']-abs(v)*duration,
            heading_rad=yaw-yaw_start, ideal_heading_rad=omega*duration,
            heading_error_rad=yaw-yaw_start-omega*duration,
            distance_rmse_m=float(np.sqrt(np.mean([(r['phase_displacement_m']-r['ideal_phase_distance_m'])**2 for r in phase_rows]))),
            heading_rmse_rad=float(np.sqrt(np.mean([(r['phase_heading_rad']-r['ideal_phase_heading_rad'])**2 for r in phase_rows]))),
            encoder_predicted_signed_distance_m=float(RADIUS * wheel_integral.sum()/2),
            encoder_predicted_heading_rad=float(RADIUS * (wheel_integral[1]-wheel_integral[0])/TRACK))
    assert np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel))
    assert not any(w.number for w in data.warning), 'MuJoCo emitted numerical warnings'
    output.mkdir(parents=True, exist_ok=True)
    stem = scene.stem + (f'_joint_friction_{friction_scale:g}' if friction_scale != 1 else '')
    with (output / f'{stem}.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    # Hash all source XML/assets for reproducibility, retaining ROBOTIS files unchanged.
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(scene.parent.rglob('*')) if p.suffix in ('.xml', '.stl')}
    result = dict(scene=str(scene.relative_to(ROOT)), mujoco=mujoco.__version__,
                  python=platform.python_version(), timestep_s=model.opt.timestep,
                  radius_m=RADIUS, track_m=TRACK, joint_friction_scale=friction_scale,
                  actuator_gain=model.actuator_gainprm[ids, 0].tolist(),
                  actuator_dyntype=model.actuator_dyntype[ids].tolist(),
                  wheel_joint_frictionloss=model.dof_frictionloss[dofs].tolist(),
                  wheel_armature=model.dof_armature[dofs].tolist(),
                  geom_friction=model.geom_friction.tolist(), source_sha256=hashes,
                  samples=len(rows), phases=metrics)
    (output / f'{stem}.json').write_text(json.dumps(result, indent=2) + '\n')
    print(stem, json.dumps(metrics, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'validation_results')
    parser.add_argument('--joint-friction-scale', type=float, default=1.0)
    args = parser.parse_args()
    if args.joint_friction_scale < 0:
        parser.error('friction scale must be nonnegative')
    for filename in ('scene_turtlebot3_burger.xml', 'scene_turtlebot3_burger_gain_test.xml'):
        run(ROOT / 'robotis_tb3' / filename, args.output, args.joint_friction_scale)
