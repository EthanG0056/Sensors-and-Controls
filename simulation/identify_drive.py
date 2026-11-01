"""Deterministic open-loop Part A identification; ground truth is evaluation only.
Run .venv/bin/python simulation/identify_drive.py (no viewer required).
"""
import csv
import hashlib
import json
import os
from pathlib import Path
import platform

os.environ.setdefault('MPLCONFIGDIR', '/tmp/tb3-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'identification_results'
R, B = .033, .160
# Split fixed before inspecting outcomes. Magnitudes interleaved for interpolation.
LEVELS = {'forward': [(v, 'train' if i % 2 == 0 else 'held_out')
                      for i, v in enumerate([.04, .07, .10, .13, .16])],
          'turn': [(v, 'train' if i % 2 == 0 else 'held_out')
                   for i, v in enumerate([.3, .5, .7, .9, 1.1])]}


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def trial(model, gain, mode, target, split, save):
    d = mujoco.MjData(model)  # fresh state, time, controls, solver warm start
    ids = [model.actuator(n).id for n in ('wheel_left', 'wheel_right')]
    dofs = [model.jnt_dofadr[model.joint(n).id] for n in ('wheel_left', 'wheel_right')]
    base = model.body('base').id
    floor = model.geom('floor').id
    command = np.array([target/R]*2 if mode == 'forward' else [-B*target/(2*R), B*target/(2*R)])
    assert np.all(command >= model.actuator_ctrlrange[ids, 0]) and np.all(command <= model.actuator_ctrlrange[ids, 1])
    dt = model.opt.timestep
    rows = []
    jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    yaw = old_yaw = 0.
    for step in range(round(8/dt)+1):
        t = step*dt
        d.ctrl[ids] = 0 if t < 2 else command
        # Align qpos/qvel, actuator and contact forces at the same sample time.
        mujoco.mj_forward(model, d)
        rot = d.xmat[base].reshape(3, 3)
        raw_yaw = np.arctan2(rot[1, 0], rot[0, 0])
        yaw += np.arctan2(np.sin(raw_yaw-old_yaw), np.cos(raw_yaw-old_yaw)); old_yaw = raw_yaw
        vel = np.zeros(6)
        # Body origin is the wheel-axis midpoint; avoid COM-offset velocities.
        mujoco.mj_jacBody(model, d, jacp, jacr, base)
        vel[:3], vel[3:] = jacr @ d.qvel, jacp @ d.qvel
        # World velocity projected onto horizontal heading axes.
        v = np.cos(raw_yaw)*vel[3] + np.sin(raw_yaw)*vel[4]
        lateral = -np.sin(raw_yaw)*vel[3] + np.cos(raw_yaw)*vel[4]
        base_force = 0.
        for contact_id in range(d.ncon):
            c = d.contact[contact_id]
            other = c.geom2 if c.geom1 == floor else c.geom1 if c.geom2 == floor else -1
            if other >= 0 and model.geom_bodyid[other] == base:
                force = np.zeros(6); mujoco.mj_contactForce(model, d, contact_id, force)
                base_force += abs(force[0])
        rows.append(dict(time_s=t, drive_time_s=t-2, command_left_rad_s=float(d.ctrl[ids[0]]),
            command_right_rad_s=float(d.ctrl[ids[1]]), left_rad_s=float(d.qvel[dofs[0]]),
            right_rad_s=float(d.qvel[dofs[1]]), torque_left_Nm=float(d.actuator_force[ids[0]]),
            torque_right_Nm=float(d.actuator_force[ids[1]]), x_m=float(d.xpos[base, 0]),
            y_m=float(d.xpos[base, 1]), heading_rad=yaw, forward_m_s=float(v),
            lateral_m_s=float(lateral), yaw_rad_s=float(vel[2] - rot[2, 0]*(rot[0, 0]*vel[0]+rot[1, 0]*vel[1])/(rot[0, 0]**2+rot[1, 0]**2)),
            base_floor_normal_N=base_force, contacts=d.ncon))
        if step < round(8/dt):
            mujoco.mj_step(model, d)
    assert not any(w.number for w in d.warning)
    assert all(np.isfinite(list(row.values())).all() for row in rows)
    name = f'kv{gain:g}_{mode}_{target:+.2f}'
    if save:
        write_csv(OUT / f'{name}.csv', rows)
    # Equal-duration adjacent late windows test residual drift, without assuming settlement.
    late = [r for r in rows if 6 <= r['time_s'] < 8]
    earlier = [r for r in rows if 4 <= r['time_s'] < 6]
    mean = lambda key, rs=late: float(np.mean([r[key] for r in rs]))
    wheels = np.array([mean('left_rad_s'), mean('right_rad_s')])
    drift = max(abs(mean(k)-mean(k, earlier)) for k in ('left_rad_s', 'right_rad_s'))
    variation = max(np.std([r[k] for r in late]) for k in ('left_rad_s', 'right_rad_s'))
    tolerance = max(.02, .02*max(abs(wheels)))
    return dict(trial=name, gain=gain, mode=mode, target=target, split=split,
        command_left=command[0], command_right=command[1], left=wheels[0], right=wheels[1],
        wheel_mean=float(wheels.mean()), wheel_difference=float(wheels[1]-wheels[0]),
        forward=mean('forward_m_s'), yaw_rate=mean('yaw_rad_s'),
        lateral_rms=float(np.sqrt(mean('lateral_m_s')**2 + np.var([r['lateral_m_s'] for r in late]))),
        wheel_rmse=float(np.sqrt(np.mean([(r[k]-c)**2 for r in late
                          for k,c in zip(('left_rad_s','right_rad_s'),command)]))),
        wheel_window_drift=drift, wheel_std_max=float(variation), settled=bool(drift<tolerance and variation<tolerance),
        base_contact_fraction=float(np.mean([r['base_floor_normal_N'] > 1e-6 for r in late])),
        base_normal_mean=mean('base_floor_normal_N'))


def fit(rows):
    models = []
    for gain in (.1, 1.):
        for mode in LEVELS:
            train = [r for r in rows if r['gain']==gain and r['mode']==mode and r['split']=='train']
            xkey, ykey = ('wheel_mean','forward') if mode=='forward' else ('wheel_difference','yaw_rate')
            x = np.array([r[xkey] for r in train]); y = np.array([r[ykey] for r in train])
            coefficient = float(x@y/(x@x))
            # Signed dead-zone wheel response, fitted separately for left and right.
            # Grid-search d, solve nonnegative slope a analytically: w=a sign(u) max(|u|-d,0).
            servo = []
            for side in ('left','right'):
                u = np.array([r['command_'+side] for r in train]); w = np.array([r[side] for r in train])
                candidates = []
                for deadzone in np.linspace(0, max(abs(u)), 2001):
                    z = np.sign(u)*np.maximum(abs(u)-deadzone, 0)
                    slope = max(0., float(z@w/(z@z))) if z@z else 0.
                    candidates.append((float(np.mean((slope*z-w)**2)), slope, float(deadzone)))
                mse, slope, deadzone = min(candidates)
                servo.append(dict(slope=slope, deadzone_rad_s=deadzone, train_rmse=mse**.5))
            model = dict(gain=gain, mode=mode, coefficient=coefficient,
                coefficient_units='m/rad' if mode=='forward' else 'rad/rad',
                trial_ratio_min=float(min(y/x)), trial_ratio_max=float(max(y/x)), servo=servo)
            for r in rows:
                if r['gain']!=gain or r['mode']!=mode: continue
                predicted_wheels = [s['slope']*np.sign(r['command_'+side])*max(abs(r['command_'+side])-s['deadzone_rad_s'],0)
                                    for side,s in zip(('left','right'),servo)]
                z = sum(predicted_wheels)/2 if mode=='forward' else predicted_wheels[1]-predicted_wheels[0]
                r['observed_response'] = r[ykey]
                r['measured_wheel_prediction'] = coefficient*r[xkey]
                r['command_prediction'] = coefficient*z
                r['nominal_measured_wheel_prediction'] = (R if mode=='forward' else R/B)*r[xkey]
                r['ideal_command_prediction'] = r['target']
            held = [r for r in rows if r['gain']==gain and r['mode']==mode and r['split']=='held_out']
            model['held_out'] = {}
            for key in ('measured_wheel_prediction','command_prediction','nominal_measured_wheel_prediction','ideal_command_prediction'):
                e = np.array([r[key]-r['observed_response'] for r in held])
                model['held_out'][key] = dict(rmse=float(np.sqrt(np.mean(e**2))), bias=float(e.mean()), max_abs=float(max(abs(e))))
            models.append(model)
    return models


def plots(rows):
    fig, axes = plt.subplots(2,2, figsize=(11,8))
    for ax,(gain,mode) in zip(axes.flat,[(g,m) for g in (.1,1.) for m in LEVELS]):
        rs = sorted([r for r in rows if r['gain']==gain and r['mode']==mode],key=lambda r:r['target'])
        for split,marker in [('train','o'),('held_out','s')]:
            subset=[r for r in rs if r['split']==split]
            ax.scatter([r['target'] for r in subset],[r['observed_response'] for r in subset],marker=marker,label=split)
        for key,label in [('command_prediction','Fitted command model'),('measured_wheel_prediction','Fitted rolling + measured wheels'),('ideal_command_prediction','Ideal command')]:
            ax.plot([r['target'] for r in rs],[r[key] for r in rs],label=label)
        unit='m/s' if mode=='forward' else 'rad/s'
        ax.set(xlabel=f'Command ({unit})',ylabel=f'Response ({unit})',title=f'kv={gain:g}: {mode}')
        ax.grid(alpha=.3); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT/'response.png',dpi=160); plt.close(fig)
    fig, axes = plt.subplots(2,2,figsize=(11,7))
    for ax,(gain,mode) in zip(axes.flat,[(g,m) for g in (.1,1.) for m in LEVELS]):
        target=.10 if mode=='forward' else .7
        data=np.genfromtxt(OUT/f'kv{gain:g}_{mode}_{target:+.2f}.csv',delimiter=',',names=True)
        for side in ('left','right'):
            ax.plot(data['drive_time_s'],data[side+'_rad_s'],label=side+' measured')
            ax.plot(data['drive_time_s'],data['command_'+side+'_rad_s'],'--',label=side+' command')
        ax.axvspan(4,6,color='grey',alpha=.15,label='measurement window')
        ax.set(xlabel='Time since command step (s)',ylabel='Wheel speed (rad/s)',title=f'kv={gain:g}, {mode}, command={target:g}')
        ax.legend(fontsize=7); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT/'wheel_steps.png',dpi=160); plt.close(fig)



def report(rows, models):
    lines = ['# Part A: executed identification results', '',
        'Generated by `simulation/identify_drive.py`. Coefficients are empirical finite-window fits, not physical dimensions.', '',
        '| kv | Mode | Rolling coefficient | Training trial ratio range | Held-out rolling RMSE | Held-out command RMSE | Nominal measured-wheel RMSE | Ideal-command RMSE |',
        '|---:|---|---:|---|---:|---:|---:|---:|']
    for m in models:
        h=m['held_out']
        lines.append(f"| {m['gain']:g} | {m['mode']} | {m['coefficient']:.8f} | {m['trial_ratio_min']:.6f}–{m['trial_ratio_max']:.6f} | {h['measured_wheel_prediction']['rmse']:.6f} | {h['command_prediction']['rmse']:.6f} | {h['nominal_measured_wheel_prediction']['rmse']:.6f} | {h['ideal_command_prediction']['rmse']:.6f} |")
    lines += ['', 'Forward coefficient units: m/rad; turn coefficient maps wheel-speed difference to yaw rate (dimensionless rad/rad). Forward errors: m/s; turn errors: rad/s.', '',
        '| kv | Mode | Left slope / deadzone (rad/s) | Right slope / deadzone (rad/s) |', '|---:|---|---|---|']
    for m in models:
        a,b=m['servo']
        lines.append(f"| {m['gain']:g} | {m['mode']} | {a['slope']:.6f} / {a['deadzone_rad_s']:.6f} | {b['slope']:.6f} / {b['deadzone_rad_s']:.6f} |")
    lines += ['', '| kv | Mode | Settled / 10 | Max timestep change in forward speed (m/s) | Max timestep change in yaw rate (rad/s) |', '|---:|---|---:|---:|---:|']
    for g in (.1,1.):
        for mode in LEVELS:
            rs=[r for r in rows if r['gain']==g and r['mode']==mode]
            lines.append(f"| {g:g} | {mode} | {sum(r['settled'] for r in rs)} / 10 | {max(abs(r['fine_forward_delta']) for r in rs):.6f} | {max(abs(r['fine_yaw_delta']) for r in rs):.6f} |")
    lines += ['', '## Per-trial predictions', '', '| kv | Mode | Command | Split | Measured | Rolling prediction | Command prediction | Settled |', '|---:|---|---:|---|---:|---:|---:|---|']
    for r in rows:
        lines.append(f"| {r['gain']:g} | {r['mode']} | {r['target']:+.2f} | {r['split']} | {r['observed_response']:+.6f} | {r['measured_wheel_prediction']:+.6f} | {r['command_prediction']:+.6f} | {r['settled']} |")
    lines += ['', '![Response and model predictions](response.png)', '', '![Representative wheel step responses](wheel_steps.png)', '', '![Held-out residuals](held_out_errors.png)', '']
    (OUT/'RESULTS.md').write_text('\n'.join(lines))
    fig, axes=plt.subplots(1,2,figsize=(11,4))
    for ax,mode in zip(axes,LEVELS):
        for g in (.1,1.):
            rs=sorted([r for r in rows if r['mode']==mode and r['gain']==g and r['split']=='held_out'],key=lambda r:r['target'])
            for key,style in [('measured_wheel_prediction','o-'),('command_prediction','s--')]:
                ax.plot([r['target'] for r in rs],[r[key]-r['observed_response'] for r in rs],style,label=f'kv={g:g}, '+('rolling' if key.startswith('measured') else 'command'))
        unit='m/s' if mode=='forward' else 'rad/s'
        ax.axhline(0,color='black',linewidth=.7); ax.grid(alpha=.3)
        ax.set(title=mode+' held-out errors',xlabel=f'Command ({unit})',ylabel=f'Predicted minus measured ({unit})'); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT/'held_out_errors.png',dpi=160); plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    protected = list((ROOT/'robotis_tb3').rglob('*'))+[ROOT/'turtlebot.py', ROOT/'validate_drive.py', ROOT/'identify_drive.py']
    hashes = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected if p.is_file()}
    rows, fine = [], []
    for gain,suffix in [(.1,''),(1.,'_gain_test')]:
        scene=ROOT/'robotis_tb3'/f'scene_turtlebot3_burger{suffix}.xml'
        for dt,destination,save in [(.002,rows,True),(.001,fine,False)]:
            m=mujoco.MjModel.from_xml_path(str(scene)); m.opt.timestep=dt
            for mode,levels in LEVELS.items():
                for magnitude,split in levels:
                    for sign in (-1,1):
                        destination.append(trial(m,gain,mode,sign*magnitude,split,save))
    models=fit(rows)
    for r,f in zip(rows,fine):
        assert r['trial']==f['trial']
        r['fine_forward_delta']=f['forward']-r['forward']
        r['fine_yaw_delta']=f['yaw_rate']-r['yaw_rate']
        r['fine_wheel_delta_max']=max(abs(f[k]-r[k]) for k in ('left','right'))
    write_csv(OUT/'summary.csv',rows); write_csv(OUT/'fine_summary.csv',fine)
    plots(rows)
    report(rows, models)
    metadata=dict(python=platform.python_version(),numpy=np.__version__,mujoco=mujoco.__version__,matplotlib=matplotlib.__version__,
        platform=platform.platform(), reset_settle_s=2,drive_s=6,measurement_window_s=[4,6],
        coarse_dt=.002,fine_dt=.001,levels=LEVELS,source_sha256=hashes,models=models)
    (OUT/'models.json').write_text(json.dumps(metadata,indent=2)+'\n')
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    print(json.dumps(models,indent=2))
    print('Trials:',len(rows),'fine:',len(fine),'settled:',sum(r['settled'] for r in rows))


if __name__=='__main__': main()
