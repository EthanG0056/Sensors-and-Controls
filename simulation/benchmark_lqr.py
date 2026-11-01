"""IDEAL-STATE DIAGNOSTIC ONLY: MuJoCo pose feeds tracker, not assessed navigation.
Run from repository root: .venv/bin/python simulation/benchmark_lqr.py
"""
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

os.environ.setdefault('MPLCONFIGDIR','/tmp/tb3-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import scipy

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from control.lqr_controller import LQRTracker,CONFIGS,tracking_error,wheel_commands,linearise,discretise,controllability
from control.trajectory import reference,DURATIONS

OUT=ROOT/'simulation/lqr_results'
INITIALS=[(0.,0.,0.),(-.08,.06,.2),(.08,-.06,-.2)]
TS=.05


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def trial(gain,name,config,initial_id,dt):
    suffix='' if gain==.1 else '_gain_test'
    m=mujoco.MjModel.from_xml_path(str(ROOT/f'simulation/robotis_tb3/scene_turtlebot3_burger{suffix}.xml'))
    m.opt.timestep=dt
    substeps=round(TS/dt); assert np.isclose(substeps*dt,TS)
    d=mujoco.MjData(m)
    adr=m.jnt_qposadr[m.joint('base_joint').id]
    x,y,yaw=INITIALS[initial_id]
    d.qpos[adr:adr+2]=[x,y]; d.qpos[adr+3:adr+7]=[np.cos(yaw/2),0,0,np.sin(yaw/2)]
    for _ in range(round(2/dt)): mujoco.mj_step(m,d)
    ids=[m.actuator(n).id for n in ('wheel_left','wheel_right')]
    dofs=[m.jnt_dofadr[m.joint(n).id] for n in ('wheel_left','wheel_right')]
    tracker=LQRTracker(TS,CONFIGS[config]); rows=[]
    for step in range(round(DURATIONS[name]/TS)+1):
        mujoco.mj_forward(m,d)
        t=step*TS
        r=reference(name,t)
        rotation=np.zeros(9);mujoco.mju_quat2Mat(rotation,d.qpos[adr+3:adr+7])
        pose=np.r_[d.qpos[adr:adr+2],np.arctan2(rotation[3],rotation[0])]
        command,mode=tracker.command(pose,r)  # IDEAL ground truth diagnostic feedback
        wheels,applied,saturated=wheel_commands(*command)
        error=tracking_error(pose,r.pose)
        rows.append(dict(time_s=t,simulation_time_s=float(d.time),x=pose[0],y=pose[1],yaw=pose[2],
            ref_x=r.pose[0],ref_y=r.pose[1],ref_yaw=r.pose[2],ref_v=r.input[0],ref_omega=r.input[1],
            error_x=error[0],error_y=error[1],error_yaw=error[2],
            raw_v=command[0],raw_omega=command[1],applied_v=applied[0],applied_omega=applied[1],
            command_left=wheels[0],command_right=wheels[1],
            actual_left=d.qvel[dofs[0]],actual_right=d.qvel[dofs[1]],saturated=int(saturated),mode=mode))
        if step==round(DURATIONS[name]/TS): break
        d.ctrl[ids]=wheels
        for _ in range(substeps): mujoco.mj_step(m,d)
    assert not any(w.number for w in d.warning)
    numeric=np.array([[v for k,v in row.items() if k!='mode'] for row in rows])
    assert np.isfinite(numeric).all()
    tag=f'kv{gain:g}_{name}_{config}_init{initial_id}_dt{dt:g}'
    write_csv(OUT/f'{tag}.csv',rows)
    errors=np.array([[r['error_x'],r['error_y'],r['error_yaw']] for r in rows])
    controls=np.array([[r['applied_v'],r['applied_omega']] for r in rows[:-1]])
    wheel_series=np.array([[r['command_left'],r['command_right']] for r in rows[:-1]])
    return dict(trial=tag,state_source='ideal_state_diagnostic',gain=gain,path=name,config=config,
        initial_id=initial_id,physics_dt=dt,controller_dt=TS,duration=DURATIONS[name],
        position_rmse=float(np.sqrt(np.mean(np.sum(errors[:,:2]**2,axis=1)))),
        heading_rmse=float(np.sqrt(np.mean(errors[:,2]**2))),
        final_position_error=float(np.linalg.norm(errors[-1,:2])),final_heading_error=float(errors[-1,2]),
        effort_v2=float(TS*np.sum(controls[:,0]**2)),effort_omega2=float(TS*np.sum(controls[:,1]**2)),
        wheel_squared_effort=float(TS*np.sum(wheel_series**2)),
        saturation_fraction=float(np.mean([r['saturated'] for r in rows[:-1]])),
        stationary_stop_fraction=float(np.mean([r['mode']=='stationary_stop' for r in rows[:-1]])))


def plots(results):
    fig,axes=plt.subplots(2,3,figsize=(12,7))
    for col,path in enumerate(('straight','curve','waypoint')):
        for gain in (.1,1.):
            tag=f'kv{gain:g}_{path}_balanced_init1_dt0.002'
            d=np.genfromtxt(OUT/f'{tag}.csv',delimiter=',',names=True,dtype=None,encoding=None)
            axes[0,col].plot(d['x'],d['y'],label=f'kv={gain:g}')
            axes[1,col].plot(d['time_s'],np.hypot(d['error_x'],d['error_y']),label=f'kv={gain:g}')
        axes[0,col].plot(d['ref_x'],d['ref_y'],'k--',label='reference')
        axes[0,col].set(title=path+' (initial condition 1)',xlabel='x (m)',ylabel='y (m)');axes[0,col].axis('equal')
        axes[1,col].set(xlabel='Time (s)',ylabel='Position error (m)')
        for row in (0,1): axes[row,col].legend(fontsize=8);axes[row,col].grid(alpha=.3)
    fig.suptitle('IDEAL-STATE DIAGNOSTIC — balanced weights');fig.tight_layout();fig.savefig(OUT/'paths.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for gain,ax in zip((.1,1.),axes):
        for config,marker in zip(CONFIGS,('o','s','^')):
            rs=[r for r in results if r['gain']==gain and r['config']==config and r['path'] in ('straight','curve','waypoint') and r['physics_dt']==.002]
            ax.scatter([r['wheel_squared_effort']/r['duration'] for r in rs],[r['position_rmse'] for r in rs],label=config,marker=marker)
        ax.set(title=f'kv={gain:g}: all paths and initial conditions',xlabel='Mean sum wheel-command squares (rad²/s²)',ylabel='Position RMSE (m)');ax.legend();ax.grid(alpha=.3)
    fig.suptitle('IDEAL-STATE DIAGNOSTIC — tracking versus input effort');fig.tight_layout();fig.savefig(OUT/'tradeoff.png',dpi=160);plt.close(fig)


    fig,axes=plt.subplots(2,2,figsize=(11,7))
    for row,gain in enumerate((.1,1.)):
        for col,path in enumerate(('low_speed','rotation')):
            d=np.genfromtxt(OUT/f'kv{gain:g}_{path}_balanced_init1_dt0.002.csv',delimiter=',',names=True,dtype=None,encoding=None)
            ax=axes[row,col]
            for side in ('left','right'):
                ax.plot(d['time_s'],d['command_'+side],'--',label=side+' command')
                ax.plot(d['time_s'],d['actual_'+side],label=side+' measured')
            ax.set(title=f'kv={gain:g}: {path}',xlabel='Time (s)',ylabel='Wheel speed (rad/s)');ax.legend(fontsize=7);ax.grid(alpha=.3)
    fig.suptitle('IDEAL-STATE DIAGNOSTIC — actuator limitations');fig.tight_layout();fig.savefig(OUT/'failure_wheels.png',dpi=160);plt.close(fig)


def report(results):
    lines=['# Part D executed results — IDEAL-STATE DIAGNOSTIC ONLY','',
           'Ground-truth pose feeds the controller in these tests. No estimated-state results or final navigation/docking claims.','',
           'Metrics include the complete tracking interval, initial transients and the final endpoint sample. Each aggregate averages nine runs (three paths × three initial conditions).','',
           '| kv | Weights | Mean position RMSE (m) | Mean heading RMSE (rad) | Mean final position error (m) | Mean wheel² effort / duration (rad²/s²) | Saturated intervals (%) |',
           '|---:|---|---:|---:|---:|---:|---:|']
    for gain in (.1,1.):
        for config in CONFIGS:
            rs=[r for r in results if r['gain']==gain and r['config']==config and r['physics_dt']==.002 and r['path'] in ('straight','curve','waypoint')]
            mean=lambda k:np.mean([r[k] for r in rs])
            lines.append(f"| {gain:g} | {config} | {mean('position_rmse'):.6f} | {mean('heading_rmse'):.6f} | {mean('final_position_error'):.6f} | {np.mean([r['wheel_squared_effort']/r['duration'] for r in rs]):.6f} | {100*mean('saturation_fraction'):.3f} |")
    lines+=['','## All trials','','| Trial | Position RMSE (m) | Heading RMSE (rad) | Final position (m) | Final heading (rad, signed) | Saturated (%) |','|---|---:|---:|---:|---:|---:|']
    for r in results:
        lines.append(f"| {r['trial']} | {r['position_rmse']:.6f} | {r['heading_rmse']:.6f} | {r['final_position_error']:.6f} | {r['final_heading_error']:.6f} | {100*r['saturation_fraction']:.3f} |")
    lines+=['','## Timestep sensitivity (balanced, initial condition 1)','','| kv | Path | Δ position RMSE (fine − coarse, m) | Δ final position error (m) |','|---:|---|---:|---:|']
    for fine in [r for r in results if r['physics_dt']==.001]:
        coarse=next(r for r in results if r['gain']==fine['gain'] and r['path']==fine['path'] and r['config']==fine['config'] and r['initial_id']==fine['initial_id'] and r['physics_dt']==.002)
        lines.append(f"| {fine['gain']:g} | {fine['path']} | {fine['position_rmse']-coarse['position_rmse']:+.6f} | {fine['final_position_error']-coarse['final_position_error']:+.6f} |")
    lines+=['','![Diagnostic tracking paths](paths.png)','','![Accuracy and effort](tradeoff.png)','','![Actuator limitation diagnostics](failure_wheels.png)','']
    (OUT/'RESULTS.md').write_text('\n'.join(lines))


def main():
    OUT.mkdir(exist_ok=True)
    protected=list((ROOT/'simulation/robotis_tb3').rglob('*'))+list((ROOT/'simulation/identification_results').glob('*'))+list((ROOT/'simulation/validation_results').glob('*'))
    protected+=[ROOT/'simulation'/n for n in ('PART_A_MODELLING.md','VALIDATION.md','validate_drive.py','turtlebot.py')]
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected if p.is_file()}
    results=[]
    for gain in (.1,1.):
        for name in ('straight','curve','waypoint'):
            for config in CONFIGS:
                for initial in range(3): results.append(trial(gain,name,config,initial,.002))
        for name in ('low_speed','rotation','stationary'):
            results.append(trial(gain,name,'balanced',1,.002))
        for name in ('straight','curve','waypoint'):
            results.append(trial(gain,name,'balanced',1,.001))
    write_csv(OUT/'summary.csv',results);plots(results);report(results)
    analysis=[]
    for name,v,w in [('straight',.08,0),('curve',.08,.3),('rotation',0,.4),('stationary',0,0),('low_speed',.01,0)]:
        a,b=linearise(v,w);ad,bd=discretise(a,b,TS)
        rank,s=controllability(a,b);drank,ds=controllability(ad,bd)
        analysis.append(dict(reference=name,continuous_rank=rank,continuous_singular_values=s.tolist(),discrete_rank=drank,discrete_singular_values=ds.tolist()))
    sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'control/lqr_controller.py',ROOT/'control/trajectory.py',ROOT/'tests/test_lqr.py']}
    meta=dict(state_source='IDEAL-STATE DIAGNOSTIC ONLY',estimated_state_runs=0,python=platform.python_version(),mujoco=mujoco.__version__,numpy=np.__version__,scipy=scipy.__version__,controller_dt=TS,initial_conditions=INITIALS,
        configs={k:dict(q=w.q,r=w.r) for k,w in CONFIGS.items()},controllability=analysis,protected_sha256=hashes,source_sha256=sources,trials=len(results))
    (OUT/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    print(f'Executed {len(results)} ideal-state diagnostic trials. Protected inputs unchanged.')
    print('\n'.join((OUT/'RESULTS.md').read_text().splitlines()[:16]))


if __name__=='__main__': main()
