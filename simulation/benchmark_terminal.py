"""IDEAL-STATE terminal diagnostics. Writes only terminal_results; preserves baselines."""
import csv
from dataclasses import asdict
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

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from control.terminal_controller import TrajectoryTerminalController,TerminalParameters
from control.lqr_controller import LQRTracker,wheel_commands,wrap
from control.trajectory import Reference,reference,DURATIONS
OUT=ROOT/'simulation/terminal_results'
TS=.05;POST=50.
CASES={'ahead':(-.20,.0,0.),'behind':(.15,0.,0.),'lateral':(0.,.15,-1.),
       'opposed':(-.15,-.10,np.pi),'heading_only':(0.,0.,1.2),
       'wrap':(.06,-.06,-np.pi+.02),'near':(.015,.005,-.15),'at_goal':(0.,0.,0.)}
SCENARIOS=list(CASES)+['handoff_straight','handoff_waypoint']


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def metrics(rows,end,goal):
    terminal=[r for r in rows if r['time_s']>=end-1e-9]
    pos=np.array([r['position_error'] for r in terminal]);heading=np.array([r['heading_error'] for r in terminal])
    good=np.array([r['position_error']<=.025 and abs(r['heading_error'])<=.07 and r['observed_v']<=.008 and r['observed_omega']<=.04 for r in terminal])
    stays=np.logical_and.accumulate(good[::-1])[::-1]
    candidates=[i for i in range(len(terminal)) if stays[i] and terminal[-1]['time_s']-terminal[i]['time_s']>=2.-1e-9]
    settling=terminal[candidates[0]]['time_s']-end if candidates else None
    initial=np.array([terminal[0]['x'],terminal[0]['y']]);vector=goal[:2]-initial
    direction=vector/np.linalg.norm(vector) if np.linalg.norm(vector)>1e-6 else np.zeros(2)
    overshoot=max(0.,max(np.dot(np.array([r['x'],r['y']])-goal[:2],direction) for r in terminal))
    modes=[r['mode'] for r in terminal]
    # Orientation overshoot is defined only from first final-heading phase onward.
    indices=[i for i,r in enumerate(terminal) if r['mode'] in ('heading','coast','stop')]
    heading_overshoot=None
    if indices:
        first=indices[0];sign=np.sign(heading[first])
        heading_overshoot=float(max(0.,max(-sign*heading[first:])))
    return dict(final_position_error=float(pos[-1]),final_heading_error=float(heading[-1]),
        settling_time_s=settling,failed=settling is None,position_rmse=float(np.sqrt(np.mean(pos**2))),
        position_plane_overshoot_m=float(overshoot),heading_phase_overshoot_rad=heading_overshoot,
        saturation_fraction=float(np.mean([r['saturated'] for r in terminal[:-1]])),
        mode_switches=sum(a!=b for a,b in zip(modes,modes[1:])),
        reapproaches=sum(a in ('heading','coast') and b=='bearing' for a,b in zip(modes,modes[1:])),
        final_mode=modes[-1],stop_time_s=next((r['time_s']-end for r in terminal if r['mode']=='stop'),None))


def trial(scenario,gain,dt,strategy):
    path=scenario.removeprefix('handoff_') if scenario.startswith('handoff_') else None
    end=DURATIONS[path] if path else 0.
    goal=reference(path,end).pose.copy() if path else np.zeros(3)
    initial=(-.08,.06,.2) if path else CASES[scenario]
    m=mujoco.MjModel.from_xml_path(str(ROOT/f"simulation/robotis_tb3/scene_turtlebot3_burger{'_gain_test' if gain==1 else ''}.xml"));m.opt.timestep=dt
    d=mujoco.MjData(m);adr=m.jnt_qposadr[m.joint('base_joint').id]
    d.qpos[adr:adr+2]=initial[:2];d.qpos[adr+3:adr+7]=[np.cos(initial[2]/2),0,0,np.sin(initial[2]/2)]
    for _ in range(round(2/dt)):mujoco.mj_step(m,d)
    ids=[m.actuator(n).id for n in ('wheel_left','wheel_right')]
    dofs=[m.jnt_dofadr[m.joint(n).id] for n in ('wheel_left','wheel_right')]
    hybrid=TrajectoryTerminalController(goal,end);baseline=LQRTracker(TS);rows=[];previous=None
    for step in range(round((end+POST)/TS)+1):
        t=step*TS;mujoco.mj_forward(m,d)
        rotation=np.zeros(9);mujoco.mju_quat2Mat(rotation,d.qpos[adr+3:adr+7])
        pose=np.r_[d.qpos[adr:adr+2],np.arctan2(rotation[3],rotation[0])]
        r=reference(path,t) if path and t<end else Reference(goal,np.zeros(2))
        if strategy=='hybrid':body,wheels,mode,saturated=hybrid.command(pose,r,t)
        else:
            raw,mode=baseline.command(pose,r);wheels,body,saturated=wheel_commands(*raw)
        speed=np.linalg.norm(pose[:2]-previous[:2])/TS if previous is not None else 0.
        rate=abs(wrap(pose[2]-previous[2]))/TS if previous is not None else 0.
        rows.append(dict(time_s=t,x=pose[0],y=pose[1],theta=pose[2],goal_x=goal[0],goal_y=goal[1],goal_theta=goal[2],
            position_error=float(np.linalg.norm(goal[:2]-pose[:2])),heading_error=float(wrap(goal[2]-pose[2])),
            observed_v=float(speed),observed_omega=float(rate),v=body[0],omega=body[1],
            wheel_left=wheels[0],wheel_right=wheels[1],actual_left=d.qvel[dofs[0]],actual_right=d.qvel[dofs[1]],
            saturated=int(saturated),mode=mode))
        previous=pose
        if step<round((end+POST)/TS):
            d.ctrl[ids]=wheels
            for _ in range(round(TS/dt)):mujoco.mj_step(m,d)
    assert not any(w.number for w in d.warning)
    assert all(np.isfinite([v for k,v in row.items() if k!='mode']).all() for row in rows)
    name=f'{scenario}_kv{gain:g}_dt{dt:g}_{strategy}'
    write_csv(OUT/f'{name}.csv',rows)
    return dict(trial=name,scenario=scenario,gain=gain,physics_dt=dt,strategy=strategy,handoff_s=end,
                **metrics(rows,end,goal))


def report(results):
    lines=['# Terminal pose results — IDEAL-STATE DIAGNOSTIC ONLY','','No estimated-state integration, obstacle avoidance or ArUco docking is tested. All failures are retained.',
        '', '| kv | dt (s) | Strategy | Failures / 10 | Mean final position (m) | Mean absolute final heading (rad) | Mean settling among successes (s) | Max position-plane overshoot (m) | Saturated intervals (%) |',
        '|---:|---:|---|---:|---:|---:|---:|---:|---:|']
    for gain in (.1,1.):
        for dt in (.002,.001):
            for strategy in ('stationary_stop','hybrid'):
                rs=[r for r in results if r['gain']==gain and r['physics_dt']==dt and r['strategy']==strategy]
                times=[r['settling_time_s'] for r in rs if not r['failed']]
                settling=f'{np.mean(times):.3f}' if times else '—'
                lines.append(f"| {gain:g} | {dt:g} | {strategy} | {sum(r['failed'] for r in rs)} / {len(rs)} | {np.mean([r['final_position_error'] for r in rs]):.6f} | {np.mean([abs(r['final_heading_error']) for r in rs]):.6f} | {settling} | {max(r['position_plane_overshoot_m'] for r in rs):.6f} | {100*np.mean([r['saturation_fraction'] for r in rs]):.3f} |")
    lines+=['','Settling requires position ≤0.025 m, absolute heading ≤0.07 rad, observed speed ≤0.008 m/s and yaw rate ≤0.04 rad/s continuously to the horizon, for at least 2 s. Unsettled trials fail. Means include all ten scenarios; settling means exclude failures.','',
        '| Trial | Final position (m) | Final heading (rad) | Settling (s) | Failed | Mode switches | Final mode |', '|---|---:|---:|---:|---|---:|---|']
    for r in results:
        time='—' if r['settling_time_s'] is None else f"{r['settling_time_s']:.2f}"
        lines.append(f"| {r['trial']} | {r['final_position_error']:.6f} | {r['final_heading_error']:.6f} | {time} | {r['failed']} | {r['mode_switches']} | {r['final_mode']} |")
    lines+=['','![Terminal errors](errors.png)','','![Handoff and actuator response](handoff.png)','']
    (OUT/'RESULTS.md').write_text('\n'.join(lines))


def plots():
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    for col,gain in enumerate((.1,1.)):
        for strategy in ('stationary_stop','hybrid'):
            d=np.genfromtxt(OUT/f'opposed_kv{gain:g}_dt0.002_{strategy}.csv',delimiter=',',names=True,dtype=None,encoding=None)
            axes[0,col].plot(d['time_s'],d['position_error'],label=strategy)
            axes[1,col].plot(d['time_s'],abs(d['heading_error']),label=strategy)
        axes[0,col].set(title=f'Opposed initial heading, kv={gain:g}',ylabel='Position error (m)')
        axes[1,col].set(xlabel='Time after handoff (s)',ylabel='Absolute heading error (rad)')
        axes[0,col].axhline(.025,color='k',linestyle=':');axes[1,col].axhline(.07,color='k',linestyle=':')
        for row in (0,1):axes[row,col].grid(alpha=.3);axes[row,col].legend()
    fig.suptitle('IDEAL-STATE TERMINAL DIAGNOSTIC');fig.tight_layout();fig.savefig(OUT/'errors.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(10,9))
    for strategy in ('stationary_stop','hybrid'):
        d=np.genfromtxt(OUT/f'handoff_waypoint_kv1_dt0.002_{strategy}.csv',delimiter=',',names=True,dtype=None,encoding=None)
        axes[0].plot(d['x'],d['y'],label=strategy)
        axes[1].plot(d['time_s'],d['position_error'],label=strategy)
    d=np.genfromtxt(OUT/'handoff_waypoint_kv1_dt0.002_hybrid.csv',delimiter=',',names=True,dtype=None,encoding=None)
    for side in ('left','right'):
        axes[2].plot(d['time_s'],d['wheel_'+side],'--',label=side+' command')
        axes[2].plot(d['time_s'],d['actual_'+side],label=side+' measured')
    axes[0].plot(d['goal_x'][-1],d['goal_y'][-1],'kx',label='goal');axes[0].axis('equal');axes[0].set(xlabel='x (m)',ylabel='y (m)')
    axes[1].set(xlabel='Time (s)',ylabel='Fixed goal distance (m)');axes[1].axvline(15,color='k',linestyle=':',label='handoff')
    axes[2].set(xlabel='Time (s)',ylabel='Wheel speed (rad/s)')
    for ax in axes:ax.grid(alpha=.3);ax.legend(fontsize=8)
    fig.suptitle('IDEAL-STATE DIAGNOSTIC — waypoint handoff, kv=1');fig.tight_layout();fig.savefig(OUT/'handoff.png',dpi=160);plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    protected=[]
    for folder in ('simulation/robotis_tb3','simulation/identification_results','simulation/validation_results','simulation/lqr_results'):
        protected.extend(p for p in (ROOT/folder).rglob('*') if p.is_file())
    protected.extend(ROOT/p for p in ('control/lqr_controller.py','control/trajectory.py','control/PART_D_CONTROL.md','simulation/benchmark_lqr.py','tests/test_lqr.py','simulation/turtlebot.py'))
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    results=[]
    for gain in (.1,1.):
        for dt in (.002,.001):
            for scenario in SCENARIOS:
                for strategy in ('stationary_stop','hybrid'):results.append(trial(scenario,gain,dt,strategy))
    write_csv(OUT/'summary.csv',results);report(results);plots()
    sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'control/terminal_controller.py',ROOT/'tests/test_terminal.py']}
    (OUT/'metadata.json').write_text(json.dumps(dict(state_source='IDEAL-STATE DIAGNOSTIC ONLY',estimated_state_trials=0,
        parameters=asdict(TerminalParameters()),post_handoff_s=POST,cases=CASES,controller_dt=TS,
        python=platform.python_version(),mujoco=mujoco.__version__,numpy=np.__version__,protected_sha256=hashes,source_sha256=sources,trials=len(results)),indent=2)+'\n')
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    print('\n'.join((OUT/'RESULTS.md').read_text().splitlines()[:16]))


if __name__=='__main__':main()
