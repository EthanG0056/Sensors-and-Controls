"""Additional geometry metrics from retained raw logs; no simulation/model mutation."""
import csv
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'simulation/terminal_comparison'


def main():
    OUT.mkdir(exist_ok=True)
    rows=[]
    for revision in ('terminal_results','terminal_staged_results'):
        folder=ROOT/'simulation'/revision
        summary=list(csv.DictReader((folder/'summary.csv').open()))
        for r in summary:
            if r['strategy']!='hybrid':continue
            d=np.genfromtxt(folder/(r['trial']+'.csv'),delimiter=',',names=True,dtype=None,encoding=None)
            end=float(r['handoff_s']);indices=np.flatnonzero((d['time_s']>=end)&(d['mode']=='approach'))
            overshoot=None
            if len(indices):
                i=indices[0];goal=np.array([d['goal_x'][i],d['goal_y'][i]])
                direction=goal-np.array([d['x'][i],d['y'][i]]);direction/=np.linalg.norm(direction)
                points=np.column_stack([d['x'][i:],d['y'][i:]])
                overshoot=float(max(0.,max((points-goal)@direction)))
            rows.append(dict(revision=revision,trial=r['trial'],gain=r['gain'],physics_dt=r['physics_dt'],scenario=r['scenario'],
                final_approach_plane_overshoot_m=overshoot,
                heading_phase_overshoot_rad=r['heading_phase_overshoot_rad'],
                full_terminal_plane_excursion_m=r['position_plane_overshoot_m']))
    with (OUT/'geometry_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['# Additional terminal geometry metrics','',
        'The original full-terminal plane excursion includes deliberate staging detours. Final-approach overshoot fixes its plane at the first non-staging approach sample; heading-only trials without an approach have no such metric. These scalar measures do not bound swept area or lateral drift.','',
        '| Revision | kv | dt (s) | Max final-approach plane overshoot (m) | Max full-terminal plane excursion (m) |',
        '|---|---:|---:|---:|---:|']
    for revision in ('terminal_results','terminal_staged_results'):
        for gain in ('0.1','1.0'):
            for dt in ('0.002','0.001'):
                rs=[r for r in rows if r['revision']==revision and r['gain']==gain and r['physics_dt']==dt]
                overs=[r['final_approach_plane_overshoot_m'] for r in rs if r['final_approach_plane_overshoot_m'] is not None]
                lines.append(f"| {revision} | {gain} | {dt} | {max(overs):.6f} | {max(float(r['full_terminal_plane_excursion_m']) for r in rs):.6f} |")
    (OUT/'GEOMETRY.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__=='__main__':main()
