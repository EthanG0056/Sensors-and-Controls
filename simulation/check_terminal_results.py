"""Validate new terminal evidence; never regenerates old modelling/LQR results."""
import csv
from evidence_paths import verify_saved_hash, transferred_layout_argument
import json
from pathlib import Path
import numpy as np
from benchmark_terminal import metrics
ROOT=Path(__file__).resolve().parents[1]


def read(path):
    rows=list(csv.DictReader(path.open()))
    for row in rows:
        for key in row:
            if key!='mode':row[key]=float(row[key])
    return rows


def main(transferred_layout=False):
    for folder in ('terminal_results','terminal_staged_results'):
        out=ROOT/'simulation'/folder
        meta=json.loads((out/'metadata.json').read_text())
        assert meta['trials']==80 and meta['estimated_state_trials']==0
        for group in ('source_sha256','protected_sha256','protected_direct_results_sha256'):
            for path,digest in meta.get(group,{}).items():
                verify_saved_hash(ROOT, path, digest, transferred_layout)
        summaries=list(csv.DictReader((out/'summary.csv').open()))
        assert len(summaries)==80
        for summary in summaries:
            rows=read(out/(summary['trial']+'.csv'))
            end=float(summary['handoff_s']);goal=np.array([rows[0]['goal_x'],rows[0]['goal_y'],rows[0]['goal_theta']])
            recomputed=metrics(rows,end,goal)
            for key,value in recomputed.items():
                if value is None:assert summary[key]==''
                elif isinstance(value,bool):assert summary[key]==str(value)
                elif isinstance(value,str):assert summary[key]==value
                else:np.testing.assert_allclose(value,float(summary[key]),atol=1e-12)
            if summary['strategy']=='hybrid' and summary['failed']=='False':assert summary['final_mode']=='stop'
            for row in rows:
                assert max(abs(row['wheel_left']),abs(row['wheel_right']))<=6.67+1e-12
                np.testing.assert_allclose(row['v'],.033*(row['wheel_left']+row['wheel_right'])/2,atol=1e-12)
                np.testing.assert_allclose(row['omega'],.033*(row['wheel_right']-row['wheel_left'])/.16,atol=1e-12)
                if row['mode'] in ('fault','stop','coast','stationary_stop'):
                    assert row['v']==row['omega']==0
            if summary['strategy']=='hybrid':
                baseline=read(out/(summary['trial'].replace('_hybrid','_stationary_stop')+'.csv'))
                for a,b in zip(rows,baseline):
                    if a['time_s']<=end:
                        for key in ('x','y','theta'):assert a[key]==b[key],(folder,summary['trial'],key)
                assert summary['mode_switches'] and int(summary['mode_switches'])<40
        print('PASS:',folder,'80 logs; recomputed metrics, limits, stop/fault, identical paired starts, bounded switching and preserved-source hashes')
    # Repeated stationary baselines in both revisions must be byte-identical.
    for p in (ROOT/'simulation/terminal_results').glob('*stationary_stop.csv'):
        assert p.read_bytes()==(ROOT/'simulation/terminal_staged_results'/p.name).read_bytes(),p.name
    print('PASS: all 40 repeated stationary baselines are byte-identical across revisions')


if __name__=='__main__':main(transferred_layout_argument())
