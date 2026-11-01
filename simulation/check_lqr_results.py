"""Recompute benchmark evidence and verify protected inputs without rerunning trials."""
import csv
from evidence_paths import verify_saved_hash, transferred_layout_argument
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'simulation/lqr_results'


def main(transferred_layout=False):
    meta=json.loads((OUT/'metadata.json').read_text())
    assert meta['estimated_state_runs']==0
    for group in ('source_sha256','protected_sha256'):
        for path,digest in meta[group].items():
            verify_saved_hash(ROOT, path, digest, transferred_layout)
    summary=list(csv.DictReader((OUT/'summary.csv').open()))
    assert len(summary)==meta['trials']==66
    for row in summary:
        data=np.genfromtxt(OUT/(row['trial']+'.csv'),delimiter=',',names=True,dtype=None,encoding=None)
        numeric=[k for k in data.dtype.names if k!='mode']
        assert all(np.isfinite(data[k]).all() for k in numeric)
        assert np.allclose(np.diff(data['time_s']),.05)
        assert max(abs(data['command_left']))<=6.67+1e-12
        assert max(abs(data['command_right']))<=6.67+1e-12
        np.testing.assert_allclose(data['applied_v'],.033*(data['command_left']+data['command_right'])/2,atol=1e-12)
        np.testing.assert_allclose(data['applied_omega'],.033*(data['command_right']-data['command_left'])/.160,atol=1e-12)
        def same(key,value): np.testing.assert_allclose(float(row[key]),value,atol=1e-12)
        position2=data['error_x']**2+data['error_y']**2
        # Body rotation must preserve world position error norm.
        np.testing.assert_allclose(position2,(data['ref_x']-data['x'])**2+(data['ref_y']-data['y'])**2,atol=1e-12)
        same('position_rmse',np.sqrt(np.mean(position2)))
        same('heading_rmse',np.sqrt(np.mean(data['error_yaw']**2)))
        same('final_position_error',np.sqrt(position2[-1]))
        same('final_heading_error',data['error_yaw'][-1])
        same('saturation_fraction',np.mean(data['saturated'][:-1]))
        same('wheel_squared_effort',.05*np.sum(data['command_left'][:-1]**2+data['command_right'][:-1]**2))
        if row['path']=='stationary':
            assert all(data['mode']=='stationary_stop')
            np.testing.assert_array_equal(data['command_left'],0)
            np.testing.assert_array_equal(data['command_right'],0)
    print('PASS: 66 trial logs, metrics, wheel limits/conversion, explicit stationary policy,')
    print('ideal-state labelling, source hashes and preserved Part A/model inputs.')


if __name__=='__main__': main(transferred_layout_argument())
