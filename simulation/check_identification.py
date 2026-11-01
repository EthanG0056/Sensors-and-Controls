"""Check saved identification evidence, leakage and deterministic repeatability."""
import csv
from evidence_paths import verify_saved_hash, transferred_layout_argument
import json

import numpy as np
import identify_drive as study


def main(transferred_layout=False):
    meta = json.loads((study.OUT / 'models.json').read_text())
    for path, digest in meta['source_sha256'].items():
        verify_saved_hash(study.ROOT, path, digest, transferred_layout)
    rows = list(csv.DictReader((study.OUT / 'summary.csv').open()))
    for row in rows:
        for key in row:
            if key == 'settled':
                row[key] = row[key] == 'True'
            elif key not in ('trial', 'mode', 'split'):
                row[key] = float(row[key])
    # Deliberately corrupt validation observations; fitted parameters must not move.
    for row in rows:
        if row['split'] == 'held_out':
            for key in ('left', 'right', 'forward', 'yaw_rate', 'wheel_mean', 'wheel_difference'):
                row[key] += 123
    refit = study.fit(rows)
    assert all(a['coefficient'] == b['coefficient'] and a['servo'] == b['servo']
               for a, b in zip(refit, meta['models']))
    model = study.mujoco.MjModel.from_xml_path(str(
        study.ROOT / 'robotis_tb3/scene_turtlebot3_burger_gain_test.xml'))
    repeated = study.trial(model, 1., 'forward', .07, 'held_out', False)
    original = next(row for row in csv.DictReader((study.OUT / 'summary.csv').open())
                    if row['trial'] == repeated['trial'])
    for key, value in repeated.items():
        if isinstance(value, (float, np.floating)):
            assert value == float(original[key]), key
    logs = list(study.OUT.glob('kv*.csv'))
    assert len(logs) == 40
    for path in logs:
        data = np.genfromtxt(path, delimiter=',', names=True)
        gain = .1 if path.name.startswith('kv0.1') else 1.
        assert len(data) == 4001
        for side in ('left', 'right'):
            residual = data['torque_' + side + '_Nm'] - gain * (
                data['command_' + side + '_rad_s'] - data[side + '_rad_s'])
            assert max(abs(residual)) < 1e-10
        # Integrated exact heading rate must agree with unwrapped orientation.
        integrated = np.trapezoid(data['yaw_rad_s'], data['time_s'])
        assert abs(integrated - (data['heading_rad'][-1] - data['heading_rad'][0])) < .005
    print('PASS: source hashes, held-out exclusion, deterministic held-out repeat,')
    print('40 complete logs, actuator torque law, heading-rate integration.')


if __name__ == '__main__':
    main(transferred_layout_argument())
