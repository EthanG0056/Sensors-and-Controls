"""Staged revision benchmark; reuses identical direct-hybrid test protocol.

Writes terminal_staged_results only. Original terminal_results remain unchanged.
"""
from pathlib import Path
import hashlib
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import benchmark_terminal as protocol
from control.staged_terminal_controller import StagedTrajectoryTerminalController


def main():
    # Substitute the experiment dependency, not the baseline module on disk.
    protocol.OUT=protocol.ROOT/'simulation/terminal_staged_results'
    protocol.TrajectoryTerminalController=StagedTrajectoryTerminalController
    protocol.main()
    p=protocol.OUT/'metadata.json';meta=json.loads(p.read_text())
    meta['revision']='Staged hybrid: 0.12 m behind goal; alignment threshold 0.35 rad'
    for path in (Path(__file__),protocol.ROOT/'control/staged_terminal_controller.py',protocol.ROOT/'tests/test_staged_terminal.py'):
        meta['source_sha256'][str(path.relative_to(protocol.ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    meta['protected_direct_results_sha256']={str(path.relative_to(protocol.ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in (protocol.ROOT/'simulation/terminal_results').glob('*') if path.is_file()}
    p.write_text(json.dumps(meta,indent=2)+'\n')
    report=protocol.OUT/'RESULTS.md'
    report.write_text(report.read_text().replace('# Terminal pose results', '# Staged terminal pose results').replace('No estimated-state integration', 'Here hybrid denotes the staged revision (0.12 m staging distance, 0.35 rad trigger). No estimated-state integration'))


if __name__=='__main__':main()
