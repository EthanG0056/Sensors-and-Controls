"""Explicit, content-preserving historical path relocation; no hash substitution."""
import argparse
import hashlib
from pathlib import Path

# Identification metadata is relative to simulation; other metadata to repo root.
RENAMES = {
    'turtlebot.py': 'turtlebot_movement_test.py',
    'simulation/turtlebot.py': 'simulation/turtlebot_movement_test.py',
}


def verify_saved_hash(root, historical_path, expected, transferred_layout=False):
    relative = RENAMES.get(historical_path, historical_path) if transferred_layout else historical_path
    candidate = Path(root) / relative
    if not candidate.is_file():
        raise AssertionError(f'Missing historical input {historical_path}: checked {candidate}')
    actual = hashlib.sha256(candidate.read_bytes()).hexdigest()
    if actual != expected:
        raise AssertionError(
            f'{historical_path}: checked {candidate}; expected SHA256 {expected}, got {actual}')
    if relative != historical_path:
        print(f'RELOCATED (explicit transferred layout): {historical_path} -> {relative}; SHA256 {actual}')
    return candidate


def transferred_layout_argument():
    parser = argparse.ArgumentParser(description='Verify historical evidence without changing saved metadata.')
    parser.add_argument('--transferred-layout', action='store_true',
                        help='Explicitly verify historical turtlebot.py against the renamed movement test; require the original SHA256.')
    return parser.parse_args().transferred_layout
