# Verification after transfer to the shared repository

Historical benchmark metadata and all modelling/LQR/terminal results are unchanged.
The original movement script was transferred under a new filename so the teammate's
existing `simulation/turtlebot.py` could remain intact.

| Role | Path in Git clone | SHA256 at inspection |
|---|---|---|
| Historical movement script | `simulation/turtlebot_movement_test.py` | `f78627dd55eeb947c07c32e3014671f25b76716334e21f95ce97bacd71719eed` |
| Teammate's current script | `simulation/turtlebot.py` | `e774a6f23f2866c09c3a281ec72b0dc84ca23b88122ca071a76aeb5448051493` |

The historical movement script's SHA256 exactly equals the original development
file and the saved expected digest. All 299 transferred evidence files, including
historical metadata, were compared byte-for-byte against the development folder.
All other recorded source/protected hashes match at their original paths. Legacy
`validation_results` XML/asset hashes use paths relative to `simulation/`, not
repository root, and also match. The teammate script matches Git HEAD.

## Exact failure and explicit correction

The reproduced traceback was the hash assertion in `check_lqr_results.py` line 17
(before this change), group `protected_sha256`, historical key
`simulation/turtlebot.py`. It correctly rejected the teammate's different file.
The same conflict affects identification (`source_sha256`, key `turtlebot.py`)
and both terminal metadata files (`protected_sha256`, key
`simulation/turtlebot.py`). Thus all three strict checker commands reject this
transferred layout, not just the first checker invoked.

Strict checking remains the default. `--transferred-layout` explicitly maps only
the historical movement-script path to `turtlebot_movement_test.py`, supporting
the two original metadata path roots. It compares that file to the **original
saved digest**. It neither searches for arbitrary matching files nor falls back
to the teammate script if the renamed file is missing/corrupt. Every other path
and all numerical checks remain strict. The mapping is printed when applied.
No saved hashes or historical results were rewritten or benchmarks regenerated.

This legitimately verifies preserved historical input bytes under an acknowledged
rename. It does not prove the old path has unchanged semantics, or that the
teammate's current script generated historical results. The original source tree
is not claimed to be literally unchanged. Future benchmarks would record the
current layout and need separate result directories; do not regenerate historical
results merely to update their manifests. Historical source digests are provenance
checks, not signatures, and existing benchmark metadata does not cryptographically
pin every logged output; numerical evidence checks retain their existing scope.

## Commands from the cloned repository root

Use a Python environment containing the previously recorded dependencies. The
clone currently has no `.venv`; the existing development interpreter can be used
without copying its environment into Git:

```sh
PYTHON="$HOME/Downloads/Sensors-and-Controls-main/.venv/bin/python"
"$PYTHON" -m unittest discover -s tests -p test_lqr.py -v
"$PYTHON" -m unittest discover -s tests -p test_terminal.py -v
"$PYTHON" -m unittest discover -s tests -p test_staged_terminal.py -v
"$PYTHON" -m unittest discover -s tests -p test_evidence_paths.py -v
"$PYTHON" simulation/check_identification.py --transferred-layout
"$PYTHON" simulation/check_lqr_results.py --transferred-layout
"$PYTHON" simulation/check_terminal_results.py --transferred-layout
git diff --check
```

The first three commands are the original 19 tests. Four additional regression
tests check explicit relocation, strict default behaviour, missing/corrupt renamed
files and unrelated paths. Unflagged evidence checkers are expected to fail on
the historical movement path in this clone; that is deliberate, not waived
verification. `.gitignore` excludes `.venv`, `__pycache__`, `.pyc` and `.DS_Store`;
no already-tracked files in these categories were present at inspection.
No commit, push, staging or Co-authored-by trailer was performed/added.
