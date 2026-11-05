# Prescribed-command MuJoCo actuator diagnostic

Actual experimental kv=1 TurtleBot3 actuators; no EKF or navigation feedback.

The spawned worker owns all model/data access and stepping. Command expiry uses monotonic wall time; simulation time is logged independently. Initial 2 s simulation settling precedes live commands. No resets occur. Zero control is not instantaneous physical rest. Scheduling latency is a measured diagnostic outcome, not a hard-real-time guarantee.

See summary.json for checks and measured wheel response, writes.json for actual readback/receipts, and samples.csv for the complete trace. Prescribed forward wheel speed is 0.10/0.033 rad/s; turn commands are (-2,+2) rad/s. Leases of 0.8/0.65 s and response thresholds are synthetic diagnostic fixtures, not deployment settings.

Reproduce from repository root with the existing development Python:

```sh
python -B simulation/demo_mujoco_command_gateway.py
python -B -m unittest discover -s tests -p test_mujoco_command_gateway.py -v
```

This does not establish EKF navigation, obstacle clearance or hardware stopping. Tests separately inject a writer failure and a physics pause, explicitly labelled; those fixtures are not part of this successful live-write trace.
