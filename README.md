
# Sensors-and-Controls

Repository for Integrated Project.
Group 23:
    Ethan Griffiths
    Lachlan Taylor
    Arnoe Kruger

## Part A: Modelling

This part describes the TurtleBot3 Burger's planar differential-drive motion and
how its simulated drive response is identified and checked. The MuJoCo setup and
drive-identification experiment are in [the navigation scene](integration/scenes/scene_turtlebot3_navigation.xml)
and [identify_drive.py](simulation/identify_drive.py). See the
[modelling and validation study](simulation/PART_A_MODELLING.md) and
[observability analysis](simulation/PART_A_OBSERVABILITY.md) for the assumptions
and results.

## Part B: Sensing

The sensing modules obtain camera images and LiDAR scans from MuJoCo. The camera
detects ArUco markers and produces range/bearing observations; the LiDAR module
reads ranges and extracts a front-obstacle measurement. See
[aruco_camera.py](sensing/aruco_camera.py) and [lidar.py](sensing/lidar.py).

## Part C: Estimation

The Extended Kalman Filter predicts the robot pose from wheel odometry and
corrects it with camera observations of known markers. Its motion and camera
models, Jacobians, and update steps are implemented in [ekf.py](estimation/ekf.py);
the [wheel adapter](simulation/ekf_wheel_adapter.py) provides a prediction-only
interface for measured wheel samples.

## Part D: Control

The controller tracks pose-and-velocity references with a discrete-time LQR.
Terminal controllers add a bounded approach and final-heading phase for reaching
a goal pose. See [lqr_controller.py](control/lqr_controller.py),
[terminal_controller.py](control/terminal_controller.py), and the reference
generator in [trajectory.py](control/trajectory.py). The
[Part D report](control/PART_D_CONTROL.md) describes the model and limitations.

## Part E: Integration and Robustness

The integrated MuJoCo run connects wheel odometry, camera-based EKF updates,
LiDAR obstacle detection, trajectory/terminal control, and wheel commands in
[main.py](integration/main.py). The
[estimated-pose interface](control/estimated_pose_interface.py) checks pose
freshness and validity before dispatching commands, while
[command_gateway.py](control/command_gateway.py) manages expiring wheel-command
leases. Related interface and gateway checks are in
[test_estimated_pose_interface.py](tests/test_estimated_pose_interface.py) and
[test_command_gateway.py](tests/test_command_gateway.py).

## Team member contributions

**Ethan Griffiths:**

- Sensing:
  - aruco camera
  - lidar
  - calibration

**Lachlan Taylor:**

put something here

**Arnoe Kruger:**

put something here
