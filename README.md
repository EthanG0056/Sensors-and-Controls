
# Sensors-and-Controls

Repository for Integrated Project.
Group 23:

- Ethan Griffiths (25364771)
- Lachlan Taylor (25992569)
- Arnoe Kruger (25477389)


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


## Robustness Testing Results ##

The integrated system was tested under several varied conditions without re-tuning the controller or estimator.


**Baseline Repeatability**

Three identical nominal trials were completed successfully with identical results, confirming deterministic and repeatable behaviour in simulation.

    - Localisation position RMSE: 0.1250 m

    - Navigation position RMSE: 0.0953 m

    - True docking position error: 0.0536 m

    - 3/3 successful runs


**Obstacle Position Variation**

Obstacle positions were shifted both closer to and farther from the nominal trajectory.

Test	             |  Nav. RMSE	 |  True Dock Error	 |  Avoidance Samples	  |  Result:
-------------------|-------------|-------------------|----------------------|-----------
Baseline:	         | 0.0953 m	   |   0.0536 m	       |      400	            |   Success
Obstacles closer:	 | 0.1369 m	   |   0.0598 m	       |      601	            |   Success
Obstacles farther: | 0.1212 m	   |   0.0489 m	       |      412	            |   Success

The system successfully completed all layouts. Moving obstacles closer increased avoidance duration and tracking error.


**Initial Pose Disturbance**

The robot was started with two disturbed initial poses:

    - (y=+0.05) m, (\theta=+5^\circ)

    - (y=-0.08) m, (\theta=-10^\circ)

Both tests completed successfully, with navigation position RMSE values of 0.0381 m and 0.0452 m respectively. 
This demonstrated recovery from moderate initial position and heading offsets.


**Camera Dropout**

Camera measurements were deliberately removed while EKF wheel-odometry prediction remained active.

    5 s dropout: successful navigation and docking

    10 s dropout: failed after camera reacquisition due to estimator/controller rejection

This showed that the EKF can tolerate short visual outages, but longer periods of odometry-only localisation can produce sufficient drift 
to cause large corrections when vision returns.


**EKF Measurement Covariance Sensitivity**

The camera measurement covariance (R) was tested at (0.5R), nominal (R), and (2R).

(R)	   | Localisation RMSE	 |  Result
-------|---------------------|-----------
(0.5R) |  0.1183 m	         |  Success
(R)	   |  0.1172 m	         |  Success
(2R)	 |  0.1222 m	         |  Success

All three cases completed successfully, indicating that the EKF is reasonably robust to moderate changes in assumed camera measurement uncertainty.

Overall, the system remained stable under obstacle-position changes, disturbed starting poses, short camera outages, and moderate EKF parameter variation. 
The main observed limitation was reduced reliability after extended loss of visual measurements.


## Team member contributions

**Ethan Griffiths:**

- Sensing:

  - aruco camera
  - lidar
  - calibration


**Lachlan Taylor:**

- State Estimation / EKF:

  - developed full Extended Kallman Filter (ekf) model with differential drive wheel odometry conversion from wheel speed to linear and angular velocity (v,w) calculations, 
    nonlinear motion model for both straight and turn trajectory, jacobian of motion model with respect to state, camera reading and jacobian, 
    and the main EKF prediction and measurement update process including State covariance (P), process-noise covariance (Q) and measurement-noise covariance (R) and Kalman gain.
  - calibration of ekf parameters and testing localisation performance against ground truth

- System Integration:

  - Integrating and modifying all team members files into the Integration folder to run the simulation in integration.main
  - Created and configured Mujoco navigation scene including Aruco Markers, obstacles and charging station
  - Integrated team member Aruco marker detection with EKF landmark map and tested + debugged in Mujoco
  - Obstacle avoidance programming and integration with LiDAR + coordination with LQR/EKF trajectory controller
  - Charger station robot docking procedure using aurco camera sensing + coordination with LQR/EKF trajectory controller
  - Mujoco simulation and testing + debugging of full running system
  - Robustness testing with varying obstacle placement and operating conditions assessing localisation, trajectory tracking, obstacle avoidance and docking.


**Arnoe Kruger:**

Mathematical Modelling (Part A):

-TurtleBot3 Burger differential-drive kinematic modelling
-MuJoCo drive-response identification and parameter estimation
-Model verification, held-out validation and actuator gain comparisons
-Controllability and observability analysis

Control (Part D):

-Discrete-time LQR trajectory-tracking controller
-Trajectory generation and reference tracking
-LQR tuning, simulation benchmarking and performance evaluation
-Hybrid terminal controller and staged goal-pose positioning
-Terminal accuracy and robustness testing

System Integration and Validation:

-EKF measured-wheel odometry adapter and timestamped pose snapshots
-Estimated-pose validation and controller interface
-Wheel-command gateway, command expiry and fault handling
-MuJoCo actuator integration and independent simulation/actuator ownership
-Localisation validation and integration safety testing
-Reproducible experimental results and verification tests


## Generative AI declaration

**Visual Studio Code copilot:**

- Test files for camera and lidar functions in MuJoCo
- Required pips installation


**ChatGPT:**

- Quick understanding and explanation of teammates code
- Debugging formating/spelling errors causing the code to not run
