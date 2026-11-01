
from pathlib import Path
import time
import numpy as np
import mujoco
import mujoco.viewer



# Load the TurtleBot3 Burger scene
MODEL_PATH = (
    Path(__file__).resolve().parent
    / "robotis_tb3"
    / "scene_turtlebot3_burger_gain_test.xml"
)

model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
data = mujoco.MjData(model)

# Get the wheel actuator IDs
left_id = model.actuator("wheel_left").id
right_id = model.actuator("wheel_right").id


# Joint IDs and velocity indices for both wheels
left_joint_id = model.joint("wheel_left").id
right_joint_id = model.joint("wheel_right").id

left_dof = model.jnt_dofadr[left_joint_id]
right_dof = model.jnt_dofadr[right_joint_id]


def print_wheel_speeds():
    """Print actual simulated wheel angular velocities."""
    print(
        f"Actual wheel speeds: "
        f"Left = {data.qvel[left_dof]:.3f} rad/s, "
        f"Right = {data.qvel[right_dof]:.3f} rad/s"
    )


# TurtleBot3 Burger differential-drive parameters
WHEEL_RADIUS = 0.033      # metres — provisional
WHEEL_SEPARATION = 0.160  # metres — from MuJoCo XML
MAX_WHEEL_SPEED = 6.67    # rad/s — actuator control limit


def velocity_to_wheels(v, omega):
    """
    Convert body velocities into wheel angular velocities.

    v     : forward linear velocity (m/s)
    omega : yaw rate (rad/s)

    Returns:
        left_speed, right_speed (rad/s)
    """
    left_speed = (
        v - 0.5 * WHEEL_SEPARATION * omega
    ) / WHEEL_RADIUS

    right_speed = (
        v + 0.5 * WHEEL_SEPARATION * omega
    ) / WHEEL_RADIUS

    return left_speed, right_speed


def get_robot_pose():
    """Read the robot's position and heading from MuJoCo."""
    position = data.qpos[:3]
    quaternion = data.qpos[3:7]

    rotation_matrix = np.zeros(9)
    mujoco.mju_quat2Mat(rotation_matrix, quaternion)

    yaw = np.arctan2(
        rotation_matrix[3],
        rotation_matrix[0]
    )

    return position[0], position[1], yaw


def drive(left_speed, right_speed, duration, viewer):
    """Command both wheel velocities for a fixed duration."""
    data.ctrl[left_id] = left_speed
    data.ctrl[right_id] = right_speed

    steps = int(duration / model.opt.timestep)

    for _ in range(steps):
        step_start = time.perf_counter()

        mujoco.mj_step(model, data)
        viewer.sync()

        # Keep the simulation close to real time
        remaining = model.opt.timestep - (
            time.perf_counter() - step_start
        )
        if remaining > 0:
            time.sleep(remaining)

    # Print once after the movement finishes
    print_wheel_speeds()




with mujoco.viewer.launch_passive(model, data) as viewer:

    initial_pose = get_robot_pose()

    print("Driving forward...")
    left, right = velocity_to_wheels(0.10, 0.0)
    print(f"Commanded wheels: L={left:.3f}, R={right:.3f}")
    drive(left, right, 3.0, viewer)

    forward_pose = get_robot_pose()

    print("Rotating at 0.5 rad/s...")
    left, right = velocity_to_wheels(0.0, 0.5)
    print(f"Commanded wheels: L={left:.3f}, R={right:.3f}")
    drive(left, right, 2.0, viewer)

    rotation_pose = get_robot_pose()

    print("Stopping...")
    drive(0.0, 0.0, 2.0, viewer)

    distance = np.hypot(
        forward_pose[0] - initial_pose[0],
        forward_pose[1] - initial_pose[1]
    )

    heading_change = np.arctan2(
        np.sin(rotation_pose[2] - forward_pose[2]),
        np.cos(rotation_pose[2] - forward_pose[2])
    )

    print(f"Forward distance: {distance:.4f} m")
    print(f"Wrapped heading change: {heading_change:.4f} rad")
