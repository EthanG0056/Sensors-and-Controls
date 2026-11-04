import numpy as np
import mujoco


# ========================== TURTLEBOT PARAMETERS ========================== #

WHEEL_RADIUS = 0.033
WHEEL_SEPARATION = 0.160


# ========================== WHEEL SPEEDS ========================== #

def get_wheel_speeds(model, data):

    # Get MuJoCo DOF indices for the wheel joints
    left_joint = model.joint("wheel_left").id
    right_joint = model.joint("wheel_right").id

    left_dof = model.jnt_dofadr[left_joint]
    right_dof = model.jnt_dofadr[right_joint]

    left_speed = data.qvel[left_dof]
    right_speed = data.qvel[right_dof]

    return np.array([
        left_speed,
        right_speed
    ])


# ========================== TRUE POSE ========================== #

def get_true_pose(model, data):

    # Ground truth pose for evaluation only
    base_joint = model.joint("base_joint").id
    address = model.jnt_qposadr[base_joint]

    position = data.qpos[address:address + 3]
    quaternion = data.qpos[address + 3:address + 7]

    rotation = np.zeros(9)
    mujoco.mju_quat2Mat(
        rotation,
        quaternion
    )

    yaw = np.arctan2(
        rotation[3],
        rotation[0]
    )

    return np.array([
        position[0],
        position[1],
        yaw
    ])