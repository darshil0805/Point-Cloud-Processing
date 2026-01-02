import pybullet as p
import pybullet_data
import numpy as np
import os

URDF_PATH = "/home/skills/varun/Point-Cloud-Processing/lite-6-updated-urdf/lite_6_new.urdf"
STATES_FILE = "/home/skills/varun/dual_data/extracted_data/joint_trajectory_1/states.txt"

p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
robot_id = p.loadURDF(URDF_PATH, useFixedBase=True)

states = np.loadtxt(STATES_FILE)
first_frame_joints = states[0, :6]
print(f"First Frame Joints: {first_frame_joints}")

for i in range(6):
    p.resetJointState(robot_id, i, first_frame_joints[i])

eef_state = p.getLinkState(robot_id, 6)
print(f"EEF Pose (Link 6): Pos={eef_state[0]}, Quat={eef_state[1]}")

p.disconnect()
