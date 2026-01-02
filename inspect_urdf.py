import pybullet as p
import pybullet_data
import os

URDF_PATH = "/home/skills/varun/Point-Cloud-Processing/lite-6-updated-urdf/lite_6_new.urdf"

p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

if not os.path.exists(URDF_PATH):
    print(f"URDF Path {URDF_PATH} not found.")
else:
    robot_id = p.loadURDF(URDF_PATH, useFixedBase=True)
    num_joints = p.getNumJoints(robot_id)
    print(f"Total Joints: {num_joints}")
    movable_joints = []
    for i in range(num_joints):
        info = p.getJointInfo(robot_id, i)
        joint_name = info[1].decode("utf-8")
        joint_type = info[2]
        if joint_type != p.JOINT_FIXED:
            movable_joints.append((i, joint_name))
            print(f"Movable Joint - Index: {i}, Name: {joint_name}")
        else:
            print(f"Fixed Joint - Index: {i}, Name: {joint_name}")

p.disconnect()
