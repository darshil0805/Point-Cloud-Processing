import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

BAG_PATH = "/home/skills/varun/dual_data/joint_trajectory_1"
TOPIC = "/ufactory/joint_states"

reader = rosbag2_py.SequentialReader()
reader.open(
    rosbag2_py.StorageOptions(uri=BAG_PATH, storage_id="sqlite3"),
    rosbag2_py.ConverterOptions("", "")
)

topics_info = reader.get_all_topics_and_types()
msg_type = None
for t in topics_info:
    if t.name == TOPIC:
        msg_type = get_message(t.type)
        break

if msg_type:
    while reader.has_next():
        topic, data, ts = reader.read_next()
        if topic == TOPIC:
            msg = deserialize_message(data, msg_type)
            print(f"Names (len={len(msg.name)}): {msg.name}")
            print(f"Pos (len={len(msg.position)}): {msg.position}")
            break
else:
    print(f"Topic {TOPIC} not found.")
