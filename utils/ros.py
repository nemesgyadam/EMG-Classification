import roslibpy
import math
# platypous wifi: s4vqjc5n3cgx
# 192.168.1.200
# irob
# bark
frame_id = "base_link"


def connect(host="192.168.1.200", port=9090, topic="/mobile_controller/set_target"):
    """
    Connect to ROS
    and subsribe to the required topics
    """
    ros = roslibpy.Ros(host=host, port=port)
    ros.run()

    if ros.is_connected:
        print("Connected to ROS")
    talker = roslibpy.Topic(ros, topic, "geometry_msgs/PoseStamped")
    return ros, talker


commands = {
    "forward": roslibpy.Message(
        {
            "header": {"frame_id": frame_id},
            "pose": {
                "position": {"x": 1.0, "y": 0.0,  "z": 0.0},
                "orientation": {"x": 0.0, "y": 0.0, "w": 1.0, "z": 0.0},
            },
        }
    ),
    "right": roslibpy.Message(
        {
           "header": {"frame_id": frame_id},
            "pose": {
                "position": {"x": 0.0, "y": 0.0,  "z": 0.0},
                "orientation": {"x": 0.0, "y": 0.0, "w": math.cos(math.pi/4), "z": math.sin(-(math.pi/4))},
            },
        }
    ),
    "left": roslibpy.Message(
        {
           "header": {"frame_id": frame_id},
            "pose": {
                "position": {"x": 0.0, "y": 0.0,  "z": 0.0},
                "orientation": {"x": 0.0, "y": 0.0, "w": math.cos(math.pi/4), "z": math.sin(math.pi/4)},
            },
        }
    ),
}
