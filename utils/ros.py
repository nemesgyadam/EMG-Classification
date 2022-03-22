import roslibpy

# platypous wifi: s4vqjc5n3cgx
# 192.168.1.200
# irob
# bark


def connect(host="192.168.1.200", port=9090, topic="/cmd_vel/other"):
    """
    Connect to ROS
    and subsribe to the required topics
    """
    ros = roslibpy.Ros(host=host, port=port)
    ros.run()

    if ros.is_connected:
        print("Connected to ROS")
    talker = roslibpy.Topic(ros, topic, "geometry_msgs/Twist")
    return ros, talker

commands = {
    "forward": roslibpy.Message({
        'linear':{'x': 0.3, 'y': 0.0, 'z': 0.0},
        'angular': {'x': 0.0, 'y': 0.0, 'z': 0.0}
        }),
    # "backward" : roslibpy.Message({
    #     'linear':{'x': -0.3, 'y': 0.0, 'z': 0.0},
    #     'angular': {'x': 0.0, 'y': 0.0, 'z': 0.0}
    #     }),
    "left" : roslibpy.Message({
        'linear':{'x': 0, 'y': 0.0, 'z': 0.0},
        'angular': {'x': 0, 'y': 0.0, 'z': 0.85}
        }),
    "right" : roslibpy.Message({
        'linear':{'x': 0, 'y': 0.0, 'z': 0.0},
        'angular': {'x': .0, 'y': 0.0, 'z': -0.8,}
        })
}