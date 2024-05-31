import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Set the path to the robot.launch.py file
    robot_launch_path = get_package_share_directory("turtlebot3_bringup") + "/launch/robot.launch.py"

    return LaunchDescription([
        IncludeLaunchDescription(robot_launch_path),
        Node(
            package="sound_turtle",
            executable="doa_node",
            name="doa_node",
            output="screen",
        ),
    ])