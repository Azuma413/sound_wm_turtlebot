from launch import LaunchDescription
from launch_ros.actions import Node

namespace = "turtle1"

def generate_launch_description():
    return LaunchDescription([
        # Node(
        #     package='mysample',
        #     executable='sample',
        #     name='sample1', # これを設定することで同じノードを別の名前で起動できる。設定しなければ，executableと同じ名前になる。
        #     parameters=[{'color': color}], # colorパラメータにredを設定
        #     remappings=[('/cmd_vel', '/cmd_vel1'),('/topic', 'topic1')] # (元のトピック名，リマッピング後のトピック名)の形式でトピック名を上書き
        # ),
        Node(
            package='v4l2_camera',
            executable='v4l2_camera_node',
            name='v4l2_camera_node',
            namespace=namespace,
            output='log',
            parameters=[{'video_device': '/dev/video2'}]
        ),
    ])