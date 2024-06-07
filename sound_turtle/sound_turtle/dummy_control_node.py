# wrap_nodeの動作確認のためのダミーノード
# R1のピックアップ部に設置されたカメラの画像からボールの有無，ボールの色を検出するノード
# ライブラリのインポート
# *************************************************************************************************
# ROS関連のライブラリをインポート
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from sound_turtle_msgs.srv import GetAction
import numpy as np
# *************************************************************************************************
# 定数の定義
# *************************************************************************************************

# *************************************************************************************************
# クラスの定義
# *************************************************************************************************
class DummyControlNode(Node):
    def __init__(self):
        super().__init__('dummy_control_node')
        # ROSの設定
        self.subscription = self.create_subscription(Image, 'obs_image', self.image_callback, 10)
        self.create_service(GetAction, 'get_action', self.get_action_callback)

    def image_callback(self, msg):
        print("get image")
    def get_action_callback(self, request, response):
        rclpy.sleep(0.3)
        response.action0 = 1 - np.random.rand()*2
        response.action1 = 1 - np.random.rand()*2
        return response

# *************************************************************************************************
# メイン関数
# *************************************************************************************************
def main(args=None):
    rclpy.init(args=args)
    node = DummyControlNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()