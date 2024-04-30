# R1のピックアップ部に設置されたカメラの画像からボールの有無，ボールの色を検出するノード
# ライブラリのインポート
# *************************************************************************************************
# ROS関連のライブラリをインポート
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32

# 画像処理関連のモジュールのインポート
from cv_bridge import CvBridge
import cv2
# *************************************************************************************************
# 定数の定義
# *************************************************************************************************

# *************************************************************************************************
# クラスの定義
# *************************************************************************************************
class ControlNode(Node):

    def __init__(self):
        super().__init__('control_node')
        # 変数
        self.raw_image = None
        self.id_color_flag = False
        # ROSの設定
        self.goal_pose_pub = self.create_publisher(PoseStamped, 'goal_pose', 10)
        self.create_subscription(Image, "obs_image", self.obs_image_callback, 10)
        self.create_subscription(Float32, "reward", self.reward_callback, 10)
        self.bridge = CvBridge() # OpenCVとROSの画像を変換するためのクラス
        self.create_timer(0.1, self.timer_callback)
    
    def timer_callback(self):
        pass
# *************************************************************************************************
# メイン関数
# *************************************************************************************************
def main(args=None):
    rclpy.init(args=args)
    node = ControlNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()