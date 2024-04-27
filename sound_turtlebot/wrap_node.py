# R1のピックアップ部に設置されたカメラの画像からボールの有無，ボールの色を検出するノード
# ライブラリのインポート
# *************************************************************************************************
# ROS関連のライブラリをインポート
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


# 画像処理関連のモジュールのインポート
from cv_bridge import CvBridge
import cv2
# *************************************************************************************************
# 定数の定義
# *************************************************************************************************
PUB_IMG = False
# *************************************************************************************************
# クラスの定義
# *************************************************************************************************
class WrapNode(Node):

    def __init__(self):
        super().__init__('wrap_node')
        # パラメータの宣言と取得
        self.declare_parameter("color", "blue")
        self.color = self.get_parameter('color').get_parameter_value().string_value # blue:右側 red:左側
        # 全体用の変数
        self.raw_image = None
        self.id_color_flag = False
        # ROSの設定
        if PUB_IMG:
            self.image_pub = self.create_publisher(Image, 'obs_image', 10)
        self.bridge = CvBridge() # OpenCVとROSの画像を変換するためのクラス
        self.create_timer(0.1, self.timer_callback)
    
    def timer_callback(self):
        pass
# *************************************************************************************************
# メイン関数
# *************************************************************************************************
def main(args=None):
    rclpy.init(args=args)
    node = WrapNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()