# R1のピックアップ部に設置されたカメラの画像からボールの有無，ボールの色を検出するノード
# ライブラリのインポート
# *************************************************************************************************
# ROS関連のライブラリをインポート
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid

# 画像処理関連のモジュールのインポート
from cv_bridge import CvBridge
import cv2
# *************************************************************************************************
# 定数の定義
# *************************************************************************************************

# *************************************************************************************************
# クラスの定義
# *************************************************************************************************
class WrapNode(Node):

    def __init__(self):
        super().__init__('wrap_node')
        # 変数
        self.raw_image = None
        self.id_color_flag = False
        # ROSの設定
        self.obs_image_pub = self.create_publisher(Image, 'obs_image', 10)
        self.sound_map_pub = self.create_publisher(OccupancyGrid, 'sound_map', 10)
        self.reward_pub = self.create_publisher(Float32, 'reward', 10)
        self.create_subscription(OccupancyGrid, 'map', self.map_callback, 10)
        self.create_subscription(PoseWithCovarianceStamped, 'amcl_pose', self.amcl_pose_callback, 10)
        self.bridge = CvBridge() # OpenCVとROSの画像を変換するためのクラス
        self.create_timer(0.1, self.timer_callback)
    
    def timer_callback(self):
        # 観測の生成，報酬の計算，音源mapの作成，それぞれのpublish
        pass
    
    def map_callback(self, msg):
        # マップの更新
        pass
    
    def amcl_pose_callback(self, msg):
        # 位置情報の更新
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