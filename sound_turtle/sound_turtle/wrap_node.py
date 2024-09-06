# R1のピックアップ部に設置されたカメラの画像からボールの有無，ボールの色を検出するノード
# ライブラリのインポート
# *************************************************************************************************
# ROS関連のライブラリをインポート
import rclpy
from rclpy.node import Node
from rclpy.task import Future
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped
from sound_turtle_msgs.srv import GetAction

# 画像処理関連のモジュールのインポート
from cv_bridge import CvBridge
import cv2

# その他
import numpy as np
import os
import yaml
from pathlib import Path
import time
# *************************************************************************************************
# 定数の定義
# *************************************************************************************************
MOVE_RANGE = 0.5 # ロボットの移動範囲[m] 到達を前提としないので，十分に大きく設定する。ただし，壁面に衝突しないように注意
THRESHOLD = 0.5 # 音源の存在判定の閾値
MAP_NAME = "327" # 使用するmapの名前
# *************************************************************************************************
# クラスの定義
# *************************************************************************************************
class WrapNode(Node):
    def __init__(self):
        super().__init__('wrap_node')
        # ROSの設定
        self.obs_image_pub = self.create_publisher(Image, 'obs_image', 10)
        self.sound_map_pub = self.create_publisher(OccupancyGrid, 'sound_map', 10)
        self.create_subscription(PoseWithCovarianceStamped, 'amcl_pose', self.amcl_pose_callback, 10)
        self.create_subscription(Float32MultiArray, 'spatial_resp', self.spatial_resp_callback, 10)
        self.get_action_cli = self.create_client(GetAction, 'get_action')
        self.goal_pose_pub = self.create_publisher(PoseStamped, 'goal_pose', 10) # navigationの目標位置を送信するためのpublisher
        self.bridge = CvBridge() # OpenCVとROSの画像を変換するためのクラス
        self.rate = 2 # Hz 制御周期
        # get_actionサーバーが立ち上がるまで待機
        while not self.get_action_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        # mapの読み込み
        yaml_path = Path.cwd() / "my_envs/map" / (MAP_NAME + ".yaml")
        with open(yaml_path, 'r') as file:
            map_data = yaml.safe_load(file)
            map_image = cv2.imread(str(Path.cwd() / "my_envs/map" / map_data["image"]), cv2.IMREAD_GRAYSCALE)
            self.resolution = map_data["resolution"] # 1pixelあたりのメートル数
            self.origin = np.array(map_data["origin"]) # mapの右下の隅のpose
        # 保存用の変数
        self.spatial_resp = None
        self.field_map = None
        self.robot_pose = None
        # 画像の初期化
        self.image = self.image = np.zeros((map_image.shape[0], map_image.shape[1], 3), dtype=np.float32) # 画像を格納する配列 (R:音源の存在確率, G:map, B:robot) 0-255
        self.image[:,:,0] = 255*THRESHOLD # Rチャンネルの値を初期化
        self.image[:,:,1] = 255 - map_image # Gチャンネルの値を初期化
        self.image[map_image == 205, 1] = 255 # 未知の領域を255にする
        # データが揃うまで待機
        while self.spatial_resp is None or self.robot_pose is None:
            print('waiting...')
            time.sleep(1)
        print('start')

    def loop(self):
        start_time = time.time()
        # 1. spatial_respをRチャンネルに描画
        points = np.array(np.meshgrid(np.arange(self.image.shape[0]), np.arange(self.image.shape[1]))).T.reshape(-1, 2) # 画像上の各pixelのマイクロフォンアレイからの角度を計算してanglesに格納
        point_angles = np.arctan2(points[:,0] - robot_pos[0], points[:,1] - robot_pos[1])*180/np.pi + 180
        point_angles = (point_angles + 360) % 360
        for i, point_angle in enumerate(point_angles):
            shift_value = 0.4 # 値を大きくするほど，確率分布が収束しにくくなる
            self.image[points[i,0], points[i,1], 0] *= max(self.spatial_resp[int(point_angle)] + shift_value, 0.95) # ピーキーな値の変動を抑える
        self.image[:,:,0] += 0.1 # 値が小さくなりすぎると更新が上手くいかなくなる
        self.image[self.image < 0] = 0.0
        self.image[self.image > 255] = 255.0
        # 2. ロボットの位置をBチャンネルにプロット
        self.image[:,:,2] = 0 # Bチャンネルの値を初期化
        robot_pos = (np.array([self.robot_pose.position.x, self.robot_pose.position.y]) - self.origin) / self.resolution
        robot_pos = robot_pos.astype(np.int)
        size = 2
        self.image[robot_pos[1]-size:robot_pos[1]+size, robot_pos[0]-size:robot_pos[0]+size, 2] = 255.0
        # 3. obs_imageをpublish
        obs_image = self.image.astype(np.uint8) # 画像をROSのImage型に変換
        obs_image = self.bridge.cv2_to_imgmsg(obs_image, encoding="bgr8")
        self.obs_image_pub.publish(obs_image)
        # 4. RチャンネルをOccupancyGrid型に変換してpublish
        sound_map = OccupancyGrid()
        sound_map.header.frame_id = "map"
        sound_map.info.resolution = self.resolution
        sound_map.info.width = self.image.shape[1]
        sound_map.info.height = self.image.shape[0]
        sound_map.info.origin.position.x = self.origin[0]
        sound_map.info.origin.position.y = self.origin[1]
        sound_map.info.origin.position.z = 0
        sound_map.info.origin.orientation.x = 0
        sound_map.info.origin.orientation.y = 0
        sound_map.info.origin.orientation.z = 0
        sound_map.info.origin.orientation.w = 1
        sound_map.data = (self.image[:,:,0]).astype(np.int8).reshape(-1).tolist()
        self.sound_map_pub.publish(sound_map)
        # 6. 行動の取得
        action = self.get_action()
        # 7. 行動の送信
        self.send_goal(action)
        # 8. 制御周期の調整
        delta_time = time.time() - start_time
        if delta_time < 1/self.rate:
            time.sleep(1/self.rate - delta_time)
        else:
            print('over time')

    def amcl_pose_callback(self, msg):
        # 位置情報の更新
        self.robot_pose = msg.pose.pose

    def spatial_resp_callback(self, msg):
        # DOA情報の更新
        self.spatial_resp = np.array(msg.data)

    def send_goal(self, action):
        # ナビゲーションの目標位置を送信
        # 自己位置に対して相対的な位置を設定
        goal = self.robot_pose
        goal.position.x += action[0]*MOVE_RANGE
        goal.position.y += action[1]*MOVE_RANGE
        goal_msg = PoseStamped()
        goal_msg.pose = goal
        goal_msg.header.frame_id = "map"
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        self.goal_pose_pub.publish(goal_msg)

    def get_action(self):
        # 行動の取得
        req = GetAction.Request()
        future = self.get_action_cli.call_async(req)
        # 結果を取得するまで待機
        rclpy.spin_until_future_complete(self, future)
        return [future.result().aciton0, future.result().aciton1]

# *************************************************************************************************
# メイン関数
# *************************************************************************************************
def main(args=None):
    rclpy.init(args=args)
    node = WrapNode()
    while rclpy.ok():
        node.loop()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()