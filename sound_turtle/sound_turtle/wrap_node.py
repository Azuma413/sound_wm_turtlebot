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
from sound_turtle_msgs.srv import GetAction
from geometry_msgs.msg import Twist

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
MOVE_TIME = 1.0 # 直線移動の時間
THRESHOLD = 0.5 # 音源の存在判定の閾値
MAP_NAME = "327" # 使用するmapの名前
CONTROL_RATE = 100 # 制御周期[Hz]
MAX_VEL = 0.2 # 最大直進速度[m/s]
EPSILON_RAD = 1/180*np.pi # 角速度の閾値[rad/s]
RAD_KP = 0.5 # 角速度の比例ゲイン
RAD_KD = 0.1 # 角速度の微分ゲイン
# *************************************************************************************************
# クラスの定義
# *************************************************************************************************
class WrapNode(Node):
    def __init__(self):
        super().__init__('wrap_node')
        # ROSの設定
        self.obs_image_pub = self.create_publisher(Image, 'obs_image', 10) # 観測画像を送信するためのpublisher
        self.sound_map_pub = self.create_publisher(OccupancyGrid, 'sound_map', 10) # 音源の存在確率を送信するためのpublisher
        self.create_subscription(PoseWithCovarianceStamped, 'amcl_pose', self.amcl_pose_callback, 10) # 自己位置を取得するためのsubscriber
        self.create_subscription(Float32MultiArray, 'spatial_resp', self.spatial_resp_callback, 10) # DOAの結果を取得するためのsubscriber
        self.get_action_cli = self.create_client(GetAction, 'get_action') # 行動を取得するためのclient
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10) # ロボットの速度を送信するためのpublisher
        self.bridge = CvBridge() # OpenCVとROSの画像を変換するためのクラス
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
        self.spatial_resp = None # DOAの結果を格納
        self.robot_pose = None # ロボットの位置を格納
        self.target_rad = 0 # ロボットの目標角度を格納
        self.target_speed = 0 # ロボットの目標速度を格納
        self.move_state = False # 移動中ならTrue，回転中ならFalse
        self.move_time = 0 # 移動開始時刻
        self.prior_diff = 0 # 前回の角度の差分
        self.get_new_action = False # 新しい行動を取得できたらTrue
        # 画像の初期化
        self.image = self.image = np.zeros((map_image.shape[0], map_image.shape[1], 3), dtype=np.float32) # 画像を格納する配列 (R:音源の存在確率, G:map, B:robot) 0-255
        self.image[:,:,0] = 255*THRESHOLD # Rチャンネルの値を初期化
        self.image[:,:,1] = 255 - map_image # Gチャンネルの値を初期化
        self.image[map_image == 205, 1] = 255 # 未知の領域を255にする
        # データが揃うまで待機
        while self.spatial_resp is None or self.robot_pose is None:
            print('waiting...')
            time.sleep(1)
        # timerの設定
        self.create_timer(CONTROL_RATE, self.timer_callback)
        print('start')

    def timer_callback(self):
        self.send_obs()
        if self.move_state: # 移動中の場合
            if time.time() - self.move_time > MOVE_TIME and self.get_new_action: # 移動時間がMOVE_TIMEを超えた場合，移動を終了
                self.move_state = False
                return
            cmd_vel = Twist()
            cmd_vel.linear.x = self.target_speed
            self.cmd_vel_pub.publish(cmd_vel)
        else: # 移動中でない場合
            if abs(self.robot_pose.orientation.z - self.target_rad) < EPSILON_RAD: # 目標角度に到達した場合，移動を開始
                self.move_state = True
                self.move_time = time.time()
                self.get_new_action = False
                self.async_get_action()
                return
            # 角速度の計算
            diff = self.target_rad - self.robot_pose.orientation.z
            diff_diff = diff - self.prior_diff
            self.prior_diff = diff
            pd = RAD_KP * diff + RAD_KD * diff_diff
            cmd_vel = Twist()
            cmd_vel.angular.z = pd
            self.cmd_vel_pub.publish(cmd_vel)

    def async_get_action(self):
        req = GetAction.Request()
        future = self.get_action_cli.call_async(req)
        future.add_done_callback(self.get_action_callback)

    def get_action_callback(self, future):
        self.get_new_action = True
        result = future.result()
        self.target_rad = result.action0*2*np.pi
        self.target_speed = result.action1*MAX_VEL

    def send_obs(self):
        # 1. ロボットの位置をBチャンネルにプロット
        self.image[:,:,2] = 0 # Bチャンネルの値を初期化
        robot_pos = (np.array([self.robot_pose.position.x, self.robot_pose.position.y]) - self.origin) / self.resolution
        robot_pos = robot_pos.astype(np.int)
        size = 2
        self.image[robot_pos[1]-size:robot_pos[1]+size, robot_pos[0]-size:robot_pos[0]+size, 2] = 255.0
        # 2. spatial_respをRチャンネルに描画
        points = np.array(np.meshgrid(np.arange(self.image.shape[0]), np.arange(self.image.shape[1]))).T.reshape(-1, 2) # 画像上の各pixelのマイクロフォンアレイからの角度を計算してanglesに格納
        point_angles = np.arctan2(points[:,0] - robot_pos[0], points[:,1] - robot_pos[1])*180/np.pi + 180
        point_angles = (point_angles + 360) % 360
        for i, point_angle in enumerate(point_angles):
            shift_value = 0.4
            self.image[points[i,0], points[i,1], 0] *= max(self.spatial_resp[int(point_angle)] + shift_value, 0.95) # ピーキーな値の変動を抑える
        self.image[:,:,0] += 0.1 # 値が小さくなりすぎると更新が上手くいかなくなる
        self.image[self.image < 0] = 0.0
        self.image[self.image > 255] = 255.0
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

    def amcl_pose_callback(self, msg):
        # 位置情報の更新
        self.robot_pose = msg.pose.pose

    def spatial_resp_callback(self, msg):
        # DOA情報の更新
        self.spatial_resp = np.array(msg.data)
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