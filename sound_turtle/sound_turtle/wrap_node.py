# R1のピックアップ部に設置されたカメラの画像からボールの有無，ボールの色を検出するノード
# ライブラリのインポート
# *************************************************************************************************
# ROS関連のライブラリをインポート
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.task import Future
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid
from nav_msgs.action import NavigateToPose
from sound_turtle_msgs.srv import GetAction

# 画像処理関連のモジュールのインポート
from cv_bridge import CvBridge
import cv2

# その他
import numpy as np
import os
import yaml
from pathlib import Path
# *************************************************************************************************
# 定数の定義
# *************************************************************************************************
MOVE_RANGE = 0.2 # ロボットの移動範囲[m]
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
        self.navigate_to_pose_acli = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.bridge = CvBridge() # OpenCVとROSの画像を変換するためのクラス
        # get_actionサーバーが立ち上がるまで待機
        while not self.get_action_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        # mapの読み込み
        yaml_path = Path.cwd() / "my_envs/map/" / MAP_NAME + ".yaml"
        with open(yaml_path, 'r') as file:
            map_data = yaml.safe_load(file)
            map_image = cv2.imread(Path.cwd() + "/my_envs/map/" + map_data["image"], cv2.IMREAD_GRAYSCALE)
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
        while self.spatial_resp is None or self.field_map is None or self.robot_pose is None:
            print('waiting...')
            rclpy.sleep(0.1)
        self.create_timer(0.1, self.timer_callback)
        rclpy.sleep(1)
        # 初回の動作
        print('start')
        action = self.get_action() # 行動の取得
        self.send_goal(action) # 行動の送信

    def timer_callback(self):
        # sound_mapとobs_imageをpublish
        if self.image is not None:
            # 画像をROSのImage型に変換
            obs_image = self.image.astype(np.uint8)
            obs_image = self.bridge.cv2_to_imgmsg(obs_image, encoding="bgr8")
            self.obs_image_pub.publish(obs_image)
            # RチャンネルをOccupancyGrid型に変換
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

    def send_goal(self, action):
        # ナビゲーションの目標位置を送信
        # 自己位置に対して相対的な位置を設定
        goal = self.robot_pose
        goal.position.x += action[0]*MOVE_RANGE
        goal.position.y += action[1]*MOVE_RANGE
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.pose = goal
        self.navigate_to_pose_acli.wait_for_server()
        self.future = self.navigate_to_pose_acli.send_goal_async(goal_msg, feedback_callback=self.feedback_callback)
        self.future.add_done_callback(self.goal_response_callback)
        
    def feedback_callback(self, feedback_msg):
        # フィードバックのコールバック関数
        feedback = feedback_msg.feedback
        print(feedback)
        
    def goal_response_callback(self, future):
        # ゴールのコールバック関数
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            return
        self.get_logger().info('Goal accepted')
        self.result_future = goal_handle.get_result_async()
        self.result_future.add_done_callback(self.get_result_callback)
        
    def get_result_callback(self, future):
        """
        ROSのOccupancyGrid型からndarray配列にmap情報を変換。Gチャンネルにする
        ロボットの位置をBチャンネルにプロット
        spatial_respをRチャンネルに描画（これはシミュレータと同じ
        画像をROSのImage型に変換
        RチャンネルをOccupancyGrid型に変換
        """
        # ロボットの位置をBチャンネルにプロット
        self.image[:,:,2] = 0 # Bチャンネルの値を初期化
        robot_pos = (np.array([self.robot_pose.position.x, self.robot_pose.position.y]) - self.origin) / self.resolution
        robot_pos = robot_pos.astype(np.int)
        size = 2
        self.image[robot_pos[1]-size:robot_pos[1]+size, robot_pos[0]-size:robot_pos[0]+size, 2] = 255.0
        
        # spatial_respをRチャンネルに描画
        # 画像上の各pixelのマイクロフォンアレイからの角度を計算してanglesに格納
        points = np.array(np.meshgrid(np.arange(self.image.shape[0]), np.arange(self.image.shape[1]))).T.reshape(-1, 2)
        point_angles = np.arctan2(points[:,0] - robot_pos[0], points[:,1] - robot_pos[1])*180/np.pi + 180
        point_angles = (point_angles + 360) % 360
        for i, point_angle in enumerate(point_angles):
            shift_value = 0.4 # 値を大きくするほど，確率分布が収束しにくくなる
            self.image[points[i,0], points[i,1], 0] *= max(self.spatial_resp[int(point_angle)] + shift_value, 0.95) # ピーキーな値の変動を抑える
        self.image[:,:,0] += 0.1 # 値が小さくなりすぎると更新が上手くいかなくなる
        self.image[self.image < 0] = 0.0
        self.image[self.image > 255] = 255.0
        
        rclpy.sleep(1) # 1秒待機
        result = future.result().result
        self.get_logger().info('Result: {0}'.format(result))
        # 行動の取得
        action = self.get_action()
        self.send_goal(action) # 行動の送信
        
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
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()