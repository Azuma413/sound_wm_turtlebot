# R1のピックアップ部に設置されたカメラの画像からボールの有無，ボールの色を検出するノード
# ライブラリのインポート
# *************************************************************************************************
# ROS関連のライブラリをインポート
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from sound_turtle_msgs.srv import GetAction
from ament_index_python.packages import get_package_share_directory
# 画像処理関連のモジュールのインポート
from cv_bridge import CvBridge
import cv2
# その他
import gym
from gym import spaces
import numpy as np
import cv2
import os
from dm_env import specs
# *************************************************************************************************
# 定数の定義
# *************************************************************************************************
# MODEL = "DreamerV3"
MODEL = "DrQ-v2"

DRQ_PATH = get_package_share_directory('sound_turtle') + "/weights/best.pt"
DREAMER_PATH = get_package_share_directory('sound_turtle') + "/weights/best.ckpt"
# *************************************************************************************************
# クラスの定義
# *************************************************************************************************
class ControlNode(Node):
    """
    全体を包括する制御用クラス
    受け取った画像はコールバック関数内で環境にセットされる
    get_actionサービスを受け取ると，agentを動かしてから，get_action関数を呼び出して行動を取得する
    行動をresponseとして返す
    """
    def __init__(self):
        super().__init__('control_node')
        # 変数
        self.image = None
        # ROSの設定
        self.subscription = self.create_subscription(Image, 'obs_image', self.image_callback, 10)
        self.create_service(GetAction, 'get_action', self.get_action_callback)
        # OpenCVの設定
        self.bridge = CvBridge()
        # 環境とagentの作成
        self.env = ROSEnv()
        if MODEL == "DrQ-v2":
            from drq_util import DrQAgent
            self.env = WrapDrQ(self.env)
            self.agent = DrQAgent(self.env) # DrQ-v2のエージェントを作成 いったん保留
        elif MODEL == "DreamerV3":
            from dreamer_util import DreamerV3Agent
            self.agent = DreamerV3Agent(self.env) # DreamerV3のエージェントを作成
            
    def image_callback(self, msg):
        """
        観測データを更新するコールバック関数
        """
        self.image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.env.set_obs(self.image)
    def get_action_callback(self, request, response):
        """
        行動を取得するサービスのコールバック関数
        """
        self.agent.action() # エージェントを動かす
        response.action0 = float(self.env.get_action()[0])
        response.action1 = float(self.env.get_action()[1])
        return response

class WrapDrQ(gym.Env):
    """
    DrQ-v2用に環境をラップするクラス
    """
    def __init__(self, env):
        self.env = env
    def reset(self):
        obs = self.env.reset()
        obs = np.transpose(obs, (2, 0, 1)) # [128, 128, 3]->[3, 128, 128]
        return {'observation': obs, 'reward': np.array([0.0], dtype=np.float32), 'discount': np.array([1.0], dtype=np.float32), 'done': False , 'action': np.array([0.0, 0.0], dtype=np.float32)}
    def step(self, action):
        obs, reward, done, _ = self.env.step(action)
        obs = np.transpose(obs, (2, 0, 1))
        return {'observation': obs, 'reward': np.array([reward], dtype=np.float32), 'discount': np.array([1.0], dtype=np.float32), 'done': done, 'action': np.array(action, dtype=np.float32)}
    def observation_spec(self):
        return specs.Array(shape=(self.env.observation_space.shape[2], self.env.observation_space.shape[0], self.env.observation_space.shape[1]), dtype=self.env.observation_space.dtype, name='observation')
    def action_spec(self):
        return specs.BoundedArray(shape=self.env.action_space.shape, dtype=self.env.action_space.dtype, name='action', minimum=-1.0, maximum=1.0)
    def render(self, mode='rgb_array'):
        return self.env.render()
    def set_obs(self, obs):
        self.env.set_obs(obs)
    def get_action(self):
        return self.env.get_action()
        
class ROSEnv(gym.Env):
    """
    ROSのデータを使って環境を作成するクラス
    self.obsの更新はset_obs()を用いて外部から行う
    get_action()から行動を取得する事ができる。
    """
    def __init__(self, env_config=None):
        super(ROSEnv, self).__init__()
        """
        action_space:エージェントがとり得る行動空間
        observation_space:エージェントが観測する空間
        reward_range:報酬の最小値と最大値
        """
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.image_size = 128
        self.observation_space = spaces.Box(low=0, high=1.0, shape=(self.image_size, self.image_size, 3), dtype=np.float32)
        self.reward_range = [-5., 5.]
        # 確率範囲の絞り込みに対する報酬の重み(音源の位置を正しく推定する事への報酬の重みは，1からこの値を引いたものになる)
        self.distribution_reward_weight = 0.4
        self.max_episode_steps = 100 # 50
        self.confidence_threshold = 0.7 # 音源の存在確率がこの値を超えたら音源が存在すると判定
        self.map_name = "327"
        # ロボットの軌跡
        self.robot_pos = np.array([0.0, 0.0])
        # 画像のリスト
        self.image_list = []
        # 描画用の変数
        self.save_video = True
        self.estimated_sound_location = []
        self.render_size = 480
        self.obs = np.zeros((self.image_size,self.image_size,3))
        self.aciton = None # 行動データを格納する変数

    def reset(self, seed=None, options=None):
        """
        環境を初期状態にして初期状態(state)の観測(observation)をreturnする
        """
        # 変数の初期化
        self.estimated_sound_location = []
        # 観測の取得
        obs = cv2.resize(self.obs, (self.image_size, self.image_size))/255.0
        # 画像の保存
        if self.save_video:
            self.image_list = []
            img = self.render()
            self.image_list.append(img)
        return obs.astype(np.float32)

    def step(self, action):
        """
        行動を受け取り行動後の環境状態(state)の観測(observation)・即時報酬(reward)・エピソードの終了判定(done)・情報(info)をreturnする
        """
        self.aciton = action
        # 変数の初期化
        reward = 0
        done = False
        info = {}
        # 観測の取得
        obs = self.obs/255.0
        points = np.argwhere(obs[:,:,0] > self.confidence_threshold) # 2次元配列のTrueの要素のインデックスを返す
        point = np.mean(points, axis=0)
        self.estimated_sound_location.append(point)
        obs = cv2.resize(self.obs, (self.image_size, self.image_size))/255.0
        # 画像の保存
        if self.save_video:
            img = self.render()
            self.image_list.append(img)
            # 終了時に動画を保存
            if done: 
                # image_listをmp4に変換して保存
                fourcc = cv2.VideoWriter.fourcc(*'mp4v')
                current_dir = os.path.dirname(os.path.abspath(__file__))
                path = f"video/video"
                path = os.path.join(current_dir, path) # 絶対pathに変換
                if not os.path.exists(path):
                    os.makedirs(path)
                i = 0
                while True:
                    filename = f'trajectory{i}.mp4'
                    filename = os.path.join(path, filename) # 絶対pathに変換
                    if os.path.exists(filename):
                        i += 1
                        continue
                    else:
                        out = cv2.VideoWriter(filename, fourcc, 8.0, (img.shape[0], img.shape[1]))
                        break
                for image in self.image_list:
                    # RGBからBGRに変換
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    out.write(image)
                out.release()
        return obs.astype(np.float32), reward, done, info

    def render(self, mode='rgb_array'):
        """
        modeで指定されたように描画,もしくは配列をreturnする
        humanなら描画し, rgb_arrayならそれをreturn
        """
        # 画像の取得
        img = np.array(self.obs, dtype=np.uint8)
        # 画像を高解像度化
        img = cv2.resize(img, (self.render_size, self.render_size), interpolation=cv2.INTER_NEAREST)
        # 倍率を保存
        scale = np.array([self.render_size/self.obs.shape[0], self.render_size/self.obs.shape[1]]) # 縦倍率，横倍率
        # Gチャンネルをリセットしてmapを描画
        gray = img[:,:,1]
        _, bin_img = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY) # 二値化
        contours, _ = cv2.findContours(bin_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE) # 輪郭抽出
        img[:,:,1] = 0
        cv2.drawContours(img, contours, -1, (0, 255, 0), 2) # 輪郭を描画
        # Bチャンネルの値の平均位置を取得し，ロボットの位置を描写
        points = np.argwhere(img[:,:,0] > 0)
        if len(points) > 0:
            x, y = np.mean(points, axis=0)
            img[:,:,2] = 0
            cv2.circle(img, (int(y), int(x)), 5, (0, 200, 200), -1)
        # self.confidence_threshold*255以上のRチャンネルの値を持つピクセルを囲う
        gray = img[:,:,0]
        _, bin_img = cv2.threshold(gray, int(self.confidence_threshold*255), 255, cv2.THRESH_BINARY) # 二値化
        contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # 輪郭抽出
        cv2.drawContours(img, contours, -1, (255, 255, 0), 1) # 輪郭を描画
        # 現在の音源の推定位置を描画
        if len(self.estimated_sound_location) > 0:
            point = (np.array(self.estimated_sound_location[-1])*scale).astype(np.int32)
            cv2.circle(img, (point[::-1]), 5, (255, 200, 100), -1)
        for i in range(len(self.estimated_sound_location) - 1):
            p1 = (self.estimated_sound_location[i]*scale).astype(np.int32)
            p2 = (self.estimated_sound_location[i+1]*scale).astype(np.int32)
            cv2.line(img, tuple(p1[::-1]), tuple(p2[::-1]), (255, 200, 100), 1)
        return img
    
    def set_obs(self, obs):
        """
        観測データをセットする関数
        """
        self.obs = obs
    def get_action(self):
        """
        行動データを取得する関数
        """
        return self.aciton
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