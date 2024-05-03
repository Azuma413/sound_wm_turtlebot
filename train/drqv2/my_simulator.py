# px:mm = 1:1
# 理想的にはシミュレータとエージェントを完全に分離して，ROS上で学習を行えると良い。
# ロボットの位置やカーブの中心はmmで管理する。
# 画像の大きさは560*720とする。
# 観測として与える画像はロボットを中心とする64*64の画像とする。
# 円弧は以下のように描画する
# cv2.ellipse(img, center, axes, angle, startAngle, endAngle, color, thickness=1, lineType=cv2.LINE_8, shift=0)
# center: 中心座標, axes: 長軸と短軸の長さ, angle: 回転角度(基本的には0), startAngle: 描画する円弧の開始角度, endAngle: 描画する円弧の終了角度

import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
import cv2
from dm_env import specs
import pandas as pd

class MyEnv(gym.Env):
    def __init__(self, env_config=None):
        super(MyEnv, self).__init__()
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.obs_size = 64
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.obs_size, self.obs_size, 1), dtype=np.float32)
        self.reward_range = (-1, 1)
        self.simulator = None
        self.episode_length = 300
        self.count = 0
        
    def reset(self):
        # self.simulator = MySimulator(init_pos=[640, 200], obs_size=[self.obs_size, self.obs_size])
        self.simulator = MySimulator(obs_size=[self.obs_size, self.obs_size])
        obs, _, _ = self.simulator.simulate(np.array([0, 0]))
        self.count = 0
        obs = obs.astype(np.float32)
        # (64, 64) -> (64, 64, 1)
        obs = obs[None,:,:]
        return { 'observation': obs, 'reward': np.array([0.0], dtype=np.float32), 'discount': np.array([1.0], dtype=np.float32), 'done': False , 'action': np.array([0.0, 0.0], dtype=np.float32)}
        # return pd.DataFrame({ 'observation': obs, 'reward': np.array([0.0], dtype=np.float32), 'discount': np.array([1.0], dtype=np.float32), 'done': False , 'action': np.array([0.0, 0.0], dtype=np.float32)})
        # return obs, {}
    
    def step(self, action):
        self.count += 1
        obs, reward, done = self.simulator.simulate(action)
        obs = obs.astype(np.float32)
        # (64, 64) -> (64, 64, 1)
        obs = obs[None,:,:]
        if self.count >= self.episode_length:
            done = True
        return { 'observation': obs, 'reward': np.array([reward], dtype=np.float32), 'discount': np.array([1.0], dtype=np.float32), 'done': done , 'action': action.astype(np.float32)}
        # return pd.DataFrame({ 'observation': obs, 'reward': np.array([reward], dtype=np.float32), 'discount': np.array([1.0], dtype=np.float32), 'done': done , 'action': action.astype(np.float32)})
        # return obs, reward, done, {}
    
    def observation_spec(self):
        return specs.Array(shape=(1, self.obs_size, self.obs_size), dtype=np.float32, name='observation')
        # return specs.Array(shape=(self.obs_size, self.obs_size, 1), dtype=np.float32, name='observation')

    def action_spec(self):
        return specs.BoundedArray(shape=(2,), dtype=np.float32, name='action', minimum=-1, maximum=1)
    
    def render(self, mode='rgb_array'):
        return self.simulator.render()

class MySimulator:
    def __init__(self, init_pos=[360, 80], obs_size=[64, 64]):
        self.image_size = [560, 720]
        self.obs_size = obs_size
        self.robot_pos = np.array(init_pos)
        update_rate = 20 # Hz
        max_speed = 300 # mm/s
        self.max_length = max_speed / update_rate # mm 1ステップで進む最大距離
        self.centers = np.array([
            [520, 200],
            [520, 400],
            [360, 400],
            [200, 400],
            [200, 200]
        ])
        self.ranges = np.array([
            120,
            80,
            80,
            80,
            120
        ])
        self.areas = np.array([
            [[520, 720], [0, 400]],
            [[360, 520], [200, 400]],
            [[200, 520], [400, 560]],
            [[200, 360], [200, 400]],
            [[0, 200], [0, 400]],
            [[200, 520], [0, 200]]
        ])
        self.obs_map = self.image_generator()
        self.reward = 0
        self.is_in_area = False
        self.action = np.array([0, 0])
    
    def image_generator(self, line_thickness=20):
        """
        mapの画像を生成する
        """
        image = np.ones(self.image_size)
        angles = np.array([
            [-90, 90],
            [180, 270],
            [0, 180],
            [270, 360],
            [90, 270]
        ])
        for center, r, angle in zip(self.centers, self.ranges, angles):
            cv2.ellipse(image, tuple(center), (r, r), 0, angle[0], angle[1], 0, line_thickness)
        cv2.line(image, (200, 80), (520, 80), 0, line_thickness)
        return image
    
    def simulate(self, action):
        """
        action:正規化された[x, y]の速度ベクトル
        """
        prior_pos = self.robot_pos.copy()
        self.robot_pos += (self.max_length * action).astype(int)
        self.is_in_area = False
        self.reward = 0
        for i, area in enumerate(self.areas):
            if area[0][0] <= self.robot_pos[0] < area[0][1] and area[1][0] <= self.robot_pos[1] < area[1][1]:
                # robotがareaに入った時の処理
                length = 0
                error = 0
                if i == 5:
                    length = self.robot_pos[0] - prior_pos[0]
                    error = np.abs(self.robot_pos[1] - 80)
                else:
                    theta = []
                    theta.append(np.arctan2(self.robot_pos[1] - self.centers[i][1], self.robot_pos[0] - self.centers[i][0]))
                    theta.append(np.arctan2(prior_pos[1] - self.centers[i][1], prior_pos[0] - self.centers[i][0]))
                    for j in range(2):
                        if theta[j] < 0:
                            theta[j] += 2*np.pi
                    length = self.ranges[i] * (theta[0] - theta[1])
                    error = np.abs(np.sqrt((self.robot_pos[0] - self.centers[i][0])**2 + (self.robot_pos[1] - self.centers[i][1])**2) - self.ranges[i])
                    if i == 1 or i == 3:
                        length = -length
                self.reward = np.clip(length/self.max_length, -1, 1) - np.tanh(error/20)*2 # 3で1くらいになる
                self.is_in_area = True
                break
        obs = self.obs_map[int(self.robot_pos[1]-self.obs_size[1]//2):int(self.robot_pos[1]+self.obs_size[1]//2), int(self.robot_pos[0]-self.obs_size[0]//2):int(self.robot_pos[0]+self.obs_size[0]//2)]
        if not self.is_in_area:
            self.reward = -1
        if obs.shape != (self.obs_size[1], self.obs_size[0]):
            self.is_in_area = False
            self.reward = -1
            # obsのサイズが合わない場合は白で埋める
            obs = np.ones(self.obs_size)
        # rewardを-1,1にclipする
        self.reward = np.clip(self.reward, -1, 1)
        # まったく同じ行動を繰り返す場合は報酬を-1にする
        # if np.all(action == self.action):
        #     self.reward = -1
        self.action = action
        return obs, self.reward, not self.is_in_area
    
    def render(self):
        """
        学習の進捗確認用の画像を生成する
        """
        image = np.zeros([self.image_size[0], self.image_size[1], 3])
        image[:,:,:] = self.obs_map[:,:,None] * 255
        cv2.drawMarker(image, (360, 80), (0, 0, 0), cv2.MARKER_CROSS, 40, 10)
        cv2.circle(image, tuple(self.robot_pos), 10, (0, 255, 0), -1)
        # 観測範囲を表示
        cv2.rectangle(image, (self.robot_pos[0]-self.obs_size[0]//2, self.robot_pos[1]-self.obs_size[1]//2), (self.robot_pos[0]+self.obs_size[0]//2, self.robot_pos[1]+self.obs_size[1]//2), (0, 0, 255), 2)
        # 進行方向を表示
        cv2.arrowedLine(image, tuple(self.robot_pos), tuple(self.robot_pos + (self.max_length*self.action).astype(int)), (255, 0, 0), 2)
        # 上下反転
        image = cv2.flip(image, 0)
        # rewardを表示
        cv2.putText(image, 'reward: {:.2f}'.format(self.reward), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        # 座標を表示
        cv2.putText(image, 'x: {}, y: {}'.format(self.robot_pos[0], self.robot_pos[1]), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        # 衝突していたら文字を表示
        if not self.is_in_area:
            cv2.putText(image, 'out of area', (300, 280), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        return image