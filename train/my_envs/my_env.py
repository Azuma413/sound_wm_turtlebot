import sys
# import gymnasium as gym
# # from gymnasium import spaces
import gym
from gym import spaces
import numpy as np
import pyroomacoustics as pra
import matplotlib.pyplot as plt
# from IPython.display import display, Audio # 使っていない
from scipy.io import wavfile
import cv2
import os
import time
from dm_env import specs
import yaml

class WrapDrQ(gym.Env):
    def __init__(self, env):
        self.env = env
    def reset(self):
        obs = self.env.reset()
        return {'observation': obs, 'reward': np.array([0.0], dtype=np.float32), 'discount': np.array([1.0], dtype=np.float32), 'done': False , 'action': np.array([0.0, 0.0], dtype=np.float32)}
    def step(self, action):
        obs, reward, done, _ = self.env.step(action)
        return {'observation': obs, 'reward': np.array([reward], dtype=np.float32), 'discount': np.array([1.0], dtype=np.float32), 'done': done, 'action': np.array(action, dtype=np.float32)}
    def observation_spec(self):
        return specs.Array(shape=self.env.observation_space.shape, dtype=self.env.observation_space.dtype, name='observation')
    def action_spec(self):
        return specs.BoundedArray(shape=self.env.action_space.shape, dtype=self.env.action_space.dtype, name='action', minimum=-1.0, maximum=1.0)
    def render(self, mode='rgb_array'):
        return self.env.render()

class MySimulator:
    def __init__(self, map_name="327", robot_pos=[0.2, 0.3], sound_locations = [[0.2, 0.4], [0.7, 0.5]], threshold=0.5):
        ### パラメータの設定 ###
        # DOA推定のための設定
        self.fs = 16000  # サンプリングレート
        self.nfft = 256  # FFT size
        self.freq_range = [300, 3500] # 推定する周波数帯域
        # mapの読み込み
        yaml_path = os.path.dirname(os.path.abspath(__file__)) + "/map/" + map_name + ".yaml"
        with open(yaml_path, 'r') as file:
            map_data = yaml.safe_load(file)
            self.map_image = cv2.imread(os.path.dirname(os.path.abspath(__file__)) + "/map/" + map_data["image"], cv2.IMREAD_GRAYSCALE)
            self.resolution = map_data["resolution"] # 1pixelあたりのメートル数
            self.origin = np.array(map_data["origin"]) # mapの右下の隅のpose
        # シミュレーションの設定
        # 部屋の形状はmapデータを参照して適切に決める。音の反響はある程度割り切る。
        self.corners = np.array([[0,0], [self.map_image.shape[0]*self.resolution,0], [self.map_image.shape[0]*self.resolution,self.map_image.shape[1]*self.resolution], [0,self.map_image.shape[1]*self.resolution]])  # [x,y] 壁がない部屋
        self.room_dim = np.array([max(self.corners[:,0]), max(self.corners[:,1])]) # 部屋の大まかな寸法
        self.height = 3. # 天井の高さ
        # ロボットの初期設定
        self.center = ((self.room_dim - np.array([1, 1])) * np.array([robot_pos]) + np.array([0.5, 0.5])).reshape(-1) # ロボットの初期位置
        self.robot_height = 0.3 # ロボットの高さ
        self.move_range = 0.5 # ロボットの移動範囲[m]
        self.mic_num = 8 # マイクロフォンアレイの数
        # 観測の設定
        self.image = np.zeros((self.map_image.shape[0], self.map_image.shape[1], 3), dtype=np.float32) # 画像を格納する配列 (R:音源の存在確率, G:map, B:robot) 0-255
        self.image[:,:,0] = 255*threshold # Rチャンネルの値を初期化
        self.image[:,:,1] = map_data # Gチャンネルの値を初期化
        # 音源の位置
        sound_height = 1.3
        generate_range = 1.0 # mapの外側からどれだけ離れた位置に音源を生成するか
        self.sound_locations_2d = (self.room_dim - np.array([2*generate_range, 2*generate_range])) * np.array(sound_locations) + np.array([generate_range, generate_range]) # 音源の位置
        self.sound_locations = np.concatenate([self.sound_locations_2d, sound_height * np.ones((len(self.sound_locations_2d), 1))], axis=1)
        use_original_sound = False # Trueなら音源をそのまま使う。Falseならランダムな音源を生成する。 
        # 音源を作成
        if use_original_sound:
            sound_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "SoundSource/sound_source.wav")
            _, self.audio = wavfile.read(sound_path)
        else:
            self.audio = np.random.randn(16000)
    
    def coord2pixel(self, coord):
        """
        coord: 2次元座標
        ROSの座標系から画像の座標系に変換する
        """
        u = self.map_image.shape[1] - int((coord[1] - self.origin[1])/self.resolution)
        v = self.map_image.shape[0] - int((coord[0] - self.origin[0])/self.resolution)
        return np.array([u, v])
    
    def pixel2coord(self, pixel):
        """
        pixel: 2次元座標
        画像の座標系からROSの座標系に変換する
        """
        x = self.origin[0] + (self.map_image.shape[0] - pixel[1])*self.resolution
        y = self.origin[1] + (self.map_image.shape[1] - pixel[0])*self.resolution
        return np.array([x, y])

    def update_robot_pos(self, action):
        """
        action: 標準化されたロボットの速度と角速度の配列
        """
        new_center = self.center.copy()
        new_center[0] -= self.move_range*action[0]
        new_center[1] += self.move_range*action[1]
        # self.centerとnew_centerの間に壁があるかどうかを調べる
        start = self.coord2pixel(self.center)
        end = self.coord2pixel(new_center)
        diff = end - start
        if np.abs(diff[0]) > np.abs(diff[1]):
            step = np.abs(diff[0])
        else:
            step = np.abs(diff[1])
        for i in range(step):
            x = int(start[0] + i*diff[0]/step)
            y = int(start[1] + i*diff[1]/step)
            if self.map_image[y, x] == 0 or self.map_image[y, x] == 205: # 0は障害物，205は未知の領域
                return False
        self.center = new_center
        return True

    def simulate(self):
        aroom = pra.Room.from_corners(self.corners.T, fs=self.fs, materials=None, max_order=3, sigma2_awgn=10**(5 / 10) / (4. * np.pi * 2.)**2, air_absorption=True)
        aroom.extrude(self.height)
        mic_locs = pra.circular_2D_array(center=self.center.tolist(), M=self.mic_num, phi0=0, radius=0.035)
        mic_locs_z = np.concatenate((mic_locs, np.ones((1, mic_locs.shape[1]))*self.robot_height), axis=0)
        aroom.add_microphone_array(mic_locs_z)
        for i in range(self.sound_locations.shape[0]):
            # エラーをキャッチする
            try:
                aroom.add_source(self.sound_locations[i], signal=self.audio) # 運が悪いと音源が壁の中に生成されることがある
            except:
                self.sound_locations[i] = np.concatenate([np.random.rand(2) * self.room_dim, [1.3]]) # 流石に2連続で壁の中に音源が生成されることはない...はず
                aroom.add_source(self.sound_locations[i], signal=self.audio)
        # シミュレーションの実行
        aroom.simulate()
        X = pra.transform.stft.analysis(aroom.mic_array.signals.T, self.nfft, self.nfft // 2)
        X = X.transpose([2, 1, 0])
        # DOAの計算
        doa = pra.doa.algorithms['MUSIC'](mic_locs, self.fs, self.nfft, c=343., num_src=2, max_four=4)
        doa.locate_sources(X, freq_range=self.freq_range)
        spatial_resp = doa.grid.values # 標準化をなくしている
        spatial_resp /= 1.1 # 2.0 良い感じに調整している
        # robotの画像上の座標
        robot_pixel = self.coord2pixel(self.center)

        # Rチャンネルの値を更新
        # 画像上の各pixelのマイクロフォンアレイからの角度を計算してanglesに格納
        points = np.array(np.meshgrid(np.arange(self.map_image.shape[0]), np.arange(self.map_image.shape[1]))).T.reshape(-1, 2)
        point_angles = np.arctan2(points[:,1] - robot_pixel[1], points[:,0] - robot_pixel[0])*180/np.pi
        point_angles = (point_angles + 360) % 360
        for i, point_angle in enumerate(point_angles):
            shift_value = 0.2 # 値を大きくするほど，確率分布が収束しにくくなる
            self.image[points[i,0], points[i,1], 0] *= spatial_resp[int(point_angle)] + shift_value
        self.image[:,:,0] += 0.1 # 値が小さくなりすぎると更新が上手くいかなくなる
        self.image[self.image < 0] = 0.0
        self.image[self.image > 255] = 255.0
        
        # Bチャンネルの値を更新
        self.image[:,:,2] = 0.0
        size = 2
        self.image[:,:,1:][robot_pixel[0]-size:robot_pixel[0]+size, robot_pixel[1]-size:robot_pixel[1]+size] = 255.0
    
class MyEnv(gym.Env):
    def __init__(self, env_config=None):
        super(MyEnv, self).__init__()
        """
        action_space:エージェントがとり得る行動空間
        observation_space:エージェントが観測する空間
        reward_range:報酬の最小値と最大値
        """
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.image_size = 128
        self.observation_space = spaces.Box(low=0, high=1.0, shape=(self.image_size, self.image_size, 3), dtype=np.float32)
        self.reward_range = [-5., 5.]
        self.my_sim = None
        # 確率範囲の絞り込みに対する報酬の重み(音源の位置を正しく推定する事への報酬の重みは，1からこの値を引いたものになる)
        self.distribution_reward_weight = 0.4
        self.max_episode_steps = 100 # 50
        self.episode_count = 0
        self.confidence_threshold = 0.7 # 音源の存在確率がこの値を超えたら音源が存在すると判定
        self.map_name = "327"
        # ロボットの軌跡
        self.trajectory = []
        # 画像のリスト
        self.image_list = []
        # 描画用の変数
        self.save_video = True
        self.reward_value = 0
        self.reward_sum = 0
        self.estimated_sound_location = []
        self.move_result = True
        self.render_size = 560

    def reset(self, seed=None, options=None):
        """
        環境を初期状態にして初期状態(state)の観測(observation)をreturnする
        """
        # 変数の初期化
        self.episode_count = 0
        self.reward_value = 0
        self.reward_sum = 0
        self.trajectory = []
        self.estimated_sound_location = []
        self.move_result = True
        # ロボットの初期位置と音源の位置をランダムに設定
        robot_pos = np.random.rand(2).tolist()
        sound_locations = np.random.rand(1, 2).tolist()
        # シミュレータの初期化
        self.my_sim = MySimulator(map_name=self.map_name, robot_pos=robot_pos, sound_locations=sound_locations, threshold=self.confidence_threshold)
        # シミュレーションの実行
        self.my_sim.simulate()
        # 観測の取得
        obs = cv2.resize(self.my_sim.image, (self.image_size, self.image_size))/255.0
        # 軌跡の記録
        self.trajectory.append(self.my_sim.coord2pixel(self.my_sim.center).tolist())
        # 画像の保存
        if self.save_video:
            self.image_list = []
            img = self.render()
            self.image_list.append(img)
        return obs

    def step(self, action):
        """
        行動を受け取り行動後の環境状態(state)の観測(observation)・即時報酬(reward)・エピソードの終了判定(done)・情報(info)をreturnする
        """
        # 変数の初期化
        obs = None
        reward = 0
        done = False
        truncateds = False
        info = {}
        # エピソード数の更新
        self.episode_count += 1
        # ロボットの位置の更新
        self.move_result = self.my_sim.update_robot_pos(np.asarray(action))
        # 軌跡の記録
        self.trajectory.append(self.my_sim.coord2pixel(self.my_sim.center).tolist())
        # シミュレーションの実行
        self.my_sim.simulate()
        # 観測の取得
        obs = self.my_sim.image/255.0
        # 報酬の計算
        if self.move_result == False:
            reward = -1.
        else:
            # 音源の位置の存在確率分布が収束度合に応じて報酬を与える
            n = np.sum(obs[:,:,0] < self.confidence_threshold)
            pixel_num = obs.shape[0]*obs.shape[1]
            x = 2 # tanhの定義域をどれだけシフトさせるか
            reward = np.tanh(4*(n - pixel_num*3/4)/(pixel_num/4)-x)*self.distribution_reward_weight
            
            # 推定された音源位置と真の位置の差に応じて報酬を与える
            points = np.argwhere(obs[:,:,0] > self.confidence_threshold) # 2次元配列のTrueの要素のインデックスを返す
            if len(points) == 0: # 音源を見失っていた場合は罰則を与える
                reward = -1.0
            else:
                # 見失っていなければ，音源との距離に応じた報酬を与える
                point = np.mean(points, axis=0)
                map_point = self.my_sim.pixel2coord(point)
                self.estimated_sound_location.append(point)
                sound_loc = np.mean(self.my_sim.sound_locations_2d)
                reward += (2*np.exp(-np.linalg.norm(map_point - sound_loc)*1.0) - 1)*(1-self.distribution_reward_weight) # 1.0はスケール

        if self.episode_count >= self.max_episode_steps:
            done = True
            
        obs = cv2.resize(self.my_sim.image, (self.image_size, self.image_size))/255.0
        # 報酬の記録
        self.reward_value = reward
        self.reward_sum += reward
        # 画像の保存
        if self.save_video:
            img = self.render()
            self.image_list.append(img)
            # 終了時に動画を保存
            if done or truncateds: 
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
        return obs, reward, done, info

    def render(self, mode='rgb_array'):
        """
        modeで指定されたように描画もしは配列をreturnする
        humanなら描画し, rgb_arrayならそれをreturn
        """
        # 画像の取得
        img = np.array(self.my_sim.image, dtype=np.uint8)
        # 画像を高解像度化
        img = cv2.resize(img, (self.render_size, self.render_size), interpolation=cv2.INTER_NEAREST)
        # Bチャンネルをリセットしてロボットの位置を描画
        img[:,:,2] = 0
        x, y = self.my_sim.center
        x = int((x + self.my_sim.max_field/2) / self.my_sim.spacing)*4 # ピクセル上の座標を取得
        y = int((y + self.my_sim.max_field/2) / self.my_sim.spacing)*4
        # img[x-8:x+8, y-8:y+8, 1] = 0 # ロボット周辺のGチャンネルをリセット
        marker_size = 15
        triangle = np.array([
                            [y + marker_size*np.sin(self.my_sim.robot_angle), x + marker_size*np.cos(self.my_sim.robot_angle)],
                            [y + marker_size*np.sin(self.my_sim.robot_angle + 2/3*np.pi), x + marker_size*np.cos(self.my_sim.robot_angle + 2/3*np.pi)],
                            [y + marker_size*np.sin(self.my_sim.robot_angle - 2/3*np.pi), x + marker_size*np.cos(self.my_sim.robot_angle - 2/3*np.pi)]
                            ], dtype=np.int32)
        cv2.fillPoly(img, [triangle.reshape((-1, 1, 2))], (0, 200, 200))
        # 観測の視界を描画 x, yを中心とする1辺がself.image_size*4の正方形を描画
        # cv2.rectangle(img, (y-self.image_size*2, x-self.image_size*2), (y+self.image_size*2, x+self.image_size*2), (255, 0, 200), 1)
        # 音源の位置を描画
        for point in self.my_sim.sound_locations_2d:
            x, y = np.array([self.my_sim.max_field/2, self.my_sim.max_field/2]) + point # map上の座標に変換
            x = int(x / self.my_sim.spacing)
            y = int(y / self.my_sim.spacing)
            cv2.circle(img, (y*4, x*4), 5, (255, 255, 255), -1)
        # self.confidence_threshold*255以上のRチャンネルの値を持つピクセルを囲う
        gray = img[:,:,0]
        _, bin_img = cv2.threshold(gray, int(self.confidence_threshold*255), 255, cv2.THRESH_BINARY) # 二値化
        contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # 輪郭抽出
        cv2.drawContours(img, contours, -1, (255, 255, 0), 1) # 輪郭を描画
        # ロボットの軌跡を描画
        for i in range(len(self.trajectory) - 1):
            p1 = self.trajectory[i]
            p2 = self.trajectory[i+1]
            x1, y1 = p1
            x2, y2 = p2
            x1 = int((x1 + self.my_sim.max_field/2) / self.my_sim.spacing)*4
            y1 = int((y1 + self.my_sim.max_field/2) / self.my_sim.spacing)*4
            x2 = int((x2 + self.my_sim.max_field/2) / self.my_sim.spacing)*4
            y2 = int((y2 + self.my_sim.max_field/2) / self.my_sim.spacing)*4
            cv2.line(img, (y1, x1), (y2, x2), (0, 200, 200), 1)
        # 現在の音源の推定位置を描画
        if len(self.estimated_sound_location) > 0:
            # self.estimated_sound_locationをndarrayのintに変換
            point = (np.array(self.estimated_sound_location[-1])*4).astype(np.int32)
            cv2.circle(img, (point[1], point[0]), 5, (255, 200, 100), -1)
        for i in range(len(self.estimated_sound_location) - 1):
            p1 = (np.array(self.estimated_sound_location[i])*4).astype(np.int32)
            p2 = (np.array(self.estimated_sound_location[i+1])*4).astype(np.int32)
            cv2.line(img, (p1[1], p1[0]), (p2[1], p2[0]), (255, 200, 100), 1)
        # 画像をクリッピング
        # img = img[184:-184, 248:-248]
        # 報酬を描画
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        fontColor = (255, 255, 255)
        lineType = 2
        cv2.putText(img, f"rwd: {self.reward_value:.3f}", (20, img.shape[0]-15), font, fontScale, fontColor, lineType)
        cv2.putText(img, f"sum: {self.reward_sum:.3f}", (20, 30), font, fontScale, fontColor, lineType)
        if not self.move_result:
            cv2.putText(img, "hit wall", (int(img.shape[1]/2), img.shape[0]-15), font, fontScale, (255, 100, 0), lineType)
        img = cv2.resize(img, (560, 560), interpolation=cv2.INTER_NEAREST) # 画像を16の倍数にリサイズ
        return img

if __name__ == "__main__":
    env = MyEnv()
    obs = env.reset()
    obs, rewad, done, info = env.step([0.5, 0.5])
    print(obs.shape)
    print(rewad)