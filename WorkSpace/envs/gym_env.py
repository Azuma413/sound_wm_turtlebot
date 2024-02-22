import sys
import gymnasium
import numpy as np
from gymnasium import spaces
import pyroomacoustics as pra
import matplotlib.pyplot as plt
from IPython.display import display, Audio
from scipy.io import wavfile

# https://kagglenote.com/ml-tips/my-environment-with-gym/

class MySimulator:
    def __init__(self, robot_pos=[0.2, 0.3], image_size=250, sound_locations = [[0.2, 0.4], [0.7, 0.5]]):
        ### パラメータの設定 ###
        # DOA推定のための設定
        self.c = 343.    # speed of sound
        self.fs = 16000  # サンプリングレート
        self.nfft = 256  # FFT size
        self.freq_range = [300, 3500]
        snr_db = 5.    # ガウスノイズのSNR
        # 部屋の形状
        self.room_dim = np.array([8, 6]) # 部屋の大まかな寸法
        self.corners = np.array([[0,0], [0,6], [5,6], [5,3], [5.1,3], [5.1,6], [8,6], [8,0]])  # [x,y]
        self.height = 3. # 天井の高さ
        # ロボットの初期設定
        self.center = ((self.room_dim - np.array([1, 1])) * np.array([robot_pos]) + np.array([0.5, 0.5])).reshape(-1) # ロボットの初期位置
        self.robot_angle = 0 # ロボットの角度[rad]0がx軸方向を指す
        self.robot_height = 0.3 # ロボットの高さ
        self.robot_vel = 0.1 # ロボットの最大速度[m/s]
        self.robot_ang_vel = 0.1 # ロボットの最大角速度[rad/s]
        self.mic_num = 6 # マイクロフォンアレイの数
        # LiDAR(LDS-01)のスペック
        self.distance_range = (120, 3500) # mm
        self.distance_accuracy = 10 # mm
        self.angular_range = 360 # degree
        self.angular_resolution = 1 # degree
        # 観測の設定
        self.max_field = 10 # 画像化するフィールドの大きさ[m]
        self.spacing = self.max_field/image_size # 1画素が示す空間の大きさ[m/pixel]
        self.image = np.zeros((image_size, image_size, 3), dtype=np.uint8) # 画像を格納する配列
        self.image[:,:,0] = int(255*0.5) # Rチャンネルの値を255*0.5で初期化
        self.delta_time = 0.5 # シミュレーションの時間間隔[s]
        # 音源の位置
        sound_height = 1.3
        self.sound_locations_2d = self.room_dim * np.array(sound_locations)
        self.sound_locations = np.concatenate([self.sound_locations_2d, sound_height * np.ones((len(self.sound_locations_2d), 1))], axis=1)
        sound_path = "D:/SourceCode/VScode/sound_turtlebot/WorkSpace/envs/SoundSource/sound_source.wav" # 音源のpath
        use_original_sound = True # Trueなら音源をそのまま使う。Falseならランダムな音源を生成する。
        # 画像作成の設定
        self.update_rate = 16 # 1回のシミュレーションで音源の存在確率を256段階中の最大何段階まで変化させるか
        self.doa_threshold = 0.5 # DOAの値がこの値を超えたら音源が存在すると判定
        
        ### 初期化処理 ###
        # ガウスノイズの設定
        self.sigma2 = 10**(snr_db / 10) / (4. * np.pi * 2.)**2
        # 音源を作成
        if use_original_sound:
            _, self.audio = wavfile.read(sound_path)
        else:
            self.audio = np.random.randn(16000)
            
    def update_robot_pos(self, action):
        """
        action: 標準化されたロボットの速度と角速度の配列
        """
        # ロボットの移動
        theta = np.pi - (self.robot_ang_vel*action[1])/2 - self.robot_angle
        chord = 2 * ((self.robot_vel*action[0])/(self.robot_ang_vel*action[1])) * np.sin((self.robot_ang_vel*action[1])/2)
        self.center[0] -= chord * np.cos(theta)
        self.center[1] += chord * np.sin(theta)
        
    def simulate(self):
        # 部屋をつくる
        aroom = pra.Room.from_corners(self.corners.T, fs=self.fs, materials=None, max_order=3, sigma2_awgn=self.sigma2, air_absorption=True)
        aroom.extrude(self.height)
        mic_locs = pra.circular_2D_array(center=self.center.tolist(), M=self.mic_num, phi0=0, radius=0.1)
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
        doa = pra.doa.algorithms['MUSIC'](mic_locs, self.fs, self.nfft, c=self.c, num_src=2, max_four=4)
        doa.locate_sources(X, freq_range=self.freq_range)
        spatial_resp = doa.grid.values
        #min_val = spatial_resp.min()
        max_val = spatial_resp.max()
        #spatial_resp = (spatial_resp - min_val) / (max_val - min_val)
        spatial_resp = (spatial_resp) / (max_val) # normalize to 0-1
        
        # 音源の方向を描画するgridの作成
        # 各グリッドの座標の中心点を計算
        x = np.arange(0, self.max_field, self.spacing) + self.spacing / 2
        y = np.arange(0, self.max_field, self.spacing) + self.spacing / 2
        points = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)
        center_ = self.max_field/2 - self.room_dim/2 + self.center
        angles = np.arctan2(points[:,1] - center_[1], points[:,0] - center_[0])*180/np.pi # 各点について、マイクロフォンアレイの座標からの角度を計算
        angles = np.round(angles) # 小数点以下を四捨五入
        angles[angles < 0] += 360 # 0から360度までの範囲にする
        # gridの各点について、anglesの値番目のdataを加算
        for i, angle in enumerate(angles):
            self.image[:,:,0][int(points[i,0]/self.spacing), int(points[i,1]/self.spacing)] += np.round((spatial_resp[int(angle)] - self.doa_threshold)*self.update_rate)
        # 画像の値を0-255にする
        self.image[self.image < 0] = 0
        self.image[self.image > 255] = 255

        # lidarのデータを描画するgridの作成
        point_list = []
        for angle in range(0, 180, self.angular_resolution):
            theta = np.deg2rad(angle + self.angular_resolution/2)
            points = [[1e5, 1e5], [-1e5, -1e5]]
            for i in range(len(self.corners)):
                epsilon = 1e-6
                a, b = self.center
                p, q = self.corners[i%len(self.corners)]
                r, s = self.corners[(i+1)%len(self.corners)]
                x = (b - s - a*np.tan(theta) + r*(q - s)/(p - r + epsilon))/((q - s)/(p - r + epsilon) - np.tan(theta))
                y = np.tan(theta)*x + b - a*np.tan(theta)
                if (x - p)*(x - r) <= epsilon and (y - q)*(y - s) <= epsilon: # 線分上にあるか
                    # x > center[0]の点からxが最も小さい点を選ぶ
                    if x >= self.center[0]:
                        if points[0][0] > x:
                            points[0] = [x, y]
                    # x < center[0]の点からxが最も大きい点を選ぶ
                    else:
                        if points[1][0] < x:
                            points[1] = [x, y]
            point_list.append(points[0])
            point_list.append(points[1])
        point_list_np = np.array(point_list) + self.max_field/2 - self.room_dim/2 # m
        # ポイントをグリッドに描画
        size = 1
        for point in point_list_np:
            x, y = point
            x = int(x / self.spacing)
            y = int(y / self.spacing)
            self.image[:,:,1] = 0 # 一旦リセット
            self.image[:,:,1][x-size:x+size, y-size:y+size] = 255

        # ロボットの位置を描画するgridの作成
        x, y = center_
        x = int(x / self.spacing)
        y = int(y / self.spacing)
        size = 2
        self.image[:,:,2] = 0 # 一旦リセット
        self.image[:,:,2][x-size:x+size, y-size:y+size] = 255
        
        # gridをプロット
        # plt.imshow(image[2].T, cmap='jet', interpolation='nearest', extent=[0, 10, 0, 10], origin='lower')

        # 画像を表示
        # plt.imshow(self.image.transpose([1,0,2]), interpolation='nearest', extent=[0, self.max_field, 0, self.max_field], origin='lower')
        pass

class MyEnv(gymnasium.Env):
    def __init__(self, env_config=None):
        super(MyEnv, self).__init__()
        """
        action_space:エージェントがとり得る行動空間
        observation_space:エージェントが観測する空間
        reward_range:報酬の最小値と最大値
        """
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.image_size = 128
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.image_size, self.image_size, 3), dtype=np.uint8)
        self.reward_range = [-1., 1.]
        self.my_sim = None
        # 確率範囲の絞り込みに対する報酬の重み(音源の位置を正しく推定する事への報酬の重みは，1からこの値を引いたものになる)
        self.distribution_reward_weight = 0.5
        self.max_episode_steps = 50
        self.episode_count = 0

    def reset(self):
        """
        環境を初期状態にして初期状態(state)の観測(observation)をreturnする
        """
        self.episode_count = 0
        robot_pos = np.random.rand(2).tolist()
        sound_locations = np.random.rand(2, 2).tolist()
        self.my_sim = MySimulator(robot_pos=robot_pos, image_size=self.image_size, sound_locations=sound_locations)
        self.my_sim.simulate()
        return self.my_sim.image
        # return self.my_sim.image, {}

    def step(self, action):
        """
        行動を受け取り行動後の環境状態(state)の観測(observation)・即時報酬(reward)・エピソードの終了判定(done)・情報(info)をreturnする
        """
        obs = None
        reward = 0
        done = False
        truncateds = False
        self.episode_count += 1
        self.my_sim.update_robot_pos(np.asarray(action))
        if self.my_sim.center[0] < 0 or self.my_sim.center[0] > self.my_sim.room_dim[0] or self.my_sim.center[1] < 0 or self.my_sim.center[1] > self.my_sim.room_dim[1]:
            # ロボットが壁にぶつかったら罰則を与えて終了
            obs = self.my_sim.image
            reward = -1
            # truncateds = True
            done = True
        else:
            # ロボットが壁にぶつかっていなければシミュレーションを実行
            self.my_sim.simulate()
            obs = self.my_sim.image
            # 報酬の計算
            # 音源の位置をできるだけ正確に推定することが報酬になる
            # 画像のRチャンネルのうち，値が255*0.5以下のピクセル数をカウントする
            n = np.sum(obs[:,:,0] < 255*0.5)
            x = 0.8 # tanhの定義域をどれだけシフトさせるか。1なら-1~1(報酬と罰則のバランス), 0なら0~2(報酬のみ)
            reward = np.tanh(2*n/(self.image_size**2)-x)*self.distribution_reward_weight
            # 音源の位置のpixelの値に応じて報酬を与える
            for point in self.my_sim.sound_locations_2d:
                x, y = np.array([self.my_sim.max_field/2, self.my_sim.max_field/2]) - self.my_sim.room_dim/2 + point # map上の座標に変換
                x = int(x / self.my_sim.spacing) # pixel座標に変換
                y = int(y / self.my_sim.spacing) # pixel座標に変換
                reward += obs[x, y, 0]/255*(1-self.distribution_reward_weight)/2
            if self.episode_count >= self.max_episode_steps:
                done = True
        info = {}
        return obs, reward, done, info
        # return obs, reward, done, truncateds, info

    def render(self, mode='rgb_array'):
        """
        modeで指定されたように描画もしは配列をreturnする
        humanなら描画し, rgb_arrayならそれをreturn
        """
        img = self.my_sim.image
        # 音源の位置を描画
        for point in self.my_sim.sound_locations_2d:
            x, y = np.array([self.my_sim.max_field/2, self.my_sim.max_field/2]) - self.my_sim.room_dim/2 + point # map上の座標に変換
            x = int(x / self.my_sim.spacing)
            y = int(y / self.my_sim.spacing)
            size = 1
            img[x-size:x+size, y-size:y+size, :] = 255
        return img
    
if __name__ == "__main__":
    env = MyEnv()
    obs = env.reset()
    obs, rewad, done, info = env.step([0.5, 0.5])
    print(obs.shape)
    print(rewad)