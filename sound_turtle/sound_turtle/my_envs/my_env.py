import gymnasium as gym # import gym
from gymnasium import spaces
import numpy as np
import pyroomacoustics as pra
from scipy.io import wavfile
import cv2
import os
from dm_env import specs
import yaml
import wandb

global_count = 0

class WrapDrQ(gym.Env):
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

class MySimulator:
    def __init__(self, map_name="327", robot_pos=[0.2, 0.3], sound_locations = [[0.2, 0.4], [0.7, 0.5]], threshold=0.5, room_num=0):
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
        use_random_obstacle = False # マップにランダムな障害物を追加するかどうか。
        if use_random_obstacle:
            max_obstacle_size = 5 # 障害物の最大サイズ（ピクセル）
            obstacle_size = np.random.randint(1, max_obstacle_size + 1, size=2)
            position = np.random.randint([self.map_image.shape[1]/4, self.map_image.shape[0]/4], [self.map_image.shape[1]*3/4, self.map_image.shape[0]*3/4])
            self.map_image[position[1]:position[1] + obstacle_size[1], position[0]:position[0] + obstacle_size[0]] = 0
        # シミュレーションの設定
        # map特有の設定
        self.sound_base_pos = np.array([0.5, 0.5])
        robot_base_pos = np.array([0.5, 0.5]) # 左上が原点 (縦，横)
        # robot_base_pos = np.array([0.4, 0.35]) # 左上が原点 (縦，横)
        # 観測の設定
        self.image = np.zeros((self.map_image.shape[0], self.map_image.shape[1], 3), dtype=np.float32) # 画像を格納する配列 (R:音源の存在確率, G:map, B:robot) 0-255
        self.image[:,:,0] = 255*threshold # Rチャンネルの値を初期化
        self.image[:,:,1] = 255 - self.map_image # Gチャンネルの値を初期化
        # 未知の領域を255にする
        self.image[self.map_image == 205, 1] = 255
        # 音源の位置
        self.sound_height = 1.3
        self.sound_range = 0.7 # 1.7
        robot_range = 0.
        self.sound_locations_2d = self.pixel2coord(self.map_image.shape*self.sound_base_pos) + self.sound_range*np.array(sound_locations) # 音源の位置
        # self.sound_locations = np.array([np.array([self.sound_locations_2d[0][0], self.sound_locations_2d[0][1], self.sound_height])])
        # sound_locations_2dの1次元目が2以上の場合は上のコードでは対応できないので以下のコードを使う
        self.sound_locations = np.array([np.array([self.sound_locations_2d[i][0], self.sound_locations_2d[i][1], self.sound_height]) for i in range(self.sound_locations_2d.shape[0])])
        use_original_sound = False # Trueなら音源をそのまま使う。Falseならランダムな音源を生成する。
        # 音源を作成
        if use_original_sound:
            sound_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "SoundSource/sound_source.wav")
            _, self.audio = wavfile.read(sound_path)
        else:
            self.audio = np.random.randn(16000)
        # ロボットの初期設定
        self.center = self.pixel2coord(self.map_image.shape*robot_base_pos) + robot_range*np.array(robot_pos) # ロボットの初期位置
        self.robot_height = 0.3 # ロボットの高さ
        self.move_range = 0.2 # ロボットの移動範囲[m]
        self.mic_num = 8 # マイクロフォンアレイの数
        if room_num == 3: # リアルな部屋の形状
            self.use_strict_room = True
            if self.use_strict_room:
                room_data = [[0,1.438],[0,2.368],[1,2.368],[1,3.438],[1.69,3.438],[1.69,4.3],[13,4.3],
                            [13,2.35],[8.5,2.35],[8.5,2.25],[13,2.25],[13,0],[4.155,0],[4.155,1.438]]
                theta = -135
                theta = theta*np.pi/180
                rot_mat = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta),np.cos(theta)]])
                room_data = np.array(room_data)
                room_data -= np.array([max(room_data[0])/2, max(room_data[1])/2])
                self.corners = np.dot(rot_mat, room_data.T).T
                self.corners = self.corners[:, ::-1]
                linear_motion = np.array([6.5,3.5])
                # linear_motion = np.array([6.5,3.1])
                self.corners += linear_motion
            else:
                room_dim0 = self.pixel2coord(self.map_image.shape) # 部屋の大まかな寸法
                room_dim1 = self.pixel2coord([0, 0])
                self.corners = np.array([room_dim1, [room_dim0[0],room_dim1[1]], room_dim0, [room_dim1[0],room_dim0[1]]])  # [x,y] 壁がない部屋
        else:
            image_ = np.ones_like(self.image[:,:,1])*255
            image_ = image_.astype(np.uint8)
            # xが上，yが左
            if room_num == 0: # 長方形の部屋
                self.corners = np.array([[0, 0], [0, 8], [5, 8], [5, 0]]) + np.array([0,-4])
                pixel_corners = np.array([self.coord2pixel(corner) for corner in self.corners])
                # 部屋の中を0で埋める
                cv2.fillPoly(image_, [pixel_corners], 0)
            elif room_num == 1: # L字の部屋
                self.corners = np.array([[0, 0], [0, 8], [7, 8], [7, 5], [5, 5], [5, 0]]) + np.array([-1,-4])
                pixel_corners = np.array([self.coord2pixel(corner) for corner in self.corners])
                # 部屋の中を0で埋める
                cv2.fillPoly(image_, [pixel_corners], 0)
            elif room_num == 2: # 仕切りのある部屋
                self.corners = np.array([[0, 0], [0, 8], [5, 8], [5, 4.5], [2, 4.5], [2, 4.3], [5, 4.3],[5, 0]]) + np.array([0,-4])
                pixel_corners = np.array([self.coord2pixel(corner) for corner in self.corners])
                # 部屋の中を0で埋める
                cv2.fillPoly(image_, [pixel_corners], 0)
            self.image[:,:,1] = image_.astype(np.float32)
            self.map_image = 255 - image_
        self.height = 3. # 天井の高さ
        self.count = 0
        self.T = 1.0 # 音源が一周する時間 0~1
        self.R = 1.0 # 音源の回転半径
        self.phi = 0.0 # 音源の初期角度
        self.sound_locs = None

    def coord2pixel(self, coord):
        """
        coord: 2次元座標
        ROSの座標系から画像の座標系に変換する
        """
        u = self.map_image.shape[0] - int((coord[1] - self.origin[1])/self.resolution)
        v = self.map_image.shape[1] - int((coord[0] - self.origin[0])/self.resolution)
        return np.array([u, v])

    def pixel2coord(self, pixel):
        """
        pixel: 2次元座標
        画像の座標系からROSの座標系に変換する
        """
        x = self.origin[0] + (self.map_image.shape[1] - pixel[1])*self.resolution
        y = self.origin[1] + (self.map_image.shape[0] - pixel[0])*self.resolution
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
            x = int(start[0] + i*diff[0]/step) + 1
            y = int(start[1] + i*diff[1]/step) + 1
            if self.map_image[x, y] < 210: # 0は障害物，205は未知の領域
                return False
        self.center = new_center
        return True

    def simulate(self):
        self.count += 1
        aroom = pra.Room.from_corners(self.corners.T, fs=self.fs, materials=None, max_order=3, sigma2_awgn=10**(5 / 10) / (4. * np.pi * 2.)**2, air_absorption=True)
        aroom.extrude(self.height)
        mic_locs = pra.circular_2D_array(center=self.center.tolist(), M=self.mic_num, phi0=0, radius=0.035)
        mic_locs_z = np.concatenate((mic_locs, np.ones((1, mic_locs.shape[1]))*self.robot_height), axis=0)
        aroom.add_microphone_array(mic_locs_z)
        # 音源の位置を回転させる
        # self.sound_locs = np.array([np.array([
        #             self.sound_locations_2d[0][0] + self.R*np.cos(2*np.pi*self.count/(self.T*100) + self.phi*2*np.pi),
        #             self.sound_locations_2d[0][1] + self.R*np.sin(2*np.pi*self.count/(self.T*100) + self.phi*2*np.pi),
        #             self.sound_height
        #         ])])
        # for i in range(self.sound_locs.shape[0]):
        #     aroom.add_source(self.sound_locs[i], signal=self.audio)
        #     success = False
        #     while not success:
        #         try:
        #             aroom.add_source(sound_locs[i], signal=self.audio) # 運が悪いと音源が壁の中に生成されることがある
        #             success = True
        #         except ValueError as e:
        #             print(f"例外が発生しました: {e}. 再試行します...")
        #             sound_locs = self.pixel2coord(self.map_image.shape*self.sound_base_pos) + self.sound_range*(np.random.rand(self.sound_locations_2d.shape[0], 2)*2 - 1) # 音源の位置
        #             sound_locs = np.array([np.array([self.sound_locations_2d[i][0], self.sound_locations_2d[i][1], self.sound_height]) for i in range(self.sound_locations_2d.shape[0])])
                    # self.sound_locations = np.array([np.array([self.sound_locations_2d[0][0], self.sound_locations_2d[0][1], self.sound_height])])
        for i in range(self.sound_locations.shape[0]):
            success = False
            while not success:
                try:
                    aroom.add_source(self.sound_locations[i], signal=self.audio) # 運が悪いと音源が壁の中に生成されることがある
                    success = True
                except ValueError as e:
                    print(f"例外が発生しました: {e}. 再試行します...")
                    self.sound_locations_2d = self.pixel2coord(self.map_image.shape*self.sound_base_pos) + self.sound_range*(np.random.rand(1, 2)*2 - 1) # 音源の位置
                    self.sound_locations = np.array([np.array([self.sound_locations_2d[0][0], self.sound_locations_2d[0][1], self.sound_height])])
        # シミュレーションの実行
        aroom.simulate()
        X = pra.transform.stft.analysis(aroom.mic_array.signals.T, self.nfft, self.nfft // 2)
        X = X.transpose([2, 1, 0])
        # DOAの計算
        doa = pra.doa.algorithms['MUSIC'](mic_locs, self.fs, self.nfft, c=343., num_src=2, max_four=4)
        doa.locate_sources(X, freq_range=self.freq_range)
        spatial_resp = doa.grid.values # 標準化をなくしている
        spatial_resp = spatial_resp # 30くらいまで行く
        # spatial_resp = (spatial_resp - spatial_resp.min())/(spatial_resp.max() - spatial_resp.min())
        # robotの画像上の座標
        robot_pixel = self.coord2pixel(self.center)

        # Rチャンネルの値を更新
        # 画像上の各pixelのマイクロフォンアレイからの角度を計算してanglesに格納
        points = np.array(np.meshgrid(np.arange(self.map_image.shape[0]), np.arange(self.map_image.shape[1]))).T.reshape(-1, 2)
        point_angles = np.arctan2(points[:,0] - robot_pixel[0], points[:,1] - robot_pixel[1])*180/np.pi + 180
        point_angles = (point_angles + 360) % 360
        # パラメータ
        A = 4.5 # spatial_respの値の内，情報を持たないと判断する値の最大値
        B = 0.05 # spatial_respの値をどれだけスケールするか
        C = 0.85 # 値を更新する際の最小値
        D = 3 # 最低値
        for i, point_angle in enumerate(point_angles):
            self.image[points[i,0], points[i,1], 0] *= max(spatial_resp[int(point_angle)], A)*B - A*B + C
        self.image[self.image[:,:,0] < D, 0] = D
        # Bチャンネルの値を更新
        # self.image[:,:,2] = 0
        # size = 3
        # self.image[
        #     max(0, robot_pixel[0]-size):min(self.map_image.shape[0], robot_pixel[0]+size), 
        #     max(0, robot_pixel[1]-size):min(self.map_image.shape[1], robot_pixel[1]+size), 2] = 255
        slice_idx = int(self.image.shape[1]/2)
        self.image[:,:slice_idx,2] = robot_pixel[0]/self.map_image.shape[0]*255.0
        self.image[:,slice_idx:,2] = robot_pixel[1]/self.map_image.shape[1]*255.0
        self.image = np.clip(self.image, 0, 255)

class MyEnv(gym.Env):
    def __init__(self, room_num=0, env_config=None):
        super(MyEnv, self).__init__()
        """
        action_space:エージェントがとり得る行動空間
        observation_space:エージェントが観測する空間
        reward_range:報酬の最小値と最大値
        """
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.image_size = 64 # 128
        self.observation_space = spaces.Box(low=0, high=1.0, shape=(self.image_size, self.image_size, 3), dtype=np.float32)
        self.reward_range = [-1., 1.]
        self.my_sim = None
        # 確率範囲の絞り込みに対する報酬の重み(音源の位置を正しく推定する事への報酬の重みは，1からこの値を引いたものになる)
        self.distribution_reward_weight = 0.4
        self.max_episode_steps = 100
        self.episode_count = 0
        self.confidence_threshold = 0.7 # 音源の存在確率がこの値を超えたら音源が存在すると判定
        self.map_name = "main"
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
        self.render_size = 480
        self.mark_corner = False # 部屋の形状をrenderに表示するか。
        self.drow_robot_trajectory = False # ロボットの軌跡を描画するか。
        self.drow_sound_trajectory = False # 音源の軌跡を描画するか。
        self.done_when_hit_wall = False # 壁に当たったときにエピソードを終了するか。
        self.log_video_step = 50 # 何エピソードごとにログを取るか。
        self.room_num = room_num

    def reset(self, seed=None, options=None):
        """
        環境を初期状態にして初期状態(state)の観測(observation)をreturnする
        """
        # 変数の初期化
        global global_count
        global_count += 1
        self.episode_count = 0
        self.reward_value = 0
        self.reward_sum = 0
        self.trajectory = []
        self.estimated_sound_location = []
        self.move_result = True
        # ロボットの初期位置と音源の位置をランダムに設定
        robot_pos = (np.random.rand(2)*2 - 1).tolist()
        sound_locations = (np.random.rand(1, 2)*2 - 1).tolist() # 音源が1つの場合
        # sound_locations = (np.random.rand(2, 2)*2 - 1).tolist() # 音源が2つの場合
        # シミュレータの初期化
        self.my_sim = MySimulator(map_name=self.map_name, robot_pos=robot_pos, sound_locations=sound_locations, threshold=self.confidence_threshold, room_num=self.room_num)
        self.my_sim.T = np.random.rand()*0.3 + 0.7
        self.my_sim.R = np.random.rand()*0.3 + 0.7
        self.my_sim.phi = np.random.rand()
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
        A = 0.001 # 分布報酬用のパラメータ 大きいほど報酬が厳しくなる。
        B = 0.5 # 推定報酬用のパラメータ 大きいほど報酬が厳しくなる。
        # 音源の位置の存在確率分布が収束度合に応じて報酬を与える
        n = np.sum(obs[:,:,0] > self.confidence_threshold)
        reward = (2*np.exp(-n*A) - 1)*self.distribution_reward_weight
        # print("分布報酬：", 2*np.exp(-n*A) - 1)
        points = np.argwhere(obs[:,:,0] > self.confidence_threshold) # 2次元配列のTrueの要素のインデックスを返す
        if len(points) == 0: # 音源を見失っていた場合は罰則を与える
            reward = -1.0
        else:
            # 見失っていなければ，音源との距離に応じた報酬を与える
            point = np.mean(points, axis=0)
            map_point = self.my_sim.pixel2coord(point)
            self.estimated_sound_location.append(point)
            sound_loc = self.my_sim.sound_locations_2d
            reward += (2*np.exp(-np.linalg.norm(map_point - sound_loc)*B) - 1)*(1-self.distribution_reward_weight)
            # print("推定報酬：",2*np.exp(-np.linalg.norm(map_point - sound_loc)*B) - 1)

        if self.move_result == False:
            reward = -1.
            done = True if self.done_when_hit_wall else False
        if self.episode_count >= self.max_episode_steps:
            done = True
            
        obs = cv2.resize(self.my_sim.image, (self.image_size, self.image_size))/255.0
        # 報酬の記録
        self.reward_value = reward
        self.reward_sum += reward
        # 画像の保存
        if self.save_video and (global_count % self.log_video_step == 0):
            img = self.render()
            self.image_list.append(img)
            # 終了時に動画を保存
            if done or truncateds:
                video = np.array(self.image_list)
                # THWC -> TCHW
                video = video.transpose(0,3,1,2)
                wandb.log({"video": wandb.Video(video)})
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
        # 倍率を保存
        scale = np.array([self.render_size/self.my_sim.image.shape[0], self.render_size/self.my_sim.image.shape[1]]) # 縦倍率，横倍率
        # Gチャンネルをリセットしてmapを描画
        gray = img[:,:,1]
        _, bin_img = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY) # 二値化
        contours, _ = cv2.findContours(bin_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE) # 輪郭抽出
        img[:,:,1] = 0
        cv2.drawContours(img, contours, -1, (0, 255, 0), 2) # 輪郭を描画
        # Bチャンネルをリセットしてロボットの位置を描画
        img[:,:,2] = 0
        x, y = self.my_sim.coord2pixel(self.my_sim.center)*scale # ピクセル上の座標を取得
        cv2.circle(img, (int(y), int(x)), 5, (0, 200, 200), -1)
        # 音源の位置を描画
        for point in self.my_sim.sound_locations_2d:
            x, y = self.my_sim.coord2pixel(point)*scale
            cv2.circle(img, (int(y), int(x)), 5, (255, 255, 255), -1)
        # for point in self.my_sim.sound_locs:
        #     point = point[:2]
        #     x, y = self.my_sim.coord2pixel(point)*scale
        #     cv2.circle(img, (int(y), int(x)), 5, (255, 255, 255), -1)
        # self.confidence_threshold*255以上のRチャンネルの値を持つピクセルを囲う
        gray = img[:,:,0]
        _, bin_img = cv2.threshold(gray, int(self.confidence_threshold*255), 255, cv2.THRESH_BINARY) # 二値化
        contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # 輪郭抽出
        cv2.drawContours(img, contours, -1, (255, 255, 0), 1) # 輪郭を描画
        # ロボットの軌跡を描画
        if self.drow_robot_trajectory:
            for i in range(len(self.trajectory) - 1):
                p1 = (np.array(self.trajectory[i])*scale).astype(np.int32)
                p2 = (np.array(self.trajectory[i+1])*scale).astype(np.int32)
                cv2.line(img, tuple(p1[::-1]), tuple(p2[::-1]), (0, 200, 200), 1)
        # 現在の音源の推定位置を描画
        if len(self.estimated_sound_location) > 0:
            # self.estimated_sound_locationをndarrayのintに変換
            point = (np.array(self.estimated_sound_location[-1])*scale).astype(np.int32)
            cv2.circle(img, (point[::-1]), 5, (255, 255, 0), -1)
        if self.drow_sound_trajectory:
            for i in range(len(self.estimated_sound_location) - 1):
                p1 = (self.estimated_sound_location[i]*scale).astype(np.int32)
                p2 = (self.estimated_sound_location[i+1]*scale).astype(np.int32)
                cv2.line(img, tuple(p1[::-1]), tuple(p2[::-1]), (255, 200, 100), 1)
        # self.my_sim.cornersを描画
        if self.mark_corner:
            img_corners = []
            for corner in self.my_sim.corners:
                x, y = self.my_sim.coord2pixel(corner)*scale
                img_corners.append([x, y])
            img_corners = np.array(img_corners).astype(np.int32)
            cv2.drawContours(img, [img_corners], -1, (255, 0, 255), 2)
        # 報酬を描画
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        fontColor = (255,255,255)
        lineType = 2
        cv2.putText(img, f"rwd: {self.reward_value:.3f}", (20, img.shape[0]-15), font, fontScale, fontColor, lineType)
        cv2.putText(img, f"sum: {self.reward_sum:.3f}", (20, 30), font, fontScale, fontColor, lineType)
        if not self.move_result:
            cv2.putText(img, "hit wall", (int(img.shape[1]/2), img.shape[0]-15), font, fontScale, (255, 100, 0), lineType)
        return img

if __name__ == "__main__":
    env = MyEnv()
    obs = env.reset()
    obs, rewad, done, info = env.step([0.5, 0.5])
    print(obs.shape)
    print(rewad)