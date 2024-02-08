import sys
import gym
import numpy as np
import gym.spaces
import gym
# https://kagglenote.com/ml-tips/my-environment-with-gym/

class MyEnv(gym.Env):
    def __init__(self):
        ACTION_NUM=2 #アクションの数
        self.action_space = gym.spaces.Discrete(ACTION_NUM) 
        
        #状態が3つの時で上限と下限の設定と仮定
        LOW=[0,0,0]
        HIGH=[1,1,1]
        self.observation_space = gym.spaces.Box(low=LOW, high=HIGH)

    def reset(self): # 環境の初期化
        return observation

    def step(self, action_index): # 1ステップ進める 状態，報酬，終了フラグを返す
        return observation, reward, done, {}

    def render(self): # 環境の描画
        pass