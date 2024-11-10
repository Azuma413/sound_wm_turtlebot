import torch
import numpy as np
import gym
from collections import deque
import random
import wandb  # WandBのインポート
from my_envs.my_env import MyEnv
import torch.nn as nn
import torch.optim as optim
# from gym.wrappers import gray_scale_observation, resize_observation, frame_stack
from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack
from PIL import Image

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.stack(state), np.array(action), np.array(reward), np.stack(next_state), np.array(done)

    def __len__(self):
        return len(self.buffer)

class Encoder(nn.Module):
    def __init__(self, img_channels):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(img_channels, 32, 3, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=2)
    def forward(self, state):
        state = state.permute(0, 3, 1, 2)
        x = torch.relu(self.conv1(state))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)
        return x

class Actor(nn.Module):
    def __init__(self, action_dim, max_action):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(64 * 15 * 15, 256)  # 画像サイズに応じて変更
        self.fc2 = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.max_action * torch.tanh(self.fc2(x))

class Critic(nn.Module):
    def __init__(self, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(64 * 15 * 15 + action_dim, 256)  # 画像サイズに応じて変更
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, action):
        x = torch.cat([x, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class SAC:
    def __init__(self, img_channels, action_dim, max_action):
        self.actor = ConvActor(img_channels, action_dim, max_action).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic1 = ConvCritic(img_channels, action_dim).to(device)
        self.critic2 = ConvCritic(img_channels, action_dim).to(device)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=3e-4)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=3e-4)

        self.target_critic1 = ConvCritic(img_channels, action_dim).to(device)
        self.target_critic2 = ConvCritic(img_channels, action_dim).to(device)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        self.replay_buffer = ReplayBuffer(100000)
        self.max_action = max_action
        self.discount = 0.99
        self.tau = 0.005
        self.alpha = 0.2

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, batch_size=256):
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)

        state = torch.FloatTensor(state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        done = torch.FloatTensor(done).to(device)

        with torch.no_grad():
            next_action = self.actor(next_state)
            next_q1 = self.target_critic1(next_state, next_action)
            next_q2 = self.target_critic2(next_state, next_action)
            next_q = torch.min(next_q1, next_q2).T
            log_prob = -self.actor(next_state).mean()  # Approximate log probability
            target_q = reward + (1 - done) * self.discount * (next_q - self.alpha * log_prob)
            target_q = target_q.T

        current_q1 = self.critic1(state, action)
        current_q2 = self.critic2(state, action)
        critic1_loss = nn.MSELoss()(current_q1, target_q)
        critic2_loss = nn.MSELoss()(current_q2, target_q)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        log_prob = -self.actor(state).mean()  # Approximate log probability
        actor_loss = (self.alpha * log_prob - self.critic1(state, self.actor(state))).mean()
        # actor_loss = -self.critic1(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        for param, target_param in zip(self.critic1.parameters(), self.target_critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.critic2.parameters(), self.target_critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # WandBに損失を記録
        wandb.log({
            "critic1_loss": critic1_loss.item(),
            "critic2_loss": critic2_loss.item(),
            "actor_loss": actor_loss.item(),
        })

if __name__ == "__main__":
    # WandBの初期化
    seed = 0
    wandb.init(project="sound_turtle", name=f"sac/run{seed}")  # 'your_wandb_username'を適切に変更
    
    # env = MyEnv()
    env = gym.make("CarRacing-v2")
    env = ResizeObservation(env, (128, 128))
    env = FrameStack(env, num_stack=4)
    
    torch.manual_seed(seed)
    img_channels = env.observation_space.shape[2]  # 画像のチャンネル数 (例: RGBなら3)
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sac = SAC(img_channels, action_dim, max_action)
    episodes = 100000000
    batch_size = 256
    episode_length = 0
    global_step = 0
    max_step = 100000
    for episode in range(episodes):
        episode_length = 0
        state = env.reset()
        state = np.array(state) # 後で消す
        print("state: ", state)
        episode_reward = 0
        for t in range(200):
            global_step += 1
            episode_length += 1
            action = sac.select_action(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.array(next_state) # 後で消す
            sac.replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            if len(sac.replay_buffer) > batch_size:
                sac.train(batch_size)
            else:
                sac.train(len(sac.replay_buffer))
            if done:
                break
            if global_step >= max_step:
                break
        # エピソード終了時に報酬をWandBにログ
        wandb.log({"train/score":episode_reward, "train/length":episode_length})
        print(f"Episode: {episode}, Reward: {episode_reward}")
