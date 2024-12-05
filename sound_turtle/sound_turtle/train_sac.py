import torch
import numpy as np
import gym
from collections import deque
import random
import wandb  # WandBのインポート
from my_envs.my_env import MyEnv
import torch.nn as nn
import torch.optim as optim
import sys
from gym import ObservationWrapper
from gym.spaces import Box
from PIL import Image

class RenderObservationWrapper(ObservationWrapper):
    def __init__(self, env, shape=(128, 128)):
        super().__init__(env)
        self.shape = shape
        self.observation_space = Box(
            low=0, high=255, shape=(shape[0], shape[1], 3), dtype=np.uint8
        )
    def observation(self, observation):
        frame = self.env.render()
        frame = Image.fromarray(frame)
        frame = frame.resize(self.shape)
        return np.array(frame, dtype=np.uint8)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=int(capacity))
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.stack(state), np.array(action), np.array(reward), np.stack(next_state), np.array(done)
    def __len__(self):
        return len(self.buffer)

class Encoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()
        assert len(obs_shape) == 3
        self.repr_dim = 103968 # Encoderの出力次元
        self.convnet = nn.Sequential(
            nn.Conv2d(obs_shape[-1], 32, 3, stride=2),
            nn.GELU(), nn.Conv2d(32, 32, 3, stride=1),
            nn.GELU(), nn.Conv2d(32, 32, 3, stride=1),
            nn.GELU(), nn.Conv2d(32, 32, 3, stride=1),
            nn.GELU()
        )
        self.optimizer = optim.AdamW(self.parameters(), lr=1e-4)
    def forward(self, obs):
        obs = obs.permute(0, 3, 1, 2)
        obs = obs - 0.5
        h = self.convnet(obs)
        h = h.reshape(h.shape[0], -1)
        return h

class Actor(nn.Module):
    def __init__(self, repr_dim, action_space: gym.Space, unit_dim=256):
        super().__init__()
        self.action_dim = action_space.shape[0] # 行動の次元
        self.action_center = (action_space.high + action_space.low) / 2 # 行動空間の中心
        self.action_scale = action_space.high - self.action_center # 行動空間のスケール
        self.action_center = torch.FloatTensor(self.action_center).to("cuda")
        self.action_scale = torch.FloatTensor(self.action_scale).to("cuda")
        self.hidden_layers = nn.Sequential(
            nn.Linear(repr_dim, unit_dim), nn.GELU(),
            nn.Linear(unit_dim, unit_dim), nn.GELU(),
            nn.Linear(unit_dim, unit_dim), nn.GELU()
        )
        self.mean_layer = nn.Linear(unit_dim, self.action_dim) # 平均
        self.std_layer = nn.Linear(unit_dim, self.action_dim) # 標準偏差
        self.optimizer = optim.AdamW(self.parameters(), lr=1e-4)
    def forward(self, input):
        h = self.hidden_layers(input)
        mean = self.mean_layer(h)
        log_std = self.std_layer(h)
        log_std = torch.clamp(log_std, -20, 2)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # detachせずにそのまま
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_center
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob

class Critic(nn.Module):
    def __init__(self, repr_dim, action_dim, unit_dim=64):
        super().__init__()
        self.hidden_layers1 = nn.Sequential(
            nn.Linear(repr_dim + action_dim, unit_dim), nn.GELU(),
            nn.Linear(unit_dim, unit_dim), nn.GELU(),
            nn.Linear(unit_dim, unit_dim), nn.GELU(),
            nn.Linear(unit_dim, 1)
        )
        self.hidden_layers2 = nn.Sequential(
            nn.Linear(repr_dim + action_dim, unit_dim), nn.GELU(),
            nn.Linear(unit_dim, unit_dim), nn.GELU(),
            nn.Linear(unit_dim, unit_dim), nn.GELU(),
            nn.Linear(unit_dim, 1)
        )
        self.optimizer = optim.AdamW(self.parameters(), lr=1e-4)
    def forward(self, repr, action):
        h = torch.cat([repr, action], dim=1)
        q1 = self.hidden_layers1(h)
        q2 = self.hidden_layers2(h)
        return q1, q2

class SAC:
    def __init__(self, action_space: gym.Space, obs_space: gym.Space, capacity=1e6, device='cuda', batch_size=256, gamma=0.99, tau=0.005):
        self.device = device
        self.encoder = Encoder(obs_space.shape).to(device)
        self.actor = Actor(self.encoder.repr_dim, action_space, unit_dim=64).to(device)
        self.actor_target = Actor(self.encoder.repr_dim, action_space, unit_dim=64).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic = Critic(self.encoder.repr_dim, action_space.shape[0]).to(device)
        self.critic_target = Critic(self.encoder.repr_dim, action_space.shape[0]).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.replay_buffer = ReplayBuffer(capacity)
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.reward_scale = 1.0
        self.alpha = 0.0
        self.target_entropy = -np.prod(action_space.shape)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = optim.AdamW([self.log_alpha], lr=1e-4)
    def train(self):
        self.encoder.train()
        self.actor.train()
        self.critic.train()
    def eval(self):
        self.encoder.eval()
        self.actor.eval()
        self.critic.eval()
    def select_action(self, obs):
        if len(obs.shape) == 3:
            obs = obs[np.newaxis]
        obs = torch.FloatTensor(obs).to(self.device)
        with torch.no_grad():
            repr = self.encoder(obs)
            action, _ = self.actor(repr)
        return action.cpu().numpy()[0]
    def update(self):
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device).unsqueeze(-1)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).to(self.device).unsqueeze(-1)
        # Critic & Encoder update
        repr = self.encoder(state)
        with torch.no_grad(): # Target Policy Smoothing
            next_repr = self.encoder(next_state)
            next_action, next_log_prob = self.actor_target(next_repr)
            next_action = next_action.to(self.device)
            next_q1, next_q2 = self.critic_target(next_repr, next_action)
            min_next_q = torch.min(next_q1, next_q2)
            target_q = reward + self.gamma * (1 - done) * (min_next_q - self.alpha * next_log_prob)
        q1, q2 = self.critic(repr, action)
        critic_loss = (q1 - target_q).pow(2).mean() + (q2 - target_q).pow(2).mean()
        self.critic.optimizer.zero_grad()
        self.encoder.optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic.optimizer.step()
        self.encoder.optimizer.step()
        # Actor update
        repr = self.encoder(state.detach())
        action, log_prob = self.actor(repr)
        action = action.to(self.device)
        q1, q2 = self.critic(repr, action)
        min_q = torch.min(q1, q2)
        actor_loss = (self.alpha * log_prob - min_q).mean()
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor.optimizer.step()
        # Target update
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        # Alpha update
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()
        # wandb.log({"train/critic_loss":critic_loss.item(), "train/actor_loss":actor_loss.item()})

if __name__ == "__main__":
    if '--seed' in sys.argv:
        seed = int(sys.argv[sys.argv.index('--seed') + 1])
    else:
        print("Please specify the seed. (--seed <seed>)")
    wandb.init(project="SoundTurtle", name=f"sac/run{seed}")
    
    # 自分の環境を使う場合
    env = MyEnv()
    print(env.action_space, env.observation_space)
    
    # gymの環境を使う場合
    # env = gym.make("MountainCarContinuous-v0", render_mode="rgb_array")
    # env = RenderObservationWrapper(env, shape=(128, 128))
    
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    episodes = 100000000 # とりあえず大きめに設定
    batch_size = 256
    episode_length = 0
    global_step = 0
    max_step = 100000 # こちらで制限
    sac = SAC(env.action_space, env.observation_space, device=device, batch_size=batch_size)
    sac.train()
    train_start = False
    for episode in range(episodes):
        episode_length = 0
        # state, _ = env.reset()
        state = env.reset()
        episode_reward = 0
        for t in range(200):
            global_step += 1
            episode_length += 1
            action = sac.select_action(state)
            next_state, reward, done, _ = env.step(action)
            # next_state, reward, terminated, truncated, _ = env.step(action)
            # done = terminated or truncated
            sac.replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            if len(sac.replay_buffer) > batch_size:
                if not train_start:
                    print("Train Start!")
                    train_start = True
                sac.update()
            if done:
                break
        if global_step >= max_step:
            break
        if train_start:
            wandb.log({"train/score":episode_reward, "train/length":episode_length})
        print(f"Episode: {episode}, Reward: {episode_reward}")