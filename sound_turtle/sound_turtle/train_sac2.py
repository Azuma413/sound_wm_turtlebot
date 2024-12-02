import ray
import ray.rllib
from ray.rllib.algorithms.sac import SACConfig
from my_envs.my_env import MyEnv
import wandb
import sys
import gym

class WrapSAC(gym.Env):
    def __init__(self, env):
        self.env = env
        # gym.spaces.Boxをray.rllib.utils.spaces.Boxに変換
        self.observation_space = ray.rllib.utils.spaces.Box(
            low=env.observation_space.low,
            high=env.observation_space.high,
            shape=env.observation_space.shape,
            dtype=env.observation_space.dtype
        )
        self.action_space = ray.rllib.utils.spaces.Box(
            low=env.action_space.low,
            high=env.action_space.high,
            shape=env.action_space.shape,
            dtype=env.action_space.dtype
        )
        
    def reset(self):
        obs = self.env.reset()
        return obs
    def step(self, action):
        obs, reward, done, _ = self.env.step(action)
        return obs, reward, done, {}
    def render(self, mode='rgb_array'):
        return self.env.render()

def main(seed):
    # Rayの初期化
    ray.init()
    env_ = MyEnv()
    env = WrapSAC(env_)
    obs_space = env.observation_space
    act_space = env.action_space
    # SACの設定
    config = (
        SACConfig()
        .environment(env=WrapSAC(env_), render_env=False)
        .rollouts(num_rollout_workers=1)  # 並列ワーカー数
        .framework("torch")  # PyTorchを使用
        .resources(num_gpus=1)  # GPUを使用する場合は適切に設定
        .training(model={
            "conv_filters": [
                [32, [3, 3], 2],
                [32, [3, 3], 1],
                [32, [3, 3], 1],
                [32, [3, 3], 1]
            ],  # CNNの設定
            "fcnet_hiddens": [256, 256]  # 全結合層のユニット数
        })
        .environment(observation_space=obs_space, action_space=act_space)
    )
    algo = config.build()
    wandb.init(project='SoundTurtle', name=f"SAC/run{seed}")
    total_timesteps = 0
    max_timesteps = 100000
    # 学習ループ
    while total_timesteps < max_timesteps:
        result = algo.train()
        total_timesteps = result["timesteps_total"]  # 現在のステップ数を取得

        # wandb にログを送信
        wandb.log({
            "train/score": result["episode_reward_mean"],
            "train/length": result["episode_len_mean"],
        })

        print(f"Steps: {total_timesteps}, Mean Reward: {result['episode_reward_mean']}")

    print("Training finished!")

    # モデルの保存
    checkpoint = algo.save("sac_model")
    print(f"Model saved at {checkpoint}")

    # wandb の終了
    wandb.finish()

    # Rayのシャットダウン
    ray.shutdown()

if __name__ == '__main__':
    # --seed
    if '--seed' in sys.argv:
        num = int(sys.argv[sys.argv.index('--seed') + 1])
        main(num)
    else:
        print('Please input seed number.')