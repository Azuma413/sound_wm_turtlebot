import ray
from ray.rllib.algorithms.sac import SACConfig
from my_envs.my_env import MyEnv
import wandb
import sys

def main(seed):
    # Rayの初期化
    ray.init()
    env = MyEnv()
    obs_space = env.observation_space
    act_space = env.action_space
    # SACの設定
    config = (
        SACConfig()
        .environment(env=MyEnv, render_env=False)
        .rollouts(num_env_runners=1)  # 並列ワーカー数
        .framework("torch")  # PyTorchを使用
        .resources(num_gpus=1)  # GPUを使用する場合は適切に設定
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