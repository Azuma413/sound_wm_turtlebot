import wandb
import warnings
import dreamerv3
from dreamerv3 import embodied
from pathlib import Path
import gc
import jax
import numpy as np
import sys
warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')

def main(num):
  size = 'medium' # 'small', 'medium', 'large', 'xlarge'
  room_num = 0 # 0:長方形, 1:L字, 2:仕切り, 3:リアル
  image_horizon = 15 # 15
  name = f'dreamerv3/{size}/ih_{image_horizon}/room{room_num}/alt_blue/run{num}'
  config = embodied.Config(dreamerv3.configs['defaults'])
  config = config.update(dreamerv3.configs[size])
  config = config.update({
      'seed': num,
      'logdir': Path(__file__).parent / 'weight' / name,
      'run.train_ratio': 64,
      'run.log_every': 30,  # Seconds
      'run.steps': 100000,
      'batch_size': 16,
      'jax.prealloc': False,
      'encoder.mlp_keys': '$^',
      'decoder.mlp_keys': '$^',
      'encoder.cnn_keys': 'image',
      'decoder.cnn_keys': 'image',
      'imag_horizon': image_horizon,
      # 'jax.platform': 'cpu',
  })
  wandb.init(project='SoundTurtle', group='dreamerv3', name=name, config=config)
  config = embodied.Flags(config).parse()
  logdir = embodied.Path(config.logdir)
  step = embodied.Counter()
  logger = embodied.Logger(step, [
      embodied.logger.TensorBoardOutput(logdir),
  ])
  from dreamerv3.embodied.envs import from_gym
  from my_envs.my_env import MyEnv # add
  train_env = MyEnv(room_num)
  train_env = from_gym.FromGym(train_env, obs_key='image')  # Or obs_key='vector'.
  train_env = dreamerv3.wrap_env(train_env, config)
  train_env = embodied.BatchEnv([train_env], parallel=False)
  eval_env = MyEnv(room_num)
  eval_env = from_gym.FromGym(eval_env, obs_key='image')  # Or obs_key='vector'.
  eval_env = dreamerv3.wrap_env(eval_env, config)
  eval_env = embodied.BatchEnv([eval_env], parallel=False)

  agent = dreamerv3.Agent(train_env.obs_space, train_env.act_space, step, config)
  train_replay = embodied.replay.Uniform(config.batch_length, config.replay_size, logdir / 'train_replay')
  eval_replay = embodied.replay.Uniform(config.batch_length, config.replay_size, logdir / 'eval_replay')
  args = embodied.Config(
      **config.run, logdir=config.logdir,
      batch_steps=config.batch_size * config.batch_length)
  embodied.run.train_eval(agent, train_env, eval_env, train_replay, eval_replay, logger, args)
  wandb.finish()
if __name__ == '__main__':
  # --seed
  if '--seed' in sys.argv:
    num = int(sys.argv[sys.argv.index('--seed') + 1])
    main(num)
  else:
    print('Please input seed number.')