def main():
  import wandb
  import warnings
  import dreamerv3
  from dreamerv3 import embodied
  from pathlib import Path
  warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')

  name = "dreamerv3/run0"

  wandb.init(project='sound_turtle', group='dreamerv3', name=name)
  config = embodied.Config(dreamerv3.configs['defaults'])
  config = config.update(dreamerv3.configs['medium'])
  config = config.update({
      'logdir': Path(__file__).parent / 'weight' / name,
      'run.train_ratio': 64,
      'run.log_every': 30,  # Seconds
      'batch_size': 16,
      'jax.prealloc': False,
      'encoder.mlp_keys': '$^',
      'decoder.mlp_keys': '$^',
      'encoder.cnn_keys': 'image',
      'decoder.cnn_keys': 'image',
      # 'jax.platform': 'cpu',
  })
  config = embodied.Flags(config).parse()
  logdir = embodied.Path(config.logdir)
  step = embodied.Counter()
  logger = embodied.Logger(step, [
      # embodied.logger.TerminalOutput(),
      # embodied.logger.WandBOutput(r'.*', logdir, config),
      # embodied.logger.JSONLOutput(logdir, 'metrics.jsonl'),
      embodied.logger.TensorBoardOutput(logdir),
  ])
  from dreamerv3.embodied.envs import from_gym
  from my_envs.my_env import MyEnv # add
  train_env = MyEnv()
  train_env = from_gym.FromGym(train_env, obs_key='image')  # Or obs_key='vector'.
  train_env = dreamerv3.wrap_env(train_env, config)
  train_env = embodied.BatchEnv([train_env], parallel=False)
  eval_env = MyEnv()
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
if __name__ == '__main__':
  main()
