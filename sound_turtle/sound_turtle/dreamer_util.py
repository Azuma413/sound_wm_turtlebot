import re
from dreamerv3 import embodied
import numpy as np
import warnings
import dreamerv3
from dreamerv3.embodied.envs import from_gym
import yaml
from pathlib import Path
warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')

class DreamerV3Agent:
    def __init__(self, env):
        # configの設定
        configs = yaml.YAML(typ='safe').load(Path(Path(__file__).parent, 'my_config/dreamerv3.yaml').read())
        config = embodied.Config(configs['defaults'])
        config = config.update(configs['medium'])
        config = config.update({
            'logdir': Path(__file__).parent + '/weight/dreamerv3/run0', # 重みファイルの場所を指定
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
        # logdirの設定
        logdir = embodied.Path(config.logdir)
        # stepの設定
        step = embodied.Counter()
        # loggerの設定
        logger = embodied.Logger(step, [embodied.logger.TerminalOutput()])
        # envの設定
        env = from_gym.FromGym(env, obs_key='image')  # Or obs_key='vector'.
        env = dreamerv3.wrap_env(env, config)
        env = embodied.BatchEnv([env], parallel=False)
        # agentの設定
        agent = dreamerv3.Agent(env.obs_space, env.act_space, step, config)
        # argsの設定
        args = embodied.Config(
            **config.run, logdir=config.logdir,
            batch_steps=config.batch_size * config.batch_length
        )
        # ここからeval_onlyの処理
        logdir.mkdirs()
        print('Logdir', logdir)
        should_log = embodied.when.Clock(args.log_every)
        step = logger.step
        metrics = embodied.Metrics()
        print('Observation space:', env.obs_space)
        print('Action space:', env.act_space)
        timer = embodied.Timer()
        timer.wrap('agent', agent, ['policy'])
        timer.wrap('env', env, ['step'])
        timer.wrap('logger', logger, ['write'])
        nonzeros = set()
        def per_episode(ep):
            length = len(ep['reward']) - 1
            score = float(ep['reward'].astype(np.float64).sum())
            logger.add({'length': length, 'score': score}, prefix='episode')
            print(f'Episode has {length} steps and return {score:.1f}.')
            stats = {}
            for key in args.log_keys_video:
                if key in ep:
                    stats[f'policy_{key}'] = ep[key]
            for key, value in ep.items():
                if not args.log_zeros and key not in nonzeros and (value == 0).all():
                    continue
                nonzeros.add(key)
                if re.match(args.log_keys_sum, key):
                    stats[f'sum_{key}'] = ep[key].sum()
                if re.match(args.log_keys_mean, key):
                    stats[f'mean_{key}'] = ep[key].mean()
                if re.match(args.log_keys_max, key):
                    stats[f'max_{key}'] = ep[key].max(0).mean()
            metrics.add(stats, prefix='stats')
        self.driver = embodied.Driver(env)
        self.driver.on_episode(lambda ep, worker: per_episode(ep))
        self.driver.on_step(lambda tran, _: step.increment())
        checkpoint = embodied.Checkpoint()
        checkpoint.agent = agent
        checkpoint.load(args.from_checkpoint, keys=['agent'])
        print('Start evaluation')
        self.policy = lambda *args: agent.policy(*args, mode='eval')
        
    def action(self):
        # 1ステップだけ実行
        self.driver(self.policy, steps=1)