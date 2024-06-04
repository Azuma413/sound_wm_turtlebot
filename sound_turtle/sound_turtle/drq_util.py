import warnings
import os
from pathlib import Path
import hydra
import torch
import utils
warnings.filterwarnings('ignore', category=DeprecationWarning)
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'
torch.backends.cudnn.benchmark = True

def make_agent(obs_spec, action_spec, cfg):
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    return hydra.utils.instantiate(cfg)

class DrQV2Agent:
    def __init__(self, env):
        cfg = self.set_config()
        cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        utils.set_seed_everywhere(cfg.seed)
        device = torch.device(cfg.device)
        self.agent = make_agent(self.env.observation_spec(),
                                self.env.action_spec(),
                                self.cfg.agent)
        self.timer = utils.Timer()
        self._global_step = 0
        
        snapshot = Path.cwd() / 'weight/drqv2/run0/snapshot.pt' # 重みの保存先
        if snapshot.exists():
            print(f'resuming: {snapshot}')
            with snapshot.open('rb') as f:
                payload = torch.load(f, map_location=device)
            for k, v in payload.items():
                self.__dict__[k] = v
        self.time_step = env.reset()
    
    @hydra.main(config_path='my_config', config_name='drqv2')
    def set_config(cfg):
        return cfg
    
    def action(self):
        # 1ステップ進める
        with torch.no_grad(), utils.eval_mode(self.agent):
            action = self.agent.act(self.time_step["observation"],
                                    self.global_step,
                                    eval_mode=True)
        self.time_step = self.eval_env.step(action) # これで環境のactionが更新される

    @property
    def global_step(self):
        return self._global_step