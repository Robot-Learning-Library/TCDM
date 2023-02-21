# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import traceback
import gym
import numpy as np
import hydra, os, wandb, yaml
from tcdm.rl import trainers
from tcdm.rl.trainers.util import make_env
from omegaconf import DictConfig, OmegaConf, open_dict
from hydra.core.hydra_config import HydraConfig

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

def rollout(config, resume_model):
    # Create environment
    env = make_env(multi_proc=False, **config.env)

    # Instantiate the agent
    model = PPO.load(resume_model, env)
    model._last_obs = None
    reset_num_timesteps = False

    # Evaluate the agent
    # NOTE: If you use wrappers with your environment that modify rewards,
    #       this will be reflected here. To evaluate with original rewards,
    #       wrap environment in a "Monitor" wrapper before other wrappers.
    # mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

    # Enjoy trained agent
    vec_env = model.get_env()
    obs = vec_env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = vec_env.step(action)
        vec_env.render()
        if np.any(dones):
            print('Episode finished')
            # break

cfg_path = os.path.dirname(__file__)
cfg_path = os.path.join(cfg_path, 'experiments')
@hydra.main(config_path=cfg_path, config_name="config.yaml")
def test(cfg: DictConfig):
    try:
        cfg_yaml = OmegaConf.to_yaml(cfg)
        # resume_model = cfg.resume_model

        # set checkpoint path
        resume_model = '/home/quantumiracle/research/TCDM/outputs/2023-02-17/03-17-50/restore_checkpoint'  # fryingpan

        if os.path.exists('exp_config.yaml'):
            old_config = yaml.load(open('exp_config.yaml', 'r'))
            params, wandb_id = old_config['params'], old_config['wandb_id']
            resume_model = 'restore_checkpoint.zip'
            assert os.path.exists(resume_model), 'restore_checkpoint.zip does not exist!'
        else:
            defaults = HydraConfig.get().runtime.choices
            params = yaml.safe_load(cfg_yaml)
            params['defaults'] = {k: defaults[k] for k in ('agent', 'env')}

        with open_dict(cfg):    
            cfg['env']['task_kwargs']['ref_only'] = False
            cfg['env']['task_kwargs']['auto_ref'] = True
            cfg['n_envs'] = 2 # set number of envs to run

        if cfg.agent.name == 'PPO':
            print('Config: ', cfg)
            rollout(cfg, resume_model)
            # trainers.ppo_trainer(cfg, resume_model)
        else:
            raise NotImplementedError
        # wandb.finish()
    except:
        pass
        # traceback.print_exc(file=open('exception.log', 'w'))
        # with open('exception.log', 'r') as f:
        #     print(f.read())


if __name__ == '__main__':
        test()
