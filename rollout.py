# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import matplotlib.pyplot as plt
import pickle as pkl
import numpy as np
import glob, yaml, os, imageio, cv2, shutil
from tcdm import suite
from stable_baselines3 import PPO
from argparse import ArgumentParser
import pandas as pd
from dm_control import viewer

"""
PLEASE DOWNLOAD AND UNZIP THE PRE-TRAINED AGENTS BEFORE RUNNING THIS
SEE: https://github.com/facebookresearch/DexMan#pre-trained-policies
"""

parser = ArgumentParser(description="Example code for loading pre-trained policies")
# parser.add_argument('--env', default='hammer_use1')
parser.add_argument('--save_folder', default='pretrained_agents/hammer_use1/', 
                                     help="Save folder containing agent checkpoint/config")
parser.add_argument('--checkpoint', default=None, help="checkpoint folder")
# parser.add_argument('--config', default=None, help="config folder")
parser.add_argument('--render', action="store_true", help="Supply flag to render mp4")


# def render(writer, physics, AA=2, height=768, width=768):
def render(writer, physics, AA=2, height=512, width=512):
    if writer is None:
        return
    img = physics.render(camera_id=0, height=height * AA, width=width * AA)
    writer.append_data(cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA))


def rollout(args, writer):
    gym_wrap = True
    if args.checkpoint is not None:
        checkpoint_path = os.path.join(args.checkpoint, 'checkpoint.zip')  # use specified checkpoint
    else:
        checkpoint_path = os.path.join(args.save_folder, 'checkpoint.zip')
    save_folder = args.save_folder
    # get experiment config
    config =  yaml.safe_load(open(os.path.join(save_folder, 'exp_config.yaml'), 'r'))
    if 'params' in config: # saved config may have one more level
        config = config['params']
    # config['env']['task_kwargs']['ref_only'] = True
    # config['env']['task_kwargs']['auto_ref'] = True
    # config['env']['task_kwargs']['traj_path'] = 'trajectories/specified_trajs'
    # config['env']['task_kwargs']['traj_path'] = 'trajectories/multi_trajs'
    # build environment and load policy
    if 'multi_obj' in config['env'] and config['env']['multi_obj']:
        env = suite.load_multi(config['env']['name'], config['env']['task_kwargs'], gym_wrap=gym_wrap, obj_only=config['env']['obj_only'])
    else:
        o, t = config['env']['name'].split('-')
        env = suite.load(o, t, config['env']['task_kwargs'], gym_wrap=gym_wrap)
    try:
        policy = PPO.load(checkpoint_path)
    except:
        policy = PPO.load(checkpoint_path.replace('checkpoint.zip', 'restore_checkpoint'))

    log = False
    logger = {'s': [], 'a':[]}
    # Launch viewer
    # viewer.launch(env)

    # Initialize window
    # fig = plt.figure(figsize=(16,16))
    # img = plt.imshow(env.render(mode='rgb_array')) # only call this once

    # rollout the policy and print total reward
    s, done, total_reward = env.reset(), False, 0
    render(writer, env.wrapped.physics) if gym_wrap else render(writer, env.physics)
    while not done:
        action, _ = policy.predict(s['state'], deterministic=True)
        s, r, done, __ = env.step(action)
        render(writer, env.wrapped.physics) if gym_wrap else render(writer, env.physics)
        total_reward += r

        # img.set_data(env.render(mode='rgb_array')) # just update the data
        # fig.canvas.draw_idle()
        # plt.pause(0.5)

        if log:
            logger['s'].append(s['state'])
            logger['a'].append(action)
    print('Total reward:', total_reward)
    if log:
        df = pd.DataFrame(logger['s'])
        df.to_csv(f's.csv', index=False, header=True)
        df = pd.DataFrame(logger['a'])
        df.to_csv(f'a.csv', index=False, header=True)


if __name__ == "__main__":
    args = parser.parse_args()

    # configure writer
    if args.render:
        writer = imageio.get_writer('rollout.mp4', fps=25)
        rollout(args, writer)
        writer.close()
    else:
        rollout(args, None)
