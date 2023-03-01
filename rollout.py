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
parser.add_argument('--traj_path', default=None, help="trajectory path folder")
parser.add_argument('--render', action="store_true", help="Supply flag to render mp4")
parser.add_argument('--auto_ref', type=bool, default=False, help="automatically generate reference trajectory for training")
parser.add_argument('--ref_only', type=bool, default=False, help="whether only shows the reference trajectory")

# def render(writer, physics, AA=2, height=768, width=768):
def render(writer, physics, AA=2, height=512, width=512):
    if writer is None:
        return
    img = physics.render(camera_id=0, height=height * AA, width=width * AA)
    writer.append_data(cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA))


def rollout(args):
    gym_wrap = True
    if args.checkpoint is not None:
        checkpoint_path = os.path.join(args.checkpoint, 'checkpoint.zip')  # use specified checkpoint
    else:
        checkpoint_path = os.path.join(args.save_folder, 'checkpoint.zip')
    save_folder = args.save_folder
    task_name = save_folder.split('/')[-1]  # 'new_agents/banana_fryingpan_pass1
    writer = imageio.get_writer(f'rollout_{task_name}.mp4', fps=25) if args.render else None
    # get experiment config
    config =  yaml.safe_load(open(os.path.join(save_folder, 'exp_config.yaml'), 'r'))
    if 'params' in config: # saved config may have one more level
        config = config['params']
    config['env']['task_kwargs']['ref_only'] = args.ref_only
    config['env']['task_kwargs']['auto_ref'] = args.auto_ref
    # config['env']['task_kwargs']['traj_path'] = 'trajectories/specified_trajs'
    # config['env']['task_kwargs']['traj_path'] = 'trajectories/multi_trajs'
    if args.traj_path is not None:
        config['env']['task_kwargs']['traj_path'] = args.traj_path
    # build environment and load policy
    if 'multi_obj' in config['env'] and config['env']['multi_obj']:
        env = suite.load_multi(config['env']['name'], 
                        config['env']['task_kwargs'], 
                        gym_wrap=gym_wrap, 
                        )
    else:
        env = suite.load(config['env']['name'], config['env']['task_kwargs'], gym_wrap=gym_wrap)

    if 'switch' in config['env']['task_kwargs'] and config['env']['task_kwargs']['switch']: 
        policies = []
        policy_paths = []
        # policy_paths = [
        #                 # 'outputs/2022-11-06/12-00-55',  # banana
        #                 # 'outputs/2022-12-29/03-52-52'   # pan
        #                 'outputs/2022-11-25/03-45-35',  # cup
        #                 'outputs/2022-11-25/03-45-35'   # cup
        #     ]
        # automatically get general policies according to the names of objects, from 'outputs/general_policy'
        obj_list = []
        saved_policy_path = 'outputs/general_policy/'
        with open("tcdm/envs/assets/task_trajs.yaml", "r") as stream:
            try:
                trajs = yaml.safe_load(stream)
                names = trajs['obj_mimic']
                for name in names:
                    obj_list.append(name.split('_')[0])
            except yaml.YAMLError as exc:
                print(exc)
        for obj in config['env']['name'].split('-'):
            if obj in obj_list:
                policy_paths.append(saved_policy_path+f'{obj}')
        print('Load general policies from: ', policy_paths)
        # load policy from path
        for path in policy_paths:
            policies.append(PPO.load(os.path.join(path, 'restore_checkpoint')))
    else:  # single policy
        try:
            policy = PPO.load(checkpoint_path)
        except:
            policy = PPO.load(checkpoint_path.replace('checkpoint.zip', 'restore_checkpoint'))

    if 'switch' in config['env']['task_kwargs'] and config['env']['task_kwargs']['switch']:
        s, done, total_reward = env.reset(), False, 0
        render(writer, env.wrapped.physics) if gym_wrap else render(writer, env.physics)
        while not done:
            policy = policies[int(s['current_move_obj_idx'][0])]
            action, _ = policy.predict(s['state'], deterministic=True)
            s, r, done, __ = env.step(action)
            render(writer, env.wrapped.physics) if gym_wrap else render(writer, env.physics)
            total_reward += r
    else:

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

    writer.close() if writer is not None else None

if __name__ == "__main__":
    args = parser.parse_args()

    rollout(args)
