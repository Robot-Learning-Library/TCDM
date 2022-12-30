# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import yaml, os
from argparse import ArgumentParser

from tcdm import suite
from dm_control import viewer


"""
PLEASE DOWNLOAD AND UNZIP THE PRE-TRAINED AGENTS BEFORE RUNNING THIS
SEE: https://github.com/facebookresearch/DexMan#pre-trained-policies
"""

parser = ArgumentParser(description="Example code for loading pre-trained policies")
parser.add_argument('--save_folder', default='pretrained_agents/hammer_use1/', 
                                     help="Save folder containing agent checkpoint/config")


def rollout(save_folder):
    # get experiment config
    config = yaml.safe_load(open(os.path.join(save_folder, 'exp_config.yaml'), 'r'))
    
    # build environment and load policy
    config['env']['task_kwargs']['ref_only'] = True
    # config['env']['task_kwargs']['auto_ref'] = True
    if 'multi_obj' in config['env'] and config['env']['multi_obj']:
        env = suite.load_multi(config['env']['name'],
                               config['env']['task_kwargs'], 
                               gym_wrap=False, 
                               obj_only=config['env']['obj_only'],
                               switch=config['env']['switch'])
    else:
        o, t = config['env']['name'].split('-')
        env = suite.load(o, t, 
                         config['env']['task_kwargs'], 
                         gym_wrap=False,    # do not wrap gym, otherwise launch interactive viewer from dm_control 
                         obj_only=config['env']['obj_only'])

    # Launch viewer
    viewer.launch(env)


if __name__ == "__main__":
    args = parser.parse_args()

    rollout(args.save_folder)
