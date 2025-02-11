# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os, tcdm


# logic for retrieving assets
def asset_abspath(resource_path):
    envs_path = os.path.dirname(__file__)
    asset_folder_path = os.path.join(envs_path, 'assets')
    return os.path.join(asset_folder_path, resource_path)


# logic for retreiving trajectories
def traj_abspath(resource_path, traj_path='trajectories'):
    base_path = os.path.dirname(os.path.dirname(tcdm.__file__))
    data_folder_path = os.path.join(base_path, traj_path)
    return os.path.join(data_folder_path, resource_path)

# logic for retreiving trajectories
def generated_traj_abspath(resource_path, traj_path, task_name):
    base_path = os.path.dirname(os.path.dirname(tcdm.__file__))
    data_folder_path = os.path.join(base_path, f'{traj_path}/{task_name}')
    return os.path.join(data_folder_path, resource_path)


# mj_models relies on asset_abspath hence import at end
from tcdm.envs.mujoco import mj_models
