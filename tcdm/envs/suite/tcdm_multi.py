# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
from tcdm.envs import mj_models, traj_abspath, generated_traj_abspath
from dm_control.utils import containers
from tcdm.envs.control import Environment
from tcdm.envs.suite.tcdm import ObjMimicTask
from tcdm.envs.suite.tcdm_obj_only import ObjOnlyMimicTask
from tcdm.envs.mujoco import physics_from_mjcf


class MultiObjOnlyMimicTask(ObjOnlyMimicTask):
    """Right now, assume two objects, and one object is fixed."""

    def __init__(self, object_name, data_path, reward_kwargs, append_time, 
                       pregrasp_init_key, ref_only=False, auto_ref=False, task_name=None, traj_path=None):
        super().__init__(object_name, data_path, reward_kwargs, append_time, 
                       ref_only, auto_ref, task_name, traj_path)
        self._multi_obj = True

class MultiObjMimicTask(ObjMimicTask):
    """Right now, assume two objects, and one object is fixed."""

    def __init__(self, object_name, data_path, reward_kwargs, append_time, 
                       pregrasp_init_key, ref_only=False, auto_ref=False, task_name=None, traj_path=None):
        super().__init__(object_name, data_path, reward_kwargs, append_time, pregrasp_init_key, ref_only, auto_ref, task_name, traj_path)
        self._multi_obj = True

def _multi_obj_mimic_task_factory(env_name, object_class_list, robot_class):
    def task(append_time=True, pregrasp='initialized', ref_only=False, auto_ref=False, traj_path=None, reward_kwargs={}, environment_kwargs={}):
        """
        ref_only: only visualize object reference trajectory, the hand is hanging
        auto_ref: automatically generate reference trajectory at the start of each episode
        """
        # load env and robot
        env = mj_models.TableEnv()
        if robot_class:
            env.attach(robot_class(limp=False))

        # object_model = object1_class()
        # object_name = '{}/object'.format(object_model.mjcf_model.model)
        # env.attach(object_model)

        # load objects
        object_model_list = []
        for object_class in object_class_list:
            object_model = object_class()
            env.attach(object_model)
            object_model_list.append(object_model)
        first_object_name = '{}/object'.format(object_model_list[0].mjcf_model.model)

        env_name_ = env_name.replace('-', '_')
        if traj_path is None:
            data_path = traj_abspath(env_name_+'.npz', traj_path='trajectories/multi_trajs')
        else:
            data_path = generated_traj_abspath(env_name_+'.npz', traj_path, env_name_)

        if robot_class:
            task_class = MultiObjMimicTask
        else:
            task_class = MultiObjOnlyMimicTask
        task = task_class(first_object_name, data_path, reward_kwargs, append_time, pregrasp, ref_only, auto_ref, env_name, traj_path)
        # build physics object and create environment
        if ref_only:
            gravity_compensation_for_all = True
        else:  
            gravity_compensation_for_all = False
        # build physics object and create environment
        physics = physics_from_mjcf(env, gravity_compensation_for_all=gravity_compensation_for_all)
        return Environment(physics, task,
                           n_sub_steps=task.substeps,
                           **environment_kwargs)

    task.__name__ = env_name
    return task


def get_domain_multi(env_name, obj_only=False):
    # with open(asset_abspath('multi_task_trajs.yaml'), 'r') as g:
    #     _TCDM_TRAJS = yaml.safe_load(g)['obj_mimic']

    obj_names = env_name.split('-')[0:-1]  # e.g., *-*-pass

    object_class_list = []
    for obj_name in obj_names:
        object_class = mj_models.get_object(obj_name)
        object_class_list.append(object_class)

    if obj_only:
        robot_class = None
    else:
        robot_class = mj_models.Adroit
    task = _multi_obj_mimic_task_factory(env_name, object_class_list, robot_class)
    domain = containers.TaggedTasks()
    domain.add('mimic')(task)

    return domain
