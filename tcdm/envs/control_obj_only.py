# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import copy
from tcdm.envs.control import Task
from tcdm.envs.reference import HandObjectReferenceMotion, random_generate_ref
from tcdm.envs import generated_traj_abspath
from tcdm.util.geom import quat2euler


class SingleObjectTask(Task):
    def __init__(self, object_name, reward_fns, reward_weights=None, random=None):
        self._object_name = object_name
        super().__init__(reward_fns, reward_weights=reward_weights, random=random)


    def get_observation(self, physics):
        obs = super().get_observation(physics)
        base_pos = obs['position']  # object relative to initial
        base_vel = obs['velocity']

        object_name = self.object_name
        obj_com = physics.named.data.xipos[object_name].copy()  # world, table, hammer, hammer/object
        obj_rot = physics.named.data.xquat[object_name].copy()
        obj_vel = physics.data.object_velocity(object_name, 'body')
        # print(base_pos, base_vel, obj_com, obj_rot, obj_vel)
        # print('obj: ', obj_com)

        obs['position'] = np.concatenate((base_pos, 
                                          )).astype(np.float32)
        obs['velocity'] = np.concatenate((base_vel, 
                                          )).astype(np.float32)
        obs['state'] = np.concatenate((obs['position'], obs['velocity']))
        return obs
    

    @property
    def object_name(self):
        return self._object_name


class ReferenceMotionTask(SingleObjectTask):
    def __init__(self, reference_motion, reward_fns,
                       reward_weights=None, random=None):
        self.reference_motion = reference_motion
        # self._init_key = init_key
        object_name = reference_motion.object_name
        super().__init__(object_name, reward_fns, reward_weights, random)


    def initialize_episode(self, physics):
        return super().initialize_episode(physics)


    def before_step(self, action, physics):
        super().before_step(action, physics)
        self.reference_motion.step()


    def get_termination(self, physics):
        if self.reference_motion.next_done:
            print('done!')
            return 0.0
        # for reward_fn in self._reward_fns:
        #     if reward_fn.check_termination(physics):
        #         return 0.0
        return None
        # return super().get_termination(physics)


    @property
    def substeps(self):
        return self.reference_motion.substeps


    def get_observation(self, physics):
        obs = super().get_observation(physics)
        obs['goal'] = self.reference_motion.goals.astype(np.float32)
        obs['state'] = np.concatenate((obs['state'], obs['goal']))
        return obs


class GeneralReferenceMotionTask(ReferenceMotionTask):
    def __init__(self, reference_motion, reward_fns, data_path, ref_only, auto_ref, task_name, object_name, traj_path, reward_weights=None, random=None):
        self.data_path = data_path
        self.ref_only = ref_only
        self.auto_ref = auto_ref
        self.traj_path = traj_path
        self.task_name = task_name
        if self.auto_ref:
            """Allen: why generates the reference motion again here? What is the purpose of auto_ref flag?"""
            reference_motion = self._generate_reference_motion(object_name)
        super().__init__(reference_motion, reward_fns, reward_weights, random)


    def _generate_reference_motion(self, object_name=None):
        motion_file = generated_traj_abspath(self.data_path, self.traj_path, self.task_name)
        if object_name is not None:
            ref_obj = HandObjectReferenceMotion(object_name, motion_file)
        else:
            ref_obj = HandObjectReferenceMotion(self.object_name, motion_file)
        ref = ref_obj._reference_motion
        rand_ref = random_generate_ref(copy.copy(ref))
        ref_obj.set_with_given_ref(rand_ref)
        return ref_obj


    def initialize_episode(self, physics):

        #! This works for now for running visualization but not sure if the correct practice
        self._step_count = 0
        
        if self.auto_ref:
            new_ref = self._generate_reference_motion()
            self.reference_motion = new_ref
        self._init_key = 'motion_planned'
        start_state = self.reference_motion.reset()[self._init_key]  # _init_key='motion_planned'
        self.start_state = start_state
        with physics.reset_context():
            # qpos use euler, start_state in euler too, object_orientation in quat
            physics.data.qpos[:6] = start_state['position'][-6:]  # object only
            physics.data.qvel[:6] = start_state['velocity'][-6:]

            # fixe object
            if self._multi_obj:
                physics.data.qpos[6:] = start_state['fixed']['position']
                physics.data.qvel[6:] = 0
        return super().initialize_episode(physics)


    def after_step(self, physics):
        super().after_step(physics)
        # physics.reset_mocap2body_xpos()
        # physics.data.xpos[-1,2] = 0.2
        # obj_com = physics.named.data.xipos[self.object_name].copy()
        physics.data.qpos[:3] = self.reference_motion._reference_motion['object_translation'][self._step_count-1]  # x,y,z
        physics.data.qpos[3:6] = quat2euler(self.reference_motion._reference_motion['object_orientation'][self._step_count-1])  # euler xyz extrinsic
        print(self._step_count, physics.data.qpos[:3], physics.data.qpos[3:6])

        # fixed object
        if self._multi_obj:
            physics.data.qpos[6:] = self.start_state['fixed']['position']

        ## Fix hand
        # physics.data.qpos[:30] = self.start_state['position'][:30]
        # physics.data.qpos[1] = 0.7  #z-axis of hand


    @property
    def substeps(self):
        # print('substeps: ', self.reference_motion.substeps)
        if self.ref_only and self.reference_motion.substeps == 10:
            substeps = int(self.reference_motion.substeps / 3) # the above substeps does not replicate the reference
        else:
            substeps = self.reference_motion.substeps
        return substeps
