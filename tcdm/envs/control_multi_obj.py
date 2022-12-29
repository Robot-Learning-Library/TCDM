# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import copy, collections
from dm_env import specs
from dm_control.rl import control # https://github.com/deepmind/dm_control/blob/main/dm_control/rl/control.py
from tcdm.envs.reference import HandObjectReferenceMotion, random_generate_ref
from tcdm.envs import generated_traj_abspath
from tcdm.util.geom import quat2euler
from tcdm.envs.control import Task, SingleObjectTask, _normalize_action, _denormalize_action
from tcdm.planner_util import motion_plan_one_obj

class GeneralReferenceMotionMultiObjectTask(SingleObjectTask):
    def __init__(self, reference_motion, reward_fns, init_key, data_path, task_name, object_names, traj_path,
                      reward_weights=None, random=None):
        self.task_name = task_name
        self.object_names = object_names
        # self.ini_object_poses = ini_object_poses
        self.target_object_poses = target_object_poses
        self.objects_manipulation_seq = objects_manipulation_seq
        self.switch_num = 0 
        self.switch_num_max = len(self.objects_manipulation_seq)-1 if self.objects_manipulation_seq is not None else len(self.object_names)-1
        self.curr_obj_idx = self.switch_num if self.objects_manipulation_seq is None else self.objects_manipulation_seq[0]
        self.reference_motion = reference_motion  # this is the first reference motion
        self._init_key = init_key
        super().__init__(task_name, reward_fns, reward_weights, random)

    def switch_obj(self, physics):
        if self.check_switch():
            self.switch_num += 1
            if self.objects_manipulation_seq is None:
                self.curr_obj_idx = self.switch_num
            else:
                self.curr_obj_idx = self.objects_manipulation_seq[self.switch_num]
            object_name = self.objects_manipulation_seq[self.curr_obj_idx]
            traj_path = f'./trajectories/multi_trajs/{self.task_name}/traj{self.switch_num}.npz'
            obj_poses = physics.data.qpos[30:].reshape(-1, 6)  # get current object poses
            traj, _ = motion_plan_one_obj(
                obj_list = self.object_names, 
                manipulated_obj_idx=self.curr_obj_idx, 
                obj_poses=obj_poses, 
                manipulated_obs_target_pose=self.target_object_poses[self.curr_obj_idx], 
                traj_path=traj_path) # save ref traj to file
            self.reference_motion = HandObjectReferenceMotion(object_name, traj_path)
            return True
        else:
            return False
    
    def check_switch(self):
        # TODO
        if self.reference_motion.next_done:
            return True
        else:
            return False

    @property
    def substeps(self):
        substeps = self.reference_motion.substeps
        # print('substeps: ', substeps)
        return substeps

    def get_observation(self, physics):
        obs = Task.get_observation(self, physics, self.curr_obj_idx)  # obj_idx specifies which object to get observation
        base_pos = obs['position']
        base_vel = obs['velocity']

        base_pos = copy.deepcopy(base_pos)
        # obj in global frame
        base_pos[30:33] -= self.offset 
        # observation minus offset; hand in initial fixed hand base frame (-x, z, y) to global (x, y, z)
        base_pos[0] += self.offset[0]
        base_pos[1] -= self.offset[2]
        base_pos[2] -= self.offset[1]

        hand_poses = physics.body_poses
        pose = copy.deepcopy(hand_poses.pos)
        pose[:] -= self.offset  # frames on hand to global frame, shape: (16, 3)

        hand_com = pose.reshape((-1, 3)) # com for center of mass
        hand_rot = hand_poses.rot.reshape((-1, 4))
        hand_lv = hand_poses.linear_vel.reshape((-1, 3))
        hand_av = hand_poses.angular_vel.reshape((-1, 3))
        hand_vel = np.concatenate((hand_lv, hand_av), 1)

        object_name = self.object_name
        obj_com = physics.named.data.xipos[object_name].copy()
        obj_com -= self.offset # object frame in global frame, shape: (3)

        obj_rot = physics.named.data.xquat[object_name].copy()
        obj_vel = physics.data.object_velocity(object_name, 'body')
        obj_vel = obj_vel.reshape((1, 6))
        
        full_com = np.concatenate((hand_com, obj_com.reshape((1,3))), 0)
        full_rot = np.concatenate((hand_rot, obj_rot.reshape((1,4))), 0)
        full_vel = np.concatenate((hand_vel, obj_vel), 0)

        obs['position'] = np.concatenate((base_pos, full_com.reshape(-1), 
                                          full_rot.reshape(-1))).astype(np.float32)
        obs['velocity'] = np.concatenate((base_vel, 
                                          full_vel.reshape(-1))).astype(np.float32)
        obs['state'] = np.concatenate((obs['position'], obs['velocity']))

        obs['goal'] = self.reference_motion.goals.astype(np.float32)
        # handle the offset
        obs['goal'][4::7] -= self.offset[0]  # shape (7,3), 7=4(oritentation)+3(position), 3=future goal position at time +(1,5,10)
        obs['goal'][5::7] -= self.offset[1]
        obs['goal'][6::7] -= self.offset[2]
        obs['state'] = np.concatenate((obs['state'], obs['goal']))
        obs['current_object_idx'] = self.current_object_idx  # get the current object index
        return obs

    def initialize_episode(self, physics):

        #! This works for now for running visualization but not sure if the correct practice
        self._step_count = 0

        self.additional_step_cnt = 0 # step after the reference motion is done
        self.additional_step = False # whether to step after the reference motion is done
        # if self.auto_ref:
        #     new_ref = self._generate_reference_motion()
        #     self.reference_motion = new_ref
        start_state = self.reference_motion.reset()[self._init_key]  # _init_key='motion_planned'
        self.start_state = start_state
        with physics.reset_context():
            # hand
            physics.data.qpos[:30] = start_state['position']
            physics.data.qvel[:30] = start_state['velocity']

            # other objects
            for i, obj_name in enumerate(self.object_names):
                physics.data.qpos[30+6*i:36+6*i] = start_state[str(i)]['position']
                physics.data.qvel[30+6*i:36+6*i] = start_state[str(i)]['position']
            
        return super().initialize_episode(physics)

    def before_step(self, action, physics):
        """
        action is 30 dimensional: 3 for hand translation, 3 for hand rotation, 24 for joint pose, all abosulte position (not delta)!
        """
        # correct hand translation in action  by adding offet
        denorm_action = _denormalize_action(physics, action)  # since offset is in denormed space (true coordinate)
        denorm_action[0] -= self.offset[0] # action match with qpos dimension, so frame change from hand base (-x, z, y, for action) to global (x,y,z, for offset)
        denorm_action[1] += self.offset[2]
        denorm_action[2] += self.offset[1]
        action = _normalize_action(physics, denorm_action)

        if self.additional_step: 
            # physics.data.qvel[:] = 0
            action = physics.data.qpos[:30].copy()  # 30 dim: 3 for hand translation, 3 for hand rotation, 24 for joint pose
            action = _normalize_action(physics, action)  # since action will be denormalized later

        super().before_step(action, physics)
        if not self.additional_step:
            self.reference_motion.step()

    def after_step(self, physics):
        super().after_step(physics)
        # TODO
        switched = self.switch_obj(physics)

    def get_termination(self, physics):
        if self.reference_motion.next_done and self.switch_num == self.switch_num_max:
            return 0.0
        return super().get_termination(physics)


