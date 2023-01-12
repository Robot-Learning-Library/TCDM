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


def _denormalize_action(physics, action):
    ac_min, ac_max = physics.ctrl_range.T
    ac_mid = 0.5 * (ac_max + ac_min)
    ac_range = 0.5 * (ac_max - ac_min)
    return np.clip(action, -1, 1) * ac_range + ac_mid


def _normalize_action(physics, action):
    ac_min, ac_max = physics.ctrl_range.T
    ac_mid = 0.5 * (ac_max + ac_min)
    ac_range = 0.5 * (ac_max - ac_min)
    return np.clip((action - ac_mid) / ac_range, -1, 1)


FLAT_OBSERVATION_KEY = 'observations'
class Environment(control.Environment):
    def __init__(self, physics, task, default_camera_id=0, **kwargs):
        self._default_camera_id = default_camera_id
        super().__init__(physics, task, **kwargs)

    def get_state(self):
        return dict(physics=self.physics.get_state(), 
                    task=self.task.get_state())
    
    def set_state(self, state):
        self.physics.set_state(state['physics'])
        self.task.set_state(state['task'])

    @property
    def flat_obs(self):
        return self._flat_observation
    
    @property
    def default_camera_id(self):
        return self._default_camera_id


class Task(control.Task):
    def __init__(self, reward_fns, reward_weights=None, random=None):
        # initialize 
        if not isinstance(random, np.random.RandomState):
            random = np.random.RandomState(random)
        self._random = random
        self._info = {}

        # store reward functions and weighting terms
        self._step_count = 0
        self._reward_fns = copy.deepcopy(reward_fns)
        reward_wgts = [1.0 for _ in self._reward_fns] if reward_weights is None \
                     else reward_weights
        self._reward_wgts = copy.deepcopy(reward_wgts)

    @property
    def random(self):
        """Task-specific `numpy.random.RandomState` instance."""
        return self._random

    def action_spec(self, physics):
        """Returns a `BoundedArraySpec` matching the `physics` actuators."""
        return specs.BoundedArray((physics.adim,), np.float32, -1, 1)

    def initialize_episode(self, physics):
        """ Sets the state of the environment at the start of each episode.
            Called by `control.Environment` at the start of each episode *within*
            `physics.reset_context()` (see the documentation for `base.Physics`)

        Args:
            physics: An instance of `mujoco.Physics`.
        """
        # initialize info dict and rewards
        self._info = {}
        self.initialize_rewards(physics)
    
    def initialize_rewards(self, physics):
        """ Initializes reward function objects with necessarily data/objects in task
        
        Args:
            physics: An instance of `mujoco.Physics`
        """
        for reward_fn in self._reward_fns:
            reward_fn.initialize_rewards(self, physics)

    def before_step(self, action, physics):
        """Sets the control signal for the actuators to values in `action`."""
        self._step_count += 1
        action = _denormalize_action(physics, action)
        physics.set_control(action)

    def after_step(self, physics):
        """Called immediately after environment step: no-op by default"""

    def get_observation(self, physics, obj_idx=None):
        """Returns a default observation of current physics state."""
        obs = collections.OrderedDict()
        if obj_idx is None:
            obs['position'] = physics.data.qpos.astype(np.float32).copy()
            obs['velocity'] = physics.data.qvel.astype(np.float32).copy()
        else:
            hand_pos = physics.data.qpos.astype(np.float32).copy()[:30]
            hand_vel = physics.data.qvel.astype(np.float32).copy()[:30]
            obj_pos = physics.data.qpos.astype(np.float32).copy()[30+6*obj_idx:36+6*obj_idx]
            obj_vel = physics.data.qvel.astype(np.float32).copy()[30+6*obj_idx:36+6*obj_idx]

            obs['position'] = np.concatenate([hand_pos, obj_pos])
            obs['velocity'] = np.concatenate([hand_vel, obj_vel])

        motor_joints = physics.data.qpos[:physics.adim]
        obs['zero_ac'] = _normalize_action(physics, motor_joints)
        return obs
    
    @property
    def step_info(self):
        """Compatability function to pipe extra step information for gym compat """
        return self._info

    @property
    def step_count(self):
        return self._step_count

    def get_reward(self, physics):
        reward = 0
        if not self.additional_step:
            for reward_fn, lambda_r in zip(self._reward_fns, self._reward_wgts):
                r_i, info_i = reward_fn(physics)
                reward += lambda_r * r_i
                self._info.update(info_i)
        return reward

    def get_termination(self, physics):
        if not self.additional_step and not self.ref_only:
            for reward_fn in self._reward_fns:
                if reward_fn.check_termination(physics):
                    return 0.0
        return None


class SingleObjectTask(Task):
    def __init__(self, object_name, reward_fns, reward_weights=None, random=None):
        self._object_name = object_name
        super().__init__(reward_fns, reward_weights=reward_weights, random=random)
    
    def get_observation(self, physics):
        obs = super().get_observation(physics)
        base_pos = obs['position']
        base_vel = obs['velocity']

        hand_poses = physics.body_poses
        hand_com = hand_poses.pos.reshape((-1, 3))
        hand_rot = hand_poses.rot.reshape((-1, 4))
        hand_lv = hand_poses.linear_vel.reshape((-1, 3))
        hand_av = hand_poses.angular_vel.reshape((-1, 3))
        hand_vel = np.concatenate((hand_lv, hand_av), 1)

        object_name = self.object_name
        obj_com = physics.named.data.xipos[object_name].copy()
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
        return obs
    
    @property
    def object_name(self):
        return self._object_name

# class MultiObjectTask(Task):
#     def __init__(self, object_names, reward_fns, reward_weights=None, random=None):
#         self._object_names = object_names
#         super().__init__(reward_fns, reward_weights=reward_weights, random=random)
    
#     def get_observation(self, physics, obj_idx):
#         obs = super().get_observation(physics, obj_idx)
#         base_pos = obs['position']
#         base_vel = obs['velocity']

#         hand_poses = physics.body_poses
#         hand_com = hand_poses.pos.reshape((-1, 3))
#         hand_rot = hand_poses.rot.reshape((-1, 4))
#         hand_lv = hand_poses.linear_vel.reshape((-1, 3))
#         hand_av = hand_poses.angular_vel.reshape((-1, 3))
#         hand_vel = np.concatenate((hand_lv, hand_av), 1)

#         object_name = self.object_name
#         obj_com = physics.named.data.xipos[object_name].copy()
#         obj_rot = physics.named.data.xquat[object_name].copy()
#         obj_vel = physics.data.object_velocity(object_name, 'body')
#         obj_vel = obj_vel.reshape((1, 6))

#         full_com = np.concatenate((hand_com, obj_com.reshape((1,3))), 0)
#         full_rot = np.concatenate((hand_rot, obj_rot.reshape((1,4))), 0)
#         full_vel = np.concatenate((hand_vel, obj_vel), 0)

#         obs['position'] = np.concatenate((base_pos, full_com.reshape(-1), 
#                                           full_rot.reshape(-1))).astype(np.float32)
#         obs['velocity'] = np.concatenate((base_vel, 
#                                           full_vel.reshape(-1))).astype(np.float32)
#         obs['state'] = np.concatenate((obs['position'], obs['velocity']))
#         return obs
    
#     @property
#     def object_name(self):
#         return self._object_names

class ReferenceMotionTask(SingleObjectTask):
    def __init__(self, reference_motion, reward_fns, init_key,
                       reward_weights=None, random=None):
        self.reference_motion =reference_motion
        self._init_key = init_key
        object_name = reference_motion.object_name
        super().__init__(object_name, reward_fns, reward_weights, random)

    def initialize_episode(self, physics):
        start_state = self.reference_motion.reset()[self._init_key]  # _init_key='motion_planned'
        with physics.reset_context():
            physics.data.qpos[:] = start_state['position']
            physics.data.qvel[:] = start_state['velocity']
        return super().initialize_episode(physics)

    def before_step(self, action, physics):
        super().before_step(action, physics)
        self.reference_motion.step()
        # physics.data.qpos[-3:] = self.start_state['position'][-3:]
        # physics.data.qvel[-3:] = self.start_state['velocity'][-3:]

    def get_termination(self, physics):
        if self.reference_motion.next_done:
            return 0.0
        return super().get_termination(physics)

    @property
    def substeps(self):
        return self.reference_motion.substeps

    def get_observation(self, physics):
        obs = super().get_observation(physics)
        obs['goal'] = self.reference_motion.goals.astype(np.float32)
        obs['state'] = np.concatenate((obs['state'], obs['goal']))
        return obs


class GeneralReferenceMotionTask(SingleObjectTask):
    def __init__(self, reference_motion, reward_fns, init_key, data_path, ref_only, auto_ref, task_name, object_name, traj_path,
                      reward_weights=None, random=None):
        self.data_path = data_path
        self.ref_only = ref_only
        self.auto_ref = auto_ref
        self.traj_path = traj_path
        self.task_name = task_name
        if self.auto_ref:
            reference_motion = self._generate_reference_motion(object_name)
        self.reference_motion =reference_motion
        if 'initial_translation_offset' in self.reference_motion._reference_motion:
            self.offset = self.reference_motion._reference_motion['initial_translation_offset']
        else:
            self.offset = np.zeros(3)
        print('offset: ', self.offset)
        self._init_key = init_key
        object_name = reference_motion.object_name
        super().__init__(object_name, reward_fns, reward_weights, random)

    def _generate_reference_motion(self, object_name=None):
        motion_file = generated_traj_abspath(self.data_path, self.traj_path, self.task_name)
        if object_name is not None:
            ref_obj = HandObjectReferenceMotion(object_name, motion_file)
        else:
            ref_obj = HandObjectReferenceMotion(self.object_name, motion_file)
        ref = ref_obj._reference_motion
        rand_ref = random_generate_ref(copy.copy(ref), 'random')
        ref_obj.set_with_given_ref(rand_ref)
        return ref_obj

    def get_observation(self, physics):
        obs = Task.get_observation(self, physics, obj_idx=0)  # obj_idx specifies which object to get observation
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
        return obs

    def initialize_episode(self, physics):

        #! This works for now for running visualization but not sure if the correct practice
        self._step_count = 0

        self.additional_step_cnt = 0 # step after the reference motion is done
        self.additional_step = False # whether to step after the reference motion is done
        if self.auto_ref:
            new_ref = self._generate_reference_motion()
            self.reference_motion = new_ref
        start_state = self.reference_motion.reset()[self._init_key]  # _init_key='motion_planned'
        self.start_state = start_state
        with physics.reset_context():
            physics.data.qpos[:36] = start_state['position']
            physics.data.qvel[:36] = start_state['velocity']

            # fixe object
            if self._multi_obj:
                physics.data.qpos[-6:] = start_state['fixed']['position']
                physics.data.qvel[-6:] = 0
            
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
        # action = np.zeros_like(action)
        # action[:3] = self.ini_pose
        # print('qpos: ', physics.data.qpos[:3] )

        super().before_step(action, physics)
        if not self.additional_step:
            self.reference_motion.step()

    def after_step(self, physics):
        super().after_step(physics)
        # print(physics.data.qpos[3:])
        if self.ref_only: # set position of objects (according to reference) and hand (fixed)
            
            print('step: ', self._step_count)
            
            # hand
            physics.data.qpos[:30] = self.start_state['position'][:30]
            physics.data.qpos[1] = 0.7  #z-axis of hand

            z_global_local_offset = 0.2
            # floating object            
            physics.data.qpos[30:32] = self.reference_motion._reference_motion['object_translation'][self._step_count-1][:2]  # x,y
            physics.data.qpos[32] = self.reference_motion._reference_motion['object_translation'][self._step_count-1][-1] - z_global_local_offset  # z, global frame to local frame
            euler = quat2euler(self.reference_motion._reference_motion['object_orientation'][self._step_count-1])
            physics.data.qpos[33:36] = euler

            # fixed object
            if self._multi_obj:
                physics.data.qpos[-6:] = self.start_state['fixed']['position']

    @property
    def substeps(self):
        substeps = self.reference_motion.substeps
        # print('substeps: ', substeps)
        return substeps

    def get_termination(self, physics):
        loose_hand = False
        if loose_hand and not self.ref_only and self.reference_motion.next_done: # after training, test with post procedure: lossen the hand
            # if done, additional steps for openning the hand
            # this will not affect the reward
            smooth_loosen_steps = 30
            self.additional_step_cnt +=1
            target_hand_pose = np.zeros(24)   # fully open hand pose
            # target_hand_pose = self.start_state['position'][6:30]  # set hand to initial joint position
            if self.additional_step_cnt == 1: 
                self.end_hand_pose = copy.deepcopy(physics.data.qpos[:6])
                self.end_hand_joint_pose = copy.deepcopy(physics.data.qpos[6:30])
                self.end_obj_pose = copy.deepcopy(physics.data.qpos[30:])

            # smoothly move to target hand pose (not grasping object)
            if self.additional_step_cnt <= smooth_loosen_steps:
                physics.data.qpos[6:30] = self.end_hand_joint_pose + (target_hand_pose - self.end_hand_joint_pose)*self.additional_step_cnt/smooth_loosen_steps  # set hand to initial joint position
            else:
                physics.data.qpos[6:30] = target_hand_pose

            # set fixed obj and hand pose
            # physics.data.qpos[:6] = self.end_hand_pose
            # physics.data.qpos[30:] = self.end_obj_pose
            # physics.data.qvel[30:] = 0

            self.additional_step = True
            if self.additional_step_cnt > 150:
                return 0.0
            else:
                None
        else: # without post procedure, or during training
            if self.reference_motion.next_done:
                return 0.0

        return super().get_termination(physics)
