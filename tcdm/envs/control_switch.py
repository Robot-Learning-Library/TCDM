# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import copy

from tcdm.envs.reference import HandObjectReferenceMotion
from tcdm.util.geom import quat2euler
from tcdm.envs.control import Task, SingleObjectTask, _normalize_action, _denormalize_action
from tcdm.planner.generate import motion_plan_one_obj


class GeneralReferenceMotionSwitchTask(SingleObjectTask):
    def __init__(self, reference_motion, 
                       reward_fns, 
                       init_key, 
                       obj_names,
                       ref_only,
                       move_obj_seq,
                       target_obj_Xs,
                       traj_folder='new_agents/banana_fryingpan_pass1',
                       use_saved_traj=False, 
                       reward_weights=None,
                    #    random=None
                       ):
        self.target_obj_Xs = target_obj_Xs
        self.move_obj_seq = move_obj_seq
        self.switch_num = 0 
        self.switch_num_max = len(self.move_obj_seq)-1 if self.move_obj_seq is not None else len(self.obj_names)-1
        self.curr_move_obj_idx = self.switch_num if self.move_obj_seq is None else self.move_obj_seq[0]
        self.traj_folder = traj_folder
        self.use_saved_traj = use_saved_traj

        self.ref_only = ref_only
        self.obj_names = obj_names
        self.reference_motion = reference_motion  # this is the first reference motion

        self._init_key = init_key
        self.z_global_local_offset = -0.2

        # object offset in global frame
        self.avoid_collision_z_shift = 0.0
        original_banana_ini_pose = [ 0.01895152, -0.01185687, -0.17970488+0.2]  # reference_motion.reset()[self._init_key]['position'][30:33] in original banana env, z+0.2 to global frame
        start_state = self.reference_motion.reset()[self._init_key]
        self.offset = start_state[str(self.curr_move_obj_idx)]['position'][:3] - original_banana_ini_pose # 3 of 6 as xyz
        print(self.offset)        
        super().__init__(obj_names[0], reward_fns, reward_weights, random=None)


    def switch_obj(self, physics):
        if self.switch_num == self.switch_num_max:   # terminate episode
            return False
        self.switch_num += 1
        if self.move_obj_seq is None:
            self.curr_move_obj_idx = self.switch_num
        else:
            self.curr_move_obj_idx = self.move_obj_seq[self.switch_num]
        object_name = self.move_obj_seq[self.curr_move_obj_idx]
        traj_path = f'./{self.traj_folder}/traj_{self.switch_num}.npz'
        if not self.use_saved_traj:
            cur_qpos = copy.deepcopy(physics.data.qpos[30:].reshape(-1, 6))
            cur_qpos[:, 2] += -self.z_global_local_offset   # local to global...
            traj, _ = motion_plan_one_obj(
                obj_list=[name.split('/')[0] for name in self.obj_names], 
                move_obj_idx=self.curr_move_obj_idx, 
                obj_Xs=cur_qpos.tolist(),  # get current object poses
                move_obj_target_X=self.target_obj_Xs[self.curr_move_obj_idx], 
                save_path=traj_path,
                ignore_collision_obj_idx_all=[idx for idx in range(len(self.obj_names)) if idx != self.curr_move_obj_idx],  # ignore collision with all other objects
                visualize=False) # TODO: cfg for visualize

        # TODO: directly pass trajectory instead of saving to file
        self.reference_motion = HandObjectReferenceMotion(object_name, traj_path)
        # reset hand pose
        start_state = self.reference_motion.reset()[self._init_key]
        
        # object offset in global frame
        original_pan_ini_pose = [0.00130683,  0.03177048, -0.17431791+0.2] # reference_motion.reset()[self._init_key]['position'][30:33] in original banana env, z+0.2 to global frame
        self.offset = start_state[str(self.curr_move_obj_idx)]['position'][:3] - original_pan_ini_pose # 3 of 6 as xyz
        print(self.offset)

        physics.data.qpos[:30] = start_state['position']   # TODO for different object, this ini pose can be different
        # physics.data.qpos[:3] -= self.offset
        # add object offset for hand; global to local (qpos)
        physics.data.qpos[0] -= self.offset[0]
        physics.data.qpos[1] += self.offset[2]
        physics.data.qpos[2] += self.offset[1]
        physics.data.qpos[2] += self.avoid_collision_z_shift
        physics.data.qvel[:30] = start_state['velocity']
        
        # Reset step
        self._step_count = 0

    def check_switch(self):
        return self.reference_motion.next_done


    @property
    def substeps(self):
        substeps = self.reference_motion.substeps
        # print('substeps: ', substeps)
        return substeps


    def get_observation(self, physics):
        obs = Task.get_observation(self, physics, self.curr_move_obj_idx)  # obj_idx specifies which object to get observation
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
        obs['current_move_obj_idx'] = np.array([self.curr_move_obj_idx], dtype=float)  # get the current object index
        return obs


    def initialize_episode(self, physics):
        # This works for now for running visualization but not sure if the correct practice
        self._step_count = 0

        self.additional_step_cnt = 0 # step after the reference motion is done
        self.additional_step = False # whether to step after the reference motion is done
        start_state = self.reference_motion.reset()[self._init_key]  # _init_key='motion_planned'
        self.start_state = start_state

        with physics.reset_context():
            # hand
            physics.data.qpos[:30] = start_state['position']
            # physics.data.qpos[:3] -= self.offset
            # add object offset for hand; global to local (qpos)
            physics.data.qpos[0] -= self.offset[0]
            physics.data.qpos[1] += self.offset[2]
            physics.data.qpos[2] += self.offset[1]
            physics.data.qpos[2] += self.avoid_collision_z_shift
            physics.data.qvel[:30] = start_state['velocity']

            # objects
            for i, obj_name in enumerate(self.obj_names):
                physics.data.qpos[30+6*i:36+6*i] = start_state[str(i)]['position']
                physics.data.qvel[30+6*i:36+6*i] = start_state[str(i)]['velocity']
                physics.data.qpos[30+6*i+2] += self.z_global_local_offset

        return super().initialize_episode(physics)


    def before_step(self, action, physics):
        """
        action is 30 dimensional: 3 for hand translation, 3 for hand rotation, 24 for joint pose, all abosulte position (not delta)!
        """
        if self.ref_only:
            action = 30*[0]
        else:
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
        switch = self.check_switch()
        print('step: ', self._step_count, ' current object index: ', self.curr_move_obj_idx)
        if self.ref_only: # set position of objects (according to reference) and hand (fixed)
            
            # hand - leave it high up
            physics.data.qpos[:30] = self.start_state['position']
            physics.data.qvel[:30] = self.start_state['velocity']
            physics.data.qpos[1] = 0.7  # z-axis of hand
            # print(physics.data.qpos[30:], self.curr_move_obj_idx)

            # objects
            for i, obj_name in enumerate(self.obj_names):
                if i == self.curr_move_obj_idx:
                    physics.data.qpos[30+6*i:33+6*i] = self.reference_motion._reference_motion['object_translation'][self._step_count-1]
                    physics.data.qpos[33+6*i:36+6*i] = quat2euler(self.reference_motion._reference_motion['object_orientation'][self._step_count-1])
                    physics.data.qpos[30+6*i+2] += self.z_global_local_offset

                    # if switch:  # make object static at switch
                    #     physics.data.qvel[30+6*i:30+6*i+6] = 6*[0]
                # else:
                #     physics.data.qvel[30+6*i:30+6*i+6] = 6*[0]  # make other objects static
        
        if switch:
            # Check if switch trajectory
            self.switch_obj(physics)
            print('Switched trajectory!')


    def get_termination(self, physics):
        if self.reference_motion.next_done and self.switch_num == self.switch_num_max:
            return 0.0
        return super().get_termination(physics)


