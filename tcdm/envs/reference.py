# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import copy
import numpy as np
from tcdm.util.motion import Pose, PoseAndVelocity
from tcdm.util.geom import euler2quat

class HandReferenceMotion(object):
    def __init__(self, motion_file, start_step=0):
        self._load_motion(motion_file)
        self._substeps = int(self._reference_motion['SIM_SUBSTEPS'])
        self._data_substeps = self._reference_motion.get('DATA_SUBSTEPS', self._substeps)
        self._step, self._start_step = 0, int(start_step)


    def _load_motion(self, motion_file):
        if '.npz' not in motion_file:
            motion_file += '.npz'
        motion_file = np.load(motion_file, allow_pickle=True)
        self._reference_motion =  {k:v for k, v in motion_file.items()}
        self._reference_motion['s_0'] = self._reference_motion['s_0'][()]


    def set_with_given_ref(self, ref):
        self._reference_motion = ref
        self._substeps = int(ref['SIM_SUBSTEPS'])
        self._data_substeps = ref.get('DATA_SUBSTEPS', self._substeps)


    def reset(self):
        self._step = 0
        return copy.deepcopy(self._reference_motion['s_0'])


    def step(self):
        self._check_valid_step()
        self._step += self._data_substeps


    def revert(self):
        self._step -= self._data_substeps
        self._check_valid_step()


    def __len__(self):
        return self.length


    @property
    def t(self):
        return self._step


    @property
    def data_substep(self):
        return self._data_substeps


    @property
    def time(self):
        return float(self._step) / self.length


    @property
    def qpos(self):
        self._check_valid_step()
        return self._reference_motion['s'][self._step].copy()


    @property
    def qvel(self):
        self._check_valid_step()
        return self._reference_motion['sdot'][self._step].copy()


    @property
    def eef_pos(self):
        self._check_valid_step()
        return self._reference_motion['eef_pos'][self._step].copy()


    @property
    def human_joint_coords(self):
        self._check_valid_step()
        return self._reference_motion['human_joint_coords'][self._step].copy()


    @property
    def eef_quat(self):
        self._check_valid_step()
        return self._reference_motion['eef_quat'][self._step].copy()


    @property
    def eef_linear_velocity(self):
        self._check_valid_step()
        return self._reference_motion['eef_velp'][self._step].copy()


    @property
    def eef_angular_velocity(self):
        self._check_valid_step()
        return self._reference_motion['eef_velr'][self._step].copy()


    @property
    def body_poses(self):
        pos = self.eef_pos
        rot = self.eef_quat
        lv = self.eef_linear_velocity
        av = self.eef_angular_velocity
        return PoseAndVelocity(pos, rot, lv, av)


    @property
    def substeps(self):
        return self._substeps


    @property
    def done(self):
        assert self._step is not None, "Motion must be reset before it can be done"
        return self._step >= self.length


    @property
    def next_done(self):
        assert self._step is not None, "Motion must be reset before it can be done"
        return self._step >= self.length - self._data_substeps


    @property
    def n_left(self):
        assert self._step is not None, "Motion must be reset before lengths calculated"
        n_left = (self.length - self._step) / float(self._data_substeps) - 1
        return int(max(n_left, 0))


    @property
    def n_steps(self):
        n_steps = self.length / float(self._data_substeps) - 1
        return int(max(n_steps, 0))


    @property
    def length(self):
        if 'length' in self._reference_motion:
            return self._reference_motion['length']
        return self._reference_motion['s'].shape[0]


    @property
    def start_step(self):
        return self._start_step


    def _check_valid_step(self):
        assert not self.done, "Attempting access data and/or step 'done' motion"
        assert self._step >= self._start_step, "step must be at least start_step"


    def __getitem__(self, key):
        value =  copy.deepcopy(self._reference_motion[key])
        if not isinstance(value, np.ndarray):
            return value
        if len(value.shape) >= 2:
            return value[self._start_step::self._data_substeps]
        return value


class HandObjectReferenceMotion(HandReferenceMotion):
    def __init__(self, object_name, motion_file):
        super().__init__(motion_file)
        self._object_name = object_name


    @property
    def object_name(self):
        return self._object_name


    @property
    def object_pos(self):
        self._check_valid_step()
        return self._reference_motion['object_translation'][self._step].copy()


    @property
    def floor_z(self):
        return float(self._reference_motion['object_translation'][0,2])


    @property
    def object_rot(self):
        self._check_valid_step()
        return self._reference_motion['object_orientation'][self._step].copy()


    @property
    def object_pose(self):
        pos = self.object_pos[None]
        rot = self.object_rot[None]
        return Pose(pos, rot)


    @property
    def goals(self):
        g = []
        for i in [1, 5, 10]:
            i = min(self._step + i, self.length-1)
            for k in ('object_orientation', 'object_translation'):
                g.append(self._reference_motion[k][i].flatten())
        return np.concatenate(g)


import scipy.interpolate as interpolate
# this is a valid operation range for object on table, global frame, centered at the table center
# OBJ_X_RANGE = [-0.3, 0.3]
# OBJ_Y_RANGE = [-0.3, 0.3]
# OBJ_Z_RANGE = [-0.1, 0.5]
# considering the range of robot hand: TCDM/tcdm/envs/assets/robots/adroit/actuators.xml
# x: [-0.25, 0.25], y: [-0.2, 0.1], z: [-0.3, 0.5]
OBJ_X_RANGE = [-0.2, 0.2]
OBJ_Y_RANGE = [-0.1, 0.1]
OBJ_Z_RANGE = [-0.1, 0.5]
# OBJ_QUAT_RANGE = [-1., 1]
OBJ_QUAT_INC_RANGE = [-0.1, 0.1] # incremental range
OBJ_TRANS_RANGE = [OBJ_X_RANGE, OBJ_Y_RANGE, OBJ_Z_RANGE]


def interpolate_data(data, range=None, offset=0, initial_point=None, incremental=False, random_sample=3): # random_sample is the number of random middle points
    """_summary_

    :param data: _description_
    :type data: _type_
    :param range: _description_, defaults to None
    :type range: _type_, optional
    :param offset: _description_, defaults to 0
    :type offset: int, optional
    :param initial_point: _description_, defaults to None
    :type initial_point: _type_, optional
    :param incremental: incremental value compared to last point, defaults to False
    :type incremental: bool, optional
    :param random_sample: _description_, defaults to 3
    :type random_sample: int, optional
    :return: _description_
    :rtype: _type_
    """
    if range is None:
        points = np.random.uniform(np.min(data), np.max(data), size=random_sample)
    else:
        points = np.random.uniform(*range, size=random_sample)

    # points += offset  # adding offset may lead to out of range

    if initial_point is not None: # set the same initial point as original traj
        points[0] = initial_point
    else:
        points[0] = data[0]
    len_points = points.shape[0]

    if incremental:
        points = np.cumsum(points)  # add incremental value to last point

    f = interpolate.interp1d(np.arange(len_points), points, kind='quadratic', fill_value="extrapolate")  # ‘linear’, ‘nearest’, ‘nearest-up’, ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’, ‘previous’, or ‘next’.

    x_new = np.arange(0, len_points-1, (len_points-1)/data.shape[0])
    y_new = f(x_new)
    y_new[-1] = points[-1]
    return y_new


def random_generate_ref(original_ref, initial_translation_offset=np.zeros(3)):
    trans_data = original_ref['object_translation']
    ori_data = original_ref['object_orientation']

    if initial_translation_offset == 'random': # random valid translation offset for x and y
        initial_translation_offset = [np.random.uniform(*OBJ_X_RANGE), np.random.uniform(*OBJ_Y_RANGE), 0]

    if original_ref['s_0']['motion_planned']['position'].shape[0] == 36:
        # set initial position for object and hand with offset
        original_ref['s_0']['motion_planned']['position'][30:33] += initial_translation_offset  # obj: (x, y, z)
        original_ref['s_0']['motion_planned']['position'][0] -= initial_translation_offset[0]  # hand: (-x, z, y)
        original_ref['s_0']['motion_planned']['position'][1] += initial_translation_offset[2]  # hand: (-x, z, y)
        original_ref['s_0']['motion_planned']['position'][2] += initial_translation_offset[1]  # hand: (-x, z, y)

    new_trans_data = []
    for i, d in enumerate(trans_data.T):
        if original_ref['s_0']['motion_planned']['position'].shape[0] == 36: # 30 hand + 3 pos + 3 ori of object
            initial_point = original_ref['s_0']['motion_planned']['position'][30+i] # set the original position as initial sampled point position
        else:
            initial_point = None
        new_trans_data.append(interpolate_data(d, OBJ_TRANS_RANGE[i], initial_translation_offset[i], initial_point=initial_point))
    new_trans_data = np.array(new_trans_data).T

    new_ori_data = []
    ini_ori_euler = original_ref['s_0']['motion_planned']['position'][33:37]  # dim 3
    ini_ori_quat = euler2quat(ini_ori_euler)  # dim 4
    for i, d in enumerate(ori_data.T):
        if original_ref['s_0']['motion_planned']['position'].shape[0] == 36: # 30 hand + 3 pos + 3 ori of object
            initial_point = ini_ori_quat[i]
        else:
            initial_point = None
        new_ori_data.append(interpolate_data(d, OBJ_QUAT_INC_RANGE, initial_point=initial_point, incremental=True))
    new_ori_data = np.array(new_ori_data).T

    new_traj = copy.copy(dict(original_ref))

    new_traj['object_translation'] = new_trans_data
    new_traj['object_orientation'] = new_ori_data
    new_traj['SIM_SUBSTEPS'] = 10

    return new_traj