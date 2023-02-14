import trimesh
import copy
import numpy as np
import os
from tcdm.planner.util.geometry import get_transform
from tcdm.planner.rrt_star_bid import RRTStarBidirectional
from tcdm.planner.search_space import SearchSpace

from tcdm.util.geom import quat2euler, euler2quat

import scipy.interpolate as interpolate
from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as R


# TODO: use cfg

# naming convention：
# X: 6-dim tuple with 3D position and 3D Euler angle (XYZ extrinsic)
# Y: 4x4 transformation matrix
# Z: 7-dim tuple with 3D position and quaternion (w,x,y,z)
# p: 3-dim position
# r: 3-dim Euler angle (XYZ intrinsic convention, 'rxyz' in trimesh, 'XYZ' in scipy)
#! X and r uses different convention
# q: 4-dim quaternion in (x,y,z,w) - scipy convention
#! mujoco/tcdm/trimesh.transformations uses (w,x,y,z) convention
# R: scipy rotation object


def interpolate_pos(p_init, p_end, num_point):
    f = interpolate.interp1d(
            np.array([0,num_point-1]), 
            np.vstack((p_init, p_end)).T, 
            kind='linear')  # ‘linear’, ‘nearest’, ‘nearest-up’, ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’, ‘previous’, or ‘next’. 
    x_new = np.arange(0, num_point, 1)
    y_new = f(x_new)
    return x_new, y_new.T


def interpolate_quat(R_init, R_end, num_point):
    quats = R.concatenate([R_init, R_end])
    f = Slerp([0, num_point-1], quats)

    x_new = np.arange(0, num_point, 1)
    # y_new = f(x_new).as_euler('XYZ', degrees=False)
    y_new = f(x_new).as_quat()
    return x_new, y_new


def load_motion(motion_file):
    motion_file = np.load(motion_file, allow_pickle=True)
    reference_motion =  {k:v for k, v in motion_file.items()}
    reference_motion['s_0'] = reference_motion['s_0'][()]
    return reference_motion


def motion_plan_one_obj(obj_list, 
                        move_obj_idx, 
                        obj_Xs, 
                        move_obj_target_X,
                        save_path,
                        ignore_collision_obj_idx_all=[],
                        collision_threshold=0.003,
                        p_bound_buffer=0.1, 
                        q_bound_buffer=0.2,
                        ref_traj_file_path='./trajectories',
                        visualize=False):
    """_summary_

    :param obj_list: list of object names
    :type obj_list: _type_
    :param move_obj_idx: the idex of the object to be moved
    :type move_obj_idx: _type_
    :param obj_Xs: the 6-dim poses of all objects
    :type obj_Xs: _type_
    :param move_obj_target_X: the 6-dim pose of the target pose of the manipulated object
    :type move_obj_target_X: _type_
    :param save_path: the path to save the motion plan
    :type save_path: _type_
    :param ignore_collision_obj_idx_all: the idx of object to ignore collision with, defaults to []
    :type ignore_collision_obj_idx_all: list, optional
    :param collision_threshold: _description_, defaults to 0.003
    :type collision_threshold: float, optional
    :param p_bound_buffer: _description_, defaults to 0.1
    :type p_bound_buffer: float, optional
    :param q_bound_buffer: _description_, defaults to 0.2
    :type q_bound_buffer: float, optional
    :param visualize: _description_, defaults to False
    :type visualize: bool, optional
    """
    def get_path(obj_name):
        obj_path = './tcdm/envs/assets/meshes/objects/'
        return os.path.join(obj_path, obj_name + '/' + obj_name + '.stl')

    # Load objects
    objects = []    
    for obj_name in obj_list:
        obj_path = get_path(obj_name)
        objects.append(trimesh.load(obj_path))
        # objects.append(trimesh.load(file_obj = trimesh.util.wrap_as_stream(obj_path), file_type='stl')) # load from string
    
    # TODO: naming to specify w at front or back?
    def X_to_Z(pose):  # from extrinsic xyz to (w,x,y,z)
        return pose[:3] + euler2quat(pose[3:]).tolist()

    # Get all poses from euler to quaternion
    obj_Ys = []
    for obj_X in obj_Xs:
        obj_Ys.append(get_transform(X_to_Z(obj_X)))

    # get target pose of the manipulated object from euler to quaternion
    move_obj = objects[move_obj_idx]
    move_obj_X = obj_Xs[move_obj_idx]
    move_obj_Y = obj_Ys[move_obj_idx]
    move_obj_target_Y = get_transform(X_to_Z(move_obj_target_X))
    move_obj_Z = X_to_Z(move_obj_X)
    move_obj_target_Z = X_to_Z(move_obj_target_X)

    # Configuration space boundaries - use quaternions of initial and end poses, plus some buffer
    # pose_lower = np.array([-0.1, -0.2, -0.2,  -1, -1, -1, -1])
    # pose_upper = np.array([ 0.1,  0.2,  0.2,  1,  1,  1,  1])
    p_min = np.minimum(move_obj_Z[:3], move_obj_target_Z[:3]) - p_bound_buffer
    p_max = np.maximum(move_obj_Z[:3], move_obj_target_Z[:3]) + p_bound_buffer
    q_min = np.minimum(move_obj_Z[3:], move_obj_target_Z[3:]) - q_bound_buffer
    q_max = np.maximum(move_obj_Z[3:], move_obj_target_Z[3:]) + q_bound_buffer
    # pose_lower[3:] = q_min
    # pose_upper[3:] = q_max
    pose_lower = np.hstack((p_min, q_min))
    pose_upper = np.hstack((p_max, q_max))

    # Visualize the final scene
    scene_end = trimesh.scene.scene.Scene()
    other_objects = []
    other_objects_Y = []
    other_objects_idx = []
    for i, obj in enumerate(objects):
        if i != move_obj_idx:
            other_objects.append(obj)
            other_objects_Y.append(obj_Ys[i])
            other_objects_idx.append(i)
            scene_end.add_geometry(obj, transform=obj_Ys[i])
    scene_end.add_geometry(move_obj, transform=move_obj_target_Y)
    scene_end.add_geometry(move_obj, transform=move_obj_Y)
    if visualize:
        print('Visualizing the final scene...')
        scene_end.show()

    # Initialize collision checker - moving object is labeled as str(move_obj_idx)
    collision_checker = trimesh.collision.CollisionManager()
    for i, obj in enumerate(objects):
        collision_checker.add_object(str(i), obj, obj_Ys[i])
    print(f'Any collision?: {collision_checker.in_collision_internal()}. '
          f'Minimum clearance: {collision_checker.min_distance_internal()}')
    # print('Time for collision checking: ', time.time()-s1)

    # Run RRT - pass in 7-dim pose instead of transformation matrix
    path, path_full = rrt(collision_checker, 
                            pose_lower, 
                            pose_upper, 
                            move_obj_idx, 
                            move_obj_Z,
                            move_obj_target_Z, 
                            # other_objects,
                            # other_objects_transformed_pose, 
                            collision_threshold,
                            ignore_collision_obj_idx_all)

    # Result
    print('Pos lower bound:', pose_lower[:3])
    print('Pos upper bound:', pose_upper[:3])
    print('Quaternion lower bound:', pose_lower[3:])
    print('Quaternion upper bound:', pose_upper[3:])
    print('Number of points before interpolation: ', len(path))
    print('Number of points after interpolation: ', len(path_full))
    # print('Position trajectory after interpolation: ', [path[:3] for path in path_full])
    # print('Quaternion (w,x,y,z) trajectory after interpolation: ', [path[3:] for path in path_full])

    # Visualize the final scene with interpolation
    scene_planned = trimesh.scene.scene.Scene()
    for i, obj in enumerate(other_objects):
        scene_planned.add_geometry(obj, transform=other_objects_Y[i])
    for X in path_full:
        scene_planned.add_geometry(move_obj, transform=get_transform(X))
    if visualize:
        print('Visualizing the planned scene...')
        scene_planned.show()

    # Feed in - no need to convert since mujoco uses (w,x,y,z)
    translation_tcdm = []
    orientation_tcdm = []
    for path in path_full:
        translation_tcdm += [path[:3]]
        # q = R.from_euler(seq='XYZ', angles=path[3:], degrees=False).as_quat()
        # orientation_tcdm += [tuple([q[-1]]+list(q[:3]))]  # (x,y,z,w) -> (w,x,y,z)
        orientation_tcdm += [path[3:]]

    # get the reference motion of corresponding object
    object_name = obj_list[move_obj_idx]
    # ref_traj_file_path = './trajectories'
    for filename in os.listdir(ref_traj_file_path):
        if object_name in filename and '.npz' in filename:
            ref_traj_file = os.path.join(ref_traj_file_path, filename)
            break
    print('plan using traj file: ', ref_traj_file)
    data = load_motion(ref_traj_file)
    # data = load_motion('./trajectories/banana_pass1.npz')
    traj = copy.copy(dict(data))
    # print(traj['s_0']['motion_planned']['position'][-6:])

    # traj['s_0']['motion_planned']['position'][-6:] = np.hstack((
    #     translation_tcdm[0],
    #     quat2euler(orientation_tcdm[0]),
    # ))  # local frame
    # traj['s_0']['motion_planned']['fixed'] = {}
    # # traj['s_0']['motion_planned']['fixed']['position'] = np.array(other_objects_idx)
    # traj['s_0']['motion_planned']['fixed']['position'] = np.array(objects_pose[other_objects_idx[0]])

    # Remove obj pose from s_0 - only keep hand's
    # TODO: different objects have different hand poses?
    traj['s_0']['motion_planned']['position'] = np.copy(traj['s_0']['motion_planned']['position'])[:30]
    traj['s_0']['motion_planned']['velocity'] = np.copy(traj['s_0']['motion_planned']['velocity'])[:30]

    # Add obj pose to s_0
    for obj_idx, obj_X in enumerate(obj_Xs):
        traj['s_0']['motion_planned'][str(obj_idx)] = {}
        if obj_idx == move_obj_idx:
            traj['s_0']['motion_planned'][str(obj_idx)]['position'] = \
                np.hstack((translation_tcdm[0],
                           quat2euler(orientation_tcdm[0]),
                          ))
        else:
            traj['s_0']['motion_planned'][str(obj_idx)]['position'] = np.array(obj_X)
        traj['s_0']['motion_planned'][str(obj_idx)]['velocity'] = np.zeros((6))

    # Copy trajectory over
    traj['object_translation'] = np.vstack((translation_tcdm))
    traj['object_orientation'] = np.vstack((orientation_tcdm))
    traj['length'] = len(translation_tcdm)
    # traj['SIM_SUBSTEPS'] = int(data['SIM_SUBSTEPS']/3)
    traj['DATA_SUBSTEPS'] = 1
    traj['offset'] = np.zeros((3))
    np.savez(save_path, **traj)  # save a dict as npz
    print('Saved trajectory to: ', save_path)

    return traj, save_path


def rrt(collision_checker, 
        pose_lower, 
        pose_upper, 
        move_obj_idx, 
        move_obj_init_Z, 
        move_obj_end_Z, 
        # target_objs, 
        # target_objs_X, 
        collision_threshold,
        ignore_collision_obj_idx_all,
        p_interp_threshold=0.015,
        q_interp_threshold=0.1,
        p_interp_dist=0.01,
        q_interp_dist=0.05):
    """Plan in 7-dim space: 3D translation + 4D quaternion (w,x,y,z)"""

    # RRT parameters
    Q = np.array([(0.01, 4)])  # length of tree edges 
    r = 0.01  # length of smallest edge to check for intersection with obstacles
    max_samples = 10240  # max number of samples to take before timing out
    rewire_count = 32  # optional, number of nearby branches to rewire
    prc = 0.1  # probability of checking for a connection to goal

    # Run RRT
    X_dimensions = np.vstack((pose_lower, pose_upper)).T
    rrt = RRTStarBidirectional(
        X=SearchSpace(X_dimensions, 
                      collision_checker, 
                      collision_threshold, 
                      move_obj_idx, 
                      ignore_collision_obj_idx_all), 
        Q=Q,
        x_init=tuple(move_obj_init_Z),
        x_goal=tuple(move_obj_end_Z),
        max_samples=max_samples, 
        r=r, 
        prc=prc, 
        rewire_count=rewire_count,
        )
    path = rrt.rrt_star_bidirectional()

    # Visualize the final scene
    # scene_planned = trimesh.scene.scene.Scene()
    # for target_obj, target_obj_X in zip(target_objs, target_objs_X):
    #     scene_planned.add_geometry(target_obj, transform=target_obj_X)
    # for X in path:
    #     scene_planned.add_geometry(move_obj, transform=get_transform(X))
    # scene_planned.show()

    path_full = []
    for X_ind, X in enumerate(path):
        if X_ind == 0:
            prev_X = X

        # Check if interp
        p_norm_diff = np.linalg.norm(np.array(X[:3]) - np.array(prev_X[:3]))
        q_norm_diff = np.linalg.norm(np.array(X[3:]) - np.array(prev_X[3:]))
        if p_norm_diff > p_interp_threshold or q_norm_diff > q_interp_threshold:
        # if p_norm_diff > p_interp_threshold:

            # Figure out number of interp points
            # num_point = int(p_norm_diff/p_interp_dist) + 1
            num_point = max(int(p_norm_diff/p_interp_dist) + 1, int(q_norm_diff/q_interp_dist) + 1)
            print(f'Interpolating {num_point} points!')

            # Interp position and orientation separately
            _, p_interp_all = interpolate_pos(prev_X[:3], X[:3], num_point)
            # _, r_interp_all = interpolate_quat(
            #     R.from_euler(seq='XYZ', angles=prev_X[3:], degrees=False),
            #     R.from_euler(seq='XYZ', angles=X[3:], degrees=False),
            #     num_point)
            _, q_interp_all = interpolate_quat(
                R.from_quat(list(prev_X[4:])+[prev_X[3]]),  # convert (w,x,y,z) to (x,y,z,w) for scipy rotation
                R.from_quat(list(X[4:])+[X[3]]),
                num_point)

            # Do not add the first one since that's prev
            for p_ind, (p_interp, q_interp) in enumerate(zip(p_interp_all, q_interp_all)):
                if p_ind == 0:
                    continue
                path_full += [tuple(list(p_interp)+[q_interp[-1]]+list(q_interp[:3]))]  # convert back to (w,x,y,z)
        else:
            path_full += [X]

        # Update previous point
        prev_X = X
    return path, path_full


if __name__ == "__main__":

    # Define objects for new plan
    # obj_list = ['cup', 'cup']
    # obj_list = ['toruslarge', 'knife']
    obj_list = ['toruslarge', 'toruslarge']
    move_obj_idx = 0
    # save_path = './new_agents/cup_cup_move1/traj_0.npz'
    # save_path = './new_agents/toruslarge_knife_move1/traj_0.npz'
    save_path = './new_agents/toruslarge_toruslarge_stack/traj_0.npz'

    # Open the original trajectory file and see the initial pose for determining poses of the new plan
    # ref_traj_file = './new_agents/torus_knife_move1/toruslarge_inspect1.npz'
    # data = load_motion(ref_traj_file)
    # print(data['s_0'])

    # Define poses for new plan
    # obj_Xs = [[-0.25, 0.1, -1.49e-01+0.2, 1.03e-04, 2.43e-04, 1.79e+00],
    #           [-2.21e-03, 4.10e-03, -1.49e-01+0.2, 1.03e-04, 2.43e-04, 1.79e+00]]
    # move_obj_target_X = [-0.15, 4.10e-03, -1.49e-01+0.2, 1.03e-04, 2.43e-04, 1.79e+00]  # cup-cup: move first cup from x=-0.25 to x=-0.10, while the fixed cup is at x=0
    # obj_Xs = [[-0.2, -1.43946832e-02,
    #    -1.79232350e-01+0.2, -5.76184115e-05,  3.68892728e-05, -1.15680692e+00],
    #           [3.27832237e-03+0.2, -3.30545511e-04,
    #    -1.94037339e-01+0.2,  3.08262443e+00,  9.59880039e-02, -2.97606327e+00]]
    # move_obj_target_X = [0, -1.43946832e-02,
    #    -1.79232350e-01+0.205, -5.76184115e-05,  3.68892728e-05, -1.15680692e+00]  # otruslarge-knife: move torus from x=-0.2 to x=0, while the fixed knife is at x=0.2
    obj_Xs = [[-0.2, -1.43946832e-02,-1.79232350e-01+0.19, -5.76184115e-05,  3.68892728e-05, -1.15680692e+00],
              [0.2, -1.43946832e-02, -1.79232350e-01+0.19, -5.76184115e-05,  3.68892728e-05, -1.15680692e+00]]
    move_obj_target_X = [0, -1.43946832e-02, -1.79232350e-01+0.19, -5.76184115e-05,  3.68892728e-05, -1.15680692e+00]  # otruslarge-toruslarge: move torus from x=-0.2 to x=0, while the other torus is at x=0.2

    # Plan
    traj, save_path = motion_plan_one_obj(obj_list, 
                                          move_obj_idx, 
                                          obj_Xs, 
                                          move_obj_target_X,
                                          save_path,
                                          ignore_collision_obj_idx_all=[],
                                          collision_threshold=0.003,
                                          p_bound_buffer=0.1, 
                                          q_bound_buffer=0.2,
                                          visualize=True)

# def motion_plan(target_obj_path, 
#                 float_obj_path, 
#                 traj_path, 
#                 collision_threshold=0.003, 
#                 q_bound_buffer=0.2):
#     # Load meshes
#     target_obj = trimesh.load(target_obj_path)
#     float_obj = trimesh.load(float_obj_path)
#     # print(target_obj.bounds)
#     # target_obj.show()

#     # Initial pose of the target and float objects - x,y,z,r,p,y - Euler extrinsic XYZ convention
#     float_obj_X_init_r_array = [0.01895152, -0.01185687, 0.021, -3.05750797, 0.08599904, -1.99028331]  # from banana_pass1.npz
#     float_obj_X_init_array = float_obj_X_init_r_array[:3] + euler2quat(float_obj_X_init_r_array[3:]).tolist()   # convert to quat (w,x,y,z)
#     # float_obj_X_init_array = [0.05, 0.0, 0-0.2+0.023, 0.0, 0.0, 0.0]  # account for 20cm offset in z when loading objects in tcdm
#     # target_obj_X_array = [-0.02, -0.165, 0-z_offset+0.04, 0.0, 0.0, 0.0]
#     target_obj_X_array_r_array = [-0.2, -0.165, 0.04, 0.0, 0.0, 0.0]  # avoid collision with the cup
#     target_obj_X_array = target_obj_X_array_r_array[:3] + euler2quat(target_obj_X_array_r_array[3:]).tolist()
#     float_obj_X_init = get_transform(float_obj_X_init_array)
#     target_obj_X = get_transform(target_obj_X_array)

#     # Final pose of the float object
#     float_obj_X_end_r_array = target_obj_X_array_r_array.copy()
#     float_obj_X_end_r_array[1] -= 0.01 
#     float_obj_X_end_r_array[2] += 0.149
#     float_obj_X_end_r_array[-2] = -1.57
#     float_obj_X_end_array = float_obj_X_end_r_array[:3] + euler2quat(float_obj_X_end_r_array[3:]).tolist()
#     float_obj_X_end = get_transform(float_obj_X_end_array)

#     # Configuration space boundaries - use quaternions of initial and end poses, plus some buffer
#     pose_lower = np.array([-0.1, -0.2, -0.2,  -1, -1, -1, -1])
#     pose_upper = np.array([ 0.1,  0.2,  0.2,  1,  1,  1,  1])
#     q_min = np.minimum(float_obj_X_init_array[3:], float_obj_X_end_array[3:]) - q_bound_buffer
#     q_max = np.maximum(float_obj_X_init_array[3:], float_obj_X_end_array[3:]) + q_bound_buffer
#     pose_lower[3:] = q_min
#     pose_upper[3:] = q_max

#     # Visualize the final scene
#     scene_end = trimesh.scene.scene.Scene()
#     scene_end.add_geometry(target_obj, transform=target_obj_X)
#     scene_end.add_geometry(float_obj, transform=float_obj_X_end)
#     scene_end.add_geometry(float_obj, transform=float_obj_X_init)
#     scene_end.show()

#     # Collision checking
#     collision_checker = trimesh.collision.CollisionManager()
#     collision_checker.add_object('target', target_obj, target_obj_X)
#     collision_checker.add_object('float', float_obj, float_obj_X_end)
#     # import time
#     # s1 = time.time()
#     print(collision_checker.in_collision_internal())    # 0.0002-0.0006s
#     print(collision_checker.min_distance_internal())    # 0.0002-0.0006s
#     # print('Time for collision checking: ', time.time()-s1)

#     path_full = rrt(collision_checker, pose_lower, pose_upper, float_obj, float_obj_X_init_array, float_obj_X_end_array, [target_obj], [target_obj_X], collision_threshold)

#     # Result
#     print('Number of points before interpolation: ', len(path))
#     print('Number of points after interpolation: ', len(path_full))
#     print('Quaternion lower bound:', pose_lower[3:])
#     print('Quaternion upper bound:', pose_upper[3:])
#     # print('Path after interpolation: ', path_full)
#     # print('Position trajectory after interpolation: ', [path[:3] for path in path_full])
#     # print('Quaternion (w,x,y,z) trajectory after interpolation: ', [path[3:] for path in path_full])

#     # Visualize the final scene with interpolation
#     scene_planned = trimesh.scene.scene.Scene()
#     scene_planned.add_geometry(target_obj, transform=target_obj_X)
#     for X in path_full:
#         scene_planned.add_geometry(float_obj, transform=get_transform(X))
#     scene_planned.show()

#     # Convert - mujoco uses (w,x,y,z)
#     translation_tcdm = []
#     orientation_tcdm = []
#     for path in path_full:
#         translation_tcdm += [path[:3]]
#         # q = R.from_euler(seq='XYZ', angles=path[3:], degrees=False).as_quat()
#         # orientation_tcdm += [tuple([q[-1]]+list(q[:3]))]  # (x,y,z,w) -> (w,x,y,z)
#         orientation_tcdm += [path[3:]]

#     # Save trajectory
#     def load_motion(motion_file):
#         motion_file = np.load(motion_file, allow_pickle=True)
#         reference_motion =  {k:v for k, v in motion_file.items()}
#         reference_motion['s_0'] = reference_motion['s_0'][()]
#         return reference_motion
    
#     data = load_motion('./trajectories/banana_pass1.npz')
#     traj = copy.copy(dict(data))
#     # print(traj['s_0']['motion_planned']['position'][-6:])

#     traj['offset'] = np.zeros((3))

#     traj['s_0']['motion_planned']['position'][-6:] = np.hstack((
#         translation_tcdm[0],
#         quat2euler(orientation_tcdm[0]),
#     ))  # local frame

#     # fixed object
#     traj['s_0']['motion_planned']['fixed'] = {}
#     traj['s_0']['motion_planned']['fixed']['position'] = np.array(target_obj_X_array_r_array)

#     traj['object_translation'] = np.vstack((translation_tcdm))
#     traj['object_orientation'] = np.vstack((orientation_tcdm))
#     traj['length'] = len(translation_tcdm)
#     # traj['SIM_SUBSTEPS'] = int(data['SIM_SUBSTEPS']/3)
#     traj['DATA_SUBSTEPS'] = 1
#     # traj['offset'] = offset
#     np.savez(traj_path, **traj)  # save a dict as npz


# if __name__ == "__main__":
    # # Fixed object, e.g., a cup
    # target_obj_path = './tcdm/envs/assets/meshes/objects/cup/cup.stl'

    # # Moving object, e.g., a banana 
    # float_obj_path = './tcdm/envs/assets/meshes/objects/cup/cup.stl'

    # # Trajectory path
    # traj_path = './new_agents/cup/cup_cup_move1/cup_cup_move1.npz'

    # # Generate trajectory
    # motion_plan_one_obj(target_obj_path, float_obj_path, traj_path)
