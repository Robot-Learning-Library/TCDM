import trimesh
import copy
import numpy as np
from planner.util.geometry import get_transform
from planner.rrt_star_bid import RRTStarBidirectional
from planner.search_space import SearchSpace

from tcdm.util.geom import quat2euler, euler2quat

import scipy.interpolate as interpolate
from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as R


# TODO: use cfg

# naming convention：
# X: 6-dim tuple with 3D position and 3D Euler angle (XYZ extrinsic)
# p: 3-dim position
# r: 3-dim Euler angle (XYZ intrinsic convention, 'rxyz' in trimesh, 'XYZ' in scipy)
# q: 4-dim quaternion in (x,y,z,w) - scipy convention
#! mujoco/tcdm uses (w,x,y,z) convention
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


##########################################################################
#######################   Mesh and key poses   ############################
##########################################################################

def motion_plan(target_obj_path, float_obj_path, traj_path, z_offset=0.2, collision_threshold=0.003, q_bound_buffer=0.2):
    # Load meshes
    target_obj = trimesh.load(target_obj_path)
    float_obj = trimesh.load(float_obj_path)
    # print(target_obj.bounds)
    # target_obj.show()


    # Initial pose of the target and float objects - x,y,z,r,p,y - Euler extrinsic XYZ convention
    float_obj_X_init_r_array = [0.01895152, -0.01185687, -0.17970488, -3.05750797, 0.08599904, -1.99028331]  # from banana_pass1.npz
    float_obj_X_init_array = float_obj_X_init_r_array[:3] + euler2quat(float_obj_X_init_r_array[3:]).tolist()   # convert to quat (w,x,y,z)
    # float_obj_X_init_array = [0.05, 0.0, 0-0.2+0.023, 0.0, 0.0, 0.0]  # account for 20cm offset in z when loading objects in tcdm
    # target_obj_X_array = [-0.02, -0.165, 0-z_offset+0.04, 0.0, 0.0, 0.0]
    target_obj_X_array_r_array = [-0.2, -0.165, 0-z_offset+0.04, 0.0, 0.0, 0.0]  # avoid collision with the cup
    target_obj_X_array = target_obj_X_array_r_array[:3] + euler2quat(target_obj_X_array_r_array[3:]).tolist()
    float_obj_X_init = get_transform(float_obj_X_init_array)
    target_obj_X = get_transform(target_obj_X_array)

    # Final pose of the float object
    float_obj_X_end_r_array = target_obj_X_array_r_array.copy()
    float_obj_X_end_r_array[1] -= 0.01 
    float_obj_X_end_r_array[2] += 0.149
    float_obj_X_end_r_array[-2] = -1.57
    float_obj_X_end_array = float_obj_X_end_r_array[:3] + euler2quat(float_obj_X_end_r_array[3:]).tolist()
    float_obj_X_end = get_transform(float_obj_X_end_array)

    # Configuration space boundaries - use quaternions of initial and end poses, plus some buffer
    # pose_lower = np.array([-0.1, -0.2, -0.2, -3, -3, -3])
    # pose_upper = np.array([ 0.1,  0.2,  0.2,  3,  3,  3])
    pose_lower = np.array([-0.1, -0.2, -0.2,  -1, -1, -1, -1])
    pose_upper = np.array([ 0.1,  0.2,  0.2,  1,  1,  1,  1])
    q_min = np.minimum(float_obj_X_init_array[3:], float_obj_X_end_array[3:]) - q_bound_buffer
    q_max = np.maximum(float_obj_X_init_array[3:], float_obj_X_end_array[3:]) + q_bound_buffer
    pose_lower[3:] = q_min
    pose_upper[3:] = q_max

    # Visualize the final scene
    scene_end = trimesh.scene.scene.Scene()
    scene_end.add_geometry(target_obj, transform=target_obj_X)
    scene_end.add_geometry(float_obj, transform=float_obj_X_end)
    scene_end.add_geometry(float_obj, transform=float_obj_X_init)
    scene_end.show()

    # Collision checking
    collision_checker = trimesh.collision.CollisionManager()
    collision_checker.add_object('target', target_obj, target_obj_X)
    collision_checker.add_object('float', float_obj, float_obj_X_end)
    # import time
    # s1 = time.time()
    print(collision_checker.in_collision_internal())    # 0.0002-0.0006s
    print(collision_checker.min_distance_internal())    # 0.0002-0.0006s
    # print('Time for collision checking: ', time.time()-s1)

    ##########################################################################
    #######################    Motion Planning    ############################
    ##########################################################################

    # RRT parameters
    Q = np.array([(0.01, 4)])  # length of tree edges 
    r = 0.01  # length of smallest edge to check for intersection with obstacles
    max_samples = 10240  # max number of samples to take before timing out
    rewire_count = 32  # optional, number of nearby branches to rewire
    prc = 0.1  # probability of checking for a connection to goal

    # Run RRT
    X_dimensions = np.vstack((pose_lower, pose_upper)).T
    rrt = RRTStarBidirectional(X=SearchSpace(X_dimensions, collision_checker, collision_threshold), 
                            Q=Q,
                            x_init=tuple(float_obj_X_init_array),
                            x_goal=tuple(float_obj_X_end_array),
                            max_samples=max_samples, 
                            r=r, 
                            prc=prc, 
                            rewire_count=rewire_count,
                            )
    path = rrt.rrt_star_bidirectional()
    print('Path before interpolation: ', path)

    # Visualize the final scene
    scene_planned = trimesh.scene.scene.Scene()
    scene_planned.add_geometry(target_obj, transform=target_obj_X)
    for X in path:
        scene_planned.add_geometry(float_obj, transform=get_transform(X))
    scene_planned.show()

    # Interpolate
    p_interp_threshold = 0.015
    q_interp_threshold = 0.1  # not using
    p_interp_dist = 0.01
    q_interp_dist = 0.05

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

    # Result
    print('Number of points before interpolation: ', len(path))
    print('Number of points after interpolation: ', len(path_full))
    print('Quaternion lower bound:', pose_lower[3:])
    print('Quaternion upper bound:', pose_upper[3:])
    # print('Path after interpolation: ', path_full)
    # print('Position trajectory after interpolation: ', [path[:3] for path in path_full])
    # print('Quaternion (w,x,y,z) trajectory after interpolation: ', [path[3:] for path in path_full])

    # Visualize the final scene with interpolation
    scene_planned = trimesh.scene.scene.Scene()
    scene_planned.add_geometry(target_obj, transform=target_obj_X)
    for X in path_full:
        scene_planned.add_geometry(float_obj, transform=get_transform(X))
    scene_planned.show()

    # Convert - mujoco uses (w,x,y,z)
    translation_tcdm = []
    orientation_tcdm = []
    for path in path_full:
        translation_tcdm += [path[:3]]
        # q = R.from_euler(seq='XYZ', angles=path[3:], degrees=False).as_quat()
        # orientation_tcdm += [tuple([q[-1]]+list(q[:3]))]  # (x,y,z,w) -> (w,x,y,z)
        orientation_tcdm += [path[3:]]

    # Save trajectory
    def load_motion(motion_file):
        motion_file = np.load(motion_file, allow_pickle=True)
        reference_motion =  {k:v for k, v in motion_file.items()}
        reference_motion['s_0'] = reference_motion['s_0'][()]
        return reference_motion
    
    data = load_motion('./trajectories/banana_pass1.npz')
    traj = copy.copy(dict(data))
    # print(traj['s_0']['motion_planned']['position'][-6:])

    traj['offset'] = np.zeros((3))

    traj['s_0']['motion_planned']['position'][-6:] = np.hstack((
        translation_tcdm[0],
        quat2euler(orientation_tcdm[0]),
    ))  # local frame

    # fixed object
    traj['s_0']['motion_planned']['fixed'] = {}
    traj['s_0']['motion_planned']['fixed']['position'] = np.array(target_obj_X_array_r_array)

    traj['object_translation'] = np.vstack((translation_tcdm)) + np.array([0,0,z_offset])
    traj['object_orientation'] = np.vstack((orientation_tcdm))
    traj['length'] = len(translation_tcdm)
    traj['SIM_SUBSTEPS'] = int(data['SIM_SUBSTEPS']/3)
    traj['DATA_SUBSTEPS'] = 1
    # traj['offset'] = offset
    np.savez(traj_path, **traj)  # save a dict as npz

# Fixed object, e.g., a cup
target_obj_path = './tcdm/envs/assets/meshes/objects/cup/cup.stl'

# Moving object, e.g., a banana 
float_obj_path = './tcdm/envs/assets/meshes/objects/banana/banana.stl'

# Trajectory path
traj_path = './trajectories/multi_trajs/banana_cup_pass1/banana_cup_pass1.npz'

motion_plan(target_obj_path, float_obj_path, traj_path)