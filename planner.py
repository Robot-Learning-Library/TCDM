import trimesh
import numpy as np
# from planner.rrt_simple import get_transform, edge_free, random_conf, nearest_vertex, extend, backtrack, rrt
from planner.util.geometry import get_transform
from planner.rrt_star_bid import RRTStarBidirectional
from planner.search_space import SearchSpace


# e.g., a cup
target_obj_path = '/home/allen/TCDM/tcdm/envs/assets/meshes/objects/cup/cup.stl'

# e.g., a banana 
float_obj_path = '/home/allen/TCDM/tcdm/envs/assets/meshes/objects/banana/banana.stl'

# Load meshes
target_obj = trimesh.load(target_obj_path)
# print(target_obj.bounds)
# target_obj.show()
float_obj = trimesh.load(float_obj_path)
# print(float_obj.bounds)
# float_obj.show()

# Initial pose of the target and float objects - x,y,z,r,p,y - Euler extrinsic XYZ convention
float_obj_R_init_array = [0.3, 0.0, 0.0, 0.0, 0.0, 0.0]
target_obj_R_array = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
float_obj_R_init = get_transform(float_obj_R_init_array)
target_obj_R = get_transform(target_obj_R_array)

# Final pose of the float object - needs to manually specify right now - assume collision free with the target
float_obj_R_end_array = [0.0, -0.01, 0.10, 0.0, -1.57, 0.0]
# float_obj_R_end_array = (0.05860377713192581, -0.006156166736869985, 0.10497456574689194, 0.010936462142528264, -0.7352193985458293, -0.007242288185940012)
float_obj_R_end = get_transform(float_obj_R_end_array)

# Visualize the final scene
scene_end = trimesh.scene.scene.Scene()
scene_end.add_geometry(target_obj, transform=target_obj_R)
scene_end.add_geometry(float_obj, transform=float_obj_R_end)
scene_end.show()

# Collision checking
collision_checker = trimesh.collision.CollisionManager()
collision_checker.add_object('target', target_obj, target_obj_R)
collision_checker.add_object('float', float_obj, float_obj_R_end)
# import time
# s1 = time.time()
# print(collision_checker.in_collision_internal())    # 0.0002-0.0006s
# print(collision_checker.min_distance_internal())    # 0.0002-0.0006s
# print('Time for collision checking: ', time.time()-s1)

# parameters
pose_lower = np.array([-0.1, -0.1, 0., -1, -2, -1])
pose_upper = np.array([0.4, 0.1, 0.2, 1, 2, 1])

# Run RRT
X_dimensions = np.vstack((pose_lower, pose_upper)).T
x_init = tuple(float_obj_R_init_array)
x_goal = tuple(float_obj_R_end_array)

Q = np.array([(0.01, 4)])  # length of tree edges
r = 0.01  # length of smallest edge to check for intersection with obstacles
max_samples = 10240  # max number of samples to take before timing out
rewire_count = 32  # optional, number of nearby branches to rewire
prc = 0.1  # probability of checking for a connection to goal

# create Search Space
X = SearchSpace(X_dimensions, collision_checker)
rrt = RRTStarBidirectional(X, Q, x_init, x_goal, max_samples, r, prc, rewire_count)
path = rrt.rrt_star_bidirectional()
print(path)

# Visualize the final scene
scene_planned = trimesh.scene.scene.Scene()
scene_planned.add_geometry(target_obj, transform=target_obj_R)
for p in path:
    scene_planned.add_geometry(float_obj, transform=get_transform(p))
scene_planned.show()

#
# vertices, parents = rrt(origin=float_obj_R_init_array, 
#                         collision_checker=collision_checker,
#                         pose_lower=pose_lower,
#                         pose_upper=pose_upper,
#                         trials=100000,
#                         pos_step_size=0.01,
#                         euler_step_size=0.10,
#                         )
# # vertices, parents = rrt(origin, width, height, obstacles)
# index = nearest_vertex(float_obj_R_end_array, vertices)
# euler_distance_weight = 0.1
# pos_distance = np.linalg.norm(vertices[index, :3] - float_obj_R_end_array[:3])
# euler_distance = np.linalg.norm(vertices[index, 3:] - float_obj_R_end_array[3:])
# weighted_distance = pos_distance + euler_distance_weight * euler_distance
# print('Nearest: ', np.array2string(vertices[index, :], separator=', '))
# print('Target: ', float_obj_R_end_array)
# print('Pos distance: ', pos_distance)
# print('Euler distance: ', euler_distance)
# print('Weighted distance: ', weighted_distance)

# # Check condition
# if weighted_distance < 0.05: # needs weighting
#     print('Path found!')
#     path_verts = backtrack(index, parents)
#     for ind, i in enumerate(path_verts):
#         if parents[i] < 0:
#             continue
#         if ind % 5 == 0:
#             print(f"Index {ind}, {np.array2string(vertices[i], separator=', ')}")
#         # plt.plot([vertices[i, 0], vertices[parents[i], 0]], 
#         #         [vertices[i, 1], vertices[parents[i], 1]], c='r')   
#     print('Number of vertices in the path: ', len(path_verts)) 
# else:
#     print('No path found!')
#     path_verts = []
