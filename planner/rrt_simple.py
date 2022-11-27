import numpy as np
import trimesh


def get_transform(pos_euler):
    pos = pos_euler[:3]
    euler = pos_euler[3:]
    transform = trimesh.transformations.euler_matrix(euler[0], euler[1], euler[2], 'sxyz') # s for static, returns 4x4
    transform[:3, -1] = pos
    return transform


def edge_free(edge, collision_checker):
    """
    Check if a graph edge is in the free space.
    
    This function checks if a graph edge, i.e. a line segment specified as two end points, lies entirely outside of
    every obstacle in the configuration space.
    
    @param edge: A tuple containing the two segment endpoints.
    @param collision_checker: From trimesh.
    @return: True if the edge is in the free space, i.e. it lies entirely outside of all the circles in `obstacles`. 
             Otherwise return False.
    """
    collision_checker.set_transform('float', get_transform(edge[0]))
    free_1 = not collision_checker.in_collision_internal()

    collision_checker.set_transform('float', get_transform(edge[1]))
    free_2 = not collision_checker.in_collision_internal()

    collision_checker.set_transform('float', get_transform((edge[0] + edge[1])/2))
    free_3 = not collision_checker.in_collision_internal()
    # print(free_1, free_2, free_3, free_1 and free_2 and free_3)
    
    return free_1 and free_2 and free_3
    # return True

    # free = True
    # for o in obstacles:
    #     dist = np.abs((edge[1][1] - edge[0][1]) * o[0][0] - (edge[1][0] - edge[0][0]) * o[0][1]  + edge[1][0] * edge[0][1] - edge[1][1] * edge[0][0]) / np.linalg.norm(edge[1] - edge[0])
    #     free = free and dist > o[1]
    # return free


def random_conf(lower, upper):
    """
    Sample a random configuration from the configuration space.
    
    This function draws a uniformly random configuration from the configuration space rectangle. The configuration 
    does not necessarily have to reside in the free space.
    
    @param width: The configuration space width.
    @param height: The configuration space height.
    @return: A random configuration uniformily distributed across the configuration space.
    """
    return np.random.uniform(0, 1, size=len(lower)) * (upper - lower) + lower


def nearest_vertex(conf, vertices) :
    """
    Finds the nearest vertex to conf in the set of vertices.
    
    This function searches through the set of vertices and finds the one that is closest to 
    conf using the L2 norm (Euclidean distance).
    
    @param conf: The configuration we are trying to find the closest vertex to.
    @param vertices: The set of vertices represented as an np.array with shape (n, 2). Each row represents
                     a vertex.
    @return: The index (i.e. row of `vertices`) of the vertex that is closest to `conf`.
    """
    return np.argmin(np.linalg.norm(vertices - conf, axis=1))


def extend(origin, target, pos_step_size=0.2, euler_step_size=0.2):
    """
    Extends the RRT at most a fixed distance toward the target configuration.
    
    Given a configuration in the RRT graph `origin`, this function returns a new configuration that takes a
    step of at most `step_size` towards the `target` configuration. That is, if the L2 distance between `origin`
    and `target` is less than `step_size`, return `target`. Otherwise, return the configuration on the line
    segment between `origin` and `target` that is `step_size` distance away from `origin`.
    
    @param origin: A vertex in the RRT graph to be extended.
    @param target: The vertex that is being extended towards.
    @param step_size: The maximum allowed distance the returned vertex can be from `origin`.
    
    @return: A new configuration that is as close to `target` as possible without being more than
            `step_size` away from `origin`.
    """
    direction_pos = target[:3] - origin[:3]
    direction_euler = target[3:] - origin[3:]
    
    if np.linalg.norm(direction_pos) < pos_step_size or np.linalg.norm(direction_euler) < euler_step_size:
        return target
    else:
        direction_pos = direction_pos / np.linalg.norm(direction_pos)
        direction_euler = direction_euler / np.linalg.norm(direction_euler)
        out = origin
        out[:3] += pos_step_size * direction_pos
        out[3:] += euler_step_size * direction_euler
        return out


def backtrack(index, parents):
    """
    Find the sequence of nodes from the origin of the graph to an index.
    
    This function returns a List of vertex indices going from the origin vertex to the vertex `index`.
    
    @param index: The vertex to find the path through the tree to.
    @param parents: The array of vertex parents as specified in the `rrt` function.
    
    @return: The list of vertex indicies such that specifies a path through the graph to `index`.
    """
    
    i = index
    vert_idx = []
    
    while i >= 0:
        vert_idx.insert(0, i)
        i = parents[i]

    return vert_idx


def rrt(origin, collision_checker, pose_lower, pose_upper, trials=1000, pos_step_size=0.2, euler_step_size=0.2):
    """
    Explore a workspace using the RRT algorithm.
    
    This function builds an RRT using `trials` samples from the free space.
    
    @param origin: The starting configuration of the robot.
    @param width: The width of the configuration space.
    @param height: The height of the configuration space.
    @param obstacles: A list of circular obstacles.
    @param trials: The number of configurations to sample from the free space.
    @param step_size: The step_size to pass to `extend`.
    
    @return: A tuple (`vertices`, `parents`), where `vertices` is an (n, 2) `np.ndarray` where each row is a configuration vertex
             and `parents` is an array identifying the parent, i.e. `parents[i]` is the parent of the vertex in
             the `i`th row of `vertices.
    """
    num_verts = 1
    
    vertices = np.zeros((trials + 1, len(origin)))
    vertices[0, :] = origin
    
    parents = np.zeros(trials + 1, dtype=int)
    parents[0] = -1

    for trial in range(trials):
        if trial % 100 == 0:
            print(f'Running {trial} out of {trials}', end='\r')
        rand_conf = random_conf(pose_lower, pose_upper)
        nearest_conf_idx = nearest_vertex(rand_conf, vertices[:num_verts, :])
        new_conf = extend(vertices[nearest_conf_idx, :], rand_conf, pos_step_size, euler_step_size)
        
        flag_edge_free = edge_free((vertices[nearest_conf_idx, :], new_conf), collision_checker)
        if flag_edge_free:
            parents[num_verts] = nearest_conf_idx
            vertices[num_verts, :] = new_conf
            num_verts += 1
    print('Edge free percentage: ', num_verts / trials)    
    return vertices[:num_verts, :], parents[:num_verts]
