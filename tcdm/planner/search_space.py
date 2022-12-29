# This file is subject to the terms and conditions defined in
# file 'LICENSE', which is part of this source code package.

import numpy as np
# from rtree import index

from planner.util.geometry import es_points_along_line, get_transform
# from planner.util.obstacle_generation import obstacle_generator


class SearchSpace(object):
    # def __init__(self, dimension_lengths, O=None):
    def __init__(self, dimension_lengths, collision_checker, collision_threshold=0.005):
        """
        Initialize Search Space
        :param dimension_lengths: range of each dimension
        :param O: list of obstacles
        """
        # sanity check
        if len(dimension_lengths) < 2:
            raise Exception("Must have at least 2 dimensions")
        self.dimensions = len(dimension_lengths)  # number of dimensions
        # sanity checks
        if any(len(i) != 2 for i in dimension_lengths):
            raise Exception("Dimensions can only have a start and end")
        if any(i[0] >= i[1] for i in dimension_lengths):
            raise Exception("Dimension start must be less than dimension end")
        self.dimension_lengths = dimension_lengths  # length of each dimension
        # p = index.Property()
        # p.dimension = self.dimensions
        # if O is None:
        #     self.obs = index.Index(interleaved=True, properties=p)
        # else:
        #     # r-tree representation of obstacles
        #     # sanity check
        #     if any(len(o) / 2 != len(dimension_lengths) for o in O):
        #         raise Exception("Obstacle has incorrect dimension definition")
        #     if any(o[i] >= o[int(i + len(o) / 2)] for o in O for i in range(int(len(o) / 2))):
        #         raise Exception("Obstacle start must be less than obstacle end")
        #     self.obs = index.Index(obstacle_generator(O), interleaved=True, properties=p)
        self.collision_checker = collision_checker
        self.collision_threshold = collision_threshold
        

    def obstacle_free(self, x):
        """
        Check if a location resides inside of an obstacle
        :param x: location to check
        :return: True if not inside an obstacle, False otherwise
        """
        self.collision_checker.set_transform('float', get_transform(x))
        return not self.collision_checker.in_collision_internal()
        # return self.collision_checker.min_distance_internal() > self.collision_threshold
        # return self.obs.count(x) == 0

    def sample_free(self):
        """
        Sample a location within X_free
        :return: random location within X_free
        """
        while True:  # sample until not inside of an obstacle
            x = self.sample()
            if self.obstacle_free(x):
                return x

    def collision_free(self, start, end, r):
        """
        Check if a line segment intersects an obstacle
        :param start: starting point of line
        :param end: ending point of line
        :param r: resolution of points to sample along edge when checking for collisions
        :return: True if line segment does not intersect an obstacle, False otherwise
        """
        points = es_points_along_line(start, end, r)
        coll_free = all(map(self.obstacle_free, points))
        return coll_free

    def sample(self):
        """
        Return a random location within X
        :return: random location within X (not necessarily X_free)
        """
        x = np.random.uniform(self.dimension_lengths[:, 0], self.dimension_lengths[:, 1])
        return tuple(x)
