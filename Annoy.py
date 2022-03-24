import time
from random import randint
import numpy as np
from multiprocessing import Pool


class Annoy:

    def __init__(
        self,
        X,
        leaf_size=40,
        n_trees=10,
        metric='minkowski',
        n_jobs=0,
        **kwargs
    ):
        self.X = np.array(X)
        self.indexes = np.linspace(0, len(self.X)-1, num=len(self.X), dtype=int)
        self.leaf_size = leaf_size
        self.n_trees = n_trees
        self.trees = []
        self.times = [0] * 100
        self.times_count = [0] * 100
        if n_jobs == 1:
            with Pool(processes=10) as pool:
                self.trees = pool.map(self.build_tree, range(self.n_trees))
        else:
            for i in range(self.n_trees):
                self.trees.append(self._build_tree(self.indexes, self.X))
        # print()
        # for i in range(6):
        #     if self.times_count[i] == 0:
        #         continue
        #     print(f'block_{i} takes {self.times[i]/self.times_count[i]}s on average and {self.times[i]}s overall')
        # print()
        # print('trees', self.trees)

    def build_tree(self, tree_idx):
        return self._build_tree(self.indexes, self.X)

    def addtime(self, idx, start): # this is for debugging, will be deleted
        self.times[idx] += time.time()-start
        self.times_count[idx] += 1

    def _build_tree(self, points, X):
        start_time = time.time()
        if len(points) <= self.leaf_size:
            node = Tree(is_leaf=True)
            node.points = points
            node.leaf_size = len(points)
            return node
        self.addtime(1, start_time)
        start_time = time.time()
        p1 = randint(0, len(points)-1)
        p2 = randint(0, len(points)-1)
        while p2 == p1:
            p2 = randint(0, len(points)-1)
        # p1 = points[p1]
        # p2 = points[p2]
        self.addtime(2, start_time)

        start_time = time.time()
        plane_norm = X[p2]-X[p1]
        plane_point = (X[p1]+X[p2])/2
        # plane equation: norm.x=b
        # b = norm.point
        b = np.dot(plane_norm, plane_point)
        self.addtime(3, start_time)

        start_time = time.time()
        left_points = points[np.sum(plane_norm*X, 1) <= b]
        right_points = points[np.sum(plane_norm*X, 1) > b]
        left_X = X[np.sum(plane_norm*X, 1) <= b]
        right_X = X[np.sum(plane_norm*X, 1) > b]
        self.addtime(4, start_time)

        start_time = time.time()
        node = Tree()
        node.leaf_size = len(points)
        node.norm = plane_norm
        node.b = b
        self.addtime(5, start_time)
        node.left = self._build_tree(left_points, left_X)
        node.right = self._build_tree(right_points, right_X)
        return node

    def query(self, X, k=1, return_distance=True):
        points_indices = []
        for tree in self.trees:
            points_indices.extend(self._query_tree(X, tree, k))
            # print('\t', points_indices)
        points_indices = list(set(points_indices))

        distances = [self.distance(X, self.X[p]) for p in points_indices]

        distance_index_pair = list(zip(distances, points_indices))
        distance_index_pair.sort()
        if len(distance_index_pair) > k:
            distance_index_pair = distance_index_pair[:k]

        points_indices_sorted = [i for d, i in distance_index_pair]
        distances_sorted = [d for d, i in distance_index_pair]

        if return_distance:
            return [distances_sorted, points_indices_sorted]
        return points_indices_sorted

    def distance(self, p1, p2):
        return np.sqrt(np.sum((np.array(p1)-np.array(p2))**2))

    def _query_tree(self, point, tree, k=1):
        if tree.is_leaf:
            return tree.points
        if np.dot(tree.norm, point) <= tree.b:
            leaves = self._query_tree(point, tree.left, k)
            if tree.left.leaf_size < k:
                # leaves.extend(self._query_tree(point, tree.right, k-tree.left.leaf_size))
                leaves = np.concatenate((leaves, self._query_tree(point, tree.right, k-tree.left.leaf_size)))
            return leaves
        leaves = self._query_tree(point, tree.right, k)
        if tree.right.leaf_size < k:
            # leaves.extend(self._query_tree(point, tree.left, k-tree.right.leaf_size))
            leaves = np.concatenate((leaves, self._query_tree(point, tree.left, k-tree.right.leaf_size)))
        return leaves


class Tree:
    def __init__(self, is_leaf=False):
        self.is_leaf = is_leaf
        self.leaf_size = 0
        # if self.is_leaf is False
        self.norm = None
        self.b = None
        self.left = None
        self.right = None
        # if self.is_leaf is True
        self.points = None
