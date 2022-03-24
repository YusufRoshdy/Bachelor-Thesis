import time
from random import randint
import numpy as np
from multiprocessing import Pool


class ItrAnnoy:

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
        self.leaf_size = leaf_size
        self.n_trees = n_trees
        self.trees = []
        self.times = [0] * 100
        self.times_count = [0] * 100

        for i in range(self.n_trees):
            # indexes = np.linspace(0, len(self.X) - 1, num=len(self.X), dtype=int)
            indexes = np.arange(len(self.X))
            self.trees.append(
                (indexes,
                 self._build_tree(indexes, self.X, 0, indexes.shape[0]-1)
                 )
            )
        # print()
        # for i in range(10):
        #     if self.times_count[i] == 0:
        #         continue
        #     print(f'block_{i} takes {self.times[i]/self.times_count[i]}s on average and {self.times[i]}s overall, it was executed {self.times_count[i]} times')
        # print()
        # print('trees', self.trees)

    # def build_tree(self, tree_idx):
        # return self._build_tree(self.indexes, self.X)

    def addtime(self, idx, start):  # this is for debugging, will be deleted
        self.times[idx] += time.time()-start
        self.times_count[idx] += 1

    def _build_tree(self, indices, X, start_idx, end_idx):
        start_time = time.time()
        if end_idx-start_idx+1 <= self.leaf_size:
            node = Tree(is_leaf=True)
            node.points = indices[start_idx: end_idx+1]
            node.leaf_size = len(indices[start_idx: end_idx+1])
            return node
        self.addtime(1, start_time)

        start_time = time.time()
        temp_indices = indices[start_idx: end_idx+1]
        length = len(temp_indices)
        # temp_X = X[temp_indices]

        p1 = randint(0, length - 1)
        p2 = randint(0, length - 1)
        while p2 == p1:
            p2 = randint(0, length - 1)
        p1 = X[p1]
        p2 = X[p2]
        self.addtime(2, start_time)

        start_time = time.time()
        plane_norm = p2 - p1
        plane_point = (p1 + p2) / 2
        b = np.dot(plane_norm, plane_point)
        self.addtime(3, start_time)

        start_time = time.time()
        # dor_product = np.sum(plane_norm * temp_X, 1)
        dor_product_less_than_b = np.sum(plane_norm * X[temp_indices], 1) <= b
        left_points = temp_indices[dor_product_less_than_b]
        # right_points = temp_indices[~dor_product_less_than_b]
        # temp_indices = np.concatenate(
        #     (
        #         left_points,
        #         temp_indices[~dor_product_less_than_b]
        #     ),
        #     axis=None
        # )

        # temp_indices = temp_indices[np.apply_along_axis(lambda x: (np.dot(plane_norm, x)), 1, temp_X).argsort()]

        # indices[start_idx: end_idx + 1] = temp_indices
        indices[start_idx: end_idx + 1] = np.concatenate(
            (
                left_points,
                temp_indices[~dor_product_less_than_b]
            ),
            axis=None
        )
        self.addtime(4, start_time)

        start_time = time.time()
        node = Tree()
        node.leaf_size = length
        node.norm = plane_norm
        node.b = b
        node.start_idx = start_idx
        node.end_idx = end_idx
        self.addtime(5, start_time)
        node.left = self._build_tree(indices, X, start_idx=start_idx, end_idx=start_idx+len(left_points)-1)
        node.right = self._build_tree(indices, X, start_idx=start_idx+len(left_points), end_idx=end_idx)
        return node

    def query(self, X, k=1, return_distance=True):
        points_indices = []
        for tree in self.trees:
            points_indices.extend(self._query_tree(X, tree[1], k))
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
        self.start_idx = None
        self.end_idx = None
        # if self.is_leaf is True
        self.points = None
