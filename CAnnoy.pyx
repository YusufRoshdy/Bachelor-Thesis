#!python
#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
# distutils: language = c++

from libcpp.vector cimport vector
from libcpp cimport bool
from libc.stdlib cimport rand
from libc.math cimport sqrt, pow
import numpy as np
cimport numpy as np

np.import_array()

DTYPE = np.float64
ITYPE = np.int64
ctypedef np.int64_t ITYPE_t
ctypedef np.float64_t DTYPE_t


cdef class Annoy:
    cdef readonly np.ndarray X
    cdef readonly ITYPE_t leaf_size, n_trees
    cdef list trees
    def __init__(
            self,
            X,
            int leaf_size=40,
            int n_trees=10,
            # metric='minkowski',
            # n_jobs=0,
            # **kwargs
    ):
        self.X = np.array(X, dtype=DTYPE)
        indexes = np.linspace(0, len(self.X) - 1, num=len(self.X), dtype=ITYPE)
        self.leaf_size = leaf_size
        self.n_trees = n_trees
        self.trees = []
        cdef int i = 0
        cdef vector[vector[DTYPE_t]] X_vec = self.ndarray_to_vector_2(self.X)
        cdef vector[ITYPE_t] points = self.ndarray_to_vector_1(indexes)

        for i in range(n_trees):
            self.trees.append(self._build_tree(points, X_vec))

#     def build_tree(self, tree_idx):
#         return self._build_tree(self.indexes, self.X)
#
    cdef vector[ITYPE_t] ndarray_to_vector_1(self, np.ndarray array):
        cdef vector[ITYPE_t] tmp_result
        cdef ITYPE_t i
        for i in range(array.shape[0]):
            tmp_result.push_back(array[i])
        return tmp_result

    cdef vector[vector[DTYPE_t]] ndarray_to_vector_2(self, np.ndarray array):
        cdef vector[vector[DTYPE_t]] tmp_result
        cdef ITYPE_t i
        for i in range(array.shape[0]):
            tmp_result.push_back(array[i])
        return tmp_result

    cdef vector[DTYPE_t] vectors_midpoint(self, vector[DTYPE_t] v1, vector[DTYPE_t] v2):
        cdef vector[DTYPE_t] tmp_result
        cdef ITYPE_t i
        for i in range(v1.size()):
            tmp_result.push_back((v1[i] + v2[i])/2)
        return tmp_result

    cdef vector[DTYPE_t] vector_diff(self, vector[DTYPE_t] v1, vector[DTYPE_t] v2):
        cdef vector[DTYPE_t] tmp_result
        cdef ITYPE_t i
        for i in range(v1.size()):
            tmp_result.push_back(v1[i] - v2[i])
        return tmp_result


    def _build_tree(self, vector[ITYPE_t] points, vector[vector[DTYPE_t]] X):
        if points.size() <= self.leaf_size:
            node = Tree(is_leaf=True)
            node.points = points
            node.leaf_size = points.size()
            return node

        # cdef int p1 = randint(0, points.size() - 1)
        # cdef int p2 = randint(0, points.size() - 1)
        cdef int p1 = rand() % points.size()
        cdef int p2 = rand() % points.size()
        if p2 == p1:
            p2 = (p2+1) % points.size()
        # p1 = points[p1]
        # p2 = points[p2]
        cdef vector[DTYPE_t] plane_norm = self.vector_diff(X[p2], X[p1])
        cdef vector[DTYPE_t] plane_point = self.vectors_midpoint(X[p1], X[p2])
        # plane equation: norm.x=b
        # b = norm.point
        cdef int i, j

        cdef DTYPE_t b = 0
        for i in range(plane_norm.size()):
            b += plane_norm[i] * plane_point[i]

        # cdef vector[DTYPE_t] norm_x_sum
        cdef vector[vector[DTYPE_t]] left_X, right_X
        left_X.reserve(X.size())
        right_X.reserve(X.size())
        cdef vector[ITYPE_t] left_points, right_points
        cdef DTYPE_t temp_sum
        for i in range(X.size()):
            temp_sum = 0
            for j in range(X[i].size()):
                temp_sum += plane_norm[j] * X[i][j]
            # norm_x_sum.push_back(temp_sum)
            if temp_sum <= b:
                left_points.push_back(points[i])
                left_X.push_back(X[i])
            else:
                right_points.push_back(points[i])
                right_X.push_back(X[i])

        # cdef np.ndarray[DTYPE_t, ndim=1] norm_x_sum = np.sum(plane_norm * X, 1)
        # cdef np.ndarray[bool, ndim=1] norm_less_eq_b = norm_x_sum <= b
        # cdef np.ndarray[bool, ndim=1] norm_larger_b = norm_x_sum > b
        # cdef np.ndarray[ITYPE_t, ndim=1] left_points = points[norm_less_eq_b]
        # cdef np.ndarray[ITYPE_t, ndim=1] right_points = points[norm_larger_b]
        # cdef np.ndarray[DTYPE_t, ndim=2] left_X = X[norm_less_eq_b]
        # cdef np.ndarray[DTYPE_t, ndim=2] right_X = X[norm_larger_b]

        node = Tree()
        node.leaf_size = points.size()
        node.norm = plane_norm
        nodeb = b
        node.left = self._build_tree(left_points, left_X)
        node.right = self._build_tree(right_points, right_X)
        return node

    def query(self, X, int k=1, return_distance=True):
        X = np.array(X, dtype=DTYPE)
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

    def distance(self, np.ndarray[DTYPE_t, ndim=1]  p1, np.ndarray[DTYPE_t, ndim=1] p2):
        cdef double d = 0
        cdef int i = 0
        for i in range(p1.shape[0]):
            d += pow(p1[i]-p2[i], 2)
        return sqrt(d)

    def _query_tree(self, point, tree, k=1):
        if tree.is_leaf:
            return tree.points
        if np.dot(tree.norm, point) <= tree.b:
            leaves = self._query_tree(point, tree.left, k)
            if tree.left.leaf_size < k:
                # leaves.extend(self._query_tree(point, tree.right, k-tree.left.leaf_size))
                leaves = np.concatenate((leaves, self._query_tree(point, tree.right, k - tree.left.leaf_size)))
            return leaves
        leaves = self._query_tree(point, tree.right, k)
        if tree.right.leaf_size < k:
            # leaves.extend(self._query_tree(point, tree.left, k-tree.right.leaf_size))
            leaves = np.concatenate((leaves, self._query_tree(point, tree.left, k - tree.right.leaf_size)))
        return leaves


# class Tree:
#     def __init__(self, is_leaf=False):
#         self.is_leaf = is_leaf
#         self.leaf_size = 0
#         # if self.is_leaf is False
#         self.norm = None
#         self.b = None
#         self.left = None
#         self.right = None
#         # if self.is_leaf is True
#         self.points = None

cdef class Tree:
    cdef public vector[DTYPE_t] norm
    cdef public DTYPE_t b
    cdef public vector[ITYPE_t] points
    cdef public Tree left, right
    cdef public int leaf_size
    cdef public int is_leaf
    def __init__(self, is_leaf=False):
        self.is_leaf = is_leaf
    #     self.leaf_size = 0
    #     # if self.is_leaf is False
    #     self.norm = None
    #     self.b = None
    #     self.left = None
    #     self.right = None
    #     # if self.is_leaf is True
    #     self.points = None
