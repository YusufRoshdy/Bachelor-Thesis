#!python
#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
# distutils: language = c++

from libcpp.vector cimport vector
from libcpp.stack cimport stack
from libcpp.map cimport map
from libcpp.pair cimport pair

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

ctypedef (int, int, int, int, double, vector[double]) NODE
# start, end, left, right, b, norm


import time
from random import randint
import numpy as np
from multiprocessing import Pool


cdef class CItrAnnoy:
    cdef readonly np.ndarray X
    cdef readonly ITYPE_t leaf_size, n_trees
    cdef vector[map[int, NODE]] trees
    cdef vector[double] times
    cdef vector[int] times_count
    cdef vector[vector[double]] X_vec
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
        self.times = vector[double](10)
        self.times_count = vector[int](10)
        # self.times = [0] * 100
        # self.times_count = [0] * 100
        self.X_vec = self.ndarray_to_vector_2(self.X)
        for i in range(self.n_trees):
            # indexes = np.linspace(0, len(self.X) - 1, num=len(self.X), dtype=int)
            # indexes = np.arange(len(self.X))
            start_time = time.time()
            self.trees.push_back(self._build_tree())
            self.times[5] += time.time() - start_time
            self.times_count[5] += 1

        # print()
        # for i in range(10):
        #     if self.times_count[i] == 0:
        #         continue
        #     print(
        #         f'block_{i} takes {self.times[i] / self.times_count[i]}s on average and {self.times[i]}s overall, it was executed {self.times_count[i]} times')
        # print()
        # print()
        # for i in range(10):
        #     if self.times_count[i] == 0:
        #         continue
        #     print(f'block_{i} takes {self.times[i]/self.times_count[i]}s on average and {self.times[i]}s overall, it was executed {self.times_count[i]} times')
        # print()

    # def addtime(self, idx, start):  # this is for debugging, will be deleted
    #     self.times[idx] += time.time()-start
    #     self.times_count[idx] += 1

    def _build_tree(self):
        start_time = time.time()
        cdef:
            int size = self.X_vec.size(), new_node_idx = 0
            int i, j, start, end, length, s, e, p1, p2
            double temp_sum, b
            vector[int] indexes = vector[int](size), ret = vector[int](size)
            map[int, NODE] tree
            stack[int] nodes_to_build
            vector[double] plane_norm, plane_point, X_vec_ret_i
            NODE root, left, right, node
        self.times[6] += time.time() - start_time
        self.times_count[6] += 1
        start_time = time.time()

        for i in range(size):
            indexes[i] = i
        ret = indexes

        cdef vector[double] new_vec
        # start, end, left, right, b, norm
        root = (0, size-1, -1, -1, 0.0, new_vec)
        tree[new_node_idx] = root
        nodes_to_build.push(new_node_idx)
        new_node_idx += 1
        self.times[7] += time.time() - start_time
        self.times_count[7] += 1
        while not nodes_to_build.empty():
            start_time = time.time()
            node = tree[nodes_to_build.top()]
            nodes_to_build.pop()
            start = node[0]
            end = node[1]
            length = end-start+1
            self.times[0] += time.time()-start_time
            self.times_count[0] += 1

            if length <= self.leaf_size:
                continue


            start_time = time.time()
            p1 = rand() % length
            p2 = rand() % length
            if p2 == p1:
                p2 = (p2 + 1) % length

            plane_norm = self.vector_diff(self.X_vec[p2], self.X_vec[p1])
            plane_point = self.vectors_midpoint(self.X_vec[p1], self.X_vec[p2])

            b = 0
            for i in range(plane_norm.size()):
                b += plane_norm[i] * plane_point[i]

            node[4] = b
            node[5] = plane_norm
            self.times[1] += time.time()-start_time
            self.times_count[1] += 1

            start_time = time.time()
            s = start
            e = end
            for i in range(start, end+1):
                temp_sum = 0
                X_vec_ret_i = self.X_vec[ret[i]]
                for j in range(X_vec_ret_i.size()):
                    temp_sum += plane_norm[j] * X_vec_ret_i[j]

                if temp_sum <= b:
                    ret[s] , ret[i] = ret[i] , ret[s]
                    s += 1
                else:
                    ret[e], ret[i] = ret[i], ret[e]
                    e -= 1
            self.times[2] += time.time() - start_time
            self.times_count[2] += 1

            start_time = time.time()
            left = (start, e, -1, -1, 0.0, new_vec)
            right = (s, end, -1, -1, 0.0, new_vec)

            tree[new_node_idx] = right
            nodes_to_build.push(new_node_idx)
            plane_norm[2] = new_node_idx
            new_node_idx += 1

            tree[new_node_idx] = left
            nodes_to_build.push(new_node_idx)
            plane_norm[2] = new_node_idx
            new_node_idx += 1

            self.times[3] += time.time()-start_time
            self.times_count[3] += 1

        # indexes = ret
        return tree

    cdef vector[double] ndarray_to_vector_1(self, np.ndarray array):
        cdef vector[double] tmp_result
        cdef ITYPE_t i
        for i in range(array.shape[0]):
            tmp_result.push_back(array[i])
        return tmp_result

    cdef vector[vector[double]] ndarray_to_vector_2(self, np.ndarray array):
        cdef vector[vector[double]] tmp_result
        cdef ITYPE_t i
        for i in range(array.shape[0]):
            tmp_result.push_back(array[i])
        return tmp_result

    cdef vector[double] vectors_midpoint(self, vector[double] v1, vector[double] v2):
        cdef vector[double] tmp_result
        cdef ITYPE_t i
        for i in range(v1.size()):
            tmp_result.push_back((v1[i] + v2[i]) / 2)
        return tmp_result

    cdef vector[double] vector_diff(self, vector[double] v1, vector[double] v2):
        cdef vector[double] tmp_result
        cdef ITYPE_t i
        for i in range(v1.size()):
            tmp_result.push_back(v1[i] - v2[i])
        return tmp_result


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
