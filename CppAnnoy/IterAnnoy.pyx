#!python
#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: profile=True
#cython: linetrace=True
# distutils: language = c++
import numpy as np
from IterAnnoy cimport Annoy
from libcpp.vector cimport vector

cdef class PyAnnoy:
    cdef Annoy *c_annoy

    def __cinit__(self, X, leaf_size=40, n_trees=10, n_jobs=1):
        X = np.array(X)
        self.c_annoy = new Annoy(X, leaf_size, n_trees, n_jobs)

    def __dealloc__(self):
        del self.c_annoy

    def build_tree(self):
        self.c_annoy.buildTree()
        return

    def query_tree(self, vector[double] X, int k=1, return_distance=False):
        return
        cdef vector[double]  point = X
        cdef vector[double]  distances
        cdef vector[int]  points_indices

        self.c_annoy.queryTreeRecursive(point, k, distances, points_indices)

        if return_distance:
            return [points_indices, distances]
        return points_indices

    def recursive_query(self, vector[double] X, int k=1, return_distance=False):
        cdef vector[double]  point = X
        cdef vector[double]  distances
        cdef vector[int]  points_indices

        self.c_annoy.queryTreeRecursive(point, k, distances, points_indices)

        if return_distance:
            return [points_indices, distances]
        return points_indices

    def iterative_query(self, vector[double] X, int k=1, return_distance=False):
        cdef vector[double]  point = X
        cdef vector[double]  distances
        cdef vector[int]  points_indices

        self.c_annoy.queryTreeIterative(point, k, distances, points_indices)

        if return_distance:
            return [points_indices, distances]
        return points_indices

    def forest_query(self, vector[double] X, int k=1, return_distance=False):
        cdef vector[double]  point = X
        cdef vector[double]  distances
        cdef vector[int]  points_indices

        self.c_annoy.queryTreeForest(point, k, distances, points_indices)

        if return_distance:
            return [points_indices, distances]
        return points_indices
