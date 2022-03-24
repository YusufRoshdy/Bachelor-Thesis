#!python
#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
# distutils: language = c++

from libcpp.memory cimport shared_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool

cdef extern from "IterAnnoyImpl.cpp":
    pass

cdef extern from "IterAnnoy.h" namespace "annoy":
    cdef cppclass Node:
        bool is_leaf
        int start, end
        shared_ptr[Node] left, right
        double b
        vector[double] norm

    cdef cppclass Annoy:
        vector[vector[double]] X
        int leaf_size, n_trees
        vector[shared_ptr[Node]] trees
        vector[vector[int]] indexes
        string last
        Annoy(vector[vector[double]] &X, int, int, int) except +
        Node buildTree() except +
        vector[int] queryTree(vector[double]& point, int k, vector[double]& distances, vector[int]&  points_indices) except +
        vector[int] _queryTree(vector[double]& point, vector[int]& index, shared_ptr[Node] tree, int k) except +
