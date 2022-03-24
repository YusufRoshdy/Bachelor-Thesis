
from CppAnnoy.IterAnnoy import PyAnnoy as annoy
import time
import numpy as np
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from scipy.sparse import csr_matrix

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import KNeighborsTransformer
from sklearn.utils._testing import assert_array_almost_equal
from sklearn.datasets import fetch_openml
from sklearn.pipeline import make_pipeline
from sklearn.manifold import TSNE
from sklearn.utils import shuffle


class AnnoyTransformer(TransformerMixin, BaseEstimator):
    """Wrapper for using annoy.AnnoyIndex as sklearn's KNeighborsTransformer"""

    def __init__(self, n_neighbors=5, metric="euclidean", n_trees=10, search_k=-1):
        self.n_neighbors = n_neighbors
        self.n_trees = n_trees
        self.search_k = search_k
        self.metric = metric

    def fit(self, X):
        self.n_samples_fit_ = X.shape[0]

        self.annoy_ = annoy(X, leaf_size=40, n_trees=self.n_trees)
        self.annoy_.build_tree()
        # my_annoy_ret = my_annoy.query_tree(query[0], k)

        return self

    def transform(self, X):
        return self._transform(X)

    def fit_transform(self, X, y=None):
        return self.fit(X)._transform(X=None)

    def _transform(self, X):
        """As `transform`, but handles X is None for faster `fit_transform`."""

        n_samples_transform = self.n_samples_fit_ if X is None else X.shape[0]

        # For compatibility reasons, as each sample is considered as its own
        # neighbor, one extra neighbor will be computed.
        n_neighbors = self.n_neighbors + 1

        indices = np.empty((n_samples_transform, n_neighbors), dtype=int)
        distances = np.empty((n_samples_transform, n_neighbors))

        if X is None:
            return
            # for i in range(self.annoy_.get_n_items()):
            #     ind, dist = self.annoy_.get_nns_by_item(
            #         i, n_neighbors, self.search_k, include_distances=True
            #     )
            #
            #     indices[i], distances[i] = ind, dist
        else:
            for i, x in enumerate(X):
                # indices[i], distances[i] = self.annoy_.get_nns_by_vector(
                #     x.tolist(), n_neighbors, self.search_k, include_distances=True
                # )

                res = self.annoy_.query_tree(x, n_neighbors, return_distance=True)
                # print(len(res[0]), len(res[1]), indices[i].shape, distances[i].shape)
                indices[i], distances[i] = res

        indptr = np.arange(0, n_samples_transform * n_neighbors + 1, n_neighbors)
        kneighbors_graph = csr_matrix(
            (distances.ravel(), indices.ravel(), indptr),
            shape=(n_samples_transform, self.n_samples_fit_),
        )

        return kneighbors_graph


f = 64
distances = False
k = 10
n_trees = 50
repeats = 10
leaf_size = 100
n = 10000
samples = np.random.random((n, f)).astype('float32')
query = np.random.random((1, f)).astype('float32')

# annoy_transformer = AnnoyTransformer()
# annoy_transformer.fit(samples)
# print('fitting complete!')
# res = annoy_transformer.transform(samples)
# print('transformed')
# print(res)

import cProfile
import numpy as np
from CppAnnoy.IterAnnoy import PyAnnoy


# cProfile.run('re.compile("foo|bar")')
f = 16
distances = False
k = 10
n_trees = 50
repeats = 100
leaf_size = 40
n = 10000

samples = np.random.random((n, f)).astype('float32')
query = np.random.random((1, f)).astype('float32')


# CPP Annoy, the full forest build is not fully implemented so I am running build tree n_trees times to get the time
# cProfile.run('PyAnnoy(samples, leaf_size=leaf_size, n_trees=n_trees)')
# my_cpp_annoy = PyAnnoy(samples, leaf_size=leaf_size, n_trees=n_trees)
#
#
#
# cProfile.run('my_cpp_annoy.build_tree()')
# my_cpp_annoy.build_tree()
#
# xcpp = my_cpp_annoy.query_tree(query[0], k)
# cProfile.run('my_cpp_annoy.query_tree(query[0], k)')


from memory_profiler import profile, memory_usage
from annoy import AnnoyIndex

# @profile
def my(samples, query, leaf_size, n_trees, k):
    my_cpp_annoy = PyAnnoy(samples, leaf_size=leaf_size, n_trees=n_trees)
    my_cpp_annoy.build_tree()
    xcpp = my_cpp_annoy.query_tree(query[0], k)

# @profile
def spotify(samples, query, n_trees, k):
    t = AnnoyIndex(samples.shape[1], 'euclidean')  # Length of item vector that will be indexed
    for i, v in enumerate(samples):
        t.add_item(i, v)
    t.build(n_trees)  # 10 trees
    x = t.get_nns_by_vector(query[0], k, include_distances=distances)

if __name__ == '__main__':
    fs = [16, 32, 64, 128]
    n_samples = [20000]
    leaf_sizes = [40]
    Ks = [10]
    n_trees_list = [1, 10, 20, 50, 100]

    for k in Ks:
        for f in fs:
            for n in n_samples:
                samples = np.random.random((n, f)).astype('float32')
                query = np.random.random((1, f)).astype('float32')
                for n_trees in n_trees_list:
                    for leaf_size in leaf_sizes:
                        print(f'{k=}, {f=}, {n=}, {n_trees=}, {leaf_size=}')
                        my_usage = memory_usage((my, (samples, query, leaf_size, n_trees, k), {}))
                        my_usage = np.array(my_usage)
                        spotify_usage = memory_usage((spotify, (samples, query, n_trees, k), {}))
                        spotify_usage = np.array(spotify_usage)
                        print('my:', my_usage.max() - my_usage.min(), my_usage.max() - my_usage[0])
                        print('spotify:', spotify_usage.max() - spotify_usage.min(), spotify_usage.max() - spotify_usage[0])
