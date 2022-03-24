from annoy import AnnoyIndex
import random
from sklearn.neighbors import NearestNeighbors
from Annoy import Annoy
from CAnnoy import Annoy as CAnnoy
from ItrAnnoy import ItrAnnoy
from CItrAnnoy import CItrAnnoy
import time
import numpy as np
from sklearn.neighbors import BallTree
from CppAnnoy.IterAnnoy import PyAnnoy

if __name__ == '__main__':

    # Setup parameters
    f = 128
    distances = False
    k = 100
    n_trees = 100
    repeats = 100
    leaf_size = 30
    n = 10000

    samples = np.random.random((n, f)).astype('float32')
    query = np.random.random((1, f)).astype('float32')


    # SKLearn
    # # ball tree
    # start_time = time.time()
    # neigh = NearestNeighbors(n_neighbors=k, radius=0.4, algorithm='ball_tree', metric='euclidean', n_jobs=-1)
    # neigh.fit(samples)
    # print(f'sklearn ball_tree fit {time.time()-start_time}s')
    # start_time = time.time()
    # for _ in range(repeats):
    #     x2 = neigh.kneighbors(query, k, return_distance=distances)[0]
    # print(f'sklearn ball_tree query {time.time()-start_time}s')

    # # KD tree
    # start_time = time.time()
    # neigh = NearestNeighbors(n_neighbors=k, radius=0.4, algorithm='kd_tree', metric='euclidean', n_jobs=-1)
    # neigh.fit(samples)
    # print(f'sklearn kd_tree fit {time.time()-start_time}s')
    # start_time = time.time()
    # for _ in range(repeats):
    #     x2 = neigh.kneighbors(query, k, return_distance=distances)[0]
    # print(f'sklearn kd_tree query {time.time()-start_time}s')

    # brute
    start_time = time.time()
    neigh = NearestNeighbors(n_neighbors=k, radius=0.4, algorithm='brute', metric='euclidean', n_jobs=-1)
    neigh.fit(samples)
    print(f'sklearn brute fit {time.time()-start_time:0.4f}s')
    start_time = time.time()
    for _ in range(repeats):
        brute = neigh.kneighbors(query, k, return_distance=distances)[0]
    print(f'sklearn brute query {time.time()-start_time:0.4f}s\n')





    # Spotify
    start_time = time.time()
    t = AnnoyIndex(f, 'euclidean')  # Length of item vector that will be indexed
    for i, v in enumerate(samples):
        t.add_item(i, v)
    t.build(n_trees)  # 10 trees
    print(f'Spotify annoy fit took {time.time() - start_time}s')
    annoy_recall = 0
    time_sum = 0
    for _ in range(repeats):
        start_time = time.time()
        x = t.get_nns_by_vector(query[0], k, include_distances=distances)
        time_sum += time.time() - start_time
        common = 0
        for a in x:
            if a in brute:
                common += 1
        common /= k
        annoy_recall += common
    annoy_recall /= repeats
    print(f'{annoy_recall:0.4f} Spotify annoy query took {time_sum:0.6f}s\n')


    # I was modifing it anfd there is a bug
    # CAnnoy - cython recursion
    # start_time = time.time()
    # my_annoy = CAnnoy(samples, leaf_size=leaf_size, n_trees=n_trees)
    # my_cannoy_time = time.time() - start_time
    # print(f'my CAnnoy fit {my_cannoy_time}s')
    #
    # start_time = time.time()
    # for _ in range(repeats):
    #     x3 = my_annoy.query(query[0], k, return_distance=distances)
    # print(f'my CAnnoy query {time.time() - start_time}s')


    # Python recursion
    # start_time = time.time()
    # my_annoy = Annoy(samples, leaf_size=leaf_size, n_trees=n_trees, n_jobs=0)
    # my_annoy_time = time.time() - start_time
    # print(f'my annoy fit {my_annoy_time:0.4f}s')
    # start_time = time.time()
    # for _ in range(repeats):
    #     x3 = my_annoy.query(query[0], k, return_distance=distances)
    # print(f'my annoy query {time.time() - start_time:0.4f}s\n')


    # Python iterative
    # start_time = time.time()
    # my_annoy = ItrAnnoy(samples, leaf_size=leaf_size, n_trees=n_trees)
    # my_itrannoy_time = time.time() - start_time
    # print(f'my ItrAnnoy fit {my_itrannoy_time:0.4f}s')
    # start_time = time.time()
    # for _ in range(repeats):
    #     x3 = my_annoy.query(query[0], k, return_distance=distances)
    # print(f'my ItrAnnoy query {time.time() - start_time:0.4f}s\n')


    # Cython iterative
    # start_time = time.time()
    # my_annoy = CItrAnnoy(samples, leaf_size=leaf_size, n_trees=n_trees)
    # my_citrannoy_time = time.time() - start_time
    # print(f'my CItrAnnoy fit {my_citrannoy_time:0.4f}s\n')
    # Query is not implemented yet
    # start_time = time.time()
    # for _ in range(repeats):
    #     x3 = my_annoy.query(query[0], k, return_distance=distances)
    # print(f'my ItrAnnoy query {time.time() - start_time}s')

    print('\n\n\nNo parallel')
    # CPP Annoy, the full forest build is not fully implemented so I am running build tree n_trees times to get the time
    my_cpp_annoy = PyAnnoy(samples, leaf_size=leaf_size, n_trees=n_trees)
    start_time = time.time()

    my_cpp_annoy.build_tree()
    my_cpp_annoy_time = time.time() - start_time

    print(f'\tmy CPP annoy fit {my_cpp_annoy_time:0.4f}s')

    cpp_recall = 0
    time_sum = 0
    try:
        for _ in range(repeats):
            start_time = time.time()
            xcpp = my_cpp_annoy.query_tree(query[0], k)
            time_sum += time.time() - start_time

            common = 0
            for a in xcpp:
                if a in brute:
                    common += 1
            common /= k
            cpp_recall += common
        cpp_recall /= repeats
    except Exception as e:
        print('error 3', e)
    print(f'\t{cpp_recall:0.4f} my CPP annoy query took {time_sum:0.4f}s\n')
    # print(x)
    # print(xcpp)

    # common = 0
    # for a in x:
    #     if a in xcpp:
    #         common += 1
    # print('other common', common)

# ***************************************************************************************************





    query_time_not_parallel = time_sum
    fit_time_not_parallel = my_cpp_annoy_time
    for n_jobs in range(2, 8):
        print(f'Parallel {n_jobs=}')

        # CPP Annoy, the full forest build is not fully implemented so I am running build tree n_trees times to get the time
        my_cpp_annoy = PyAnnoy(samples, leaf_size=leaf_size, n_trees=n_trees, n_jobs=n_jobs)
        start_time = time.time()

        my_cpp_annoy.build_tree()
        my_cpp_annoy_time = time.time() - start_time

        print(f'\tmy CPP annoy fit {my_cpp_annoy_time:0.4f}s ({my_cpp_annoy_time / fit_time_not_parallel * 100:0.2f}%)')

        cpp_recall = 0
        time_sum = 0
        try:
            for _ in range(repeats):
                start_time = time.time()
                xcpp = my_cpp_annoy.query_tree(query[0], k)
                time_sum += time.time() - start_time

                common = 0
                for a in xcpp:
                    if a in brute:
                        common += 1
                common /= k
                cpp_recall += common
            cpp_recall /= repeats
        except Exception as e:
            print('error 3', e)
        print(f'\t{cpp_recall:0.4f} my CPP annoy query took {time_sum:0.4f}s ({time_sum / query_time_not_parallel * 100:0.2f}%)\n')
