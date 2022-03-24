#include <algorithm>
#include <cstdlib>
#include <memory>
#include <numeric>
#include <stack>
#include <vector>
#include <queue>
#include <set>
#include <iostream>
#include <string>
#include <omp.h>
#include <float.h>
//#include <execution> for parallel sort needs c++17
#include "IterAnnoy.h"

#include <chrono>
#include <ctime>

using namespace std;

namespace annoy {
    Node::Node(int start, int end) : start(start), end(end) {}

    Annoy::Annoy(vector<vector<double>> X_in, int leaf_size, int n_trees, int n_jobs) : X(move(X_in)), leaf_size(leaf_size), n_trees(n_trees), n_jobs(n_jobs) {
        trees.resize(n_trees);
        indexes.resize(n_trees);
    }


    void Annoy::buildTree() {
        auto size = static_cast<int>(X.size());
        #pragma omp parallel for num_threads(n_jobs)
        for(int tree_idx = 0; tree_idx < n_trees; tree_idx++){
            int i, j, start_idx, end_idx, length, s, e, p1, p2;
            double temp_sum, b, imbalance_ratio;
            vector<int> index(size);
            stack<shared_ptr<Node>> nodes_to_build;
            vector<double> plane_norm(X[0].size()), plane_point(X[0].size());
            shared_ptr<Node> node;
            // TODO: readability - move declarations inside loop (and verify with assembly)

            // Fill with [0..size-1]
            iota(begin(index), end(index), 0);

            Node root(0, size - 1);
            auto root_ref = make_shared<Node>(root);
            nodes_to_build.push(root_ref);
            int imbalance_flag = 0;
            while (!nodes_to_build.empty()) {
                node = nodes_to_build.top();
                nodes_to_build.pop();

                start_idx = node->start;
                end_idx = node->end;
                length = end_idx - start_idx + 1;

                if (length <= leaf_size) {
                    node->is_leaf = true;
                    continue;
                }

                p1 = rand() % length;
                p2 = rand() % length;
                if (p2 == p1) p2 = (p2 + 1) % length;


                vector_diff(X[p2], X[p1], plane_norm);
                vectors_midpoint(X[p1], X[p2], plane_point);

                b = 0;
                for (i = 0; i < plane_norm.size(); i++)
                    b += plane_norm[i] * plane_point[i];

                node->b = b;
                node->norm = plane_norm;

                s = start_idx;
                e = end_idx;
                for (i = start_idx; i < end_idx+1; i++) {
                    temp_sum = 0;
                    for (j = 0; j < X[index[i]].size(); j++)
                        temp_sum += plane_norm[j] * X[index[i]][j];

                    if (temp_sum <= b) {
                        swap(index[s], index[i]);
                        s += 1;
                    }
                    else {
                        swap(index[e], index[i]);
                        e -= 1;
                    }
                }

                imbalance_ratio = split_imbalance(start_idx, s - 1, e - 1, end_idx);
                if (imbalance_ratio >= 0.80) {
                    // TODO: handle limiting the retry count so we don't go into an infinite loop
                    // Currently I just make that node a leaf
                    if (imbalance_flag > 3) {
                        imbalance_flag = 0;
                        node->is_leaf = true;
                        continue;
                    }
                    imbalance_flag += 1;
                    nodes_to_build.push(node);
                    continue;
                }
                imbalance_flag = 0;

                shared_ptr<Node> left = make_shared<Node>(start_idx, s - 1);
                shared_ptr<Node> right = make_shared<Node>(e + 1, end_idx);

                /////
                node->left = left;
                node->right = right;
                /////

                nodes_to_build.push(left);
                nodes_to_build.push(right);
            }
            trees[tree_idx] = root_ref;
            indexes[tree_idx] = index;
        }
    }

    void Annoy::queryTree(const vector<double>& point, int k, vector<double>& distances,vector<int>&  points_indices) /* const */ {
//        queryTree2(point, k, distances, points_indices);
//        return;



        //
        auto start = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds;
        if (n_jobs > 1){
            vector<vector<int>> p_i(n_jobs);
            #pragma omp parallel for num_threads(n_jobs) schedule(dynamic)
            for(int i = 0; i < n_trees; i++){
                _queryTree(point, indexes[i], trees[i], k, p_i[omp_get_thread_num()]);
            }

            for(int i = 0; i < n_jobs; i++){
                points_indices.insert(end(points_indices), p_i[i].begin(), p_i[i].end());
            }
        }
        else{
            for(int i = 0; i < n_trees; i++){
                _queryTree(point, indexes[i], trees[i], k, points_indices);
            }
        }

        auto end = std::chrono::system_clock::now();
        elapsed_seconds = end-start;
        cout << "\t\ttrees time: " << elapsed_seconds.count() << "s\n";

        //e-5
        vector<int>::iterator ip = unique(points_indices.begin(), points_indices.end());
        points_indices.resize(distance(points_indices.begin(), ip));
        distances.resize(points_indices.size());



        //
        start = std::chrono::system_clock::now();
        #pragma omp parallel for num_threads(n_jobs) schedule(static)
        for(int i = 0; i < points_indices.size(); i++) {
            distances[i] = vector_distance(point, X[points_indices[i]]);
        }
        end = std::chrono::system_clock::now();
        elapsed_seconds = end-start;
        cout << "\t\tdistances time: " << elapsed_seconds.count() << "s\n";



        // e-5
        vector<pair<double, int>> pairs(points_indices.size());
        #pragma omp parallel for num_threads(n_jobs) schedule(static)
        for(int i = 0; i < points_indices.size(); i++){
            pairs[i] = make_pair(distances[i], points_indices[i]);
        }

        //
        start = std::chrono::system_clock::now();
//        sort(execution::par_unseq, pairs.begin(), pairs.end()); // for parallel sort needs c++17
        sort(pairs.begin(), pairs.end());
        end = std::chrono::system_clock::now();
        elapsed_seconds = end-start;
        cout << "\t\tsort time: " << elapsed_seconds.count() << "s\n";

        // e-5
        #pragma omp parallel for num_threads(n_jobs) schedule(static)
        for(int i = 0; i < points_indices.size(); i++){
            distances[i] = pairs[i].first;
            points_indices[i] = pairs[i].second;
        }

        if(points_indices.size() > k){
            points_indices.resize(k);
            distances.resize(k);
        }
    }

//    int Annoy::_queryTree(const vector<double>& point, const vector<int>& index, const shared_ptr<Node> tree, int k, vector<int>& results)  {
//        priority_queue<pair<double, shared_ptr<Node> > > q;
//        q.push(make_pair(DBL_MAX, tree));
//        vector<int> nns;
//
//        while (nns.size() < k*2 && !q.empty()) {
//            const pair<double, shared_ptr<Node> >& top = q.top();
//            double d = top.first;
//            shared_ptr<Node> nd = top.second;
//            q.pop();
//
//            int n_descendants = nd->end - nd->start;
//
//            if (nd->is_leaf) {
//                if (nd->start > nd->end) continue; // empty leaf
//                nns.insert(end(nns), index.begin() + nd->start, index.begin() + nd->end);
//            }
//            else {
//                double dot_product = 0, margin = 0;
//                for (int j = 0; j < point.size(); j++)
//                    dot_product += nd->norm[j] * point[j];
//
//                margin = dot_product - nd->b;
//
//                if (dot_product <= nd->b) {
////                    if (-margin < 0)
////                        cout << "margin\n";
//                    q.push( make_pair(min(d, -margin), nd->left) );
//                    q.push( make_pair(min(d, margin), nd->right) );
//                }else{
////                    if (margin < 0)
////                        cout << "margin\n";
//                    q.push( make_pair(min(d, margin), nd->right) );
//                    q.push( make_pair(min(d, -margin), nd->left) );
//                }
//
//            }
//        }
//
//        results.insert(end(results), nns.begin(), nns.end());
//        return 0;
//    }



    void Annoy::queryTree2(const vector<double>& point, int k, vector<double>& distances,vector<int>&  points_indices) /* const */ {
//        n_jobs = 1;
        //
        auto start = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds;

        vector<int> nns;
        priority_queue<pair<double, pair<int, shared_ptr<Node>> > > q;
        for(int i = 0; i < n_trees; i++){
            q.push( make_pair(DBL_MAX, make_pair(i, trees[i]) ) );
        }
        pair<double, pair<int, shared_ptr<Node>> > temp;
        while (nns.size() < k*n_trees && !q.empty()) {
            const pair<double, pair<int, shared_ptr<Node>> >& top = q.top();
            double d = top.first;
            int i = top.second.first;
            shared_ptr<Node> nd = top.second.second;
            q.pop();

            int n_descendants = nd->end - nd->start;

            if (nd->is_leaf) {
                if (nd->start > nd->end) continue; // empty leaf
                nns.insert(end(nns), indexes[i].begin() + nd->start, indexes[i].begin() + nd->end);
            }
            else if (n_descendants <= leaf_size) {
//                cout << "********************************************************************************************************************************************************************************************\n" ;
                nns.insert(end(nns), indexes[i].begin() + nd->start, indexes[i].begin() + nd->end);
            }
            else {
                double dot_product = 0, margin = 0;
                for (int j = 0; j < point.size(); j++)
                    dot_product += nd->norm[j] * point[j];

                margin = dot_product - nd->b;

                if (dot_product <= nd->b) {
//                    if (-margin < 0)
//                        cout << "margin\n";
                    q.push( make_pair(min(d, -margin), make_pair(i, nd->left) ) );
                    q.push( make_pair(min(d, margin), make_pair(i, nd->right) ) );
                }else{
//                    if (margin < 0)
//                        cout << "margin\n";
                    q.push( make_pair(min(d, margin), make_pair(i, nd->right) ) );
                    q.push( make_pair(min(d, -margin), make_pair(i, nd->left) ) );
                }

            }
        }

        points_indices.insert(end(points_indices), nns.begin(), nns.end());
//////////////////////////////////////////////////////////////////////////////////////////////////
        auto end = std::chrono::system_clock::now();
        elapsed_seconds = end-start;
//        cout << "\t\ttrees time: " << elapsed_seconds.count() << "s,\tpoints_indices.size: " << points_indices.size() << "\n";

        //e-5
        vector<int>::iterator ip = unique(points_indices.begin(), points_indices.end());
        points_indices.resize(distance(points_indices.begin(), ip));
        distances.resize(points_indices.size());



        //
        start = std::chrono::system_clock::now();
        #pragma omp parallel for num_threads(n_jobs) schedule(static)
        for(int i = 0; i < points_indices.size(); i++) {
            distances[i] = vector_distance(point, X[points_indices[i]]);
        }
        end = std::chrono::system_clock::now();
        elapsed_seconds = end-start;
//        cout << "\t\tdistances time: " << elapsed_seconds.count() << "s\n";



        // e-5
        vector<pair<double, int>> pairs(points_indices.size());
        #pragma omp parallel for num_threads(n_jobs) schedule(static)
        for(int i = 0; i < points_indices.size(); i++){
            pairs[i] = make_pair(distances[i], points_indices[i]);
        }

        //
        start = std::chrono::system_clock::now();
//        sort(execution::par_unseq, pairs.begin(), pairs.end()); // for parallel sort needs c++17
        sort(pairs.begin(), pairs.end());
        end = std::chrono::system_clock::now();
        elapsed_seconds = end-start;
//        cout << "\t\tsort time: " << elapsed_seconds.count() << "s\n";

        // e-5
        #pragma omp parallel for num_threads(n_jobs) schedule(static)
        for(int i = 0; i < points_indices.size(); i++){
            distances[i] = pairs[i].first;
            points_indices[i] = pairs[i].second;
        }

        if(points_indices.size() > k){
            points_indices.resize(k);
            distances.resize(k);
        }
    }

    int Annoy::_queryTree(const vector<double>& point, const vector<int>& index, const shared_ptr<Node> tree, int k,vector<int>& results)  {
        if (tree->is_leaf) {
            if (tree->start > tree->end) {
                return 0; // empty leaf
            }
            results.insert(end(results), index.begin() + tree->start, index.begin() + tree->end);
            return (tree->end - tree->start);
        }

        double dot_product = 0;
        for (int j = 0; j < point.size(); j++)
            dot_product += tree->norm[j] * point[j];

        if (dot_product <= tree->b) {
            auto leaves_size = _queryTree(point, index, tree->left, k, results);
            if (leaves_size < k) {
                int temp_k = k - leaves_size;
                auto rightLeaves = _queryTree(point, index, tree->right, temp_k, results);
                leaves_size += rightLeaves;
            }
            return leaves_size;
        }

        auto leaves_size = _queryTree(point, index, tree->right, k, results);
        if (leaves_size < k) {
            int temp_k = k - leaves_size;
            auto leftLeaves = _queryTree(point, index, tree->left, temp_k, results);
            leaves_size += leftLeaves;
        }
        return leaves_size;
    }


    void Annoy::vectors_midpoint(const vector<double>& v1, const vector<double>& v2, vector<double>& res) {
        size_t size = v1.size();
        for (size_t i = 0; i < size; i++)
            res[i] = (v1[i] + v2[i]) / 2;
    }

    void Annoy::vector_diff(const vector<double> &v1, const vector<double> &v2, vector<double>& res) {
        size_t size = v1.size();
        for (size_t i = 0; i < size; i++)
            res[i] = v1[i] - v2[i];
    }

    double Annoy::vector_distance(const vector<double> &v1, const vector<double> &v2) {
        double res = 0.0, dist;
        size_t size = v1.size();
        for (size_t i = 0; i < size; i++){
            dist = v1[i]-v2[i];
            res += dist*dist;
        }
        return res > 0.0 ? sqrt(res) : 0.0;
    }

    double Annoy::split_imbalance(double left_start, double left_end, double right_start, double right_end){
        return max(left_end - left_start, right_end - right_start) / (right_end - left_start);
    }
}

