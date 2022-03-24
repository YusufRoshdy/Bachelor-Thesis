#ifndef ITER_ANNOY_H
#define ITER_ANNOY_H

#include <vector>
#include <memory>


namespace annoy {
    struct Node {
        bool is_leaf = false;
        int start = -1, end = -1;
        std::shared_ptr<Node> left = nullptr, right = nullptr;
        double b = 0;
        std::vector<double> norm;

        Node(int start, int end);
    };

    class Annoy {
    private:
        static void vectors_midpoint(const std::vector<double> &v1, const std::vector<double> &v2, std::vector<double>& res);
        static void vector_diff(const std::vector<double> &v1, const std::vector<double> &v2, std::vector<double>& res);
        static double vector_distance(const std::vector<double> &v1, const std::vector<double> &v2);
        static double split_imbalance(double left_start, double left_end, double right_start, double right_end);
    public:
        std::vector<std::vector<double>> X;
        int leaf_size, n_trees, n_jobs;
        std::vector<std::shared_ptr<Node>> trees;
        std::vector<std::vector<int>> indexes;
        std::string last;

        Annoy(std::vector<std::vector<double>> X, int leaf_size, int n_trees, int n_jobs);

        void buildTree();

        void queryTree(const std::vector<double>& point, int k, std::vector<double>& distances, std::vector<int>&  points_indices);
        void queryTree2(const std::vector<double>& point, int k, std::vector<double>& distances, std::vector<int>&  points_indices);

        int _queryTree(const std::vector<double>& point, const std::vector<int>& index, const std::shared_ptr<Node> tree, int k, std::vector<int>& results);
    };
}

#endif
