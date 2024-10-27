#pragma once

#include "utils.hpp"

#include <map>
#include <list>
#include <algorithm>

using namespace std;

namespace vl {

struct tree {
    tree() : m_num_leaves(0), m_num_features(0), m_num_classes(0) {}

    void parse(std::ifstream& in) {
        std::string line;
        std::getline(in, line);
        std::istringstream is(line);
        is >> line;
        assert(line == "classifier-decision-tree");
        is >> m_num_features;
        is >> m_num_classes;
        std::getline(in, line);  // skip line containing labels
        m_root = new node;
        uint32_t max_depth = parse_node(in, m_root, 1);  // parse root
        (void)max_depth;
    }

    ~tree() { delete_node(m_root); }

    struct node {
        node()
            : feature(constants::invalid_feature)
            , threshold(constants::invalid_threshold)
            , score(constants::invalid_score)
            , left(nullptr)
            , right(nullptr) {}
        feature_t feature;
        float_t threshold;
        float_t score;
        node* left;
        node* right;
    };

    uint32_t num_leaves() const {
        assert(m_hyper_rectangles.size() == m_num_leaves);
        return m_num_leaves;
    }

    bool annotated() const { return !m_hyper_rectangles.empty(); }

    void annotate() {
        if (!annotated()) {
            m_hyper_rectangles.reserve(m_num_leaves);
            hyper_rectangle_t root_hr;
            root_hr.H.resize(m_num_features, {-constants::inf, constants::inf});
            annotate(m_root, root_hr);  // annotate recursively from root
            assert(m_hyper_rectangles.size() == m_num_leaves);
        }
    }

    float_t raw_prediction_score(instance_t const& x) const {
        return raw_prediction_score(x, m_root);
    }
    
    void compute_norms(instance_t const& x, const float_t p,  //
                       const bool sort)                       //
    {
        std::vector<float_t> delta(m_num_features);
        for (auto& hr : m_hyper_rectangles) {
            if (!hr.empty) {
                assert(hr.H.size() == m_num_features);
                assert(hr.score != constants::invalid_score);
                for (uint32_t i = 0; i != m_num_features; ++i) {
                    auto l_i = hr.H[i].first;
                    auto r_i = hr.H[i].second;
                    auto x_i = x[i];
                    if (x_i <= l_i) {
                        delta[i] = (l_i - x_i); //+ 1e-4;
                    } else if (x_i > r_i) {
                        delta[i] = (r_i - x_i); //- 1e-4;
                    } else {
                        delta[i] = 0.0;
                    }
                }

                hr.norm = norm(delta, p);

#ifdef DEBUG
                /* For computing the optimal adversarial attack. */
                hr.delta = delta;
#endif

                if (p != constants::inf and p != 0) {
                    hr.norm = eta(pow(hr.norm, p), constants::scaling_factor);
                }
            }
        }

        if (sort) {
            std::sort(m_hyper_rectangles.begin(), m_hyper_rectangles.end(),
                      [](auto const& l, auto const& r) { return l.norm < r.norm; });
        }
    }

    void compute_norms(instance_t const& x, const float_t p,  //
                       const bool sort, vector<double>& stds_list)                       //
    {
        std::vector<float_t> delta(m_num_features);
        for (auto& hr : m_hyper_rectangles) {
            if (!hr.empty) {
                assert(hr.H.size() == m_num_features);
                assert(hr.score != constants::invalid_score);
                for (uint32_t i = 0; i != m_num_features; ++i) {
                    auto l_i = hr.H[i].first;
                    auto r_i = hr.H[i].second;
                    auto x_i = x[i];
                    if (x_i <= l_i) {
                        delta[i] = (l_i - x_i)/stds_list[i] + 1e-4;
                    } else if (x_i > r_i) {
                        delta[i] = (r_i - x_i)/stds_list[i] - 1e-4;
                    } else {
                        delta[i] = 0.0;
                    }
                }

                hr.norm = norm(delta, p);

#ifdef DEBUG
                /* For computing the optimal adversarial attack. */
                hr.delta = delta;
#endif

                if (p != constants::inf and p != 0) {
                    hr.norm = eta(pow(hr.norm, p), constants::scaling_factor);
                }
            }
        }

        if (sort) {
            std::sort(m_hyper_rectangles.begin(), m_hyper_rectangles.end(),
                      [](auto const& l, auto const& r) { return l.norm < r.norm; });
        }
    }

    void compute_norms(instance_t const& x, const float_t p,  //
                       const bool sort, features_list_t& attacked_features_list, vector<double>& stds_list)                       //
    {
        std::vector<float_t> delta(m_num_features);
        for (auto& hr : m_hyper_rectangles) {
            if (!hr.empty) {
                assert(hr.H.size() == m_num_features);
                assert(hr.score != constants::invalid_score);
                for (uint32_t i = 0; i != m_num_features; ++i) {
                    auto l_i = hr.H[i].first;
                    auto r_i = hr.H[i].second;
                    auto x_i = x[i];
                    if (x_i <= l_i) {
                        if(std::find(attacked_features_list.begin(), attacked_features_list.end(), i) != attacked_features_list.end())
                            delta[i] = (l_i - x_i)/stds_list[i] + 1e-4;
                        else
                            //TODO: what happens with norm 0?
                            delta[i] = constants::inf;
                    } else if (x_i > r_i) {
                        if(std::find(attacked_features_list.begin(), attacked_features_list.end(), i) != attacked_features_list.end())
                            delta[i] = (r_i - x_i)/stds_list[i] - 1e-4;
                        else
                            delta[i] = constants::inf;
                    } else {
                        delta[i] = 0.0;
                    }
                }

                hr.norm = norm(delta, p);

#ifdef DEBUG
                /* For computing the optimal adversarial attack. */
                hr.delta = delta;
#endif

                if (p != constants::inf and p != 0) {
                    hr.norm = eta(pow(hr.norm, p), constants::scaling_factor);
                }
            }
        }

        if (sort) {
            std::sort(m_hyper_rectangles.begin(), m_hyper_rectangles.end(),
                      [](auto const& l, auto const& r) { return l.norm < r.norm; });
        }
    }

    /*
        Compute adversarial gains for each leaf of the tree
        and returns the maximum one.
     */
    float_t compute_gains(instance_t const& x, const label_t y, const float_t k,
                          const bool scale_gains, const bool sort)  //
    {
        assert(y == 0 or y == 1);
        float_t max_gain = -constants::inf;
        const float_t rps = raw_prediction_score(x);

#ifdef DEBUG
        clear_attacks();
        uint64_t pos_of_max = 0;
#endif

        for (uint64_t i = 0; i != m_hyper_rectangles.size(); ++i) {
            auto& hr = m_hyper_rectangles[i];
            if (!hr.empty and hr.norm <= k) {  // only if valid
                float_t gain = 0;
                if (y == 0) gain = hr.score - rps;
                if (y == 1) gain = rps - hr.score;
                if (gain >= 0) {
                    hr.gain = scale_gains ? eta(gain, constants::scaling_factor) : gain;
                    if (hr.gain > max_gain) {
                        max_gain = hr.gain;
#ifdef DEBUG
                        pos_of_max = i;
#endif
                    }
                } else {
                    hr.gain = constants::inf;
                }
            }
        }

#ifdef DEBUG
        set_attack_on_leaf(pos_of_max);
#endif

        if (sort) {
            std::sort(m_hyper_rectangles.begin(), m_hyper_rectangles.end(),
                      [](auto const& l, auto const& r) { return l.gain < r.gain; });
        }

        return max_gain;
    }

    hyper_rectangle_t const& hyper_rectangle(uint64_t leaf_index) const {
        assert(leaf_index < num_leaves());
        return m_hyper_rectangles[leaf_index];
    }

#ifdef DEBUG
    void set_attack_on_leaf(uint64_t leaf_index) {
        assert(leaf_index < num_leaves());
        m_hyper_rectangles[leaf_index].is_part_of_attack = true;
    }
    void clear_attacks() {
        for (auto& hr : m_hyper_rectangles) hr.is_part_of_attack = false;
    }
#endif

    float_t max_gain(instance_t const& x, const label_t y,  //
                     const float_t p,                       //
                     const float_t k
    ) {
        assert(y == 0 or y == 1);
        constexpr bool sort_norms = false;
        constexpr bool sort_gains = false;
        constexpr bool scale_gains = false;
        compute_norms(x, p, sort_norms);
        return compute_gains(x, y, k, scale_gains, sort_gains);
    }

    float_t max_gain(instance_t const& x, const label_t y,  //
                     const float_t p,                       //
                     const float_t k,                        //
                     vector<double>& stds_list
    ) {
        assert(y == 0 or y == 1);
        constexpr bool sort_norms = false;
        constexpr bool sort_gains = false;
        constexpr bool scale_gains = false;
        compute_norms(x, p, sort_norms, stds_list);
        return compute_gains(x, y, k, scale_gains, sort_gains);
    }

    float_t max_gain(instance_t const& x, const label_t y,  //
                     const float_t p,                       //
                     const float_t k,                        //
                     features_list_t& attacked_features_list,
                     vector<double>& stds_list
    ) {
        assert(y == 0 or y == 1);
        constexpr bool sort_norms = false;
        constexpr bool sort_gains = false;
        constexpr bool scale_gains = false;
        compute_norms(x, p, sort_norms, attacked_features_list, stds_list);
        return compute_gains(x, y, k, scale_gains, sort_gains);
    }

    uint64_t divide_gains_by_t(const float_t t) {
        uint64_t max_gain = 0;
        for (auto& hr : m_hyper_rectangles) {
            if (!hr.empty and hr.gain != constants::inf and hr.gain >= 0) {
                hr.gain = uint64_t(hr.gain / t);
                if (hr.gain > max_gain) max_gain = hr.gain;
            }
        }
        return max_gain;
    }

    uint32_t num_features() const { return m_num_features; }

    void print(std::ostream& out) const {
        out << "Print decision tree, num features " << m_num_features << ", num classes "
            << m_num_classes << endl;
        print_aux(m_root, out, "");
    }

private:
    uint32_t m_num_leaves;
    uint32_t m_num_features;
    uint32_t m_num_classes;
    node* m_root;
    std::vector<hyper_rectangle_t> m_hyper_rectangles;

    bool is_leaf(node const* n) const { return n->left == nullptr and n->right == nullptr; }

    float_t raw_prediction_score(instance_t const& x, node const* n) const {
        if (is_leaf(n)) return n->score;
        if (x[n->feature] <= n->threshold) return raw_prediction_score(x, n->left);
        return raw_prediction_score(x, n->right);
    }

    void annotate(node const* n, hyper_rectangle_t& parent_hr) {
        if (is_leaf(n)) {
            parent_hr.set_empty();
            parent_hr.score = n->score;
            m_hyper_rectangles.push_back(parent_hr);
            return;
        }
        hyper_rectangle_t l_hr = parent_hr;
        hyper_rectangle_t r_hr = parent_hr;
        feature_t feature = n->feature;
        l_hr.H[feature].second = std::min(l_hr.H[feature].second, n->threshold);
        r_hr.H[feature].first = std::max(r_hr.H[feature].first, n->threshold);
        annotate(n->left, l_hr);
        annotate(n->right, r_hr);
    }

    uint32_t parse_node(std::ifstream& in, node* n, uint32_t depth) {
        assert(n->feature == constants::invalid_feature);
        assert(n->threshold == constants::invalid_threshold);
        assert(n->score == constants::invalid_score);

        std::string line;
        std::getline(in, line);
        std::istringstream is(line);
        std::string node_type;
        is >> node_type;
        uint32_t max_depth = depth;

        /* internal node*/
        if (node_type == "SPLIT") {
            feature_t feature;
            is >> feature;
            assert(feature < m_num_features);
            float_t threshold;
            is >> threshold;
            n->feature = feature;
            n->threshold = threshold;
            n->left = new node;
            n->right = new node;
            uint32_t l_depth = parse_node(in, n->left, depth + 1);
            uint32_t r_depth = parse_node(in, n->right, depth + 1);
            max_depth = std::max(l_depth, r_depth);
        }
        /* leaf */
        else if (node_type == "LEAF_LOGARITHMIC") {
            is >> n->score;  // skip first float: the score towards label 0
            is >> n->score;  // take second float: the score towards label 1
            ++m_num_leaves;
        }

        return max_depth;
    }

    void delete_node(node const* n) {
        if (!is_leaf(n)) {
            delete_node(n->left);
            delete_node(n->right);
        }
        delete n;
    }

    void print_aux(node const* n, std::ostream& out, std::string const& identation_str) const {
        if (n) {
            if (n->left or n->right) {
                out << identation_str + "INTERNAL NODE: " << n->feature << " <= " << n->threshold
                    << std::endl;
                print_aux(n->left, out, identation_str + "\t");
                print_aux(n->right, out, identation_str + "\t");
            } else {
                out << identation_str + "LEAF NODE: " << n->score << std::endl;
            }
        }
    }
};

}  // namespace vl