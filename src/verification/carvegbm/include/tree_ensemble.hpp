#pragma once

#include "tree.hpp"

#include <algorithm>
#include <map>
#include <list>

namespace vl {

struct tree_ensemble {
    tree_ensemble() {}

    enum stable_result_t { yes, no, unknown };

    void parse(std::ifstream& in) {
        std::string line;
        std::getline(in, line);
        std::istringstream is(line);
        is >> line;
        assert(line == "classifier-forest");
        uint64_t num_trees;
        is >> num_trees;
        std::cout << "num_trees = " << num_trees << std::endl;
        m_trees.resize(num_trees);
        for (uint64_t i = 0; i != num_trees; ++i) m_trees[i].parse(in);
    }

    float_t raw_prediction_score(instance_t const& x) const {
        float_t rps = 0.0;
        for (auto const& t : m_trees) rps += t.raw_prediction_score(x);
        return rps;
    }

    template <typename InverseLinkFunction>
    std::pair<label_t, float_t> predict(instance_t const& x, InverseLinkFunction const& i,
                                        const float_t tau) const {
        const float_t rps = raw_prediction_score(x);
        if (i(rps) >= tau) return {1, rps};
        return {0, rps};
    }

    float_t solve_approx(instance_t const& x, const label_t y, const float_t p, const float_t k,
                         const float_t eps) {
        return solve_opt_problem(x, y, p, k, eps);
    }
    float_t solve_exact(instance_t const& x, const label_t y, const float_t p, const float_t k) {
        // return solve_dp_opt_problem_alt_form(x, y, p, k);
        return solve_dp_opt_problem(x, y, p, k);
    }

    template <typename InverseLinkFunction>
    stable_result_t stable(instance_t const& x, const label_t y, const float_t rps,  //
                           InverseLinkFunction const& i, const float_t tau,          //
                           const float_t p, const float_t k,                         //
                           vector<double>& stds_list,
                           const float_t eps = 0.0)                                  //
    {
        assert(y == 0 or y == 1);

        const float_t gamma = solve_opt_problem(x, y, p, k, stds_list, eps);
        cout << "RPS: " << rps << endl;
        cout << "GAMMA: " << gamma << endl;
        if (eps == 0) {
            if ((y == 0 and i(rps + gamma) < tau) or  //
                (y == 1 and i(rps - gamma) >= tau)) {
                return stable_result_t::yes;
            }
            return stable_result_t::no;
        }

        if (y == 0) {
            assert(rps < 0);
            if (i(rps + gamma / (1.0 - eps)) < tau) {
                return stable_result_t::yes;
            } else if (i(rps + gamma) >= tau) {
                return stable_result_t::no;
            }
        } else {
            assert(rps >= 0);
            if (i(rps - gamma) < tau) {
                return stable_result_t::no;
            } else if (i(rps - gamma / (1.0 - eps)) >= tau) {
                return stable_result_t::yes;
            }
        }

        return stable_result_t::unknown;
    }

    template <typename InverseLinkFunction>
    stable_result_t stable(instance_t const& x, const label_t y, const float_t rps,  //
                           InverseLinkFunction const& i, const float_t tau,          //
                           const float_t p, const float_t k,                         //
                           features_list_t attacked_feature_list, vector<double> stds_list, const float_t eps = 0.0)                                  //
    {
        assert(y == 0 or y == 1);

        const float_t gamma = solve_opt_problem(x, y, p, k, attacked_feature_list, stds_list, eps);
        cout << "RPS: " << rps << endl;
        cout << "GAMMA: " << gamma << endl;
        if (eps == 0) {
            if ((y == 0 and i(rps + gamma) < tau) or  //
                (y == 1 and i(rps - gamma) >= tau)) {
                return stable_result_t::yes;
            }
            return stable_result_t::no;
        }

        if (y == 0) {
            assert(rps < 0);
            if (i(rps + gamma / (1.0 - eps)) < tau) {
                return stable_result_t::yes;
            } else if (i(rps + gamma) >= tau) {
                return stable_result_t::no;
            }
        } else {
            assert(rps >= 0);
            if (i(rps - gamma) < tau) {
                return stable_result_t::no;
            } else if (i(rps - gamma / (1.0 - eps)) >= tau) {
                return stable_result_t::yes;
            }
        }

        return stable_result_t::unknown;
    }

    void annotate() {
        for (auto& t : m_trees) t.annotate();
    }

#ifdef DEBUG
    std::vector<float_t> optimal_delta() const {
        const uint64_t d = num_features();
        std::vector<float_t> opt_delta(d, 0.0);
        for (auto const& tree : m_trees) {
            for (uint64_t j = 0; j != tree.num_leaves(); ++j) {
                auto const& hr = tree.hyper_rectangle(j);
                auto const& delta = hr.delta;
                if (hr.is_part_of_attack) {
                    for (uint64_t i = 0; i != d; ++i) opt_delta[i] += delta[i];
                    break;  // there must be at most 1 leaf in optimal attack
                }
            }
        }
        return opt_delta;
    }
#endif

    uint64_t num_trees() const { return m_trees.size(); }

    uint64_t num_features() const {
        assert(!m_trees.empty());
        return m_trees.front().num_features();
    }

    void print(std::ostream& out) const {
        out << "Print tree ensemble, contains " << m_trees.size() << " trees" << endl;
        for (auto const& t : m_trees) t.print(out);
    }

    float_t solve_opt_problem(instance_t const& x, const label_t y,  //
                              const float_t p, const float_t k,      //
                              const float_t eps = 0.0)               //
    {
        if (p == constants::inf) {
            float_t max_gain = 0.0;
            for (auto& t : m_trees) max_gain += t.max_gain(x, y, p, k);
            return max_gain;
        }

        assert(p >= 0);

        /* No approximation. */
        if (eps == 0.0) return solve_dp_opt_problem(x, y, p, k);

        /* Approximate solution with 0 < eps < 1. */
        return solve_eps_opt_problem(x, y, p, k, eps);
    }

    float_t solve_opt_problem(instance_t const& x, const label_t y,  //
                              const float_t p, const float_t k,      //
                              features_list_t& attacked_features_list, vector<double>& stds_list, const float_t eps = 0.0)               //
    {
        if (p == constants::inf) {
            float_t max_gain = 0.0;
            for (auto& t : m_trees) max_gain += t.max_gain(x, y, p, k, attacked_features_list, stds_list);
            return max_gain;
        }

        assert(p >= 0);

        /* No approximation. */
        cout << "NO APPROX" << endl;
        if (eps == 0.0) return solve_dp_opt_problem(x, y, p, k, attacked_features_list, stds_list);

        /* Approximate solution with 0 < eps < 1. */
        return solve_eps_opt_problem(x, y, p, k, eps);
    }

    float_t solve_opt_problem(instance_t const& x, const label_t y,  //
                              const float_t p, const float_t k,      //
                              vector<double>& stds_list,
                              const float_t eps = 0.0)               //
    {
        if (p == constants::inf) {
            float_t max_gain = 0.0;
            for (auto& t : m_trees) max_gain += t.max_gain(x, y, p, k, stds_list);
            return max_gain;
        }

        assert(p >= 0);

        /* No approximation. */
        if (eps == 0.0) return solve_dp_opt_problem(x, y, p, k, stds_list);

        /* Approximate solution with 0 < eps < 1. */
        return solve_eps_opt_problem(x, y, p, k, eps);
    }

private:
    std::vector<tree> m_trees;

    float_t solve_dp_opt_problem(instance_t const& x, const label_t y,  //
                                 const float_t p, const float_t k, features_list_t& attacked_features_list, vector<double>& stds_list)      //
    {
        cout << "WITH ATTACKED FLIST" << endl;
        const uint64_t K = p == 0 ? k : eta(pow(k, p), constants::scaling_factor);
        const uint64_t m = num_trees();
        constexpr bool sort_norms = true;
        constexpr bool sort_gains = false;
        constexpr bool scale_gains = false;
        for (auto& tree : m_trees) {
            tree.compute_norms(x, p, sort_norms, attacked_features_list, stds_list);
            tree.compute_gains(x, y, K, scale_gains, sort_gains);
#ifdef DEBUG
            tree.clear_attacks();
#endif
        }

        std::vector<std::vector<float_t>> M(m + 1, std::vector<float_t>(K + 1, 0.0));

#ifdef DEBUG
        /* Temporary data to reconstruct the optimal DP solution. */
        std::vector<std::vector<float_t>> V(
            m, std::vector<float_t>(K + 1, 0.0));  // best value per tree
        std::vector<std::vector<float_t>> W(
            m, std::vector<float_t>(K + 1, 0.0));  // best weight per tree
        std::vector<std::vector<uint32_t>> L(
            m, std::vector<uint32_t>(K + 1, 0));  // leaves under attack (at most 1 per tree)
#endif

        for (uint64_t i = 1; i <= m; ++i) {
            auto& tree = m_trees[i - 1];
            for (uint64_t q = 0; q <= K; ++q) {
                float_t max_q = M[i - 1][q];
#ifdef DEBUG
                uint32_t leaf_to_attack = 0;
#endif
                for (uint64_t j = 0; j != tree.num_leaves(); ++j) {
                    auto const& hr = tree.hyper_rectangle(j);
                    if (!hr.empty) {
                        if (hr.norm <= q) {
                            if (hr.gain != constants::inf and
                                M[i - 1][q - hr.norm] + hr.gain > max_q) {
                                max_q = M[i - 1][q - hr.norm] + hr.gain;
#ifdef DEBUG
                                V[i - 1][q] = hr.gain;
                                W[i - 1][q] = hr.norm;
                                leaf_to_attack = j;
#endif
                            }
                        } else {
                            break;  // since hyper-rectangles are sorted by norm
                        }
                    }
                }
                M[i][q] = max_q;
#ifdef DEBUG
                L[i - 1][q] = leaf_to_attack;
#endif
            }
        }

#ifdef DEBUG
        /* Reconstruct optimal DP solution. */
        float_t DP_sol = M[m][K];
        int w = K;
        for (int i = m; i > 0 and DP_sol > 0; --i) {
            if (DP_sol == M[i - 1][w]) {
                continue;
            } else {
                uint32_t leaf_index = L[i - 1][w];
                m_trees[i - 1].set_attack_on_leaf(leaf_index);
                DP_sol -= V[i - 1][w];
                w -= W[i - 1][w];
            }
        }
        assert(w >= 0);
#endif

        return M[m][K];
    }

    float_t solve_dp_opt_problem(instance_t const& x, const label_t y,  //
                                 const float_t p, const float_t k, vector<double>& stds_list)      //
    {
        const uint64_t K = p == 0 ? k : eta(pow(k, p), constants::scaling_factor);
        const uint64_t m = num_trees();
        constexpr bool sort_norms = true;
        constexpr bool sort_gains = false;
        constexpr bool scale_gains = false;
        for (auto& tree : m_trees) {
            tree.compute_norms(x, p, sort_norms, stds_list);
            tree.compute_gains(x, y, K, scale_gains, sort_gains);
#ifdef DEBUG
            tree.clear_attacks();
#endif
        }

        std::vector<std::vector<float_t>> M(m + 1, std::vector<float_t>(K + 1, 0.0));

#ifdef DEBUG
        /* Temporary data to reconstruct the optimal DP solution. */
        std::vector<std::vector<float_t>> V(
            m, std::vector<float_t>(K + 1, 0.0));  // best value per tree
        std::vector<std::vector<float_t>> W(
            m, std::vector<float_t>(K + 1, 0.0));  // best weight per tree
        std::vector<std::vector<uint32_t>> L(
            m, std::vector<uint32_t>(K + 1, 0));  // leaves under attack (at most 1 per tree)
#endif

        for (uint64_t i = 1; i <= m; ++i) {
            auto& tree = m_trees[i - 1];
            for (uint64_t q = 0; q <= K; ++q) {
                float_t max_q = M[i - 1][q];
#ifdef DEBUG
                uint32_t leaf_to_attack = 0;
#endif
                for (uint64_t j = 0; j != tree.num_leaves(); ++j) {
                    auto const& hr = tree.hyper_rectangle(j);
                    if (!hr.empty) {
                        if (hr.norm <= q) {
                            if (hr.gain != constants::inf and
                                M[i - 1][q - hr.norm] + hr.gain > max_q) {
                                max_q = M[i - 1][q - hr.norm] + hr.gain;
#ifdef DEBUG
                                V[i - 1][q] = hr.gain;
                                W[i - 1][q] = hr.norm;
                                leaf_to_attack = j;
#endif
                            }
                        } else {
                            break;  // since hyper-rectangles are sorted by norm
                        }
                    }
                }
                M[i][q] = max_q;
#ifdef DEBUG
                L[i - 1][q] = leaf_to_attack;
#endif
            }
        }

#ifdef DEBUG
        /* Reconstruct optimal DP solution. */
        float_t DP_sol = M[m][K];
        int w = K;
        for (int i = m; i > 0 and DP_sol > 0; --i) {
            if (DP_sol == M[i - 1][w]) {
                continue;
            } else {
                uint32_t leaf_index = L[i - 1][w];
                m_trees[i - 1].set_attack_on_leaf(leaf_index);
                DP_sol -= V[i - 1][w];
                w -= W[i - 1][w];
            }
        }
        assert(w >= 0);
#endif

        return M[m][K];
    }

    float_t solve_dp_opt_problem(instance_t const& x, const label_t y,  //
                                 const float_t p, const float_t k)      //
    {
        const uint64_t K = p == 0 ? k : eta(pow(k, p), constants::scaling_factor);
        const uint64_t m = num_trees();
        constexpr bool sort_norms = true;
        constexpr bool sort_gains = false;
        constexpr bool scale_gains = false;
        for (auto& tree : m_trees) {
            tree.compute_norms(x, p, sort_norms);
            tree.compute_gains(x, y, K, scale_gains, sort_gains);
#ifdef DEBUG
            tree.clear_attacks();
#endif
        }

        std::vector<std::vector<float_t>> M(m + 1, std::vector<float_t>(K + 1, 0.0));

#ifdef DEBUG
        /* Temporary data to reconstruct the optimal DP solution. */
        std::vector<std::vector<float_t>> V(
            m, std::vector<float_t>(K + 1, 0.0));  // best value per tree
        std::vector<std::vector<float_t>> W(
            m, std::vector<float_t>(K + 1, 0.0));  // best weight per tree
        std::vector<std::vector<uint32_t>> L(
            m, std::vector<uint32_t>(K + 1, 0));  // leaves under attack (at most 1 per tree)
#endif

        for (uint64_t i = 1; i <= m; ++i) {
            auto& tree = m_trees[i - 1];
            for (uint64_t q = 0; q <= K; ++q) {
                float_t max_q = M[i - 1][q];
#ifdef DEBUG
                uint32_t leaf_to_attack = 0;
#endif
                for (uint64_t j = 0; j != tree.num_leaves(); ++j) {
                    auto const& hr = tree.hyper_rectangle(j);
                    if (!hr.empty) {
                        if (hr.norm <= q) {
                            if (hr.gain != constants::inf and
                                M[i - 1][q - hr.norm] + hr.gain > max_q) {
                                max_q = M[i - 1][q - hr.norm] + hr.gain;
#ifdef DEBUG
                                V[i - 1][q] = hr.gain;
                                W[i - 1][q] = hr.norm;
                                leaf_to_attack = j;
#endif
                            }
                        } else {
                            break;  // since hyper-rectangles are sorted by norm
                        }
                    }
                }
                M[i][q] = max_q;
#ifdef DEBUG
                L[i - 1][q] = leaf_to_attack;
#endif
            }
        }

#ifdef DEBUG
        /* Reconstruct optimal DP solution. */
        float_t DP_sol = M[m][K];
        int w = K;
        for (int i = m; i > 0 and DP_sol > 0; --i) {
            if (DP_sol == M[i - 1][w]) {
                continue;
            } else {
                uint32_t leaf_index = L[i - 1][w];
                m_trees[i - 1].set_attack_on_leaf(leaf_index);
                DP_sol -= V[i - 1][w];
                w -= W[i - 1][w];
            }
        }
        assert(w >= 0);
#endif

        return M[m][K];
    }

    /*
        Returns a solution that is a (1-eps)-approximation of the optimal solution.
    */
    float_t solve_eps_opt_problem(instance_t const& x, const label_t y,  //
                                  const float_t p, const float_t k,
                                  const float_t eps)  //
    {
        assert(p >= 0);
        assert(eps > 0.0);
        if (eps >= 1.0) throw std::runtime_error("Error: eps must be < 1.0");

        const uint64_t K = p == 0 ? k : eta(pow(k, p), constants::scaling_factor);
        const uint64_t m = num_trees();

        uint64_t C = 0;  // upper bound on profit
        uint64_t max_ensemble_gain = 0;
        constexpr bool sort_norms = false;
        constexpr bool sort_gains = true;
        constexpr bool scale_gains = true;
        for (auto& tree : m_trees) {
            tree.compute_norms(x, p, sort_norms);
            auto max_tree_gain = tree.compute_gains(x, y, K, scale_gains, sort_gains);
            C += max_tree_gain;
            if (max_tree_gain > max_ensemble_gain) max_ensemble_gain = max_tree_gain;
        }

        if (C == 0) return 0;

        /* Divide all gains by t and recompute upper bound on profit. */
        C = 0;
        const float_t t = std::max(1.0, (eps * max_ensemble_gain) / m);
        for (auto& tree : m_trees) C += tree.divide_gains_by_t(t);

        std::vector<std::vector<float_t>> M(m + 1, std::vector<float_t>(C + 1, constants::inf));
        M[0][0] = 0.0;

        for (uint64_t i = 1; i <= m; ++i) {
            auto const& tree = m_trees[i - 1];
            for (uint64_t q = 0; q <= C; ++q) {
                float_t min_q = M[i - 1][q];
                for (uint64_t j = 0; j != tree.num_leaves(); ++j) {
                    auto const& hr = tree.hyper_rectangle(j);
                    if (!hr.empty) {
                        if (hr.gain <= q) {
                            if (hr.norm != constants::inf and
                                M[i - 1][q - hr.gain] + hr.norm < min_q) {
                                min_q = M[i - 1][q - hr.gain] + hr.norm;
                            }
                        } else {
                            break;  // since hyper-rectangles are sorted by gain
                        }
                    }
                }
                M[i][q] = min_q <= std::min(float_t(K), M[i - 1][q]) ? min_q : M[i - 1][q];
            }
        }

        /* Compute DP solution by scanning M[m]. */
        float_t dp_sol = 0;
        for (uint64_t i = 0; i <= C; ++i) {
            if (M[m][i] != constants::inf) dp_sol = i;
        }

        /* Scale back. */
        dp_sol *= t;

        return eta_inv(dp_sol, constants::scaling_factor);
    }

    /* Alternative form of the optimization problem. */
    // float_t solve_dp_opt_problem_alt_form(instance_t const& x, const label_t y,  //
    //                                       const float_t p, const float_t k)      //
    // {
    //     assert(p >= 0);

    //     const uint64_t K = p == 0 ? k : eta(pow(k, p), constants::scaling_factor);
    //     const uint64_t m = num_trees();

    //     uint64_t C = 0;  // upper bound on profit
    //     constexpr bool sort_norms = false;
    //     constexpr bool sort_gains = true;
    //     constexpr bool scale_gains = true;
    //     for (auto& tree : m_trees) {
    //         tree.compute_norms(x, p, sort_norms);
    //         C += tree.compute_gains(x, y, K, scale_gains, sort_gains);
    //     }

    //     if (C == 0) return 0;

    //     std::vector<std::vector<float_t>> M(m + 1, std::vector<float_t>(C + 1, constants::inf));
    //     M[0][0] = 0.0;

    //     for (uint64_t i = 1; i <= m; ++i) {
    //         auto const& tree = m_trees[i - 1];
    //         for (uint64_t q = 0; q <= C; ++q) {
    //             float_t min_q = M[i - 1][q];
    //             for (uint64_t j = 0; j != tree.num_leaves(); ++j) {
    //                 auto const& hr = tree.hyper_rectangle(j);
    //                 if (!hr.empty) {
    //                     if (hr.gain <= q) {
    //                         if (hr.norm != constants::inf and
    //                             M[i - 1][q - hr.gain] + hr.norm < min_q) {
    //                             min_q = M[i - 1][q - hr.gain] + hr.norm;
    //                         }
    //                     } else {
    //                         break;  // since hyper-rectangles are sorted by gain
    //                     }
    //                 }
    //             }
    //             M[i][q] = min_q <= std::min(float_t(K), M[i - 1][q]) ? min_q : M[i - 1][q];
    //         }
    //     }

    //     /* Compute DP solution by scanning M[m]. */
    //     uint64_t dp_sol = 0;
    //     for (uint64_t i = 0; i != C + 1; ++i) {
    //         if (M[m][i] != constants::inf) dp_sol = i;
    //     }

    //     return eta_inv(dp_sol, constants::scaling_factor);
    // }
};

}  // namespace vl
