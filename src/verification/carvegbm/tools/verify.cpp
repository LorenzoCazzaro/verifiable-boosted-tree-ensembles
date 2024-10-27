#include <iostream>
#include <random>
#include <chrono>
#include <sys/resource.h>
#include <cstring>
#include <cmath>

#include "tree_ensemble.hpp"
#include "inverse_link_functions.hpp"
#include "../external/include/parser.hpp"
#include <iomanip>

using namespace vl;

typedef std::chrono::high_resolution_clock clock_type;

void parse_row_in_csv(std::ifstream& in, vector<double>& feature_list) {
    std::string line;
    std::getline(in, line);
    std::istringstream is(line);
    std::string str;
    while (std::getline(is, str, ',')) feature_list.push_back(std::stod(str));
}

void parse_row_in_csv(std::ifstream& in, features_list_t& feature_list) {
    std::string line;
    std::getline(in, line);
    std::istringstream is(line);
    std::string str;
    while (std::getline(is, str, ',')) feature_list.push_back(std::stoul(str));
}

void parse_row_in_csv(std::ifstream& in, std::string& line, instance_t& x, label_t& y) {
    std::getline(in, line);
    std::istringstream is(line);
    std::string str;
    x.clear();
    std::getline(is, str, ',');
    y = std::stof(str);
    while (std::getline(is, str, ',')) x.push_back(std::stof(str));
}

int main(int argc, char** argv) {
    cmd_line_parser::parser parser(argc, argv);
    parser.add("model_filename", "Model filename: must be a .silva file.", "-i", true);
    parser.add("testset_filename", "Testset filename: must be a .csv file.", "-t", true);
    parser.add("p", "Determines the norm used.", "-p", true);
    parser.add("k", "Determines the strength of the attacker.", "-k", true);
    parser.add("eps",
               "Tolerance threshold for approximate solution. The number eps should be a real "
               "number in [0,1). Eps = 0 by default (exact solution).",
               "-e", false);
    parser.add(
        "verbose",
        "Print query status: ROBUST, VULNERABLE, FRAGILE, BROKEN, or UNKNOWN. (Default is false.)",
        "--verbose", false, true);
    parser.add("index_of_instance", "Index of the instance of the dataset to verify.", "-ioi", true);
    parser.add("subset_attacked_features_flag", "Flag for verifying the robustness against an attacker that can manipulate only a subset of features", "--saff", false, true);
    parser.add("attacked_features_list_filename", "List of the features to be considered as attacked: must be a .csv file containing only a row.", "--afl", false);
    parser.add("chen_stability", "Verify Chen Stability", "--cs", false, true);
    parser.add("chen_high_confidence", "Verify Chen High Confidence", "--chc", false, true);
    parser.add("chen_small_neighborhood", "Verify Chen Small Neighborhood", "--csn", false, true);
    parser.add("score_variation", "Maximum score variation for Chen Stability and Small Neighborhood", "--c", false);
    parser.add("high_conf_prob", "Minimum probability for Chen High Confidence", "--prob", false);
    parser.add("stds_list", "List of standard deviations of feature values", "--stdl", false);
    if (!parser.parse()) return 1;

    auto model_filename = parser.get<std::string>("model_filename");
    auto testset_filename = parser.get<std::string>("testset_filename");
    
    vl::float_t p = 0.0;
    if (parser.get<std::string>("p") == "inf") {
        p = constants::inf;
    } else {
        p = parser.get<vl::float_t>("p");
    }
    std::cout << "p: " << p << std::endl;

    const vl::float_t k = parser.get<vl::float_t>("k") + 0.00001f;
    std::cout << std::setprecision(20) << "k: " << k << std::endl;

    vl::float_t eps = 0.0;
    if (parser.parsed("eps")) eps = parser.get<vl::float_t>("eps");
    std::cout << "eps: " << eps << std::endl;

    auto stds_list_filename = parser.get<std::string>("stds_list");

#ifdef DEBUG
    if (eps != 0) {
        std::cerr << "\n### WARNING: Sanity checks are only available for exact solutions (eps "
                     "must be 0). Recompile in Release mode or set eps to 0 to enable them.\n"
                  << std::endl;
    }
#endif

    bool verbose = false;
    if (parser.parsed("verbose")) verbose = true;

    int ioi = parser.get<int>("index_of_instance");
    std::cout << "Index of instance on which verify the model: " << ioi << std::endl;

    bool saff = false;
    string attacked_features_list_filename = "";
    if (parser.parsed("subset_attacked_features_flag")) {
        saff = true;
        cout << "SAFF" << endl;
        attacked_features_list_filename = parser.get<std::string>("attacked_features_list_filename");
        cout << "AFL: " << attacked_features_list_filename << endl;
    }
    
    bool cs = false;
    vl::float_t c = 0.0;
    if (parser.parsed("chen_stability")) {
        cs = true;
        cout << "CS" << endl;
        c = parser.get<vl::float_t>("score_variation");
        cout << "C: " << c << endl;
    }

    bool chc = false;
    vl::float_t prob = 0.0;
    if (parser.parsed("chen_high_confidence")) {
        chc = true;
        prob = parser.get<vl::float_t>("high_conf_prob");
    }

    bool csn = false;
    if (parser.parsed("chen_small_neighborhood")) {
        csn = true;
        c = parser.get<vl::float_t>("score_variation");
    }

    tree_ensemble T;

    {
        /* 1. load the model */
        auto start = clock_type::now();
        std::ifstream in(model_filename);
        T.parse(in);
        in.close();
        auto stop = clock_type::now();
        std::cout << "1. loading model: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count()
                  << " [msec]" << std::endl;
    }
    {
        /* 2. annotate */
        auto start = clock_type::now();
        T.annotate();
        auto stop = clock_type::now();
        std::cout << "2. annotate: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count()
                  << " [msec]" << std::endl;
    }
    {
        /* 3. test */
        inverse_link_function_type inv_link_funct;
        vl::float_t tau;

        std::ifstream in(testset_filename);
        std::string line;

        std::getline(in, line);
        std::istringstream is(line);
        uint64_t num_queries = 0;
        uint64_t num_features = 0;
        is >> line;  // skip #
        is >> num_queries;
        if (ioi >= 0) num_queries = 1;
        is >> num_features;
        assert(num_features == T.num_features());

        vector<instance_t> queries;
        vector<label_t> labels;

        if (ioi == -1) {
            instance_t x;
            label_t y;
            x.reserve(num_features);
            for (uint64_t i = 0; i != num_queries; ++i) {
                parse_row_in_csv(in, line, x, y);
                queries.push_back(x);
                labels.push_back(y);
            }
        } else {
            instance_t x;
            label_t y;
            x.reserve(num_features);
            for (int i = 0; i != ioi; ++i) std::getline(in, line);
            parse_row_in_csv(in, line, x, y);
            queries.push_back(x);
            labels.push_back(y);
        }

        cout << queries.size() << endl;

        features_list_t attacked_feature_list;
        if (saff) {
            /*Extract list of attacked features*/
            std::ifstream in_afl(attacked_features_list_filename);
            parse_row_in_csv(in_afl, attacked_feature_list);
        }

        cout << "Feature list size: " << attacked_feature_list.size() << endl;
        cout << "Attacked feats list: ";
        for(auto iter = attacked_feature_list.begin(); iter != attacked_feature_list.end(); iter++)
            cout << *iter << ", ";
        cout << endl;
        
        std::vector<double> stds_list;
        if (csn and stds_list_filename != "") {
            /*Extract list of attacked features*/
            std::ifstream in_stdsl(stds_list_filename);
            parse_row_in_csv(in_stdsl, stds_list);
        } else {
            stds_list = vector<double>(num_features, 1);
        }

        cout << "Stds list: ";
        for(auto iter = stds_list.begin(); iter != stds_list.end(); iter++)
            cout << *iter << ", ";
        cout << endl;

        in.close();

        std::cout << "performing " << num_queries << " queries..." << std::endl;
        
        using stable_result_t = vl::tree_ensemble::stable_result_t;

        double total_verification_time = 0.0;

        if (!cs & !chc & !csn) {
            uint64_t n_correct = 0;
            uint64_t n_stable = 0;
            uint64_t n_unstable = 0;
            uint64_t n_unknown = 0;
            uint64_t n_robust = 0;
            uint64_t n_fragile = 0;
            uint64_t true_positive = 0;
            uint64_t true_negative = 0;
            uint64_t false_positive = 0;
            uint64_t false_negative = 0;

            for (uint64_t i = 0; i != num_queries; ++i) {
                auto start_x_sample = clock_type::now();
                auto const& x = queries[i];
                auto const y = labels[i];
                auto [y_pred, raw_pred_score] = T.predict(x, inv_link_funct, tau);

                stable_result_t stable_result;
                bool same_prediction = y_pred == y;

                tau = 0.0;
                //since the side to verify is only one here, we use stable_result_less_side
                stable_result = !saff ? 
                    T.stable(x, y_pred, raw_pred_score, inv_link_funct, tau, p, k, stds_list, eps) :
                    T.stable(x, y_pred, raw_pred_score, inv_link_funct, tau, p, k, attacked_feature_list, stds_list, eps);

                n_correct += same_prediction;

                //positive is y==1, negative is y==0
                //cout << y << endl;
                //cout << y_pred << endl;
                true_positive += (y == 1) && (y == y_pred);
                false_positive += (y == 0) && (y != y_pred);
                true_negative += (y == 0) && (y == y_pred);
                false_negative += (y == 1) && (y != y_pred);

                bool stable = stable_result == stable_result_t::yes;
                bool unstable = stable_result == stable_result_t::no;
                bool unknown = stable_result == stable_result_t::unknown;

                n_stable += stable;
                n_unstable += unstable;
                n_unknown += unknown;

                n_robust += same_prediction && stable;
                n_fragile += same_prediction && unstable;
                
                auto end_x_sample = clock_type::now();

                double verification_time_x_instance =
                    std::chrono::duration_cast<std::chrono::microseconds>(end_x_sample - start_x_sample)
                        .count();

                total_verification_time += verification_time_x_instance;

                if (verbose) {
                    std::cout << "query " << i << ": ";
                    cout << "RAW SCORE: " << raw_pred_score << ", ";
                    cout << "PROB: " << 1/(1+exp(-raw_pred_score)) << ", ";
                    std::cout << "pred label " << y_pred << ", true label " << y << ", STATUS: ";
                    std::cout << (stable ? (same_prediction ? "ROBUST" : "VULNERABLE")
                                        : (unstable ? (same_prediction ? "FRAGILE" : "BROKEN")
                                                    : "UNKNOWN"))
                            << std::endl;
                    std::cout << "Time required per query " << i << ": " << verification_time_x_instance
                            << " [msec]" << std::endl;
                }

            #ifdef DEBUG
                if (eps == 0) {
                    const uint64_t d = T.num_features();
                    auto const& opt_delta = T.optimal_delta();
                    instance_t attack(d, 0.0);
                    for (uint64_t i = 0; i != d; ++i) attack[i] = x[i] + opt_delta[i];
                    assert(norm(opt_delta, p) <= k + 0.001f);

                    auto [y_pred_with_optimal_attack, r] = T.predict(attack, inv_link_funct, tau);

                    if (stable) {
                        if (!same_prediction) {  // VULNERABLE
                            if (y_pred_with_optimal_attack == y) {
                                std::cerr << " ##> Error: opt attack should have changed prediction !"
                                        << std::endl;
                                assert(false);
                            }
                        } else {  // ROBUST

                            if (y == 0) {
                                if (r < raw_pred_score) {
                                    std::cerr << "query " << i << ": ";
                                    std::cerr << "true label " << (y ? "+1" : "-1")
                                            << ": score after attack = " << r
                                            << " vs. score pre attack = " << raw_pred_score << "\n"
                                            << std::endl;
                                }
                                assert(r >= raw_pred_score);
                            } else if (y == 1) {
                                if (r > raw_pred_score) {
                                    std::cerr << "query " << i << ": ";
                                    std::cerr << "true label " << (y ? "+1" : "-1")
                                            << ": score after attack = " << r
                                            << " vs. score pre attack = " << raw_pred_score << "\n"
                                            << std::endl;
                                }
                                assert(r <= raw_pred_score);
                            }

                            if (y_pred_with_optimal_attack != y) {
                                std::cerr
                                    << " ==> Error: opt attack should have kept the same prediction !"
                                    << std::endl;
                                assert(false);
                            }
                        }
                    } else if (unstable) {
                        if (!same_prediction) {  // BROKEN
                            if (y_pred_with_optimal_attack != y) {
                                std::cerr
                                    << " **> Error: opt attack should have kept the same prediction !"
                                    << std::endl;
                            }
                        } else {  // FRAGILE
                            if (y_pred_with_optimal_attack == y) {
                                std::cerr << " --> Error: opt attack should have changed prediction !"
                                        << std::endl;
                            }
                        }
                    }
                }
            #endif
            }

            std::cout << "3. test " << num_queries << " queries: " << total_verification_time
                    << " [microsec] (" << total_verification_time / num_queries << " microsec/query)"
                    << std::endl;

            std::cout << "n queries correctly classified: " << n_correct << std::endl;
            std::cout << "accuracy: " << (n_correct * 100.0) / num_queries << "%" << std::endl;
            std::cout << "robustness: " << (n_robust * 100.0) / num_queries << "%" << std::endl;
            std::cout << "n robust queries: " << n_robust << std::endl;
            std::cout << "n fragile queries: " << n_fragile << std::endl;
            std::cout << "n vulnerable queries: " << n_stable - n_robust << std::endl;
            std::cout << "n broken queries: " << n_unstable - n_fragile << std::endl;
            std::cout << "n unknown queries: " << n_unknown << " (" << (n_unknown * 100.0) / num_queries
                    << "%)" << std::endl;

            std::cout << "true positive: " << true_positive
                    << std::endl;
            std::cout << "true negative: " << true_negative
                    << std::endl;
            std::cout << "false positive: " << false_positive
                    << std::endl;
            std::cout << "false negative: " << false_negative
                    << std::endl;

            assert((true_positive + false_positive + true_negative + false_negative) == num_queries);

            float true_positive_rate = (float)true_positive / (true_positive + false_negative);
            float false_positive_rate = (float)false_positive / (false_positive + true_negative);

            std::cout << "true positive rate: " << (true_positive_rate)*100
                    << "%" << std::endl;

            std::cout << "false positive rate: " << (false_positive_rate)*100
                    << "%" << std::endl;

            std::cout << "F1 score: " << ((float)(2*true_positive)/(2*true_positive + false_positive + false_negative))*100
                    << "%" << std::endl;

        } else if (cs) {

            uint64_t n_correct = 0;
            uint64_t n_satisfied = 0;

            for (uint64_t i = 0; i != num_queries; ++i) {
                auto start_x_sample = clock_type::now();
                auto const& x = queries[i];
                auto const y = labels[i];
                tau = 0.0;
                auto [y_pred, raw_pred_score] = T.predict(x, inv_link_funct, tau);
                bool same_prediction = y_pred == y;

                //cout << "TAU INIT: " << tau << endl;
                //note that it works with this tau since inv_link_funct is the identity
                tau = raw_pred_score - c;
                //cout << "TAU LESS: " << tau << endl;
                cout << "SAFF IN CS: " << saff << endl;
                auto stable_result_less_side = !saff ? 
                    T.stable(x, 1, raw_pred_score, inv_link_funct, tau, 0, k, stds_list, eps) :
                    T.stable(x, 1, raw_pred_score, inv_link_funct, tau, 0, k, attacked_feature_list, stds_list, eps);

                bool stable_less = stable_result_less_side == stable_result_t::yes;

                tau = raw_pred_score + c;
                //cout << "TAU GREATER: " << tau << endl;
                auto stable_result_great_side = !saff ? 
                    T.stable(x, 0, raw_pred_score, inv_link_funct, tau, 0, k, stds_list, eps) :
                    T.stable(x, 0, raw_pred_score, inv_link_funct, tau, 0, k, attacked_feature_list, stds_list, eps);
                
                bool stable_great = stable_result_great_side == stable_result_t::yes;

                //cout << "LESS SIDE: " << stable_less << endl;
                //cout << "GREATER SIDE: " << stable_great << endl;

                bool satisfied = stable_less && stable_great;

                n_correct += same_prediction;
                n_satisfied += satisfied;

                //positive is y==1, negative is y==0
                //cout << y << endl;
                //cout << y_pred << endl;

                auto end_x_sample = clock_type::now();

                double verification_time_x_instance =
                    std::chrono::duration_cast<std::chrono::microseconds>(end_x_sample - start_x_sample)
                        .count();

                total_verification_time += verification_time_x_instance;

                if (verbose) {
                    std::cout << "query " << i << ": ";
                    std::cout << "pred label " << y_pred << ", true label " << y;
                    std::cout << " SATISFIED: " << satisfied;
                    std::cout << " Time required per query " << i << ": " << verification_time_x_instance
                            << " [msec]" << std::endl;
                }
            }

            std::cout << "3. test " << num_queries << " queries: " << total_verification_time
                    << " [microsec] (" << total_verification_time / num_queries << " microsec/query)"
                    << std::endl;

            std::cout << "n queries correctly classified: " << n_correct << std::endl;
            std::cout << "accuracy: " << (n_correct * 100.0) / num_queries << "%" << std::endl;
            std::cout << "perc. queries on which chen stability is satisfied: " << (n_satisfied * 100.0) / num_queries << "%" << std::endl;
            std::cout << "n satisfied queries: " << n_satisfied << std::endl;

        } else if (chc) {

            uint64_t n_correct = 0;
            uint64_t n_satisfied = 0;

            for (uint64_t i = 0; i != num_queries; ++i) {
                auto start_x_sample = clock_type::now();
                auto const& x = queries[i];
                auto const y = labels[i];
                tau = 0.0;
                auto [y_pred, raw_pred_score] = T.predict(x, inv_link_funct, tau);
                bool same_prediction = y_pred == y;
                
                //apply the logit function to the high confidence prob
                double maximum_decrease_in_confidence = log(prob/(1-prob));
                cout << "MAXIMUM DECREASE IN CONFIDENCE" << maximum_decrease_in_confidence << endl;

                cout << "TAU INIT: " << tau << endl;
                //note that it works with this tau since inv_link_funct is the identity
                tau = raw_pred_score - maximum_decrease_in_confidence;
                cout << "TAU LESS: " << tau << endl;
                auto stable_result_less_side = !saff ? 
                    T.stable(x, 1, raw_pred_score, inv_link_funct, tau, 0, k, stds_list, eps) :
                    T.stable(x, 1, raw_pred_score, inv_link_funct, tau, 0, k, attacked_feature_list, stds_list, eps);

                bool stable_less = stable_result_less_side == stable_result_t::yes;

                cout << "LESS SIDE: " << stable_less << endl;

                bool satisfied = stable_less;

                n_correct += same_prediction;
                n_satisfied += satisfied;

                //positive is y==1, negative is y==0
                //cout << y << endl;
                //cout << y_pred << endl;

                auto end_x_sample = clock_type::now();

                double verification_time_x_instance =
                    std::chrono::duration_cast<std::chrono::microseconds>(end_x_sample - start_x_sample)
                        .count();

                total_verification_time += verification_time_x_instance;

                if (verbose) {
                    std::cout << "query " << i << ": ";
                    std::cout << "pred label " << y_pred << ", true label " << y;
                    std::cout << " SATISFIED: " << satisfied;
                    std::cout << " Time required per query " << i << ": " << verification_time_x_instance
                            << " [msec]" << std::endl;
                }
            }

            std::cout << "3. test " << num_queries << " queries: " << total_verification_time
                    << " [microsec] (" << total_verification_time / num_queries << " microsec/query)"
                    << std::endl;

            std::cout << "n queries correctly classified: " << n_correct << std::endl;
            std::cout << "accuracy: " << (n_correct * 100.0) / num_queries << "%" << std::endl;
            std::cout << "perc. queries on which chen stability is satisfied: " << (n_satisfied * 100.0) / num_queries << "%" << std::endl;
            std::cout << "n satisfied queries: " << n_satisfied << std::endl;

        } else if (csn) {

            uint64_t n_correct = 0;
            uint64_t n_satisfied = 0;

            for (uint64_t i = 0; i != num_queries; ++i) {
                auto start_x_sample = clock_type::now();
                auto const& x = queries[i];
                auto const y = labels[i];
                tau = 0.0;
                auto [y_pred, raw_pred_score] = T.predict(x, inv_link_funct, tau);
                bool same_prediction = y_pred == y;

                //cout << "TAU INIT: " << tau << endl;
                //note that it works with this tau since inv_link_funct is the identity
                tau = raw_pred_score - c * k;
                //cout << "TAU LESS: " << tau << endl;
                auto stable_result_less_side = !saff ? 
                    T.stable(x, 1, raw_pred_score, inv_link_funct, tau, constants::inf, k, stds_list, eps) :
                    T.stable(x, 1, raw_pred_score, inv_link_funct, tau, constants::inf, k, attacked_feature_list, stds_list, eps);

                bool stable_less = stable_result_less_side == stable_result_t::yes;

                tau = raw_pred_score + c * k;
                //cout << "TAU GREATER: " << tau << endl;
                auto stable_result_great_side = !saff ? 
                    T.stable(x, 0, raw_pred_score, inv_link_funct, tau, constants::inf, k, stds_list, eps) :
                    T.stable(x, 0, raw_pred_score, inv_link_funct, tau, constants::inf, k, attacked_feature_list, stds_list, eps);
                
                bool stable_great = stable_result_great_side == stable_result_t::yes;

                //cout << "LESS SIDE: " << stable_less << endl;
                //cout << "GREATER SIDE: " << stable_great << endl;

                bool satisfied = stable_less && stable_great;

                n_correct += same_prediction;
                n_satisfied += satisfied;

                //positive is y==1, negative is y==0
                //cout << y << endl;
                //cout << y_pred << endl;

                auto end_x_sample = clock_type::now();

                double verification_time_x_instance =
                    std::chrono::duration_cast<std::chrono::microseconds>(end_x_sample - start_x_sample)
                        .count();

                total_verification_time += verification_time_x_instance;

                if (verbose) {
                    std::cout << "query " << i << ": ";
                    std::cout << "pred label " << y_pred << ", true label " << y;
                    std::cout << " SATISFIED: " << satisfied;
                    std::cout << " Time required per query " << i << ": " << verification_time_x_instance
                            << " [msec]" << std::endl;
                }
            }

            std::cout << "3. test " << num_queries << " queries: " << total_verification_time
                    << " [microsec] (" << total_verification_time / num_queries << " microsec/query)"
                    << std::endl;

            std::cout << "n queries correctly classified: " << n_correct << std::endl;
            std::cout << "accuracy: " << (n_correct * 100.0) / num_queries << "%" << std::endl;
            std::cout << "perc. queries on which chen small neighbourhood is satisfied: " << (n_satisfied * 100.0) / num_queries << "%" << std::endl;
            std::cout << "n satisfied queries: " << n_satisfied << std::endl;

        }
    }

    return 0;
}
