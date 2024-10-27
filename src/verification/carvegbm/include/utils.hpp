#pragma once

#include <vector>
#include <cassert>
#include <cmath>
#include <fstream>
#include <string>
#include <sstream>
#include <algorithm>

#include <iostream>

using namespace std;

namespace vl {

typedef uint32_t feature_t;
typedef double float_t;
typedef int label_t;
typedef std::vector<uint32_t> features_list_t;

typedef std::vector<float_t> instance_t;

namespace constants {

constexpr uint64_t max_num_digits = 5;
const uint64_t scaling_factor = pow(10, constants::max_num_digits);
constexpr feature_t invalid_feature = feature_t(-1);
constexpr float_t invalid_threshold = float_t(-1);
constexpr float_t invalid_score = float_t(-1);

/* Note: throughout the code, we assume valid labels are 0 and 1. */
constexpr label_t invalid_label = label_t(-1);

constexpr float_t inf = std::numeric_limits<float_t>::max();

}  // namespace constants

struct hyper_rectangle_t {
    hyper_rectangle_t()
        : score(0.0)
        , norm(constants::inf)
        , gain(constants::inf)
#ifdef DEBUG
        , is_part_of_attack(false)
#endif
        , empty(false)  //
    {
    }

    float_t score;
    float_t norm;  // ||dist(x,H)||_p
    float_t gain;

#ifdef DEBUG
    bool is_part_of_attack;
    std::vector<float_t> delta;
#endif

    bool empty;
    std::vector<std::pair<float_t, float_t>> H;

    void set_empty() {
        for (uint32_t i = 0; i < H.size(); i++) {
            if (H[i].first >= H[i].second) {
                empty = true;
                return;
            }
        }
    }
};

static uint64_t eta(const float_t x, const uint64_t scaling_factor) {
    return uint64_t(x * scaling_factor);
}

static float_t eta_inv(const uint64_t x, const uint64_t scaling_factor) {
    return float_t(x) / scaling_factor;
}

double norm(instance_t const& x, const float_t p) {
    if (p == 0) {
        double ret = 0.0;
        for (auto x_i : x) {
            if (x_i == constants::inf)
                return constants::inf;
            else
                ret += x_i != 0;
        }
        return ret;
    } else if (p != constants::inf) {
        double ret = 0.0;
        for (auto x_i : x) ret += pow(abs(x_i), p);
        return pow(ret, 1.0 / p);
    } else {
        double ret = 0.0;
        for (auto x_i : x) ret = std::max<float_t>(abs(x_i), ret);
        return ret;
    }
}

}  // namespace vl