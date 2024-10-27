#pragma once

namespace vl {

struct identity {
    vl::float_t operator()(const vl::float_t score) const { return score; }
};

typedef identity inverse_link_function_type;

}  // namespace vl