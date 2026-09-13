#pragma once

#include "flyquant/connectome.hpp"

#include <cstddef>
#include <cstdint>

namespace flyquant {

ConnectomeGraph degree_preserving_shuffle(const ConnectomeGraph& graph, std::uint64_t seed,
                                          std::size_t attempted_swaps_per_edge = 20);
ConnectomeGraph random_graph_control(const ConnectomeGraph& graph, std::uint64_t seed);

}  // namespace flyquant
