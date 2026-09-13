#include "flyquant/controls.hpp"

#include <algorithm>
#include <random>
#include <stdexcept>
#include <unordered_set>

namespace flyquant {
namespace {

std::uint64_t pair_key(std::size_t a, std::size_t b) {
  return (static_cast<std::uint64_t>(a) << 32U) ^ static_cast<std::uint64_t>(b);
}

std::vector<Neuron> copy_neurons(const ConnectomeGraph& g) { return g.neurons(); }

}  // namespace

ConnectomeGraph degree_preserving_shuffle(const ConnectomeGraph& graph, std::uint64_t seed,
                                          std::size_t attempted_swaps_per_edge,
                                          WeightTransform transform) {
  auto edges = graph.edges();
  if (edges.size() < 2) {
    return ConnectomeGraph::from_parts(copy_neurons(graph), std::move(edges), transform);
  }

  std::unordered_set<std::uint64_t> occupied;
  for (const auto& e : edges) occupied.insert(pair_key(e.source, e.target));

  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<std::size_t> pick(0, edges.size() - 1);
  const std::size_t attempts = attempted_swaps_per_edge * edges.size();
  for (std::size_t k = 0; k < attempts; ++k) {
    const std::size_t i = pick(rng);
    const std::size_t j = pick(rng);
    if (i == j) continue;
    const auto a = edges[i].source;
    const auto b = edges[i].target;
    const auto c = edges[j].source;
    const auto d = edges[j].target;
    if (a == d || c == b) continue;
    if (a == c || b == d) continue;
    const auto old1 = pair_key(a, b);
    const auto old2 = pair_key(c, d);
    const auto new1 = pair_key(a, d);
    const auto new2 = pair_key(c, b);
    if ((new1 != old1 && new1 != old2 && occupied.contains(new1)) ||
        (new2 != old1 && new2 != old2 && occupied.contains(new2))) continue;

    occupied.erase(old1);
    occupied.erase(old2);
    edges[i].target = d;
    edges[j].target = b;
    occupied.insert(new1);
    occupied.insert(new2);
  }

  return ConnectomeGraph::from_parts(copy_neurons(graph), std::move(edges), transform);
}

ConnectomeGraph random_graph_control(const ConnectomeGraph& graph, std::uint64_t seed,
                                     WeightTransform transform) {
  const auto n = graph.neurons().size();
  const auto m = graph.edges().size();
  if (m == 0) return ConnectomeGraph::from_parts(copy_neurons(graph), {}, transform);
  if (n < 2) throw std::runtime_error("cannot generate non-self random graph");
  const std::uint64_t max_pairs = static_cast<std::uint64_t>(n) * static_cast<std::uint64_t>(n - 1);
  if (static_cast<std::uint64_t>(m) > max_pairs) throw std::runtime_error("too many edges for simple random graph");

  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<std::size_t> node(0, n - 1);
  std::unordered_set<std::uint64_t> occupied;
  std::vector<Edge> edges;
  edges.reserve(m);

  std::vector<Edge> attributes = graph.edges();
  std::shuffle(attributes.begin(), attributes.end(), rng);
  while (edges.size() < m) {
    const auto src = node(rng);
    const auto dst = node(rng);
    if (src == dst) continue;
    if (!occupied.insert(pair_key(src, dst)).second) continue;
    auto e = attributes[edges.size()];
    e.source = src;
    e.target = dst;
    edges.push_back(e);
  }
  return ConnectomeGraph::from_parts(copy_neurons(graph), std::move(edges), transform);
}

}  // namespace flyquant
