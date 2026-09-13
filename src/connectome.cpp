#include "flyquant/connectome.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace flyquant {
namespace {

std::vector<std::string> split_simple_csv(const std::string& line) {
  std::vector<std::string> out;
  std::stringstream ss(line);
  std::string item;
  while (std::getline(ss, item, ',')) out.push_back(item);
  return out;
}

SynapticSign parse_sign(const std::string& s) {
  if (s == "1" || s == "+1" || s == "E" || s == "EXCITATORY") return SynapticSign::Excitatory;
  if (s == "-1" || s == "I" || s == "INHIBITORY") return SynapticSign::Inhibitory;
  if (s.empty() || s == "0" || s == "U" || s == "UNKNOWN") return SynapticSign::Unknown;
  throw std::runtime_error("invalid synaptic sign: " + s);
}

}  // namespace

ConnectomeGraph ConnectomeGraph::load_csv(const std::filesystem::path& neurons_path,
                                          const std::filesystem::path& edges_path,
                                          WeightTransform transform) {
  std::ifstream nin(neurons_path);
  if (!nin) throw std::runtime_error("cannot open neurons CSV: " + neurons_path.string());
  std::string line;
  if (!std::getline(nin, line) || line != "id,type,region,neurotransmitter") {
    throw std::runtime_error("neurons CSV header must be id,type,region,neurotransmitter");
  }
  std::vector<Neuron> neurons;
  while (std::getline(nin, line)) {
    if (line.empty()) continue;
    const auto f = split_simple_csv(line);
    if (f.size() != 4) throw std::runtime_error("invalid neurons CSV row");
    neurons.push_back({std::stoull(f[0]), f[1], f[2], f[3], 0.0});
  }

  std::unordered_map<NeuronId, std::size_t> map;
  for (std::size_t i = 0; i < neurons.size(); ++i) {
    if (!map.emplace(neurons[i].id, i).second) throw std::runtime_error("duplicate neuron ID");
  }

  std::ifstream ein(edges_path);
  if (!ein) throw std::runtime_error("cannot open edges CSV: " + edges_path.string());
  if (!std::getline(ein, line) || line != "source_id,target_id,synapse_count,sign") {
    throw std::runtime_error("edges CSV header must be source_id,target_id,synapse_count,sign");
  }
  std::vector<Edge> edges;
  std::unordered_set<std::string> pairs;
  while (std::getline(ein, line)) {
    if (line.empty()) continue;
    const auto f = split_simple_csv(line);
    if (f.size() != 4) throw std::runtime_error("invalid edges CSV row");
    const auto src_id = std::stoull(f[0]);
    const auto dst_id = std::stoull(f[1]);
    const auto src = map.find(src_id);
    const auto dst = map.find(dst_id);
    if (src == map.end() || dst == map.end()) throw std::runtime_error("edge references unknown neuron");
    if (src->second == dst->second) throw std::runtime_error("self-edge found in normalized graph");
    const auto count = static_cast<std::uint32_t>(std::stoul(f[2]));
    if (count == 0) throw std::runtime_error("zero synapse_count edge");
    const auto key = std::to_string(src_id) + ":" + std::to_string(dst_id);
    if (!pairs.insert(key).second) throw std::runtime_error("duplicate source/target pair in normalized graph");
    edges.push_back({src->second, dst->second, count, parse_sign(f[3]), 0.0});
  }
  return from_parts(std::move(neurons), std::move(edges), transform);
}

ConnectomeGraph ConnectomeGraph::from_parts(std::vector<Neuron> neurons, std::vector<Edge> edges,
                                            WeightTransform transform) {
  ConnectomeGraph g;
  g.neurons_ = std::move(neurons);
  g.edges_ = std::move(edges);
  g.rebuild(transform);
  return g;
}

std::size_t ConnectomeGraph::neuron_index(NeuronId id) const {
  const auto it = id_to_index_.find(id);
  if (it == id_to_index_.end()) throw std::runtime_error("unknown neuron ID: " + std::to_string(id));
  return it->second;
}

std::vector<std::size_t> ConnectomeGraph::in_degrees() const {
  std::vector<std::size_t> d(neurons_.size(), 0);
  for (const auto& e : edges_) ++d[e.target];
  return d;
}

std::vector<std::size_t> ConnectomeGraph::out_degrees() const {
  std::vector<std::size_t> d(neurons_.size(), 0);
  for (const auto& e : edges_) ++d[e.source];
  return d;
}

void ConnectomeGraph::rebuild(WeightTransform transform) {
  id_to_index_.clear();
  for (std::size_t i = 0; i < neurons_.size(); ++i) {
    if (!id_to_index_.emplace(neurons_[i].id, i).second) throw std::runtime_error("duplicate neuron ID");
  }

  std::vector<double> incoming_magnitude(neurons_.size(), 0.0);
  for (const auto& e : edges_) {
    if (e.source >= neurons_.size() || e.target >= neurons_.size()) throw std::runtime_error("edge index out of range");
    incoming_magnitude[e.target] += std::log1p(static_cast<double>(e.synapse_count));
  }

  for (auto& e : edges_) {
    const double magnitude = std::log1p(static_cast<double>(e.synapse_count));
    const double denom = incoming_magnitude[e.target] > 0.0 ? incoming_magnitude[e.target] : 1.0;
    double sign = 1.0;
    if (transform == WeightTransform::SignedLog1pIncomingNormalized) {
      if (e.sign == SynapticSign::Inhibitory) sign = -1.0;
      else if (e.sign == SynapticSign::Excitatory) sign = 1.0;
      else sign = 0.0;
    }
    e.weight = sign * magnitude / denom;
  }

  incoming_offsets_.assign(neurons_.size() + 1, 0);
  for (const auto& e : edges_) ++incoming_offsets_[e.target + 1];
  for (std::size_t i = 1; i < incoming_offsets_.size(); ++i) incoming_offsets_[i] += incoming_offsets_[i - 1];
  incoming_edge_indices_.assign(edges_.size(), 0);
  auto cursor = incoming_offsets_;
  for (std::size_t i = 0; i < edges_.size(); ++i) incoming_edge_indices_[cursor[edges_[i].target]++] = i;
}

}  // namespace flyquant
