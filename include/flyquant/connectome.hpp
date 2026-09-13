#pragma once

#include "flyquant/types.hpp"

#include <cstddef>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

namespace flyquant {

struct Neuron {
  NeuronId id{};
  std::string type;
  std::string region;
  std::string neurotransmitter;
  double activation{};
};

struct Edge {
  std::size_t source{};
  std::size_t target{};
  std::uint32_t synapse_count{};
  SynapticSign sign{SynapticSign::Unknown};
  double weight{};
};

enum class WeightTransform {
  UnsignedLog1pIncomingNormalized,
  SignedLog1pIncomingNormalized,
};

class ConnectomeGraph {
 public:
  static ConnectomeGraph load_csv(const std::filesystem::path& neurons_csv,
                                  const std::filesystem::path& edges_csv,
                                  WeightTransform transform);
  static ConnectomeGraph from_parts(std::vector<Neuron> neurons, std::vector<Edge> edges,
                                    WeightTransform transform);
  [[nodiscard]] const std::vector<Neuron>& neurons() const noexcept { return neurons_; }
  [[nodiscard]] const std::vector<Edge>& edges() const noexcept { return edges_; }
  [[nodiscard]] const std::vector<std::size_t>& incoming_offsets() const noexcept { return incoming_offsets_; }
  [[nodiscard]] const std::vector<std::size_t>& incoming_edge_indices() const noexcept { return incoming_edge_indices_; }
  [[nodiscard]] const std::unordered_map<NeuronId, std::size_t>& id_to_index() const noexcept { return id_to_index_; }
  [[nodiscard]] std::size_t neuron_index(NeuronId id) const;
  [[nodiscard]] std::vector<std::size_t> in_degrees() const;
  [[nodiscard]] std::vector<std::size_t> out_degrees() const;
  void rebuild(WeightTransform transform);
 private:
  std::vector<Neuron> neurons_;
  std::vector<Edge> edges_;
  std::unordered_map<NeuronId, std::size_t> id_to_index_;
  std::vector<std::size_t> incoming_offsets_;
  std::vector<std::size_t> incoming_edge_indices_;
};

}  // namespace flyquant
