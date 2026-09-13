#include "flyquant/neural.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace flyquant {
namespace {

std::vector<std::string> split_csv(const std::string& line) {
  std::vector<std::string> out;
  std::stringstream ss(line);
  std::string item;
  while (std::getline(ss, item, ',')) out.push_back(item);
  return out;
}

}  // namespace

SensoryEncoder SensoryEncoder::load_csv(const std::filesystem::path& path,
                                        const ConnectomeGraph& graph,
                                        std::size_t feature_count) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open sensory map: " + path.string());
  std::string line;
  if (!std::getline(in, line) || line != "neuron_id,feature_index,gain,bias") {
    throw std::runtime_error("sensory map header must be neuron_id,feature_index,gain,bias");
  }
  std::vector<SensoryAssignment> a;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    const auto f = split_csv(line);
    if (f.size() != 4) throw std::runtime_error("invalid sensory map row");
    const auto neuron_id = std::stoull(f[0]);
    const auto feature_index = static_cast<std::size_t>(std::stoull(f[1]));
    if (feature_index >= feature_count) throw std::runtime_error("sensory feature_index out of range");
    a.push_back({graph.neuron_index(neuron_id), feature_index, std::stod(f[2]), std::stod(f[3]), neuron_id});
  }
  return from_assignments(std::move(a), feature_count);
}

SensoryEncoder SensoryEncoder::from_assignments(std::vector<SensoryAssignment> assignments,
                                                std::size_t feature_count) {
  if (assignments.empty()) throw std::runtime_error("sensory assignment must not be empty");
  SensoryEncoder e;
  e.assignments_ = std::move(assignments);
  e.feature_count_ = feature_count;
  return e;
}

std::vector<double> SensoryEncoder::encode(const std::vector<double>& features,
                                           std::size_t neuron_count) const {
  if (features.size() != feature_count_) throw std::runtime_error("sensory encoder feature width mismatch");
  std::vector<double> drive(neuron_count, 0.0);
  for (const auto& a : assignments_) {
    if (a.neuron_index >= neuron_count) throw std::runtime_error("sensory neuron index out of range");
    drive[a.neuron_index] += a.gain * std::clamp(features[a.feature_index], -8.0, 8.0) + a.bias;
  }
  return drive;
}

NeuralDynamics::NeuralDynamics(DynamicsConfig config) : config_(config) {
  if (config_.steps == 0 || !(config_.leak > 0.0 && config_.leak <= 1.0) ||
      config_.recurrent_gain < 0.0 || config_.input_gain < 0.0) {
    throw std::runtime_error("invalid neural dynamics configuration");
  }
}

std::vector<double> NeuralDynamics::run(const ConnectomeGraph& graph,
                                        const std::vector<double>& external_drive) const {
  const auto n = graph.neurons().size();
  if (external_drive.size() != n) throw std::runtime_error("external drive width mismatch");
  std::vector<double> state(n, 0.0);
  std::vector<double> next(n, 0.0);
  const auto& offsets = graph.incoming_offsets();
  const auto& incoming = graph.incoming_edge_indices();
  const auto& edges = graph.edges();

  for (std::size_t step = 0; step < config_.steps; ++step) {
    for (std::size_t i = 0; i < n; ++i) {
      double recurrent = 0.0;
      for (std::size_t k = offsets[i]; k < offsets[i + 1]; ++k) {
        const auto& edge = edges[incoming[k]];
        recurrent += edge.weight * state[edge.source];
      }
      const double candidate = std::tanh(config_.input_gain * external_drive[i] +
                                         config_.recurrent_gain * recurrent);
      next[i] = (1.0 - config_.leak) * state[i] + config_.leak * candidate;
    }
    state.swap(next);
  }
  return state;
}

UpDownDecoder UpDownDecoder::load_csv(const std::filesystem::path& path,
                                      const ConnectomeGraph& graph) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open output map: " + path.string());
  std::string line;
  if (!std::getline(in, line) || line != "neuron_id,channel") {
    throw std::runtime_error("output map header must be neuron_id,channel");
  }
  std::vector<OutputAssignment> a;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    const auto f = split_csv(line);
    if (f.size() != 2) throw std::runtime_error("invalid output map row");
    OutputChannel channel;
    if (f[1] == "UP") channel = OutputChannel::Up;
    else if (f[1] == "DOWN") channel = OutputChannel::Down;
    else throw std::runtime_error("output channel must be UP or DOWN");
    const auto neuron_id = std::stoull(f[0]);
    a.push_back({graph.neuron_index(neuron_id), channel, neuron_id});
  }
  return from_assignments(std::move(a));
}

UpDownDecoder UpDownDecoder::from_assignments(std::vector<OutputAssignment> assignments) {
  std::size_t up = 0;
  std::size_t down = 0;
  for (const auto& a : assignments) a.channel == OutputChannel::Up ? ++up : ++down;
  if (up == 0 || down == 0) throw std::runtime_error("decoder requires at least one UP and one DOWN neuron");
  UpDownDecoder d;
  d.assignments_ = std::move(assignments);
  return d;
}

UpDownDecoder::Result UpDownDecoder::decode(const std::vector<double>& state) const {
  double up = 0.0;
  double down = 0.0;
  std::size_t up_n = 0;
  std::size_t down_n = 0;
  for (const auto& a : assignments_) {
    if (a.neuron_index >= state.size()) throw std::runtime_error("output neuron index out of range");
    if (a.channel == OutputChannel::Up) {
      up += state[a.neuron_index];
      ++up_n;
    } else {
      down += state[a.neuron_index];
      ++down_n;
    }
  }
  up /= static_cast<double>(up_n);
  down /= static_cast<double>(down_n);
  const double m = std::max(up, down);
  const double eup = std::exp(up - m);
  const double edown = std::exp(down - m);
  return {up, down, eup / (eup + edown)};
}

}  // namespace flyquant
