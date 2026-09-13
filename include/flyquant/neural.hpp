#pragma once

#include "flyquant/connectome.hpp"

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

namespace flyquant {

struct SensoryAssignment {
  std::size_t neuron_index{};
  std::size_t feature_index{};
  double gain{1.0};
  double bias{};
};

class SensoryEncoder {
 public:
  static SensoryEncoder load_csv(const std::filesystem::path& path,
                                 const ConnectomeGraph& graph,
                                 std::size_t feature_count);
  static SensoryEncoder from_assignments(std::vector<SensoryAssignment> assignments,
                                         std::size_t feature_count);
  [[nodiscard]] std::vector<double> encode(const std::vector<double>& features,
                                           std::size_t neuron_count) const;
 private:
  std::vector<SensoryAssignment> assignments_;
  std::size_t feature_count_{};
};

struct DynamicsConfig {
  std::size_t steps{8};
  double leak{0.5};
  double recurrent_gain{1.0};
  double input_gain{1.0};
};

class NeuralDynamics {
 public:
  explicit NeuralDynamics(DynamicsConfig config);
  [[nodiscard]] std::vector<double> run(const ConnectomeGraph& graph,
                                        const std::vector<double>& external_drive) const;
 private:
  DynamicsConfig config_;
};

enum class OutputChannel { Up, Down };
struct OutputAssignment {
  std::size_t neuron_index{};
  OutputChannel channel{OutputChannel::Up};
};

class UpDownDecoder {
 public:
  static UpDownDecoder load_csv(const std::filesystem::path& path,
                                const ConnectomeGraph& graph);
  static UpDownDecoder from_assignments(std::vector<OutputAssignment> assignments);
  struct Result {
    double activity_up{};
    double activity_down{};
    double p_up{};
  };
  [[nodiscard]] Result decode(const std::vector<double>& state) const;
 private:
  std::vector<OutputAssignment> assignments_;
};

}  // namespace flyquant
