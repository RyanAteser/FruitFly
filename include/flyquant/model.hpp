#pragma once

#include "flyquant/connectome.hpp"
#include "flyquant/neural.hpp"
#include "flyquant/split.hpp"
#include "flyquant/types.hpp"

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

namespace flyquant {

struct ReservoirTrainConfig {
  std::size_t epochs{300};
  double learning_rate{0.05};
  double l2{1e-4};
};

struct ModelSourceHashes {
  std::string neurons_sha256;
  std::string edges_sha256;
  std::string sensory_sha256;
  std::string outputs_sha256;
};

class ConnectomeReservoirModel {
 public:
  static ConnectomeReservoirModel train(const ConnectomeGraph& graph,
                                        const SensoryEncoder& encoder,
                                        const UpDownDecoder& decoder,
                                        DynamicsConfig dynamics,
                                        const std::vector<FeatureSample>& raw_train,
                                        ReservoirTrainConfig train_config,
                                        ModelSourceHashes hashes);

  static ConnectomeReservoirModel load(const std::filesystem::path& path,
                                       const ConnectomeGraph& graph,
                                       const std::string& neurons_sha256,
                                       const std::string& edges_sha256);

  void save(const std::filesystem::path& path) const;

  [[nodiscard]] Prediction predict_one(const ConnectomeGraph& graph,
                                       const FeatureSample& raw_sample) const;
  [[nodiscard]] std::vector<Prediction> predict(const ConnectomeGraph& graph,
                                                const std::vector<FeatureSample>& raw_samples) const;

  [[nodiscard]] const Standardizer& standardizer() const noexcept { return standardizer_; }
  [[nodiscard]] const DynamicsConfig& dynamics_config() const noexcept { return dynamics_; }
  [[nodiscard]] const ReservoirTrainConfig& train_config() const noexcept { return train_config_; }
  [[nodiscard]] const ModelSourceHashes& source_hashes() const noexcept { return hashes_; }
  [[nodiscard]] const SensoryEncoder& encoder() const noexcept { return encoder_; }
  [[nodiscard]] const UpDownDecoder& decoder() const noexcept { return decoder_; }
  [[nodiscard]] const std::vector<double>& readout_weights() const noexcept { return readout_weights_; }
  [[nodiscard]] double bias_up() const noexcept { return bias_up_; }
  [[nodiscard]] double bias_down() const noexcept { return bias_down_; }

 private:
  Standardizer standardizer_;
  DynamicsConfig dynamics_;
  ReservoirTrainConfig train_config_;
  ModelSourceHashes hashes_;
  SensoryEncoder encoder_;
  UpDownDecoder decoder_;
  std::vector<double> readout_weights_;
  double bias_up_{};
  double bias_down_{};
};

}  // namespace flyquant
