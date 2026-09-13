#pragma once

#include "flyquant/config.hpp"
#include "flyquant/metrics.hpp"
#include "flyquant/types.hpp"

#include <filesystem>
#include <string>
#include <tuple>
#include <vector>

namespace flyquant {

struct RunProvenance {
  std::string experiment_id;
  std::string timestamp_utc;
  std::string git_commit;
  std::string dataset_hash;
  std::string config_hash;
  std::string feature_version{"features-v0.1"};
  std::string connectome_version{"not-used"};
  std::string encoder_version{"not-used"};
  std::string decoder_version{"not-used"};
  std::uint64_t seed{};
};

class ImmutableRunWriter {
 public:
  ImmutableRunWriter(const std::filesystem::path& root, RunProvenance provenance);
  void write_manifest(const ExperimentConfig& cfg, const std::string& model_name,
                      const std::string& evaluation_split, std::uint64_t runtime_ms);
  void write_predictions(const std::string& model_name, const std::string& split,
                         const std::vector<Prediction>& predictions);
  void write_metrics(const std::vector<std::tuple<std::string, std::string, ClassificationMetrics>>& rows);
  void write_errors(const std::string& text);
  [[nodiscard]] const std::filesystem::path& directory() const noexcept { return directory_; }
 private:
  RunProvenance provenance_;
  std::filesystem::path directory_;
};

std::string utc_timestamp_compact();

}  // namespace flyquant
