#pragma once

#include "flyquant/types.hpp"

#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

namespace flyquant {

struct ExperimentConfig {
  std::string experiment_id;
  std::filesystem::path dataset_path;
  std::filesystem::path results_dir{"results"};
  TimestampNs train_start_ns{};
  TimestampNs train_end_ns{};
  TimestampNs validation_start_ns{};
  TimestampNs validation_end_ns{};
  TimestampNs test_start_ns{};
  TimestampNs test_end_ns{};
  TimestampNs anchor_period_ns{kNsPerMinute};
  TimestampNs horizon_ns{5LL * kNsPerMinute};
  TimestampNs max_quote_age_ns{2LL * kNsPerSecond};
  std::size_t logistic_epochs{500};
  double logistic_learning_rate{0.05};
  double logistic_l2{1e-4};
  bool test_locked{true};
  std::vector<std::uint64_t> seeds{42, 1337, 2026, 9001, 314159};
};

class KeyValueConfig {
 public:
  static KeyValueConfig load(const std::filesystem::path& path);
  [[nodiscard]] const std::string& require(const std::string& key) const;
  [[nodiscard]] std::string get(const std::string& key, std::string fallback) const;
  [[nodiscard]] std::int64_t get_i64(const std::string& key, std::int64_t fallback) const;
  [[nodiscard]] std::size_t get_size(const std::string& key, std::size_t fallback) const;
  [[nodiscard]] double get_double(const std::string& key, double fallback) const;
  [[nodiscard]] bool get_bool(const std::string& key, bool fallback) const;
  [[nodiscard]] std::vector<std::uint64_t> get_u64_list(const std::string& key,
                                                         std::vector<std::uint64_t> fallback) const;
  [[nodiscard]] std::string canonical_text() const;
 private:
  std::unordered_map<std::string, std::string> values_;
};

ExperimentConfig parse_experiment_config(const KeyValueConfig& cfg);
void validate_experiment_config(const ExperimentConfig& cfg);

}  // namespace flyquant
