#include "flyquant/split.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace flyquant {
namespace {

bool inside(const FeatureSample& s, TimestampNs start, TimestampNs end) {
  return s.anchor_ts_ns >= start && s.target_ts_ns <= end;
}

void audit_one(const std::vector<FeatureSample>& samples, TimestampNs start, TimestampNs end) {
  for (const auto& s : samples) {
    if (!inside(s, start, end)) throw std::runtime_error("sample crosses split boundary");
  }
}

}  // namespace

SplitSamples chronological_split(const std::vector<FeatureSample>& samples,
                                 const ExperimentConfig& cfg) {
  SplitSamples out;
  for (const auto& s : samples) {
    if (inside(s, cfg.train_start_ns, cfg.train_end_ns)) out.train.push_back(s);
    else if (inside(s, cfg.validation_start_ns, cfg.validation_end_ns)) out.validation.push_back(s);
    else if (inside(s, cfg.test_start_ns, cfg.test_end_ns)) out.test.push_back(s);
  }
  audit_split_integrity(out, cfg);
  return out;
}

void audit_split_integrity(const SplitSamples& split, const ExperimentConfig& cfg) {
  validate_experiment_config(cfg);
  audit_one(split.train, cfg.train_start_ns, cfg.train_end_ns);
  audit_one(split.validation, cfg.validation_start_ns, cfg.validation_end_ns);
  audit_one(split.test, cfg.test_start_ns, cfg.test_end_ns);

  if (!split.train.empty() && !split.validation.empty() &&
      split.train.back().target_ts_ns >= split.validation.front().anchor_ts_ns) {
    throw std::runtime_error("train/validation temporal overlap detected");
  }
  if (!split.validation.empty() && !split.test.empty() &&
      split.validation.back().target_ts_ns >= split.test.front().anchor_ts_ns) {
    throw std::runtime_error("validation/test temporal overlap detected");
  }
}

Standardizer Standardizer::fit(const std::vector<FeatureSample>& train) {
  if (train.empty()) throw std::runtime_error("cannot fit standardizer on empty train split");
  const std::size_t d = train.front().x.size();
  Standardizer s;
  s.mean.assign(d, 0.0);
  s.stdev.assign(d, 0.0);

  for (const auto& row : train) {
    if (row.x.size() != d) throw std::runtime_error("inconsistent feature width");
    for (std::size_t j = 0; j < d; ++j) s.mean[j] += row.x[j];
  }
  for (double& m : s.mean) m /= static_cast<double>(train.size());

  for (const auto& row : train) {
    for (std::size_t j = 0; j < d; ++j) {
      const double diff = row.x[j] - s.mean[j];
      s.stdev[j] += diff * diff;
    }
  }
  for (double& v : s.stdev) {
    v = std::sqrt(v / static_cast<double>(train.size()));
    if (v < 1e-12) v = 1.0;
  }
  return s;
}

void Standardizer::transform(std::vector<FeatureSample>& samples) const {
  for (auto& row : samples) {
    if (row.x.size() != mean.size()) throw std::runtime_error("standardizer width mismatch");
    for (std::size_t j = 0; j < row.x.size(); ++j) {
      row.x[j] = (row.x[j] - mean[j]) / stdev[j];
      row.x[j] = std::clamp(row.x[j], -8.0, 8.0);
    }
  }
}

}  // namespace flyquant
