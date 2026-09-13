#pragma once

#include "flyquant/config.hpp"
#include "flyquant/types.hpp"

#include <array>
#include <vector>

namespace flyquant {

enum class Split { Train = 0, Validation = 1, Test = 2 };

struct SplitSamples {
  std::vector<FeatureSample> train;
  std::vector<FeatureSample> validation;
  std::vector<FeatureSample> test;
};

SplitSamples chronological_split(const std::vector<FeatureSample>& samples,
                                 const ExperimentConfig& cfg);
void audit_split_integrity(const SplitSamples& split, const ExperimentConfig& cfg);

struct Standardizer {
  std::vector<double> mean;
  std::vector<double> stdev;
  static Standardizer fit(const std::vector<FeatureSample>& train);
  void transform(std::vector<FeatureSample>& samples) const;
};

}  // namespace flyquant
