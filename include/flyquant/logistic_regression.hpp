#pragma once

#include "flyquant/types.hpp"

#include <cstddef>
#include <vector>

namespace flyquant {

class LogisticRegression {
 public:
  LogisticRegression(std::size_t dimensions, double learning_rate, double l2,
                     std::size_t epochs);
  void fit(const std::vector<FeatureSample>& train);
  [[nodiscard]] double predict_proba(const std::vector<double>& x) const;
  [[nodiscard]] std::vector<Prediction> predict(const std::vector<FeatureSample>& samples) const;
  [[nodiscard]] const std::vector<double>& weights() const noexcept { return weights_; }
  [[nodiscard]] double bias() const noexcept { return bias_; }
 private:
  std::vector<double> weights_;
  double bias_{};
  double learning_rate_{};
  double l2_{};
  std::size_t epochs_{};
};

std::vector<Prediction> always_up_predictions(const std::vector<FeatureSample>& samples);
std::vector<Prediction> always_down_predictions(const std::vector<FeatureSample>& samples);

}  // namespace flyquant
