#include "flyquant/logistic_regression.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace flyquant {
namespace {

double sigmoid(double z) {
  if (z >= 0.0) {
    const double e = std::exp(-z);
    return 1.0 / (1.0 + e);
  }
  const double e = std::exp(z);
  return e / (1.0 + e);
}

}  // namespace

LogisticRegression::LogisticRegression(std::size_t dimensions, double learning_rate,
                                       double l2, std::size_t epochs)
    : weights_(dimensions, 0.0), learning_rate_(learning_rate), l2_(l2), epochs_(epochs) {
  if (dimensions == 0 || learning_rate <= 0.0 || l2 < 0.0 || epochs == 0) {
    throw std::runtime_error("invalid logistic-regression constructor arguments");
  }
}

void LogisticRegression::fit(const std::vector<FeatureSample>& train) {
  if (train.empty()) throw std::runtime_error("cannot train logistic regression on empty data");
  const std::size_t d = weights_.size();
  std::vector<double> grad(d, 0.0);

  for (std::size_t epoch = 0; epoch < epochs_; ++epoch) {
    std::fill(grad.begin(), grad.end(), 0.0);
    double grad_bias = 0.0;
    for (const auto& row : train) {
      if (row.x.size() != d) throw std::runtime_error("logistic-regression feature width mismatch");
      const double p = predict_proba(row.x);
      const double err = p - static_cast<double>(row.y);
      grad_bias += err;
      for (std::size_t j = 0; j < d; ++j) grad[j] += err * row.x[j];
    }

    const double inv_n = 1.0 / static_cast<double>(train.size());
    bias_ -= learning_rate_ * grad_bias * inv_n;
    for (std::size_t j = 0; j < d; ++j) {
      const double g = grad[j] * inv_n + l2_ * weights_[j];
      weights_[j] -= learning_rate_ * g;
    }
  }
}

double LogisticRegression::predict_proba(const std::vector<double>& x) const {
  if (x.size() != weights_.size()) throw std::runtime_error("logistic-regression feature width mismatch");
  double z = bias_;
  for (std::size_t j = 0; j < x.size(); ++j) z += weights_[j] * x[j];
  return sigmoid(z);
}

std::vector<Prediction> LogisticRegression::predict(const std::vector<FeatureSample>& samples) const {
  std::vector<Prediction> out;
  out.reserve(samples.size());
  for (const auto& s : samples) {
    out.push_back({s.anchor_ts_ns, s.y, s.future_log_return, predict_proba(s.x)});
  }
  return out;
}

std::vector<Prediction> always_up_predictions(const std::vector<FeatureSample>& samples) {
  std::vector<Prediction> out;
  out.reserve(samples.size());
  for (const auto& s : samples) out.push_back({s.anchor_ts_ns, s.y, s.future_log_return, 1.0});
  return out;
}

std::vector<Prediction> always_down_predictions(const std::vector<FeatureSample>& samples) {
  std::vector<Prediction> out;
  out.reserve(samples.size());
  for (const auto& s : samples) out.push_back({s.anchor_ts_ns, s.y, s.future_log_return, 0.0});
  return out;
}

}  // namespace flyquant
