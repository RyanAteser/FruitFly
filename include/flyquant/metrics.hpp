#pragma once

#include "flyquant/types.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace flyquant {

struct ClassificationMetrics {
  std::size_t n{};
  double accuracy{};
  double balanced_accuracy{};
  double precision{};
  double recall{};
  double roc_auc{};
  double log_loss{};
  double brier_score{};
  double expected_calibration_error_10{};
  double score_return_correlation{};
};

ClassificationMetrics evaluate_predictions(const std::vector<Prediction>& predictions);
std::string metrics_csv_header();
std::string metrics_csv_row(const std::string& model, const std::string& split,
                            const ClassificationMetrics& m);

}  // namespace flyquant
