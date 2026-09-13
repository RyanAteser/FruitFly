#include "flyquant/metrics.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace flyquant {
namespace {

double safe_div(double a, double b) {
  return b == 0.0 ? std::numeric_limits<double>::quiet_NaN() : a / b;
}

double roc_auc(const std::vector<Prediction>& p) {
  std::size_t positives = 0;
  std::size_t negatives = 0;
  for (const auto& x : p) x.y == 1 ? ++positives : ++negatives;
  if (positives == 0 || negatives == 0) return std::numeric_limits<double>::quiet_NaN();

  std::vector<std::pair<double, int>> values;
  values.reserve(p.size());
  for (const auto& x : p) values.emplace_back(x.p_up, x.y);
  std::sort(values.begin(), values.end(), [](const auto& a, const auto& b) {
    return a.first < b.first;
  });

  double rank_sum_pos = 0.0;
  std::size_t i = 0;
  while (i < values.size()) {
    std::size_t j = i + 1;
    while (j < values.size() && values[j].first == values[i].first) ++j;
    const double avg_rank = 0.5 * (static_cast<double>(i + 1) + static_cast<double>(j));
    for (std::size_t k = i; k < j; ++k) if (values[k].second == 1) rank_sum_pos += avg_rank;
    i = j;
  }
  const double pos = static_cast<double>(positives);
  const double neg = static_cast<double>(negatives);
  return (rank_sum_pos - pos * (pos + 1.0) / 2.0) / (pos * neg);
}

double correlation(const std::vector<Prediction>& p) {
  if (p.size() < 2) return std::numeric_limits<double>::quiet_NaN();
  double mean_score = 0.0;
  double mean_return = 0.0;
  for (const auto& x : p) {
    mean_score += x.p_up;
    mean_return += x.future_log_return;
  }
  mean_score /= static_cast<double>(p.size());
  mean_return /= static_cast<double>(p.size());

  double cov = 0.0;
  double var_score = 0.0;
  double var_return = 0.0;
  for (const auto& x : p) {
    const double ds = x.p_up - mean_score;
    const double dr = x.future_log_return - mean_return;
    cov += ds * dr;
    var_score += ds * ds;
    var_return += dr * dr;
  }
  if (var_score == 0.0 || var_return == 0.0) return std::numeric_limits<double>::quiet_NaN();
  return cov / std::sqrt(var_score * var_return);
}

}  // namespace

ClassificationMetrics evaluate_predictions(const std::vector<Prediction>& p) {
  if (p.empty()) throw std::runtime_error("cannot evaluate empty predictions");
  std::size_t tp = 0, tn = 0, fp = 0, fn = 0;
  double logloss = 0.0;
  double brier = 0.0;
  std::array<double, 10> conf_sum{};
  std::array<double, 10> label_sum{};
  std::array<std::size_t, 10> bin_count{};

  for (const auto& x : p) {
    const double prob = std::clamp(x.p_up, 1e-15, 1.0 - 1e-15);
    const int pred = prob >= 0.5 ? 1 : 0;
    if (pred == 1 && x.y == 1) ++tp;
    if (pred == 0 && x.y == 0) ++tn;
    if (pred == 1 && x.y == 0) ++fp;
    if (pred == 0 && x.y == 1) ++fn;

    logloss += -(static_cast<double>(x.y) * std::log(prob) +
                 (1.0 - static_cast<double>(x.y)) * std::log(1.0 - prob));
    const double diff = prob - static_cast<double>(x.y);
    brier += diff * diff;

    const auto bin = std::min<std::size_t>(9, static_cast<std::size_t>(prob * 10.0));
    conf_sum[bin] += prob;
    label_sum[bin] += static_cast<double>(x.y);
    ++bin_count[bin];
  }

  ClassificationMetrics m;
  m.n = p.size();
  const double n = static_cast<double>(p.size());
  m.accuracy = static_cast<double>(tp + tn) / n;
  const double tpr = safe_div(static_cast<double>(tp), static_cast<double>(tp + fn));
  const double tnr = safe_div(static_cast<double>(tn), static_cast<double>(tn + fp));
  m.balanced_accuracy = (std::isnan(tpr) || std::isnan(tnr))
                            ? std::numeric_limits<double>::quiet_NaN()
                            : 0.5 * (tpr + tnr);
  m.precision = safe_div(static_cast<double>(tp), static_cast<double>(tp + fp));
  m.recall = tpr;
  m.roc_auc = roc_auc(p);
  m.log_loss = logloss / n;
  m.brier_score = brier / n;
  m.score_return_correlation = correlation(p);

  double ece = 0.0;
  for (std::size_t b = 0; b < 10; ++b) {
    if (bin_count[b] == 0) continue;
    const double count = static_cast<double>(bin_count[b]);
    const double avg_conf = conf_sum[b] / count;
    const double avg_label = label_sum[b] / count;
    ece += (count / n) * std::abs(avg_conf - avg_label);
  }
  m.expected_calibration_error_10 = ece;
  return m;
}

std::string metrics_csv_header() {
  return "model,split,n,accuracy,balanced_accuracy,precision,recall,roc_auc,log_loss,brier_score,ece10,score_return_correlation\n";
}

std::string metrics_csv_row(const std::string& model, const std::string& split,
                            const ClassificationMetrics& m) {
  std::ostringstream out;
  out << std::setprecision(17) << model << ',' << split << ',' << m.n << ',' << m.accuracy << ','
      << m.balanced_accuracy << ',' << m.precision << ',' << m.recall << ',' << m.roc_auc << ','
      << m.log_loss << ',' << m.brier_score << ',' << m.expected_calibration_error_10 << ','
      << m.score_return_correlation << '\n';
  return out.str();
}

}  // namespace flyquant
