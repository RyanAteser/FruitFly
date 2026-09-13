#include "flyquant/experiment.hpp"

#include <chrono>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <tuple>

namespace flyquant {

std::string utc_timestamp_compact() {
  const auto now = std::chrono::system_clock::now();
  const std::time_t t = std::chrono::system_clock::to_time_t(now);
  std::tm tm{};
#ifdef _WIN32
  gmtime_s(&tm, &t);
#else
  gmtime_r(&t, &tm);
#endif
  std::ostringstream out;
  out << std::put_time(&tm, "%Y%m%dT%H%M%SZ");
  return out.str();
}

ImmutableRunWriter::ImmutableRunWriter(const std::filesystem::path& root, RunProvenance provenance)
    : provenance_(std::move(provenance)) {
  const std::string suffix = provenance_.timestamp_utc + "_seed" + std::to_string(provenance_.seed) +
                             "_" + provenance_.config_hash.substr(0, 12);
  directory_ = root / provenance_.experiment_id / suffix;
  if (!std::filesystem::create_directories(directory_)) {
    throw std::runtime_error("immutable run directory already exists: " + directory_.string());
  }
}

void ImmutableRunWriter::write_manifest(
    const ExperimentConfig& c, const std::string& model_name,
    const std::string& evaluation_split, std::uint64_t runtime_ms,
    const std::vector<std::pair<std::string, std::string>>& extras) {
  std::ofstream out(directory_ / "manifest.txt", std::ios::out | std::ios::trunc);
  if (!out) throw std::runtime_error("cannot write manifest");
  out << "experiment_id=" << provenance_.experiment_id << '\n'
      << "timestamp_utc=" << provenance_.timestamp_utc << '\n'
      << "git_commit=" << provenance_.git_commit << '\n'
      << "dataset_hash_sha256=" << provenance_.dataset_hash << '\n'
      << "config_hash_sha256=" << provenance_.config_hash << '\n'
      << "feature_version=" << provenance_.feature_version << '\n'
      << "connectome_version=" << provenance_.connectome_version << '\n'
      << "encoder_version=" << provenance_.encoder_version << '\n'
      << "decoder_version=" << provenance_.decoder_version << '\n'
      << "seed=" << provenance_.seed << '\n'
      << "model=" << model_name << '\n'
      << "evaluation_split=" << evaluation_split << '\n'
      << "train_period_ns=" << c.train_start_ns << ':' << c.train_end_ns << '\n'
      << "validation_period_ns=" << c.validation_start_ns << ':' << c.validation_end_ns << '\n'
      << "test_period_ns=" << c.test_start_ns << ':' << c.test_end_ns << '\n'
      << "horizon_ns=" << c.horizon_ns << '\n'
      << "anchor_period_ns=" << c.anchor_period_ns << '\n'
      << "max_quote_age_ns=" << c.max_quote_age_ns << '\n'
      << "logistic_epochs=" << c.logistic_epochs << '\n'
      << "logistic_learning_rate=" << c.logistic_learning_rate << '\n'
      << "logistic_l2=" << c.logistic_l2 << '\n';
  for (const auto& [key, value] : extras) {
    if (key.empty() || key.find('=') != std::string::npos || key.find('\n') != std::string::npos ||
        value.find('\n') != std::string::npos) {
      throw std::runtime_error("invalid manifest extra field");
    }
    out << key << '=' << value << '\n';
  }
  out << "runtime_ms=" << runtime_ms << '\n';
}

void ImmutableRunWriter::write_predictions(const std::string& model_name, const std::string& split,
                                           const std::vector<Prediction>& predictions) {
  std::ofstream out(directory_ / ("predictions_" + model_name + "_" + split + ".csv"),
                    std::ios::out | std::ios::trunc);
  if (!out) throw std::runtime_error("cannot write predictions");
  out << "anchor_ts_ns,y,future_log_return,p_up,activity_up,activity_down\n"
      << std::setprecision(17);
  for (const auto& p : predictions) {
    out << p.anchor_ts_ns << ',' << p.y << ',' << p.future_log_return << ',' << p.p_up
        << ',' << p.activity_up << ',' << p.activity_down << '\n';
  }
}

void ImmutableRunWriter::write_metrics(
    const std::vector<std::tuple<std::string, std::string, ClassificationMetrics>>& rows) {
  std::ofstream out(directory_ / "metrics.csv", std::ios::out | std::ios::trunc);
  if (!out) throw std::runtime_error("cannot write metrics");
  out << metrics_csv_header();
  for (const auto& [model, split, metrics] : rows) out << metrics_csv_row(model, split, metrics);
}

void ImmutableRunWriter::write_errors(const std::string& text) {
  std::ofstream out(directory_ / "errors.log", std::ios::out | std::ios::trunc);
  if (!out) throw std::runtime_error("cannot write errors log");
  out << text;
}

}  // namespace flyquant
