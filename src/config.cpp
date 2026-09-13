#include "flyquant/config.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace flyquant {
namespace {

std::string trim(std::string s) {
  auto not_space = [](unsigned char c) { return !std::isspace(c); };
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
  s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
  return s;
}

std::vector<std::string> split_csv(const std::string& s) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) {
    item = trim(item);
    if (!item.empty()) out.push_back(item);
  }
  return out;
}

}  // namespace

KeyValueConfig KeyValueConfig::load(const std::filesystem::path& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open config: " + path.string());

  KeyValueConfig cfg;
  std::string line;
  std::size_t line_no = 0;
  while (std::getline(in, line)) {
    ++line_no;
    const auto hash = line.find('#');
    if (hash != std::string::npos) line.resize(hash);
    line = trim(line);
    if (line.empty()) continue;
    const auto eq = line.find('=');
    if (eq == std::string::npos) throw std::runtime_error("invalid config line " + std::to_string(line_no));
    auto key = trim(line.substr(0, eq));
    auto value = trim(line.substr(eq + 1));
    if (key.empty()) throw std::runtime_error("empty config key on line " + std::to_string(line_no));
    if (!cfg.values_.emplace(std::move(key), std::move(value)).second) {
      throw std::runtime_error("duplicate config key on line " + std::to_string(line_no));
    }
  }
  return cfg;
}

const std::string& KeyValueConfig::require(const std::string& key) const {
  const auto it = values_.find(key);
  if (it == values_.end()) throw std::runtime_error("missing required config key: " + key);
  return it->second;
}

std::string KeyValueConfig::get(const std::string& key, std::string fallback) const {
  const auto it = values_.find(key);
  return it == values_.end() ? std::move(fallback) : it->second;
}

std::int64_t KeyValueConfig::get_i64(const std::string& key, std::int64_t fallback) const {
  const auto it = values_.find(key);
  return it == values_.end() ? fallback : std::stoll(it->second);
}

std::size_t KeyValueConfig::get_size(const std::string& key, std::size_t fallback) const {
  const auto it = values_.find(key);
  return it == values_.end() ? fallback : static_cast<std::size_t>(std::stoull(it->second));
}

double KeyValueConfig::get_double(const std::string& key, double fallback) const {
  const auto it = values_.find(key);
  return it == values_.end() ? fallback : std::stod(it->second);
}

bool KeyValueConfig::get_bool(const std::string& key, bool fallback) const {
  const auto it = values_.find(key);
  if (it == values_.end()) return fallback;
  if (it->second == "true" || it->second == "1") return true;
  if (it->second == "false" || it->second == "0") return false;
  throw std::runtime_error("invalid boolean for key: " + key);
}

std::vector<std::uint64_t> KeyValueConfig::get_u64_list(
    const std::string& key, std::vector<std::uint64_t> fallback) const {
  const auto it = values_.find(key);
  if (it == values_.end()) return fallback;
  std::vector<std::uint64_t> out;
  for (const auto& item : split_csv(it->second)) out.push_back(std::stoull(item));
  return out;
}

std::string KeyValueConfig::canonical_text() const {
  std::vector<std::pair<std::string, std::string>> pairs(values_.begin(), values_.end());
  std::sort(pairs.begin(), pairs.end());
  std::ostringstream out;
  for (const auto& [key, value] : pairs) out << key << '=' << value << '\n';
  return out.str();
}

ExperimentConfig parse_experiment_config(const KeyValueConfig& c) {
  ExperimentConfig cfg;
  cfg.experiment_id = c.require("experiment_id");
  cfg.dataset_path = c.require("dataset_path");
  cfg.results_dir = c.get("results_dir", "results");
  cfg.train_start_ns = c.get_i64("train_start_ns", 0);
  cfg.train_end_ns = c.get_i64("train_end_ns", 0);
  cfg.validation_start_ns = c.get_i64("validation_start_ns", 0);
  cfg.validation_end_ns = c.get_i64("validation_end_ns", 0);
  cfg.test_start_ns = c.get_i64("test_start_ns", 0);
  cfg.test_end_ns = c.get_i64("test_end_ns", 0);
  cfg.anchor_period_ns = c.get_i64("anchor_period_ns", cfg.anchor_period_ns);
  cfg.horizon_ns = c.get_i64("horizon_ns", cfg.horizon_ns);
  cfg.max_quote_age_ns = c.get_i64("max_quote_age_ns", cfg.max_quote_age_ns);
  cfg.logistic_epochs = c.get_size("logistic_epochs", cfg.logistic_epochs);
  cfg.logistic_learning_rate = c.get_double("logistic_learning_rate", cfg.logistic_learning_rate);
  cfg.logistic_l2 = c.get_double("logistic_l2", cfg.logistic_l2);
  cfg.test_locked = c.get_bool("test_locked", cfg.test_locked);
  cfg.seeds = c.get_u64_list("seeds", cfg.seeds);
  return cfg;
}

void validate_experiment_config(const ExperimentConfig& c) {
  if (c.experiment_id.empty()) throw std::runtime_error("experiment_id must not be empty");
  if (c.dataset_path.empty()) throw std::runtime_error("dataset_path must not be empty");
  if (!(c.train_start_ns < c.train_end_ns && c.validation_start_ns < c.validation_end_ns &&
        c.test_start_ns < c.test_end_ns)) {
    throw std::runtime_error("each split must have start < end");
  }
  if (c.validation_start_ns - c.train_end_ns < c.horizon_ns) {
    throw std::runtime_error("validation split must be embargoed from train by at least horizon_ns");
  }
  if (c.test_start_ns - c.validation_end_ns < c.horizon_ns) {
    throw std::runtime_error("test split must be embargoed from validation by at least horizon_ns");
  }
  if (c.anchor_period_ns <= 0 || c.horizon_ns <= 0 || c.max_quote_age_ns < 0) {
    throw std::runtime_error("invalid timing parameter");
  }
  if (c.logistic_epochs == 0 || c.logistic_learning_rate <= 0.0 || c.logistic_l2 < 0.0) {
    throw std::runtime_error("invalid logistic-regression parameters");
  }
  if (c.seeds.empty()) throw std::runtime_error("at least one predetermined seed is required");
}

}  // namespace flyquant
