#include "flyquant/model.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace flyquant {
namespace {

constexpr std::string_view kMagic = "FLYQUANT_CONNECTOME_MODEL_V1";

std::vector<std::string> split(const std::string& text, char delimiter) {
  std::vector<std::string> out;
  std::stringstream ss(text);
  std::string item;
  while (std::getline(ss, item, delimiter)) out.push_back(item);
  return out;
}

std::vector<double> parse_double_list(const std::string& text) {
  std::vector<double> out;
  if (text.empty()) return out;
  for (const auto& item : split(text, ',')) out.push_back(std::stod(item));
  return out;
}

std::string double_list(const std::vector<double>& values) {
  std::ostringstream out;
  out << std::setprecision(17);
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i != 0) out << ',';
    out << values[i];
  }
  return out.str();
}

std::string require_key(std::ifstream& in, const std::string& expected_key) {
  std::string line;
  if (!std::getline(in, line)) throw std::runtime_error("unexpected end of model file");
  const auto pos = line.find('=');
  if (pos == std::string::npos || line.substr(0, pos) != expected_key) {
    throw std::runtime_error("model key mismatch: expected " + expected_key);
  }
  return line.substr(pos + 1);
}

std::vector<double> standardized_features(const Standardizer& standardizer,
                                          const std::vector<double>& raw) {
  if (raw.size() != standardizer.mean.size() || raw.size() != standardizer.stdev.size()) {
    throw std::runtime_error("model feature width mismatch");
  }
  std::vector<double> out(raw.size());
  for (std::size_t i = 0; i < raw.size(); ++i) {
    const double sd = standardizer.stdev[i];
    if (!(sd > 0.0) || !std::isfinite(sd)) throw std::runtime_error("invalid stored standard deviation");
    out[i] = std::clamp((raw[i] - standardizer.mean[i]) / sd, -8.0, 8.0);
  }
  return out;
}

double stable_p_up(double activity_up, double activity_down) {
  const double m = std::max(activity_up, activity_down);
  const double up = std::exp(activity_up - m);
  const double down = std::exp(activity_down - m);
  return up / (up + down);
}

struct CachedState {
  std::vector<double> readout;
  int y{};
};

void validate_train_config(const ReservoirTrainConfig& cfg) {
  if (cfg.epochs == 0) throw std::runtime_error("reservoir epochs must be positive");
  if (!(cfg.learning_rate > 0.0) || !std::isfinite(cfg.learning_rate)) {
    throw std::runtime_error("reservoir learning rate must be positive and finite");
  }
  if (cfg.l2 < 0.0 || !std::isfinite(cfg.l2)) {
    throw std::runtime_error("reservoir l2 must be finite and non-negative");
  }
}

}  // namespace

ConnectomeReservoirModel ConnectomeReservoirModel::train(
    const ConnectomeGraph& graph, const SensoryEncoder& encoder, const UpDownDecoder& decoder,
    DynamicsConfig dynamics, const std::vector<FeatureSample>& raw_train,
    ReservoirTrainConfig train_config, ModelSourceHashes hashes) {
  if (raw_train.empty()) throw std::runtime_error("cannot train connectome model on empty TRAIN split");
  validate_train_config(train_config);
  if (encoder.feature_count() == 0) throw std::runtime_error("sensory encoder has zero feature width");

  ConnectomeReservoirModel model;
  model.standardizer_ = Standardizer::fit(raw_train);
  model.dynamics_ = dynamics;
  model.train_config_ = train_config;
  model.hashes_ = std::move(hashes);
  model.encoder_ = encoder;
  model.decoder_ = decoder;

  const auto& outputs = model.decoder_.assignments();
  if (outputs.empty()) throw std::runtime_error("connectome model requires output assignments");
  model.readout_weights_.assign(outputs.size(), 0.0);

  std::size_t up_count = 0;
  std::size_t down_count = 0;
  for (const auto& output : outputs) {
    if (output.channel == OutputChannel::Up) ++up_count;
    else ++down_count;
  }
  if (up_count == 0 || down_count == 0) {
    throw std::runtime_error("connectome model requires both UP and DOWN output populations");
  }
  for (std::size_t i = 0; i < outputs.size(); ++i) {
    model.readout_weights_[i] = outputs[i].channel == OutputChannel::Up
                                    ? 1.0 / static_cast<double>(up_count)
                                    : 1.0 / static_cast<double>(down_count);
  }

  std::vector<FeatureSample> train = raw_train;
  model.standardizer_.transform(train);
  NeuralDynamics dynamics_runner(model.dynamics_);

  std::vector<CachedState> cached;
  cached.reserve(train.size());
  for (const auto& sample : train) {
    const auto drive = model.encoder_.encode(sample.x, graph.neurons().size());
    const auto state = dynamics_runner.run(graph, drive);
    CachedState row;
    row.y = sample.y;
    row.readout.reserve(outputs.size());
    for (const auto& output : outputs) {
      if (output.neuron_index >= state.size()) throw std::runtime_error("output neuron index out of range");
      row.readout.push_back(state[output.neuron_index]);
    }
    cached.push_back(std::move(row));
  }

  std::vector<double> grad(model.readout_weights_.size(), 0.0);
  const double inv_n = 1.0 / static_cast<double>(cached.size());
  for (std::size_t epoch = 0; epoch < train_config.epochs; ++epoch) {
    std::fill(grad.begin(), grad.end(), 0.0);
    double grad_bias_up = 0.0;
    double grad_bias_down = 0.0;

    for (const auto& row : cached) {
      double activity_up = model.bias_up_;
      double activity_down = model.bias_down_;
      for (std::size_t j = 0; j < outputs.size(); ++j) {
        if (outputs[j].channel == OutputChannel::Up) {
          activity_up += model.readout_weights_[j] * row.readout[j];
        } else {
          activity_down += model.readout_weights_[j] * row.readout[j];
        }
      }
      const double p_up = stable_p_up(activity_up, activity_down);
      const double error_up = p_up - static_cast<double>(row.y);
      grad_bias_up += error_up;
      grad_bias_down -= error_up;
      for (std::size_t j = 0; j < outputs.size(); ++j) {
        const double score_grad = outputs[j].channel == OutputChannel::Up ? error_up : -error_up;
        grad[j] += score_grad * row.readout[j];
      }
    }

    for (std::size_t j = 0; j < model.readout_weights_.size(); ++j) {
      const double g = grad[j] * inv_n + train_config.l2 * model.readout_weights_[j];
      model.readout_weights_[j] -= train_config.learning_rate * g;
    }
    model.bias_up_ -= train_config.learning_rate * grad_bias_up * inv_n;
    model.bias_down_ -= train_config.learning_rate * grad_bias_down * inv_n;
  }

  return model;
}

void ConnectomeReservoirModel::save(const std::filesystem::path& path) const {
  if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
  std::ofstream out(path, std::ios::out | std::ios::trunc);
  if (!out) throw std::runtime_error("cannot write model file: " + path.string());
  out << kMagic << '\n' << std::setprecision(17)
      << "model_type=connectome_reservoir_v1\n"
      << "feature_version=features-v0.1\n"
      << "weight_transform=unsigned_log1p_incoming_normalized\n"
      << "feature_count=" << standardizer_.mean.size() << '\n'
      << "neurons_sha256=" << hashes_.neurons_sha256 << '\n'
      << "edges_sha256=" << hashes_.edges_sha256 << '\n'
      << "sensory_sha256=" << hashes_.sensory_sha256 << '\n'
      << "outputs_sha256=" << hashes_.outputs_sha256 << '\n'
      << "dynamics_steps=" << dynamics_.steps << '\n'
      << "dynamics_leak=" << dynamics_.leak << '\n'
      << "dynamics_recurrent_gain=" << dynamics_.recurrent_gain << '\n'
      << "dynamics_input_gain=" << dynamics_.input_gain << '\n'
      << "train_epochs=" << train_config_.epochs << '\n'
      << "train_learning_rate=" << train_config_.learning_rate << '\n'
      << "train_l2=" << train_config_.l2 << '\n'
      << "bias_up=" << bias_up_ << '\n'
      << "bias_down=" << bias_down_ << '\n'
      << "mean=" << double_list(standardizer_.mean) << '\n'
      << "stdev=" << double_list(standardizer_.stdev) << '\n'
      << "sensory_count=" << encoder_.assignments().size() << '\n';
  for (const auto& a : encoder_.assignments()) {
    out << "S," << a.neuron_id << ',' << a.feature_index << ',' << a.gain << ',' << a.bias << '\n';
  }
  out << "output_count=" << decoder_.assignments().size() << '\n';
  for (std::size_t i = 0; i < decoder_.assignments().size(); ++i) {
    const auto& a = decoder_.assignments()[i];
    out << "O," << a.neuron_id << ','
        << (a.channel == OutputChannel::Up ? "UP" : "DOWN") << ','
        << readout_weights_[i] << '\n';
  }
  out << "END\n";
}

ConnectomeReservoirModel ConnectomeReservoirModel::load(
    const std::filesystem::path& path, const ConnectomeGraph& graph,
    const std::string& neurons_sha256, const std::string& edges_sha256) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open model file: " + path.string());
  std::string line;
  if (!std::getline(in, line) || line != kMagic) throw std::runtime_error("unsupported model format");

  ConnectomeReservoirModel model;
  if (require_key(in, "model_type") != "connectome_reservoir_v1") {
    throw std::runtime_error("unsupported model type");
  }
  if (require_key(in, "feature_version") != "features-v0.1") {
    throw std::runtime_error("unsupported feature version");
  }
  if (require_key(in, "weight_transform") != "unsigned_log1p_incoming_normalized") {
    throw std::runtime_error("unsupported weight transform");
  }
  const auto feature_count = static_cast<std::size_t>(std::stoull(require_key(in, "feature_count")));
  model.hashes_.neurons_sha256 = require_key(in, "neurons_sha256");
  model.hashes_.edges_sha256 = require_key(in, "edges_sha256");
  model.hashes_.sensory_sha256 = require_key(in, "sensory_sha256");
  model.hashes_.outputs_sha256 = require_key(in, "outputs_sha256");
  if (model.hashes_.neurons_sha256 != neurons_sha256 || model.hashes_.edges_sha256 != edges_sha256) {
    throw std::runtime_error("model/connectome source hash mismatch");
  }

  model.dynamics_.steps = static_cast<std::size_t>(std::stoull(require_key(in, "dynamics_steps")));
  model.dynamics_.leak = std::stod(require_key(in, "dynamics_leak"));
  model.dynamics_.recurrent_gain = std::stod(require_key(in, "dynamics_recurrent_gain"));
  model.dynamics_.input_gain = std::stod(require_key(in, "dynamics_input_gain"));
  model.train_config_.epochs = static_cast<std::size_t>(std::stoull(require_key(in, "train_epochs")));
  model.train_config_.learning_rate = std::stod(require_key(in, "train_learning_rate"));
  model.train_config_.l2 = std::stod(require_key(in, "train_l2"));
  model.bias_up_ = std::stod(require_key(in, "bias_up"));
  model.bias_down_ = std::stod(require_key(in, "bias_down"));
  model.standardizer_.mean = parse_double_list(require_key(in, "mean"));
  model.standardizer_.stdev = parse_double_list(require_key(in, "stdev"));
  if (model.standardizer_.mean.size() != feature_count || model.standardizer_.stdev.size() != feature_count) {
    throw std::runtime_error("stored standardizer width mismatch");
  }
  validate_train_config(model.train_config_);
  NeuralDynamics validate_dynamics(model.dynamics_);
  (void)validate_dynamics;

  const auto sensory_count = static_cast<std::size_t>(std::stoull(require_key(in, "sensory_count")));
  std::vector<SensoryAssignment> sensory;
  sensory.reserve(sensory_count);
  for (std::size_t i = 0; i < sensory_count; ++i) {
    if (!std::getline(in, line)) throw std::runtime_error("unexpected end of sensory section");
    const auto f = split(line, ',');
    if (f.size() != 5 || f[0] != "S") throw std::runtime_error("invalid sensory model row");
    const auto neuron_id = static_cast<NeuronId>(std::stoull(f[1]));
    sensory.push_back({graph.neuron_index(neuron_id),
                       static_cast<std::size_t>(std::stoull(f[2])), std::stod(f[3]), std::stod(f[4]), neuron_id});
  }
  model.encoder_ = SensoryEncoder::from_assignments(std::move(sensory), feature_count);

  const auto output_count = static_cast<std::size_t>(std::stoull(require_key(in, "output_count")));
  std::vector<OutputAssignment> outputs;
  outputs.reserve(output_count);
  model.readout_weights_.reserve(output_count);
  for (std::size_t i = 0; i < output_count; ++i) {
    if (!std::getline(in, line)) throw std::runtime_error("unexpected end of output section");
    const auto f = split(line, ',');
    if (f.size() != 4 || f[0] != "O") throw std::runtime_error("invalid output model row");
    const auto neuron_id = static_cast<NeuronId>(std::stoull(f[1]));
    OutputChannel channel;
    if (f[2] == "UP") channel = OutputChannel::Up;
    else if (f[2] == "DOWN") channel = OutputChannel::Down;
    else throw std::runtime_error("invalid stored output channel");
    outputs.push_back({graph.neuron_index(neuron_id), channel, neuron_id});
    model.readout_weights_.push_back(std::stod(f[3]));
  }
  model.decoder_ = UpDownDecoder::from_assignments(std::move(outputs));
  if (!std::getline(in, line) || line != "END") throw std::runtime_error("model file missing END marker");
  return model;
}

Prediction ConnectomeReservoirModel::predict_one(const ConnectomeGraph& graph,
                                                  const FeatureSample& raw_sample) const {
  const auto features = standardized_features(standardizer_, raw_sample.x);
  const auto drive = encoder_.encode(features, graph.neurons().size());
  NeuralDynamics runner(dynamics_);
  const auto state = runner.run(graph, drive);
  const auto& outputs = decoder_.assignments();
  if (outputs.size() != readout_weights_.size()) throw std::runtime_error("stored readout width mismatch");

  double activity_up = bias_up_;
  double activity_down = bias_down_;
  for (std::size_t i = 0; i < outputs.size(); ++i) {
    if (outputs[i].neuron_index >= state.size()) throw std::runtime_error("output neuron index out of range");
    if (outputs[i].channel == OutputChannel::Up) {
      activity_up += readout_weights_[i] * state[outputs[i].neuron_index];
    } else {
      activity_down += readout_weights_[i] * state[outputs[i].neuron_index];
    }
  }
  return {raw_sample.anchor_ts_ns, raw_sample.y, raw_sample.future_log_return,
          stable_p_up(activity_up, activity_down), activity_up, activity_down};
}

std::vector<Prediction> ConnectomeReservoirModel::predict(
    const ConnectomeGraph& graph, const std::vector<FeatureSample>& raw_samples) const {
  std::vector<Prediction> out;
  out.reserve(raw_samples.size());
  for (const auto& sample : raw_samples) out.push_back(predict_one(graph, sample));
  return out;
}

}  // namespace flyquant
