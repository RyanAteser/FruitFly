#include "flyquant/config.hpp"
#include "flyquant/connectome.hpp"
#include "flyquant/experiment.hpp"
#include "flyquant/features.hpp"
#include "flyquant/logistic_regression.hpp"
#include "flyquant/market_data.hpp"
#include "flyquant/metrics.hpp"
#include "flyquant/model.hpp"
#include "flyquant/neural.hpp"
#include "flyquant/sha256.hpp"
#include "flyquant/split.hpp"

#include <chrono>
#include <exception>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#ifndef FLYQUANT_GIT_COMMIT
#define FLYQUANT_GIT_COMMIT "unknown"
#endif

namespace {

using flyquant::ClassificationMetrics;

struct Args {
  std::string command;
  std::filesystem::path config;
  std::filesystem::path neurons;
  std::filesystem::path edges;
  std::filesystem::path sensory;
  std::filesystem::path outputs;
  std::filesystem::path model_out;
  std::filesystem::path model_in;
  std::string split{"validation"};
  bool allow_test{false};
};

Args parse_args(int argc, char** argv) {
  if (argc < 2) {
    throw std::runtime_error(
        "usage: flyquant <baseline|inspect-connectome|train-connectome|predict-connectome> [options]");
  }
  Args args;
  args.command = argv[1];
  for (int i = 2; i < argc; ++i) {
    const std::string a = argv[i];
    auto require_value = [&](const std::string& flag) -> std::string {
      if (i + 1 >= argc) throw std::runtime_error("missing value for " + flag);
      return argv[++i];
    };
    if (a == "--config") args.config = require_value(a);
    else if (a == "--split") args.split = require_value(a);
    else if (a == "--allow-test") args.allow_test = true;
    else if (a == "--neurons") args.neurons = require_value(a);
    else if (a == "--edges") args.edges = require_value(a);
    else if (a == "--sensory") args.sensory = require_value(a);
    else if (a == "--outputs") args.outputs = require_value(a);
    else if (a == "--model-out") args.model_out = require_value(a);
    else if (a == "--model-in") args.model_in = require_value(a);
    else throw std::runtime_error("unknown argument: " + a);
  }
  return args;
}

void require_eval_split(const Args& args, const flyquant::ExperimentConfig& cfg) {
  if (args.split != "validation" && args.split != "test") {
    throw std::runtime_error("--split must be validation or test");
  }
  if (args.split == "test" && cfg.test_locked && !args.allow_test) {
    throw std::runtime_error("TEST is locked; rerun with --allow-test only after the protocol is frozen");
  }
}

void print_metrics(const std::string& model, const ClassificationMetrics& m) {
  std::cout << model << ": n=" << m.n
            << " accuracy=" << m.accuracy
            << " balanced_accuracy=" << m.balanced_accuracy
            << " roc_auc=" << m.roc_auc
            << " log_loss=" << m.log_loss
            << " brier=" << m.brier_score << '\n';
}

flyquant::SplitSamples load_split_samples(const flyquant::ExperimentConfig& cfg,
                                          const std::string& evaluation_split) {
  const auto market = flyquant::load_market_csv(cfg.dataset_path);
  const flyquant::FeatureConfig feature_cfg{cfg.anchor_period_ns, cfg.horizon_ns, cfg.max_quote_age_ns};
  const auto last_anchor = (evaluation_split == "test" ? cfg.test_end_ns : cfg.validation_end_ns) - cfg.horizon_ns;
  auto samples = flyquant::build_feature_samples(market, cfg.train_start_ns, last_anchor, feature_cfg);
  return flyquant::chronological_split(samples, cfg);
}

flyquant::RunProvenance make_provenance(const flyquant::ExperimentConfig& cfg,
                                        const flyquant::KeyValueConfig& raw_cfg) {
  flyquant::RunProvenance provenance;
  provenance.experiment_id = cfg.experiment_id;
  provenance.timestamp_utc = flyquant::utc_timestamp_compact();
  provenance.git_commit = FLYQUANT_GIT_COMMIT;
  provenance.dataset_hash = flyquant::sha256_file(cfg.dataset_path);
  provenance.config_hash = flyquant::sha256_string(raw_cfg.canonical_text());
  provenance.seed = cfg.seeds.front();
  return provenance;
}

int run_baseline(const Args& args) {
  const auto run_started = std::chrono::steady_clock::now();
  if (args.config.empty()) throw std::runtime_error("baseline requires --config <path>");
  const auto raw_cfg = flyquant::KeyValueConfig::load(args.config);
  const auto cfg = flyquant::parse_experiment_config(raw_cfg);
  flyquant::validate_experiment_config(cfg);
  require_eval_split(args, cfg);

  auto split = load_split_samples(cfg, args.split);
  if (split.train.empty()) throw std::runtime_error("TRAIN contains no usable samples");
  auto* eval = args.split == "test" ? &split.test : &split.validation;
  if (eval->empty()) throw std::runtime_error("requested evaluation split contains no usable samples");

  const auto standardizer = flyquant::Standardizer::fit(split.train);
  standardizer.transform(split.train);
  standardizer.transform(split.validation);
  standardizer.transform(split.test);

  flyquant::LogisticRegression logistic(flyquant::kFeatureCount, cfg.logistic_learning_rate,
                                        cfg.logistic_l2, cfg.logistic_epochs);
  logistic.fit(split.train);

  const auto up = flyquant::always_up_predictions(*eval);
  const auto down = flyquant::always_down_predictions(*eval);
  const auto lr = logistic.predict(*eval);
  const auto m_up = flyquant::evaluate_predictions(up);
  const auto m_down = flyquant::evaluate_predictions(down);
  const auto m_lr = flyquant::evaluate_predictions(lr);

  auto provenance = make_provenance(cfg, raw_cfg);
  flyquant::ImmutableRunWriter writer(cfg.results_dir, provenance);
  const auto runtime_ms = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - run_started).count());
  writer.write_manifest(cfg, "baseline_bundle", args.split, runtime_ms);
  writer.write_predictions("always_up", args.split, up);
  writer.write_predictions("always_down", args.split, down);
  writer.write_predictions("logistic_regression", args.split, lr);
  writer.write_metrics({{"always_up", args.split, m_up},
                        {"always_down", args.split, m_down},
                        {"logistic_regression", args.split, m_lr}});
  writer.write_errors("");

  print_metrics("always_up", m_up);
  print_metrics("always_down", m_down);
  print_metrics("logistic_regression", m_lr);
  std::cout << "immutable_run=" << writer.directory().string() << '\n';
  return 0;
}

int inspect_connectome(const Args& args) {
  if (args.neurons.empty() || args.edges.empty()) {
    throw std::runtime_error("inspect-connectome requires --neurons and --edges");
  }
  const auto graph = flyquant::ConnectomeGraph::load_csv(
      args.neurons, args.edges, flyquant::WeightTransform::UnsignedLog1pIncomingNormalized);
  std::cout << "neurons=" << graph.neurons().size() << '\n';
  std::cout << "edges=" << graph.edges().size() << '\n';
  return 0;
}

int train_connectome(const Args& args) {
  const auto run_started = std::chrono::steady_clock::now();
  if (args.config.empty() || args.neurons.empty() || args.edges.empty() || args.sensory.empty() ||
      args.outputs.empty() || args.model_out.empty()) {
    throw std::runtime_error(
        "train-connectome requires --config --neurons --edges --sensory --outputs --model-out");
  }

  const auto raw_cfg = flyquant::KeyValueConfig::load(args.config);
  const auto cfg = flyquant::parse_experiment_config(raw_cfg);
  flyquant::validate_experiment_config(cfg);
  require_eval_split(args, cfg);

  auto split = load_split_samples(cfg, args.split);
  if (split.train.empty()) throw std::runtime_error("TRAIN contains no usable samples");
  const auto& eval = args.split == "test" ? split.test : split.validation;
  if (eval.empty()) throw std::runtime_error("requested evaluation split contains no usable samples");

  const auto weight_transform = flyquant::WeightTransform::UnsignedLog1pIncomingNormalized;
  const auto graph = flyquant::ConnectomeGraph::load_csv(args.neurons, args.edges, weight_transform);
  const auto encoder = flyquant::SensoryEncoder::load_csv(args.sensory, graph, flyquant::kFeatureCount);
  const auto decoder = flyquant::UpDownDecoder::load_csv(args.outputs, graph);

  flyquant::DynamicsConfig dynamics;
  dynamics.steps = raw_cfg.get_size("reservoir_steps", 8);
  dynamics.leak = raw_cfg.get_double("reservoir_leak", 0.5);
  dynamics.recurrent_gain = raw_cfg.get_double("reservoir_recurrent_gain", 1.0);
  dynamics.input_gain = raw_cfg.get_double("reservoir_input_gain", 1.0);

  flyquant::ReservoirTrainConfig train_cfg;
  train_cfg.epochs = raw_cfg.get_size("reservoir_epochs", 300);
  train_cfg.learning_rate = raw_cfg.get_double("reservoir_learning_rate", 0.05);
  train_cfg.l2 = raw_cfg.get_double("reservoir_l2", 1e-4);

  flyquant::ModelSourceHashes hashes;
  hashes.neurons_sha256 = flyquant::sha256_file(args.neurons);
  hashes.edges_sha256 = flyquant::sha256_file(args.edges);
  hashes.sensory_sha256 = flyquant::sha256_file(args.sensory);
  hashes.outputs_sha256 = flyquant::sha256_file(args.outputs);

  auto model = flyquant::ConnectomeReservoirModel::train(
      graph, encoder, decoder, dynamics, split.train, train_cfg, hashes);
  model.save(args.model_out);

  const auto reservoir_predictions = model.predict(graph, eval);
  const auto reservoir_metrics = flyquant::evaluate_predictions(reservoir_predictions);

  auto baseline_split = split;
  model.standardizer().transform(baseline_split.train);
  model.standardizer().transform(baseline_split.validation);
  model.standardizer().transform(baseline_split.test);
  const auto& baseline_eval = args.split == "test" ? baseline_split.test : baseline_split.validation;
  flyquant::LogisticRegression logistic(flyquant::kFeatureCount, cfg.logistic_learning_rate,
                                        cfg.logistic_l2, cfg.logistic_epochs);
  logistic.fit(baseline_split.train);
  const auto logistic_predictions = logistic.predict(baseline_eval);
  const auto up_predictions = flyquant::always_up_predictions(baseline_eval);
  const auto down_predictions = flyquant::always_down_predictions(baseline_eval);
  const auto logistic_metrics = flyquant::evaluate_predictions(logistic_predictions);
  const auto up_metrics = flyquant::evaluate_predictions(up_predictions);
  const auto down_metrics = flyquant::evaluate_predictions(down_predictions);

  auto provenance = make_provenance(cfg, raw_cfg);
  provenance.connectome_version = "neurons:" + hashes.neurons_sha256.substr(0, 12) +
                                  ";edges:" + hashes.edges_sha256.substr(0, 12);
  provenance.encoder_version = "sensory:" + hashes.sensory_sha256.substr(0, 12);
  provenance.decoder_version = "outputs:" + hashes.outputs_sha256.substr(0, 12);
  flyquant::ImmutableRunWriter writer(cfg.results_dir, provenance);

  const auto model_hash = flyquant::sha256_file(args.model_out);
  const auto runtime_ms = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - run_started).count());
  writer.write_manifest(
      cfg, "connectome_reservoir_v1", args.split, runtime_ms,
      {{"model_artifact", args.model_out.string()},
       {"model_sha256", model_hash},
       {"neurons_sha256", hashes.neurons_sha256},
       {"edges_sha256", hashes.edges_sha256},
       {"sensory_sha256", hashes.sensory_sha256},
       {"outputs_sha256", hashes.outputs_sha256},
       {"reservoir_steps", std::to_string(dynamics.steps)},
       {"reservoir_leak", std::to_string(dynamics.leak)},
       {"reservoir_recurrent_gain", std::to_string(dynamics.recurrent_gain)},
       {"reservoir_input_gain", std::to_string(dynamics.input_gain)},
       {"reservoir_epochs", std::to_string(train_cfg.epochs)},
       {"reservoir_learning_rate", std::to_string(train_cfg.learning_rate)},
       {"reservoir_l2", std::to_string(train_cfg.l2)}});
  writer.write_predictions("connectome_reservoir_v1", args.split, reservoir_predictions);
  writer.write_predictions("logistic_regression", args.split, logistic_predictions);
  writer.write_predictions("always_up", args.split, up_predictions);
  writer.write_predictions("always_down", args.split, down_predictions);
  writer.write_metrics({{"connectome_reservoir_v1", args.split, reservoir_metrics},
                        {"logistic_regression", args.split, logistic_metrics},
                        {"always_up", args.split, up_metrics},
                        {"always_down", args.split, down_metrics}});
  writer.write_errors("");

  print_metrics("connectome_reservoir_v1", reservoir_metrics);
  print_metrics("logistic_regression", logistic_metrics);
  print_metrics("always_up", up_metrics);
  print_metrics("always_down", down_metrics);
  std::cout << "model_artifact=" << args.model_out.string() << '\n';
  std::cout << "model_sha256=" << model_hash << '\n';
  std::cout << "immutable_run=" << writer.directory().string() << '\n';
  return 0;
}

int predict_connectome(const Args& args) {
  if (args.config.empty() || args.neurons.empty() || args.edges.empty() || args.model_in.empty()) {
    throw std::runtime_error("predict-connectome requires --config --neurons --edges --model-in");
  }
  const auto raw_cfg = flyquant::KeyValueConfig::load(args.config);
  const auto cfg = flyquant::parse_experiment_config(raw_cfg);
  flyquant::validate_experiment_config(cfg);
  require_eval_split(args, cfg);

  auto split = load_split_samples(cfg, args.split);
  const auto& eval = args.split == "test" ? split.test : split.validation;
  if (eval.empty()) throw std::runtime_error("requested evaluation split contains no usable samples");

  const auto neurons_hash = flyquant::sha256_file(args.neurons);
  const auto edges_hash = flyquant::sha256_file(args.edges);
  const auto graph = flyquant::ConnectomeGraph::load_csv(
      args.neurons, args.edges, flyquant::WeightTransform::UnsignedLog1pIncomingNormalized);
  const auto model = flyquant::ConnectomeReservoirModel::load(
      args.model_in, graph, neurons_hash, edges_hash);
  const auto predictions = model.predict(graph, eval);
  const auto metrics = flyquant::evaluate_predictions(predictions);
  print_metrics("connectome_reservoir_v1", metrics);
  std::cout << "model_sha256=" << flyquant::sha256_file(args.model_in) << '\n';
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto args = parse_args(argc, argv);
    if (args.command == "baseline") return run_baseline(args);
    if (args.command == "inspect-connectome") return inspect_connectome(args);
    if (args.command == "train-connectome") return train_connectome(args);
    if (args.command == "predict-connectome") return predict_connectome(args);
    throw std::runtime_error("unknown command: " + args.command);
  } catch (const std::exception& e) {
    std::cerr << "flyquant: " << e.what() << '\n';
    return 1;
  }
}
