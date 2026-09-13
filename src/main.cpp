#include "flyquant/config.hpp"
#include "flyquant/connectome.hpp"
#include "flyquant/experiment.hpp"
#include "flyquant/features.hpp"
#include "flyquant/logistic_regression.hpp"
#include "flyquant/market_data.hpp"
#include "flyquant/metrics.hpp"
#include "flyquant/sha256.hpp"
#include "flyquant/split.hpp"

#include <chrono>
#include <exception>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>
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
  std::string split{"validation"};
  bool allow_test{false};
};

Args parse_args(int argc, char** argv) {
  if (argc < 2) throw std::runtime_error("usage: flyquant <baseline|inspect-connectome> [options]");
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
    else throw std::runtime_error("unknown argument: " + a);
  }
  return args;
}

void print_metrics(const std::string& model, const ClassificationMetrics& m) {
  std::cout << model << ": n=" << m.n
            << " accuracy=" << m.accuracy
            << " balanced_accuracy=" << m.balanced_accuracy
            << " roc_auc=" << m.roc_auc
            << " log_loss=" << m.log_loss
            << " brier=" << m.brier_score << '\n';
}

int run_baseline(const Args& args) {
  const auto run_started = std::chrono::steady_clock::now();
  if (args.config.empty()) throw std::runtime_error("baseline requires --config <path>");
  const auto raw_cfg = flyquant::KeyValueConfig::load(args.config);
  const auto cfg = flyquant::parse_experiment_config(raw_cfg);
  flyquant::validate_experiment_config(cfg);

  if (args.split != "validation" && args.split != "test") throw std::runtime_error("--split must be validation or test");
  if (args.split == "test" && cfg.test_locked && !args.allow_test) {
    throw std::runtime_error("TEST is locked; rerun with --allow-test only after the protocol is frozen");
  }

  const auto market = flyquant::load_market_csv(cfg.dataset_path);
  const flyquant::FeatureConfig feature_cfg{cfg.anchor_period_ns, cfg.horizon_ns, cfg.max_quote_age_ns};
  const auto last_anchor = (args.split == "test" ? cfg.test_end_ns : cfg.validation_end_ns) - cfg.horizon_ns;
  auto samples = flyquant::build_feature_samples(market, cfg.train_start_ns, last_anchor, feature_cfg);
  auto split = flyquant::chronological_split(samples, cfg);
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

  flyquant::RunProvenance provenance;
  provenance.experiment_id = cfg.experiment_id;
  provenance.timestamp_utc = flyquant::utc_timestamp_compact();
  provenance.git_commit = FLYQUANT_GIT_COMMIT;
  provenance.dataset_hash = flyquant::sha256_file(cfg.dataset_path);
  provenance.config_hash = flyquant::sha256_string(raw_cfg.canonical_text());
  provenance.seed = cfg.seeds.front();

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
  if (args.neurons.empty() || args.edges.empty()) throw std::runtime_error("inspect-connectome requires --neurons and --edges");
  const auto graph = flyquant::ConnectomeGraph::load_csv(
      args.neurons, args.edges, flyquant::WeightTransform::UnsignedLog1pIncomingNormalized);
  std::cout << "neurons=" << graph.neurons().size() << '\n';
  std::cout << "edges=" << graph.edges().size() << '\n';
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto args = parse_args(argc, argv);
    if (args.command == "baseline") return run_baseline(args);
    if (args.command == "inspect-connectome") return inspect_connectome(args);
    throw std::runtime_error("unknown command: " + args.command);
  } catch (const std::exception& e) {
    std::cerr << "flyquant: " << e.what() << '\n';
    return 1;
  }
}
