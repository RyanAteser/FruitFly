#include "flyquant/config.hpp"
#include "flyquant/connectome.hpp"
#include "flyquant/controls.hpp"
#include "flyquant/features.hpp"
#include "flyquant/logistic_regression.hpp"
#include "flyquant/neural.hpp"
#include "flyquant/sha256.hpp"
#include "flyquant/split.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void expect(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error("test failed: " + message);
}

void test_sha256() {
  expect(flyquant::sha256_string("abc") ==
             "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
         "SHA-256 known vector");
}

void test_leakage_audit() {
  flyquant::FeatureSample s;
  s.anchor_ts_ns = 100;
  s.max_feature_event_ts_ns = 101;
  s.target_ts_ns = 200;
  s.target_event_ts_ns = 200;
  s.x.assign(flyquant::kFeatureCount, 0.0);
  bool threw = false;
  try { flyquant::audit_no_feature_leakage({s}); } catch (...) { threw = true; }
  expect(threw, "future feature event must be rejected");
}

void test_feature_builder_ignores_future_quotes() {
  flyquant::MarketData a;
  flyquant::MarketData b;
  for (int sec = 0; sec <= 700; ++sec) {
    const auto ts = static_cast<flyquant::TimestampNs>(sec) * flyquant::kNsPerSecond;
    const double mid = 100.0 + 0.001 * static_cast<double>(sec);
    flyquant::QuoteSnapshot q{ts, mid - 0.5, 10.0, mid + 0.5, 10.0};
    a.quotes.push_back(q);
    b.quotes.push_back(q);
  }
  const auto anchor = 360LL * flyquant::kNsPerSecond;
  for (auto& q : b.quotes) {
    if (q.ts_ns > anchor && q.ts_ns < anchor + 300LL * flyquant::kNsPerSecond) {
      q.bid_px += 1000.0;
      q.ask_px += 1000.0;
    }
  }
  const flyquant::FeatureConfig cfg{flyquant::kNsPerMinute, 300LL * flyquant::kNsPerSecond,
                                    2LL * flyquant::kNsPerSecond};
  const auto sa = flyquant::build_feature_samples(a, anchor, anchor, cfg);
  const auto sb = flyquant::build_feature_samples(b, anchor, anchor, cfg);
  expect(sa.size() == 1 && sb.size() == 1, "causality fixture should produce one sample");
  expect(sa[0].x == sb[0].x, "future quotes must not alter current features");
}

void test_split_embargo() {
  flyquant::ExperimentConfig c;
  c.experiment_id = "test";
  c.dataset_path = "unused";
  c.horizon_ns = 300;
  c.train_start_ns = 0;
  c.train_end_ns = 1000;
  c.validation_start_ns = 1300;
  c.validation_end_ns = 2300;
  c.test_start_ns = 2600;
  c.test_end_ns = 3600;
  flyquant::validate_experiment_config(c);
  c.test_start_ns = 2400;
  bool threw = false;
  try { flyquant::validate_experiment_config(c); } catch (...) { threw = true; }
  expect(threw, "insufficient split embargo must be rejected");
}

void test_logistic() {
  std::vector<flyquant::FeatureSample> rows;
  for (int i = -20; i <= 20; ++i) {
    if (i == 0) continue;
    flyquant::FeatureSample s;
    s.x = {static_cast<double>(i)};
    s.y = i > 0 ? 1 : 0;
    rows.push_back(s);
  }
  flyquant::LogisticRegression lr(1, 0.1, 0.0, 200);
  lr.fit(rows);
  expect(lr.predict_proba({2.0}) > 0.5, "positive sample should score UP");
  expect(lr.predict_proba({-2.0}) < 0.5, "negative sample should score DOWN");
}

flyquant::ConnectomeGraph tiny_graph() {
  std::vector<flyquant::Neuron> n = {{1,"a","r","",0.0},{2,"b","r","",0.0},
                                     {3,"c","r","",0.0},{4,"d","r","",0.0}};
  std::vector<flyquant::Edge> e = {
      {0,1,3,flyquant::SynapticSign::Unknown,0.0},
      {0,2,2,flyquant::SynapticSign::Unknown,0.0},
      {1,3,5,flyquant::SynapticSign::Unknown,0.0},
      {2,3,7,flyquant::SynapticSign::Unknown,0.0}};
  return flyquant::ConnectomeGraph::from_parts(std::move(n), std::move(e),
      flyquant::WeightTransform::UnsignedLog1pIncomingNormalized);
}

void test_degree_shuffle() {
  const auto g = tiny_graph();
  const auto shuffled = flyquant::degree_preserving_shuffle(g, 42, 100);
  expect(g.in_degrees() == shuffled.in_degrees(), "shuffle preserves in-degree");
  expect(g.out_degrees() == shuffled.out_degrees(), "shuffle preserves out-degree");
}

void test_neural_decoder() {
  const auto g = tiny_graph();
  const auto encoder = flyquant::SensoryEncoder::from_assignments({{0,0,1.0,0.0}}, 1);
  const auto drive = encoder.encode({1.0}, g.neurons().size());
  const flyquant::NeuralDynamics dynamics({4,0.5,1.0,1.0});
  const auto state = dynamics.run(g, drive);
  const auto decoder = flyquant::UpDownDecoder::from_assignments(
      {{3,flyquant::OutputChannel::Up},{1,flyquant::OutputChannel::Down}});
  const auto result = decoder.decode(state);
  expect(result.p_up >= 0.0 && result.p_up <= 1.0, "decoder probability bounds");
}

}  // namespace

int main() {
  try {
    test_sha256();
    test_leakage_audit();
    test_feature_builder_ignores_future_quotes();
    test_split_embargo();
    test_logistic();
    test_degree_shuffle();
    test_neural_decoder();
    std::cout << "all tests passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
