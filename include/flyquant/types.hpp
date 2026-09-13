#pragma once

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace flyquant {

using TimestampNs = std::int64_t;
using Sequence = std::uint64_t;
using NeuronId = std::uint64_t;

constexpr TimestampNs kNsPerSecond = 1'000'000'000LL;
constexpr TimestampNs kNsPerMinute = 60LL * kNsPerSecond;
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

enum class EventType { Quote, Trade };
enum class AggressorSide { Buy, Sell, Unknown };
enum class SynapticSign : std::int8_t { Inhibitory = -1, Unknown = 0, Excitatory = 1 };

struct MarketEvent {
  TimestampNs exchange_ts_ns{};
  TimestampNs receive_ts_ns{};
  Sequence sequence{};
  EventType type{EventType::Quote};
  double bid_px{kNaN};
  double bid_qty{kNaN};
  double ask_px{kNaN};
  double ask_qty{kNaN};
  double trade_px{kNaN};
  double trade_qty{kNaN};
  AggressorSide aggressor{AggressorSide::Unknown};
};

struct Candle {
  TimestampNs open_ts_ns{};
  TimestampNs close_ts_ns{};
  double open{kNaN};
  double high{kNaN};
  double low{kNaN};
  double close{kNaN};
  double volume{};
  double buy_volume{};
  double sell_volume{};
  std::uint64_t trade_count{};
};

struct FeatureSample {
  TimestampNs anchor_ts_ns{};
  TimestampNs max_feature_event_ts_ns{};
  TimestampNs target_ts_ns{};
  TimestampNs target_event_ts_ns{};
  std::vector<double> x;
  int y{};
  double future_log_return{};
};

struct Prediction {
  TimestampNs anchor_ts_ns{};
  int y{};
  double future_log_return{};
  double p_up{};
};

}  // namespace flyquant
