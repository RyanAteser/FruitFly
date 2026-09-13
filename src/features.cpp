#include "flyquant/features.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>

namespace flyquant {
namespace {

std::optional<double> mid_at(const std::vector<QuoteSnapshot>& quotes,
                             TimestampNs ts,
                             TimestampNs max_age,
                             TimestampNs* event_ts = nullptr) {
  const auto q = quote_at_or_before(quotes, ts, max_age);
  if (!q) return std::nullopt;
  if (event_ts) *event_ts = q->ts_ns;
  return q->mid();
}

std::optional<double> log_return(const std::vector<QuoteSnapshot>& quotes,
                                 TimestampNs now,
                                 TimestampNs lookback,
                                 TimestampNs max_age,
                                 TimestampNs& max_feature_ts) {
  TimestampNs ts_now{};
  TimestampNs ts_then{};
  const auto p_now = mid_at(quotes, now, max_age, &ts_now);
  const auto p_then = mid_at(quotes, now - lookback, max_age, &ts_then);
  if (!p_now || !p_then || *p_now <= 0.0 || *p_then <= 0.0) return std::nullopt;
  max_feature_ts = std::max({max_feature_ts, ts_now, ts_then});
  return std::log(*p_now / *p_then);
}

std::optional<double> realized_vol_60s(const std::vector<QuoteSnapshot>& quotes,
                                       TimestampNs now,
                                       TimestampNs max_age,
                                       TimestampNs& max_feature_ts) {
  double sum_sq = 0.0;
  TimestampNs ts_prev{};
  auto prev = mid_at(quotes, now - 60 * kNsPerSecond, max_age, &ts_prev);
  if (!prev || *prev <= 0.0) return std::nullopt;
  max_feature_ts = std::max(max_feature_ts, ts_prev);
  for (int i = 59; i >= 0; --i) {
    TimestampNs ts_cur{};
    auto cur = mid_at(quotes, now - static_cast<TimestampNs>(i) * kNsPerSecond,
                      max_age, &ts_cur);
    if (!cur || *cur <= 0.0) return std::nullopt;
    const double r = std::log(*cur / *prev);
    sum_sq += r * r;
    prev = cur;
    max_feature_ts = std::max(max_feature_ts, ts_cur);
  }
  return std::sqrt(sum_sq);
}

std::pair<double, double> trade_flow_60s(const std::vector<TradePrint>& trades,
                                        TimestampNs now) {
  const auto begin = std::upper_bound(trades.begin(), trades.end(), now - 60 * kNsPerSecond,
      [](TimestampNs ts, const TradePrint& t) { return ts < t.ts_ns; });
  const auto end = std::upper_bound(trades.begin(), trades.end(), now,
      [](TimestampNs ts, const TradePrint& t) { return ts < t.ts_ns; });
  double signed_qty = 0.0;
  double total_qty = 0.0;
  std::size_t count = 0;
  for (auto it = begin; it != end; ++it) {
    total_qty += it->quantity;
    if (it->aggressor == AggressorSide::Buy) signed_qty += it->quantity;
    if (it->aggressor == AggressorSide::Sell) signed_qty -= it->quantity;
    ++count;
  }
  return {total_qty > 0.0 ? signed_qty / total_qty : 0.0,
          static_cast<double>(count) / 60.0};
}

TimestampNs ceil_to_grid(TimestampNs ts, TimestampNs period) {
  const auto rem = ts % period;
  return rem == 0 ? ts : ts + (period - rem);
}

}  // namespace

const std::array<std::string, kFeatureCount>& feature_names() {
  static const std::array<std::string, kFeatureCount> names = {
      "log_return_1s", "log_return_5s", "log_return_15s", "log_return_30s",
      "log_return_60s", "log_return_300s", "momentum_acceleration",
      "realized_volatility_60s", "relative_spread", "top_book_imbalance",
      "microprice_displacement", "signed_trade_flow_60s", "trade_arrival_intensity_60s"};
  return names;
}

std::vector<FeatureSample> build_feature_samples(const MarketData& data,
                                                 TimestampNs first_anchor_ns,
                                                 TimestampNs last_anchor_ns,
                                                 const FeatureConfig& cfg) {
  if (cfg.anchor_period_ns <= 0 || cfg.horizon_ns <= 0) {
    throw std::runtime_error("feature timing must be positive");
  }
  std::vector<FeatureSample> out;
  const auto start = ceil_to_grid(first_anchor_ns, cfg.anchor_period_ns);
  for (TimestampNs t = start; t <= last_anchor_ns; t += cfg.anchor_period_ns) {
    TimestampNs max_feature_ts = std::numeric_limits<TimestampNs>::min();
    const auto r1 = log_return(data.quotes, t, 1 * kNsPerSecond, cfg.max_quote_age_ns, max_feature_ts);
    const auto r5 = log_return(data.quotes, t, 5 * kNsPerSecond, cfg.max_quote_age_ns, max_feature_ts);
    const auto r15 = log_return(data.quotes, t, 15 * kNsPerSecond, cfg.max_quote_age_ns, max_feature_ts);
    const auto r30 = log_return(data.quotes, t, 30 * kNsPerSecond, cfg.max_quote_age_ns, max_feature_ts);
    const auto r60 = log_return(data.quotes, t, 60 * kNsPerSecond, cfg.max_quote_age_ns, max_feature_ts);
    const auto r300 = log_return(data.quotes, t, 300 * kNsPerSecond, cfg.max_quote_age_ns, max_feature_ts);
    const auto rv = realized_vol_60s(data.quotes, t, cfg.max_quote_age_ns, max_feature_ts);
    const auto q = quote_at_or_before(data.quotes, t, cfg.max_quote_age_ns);
    TimestampNs target_event_ts{};
    const auto target_mid = mid_at(data.quotes, t + cfg.horizon_ns, cfg.max_quote_age_ns, &target_event_ts);
    if (!r1 || !r5 || !r15 || !r30 || !r60 || !r300 || !rv || !q || !target_mid) continue;
    if (!(q->mid() > 0.0 && *target_mid > 0.0)) continue;
    max_feature_ts = std::max(max_feature_ts, q->ts_ns);

    const double denom = q->bid_qty + q->ask_qty;
    const double book_imbalance = denom > 0.0 ? (q->bid_qty - q->ask_qty) / denom : 0.0;
    const double rel_spread = q->spread() / q->mid();
    const double micro_disp = (q->microprice() - q->mid()) / q->mid();
    const double acceleration = (*r5 / 5.0) - (*r30 / 30.0);
    const auto [trade_flow, trade_intensity] = trade_flow_60s(data.trades, t);

    FeatureSample sample;
    sample.anchor_ts_ns = t;
    sample.max_feature_event_ts_ns = max_feature_ts;
    sample.target_ts_ns = t + cfg.horizon_ns;
    sample.target_event_ts_ns = target_event_ts;
    sample.future_log_return = std::log(*target_mid / q->mid());
    sample.y = *target_mid > q->mid() ? 1 : 0;
    sample.x = {*r1, *r5, *r15, *r30, *r60, *r300, acceleration, *rv,
                rel_spread, book_imbalance, micro_disp, trade_flow, trade_intensity};
    out.push_back(std::move(sample));
  }
  audit_no_feature_leakage(out);
  return out;
}

void audit_no_feature_leakage(const std::vector<FeatureSample>& samples) {
  TimestampNs prior_anchor = std::numeric_limits<TimestampNs>::min();
  for (const auto& s : samples) {
    if (s.x.size() != kFeatureCount) throw std::runtime_error("unexpected feature count");
    if (s.max_feature_event_ts_ns > s.anchor_ts_ns) {
      throw std::runtime_error("leakage: feature event occurs after anchor");
    }
    if (s.target_event_ts_ns > s.target_ts_ns) {
      throw std::runtime_error("invalid target: target event occurs after target timestamp");
    }
    if (s.target_ts_ns <= s.anchor_ts_ns) throw std::runtime_error("invalid target horizon");
    if (s.anchor_ts_ns <= prior_anchor) throw std::runtime_error("samples are not chronological");
    prior_anchor = s.anchor_ts_ns;
    for (const double v : s.x) {
      if (!std::isfinite(v)) throw std::runtime_error("non-finite feature value");
    }
  }
}

}  // namespace flyquant
