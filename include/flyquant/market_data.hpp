#pragma once

#include "flyquant/types.hpp"

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace flyquant {

struct QuoteSnapshot {
  TimestampNs ts_ns{};
  double bid_px{};
  double bid_qty{};
  double ask_px{};
  double ask_qty{};
  [[nodiscard]] double mid() const noexcept { return 0.5 * (bid_px + ask_px); }
  [[nodiscard]] double spread() const noexcept { return ask_px - bid_px; }
  [[nodiscard]] double microprice() const noexcept;
};

struct TradePrint {
  TimestampNs ts_ns{};
  double price{};
  double quantity{};
  AggressorSide aggressor{AggressorSide::Unknown};
};

struct MarketData {
  std::vector<MarketEvent> events;
  std::vector<QuoteSnapshot> quotes;
  std::vector<TradePrint> trades;
};

MarketData load_market_csv(const std::filesystem::path& path);
void validate_market_events(const std::vector<MarketEvent>& events);
std::vector<Candle> build_trade_candles(const std::vector<TradePrint>& trades,
                                        TimestampNs interval_ns);
std::optional<QuoteSnapshot> quote_at_or_before(const std::vector<QuoteSnapshot>& quotes,
                                                 TimestampNs ts_ns,
                                                 TimestampNs max_age_ns);

}  // namespace flyquant
