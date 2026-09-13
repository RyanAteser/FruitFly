#include "flyquant/market_data.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace flyquant {
namespace {

std::vector<std::string> split_csv_line(const std::string& line) {
  std::vector<std::string> out;
  std::string current;
  bool quoted = false;
  for (std::size_t i = 0; i < line.size(); ++i) {
    const char c = line[i];
    if (c == '"') {
      if (quoted && i + 1 < line.size() && line[i + 1] == '"') {
        current.push_back('"');
        ++i;
      } else {
        quoted = !quoted;
      }
    } else if (c == ',' && !quoted) {
      out.push_back(current);
      current.clear();
    } else {
      current.push_back(c);
    }
  }
  out.push_back(current);
  if (quoted) throw std::runtime_error("unterminated CSV quote");
  return out;
}

double parse_optional_double(const std::string& s) { return s.empty() ? kNaN : std::stod(s); }

EventType parse_event_type(const std::string& s) {
  if (s == "QUOTE") return EventType::Quote;
  if (s == "TRADE") return EventType::Trade;
  throw std::runtime_error("invalid event_type: " + s);
}

AggressorSide parse_aggressor(const std::string& s) {
  if (s.empty() || s == "U" || s == "UNKNOWN") return AggressorSide::Unknown;
  if (s == "B" || s == "BUY") return AggressorSide::Buy;
  if (s == "S" || s == "SELL") return AggressorSide::Sell;
  throw std::runtime_error("invalid aggressor: " + s);
}

std::size_t require_column(const std::unordered_map<std::string, std::size_t>& columns,
                           const std::string& name) {
  const auto it = columns.find(name);
  if (it == columns.end()) throw std::runtime_error("missing CSV column: " + name);
  return it->second;
}

}  // namespace

double QuoteSnapshot::microprice() const noexcept {
  const double denom = bid_qty + ask_qty;
  if (!(denom > 0.0)) return mid();
  return (ask_px * bid_qty + bid_px * ask_qty) / denom;
}

MarketData load_market_csv(const std::filesystem::path& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open market CSV: " + path.string());

  std::string line;
  if (!std::getline(in, line)) throw std::runtime_error("market CSV is empty");
  const auto header = split_csv_line(line);
  std::unordered_map<std::string, std::size_t> columns;
  for (std::size_t i = 0; i < header.size(); ++i) columns[header[i]] = i;

  const auto c_exchange = require_column(columns, "exchange_ts_ns");
  const auto c_receive = require_column(columns, "receive_ts_ns");
  const auto c_sequence = require_column(columns, "sequence");
  const auto c_type = require_column(columns, "event_type");
  const auto c_bid_px = require_column(columns, "bid_px");
  const auto c_bid_qty = require_column(columns, "bid_qty");
  const auto c_ask_px = require_column(columns, "ask_px");
  const auto c_ask_qty = require_column(columns, "ask_qty");
  const auto c_trade_px = require_column(columns, "trade_px");
  const auto c_trade_qty = require_column(columns, "trade_qty");
  const auto c_aggressor = require_column(columns, "aggressor");
  const auto max_col = std::max({c_exchange, c_receive, c_sequence, c_type, c_bid_px, c_bid_qty,
                                 c_ask_px, c_ask_qty, c_trade_px, c_trade_qty, c_aggressor});

  MarketData data;
  std::size_t line_no = 1;
  while (std::getline(in, line)) {
    ++line_no;
    if (line.empty()) continue;
    const auto f = split_csv_line(line);
    if (f.size() <= max_col) throw std::runtime_error("short CSV row at line " + std::to_string(line_no));

    MarketEvent e;
    e.exchange_ts_ns = std::stoll(f[c_exchange]);
    e.receive_ts_ns = std::stoll(f[c_receive]);
    e.sequence = std::stoull(f[c_sequence]);
    e.type = parse_event_type(f[c_type]);
    e.bid_px = parse_optional_double(f[c_bid_px]);
    e.bid_qty = parse_optional_double(f[c_bid_qty]);
    e.ask_px = parse_optional_double(f[c_ask_px]);
    e.ask_qty = parse_optional_double(f[c_ask_qty]);
    e.trade_px = parse_optional_double(f[c_trade_px]);
    e.trade_qty = parse_optional_double(f[c_trade_qty]);
    e.aggressor = parse_aggressor(f[c_aggressor]);
    data.events.push_back(e);
  }

  validate_market_events(data.events);
  data.quotes.reserve(data.events.size());
  data.trades.reserve(data.events.size());
  for (const auto& e : data.events) {
    if (e.type == EventType::Quote) {
      if (!(std::isfinite(e.bid_px) && std::isfinite(e.bid_qty) && std::isfinite(e.ask_px) &&
            std::isfinite(e.ask_qty) && e.bid_px > 0.0 && e.ask_px >= e.bid_px &&
            e.bid_qty >= 0.0 && e.ask_qty >= 0.0)) {
        throw std::runtime_error("invalid quote event");
      }
      data.quotes.push_back({e.exchange_ts_ns, e.bid_px, e.bid_qty, e.ask_px, e.ask_qty});
    } else {
      if (!(std::isfinite(e.trade_px) && std::isfinite(e.trade_qty) && e.trade_px > 0.0 &&
            e.trade_qty > 0.0)) {
        throw std::runtime_error("invalid trade event");
      }
      data.trades.push_back({e.exchange_ts_ns, e.trade_px, e.trade_qty, e.aggressor});
    }
  }
  if (data.quotes.empty()) throw std::runtime_error("market CSV contains no quotes");
  return data;
}

void validate_market_events(const std::vector<MarketEvent>& events) {
  for (std::size_t i = 1; i < events.size(); ++i) {
    const auto& a = events[i - 1];
    const auto& b = events[i];
    if (b.exchange_ts_ns < a.exchange_ts_ns ||
        (b.exchange_ts_ns == a.exchange_ts_ns && b.sequence <= a.sequence)) {
      throw std::runtime_error("market events are not strictly ordered by (exchange_ts_ns, sequence)");
    }
  }
}

std::vector<Candle> build_trade_candles(const std::vector<TradePrint>& trades,
                                        TimestampNs interval_ns) {
  if (interval_ns <= 0) throw std::runtime_error("candle interval must be positive");
  std::vector<Candle> out;
  if (trades.empty()) return out;

  Candle current;
  current.open_ts_ns = (trades.front().ts_ns / interval_ns) * interval_ns;
  current.close_ts_ns = current.open_ts_ns + interval_ns;
  auto add_trade = [](Candle& c, const TradePrint& t) {
    if (c.trade_count == 0) c.open = c.high = c.low = c.close = t.price;
    c.high = std::max(c.high, t.price);
    c.low = std::min(c.low, t.price);
    c.close = t.price;
    c.volume += t.quantity;
    if (t.aggressor == AggressorSide::Buy) c.buy_volume += t.quantity;
    if (t.aggressor == AggressorSide::Sell) c.sell_volume += t.quantity;
    ++c.trade_count;
  };

  for (const auto& t : trades) {
    const auto bucket = (t.ts_ns / interval_ns) * interval_ns;
    if (bucket != current.open_ts_ns) {
      out.push_back(current);
      current = Candle{};
      current.open_ts_ns = bucket;
      current.close_ts_ns = bucket + interval_ns;
    }
    add_trade(current, t);
  }
  out.push_back(current);
  return out;
}

std::optional<QuoteSnapshot> quote_at_or_before(const std::vector<QuoteSnapshot>& quotes,
                                                 TimestampNs ts_ns,
                                                 TimestampNs max_age_ns) {
  const auto it = std::upper_bound(quotes.begin(), quotes.end(), ts_ns,
      [](TimestampNs ts, const QuoteSnapshot& q) { return ts < q.ts_ns; });
  if (it == quotes.begin()) return std::nullopt;
  const auto& q = *std::prev(it);
  if (q.ts_ns > ts_ns) return std::nullopt;
  if (max_age_ns >= 0 && ts_ns - q.ts_ns > max_age_ns) return std::nullopt;
  return q;
}

}  // namespace flyquant
