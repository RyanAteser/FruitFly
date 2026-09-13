#pragma once

#include "flyquant/market_data.hpp"
#include "flyquant/types.hpp"

#include <array>
#include <string>
#include <vector>

namespace flyquant {

inline constexpr std::size_t kFeatureCount = 13;
const std::array<std::string, kFeatureCount>& feature_names();

struct FeatureConfig {
  TimestampNs anchor_period_ns{kNsPerMinute};
  TimestampNs horizon_ns{5LL * kNsPerMinute};
  TimestampNs max_quote_age_ns{2LL * kNsPerSecond};
};

std::vector<FeatureSample> build_feature_samples(const MarketData& data,
                                                 TimestampNs first_anchor_ns,
                                                 TimestampNs last_anchor_ns,
                                                 const FeatureConfig& config);
void audit_no_feature_leakage(const std::vector<FeatureSample>& samples);

}  // namespace flyquant
