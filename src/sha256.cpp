#include "flyquant/sha256.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string_view>

namespace flyquant {
namespace {

constexpr std::array<std::uint32_t, 64> k = {
    0x428a2f98U,0x71374491U,0xb5c0fbcfU,0xe9b5dba5U,0x3956c25bU,0x59f111f1U,0x923f82a4U,0xab1c5ed5U,
    0xd807aa98U,0x12835b01U,0x243185beU,0x550c7dc3U,0x72be5d74U,0x80deb1feU,0x9bdc06a7U,0xc19bf174U,
    0xe49b69c1U,0xefbe4786U,0x0fc19dc6U,0x240ca1ccU,0x2de92c6fU,0x4a7484aaU,0x5cb0a9dcU,0x76f988daU,
    0x983e5152U,0xa831c66dU,0xb00327c8U,0xbf597fc7U,0xc6e00bf3U,0xd5a79147U,0x06ca6351U,0x14292967U,
    0x27b70a85U,0x2e1b2138U,0x4d2c6dfcU,0x53380d13U,0x650a7354U,0x766a0abbU,0x81c2c92eU,0x92722c85U,
    0xa2bfe8a1U,0xa81a664bU,0xc24b8b70U,0xc76c51a3U,0xd192e819U,0xd6990624U,0xf40e3585U,0x106aa070U,
    0x19a4c116U,0x1e376c08U,0x2748774cU,0x34b0bcb5U,0x391c0cb3U,0x4ed8aa4aU,0x5b9cca4fU,0x682e6ff3U,
    0x748f82eeU,0x78a5636fU,0x84c87814U,0x8cc70208U,0x90befffaU,0xa4506cebU,0xbef9a3f7U,0xc67178f2U};

std::uint32_t rotr(std::uint32_t x, std::uint32_t n) { return (x >> n) | (x << (32U - n)); }

class Sha256 {
 public:
  void update(const std::uint8_t* data, std::size_t size) {
    total_bytes_ += static_cast<std::uint64_t>(size);
    std::size_t offset = 0;
    if (buffer_size_ != 0) {
      const std::size_t take = std::min(size, buffer_.size() - buffer_size_);
      for (std::size_t i = 0; i < take; ++i) buffer_[buffer_size_ + i] = data[i];
      buffer_size_ += take;
      offset += take;
      if (buffer_size_ == buffer_.size()) { transform(buffer_.data()); buffer_size_ = 0; }
    }
    while (offset + 64 <= size) { transform(data + offset); offset += 64; }
    while (offset < size) buffer_[buffer_size_++] = data[offset++];
  }

  std::string finish() {
    const std::uint64_t bit_len = total_bytes_ * 8ULL;
    buffer_[buffer_size_++] = 0x80U;
    if (buffer_size_ > 56) {
      while (buffer_size_ < 64) buffer_[buffer_size_++] = 0U;
      transform(buffer_.data());
      buffer_size_ = 0;
    }
    while (buffer_size_ < 56) buffer_[buffer_size_++] = 0U;
    for (int i = 7; i >= 0; --i) {
      buffer_[buffer_size_++] = static_cast<std::uint8_t>((bit_len >> (8U * static_cast<unsigned>(i))) & 0xffU);
    }
    transform(buffer_.data());
    buffer_size_ = 0;
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (const auto x : h_) out << std::setw(8) << x;
    return out.str();
  }

 private:
  void transform(const std::uint8_t* block) {
    std::array<std::uint32_t, 64> w{};
    for (std::size_t i = 0; i < 16; ++i) {
      const std::size_t j = i * 4;
      w[i] = (static_cast<std::uint32_t>(block[j]) << 24U) |
             (static_cast<std::uint32_t>(block[j + 1]) << 16U) |
             (static_cast<std::uint32_t>(block[j + 2]) << 8U) |
             static_cast<std::uint32_t>(block[j + 3]);
    }
    for (std::size_t i = 16; i < 64; ++i) {
      const std::uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3U);
      const std::uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10U);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }

    auto [a,b,c,d,e,f,g,hh] = h_;
    for (std::size_t i = 0; i < 64; ++i) {
      const std::uint32_t s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
      const std::uint32_t ch = (e & f) ^ ((~e) & g);
      const std::uint32_t temp1 = hh + s1 + ch + k[i] + w[i];
      const std::uint32_t s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
      const std::uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
      const std::uint32_t temp2 = s0 + maj;
      hh = g; g = f; f = e; e = d + temp1; d = c; c = b; b = a; a = temp1 + temp2;
    }
    h_[0] += a; h_[1] += b; h_[2] += c; h_[3] += d;
    h_[4] += e; h_[5] += f; h_[6] += g; h_[7] += hh;
  }

  std::array<std::uint32_t, 8> h_ = {0x6a09e667U,0xbb67ae85U,0x3c6ef372U,0xa54ff53aU,
                                      0x510e527fU,0x9b05688cU,0x1f83d9abU,0x5be0cd19U};
  std::array<std::uint8_t, 64> buffer_{};
  std::size_t buffer_size_{};
  std::uint64_t total_bytes_{};
};

}  // namespace

std::string sha256_string(std::string_view input) {
  Sha256 sha;
  sha.update(reinterpret_cast<const std::uint8_t*>(input.data()), input.size());
  return sha.finish();
}

std::string sha256_file(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot hash file: " + path.string());
  Sha256 sha;
  std::array<char, 64 * 1024> buffer{};
  while (in) {
    in.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    const auto n = in.gcount();
    if (n > 0) sha.update(reinterpret_cast<const std::uint8_t*>(buffer.data()), static_cast<std::size_t>(n));
  }
  return sha.finish();
}

}  // namespace flyquant
