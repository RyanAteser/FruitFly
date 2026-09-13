#pragma once

#include <filesystem>
#include <string>
#include <string_view>

namespace flyquant {

std::string sha256_string(std::string_view input);
std::string sha256_file(const std::filesystem::path& path);

}  // namespace flyquant
