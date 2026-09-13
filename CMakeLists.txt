cmake_minimum_required(VERSION 3.20)
project(FruitFly LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(FLYQUANT_ENABLE_WARNINGS "Enable strict compiler warnings" ON)

find_package(Git QUIET)
set(FLYQUANT_GIT_COMMIT "unknown")
if(GIT_FOUND)
  execute_process(
    COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_VARIABLE FLYQUANT_GIT_COMMIT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
endif()

add_library(flyquant_core
  src/config.cpp
  src/market_data.cpp
  src/features.cpp
  src/split.cpp
  src/logistic_regression.cpp
  src/metrics.cpp
  src/connectome.cpp
  src/controls.cpp
  src/neural.cpp
  src/sha256.cpp
  src/experiment.cpp)

target_include_directories(flyquant_core PUBLIC include)
target_compile_definitions(flyquant_core PUBLIC FLYQUANT_GIT_COMMIT="${FLYQUANT_GIT_COMMIT}")

if(FLYQUANT_ENABLE_WARNINGS)
  if(MSVC)
    target_compile_options(flyquant_core PRIVATE /W4 /permissive-)
  else()
    target_compile_options(flyquant_core PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
  endif()
endif()

add_executable(flyquant src/main.cpp)
target_link_libraries(flyquant PRIVATE flyquant_core)

include(CTest)
if(BUILD_TESTING)
  add_executable(flyquant_tests tests/test_main.cpp)
  target_link_libraries(flyquant_tests PRIVATE flyquant_core)
  add_test(NAME flyquant_tests COMMAND flyquant_tests)
endif()
