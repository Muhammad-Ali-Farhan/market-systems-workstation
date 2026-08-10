cmake_minimum_required(VERSION 3.21)

if(POLICY CMP0167)
    cmake_policy(SET CMP0167 NEW)
endif()

# Windows vcpkg builds use static third-party libraries linked against the
# dynamic MSVC runtime. Pinning the triplet here prevents accidental ABI drift.
if(WIN32 AND NOT DEFINED VCPKG_TARGET_TRIPLET)
    set(
        VCPKG_TARGET_TRIPLET
        "x64-windows-static-md"
        CACHE STRING
        "Use static third-party libraries with the dynamic MSVC runtime."
    )
endif()

# Keep the public project name aligned with the repository while preserving the
# existing quant_engine extension module for ABI/import compatibility.
project(market_data_execution_engine VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(MARKET_ENABLE_NATIVE_ARCH "Enable machine-specific CPU instructions" OFF)
option(MARKET_BUILD_TESTS "Build native correctness tests" ON)
option(MARKET_BUILD_BENCHMARKS "Build native microbenchmarks" OFF)
option(MARKET_BUILD_FUZZERS "Build libFuzzer targets with Clang" OFF)

find_package(Python3 3.12 EXACT REQUIRED COMPONENTS Interpreter Development.Module)

execute_process(
    COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE PYBIND11_DIRECTORY_RESULT
)
if(NOT PYBIND11_DIRECTORY_RESULT EQUAL 0 OR pybind11_DIR STREQUAL "")
    message(FATAL_ERROR "Could not locate pybind11. Install requirements-build.txt with Python 3.12.")
endif()

find_package(pybind11 CONFIG REQUIRED)
find_package(Boost REQUIRED COMPONENTS system)
find_package(OpenSSL REQUIRED)
find_package(simdjson CONFIG REQUIRED)

set(MARKET_NATIVE_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/native/include")

pybind11_add_module(quant_engine native/src/PythonBindings.cpp)
target_include_directories(quant_engine PRIVATE "${MARKET_NATIVE_INCLUDE_DIR}")
target_link_libraries(
    quant_engine
    PRIVATE
    Boost::system
    OpenSSL::SSL
    OpenSSL::Crypto
    simdjson::simdjson
)

if(WIN32)
    target_compile_definitions(
        quant_engine
        PRIVATE
        NOMINMAX
        WIN32_LEAN_AND_MEAN
        _WIN32_WINNT=0x0A00
    )
endif()

function(market_apply_warnings target)
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4 /EHsc /permissive-)
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endfunction()

market_apply_warnings(quant_engine)
if(MSVC)
    # Bindings.cpp historically required /bigobj on MSVC; keep that build
    # requirement even though the source file has moved and been renamed.
    target_compile_options(quant_engine PRIVATE /bigobj)
endif()
if(MARKET_ENABLE_NATIVE_ARCH)
    if(MSVC)
        target_compile_options(quant_engine PRIVATE /arch:AVX2)
    else()
        target_compile_options(quant_engine PRIVATE -march=native)
    endif()
endif()

if(MARKET_BUILD_TESTS)
    include(CTest)
    if(BUILD_TESTING)
        find_package(Threads REQUIRED)

        add_executable(spsc_ring_buffer_tests tests/native/spsc_ring_buffer_tests.cpp)
        target_include_directories(spsc_ring_buffer_tests PRIVATE "${MARKET_NATIVE_INCLUDE_DIR}")
        target_link_libraries(spsc_ring_buffer_tests PRIVATE Threads::Threads)
        market_apply_warnings(spsc_ring_buffer_tests)
        add_test(NAME spsc_ring_buffer_tests COMMAND spsc_ring_buffer_tests)

        add_executable(top_of_book_binary_io_tests tests/native/top_of_book_binary_io_tests.cpp)
        target_include_directories(top_of_book_binary_io_tests PRIVATE "${MARKET_NATIVE_INCLUDE_DIR}")
        target_link_libraries(top_of_book_binary_io_tests PRIVATE Threads::Threads)
        market_apply_warnings(top_of_book_binary_io_tests)
        add_test(NAME top_of_book_binary_io_tests COMMAND top_of_book_binary_io_tests)

        add_executable(l2_order_book_tests tests/native/l2_order_book_tests.cpp)
        target_include_directories(l2_order_book_tests PRIVATE "${MARKET_NATIVE_INCLUDE_DIR}")
        target_link_libraries(l2_order_book_tests PRIVATE Threads::Threads)
        market_apply_warnings(l2_order_book_tests)
        add_test(NAME l2_order_book_tests COMMAND l2_order_book_tests)

        add_executable(l2_binary_format_tests tests/native/l2_binary_format_tests.cpp)
        target_include_directories(l2_binary_format_tests PRIVATE "${MARKET_NATIVE_INCLUDE_DIR}")
        market_apply_warnings(l2_binary_format_tests)
        add_test(NAME l2_binary_format_tests COMMAND l2_binary_format_tests)
    endif()
endif()

if(MARKET_BUILD_BENCHMARKS)
    find_package(Threads REQUIRED)

    add_executable(spsc_queue_benchmark benchmarks/native/spsc_queue_benchmark.cpp)
    target_include_directories(spsc_queue_benchmark PRIVATE "${MARKET_NATIVE_INCLUDE_DIR}")
    target_link_libraries(spsc_queue_benchmark PRIVATE Threads::Threads)
    market_apply_warnings(spsc_queue_benchmark)

    add_executable(l2_order_book_benchmark benchmarks/native/l2_order_book_benchmark.cpp)
    target_include_directories(l2_order_book_benchmark PRIVATE "${MARKET_NATIVE_INCLUDE_DIR}")
    target_link_libraries(l2_order_book_benchmark PRIVATE Threads::Threads)
    market_apply_warnings(l2_order_book_benchmark)
endif()

if(MARKET_BUILD_FUZZERS)
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        message(FATAL_ERROR "MARKET_BUILD_FUZZERS requires Clang/libFuzzer.")
    endif()
    add_executable(l2_binary_fuzzer fuzz/l2_binary_fuzzer.cpp)
    target_include_directories(l2_binary_fuzzer PRIVATE "${MARKET_NATIVE_INCLUDE_DIR}")
    target_compile_options(
        l2_binary_fuzzer
        PRIVATE
        -fsanitize=fuzzer,address,undefined
        -fno-omit-frame-pointer
    )
    target_link_options(l2_binary_fuzzer PRIVATE -fsanitize=fuzzer,address,undefined)
endif()
