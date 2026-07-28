
cmake_minimum_required(VERSION 3.21)

if(POLICY CMP0167)
    cmake_policy(SET CMP0167 NEW)
endif()

if(WIN32 AND NOT DEFINED VCPKG_TARGET_TRIPLET)
    set(
        VCPKG_TARGET_TRIPLET
        "x64-windows-static-md"
        CACHE STRING
        "Use static third-party libraries with the dynamic MSVC runtime."
    )
endif()

project(market_systems_workstation VERSION 1.0 LANGUAGES CXX)

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

pybind11_add_module(quant_engine Bindings.cpp)
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

if(MSVC)
    target_compile_options(quant_engine PRIVATE /W4 /EHsc /permissive- /bigobj)
    if(MARKET_ENABLE_NATIVE_ARCH)
        target_compile_options(quant_engine PRIVATE /arch:AVX2)
    endif()
else()
    target_compile_options(quant_engine PRIVATE -Wall -Wextra -Wpedantic)
    if(MARKET_ENABLE_NATIVE_ARCH)
        target_compile_options(quant_engine PRIVATE -march=native)
    endif()
endif()

if(MARKET_BUILD_TESTS)
    include(CTest)
    if(BUILD_TESTING)
        find_package(Threads REQUIRED)
        add_executable(ring_buffer_tests tests/ring_buffer_tests.cpp)
        target_include_directories(ring_buffer_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
        target_link_libraries(ring_buffer_tests PRIVATE Threads::Threads)
        add_test(NAME ring_buffer_tests COMMAND ring_buffer_tests)

        add_executable(binary_io_tests tests/binary_io_tests.cpp)
        target_include_directories(binary_io_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
        target_link_libraries(binary_io_tests PRIVATE Threads::Threads)
        add_test(NAME binary_io_tests COMMAND binary_io_tests)

        add_executable(l2_order_book_tests tests/l2_order_book_tests.cpp)
        target_include_directories(l2_order_book_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
        target_link_libraries(l2_order_book_tests PRIVATE Threads::Threads)
        add_test(NAME l2_order_book_tests COMMAND l2_order_book_tests)

        add_executable(l2_binary_format_tests tests/l2_binary_format_tests.cpp)
        target_include_directories(l2_binary_format_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
        add_test(NAME l2_binary_format_tests COMMAND l2_binary_format_tests)

        if(MSVC)
            target_compile_options(ring_buffer_tests PRIVATE /W4 /EHsc /permissive-)
            target_compile_options(binary_io_tests PRIVATE /W4 /EHsc /permissive-)
            target_compile_options(l2_order_book_tests PRIVATE /W4 /EHsc /permissive-)
            target_compile_options(l2_binary_format_tests PRIVATE /W4 /EHsc /permissive-)
        else()
            target_compile_options(ring_buffer_tests PRIVATE -Wall -Wextra -Wpedantic)
            target_compile_options(binary_io_tests PRIVATE -Wall -Wextra -Wpedantic)
            target_compile_options(l2_order_book_tests PRIVATE -Wall -Wextra -Wpedantic)
            target_compile_options(l2_binary_format_tests PRIVATE -Wall -Wextra -Wpedantic)
        endif()
    endif()
endif()

if(MARKET_BUILD_BENCHMARKS)
    find_package(Threads REQUIRED)
    add_executable(native_queue_benchmark benchmarks/native_queue_benchmark.cpp)
    target_include_directories(native_queue_benchmark PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(native_queue_benchmark PRIVATE Threads::Threads)

    add_executable(l2_book_benchmark benchmarks/l2_book_benchmark.cpp)
    target_include_directories(l2_book_benchmark PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(l2_book_benchmark PRIVATE Threads::Threads)
    if(MSVC)
        target_compile_options(native_queue_benchmark PRIVATE /W4 /EHsc /permissive-)
        target_compile_options(l2_book_benchmark PRIVATE /W4 /EHsc /permissive-)
    else()
        target_compile_options(native_queue_benchmark PRIVATE -Wall -Wextra -Wpedantic)
        target_compile_options(l2_book_benchmark PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endif()


if(MARKET_BUILD_FUZZERS)
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        message(FATAL_ERROR "MARKET_BUILD_FUZZERS requires Clang/libFuzzer.")
    endif()
    add_executable(l2_binary_fuzzer fuzz/l2_binary_fuzzer.cpp)
    target_include_directories(l2_binary_fuzzer PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_compile_options(l2_binary_fuzzer PRIVATE -fsanitize=fuzzer,address,undefined -fno-omit-frame-pointer)
    target_link_options(l2_binary_fuzzer PRIVATE -fsanitize=fuzzer,address,undefined)
endif()
