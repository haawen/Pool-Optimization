cmake_minimum_required(VERSION 3.10)
project(pool_benchmark C)

# Find required packages
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Benchmark-specific compiler flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -march=native")

# Include directories from main project
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

# Source files
set(SOURCES
    benchmark.c
    ../src/pool.c
    ../src/math_helper.c
)

# Benchmark executable
add_executable(pool_benchmark ${SOURCES})

if(UNIX)
    target_link_libraries(pool_benchmark m)
endif()

# Add custom target for running benchmarks
add_custom_target(run_benchmarks
    COMMAND ./pool_benchmark > benchmark_results.csv
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/plot_results.py benchmark_results.csv -o plots
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    DEPENDS pool_benchmark
)

# Verify Python dependencies
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import matplotlib, numpy"
    RESULT_VARIABLE PYTHON_DEPS_CHECK
    ERROR_QUIET
)
if(NOT PYTHON_DEPS_CHECK EQUAL 0)
    message(WARNING "matplotlib and/or numpy not found. Plotting will not work.")
endif()