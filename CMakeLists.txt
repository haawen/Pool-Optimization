cmake_minimum_required(VERSION 3.10)
project(pool C)

set(UNITY_DIR ext/unity/src)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

include_directories(include)

set(SOURCES
    src/pool.c
    src/math_helper.c
)

add_library(pool_shared SHARED ${SOURCES})
target_compile_options(pool_shared PRIVATE -mavx)

include_directories(${UNITY_DIR})
add_executable(profiling src/profiling.c ${SOURCES} ${UNITY_DIR}/unity.c)
target_compile_definitions(profiling PRIVATE PROFILE)
target_compile_options(profiling PRIVATE -O3 -ffast-math -march=native -Wall -pedantic -mavx)
target_compile_definitions(profiling PRIVATE UNITY_INCLUDE_DOUBLE)

add_executable(benchmark src/profiling.c ${SOURCES} ${UNITY_DIR}/unity.c)
target_compile_options(benchmark PRIVATE -O3 -ffast-math -march=native -Wall -pedantic -mavx)
target_compile_definitions(benchmark PRIVATE UNITY_INCLUDE_DOUBLE)

add_executable(flops src/flops.c ${SOURCES})
target_compile_definitions(flops PRIVATE PROFILE FLOP_COUNT)
target_compile_options(flops PRIVATE -mavx)

if(UNIX)
    target_link_libraries(pool_shared m)
    target_link_libraries(profiling m)
    target_link_libraries(benchmark m)
    target_link_libraries(flops m)
endif()
