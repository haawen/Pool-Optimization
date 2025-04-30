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

include_directories(${UNITY_DIR})
add_executable(profiling src/profiling.c ${SOURCES} ${UNITY_DIR}/unity.c)
target_compile_definitions(profiling PRIVATE PROFILE)
target_compile_definitions(profiling PRIVATE FLOP_COUNT)
target_compile_options(profiling PRIVATE -O3 -ffast-math -march=native -Wall -Wextra -pedantic)
target_compile_definitions(profiling PRIVATE UNITY_INCLUDE_DOUBLE)

if(UNIX)
    target_link_libraries(profiling m)
    target_link_libraries(pool_shared m)
endif()
