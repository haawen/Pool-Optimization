cmake_minimum_required(VERSION 3.10)
project(pool C)

set(UNITY_DIR ext/unity/src)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include_directories(include)
include_directories(tests)

set(SOURCES
    src/pool.c
)

add_library(pool_shared SHARED ${SOURCES})
add_executable(pool src/main.c ${SOURCES})

include_directories(${UNITY_DIR})

add_executable(pool_tests
    tests/test_pool.c
    ${UNITY_DIR}/unity.c
    ${SOURCES}
)
