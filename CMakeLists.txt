cmake_minimum_required(VERSION 3.10)
project(pool C)

set(UNITY_DIR ext/unity/src)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include_directories(include)
include_directories(tests)

set(SOURCES
    src/pool.c
    src/math_helper.c
)

add_library(pool_shared SHARED ${SOURCES})

add_executable(profiling src/profiling.c ${SOURCES})
target_compile_definitions(profiling PRIVATE PROFILE)

include_directories(${UNITY_DIR})

add_executable(pool_tests
    tests/test_pool.c
    ${UNITY_DIR}/unity.c
    ${SOURCES}
)

target_compile_definitions(pool_tests PRIVATE UNITY_INCLUDE_DOUBLE)

if(UNIX)
    target_link_libraries(profiling m)
    target_link_libraries(pool_shared m)
    target_link_libraries(pool_tests m)
endif()
