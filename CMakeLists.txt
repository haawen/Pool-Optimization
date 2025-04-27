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
add_executable(pool src/main.c ${SOURCES})

# set(GCC_COMPILER_FLAGS "-O3" "-march=native" "-ffast-math" "-ftree-vectorize")
set(GCC_COMPILER_FLAGS "-O3" "-march=native" "-ffast-math" "-ftree-vectorize")

# Compile down to asm to see what gets optimized away. Hardcoded for GCC.
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/main.s
    COMMAND ${CMAKE_C_COMPILER} ${GCC_COMPILER_FLAGS} -S -I${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src/main.c -o ${CMAKE_CURRENT_BINARY_DIR}/main.s
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/main.c
    COMMENT "Generating assembly for main.c"
)

add_custom_target(asm_pool DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/main.s)

include_directories(${UNITY_DIR})

add_executable(pool_tests
    tests/test_pool.c
    ${UNITY_DIR}/unity.c
    ${SOURCES}
)

target_compile_options(pool PRIVATE
  "$<$<C_COMPILER_ID:GNU>:${GCC_COMPILER_FLAGS}>"
)

if(UNIX)
    target_link_libraries(pool m)
    target_link_libraries(pool_shared m)
    target_link_libraries(pool_tests m)
endif()