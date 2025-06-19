cmake_minimum_required(VERSION 3.10)
project(pool C)

# set(CMAKE_C_COMPILER "clang")

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

set(FLAGS
    # -O3
    -mfma
    -mavx2
    -Ofast
    # #-march=native # unpredictable, most of the times worsens
    # -Wall
    # -pedantic
    -fno-math-errno
    -funroll-loops
    -ftree-vectorize
    #-fno-omit-frame-pointer

    #-flto          # Worsens
    #-g
)

add_library(pool_shared SHARED ${SOURCES})
target_compile_options(pool_shared PRIVATE ${FLAGS})

include_directories(${UNITY_DIR})

add_executable(profiling src/profiling.c ${SOURCES} ${UNITY_DIR}/unity.c)
target_compile_definitions(profiling PRIVATE PROFILE)
target_compile_options(profiling PRIVATE ${FLAGS})
target_compile_definitions(profiling PRIVATE UNITY_INCLUDE_DOUBLE)


add_executable(benchmark src/profiling.c ${SOURCES} ${UNITY_DIR}/unity.c)
target_compile_options(benchmark PRIVATE ${FLAGS} -g)
target_compile_definitions(benchmark PRIVATE UNITY_INCLUDE_DOUBLE)

add_executable(benchmark_n src/profiling_multiple_n.c ${SOURCES} ${UNITY_DIR}/unity.c)
target_compile_options(benchmark_n PRIVATE ${FLAGS} -g)
target_compile_definitions(benchmark_n PRIVATE UNITY_INCLUDE_DOUBLE)

add_executable(single_performance src/single_performance.c ${SOURCES} )
target_compile_options(single_performance PRIVATE
    ${FLAGS}
    -g # Debug information, not a optimization, used for valgrind
)

add_executable(flops src/flops.c ${SOURCES})
target_compile_definitions(flops PRIVATE PROFILE FLOP_COUNT)
target_compile_options(flops PRIVATE ${FLAGS})

add_custom_target(collide_balls_asm
  COMMAND ${CMAKE_C_COMPILER}
          ${FLAGS}
          -I${CMAKE_SOURCE_DIR}/include
          -I${CMAKE_SOURCE_DIR}/${UNITY_DIR}
          -S
          -o ${CMAKE_BINARY_DIR}/pool.s
          ${CMAKE_SOURCE_DIR}/src/pool.c
)

# SPMD compilations
add_executable(spmd_profiling src/profiling_spmd.c ${SOURCES} src/pool_spmd.c ${UNITY_DIR}/unity.c)
target_compile_definitions(spmd_profiling PRIVATE PROFILE)
target_compile_options(spmd_profiling PRIVATE ${FLAGS})
target_compile_definitions(spmd_profiling PRIVATE UNITY_INCLUDE_DOUBLE)

add_executable(spmd_benchmark src/profiling_spmd.c ${SOURCES} src/pool_spmd.c ${UNITY_DIR}/unity.c)
target_compile_options(spmd_benchmark PRIVATE ${FLAGS} -g)
target_compile_definitions(spmd_benchmark PRIVATE UNITY_INCLUDE_DOUBLE)

add_executable(spmd_single_performance src/single_performance_spmd.c ${SOURCES} src/pool_spmd.c)
target_compile_options(spmd_single_performance PRIVATE ${FLAGS} -g)

add_executable(spmd_flops src/flops_spmd.c ${SOURCES} src/pool_spmd.c)
target_compile_options(spmd_flops PRIVATE ${FLAGS})
target_compile_definitions(spmd_flops PRIVATE PROFILE FLOP_COUNT)

if(UNIX)
    target_link_libraries(pool_shared m)
    target_link_libraries(profiling m)
    target_link_libraries(benchmark_n m)
    target_link_libraries(single_performance m)
    target_link_libraries(benchmark m)
    target_link_libraries(flops m)
endif()
