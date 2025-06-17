#include "pool.h"
#include "pool_spmd.h"

#include <emmintrin.h>
#include <immintrin.h>
#include <smmintrin.h>
#include <stdio.h>
#include <math.h>
#include <stdbool.h>

#ifdef _MSC_VER
#include <intrin.h>
#include <windows.h>
#else
#include <x86intrin.h>
#endif

static inline void start_profiling_section(Profile *profile)
{
    profile->cycle_start = start_tsc();
}

static inline void end_profiling_section(Profile *profile)
{
    profile->cycles_cumulative += stop_tsc(profile->cycle_start);
}

#ifdef PROFILE

#define START_PROFILE(profile) start_profiling_section(profile)
#define END_PROFILE(profile) end_profiling_section(profile)

#else

#define START_PROFILE(profile)
#define END_PROFILE(profile)

#endif

DLL_EXPORT void spmd_4x_linear(
    double *col1_rvw1, double *col1_rvw2,
    double *col2_rvw1, double *col2_rvw2,
    double *col3_rvw1, double *col3_rvw2,
    double *col4_rvw1, double *col4_rvw2,
    float R,
    float M,
    float u_s1,
    float u_s2,
    float col1_u_b,
    float col2_u_b,
    float col3_u_b,
    float col4_u_b,
    float e_b,
    float deltaP,
    int N,
    double *col1_rvw1_result, double *col1_rvw2_result,
    double *col2_rvw1_result, double *col2_rvw2_result,
    double *col3_rvw1_result, double *col3_rvw2_result,
    double *col4_rvw1_result, double *col4_rvw2_result,
    Profile *profiles, Branch *branches)
{
#ifdef PROFILE
    Profile dummy_profile[6];
    Profile *complete_function = dummy_profile[0];
#endif
    START_PROFILE(complete_function);
    collide_balls(col1_rvw1, col1_rvw2, R, M, u_s1, u_s2, col1_u_b, e_b, deltaP, N, col1_rvw1_result, col1_rvw2_result, profiles, branches);
    collide_balls(col2_rvw1, col2_rvw2, R, M, u_s1, u_s2, col2_u_b, e_b, deltaP, N, col2_rvw1_result, col2_rvw2_result, profiles, branches);
    collide_balls(col3_rvw1, col3_rvw2, R, M, u_s1, u_s2, col3_u_b, e_b, deltaP, N, col3_rvw1_result, col3_rvw2_result, profiles, branches);
    collide_balls(col4_rvw1, col4_rvw2, R, M, u_s1, u_s2, col4_u_b, e_b, deltaP, N, col4_rvw1_result, col4_rvw2_result, profiles, branches);
    END_PROFILE(complete_function);
#ifdef PROFILE
    profiles[0] = dummy_profile[0];
#endif
}

DLL_EXPORT void spmd_4x_recip_sqrt(
    double *col1_rvw1, double *col1_rvw2,
    double *col2_rvw1, double *col2_rvw2,
    double *col3_rvw1, double *col3_rvw2,
    double *col4_rvw1, double *col4_rvw2,
    float R,
    float M,
    float u_s1,
    float u_s2,
    float col1_u_b,
    float col2_u_b,
    float col3_u_b,
    float col4_u_b,
    float e_b,
    float deltaP,
    int N,
    double *col1_rvw1_result, double *col1_rvw2_result,
    double *col2_rvw1_result, double *col2_rvw2_result,
    double *col3_rvw1_result, double *col3_rvw2_result,
    double *col4_rvw1_result, double *col4_rvw2_result,
    Profile *profiles, Branch *branches)
{
#ifdef PROFILE
    Profile dummy_profile[6];
    Profile *complete_function = dummy_profile[0];
#endif
    START_PROFILE(complete_function);
    recip_sqrt(col1_rvw1, col1_rvw2, R, M, u_s1, u_s2, col1_u_b, e_b, deltaP, N, col1_rvw1_result, col1_rvw2_result, profiles, branches);
    recip_sqrt(col2_rvw1, col2_rvw2, R, M, u_s1, u_s2, col2_u_b, e_b, deltaP, N, col2_rvw1_result, col2_rvw2_result, profiles, branches);
    recip_sqrt(col3_rvw1, col3_rvw2, R, M, u_s1, u_s2, col3_u_b, e_b, deltaP, N, col3_rvw1_result, col3_rvw2_result, profiles, branches);
    recip_sqrt(col4_rvw1, col4_rvw2, R, M, u_s1, u_s2, col4_u_b, e_b, deltaP, N, col4_rvw1_result, col4_rvw2_result, profiles, branches);
    END_PROFILE(complete_function);
#ifdef PROFILE
    profiles[0] = dummy_profile[0];
#endif
}

/**
 * SPMD version of the basic implementation of the collide_balls function.
 */
DLL_EXPORT void spmd_basic_collide_balls(
    double *col1_rvw1, double *col1_rvw2,
    double *col2_rvw1, double *col2_rvw2,
    double *col3_rvw1, double *col3_rvw2,
    double *col4_rvw1, double *col4_rvw2,
    float R,
    float M,
    float u_s1,
    float u_s2,
    float col1_u_b,
    float col2_u_b,
    float col3_u_b,
    float col4_u_b,
    float e_b,
    float deltaP,
    int N,
    double *col1_rvw1_result, double *col1_rvw2_result,
    double *col2_rvw1_result, double *col2_rvw2_result,
    double *col3_rvw1_result, double *col3_rvw2_result,
    double *col4_rvw1_result, double *col4_rvw2_result,
    Profile *profiles, Branch *branches)
{
    __asm volatile("# LLVM-MCA-BEGIN collide_balls_scalar_less_sqrt" ::: "memory");
#ifdef PROFILE
    Profile *complete_function = &profiles[0];
    Profile *before_loop = &profiles[1];
    Profile *impulse = &profiles[2];
    Profile *delta = &profiles[3];
    Profile *velocity = &profiles[4];
    Profile *after_loop = &profiles[5];
#endif
    START_PROFILE(complete_function);
    START_PROFILE(before_loop);
    // Conversion to SPMD vectors
    MEMORY(18 * 4, complete_function, before_loop)
    __m256d translation_0_rvw1 = _mm256_set_pd(col1_rvw1[0], col2_rvw1[0], col3_rvw1[0], col4_rvw1[0]);
    __m256d translation_0_rvw2 = _mm256_set_pd(col1_rvw2[0], col2_rvw2[0], col3_rvw2[0], col4_rvw2[0]);

    __m256d translation_1_rvw1 = _mm256_set_pd(col1_rvw1[1], col2_rvw1[1], col3_rvw1[1], col4_rvw1[1]);
    __m256d translation_1_rvw2 = _mm256_set_pd(col1_rvw2[1], col2_rvw2[1], col3_rvw2[1], col4_rvw2[1]);

    __m256d translation_2_rvw1 = _mm256_set_pd(col1_rvw1[2], col2_rvw1[2], col3_rvw1[2], col4_rvw1[2]);
    __m256d translation_2_rvw2 = _mm256_set_pd(col1_rvw2[2], col2_rvw2[2], col3_rvw2[2], col4_rvw2[2]);

    __m256d velocity_0_rvw1 = _mm256_set_pd(col1_rvw1[3], col2_rvw1[3], col3_rvw1[3], col4_rvw1[3]);
    __m256d velocity_0_rvw2 = _mm256_set_pd(col1_rvw2[3], col2_rvw2[3], col3_rvw2[3], col4_rvw2[3]);

    __m256d velocity_1_rvw1 = _mm256_set_pd(col1_rvw1[4], col2_rvw1[4], col3_rvw1[4], col4_rvw1[4]);
    __m256d velocity_1_rvw2 = _mm256_set_pd(col1_rvw2[4], col2_rvw2[4], col3_rvw2[4], col4_rvw2[4]);

    __m256d velocity_2_rvw1 = _mm256_set_pd(col1_rvw1[5], col2_rvw1[5], col3_rvw1[5], col4_rvw1[5]);
    __m256d velocity_2_rvw2 = _mm256_set_pd(col1_rvw2[5], col2_rvw2[5], col3_rvw2[5], col4_rvw2[5]);

    __m256d angular_velocity_0_rvw1 = _mm256_set_pd(col1_rvw1[6], col2_rvw1[6], col3_rvw1[6], col4_rvw1[6]);
    __m256d angular_velocity_0_rvw2 = _mm256_set_pd(col1_rvw2[6], col2_rvw2[6], col3_rvw2[6], col4_rvw2[6]);

    __m256d angular_velocity_1_rvw1 = _mm256_set_pd(col1_rvw1[7], col2_rvw1[7], col3_rvw1[7], col4_rvw1[7]);
    __m256d angular_velocity_1_rvw2 = _mm256_set_pd(col1_rvw2[7], col2_rvw2[7], col3_rvw2[7], col4_rvw2[7]);

    __m256d angular_velocity_2_rvw1 = _mm256_set_pd(col1_rvw1[8], col2_rvw1[8], col3_rvw1[8], col4_rvw1[8]);
    __m256d angular_velocity_2_rvw2 = _mm256_set_pd(col1_rvw2[8], col2_rvw2[8], col3_rvw2[8], col4_rvw2[8]);

    // Offset
    FLOPS(3 * 4, 0, 0, 0, complete_function, before_loop);
    __m256d offset_0 = _mm256_sub_pd(translation_0_rvw2, translation_0_rvw1);
    __m256d offset_1 = _mm256_sub_pd(translation_1_rvw2, translation_1_rvw1);
    __m256d offset_2 = _mm256_sub_pd(translation_2_rvw2, translation_2_rvw1);

    // Offset magnitude squared
    FLOPS(2 * 4, 3 * 4, 0, 0, complete_function, before_loop);
    __m256d offset_squared_0 = _mm256_mul_pd(offset_0, offset_0);
    __m256d offset_squared_1 = _mm256_mul_pd(offset_1, offset_1);
    __m256d offset_squared_2 = _mm256_mul_pd(offset_2, offset_2);
    __m256d offset_tmp = _mm256_add_pd(offset_squared_0, offset_squared_1);
    __m256d offset_magnitude_squared = _mm256_add_pd(offset_tmp, offset_squared_2);

    // Offset magnitude
    FLOPS(0, 0, 0, 1 * 4, complete_function, before_loop);
    __m256d offset_magnitude = _mm256_sqrt_pd(offset_magnitude_squared);

    // Forward
    FLOPS(0, 0, 3 * 4, 0, complete_function, before_loop);
    __m256d forward_0 = _mm256_div_pd(offset_0, offset_magnitude);
    __m256d forward_1 = _mm256_div_pd(offset_1, offset_magnitude);
    __m256d forward_2 = _mm256_div_pd(offset_2, offset_magnitude);

    // Up
    __m256d up_0 = _mm256_setzero_pd();
    __m256d up_1 = _mm256_setzero_pd();
    __m256d up_2 = _mm256_set1_pd(1.0);

    // Right
    FLOPS(3 * 4, 6 * 4, 0, 0, complete_function, before_loop);
    __m256d right_0_tmp_0 = _mm256_mul_pd(forward_1, up_2);
    __m256d right_0_tmp_1 = _mm256_mul_pd(forward_2, up_1);
    __m256d right_0 = _mm256_sub_pd(right_0_tmp_0, right_0_tmp_1);

    __m256d right_1_tmp_0 = _mm256_mul_pd(forward_2, up_0);
    __m256d right_1_tmp_1 = _mm256_mul_pd(forward_0, up_2);
    __m256d right_1 = _mm256_sub_pd(right_1_tmp_0, right_1_tmp_1);

    __m256d right_2_tmp_0 = _mm256_mul_pd(forward_0, up_1);
    __m256d right_2_tmp_1 = _mm256_mul_pd(forward_1, up_0);
    __m256d right_2 = _mm256_sub_pd(right_2_tmp_0, right_2_tmp_1);

    // Transform velocities to local frame
    FLOPS(2 * 4 * 4, 3 * 4 * 4, 0, 0, complete_function, before_loop);
    __m256d local_velocity_x_1_tmp_0 = _mm256_mul_pd(velocity_0_rvw1, right_0);
    __m256d local_velocity_x_1_tmp_1 = _mm256_mul_pd(velocity_1_rvw1, right_1);
    __m256d local_velocity_x_1_tmp_2 = _mm256_mul_pd(velocity_2_rvw1, right_2);
    __m256d local_velocity_x_1_tmp_3 = _mm256_add_pd(local_velocity_x_1_tmp_0, local_velocity_x_1_tmp_1);
    __m256d local_velocity_x_1 = _mm256_add_pd(local_velocity_x_1_tmp_3, local_velocity_x_1_tmp_2);

    __m256d local_velocity_y_1_tmp_0 = _mm256_mul_pd(velocity_0_rvw1, forward_0);
    __m256d local_velocity_y_1_tmp_1 = _mm256_mul_pd(velocity_1_rvw1, forward_1);
    __m256d local_velocity_y_1_tmp_2 = _mm256_mul_pd(velocity_2_rvw1, forward_2);
    __m256d local_velocity_y_1_tmp_3 = _mm256_add_pd(local_velocity_y_1_tmp_0, local_velocity_y_1_tmp_1);
    __m256d local_velocity_y_1 = _mm256_add_pd(local_velocity_y_1_tmp_3, local_velocity_y_1_tmp_2);

    __m256d local_velocity_x_2_tmp_0 = _mm256_mul_pd(velocity_0_rvw2, right_0);
    __m256d local_velocity_x_2_tmp_1 = _mm256_mul_pd(velocity_1_rvw2, right_1);
    __m256d local_velocity_x_2_tmp_2 = _mm256_mul_pd(velocity_2_rvw2, right_2);
    __m256d local_velocity_x_2_tmp_3 = _mm256_add_pd(local_velocity_x_2_tmp_0, local_velocity_x_2_tmp_1);
    __m256d local_velocity_x_2 = _mm256_add_pd(local_velocity_x_2_tmp_3, local_velocity_x_2_tmp_2);

    __m256d local_velocity_y_2_tmp_0 = _mm256_mul_pd(velocity_0_rvw2, forward_0);
    __m256d local_velocity_y_2_tmp_1 = _mm256_mul_pd(velocity_1_rvw2, forward_1);
    __m256d local_velocity_y_2_tmp_2 = _mm256_mul_pd(velocity_2_rvw2, forward_2);
    __m256d local_velocity_y_2_tmp_3 = _mm256_add_pd(local_velocity_y_2_tmp_0, local_velocity_y_2_tmp_1);
    __m256d local_velocity_y_2 = _mm256_add_pd(local_velocity_y_2_tmp_3, local_velocity_y_2_tmp_2);

    // Transform angular velocities to local frame
    FLOPS(2 * 6 * 4, 3 * 6 * 4, 0, 0, complete_function, before_loop);
    __m256d local_angular_velocity_x_1_tmp_0 = _mm256_mul_pd(angular_velocity_0_rvw1, right_0);
    __m256d local_angular_velocity_x_1_tmp_1 = _mm256_mul_pd(angular_velocity_1_rvw1, right_1);
    __m256d local_angular_velocity_x_1_tmp_2 = _mm256_mul_pd(angular_velocity_2_rvw1, right_2);
    __m256d local_angular_velocity_x_1_tmp_3 = _mm256_add_pd(local_angular_velocity_x_1_tmp_0, local_angular_velocity_x_1_tmp_1);
    __m256d local_angular_velocity_x_1 = _mm256_add_pd(local_angular_velocity_x_1_tmp_3, local_angular_velocity_x_1_tmp_2);

    __m256d local_angular_velocity_y_1_tmp_0 = _mm256_mul_pd(angular_velocity_0_rvw1, forward_0);
    __m256d local_angular_velocity_y_1_tmp_1 = _mm256_mul_pd(angular_velocity_1_rvw1, forward_1);
    __m256d local_angular_velocity_y_1_tmp_2 = _mm256_mul_pd(angular_velocity_2_rvw1, forward_2);
    __m256d local_angular_velocity_y_1_tmp_3 = _mm256_add_pd(local_angular_velocity_y_1_tmp_0, local_angular_velocity_y_1_tmp_1);
    __m256d local_angular_velocity_y_1 = _mm256_add_pd(local_angular_velocity_y_1_tmp_3, local_angular_velocity_y_1_tmp_2);

    __m256d local_angular_velocity_z_1_tmp_0 = _mm256_mul_pd(angular_velocity_0_rvw1, up_0);
    __m256d local_angular_velocity_z_1_tmp_1 = _mm256_mul_pd(angular_velocity_1_rvw1, up_1);
    __m256d local_angular_velocity_z_1_tmp_2 = _mm256_mul_pd(angular_velocity_2_rvw1, up_2);
    __m256d local_angular_velocity_z_1_tmp_3 = _mm256_add_pd(local_angular_velocity_z_1_tmp_0, local_angular_velocity_z_1_tmp_1);
    __m256d local_angular_velocity_z_1 = _mm256_add_pd(local_angular_velocity_z_1_tmp_3, local_angular_velocity_z_1_tmp_2);

    __m256d local_angular_velocity_x_2_tmp_0 = _mm256_mul_pd(angular_velocity_0_rvw2, right_0);
    __m256d local_angular_velocity_x_2_tmp_1 = _mm256_mul_pd(angular_velocity_1_rvw2, right_1);
    __m256d local_angular_velocity_x_2_tmp_2 = _mm256_mul_pd(angular_velocity_2_rvw2, right_2);
    __m256d local_angular_velocity_x_2_tmp_3 = _mm256_add_pd(local_angular_velocity_x_2_tmp_0, local_angular_velocity_x_2_tmp_1);
    __m256d local_angular_velocity_x_2 = _mm256_add_pd(local_angular_velocity_x_2_tmp_3, local_angular_velocity_x_2_tmp_2);

    __m256d local_angular_velocity_y_2_tmp_0 = _mm256_mul_pd(angular_velocity_0_rvw2, forward_0);
    __m256d local_angular_velocity_y_2_tmp_1 = _mm256_mul_pd(angular_velocity_1_rvw2, forward_1);
    __m256d local_angular_velocity_y_2_tmp_2 = _mm256_mul_pd(angular_velocity_2_rvw2, forward_2);
    __m256d local_angular_velocity_y_2_tmp_3 = _mm256_add_pd(local_angular_velocity_y_2_tmp_0, local_angular_velocity_y_2_tmp_1);
    __m256d local_angular_velocity_y_2 = _mm256_add_pd(local_angular_velocity_y_2_tmp_3, local_angular_velocity_y_2_tmp_2);

    __m256d local_angular_velocity_z_2_tmp_0 = _mm256_mul_pd(angular_velocity_0_rvw2, up_0);
    __m256d local_angular_velocity_z_2_tmp_1 = _mm256_mul_pd(angular_velocity_1_rvw2, up_1);
    __m256d local_angular_velocity_z_2_tmp_2 = _mm256_mul_pd(angular_velocity_2_rvw2, up_2);
    __m256d local_angular_velocity_z_2_tmp_3 = _mm256_add_pd(local_angular_velocity_z_2_tmp_0, local_angular_velocity_z_2_tmp_1);
    __m256d local_angular_velocity_z_2 = _mm256_add_pd(local_angular_velocity_z_2_tmp_3, local_angular_velocity_z_2_tmp_2);

    // Surface velocity
    __m256d spmd_R = _mm256_set1_pd(R);
    FLOPS(4 * 4, 4 * 4, 0, 0, complete_function, before_loop);
    __m256d surface_velocity_x_1_tmp = _mm256_mul_pd(local_angular_velocity_y_1, spmd_R);
    __m256d surface_velocity_x_1 = _mm256_add_pd(local_velocity_x_1, surface_velocity_x_1_tmp);

    __m256d surface_velocity_y_1_tmp = _mm256_mul_pd(local_angular_velocity_x_1, spmd_R);
    __m256d surface_velocity_y_1 = _mm256_sub_pd(local_velocity_y_1, surface_velocity_y_1_tmp);

    __m256d surface_velocity_x_2_tmp = _mm256_mul_pd(local_angular_velocity_y_2, spmd_R);
    __m256d surface_velocity_x_2 = _mm256_add_pd(local_velocity_x_2, surface_velocity_x_2_tmp);

    __m256d surface_velocity_y_2_tmp = _mm256_mul_pd(local_angular_velocity_x_2, spmd_R);
    __m256d surface_velocity_y_2 = _mm256_sub_pd(local_velocity_y_2, surface_velocity_y_2_tmp);

    // Surface Velocity Magnitude
    FLOPS(2 * 4, 4 * 4, 0, 2 * 4, complete_function, before_loop);
    __m256d surface_velocity_x_1_squared = _mm256_mul_pd(surface_velocity_x_1, surface_velocity_x_1);
    __m256d surface_velocity_y_1_squared = _mm256_mul_pd(surface_velocity_y_1, surface_velocity_y_1);
    __m256d addded_squared_1 = _mm256_add_pd(surface_velocity_x_1_squared, surface_velocity_y_1_squared);
    __m256d surface_velocity_magnitude_1 = _mm256_sqrt_pd(addded_squared_1);

    __m256d surface_velocity_x_2_squared = _mm256_mul_pd(surface_velocity_x_2, surface_velocity_x_2);
    __m256d surface_velocity_y_2_squared = _mm256_mul_pd(surface_velocity_y_2, surface_velocity_y_2);
    __m256d addded_squared_2 = _mm256_add_pd(surface_velocity_x_2_squared, surface_velocity_y_2_squared);
    __m256d surface_velocity_magnitude_2 = _mm256_sqrt_pd(addded_squared_2);

    // Ball-ball slip
    FLOPS(5 * 4, 4 * 4, 0, 1 * 4, complete_function, before_loop);
    __m256d contact_point_velocity_x_tmp_0 = _mm256_add_pd(local_angular_velocity_z_1, local_angular_velocity_z_2);
    __m256d contact_point_velocity_x_tmp_1 = _mm256_mul_pd(spmd_R, contact_point_velocity_x_tmp_0);
    __m256d contact_point_velocity_x_tmp_2 = _mm256_sub_pd(local_velocity_x_1, local_velocity_x_2);
    __m256d contact_point_velocity_x = _mm256_sub_pd(contact_point_velocity_x_tmp_2, contact_point_velocity_x_tmp_1);

    __m256d contact_point_velocity_z_tmp_0 = _mm256_add_pd(local_angular_velocity_x_1, local_angular_velocity_x_2);
    __m256d contact_point_velocity_z = _mm256_mul_pd(spmd_R, contact_point_velocity_z_tmp_0);

    __m256d ball_ball_contact_point_magnitude_tmp_0 = _mm256_mul_pd(contact_point_velocity_x, contact_point_velocity_x);
    __m256d ball_ball_contact_point_magnitude_tmp_1 = _mm256_mul_pd(contact_point_velocity_z, contact_point_velocity_z);
    __m256d ball_ball_contact_point_magnitude_tmp_2 = _mm256_add_pd(ball_ball_contact_point_magnitude_tmp_0, ball_ball_contact_point_magnitude_tmp_1);
    __m256d ball_ball_contact_point_magnitude = _mm256_sqrt_pd(ball_ball_contact_point_magnitude_tmp_2);

    // Main collision loop
    FLOPS(1 * 4, 0, 0, 0, complete_function, before_loop);
    __m256d velocity_difference_y = _mm256_sub_pd(local_velocity_y_2, local_velocity_y_1);

    // deltaP branch
    __m256d spmd_deltaP = _mm256_set1_pd(deltaP);
    FLOPS(1, 2, 0, 0, complete_function, before_loop);
    __m256d deltaP_new_tmp_0 = _mm256_set1_pd(0.5 * (1.0 + e_b) * M);
    FLOPS(0, 1, 1, 0, complete_function, before_loop);
    __m256d sign_mask = _mm256_set1_pd(-0.0);
    __m256d deltaP_new_tmp_1 = _mm256_andnot_pd(sign_mask, velocity_difference_y);
    __m256d deltaP_new_tmp_2 = _mm256_mul_pd(deltaP_new_tmp_0, deltaP_new_tmp_1);
    __m256d spmd_N = _mm256_set1_pd((double)N);
    __m256d deltaP_new = _mm256_div_pd(deltaP_new_tmp_2, spmd_N);

    __m256d zero = _mm256_setzero_pd();
    __m256d mask = _mm256_cmp_pd(spmd_deltaP, zero, _CMP_EQ_OQ);
    __m256d deltaP_vec = _mm256_blendv_pd(spmd_deltaP, deltaP_new, mask);

    // Pre-loop setup
    FLOPS(0, 2, 1, 0, complete_function, before_loop);
    __m256d spmd_C = _mm256_set1_pd(5.0 / (2.0 * M * R));
    __m256d spmd_M = _mm256_set1_pd(M);
    __m256d spmd_e_b = _mm256_set1_pd(e_b);
    __m256d total_work = _mm256_setzero_pd();
    __m256d work_required = _mm256_set1_pd(INFINITY);
    __m256d work_compression = _mm256_setzero_pd();

    __m256d eps = _mm256_set1_pd(1e-16);
    FLOPS(1 * 4, 0, 0, 0, complete_function, before_loop);
    __m256d spmd_minus_u_b = _mm256_set_pd(-col1_u_b, -col2_u_b, -col3_u_b, -col4_u_b);
    __m256d spmd_u_s1 = _mm256_set1_pd(u_s1);
    FLOPS(1, 0, 0, 0, complete_function, before_loop);
    __m256d spmd_minus_u_s2 = _mm256_set1_pd(-u_s2);

    __m256d loop_mask_tmp_0 = _mm256_cmp_pd(velocity_difference_y, zero, _CMP_LT_OQ);
    __m256d loop_mask_tmp_1 = _mm256_cmp_pd(total_work, work_required, _CMP_LT_OQ);
    __m256d loop_mask = _mm256_or_pd(loop_mask_tmp_0, loop_mask_tmp_1);

    END_PROFILE(before_loop);

    while (_mm256_movemask_pd(loop_mask) != 0)
    {
        START_PROFILE(impulse);

        // Branch 1
        FLOPS(0, 2 * 4, 1 * 4, 0, complete_function, impulse);
        __m256d deltaP_1_new_tmp_0 = _mm256_mul_pd(spmd_minus_u_b, deltaP_vec);
        __m256d deltaP_1_new_tmp_1 = _mm256_mul_pd(deltaP_1_new_tmp_0, contact_point_velocity_x);
        __m256d deltaP_1_new = _mm256_div_pd(deltaP_1_new_tmp_1, ball_ball_contact_point_magnitude);

        // Branch 3
        FLOPS(0, 2 * 4, 1 * 4, 0, complete_function, impulse);
        __m256d deltaP_2_new_tmp_0 = _mm256_mul_pd(spmd_minus_u_b, deltaP_vec);
        __m256d deltaP_2_new_tmp_1 = _mm256_mul_pd(deltaP_2_new_tmp_0, contact_point_velocity_z);
        __m256d deltaP_2_new = _mm256_div_pd(deltaP_2_new_tmp_1, ball_ball_contact_point_magnitude);

        // Branch 6
        FLOPS(0, 4 * 4, 2 * 4, 0, complete_function, impulse);
        __m256d deltaP_x_2_new_tmp_0 = _mm256_div_pd(surface_velocity_x_2, surface_velocity_magnitude_2);
        __m256d deltaP_x_2_new_tmp_1 = _mm256_mul_pd(spmd_minus_u_s2, deltaP_x_2_new_tmp_0);
        __m256d deltaP_x_2_new = _mm256_mul_pd(deltaP_x_2_new_tmp_1, deltaP_2_new);

        __m256d deltaP_y_2_new_tmp_0 = _mm256_div_pd(surface_velocity_y_2, surface_velocity_magnitude_2);
        __m256d deltaP_y_2_new_tmp_1 = _mm256_mul_pd(spmd_minus_u_s2, deltaP_y_2_new_tmp_0);
        __m256d deltaP_y_2_new = _mm256_mul_pd(deltaP_y_2_new_tmp_1, deltaP_2_new);

        // Branch 9
        FLOPS(0, 4 * 4, 2 * 4, 0, complete_function, impulse);
        __m256d deltaP_x_1_new_tmp_0 = _mm256_div_pd(surface_velocity_x_1, surface_velocity_magnitude_1);
        __m256d deltaP_x_1_new_tmp_1 = _mm256_mul_pd(spmd_u_s1, deltaP_x_1_new_tmp_0);
        __m256d deltaP_x_1_new = _mm256_mul_pd(deltaP_x_1_new_tmp_1, deltaP_2_new);

        __m256d deltaP_y_1_new_tmp_0 = _mm256_div_pd(surface_velocity_y_1, surface_velocity_magnitude_1);
        __m256d deltaP_y_1_new_tmp_1 = _mm256_mul_pd(spmd_u_s1, deltaP_y_1_new_tmp_0);
        __m256d deltaP_y_1_new = _mm256_mul_pd(deltaP_y_1_new_tmp_1, deltaP_2_new);

        // Branching logic
        __m256d mask_branch_1 = _mm256_cmp_pd(ball_ball_contact_point_magnitude, eps, _CMP_GE_OQ);

        __m256d absolute_value_contact_point_velocity_z = _mm256_andnot_pd(sign_mask, contact_point_velocity_z);
        __m256d mask_branch_3_only = _mm256_cmp_pd(absolute_value_contact_point_velocity_z, eps, _CMP_GE_OQ);
        __m256d mask_branch_3 = _mm256_and_pd(mask_branch_1, mask_branch_3_only);

        __m256d mask_branch_4_only = _mm256_cmp_pd(deltaP_2_new, zero, _CMP_GT_OQ);

        __m256d mask_branch_6_only = _mm256_cmp_pd(surface_velocity_magnitude_2, zero, _CMP_NEQ_OQ);
        __m256d mask_branch_6_tmp = _mm256_and_pd(mask_branch_3, mask_branch_4_only);
        __m256d mask_branch_6 = _mm256_and_pd(mask_branch_6_tmp, mask_branch_6_only);

        __m256d mask_branch_7_only = _mm256_cmp_pd(deltaP_2_new, zero, _CMP_LE_OQ);

        __m256d mask_branch_9_only = _mm256_cmp_pd(surface_velocity_magnitude_1, zero, _CMP_NEQ_OQ);
        __m256d mask_branch_9_tmp = _mm256_and_pd(mask_branch_3, mask_branch_7_only);
        __m256d mask_branch_9 = _mm256_and_pd(mask_branch_9_tmp, mask_branch_9_only);

        // Blending
        __m256d deltaP_1 = _mm256_blendv_pd(zero, deltaP_1_new, mask_branch_1);
        __m256d deltaP_2 = _mm256_blendv_pd(zero, deltaP_2_new, mask_branch_3);
        __m256d deltaP_x_2 = _mm256_blendv_pd(zero, deltaP_x_2_new, mask_branch_6);
        __m256d deltaP_y_2 = _mm256_blendv_pd(zero, deltaP_y_2_new, mask_branch_6);
        __m256d deltaP_x_1 = _mm256_blendv_pd(zero, deltaP_x_1_new, mask_branch_9);
        __m256d deltaP_y_1 = _mm256_blendv_pd(zero, deltaP_y_1_new, mask_branch_9);

        END_PROFILE(impulse);
        START_PROFILE(delta);

        // Velocity changes
        FLOPS(4 * 4, 0, 4 * 4, 0, complete_function, delta);
        __m256d velocity_change_x_1_tmp = _mm256_add_pd(deltaP_1, deltaP_x_1);
        __m256d velocity_change_x_1 = _mm256_div_pd(velocity_change_x_1_tmp, spmd_M);

        __m256d neg_deltaP = _mm256_xor_pd(deltaP_vec, sign_mask);
        __m256d velocity_change_y_1_tmp = _mm256_add_pd(neg_deltaP, deltaP_y_1);
        __m256d velocity_change_y_1 = _mm256_div_pd(velocity_change_y_1_tmp, spmd_M);

        __m256d neg_deltaP_1 = _mm256_xor_pd(deltaP_1, sign_mask);
        __m256d velocity_change_x_2_tmp = _mm256_add_pd(neg_deltaP_1, deltaP_x_2);
        __m256d velocity_change_x_2 = _mm256_div_pd(velocity_change_x_2_tmp, spmd_M);

        __m256d velocity_change_y_2_tmp = _mm256_add_pd(deltaP_vec, deltaP_y_2);
        __m256d velocity_change_y_2 = _mm256_div_pd(velocity_change_y_2_tmp, spmd_M);

        // Update velocities
        FLOPS(4 * 4, 0, 0, 0, complete_function, delta);
        __m256d new_local_velocity_x_1 = _mm256_add_pd(local_velocity_x_1, velocity_change_x_1);
        __m256d new_local_velocity_y_1 = _mm256_add_pd(local_velocity_y_1, velocity_change_y_1);
        __m256d new_local_velocity_x_2 = _mm256_add_pd(local_velocity_x_2, velocity_change_x_2);
        __m256d new_local_velocity_y_2 = _mm256_add_pd(local_velocity_y_2, velocity_change_y_2);

        local_velocity_x_1 = _mm256_blendv_pd(local_velocity_x_1, new_local_velocity_x_1, loop_mask);
        local_velocity_y_1 = _mm256_blendv_pd(local_velocity_y_1, new_local_velocity_y_1, loop_mask);
        local_velocity_x_2 = _mm256_blendv_pd(local_velocity_x_2, new_local_velocity_x_2, loop_mask);
        local_velocity_y_2 = _mm256_blendv_pd(local_velocity_y_2, new_local_velocity_y_2, loop_mask);

        // Angular velocity changes
        FLOPS(2 * 4, 6 * 4, 0, 0, complete_function, delta);
        __m256d delta_angular_velocity_x_1_tmp = _mm256_add_pd(deltaP_2, deltaP_y_1);
        __m256d delta_angular_velocity_x_1 = _mm256_mul_pd(spmd_C, delta_angular_velocity_x_1_tmp);

        __m256d neg_deltaP_x_1 = _mm256_xor_pd(deltaP_x_1, sign_mask);
        __m256d delta_angular_velocity_y_1 = _mm256_mul_pd(spmd_C, neg_deltaP_x_1);

        __m256d delta_angular_velocity_z_1 = _mm256_mul_pd(spmd_C, neg_deltaP_1);

        __m256d delta_angular_velocity_x_2_tmp = _mm256_add_pd(deltaP_2, deltaP_y_2);
        __m256d delta_angular_velocity_x_2 = _mm256_mul_pd(spmd_C, delta_angular_velocity_x_2_tmp);

        __m256d neg_deltaP_x_2 = _mm256_xor_pd(deltaP_x_2, sign_mask);
        __m256d delta_angular_velocity_y_2 = _mm256_mul_pd(spmd_C, neg_deltaP_x_2);

        __m256d delta_angular_velocity_z_2 = _mm256_mul_pd(spmd_C, neg_deltaP_1);

        // Update Angular Velocities
        FLOPS(6 * 4, 0, 0, 0, complete_function, delta);
        __m256d new_local_angular_velocity_x_1 = _mm256_add_pd(local_angular_velocity_x_1, delta_angular_velocity_x_1);
        __m256d new_local_angular_velocity_y_1 = _mm256_add_pd(local_angular_velocity_y_1, delta_angular_velocity_y_1);
        __m256d new_local_angular_velocity_z_1 = _mm256_add_pd(local_angular_velocity_z_1, delta_angular_velocity_z_1);
        __m256d new_local_angular_velocity_x_2 = _mm256_add_pd(local_angular_velocity_x_2, delta_angular_velocity_x_2);
        __m256d new_local_angular_velocity_y_2 = _mm256_add_pd(local_angular_velocity_y_2, delta_angular_velocity_y_2);
        __m256d new_local_angular_velocity_z_2 = _mm256_add_pd(local_angular_velocity_z_2, delta_angular_velocity_z_2);

        local_angular_velocity_x_1 = _mm256_blendv_pd(local_angular_velocity_x_1, new_local_angular_velocity_x_1, loop_mask);
        local_angular_velocity_y_1 = _mm256_blendv_pd(local_angular_velocity_y_1, new_local_angular_velocity_y_1, loop_mask);
        local_angular_velocity_z_1 = _mm256_blendv_pd(local_angular_velocity_z_1, new_local_angular_velocity_z_1, loop_mask);
        local_angular_velocity_x_2 = _mm256_blendv_pd(local_angular_velocity_x_2, new_local_angular_velocity_x_2, loop_mask);
        local_angular_velocity_y_2 = _mm256_blendv_pd(local_angular_velocity_y_2, new_local_angular_velocity_y_2, loop_mask);
        local_angular_velocity_z_2 = _mm256_blendv_pd(local_angular_velocity_z_2, new_local_angular_velocity_z_2, loop_mask);

        END_PROFILE(delta);
        START_PROFILE(velocity);

        // Update ball-table slips
        FLOPS(4 * 4, 4 * 4, 0, 0, complete_function, velocity);
        __m256d new_surface_velocity_x_1_tmp = _mm256_mul_pd(spmd_R, local_angular_velocity_y_1);
        __m256d new_surface_velocity_x_1 = _mm256_add_pd(local_velocity_x_1, new_surface_velocity_x_1_tmp);
        surface_velocity_x_1 = _mm256_blendv_pd(surface_velocity_x_1, new_surface_velocity_x_1, loop_mask);

        __m256d new_surface_velocity_y_1_tmp = _mm256_mul_pd(spmd_R, local_angular_velocity_x_1);
        __m256d new_surface_velocity_y_1 = _mm256_sub_pd(local_velocity_y_1, new_surface_velocity_y_1_tmp);
        surface_velocity_y_1 = _mm256_blendv_pd(surface_velocity_y_1, new_surface_velocity_y_1, loop_mask);

        __m256d new_surface_velocity_x_2_tmp = _mm256_mul_pd(spmd_R, local_angular_velocity_y_2);
        __m256d new_surface_velocity_x_2 = _mm256_add_pd(local_velocity_x_2, new_surface_velocity_x_2_tmp);
        surface_velocity_x_2 = _mm256_blendv_pd(surface_velocity_x_2, new_surface_velocity_x_2, loop_mask);

        __m256d new_surface_velocity_y_2_tmp = _mm256_mul_pd(spmd_R, local_angular_velocity_x_2);
        __m256d new_surface_velocity_y_2 = _mm256_sub_pd(local_velocity_y_2, new_surface_velocity_y_2_tmp);
        surface_velocity_y_2 = _mm256_blendv_pd(surface_velocity_y_2, new_surface_velocity_y_2, loop_mask);

        FLOPS(2 * 4, 4 * 4, 0, 2 * 4, complete_function, velocity);
        __m256d new_surface_velocity_x_1_squared = _mm256_mul_pd(surface_velocity_x_1, surface_velocity_x_1);
        __m256d new_surface_velocity_y_1_squared = _mm256_mul_pd(surface_velocity_y_1, surface_velocity_y_1);
        __m256d new_addded_squared_1 = _mm256_add_pd(new_surface_velocity_x_1_squared, new_surface_velocity_y_1_squared);
        __m256d new_surface_velocity_magnitude_1 = _mm256_sqrt_pd(new_addded_squared_1);
        surface_velocity_magnitude_1 = _mm256_blendv_pd(surface_velocity_magnitude_1, new_surface_velocity_magnitude_1, loop_mask);

        __m256d new_surface_velocity_x_2_squared = _mm256_mul_pd(surface_velocity_x_2, surface_velocity_x_2);
        __m256d new_surface_velocity_y_2_squared = _mm256_mul_pd(surface_velocity_y_2, surface_velocity_y_2);
        __m256d new_addded_squared_2 = _mm256_add_pd(new_surface_velocity_x_2_squared, new_surface_velocity_y_2_squared);
        __m256d new_surface_velocity_magnitude_2 = _mm256_sqrt_pd(new_addded_squared_2);
        surface_velocity_magnitude_2 = _mm256_blendv_pd(surface_velocity_magnitude_2, new_surface_velocity_magnitude_2, loop_mask);

        // Update ball-ball slip
        FLOPS(5 * 4, 4 * 4, 0, 1 * 4, complete_function, velocity);
        __m256d new_contact_point_velocity_x_tmp_0 = _mm256_add_pd(local_angular_velocity_z_1, local_angular_velocity_z_2);
        __m256d new_contact_point_velocity_x_tmp_1 = _mm256_mul_pd(spmd_R, new_contact_point_velocity_x_tmp_0);
        __m256d new_contact_point_velocity_x_tmp_2 = _mm256_sub_pd(local_velocity_x_1, local_velocity_x_2);
        __m256d new_contact_point_velocity_x = _mm256_sub_pd(new_contact_point_velocity_x_tmp_2, new_contact_point_velocity_x_tmp_1);
        contact_point_velocity_x = _mm256_blendv_pd(contact_point_velocity_x, new_contact_point_velocity_x, loop_mask);

        __m256d new_contact_point_velocity_z_tmp = _mm256_add_pd(local_angular_velocity_x_1, local_angular_velocity_x_2);
        __m256d new_contact_point_velocity_z = _mm256_mul_pd(spmd_R, new_contact_point_velocity_z_tmp);
        contact_point_velocity_z = _mm256_blendv_pd(contact_point_velocity_z, new_contact_point_velocity_z, loop_mask);

        __m256d new_ball_ball_contact_point_magnitude_tmp_0 = _mm256_mul_pd(contact_point_velocity_x, contact_point_velocity_x);
        __m256d new_ball_ball_contact_point_magnitude_tmp_1 = _mm256_mul_pd(contact_point_velocity_z, contact_point_velocity_z);
        __m256d new_ball_ball_contact_point_magnitude_tmp_2 = _mm256_add_pd(new_ball_ball_contact_point_magnitude_tmp_0, new_ball_ball_contact_point_magnitude_tmp_1);
        __m256d new_ball_ball_contact_point_magnitude = _mm256_sqrt_pd(new_ball_ball_contact_point_magnitude_tmp_2);
        ball_ball_contact_point_magnitude = _mm256_blendv_pd(ball_ball_contact_point_magnitude, new_ball_ball_contact_point_magnitude, loop_mask);

        // Update work and check compression phase
        FLOPS(3 * 4, 2 * 4, 0, 0, complete_function, velocity);
        __m256d velocity_difference_y_temp = velocity_difference_y;
        __m256d new_velocity_difference_y = _mm256_sub_pd(local_velocity_y_2, local_velocity_y_1);
        velocity_difference_y = _mm256_blendv_pd(velocity_difference_y, new_velocity_difference_y, loop_mask);
        __m256d total_work_tmp_0 = _mm256_add_pd(velocity_difference_y_temp, velocity_difference_y);
        __m256d total_work_tmp_1 = _mm256_andnot_pd(sign_mask, total_work_tmp_0);
        __m256d total_work_tmp_2 = _mm256_set1_pd(0.5);
        __m256d total_work_tmp_3 = _mm256_mul_pd(total_work_tmp_2, deltaP_vec);
        __m256d total_work_tmp_4 = _mm256_mul_pd(total_work_tmp_3, total_work_tmp_1);
        __m256d new_total_work = _mm256_add_pd(total_work, total_work_tmp_4);
        total_work = _mm256_blendv_pd(total_work, new_total_work, loop_mask);

        // Work branch
        FLOPS(1 * 4, 2 * 4, 0, 0, complete_function, velocity);
        __m256d new_work_required_tmp_0 = _mm256_set1_pd(1.0);
        __m256d new_work_required_tmp_1 = _mm256_mul_pd(spmd_e_b, spmd_e_b);
        __m256d new_work_required_tmp_2 = _mm256_add_pd(new_work_required_tmp_0, new_work_required_tmp_1);
        __m256d new_work_required = _mm256_mul_pd(new_work_required_tmp_2, total_work);

        __m256d work_compression_mask = _mm256_cmp_pd(work_compression, zero, _CMP_EQ_OQ);
        __m256d velocity_difference_y_mask = _mm256_cmp_pd(velocity_difference_y, zero, _CMP_GT_OQ);
        __m256d work_branch_mask = _mm256_and_pd(work_compression_mask, velocity_difference_y_mask);

        work_required = _mm256_blendv_pd(work_required, new_work_required, work_branch_mask);
        work_compression = _mm256_blendv_pd(work_compression, total_work, work_branch_mask);

        // Update loop mask
        __m256d new_loop_mask_tmp_0 = _mm256_cmp_pd(velocity_difference_y, zero, _CMP_LT_OQ);
        __m256d new_loop_mask_tmp_1 = _mm256_cmp_pd(total_work, work_required, _CMP_LT_OQ);
        loop_mask = _mm256_or_pd(new_loop_mask_tmp_0, new_loop_mask_tmp_1);

        END_PROFILE(velocity);
    }

    START_PROFILE(after_loop);

    FLOPS(2 * 4, 4 * 4, 0, 0, complete_function, after_loop);
    __m256d world_v1_x = _mm256_add_pd(_mm256_mul_pd(local_velocity_x_1, right_0),
                                       _mm256_mul_pd(local_velocity_y_1, forward_0));
    __m256d world_v1_y = _mm256_add_pd(_mm256_mul_pd(local_velocity_x_1, right_1),
                                       _mm256_mul_pd(local_velocity_y_1, forward_1));
    __m256d world_v1_z = _mm256_add_pd(_mm256_mul_pd(local_velocity_x_1, right_2),
                                       _mm256_mul_pd(local_velocity_y_1, forward_2));

    __m256d world_v2_x = _mm256_add_pd(_mm256_mul_pd(local_velocity_x_2, right_0),
                                       _mm256_mul_pd(local_velocity_y_2, forward_0));
    __m256d world_v2_y = _mm256_add_pd(_mm256_mul_pd(local_velocity_x_2, right_1),
                                       _mm256_mul_pd(local_velocity_y_2, forward_1));
    __m256d world_v2_z = _mm256_add_pd(_mm256_mul_pd(local_velocity_x_2, right_2),
                                       _mm256_mul_pd(local_velocity_y_2, forward_2));

    FLOPS(2 * 6 * 4, 3 * 6 * 4, 0, 0, complete_function, after_loop);
    __m256d world_w1_x = _mm256_add_pd(_mm256_mul_pd(local_angular_velocity_x_1, right_0),
                                       _mm256_mul_pd(local_angular_velocity_y_1, forward_0));
    __m256d world_w1_y = _mm256_add_pd(_mm256_mul_pd(local_angular_velocity_x_1, right_1),
                                       _mm256_mul_pd(local_angular_velocity_y_1, forward_1));
    __m256d world_w1_z = local_angular_velocity_z_1;

    __m256d world_w2_x = _mm256_add_pd(_mm256_mul_pd(local_angular_velocity_x_2, right_0),
                                       _mm256_mul_pd(local_angular_velocity_y_2, forward_0));
    __m256d world_w2_y = _mm256_add_pd(_mm256_mul_pd(local_angular_velocity_x_2, right_1),
                                       _mm256_mul_pd(local_angular_velocity_y_2, forward_1));
    __m256d world_w2_z = local_angular_velocity_z_2;

    double buf[4];

    _mm256_storeu_pd(buf, world_v1_x);
    col1_rvw1_result[3] = buf[3];
    col2_rvw1_result[3] = buf[2];
    col3_rvw1_result[3] = buf[1];
    col4_rvw1_result[3] = buf[0];
    _mm256_storeu_pd(buf, world_v1_y);
    col1_rvw1_result[4] = buf[3];
    col2_rvw1_result[4] = buf[2];
    col3_rvw1_result[4] = buf[1];
    col4_rvw1_result[4] = buf[0];
    _mm256_storeu_pd(buf, world_v1_z);
    col1_rvw1_result[5] = buf[3];
    col2_rvw1_result[5] = buf[2];
    col3_rvw1_result[5] = buf[1];
    col4_rvw1_result[5] = buf[0];

    _mm256_storeu_pd(buf, world_v2_x);
    col1_rvw2_result[3] = buf[3];
    col2_rvw2_result[3] = buf[2];
    col3_rvw2_result[3] = buf[1];
    col4_rvw2_result[3] = buf[0];
    _mm256_storeu_pd(buf, world_v2_y);
    col1_rvw2_result[4] = buf[3];
    col2_rvw2_result[4] = buf[2];
    col3_rvw2_result[4] = buf[1];
    col4_rvw2_result[4] = buf[0];
    _mm256_storeu_pd(buf, world_v2_z);
    col1_rvw2_result[5] = buf[3];
    col2_rvw2_result[5] = buf[2];
    col3_rvw2_result[5] = buf[1];
    col4_rvw2_result[5] = buf[0];

    _mm256_storeu_pd(buf, world_w1_x);
    col1_rvw1_result[6] = buf[3];
    col2_rvw1_result[6] = buf[2];
    col3_rvw1_result[6] = buf[1];
    col4_rvw1_result[6] = buf[0];
    _mm256_storeu_pd(buf, world_w1_y);
    col1_rvw1_result[7] = buf[3];
    col2_rvw1_result[7] = buf[2];
    col3_rvw1_result[7] = buf[1];
    col4_rvw1_result[7] = buf[0];
    _mm256_storeu_pd(buf, world_w1_z);
    col1_rvw1_result[8] = buf[3];
    col2_rvw1_result[8] = buf[2];
    col3_rvw1_result[8] = buf[1];
    col4_rvw1_result[8] = buf[0];

    _mm256_storeu_pd(buf, world_w2_x);
    col1_rvw2_result[6] = buf[3];
    col2_rvw2_result[6] = buf[2];
    col3_rvw2_result[6] = buf[1];
    col4_rvw2_result[6] = buf[0];
    _mm256_storeu_pd(buf, world_w2_y);
    col1_rvw2_result[7] = buf[3];
    col2_rvw2_result[7] = buf[2];
    col3_rvw2_result[7] = buf[1];
    col4_rvw2_result[7] = buf[0];
    _mm256_storeu_pd(buf, world_w2_z);
    col1_rvw2_result[8] = buf[3];
    col2_rvw2_result[8] = buf[2];
    col3_rvw2_result[8] = buf[1];
    col4_rvw2_result[8] = buf[0];

    END_PROFILE(after_loop);
    END_PROFILE(complete_function);
    __asm volatile("# LLVM-MCA-END" ::: "memory");
}