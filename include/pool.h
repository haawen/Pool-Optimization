#ifndef POOL_H
#define POOL_H

#ifdef _WIN32
#define DLL_EXPORT __declspec(dllexport)
#else
#define DLL_EXPORT
#endif
#ifdef _MSC_VER
#include <intrin.h>
#include <windows.h>
#else
#include <x86intrin.h>
#endif

#include <stdint.h>
#include "tsc_x86.h"

#ifndef likely
#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)
#endif

#ifdef FLOP_COUNT

#define FLOPS(adds, muls, divs, sqrt, ...)                             \
    do                                                                 \
    {                                                                  \
        void *arr[] = {__VA_ARGS__};                                   \
        for (size_t i = 0; i < sizeof(arr) / sizeof(void *); i++)      \
        {                                                              \
            ((Profile *)arr[i])->flops += (adds + muls + divs + sqrt); \
            ((Profile *)arr[i])->ADDS += (adds);                       \
            ((Profile *)arr[i])->MULS += (muls);                       \
            ((Profile *)arr[i])->DIVS += (divs);                       \
            ((Profile *)arr[i])->SQRT += (sqrt);                       \
        }                                                              \
    } while (0)

#define MEMORY(count, ...)                                        \
    do                                                            \
    {                                                             \
        void *arr[] = {__VA_ARGS__};                              \
        for (size_t i = 0; i < sizeof(arr) / sizeof(void *); i++) \
        {                                                         \
            ((Profile *)arr[i])->memory += (count);               \
        }                                                         \
    } while (0)

#define BRANCH(i) branches[i].count++

#else

#define FLOPS(adds, muls, divs, sqrt, ...)
#define MEMORY(count, ...)
#define BRANCH(i)
#endif

// Class Ball defined
// https://github.com/ekiefl/pooltool/blob/main/pooltool/objects/ball/datatypes.py#L321
//
// Class BallState
// https://github.com/ekiefl/pooltool/blob/main/pooltool/objects/ball/datatypes.py#L70

typedef struct
{
    myInt64 cycle_start;
    myInt64 cycles_cumulative;
    long int flops;  // W
    long int memory; // Q -> loads and stores
    long int ADDS;
    long int MULS;
    long int DIVS;
    long int SQRT;
} Profile;

typedef struct
{
    long int count;
} Branch;

DLL_EXPORT myInt64 python_start_tsc();
DLL_EXPORT myInt64 python_stop_tsc(myInt64 start);

 DLL_EXPORT myInt64 python_start_tsc();
 DLL_EXPORT myInt64 python_stop_tsc(myInt64 start);

DLL_EXPORT void collide_balls(double* rvw1, double* rvw2, float R, float M, float u_s1, float u_s2, float u_b, float e_b, float deltaP, int N, double* rvw1_result, double* rvw2_result, Profile* profiles, Branch* branches);
DLL_EXPORT void collide_balls_fma(double *rvw1, double *rvw2, float R, float M, float u_s1, float u_s2, float u_b, float e_b, float deltaP, int N, double *rvw1_result, double *rvw2_result, Profile *profiles, Branch *branches);
DLL_EXPORT void scalar_improvements(double* rvw1, double* rvw2, float R, float M, float u_s1, float u_s2, float u_b, float e_b, float deltaP, int N, double* rvw1_result, double* rvw2_result, Profile* profiles, Branch* branches);
DLL_EXPORT void scalar_less_sqrt(double* rvw1, double* rvw2, float R, float M, float u_s1, float u_s2, float u_b, float e_b, float deltaP, int N, double* rvw1_result, double* rvw2_result, Profile* profiles, Branch* branches);
DLL_EXPORT void approxsq_collide_balls(
    double* rvw1, double* rvw2,
    float   R, float   M,
    float   u_s1, float u_s2,
    float   u_b,  float e_b,
    float   deltaP,  int N,
    double* rvw1_result, double* rvw2_result,
    Profile* profiles,  Branch* branches);
DLL_EXPORT void recip_sqrt(double* rvw1, double* rvw2, float R, float M, float u_s1, float u_s2, float u_b, float e_b, float deltaP, int N, double* rvw1_result, double* rvw2_result, Profile* profiles, Branch* branches);
DLL_EXPORT void recip_sqrt_masks(double *restrict rvw1, double *restrict rvw2, float R, float M, float u_s1, float u_s2, float u_b, float e_b, float deltaP, int N, double *restrict rvw1_result, double *restrict rvw2_result, Profile *profiles, Branch *branches);
DLL_EXPORT void recip_sqrt_double_while(double* rvw1, double* rvw2, float R, float M, float u_s1, float u_s2, float u_b, float e_b, float deltaP, int N, double* rvw1_result, double* rvw2_result, Profile* profiles, Branch* branches);
DLL_EXPORT void less_sqrt_collide_balls(double* rvw1, double* rvw2, float R, float M, float u_s1, float u_s2, float u_b, float e_b, float deltaP, int N, double* rvw1_result, double* rvw2_result, Profile* profiles, Branch* branches);
DLL_EXPORT void less_sqrt_collide_balls2(double* rvw1, double* rvw2, float R, float M, float u_s1, float u_s2, float u_b, float e_b, float deltaP, int N, double* rvw1_result, double* rvw2_result, Profile* profiles, Branch* branches);
DLL_EXPORT void simple_precompute_cb(double* rvw1, double* rvw2, float Rf, float Mf, float u_s1f, float u_s2f, float u_bf, float e_bf, float deltaPf, int N, double* rvw1_result, double* rvw2_result, Profile* profiles, Branch* branches);
DLL_EXPORT void branch_prediction_collide_balls(double* rvw1, double* rvw2, float R, float M, float u_s1, float u_s2, float u_b, float e_b, float deltaP, int N, double* rvw1_result, double* rvw2_result, Profile* profiles, Branch* branches);
DLL_EXPORT void remove_unused_branches(double* rvw1, double* rvw2, float R, float M, float u_s1, float u_s2, float u_b, float e_b, float deltaP, int N, double* rvw1_result, double* rvw2_result, Profile* profiles, Branch* branches);
DLL_EXPORT void code_motion_collide_balls(double* rvw1, double* rvw2, float R, float M, float u_s1, float u_s2, float u_b, float e_b, float deltaP, int N, double* rvw1_result, double* rvw2_result, Profile* profiles, Branch* branches);
DLL_EXPORT void code_motion_register_relieve(double* rvw1, double* rvw2, float R, float M, float u_s1, float u_s2, float u_b, float e_b, float deltaP, int N, double* rvw1_result, double* rvw2_result, Profile* profiles, Branch* branches);
DLL_EXPORT void code_motion_collide_balls2(double* rvw1, double* rvw2, float R, float M, float u_s1, float u_s2, float u_b, float e_b, float deltaP, int N, double* rvw1_result, double* rvw2_result, Profile* profiles, Branch* branches);
DLL_EXPORT void simd_collide_balls(double* rvw1, double* rvw2, float R, float M, float u_s1, float u_s2, float u_b, float e_b, float deltaP, int N, double* rvw1_result, double* rvw2_result, Profile* profiles, Branch* branches);
DLL_EXPORT void simd_collide_ball_2(double* rvw1, double* rvw2, float R_float, float M_float, float u_s1_float, float u_s2_float, float u_b_float, float e_b_float, float deltaP_float, int N_int, double* rvw1_result, double* rvw2_result, Profile* profiles, Branch* branches);
DLL_EXPORT void improved_symmetry_collide_balls(double* rvw1, double* rvw2, float R, float M, float u_s1, float u_s2, float u_b, float e_b, float deltaP, int N, double* rvw1_result, double* rvw2_result, Profile* profiles, Branch* branches);
//DLL_EXPORT void SIMD_Full_basic(double* rvw1, double* rvw2, float R, float M, float u_s1, float u_s2, float u_b, float e_b, float deltaP, int N, double* rvw1_result, double* rvw2_result, Profile* profiles, Branch* branches);
DLL_EXPORT void simd_scalar_loop(double* rvw1, double* rvw2, float R_float, float M_float, float u_s1_float, float u_s2_float, float u_b_float, float e_b_float, float deltaP_float, int N_int, double* rvw1_result, double* rvw2_result, Profile* profiles, Branch* branches);
DLL_EXPORT void recip_sqrt_hoist(double* rvw1, double* rvw2, float R, float M, float u_s1, float u_s2, float u_b, float e_b, float deltaP, int N, double* rvw1_result, double* rvw2_result, Profile* profiles, Branch* branches);
DLL_EXPORT void simd_ssa(double *restrict rvw1, double *restrict rvw2, float R, float M, float u_s1, float u_s2, float u_b, float e_b, float deltaP, int N, double *restrict rvw1_result, double *restrict rvw2_result, Profile *profiles, Branch *branches);
DLL_EXPORT void recip_sqrt_better_ifs(double *rvw1, double *rvw2, float R, float M, float u_s1, float u_s2, float u_b, float e_b, float deltaP, int N, double *rvw1_result, double *rvw2_result, Profile *profiles, Branch *branches);
DLL_EXPORT void recip_sqrt_less_if(double *rvw1, double *rvw2, float R, float M, float u_s1, float u_s2, float u_b, float e_b, float deltaP, int N, double *rvw1_result, double *rvw2_result, Profile *profiles, Branch *branches);

/* Assuming rvw is row-major (passed from pooltool) */
double *get_displacement(double *rvw);

/* Assuming rvw is row-major (passed from pooltool) */
double *get_velocity(double *rvw);

/* Assuming rvw is row-major (passed from pooltool) */
double *get_angular_velocity(double *rvw);

void init_profiling_section(Profile *profile);
#endif
