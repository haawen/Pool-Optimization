#ifndef PROFILING_SPMD_H
#define PROFILING_SPMD_H

#include "pool.h"

typedef struct
{
    float R;    // Ball radius (constant across runs)
    float M;    // Ball mass (contant across runs)
    float u_s1; // Coefficient of sliding friction (ball 1) (constant across runs)
    float u_s2; // Coefficient of sliding friction (ball 2) (constant across runs)
    float e_b;  // Coefficient of restitution (constant across runs)
    int N;      // Number of iterations (deltaP is None, so we skip it)

    float col1_u_b; // Coefficient of ball-ball friction (variable across runs)
    float col2_u_b; // Coefficient of ball-ball friction (variable across runs)
    float col3_u_b; // Coefficient of ball-ball friction (variable across runs)
    float col4_u_b; // Coefficient of ball-ball friction (variable across runs)

    // Initial rvw (position, velocity, angular velocity)
    double col1_rvw1[9];
    double col1_rvw2[9];
    double col2_rvw1[9];
    double col2_rvw2[9];
    double col3_rvw1[9];
    double col3_rvw2[9];
    double col4_rvw1[9];
    double col4_rvw2[9];

    struct
    {
        double velocity[3];
        double angular[3];
    } col1_ball1, col1_ball2,
        col2_ball1, col2_ball2,
        col3_ball1, col3_ball2,
        col4_ball1, col4_ball2;
} CollisionDataSPMD;

typedef void (*CollideBallsFnSPMD)(
    double *, double *, // col1_rvw1, col1_rvw2
    double *, double *, // col2_rvw1, col2_rvw2
    double *, double *, // col3_rvw1, col3_rvw2
    double *, double *, // col4_rvw1, col4_rvw2
    float,              // R
    float,              // M
    float,              // u_s1
    float,              // u_s2
    float,              // col1_u_b
    float,              // col2_u_b
    float,              // col3_u_b
    float,              // col4_u_b
    float,              // e_b
    float,              // deltaP
    int,                // N
    double *, double *, // col1_rvw1_result, col1_rvw2_result
    double *, double *, // col2_rvw1_result, col2_rvw2_result
    double *, double *, // col3_rvw1_result, col3_rvw2_result
    double *, double *, // col4_rvw1_result, col4_rvw2_result
    Profile *, Branch *);

#endif