#ifndef PROFILING_H
#define PROFILING_H

#define WARMUP 100
#define ITERATIONS 500
#define TEST_RUNNER_ITERATIONS 3      // Rerun all TestCases (so warmup + iterations) in Random Order
#define FLUSH_SIZE (32 * 1024 * 1024) // 32MB buffer

#define TEST_CASES 5
double tolerance = 1e-6;

typedef struct
{
    float R;    // Ball radius
    float M;    // Ball mass
    float u_s1; // Coefficient of sliding friction (ball 1)
    float u_s2; // Coefficient of sliding friction (ball 2)
    float u_b;  // Coefficient of ball-ball friction
    float e_b;  // Coefficient of restitution
    int N;      // Number of iterations (deltaP is None, so we skip it)

    // Initial rvw (position, velocity, angular velocity)
    double rvw1[9];
    double rvw2[9];

    struct
    {
        double velocity[3];
        double angular[3];
    } ball1, ball2;
} CollisionData;

typedef void (*CollideBallsFn)(double *, double *, float, float, float, float, float, float, float, int, double *, double *, Profile *, Branch *);

#endif