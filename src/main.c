#include <stdlib.h>
#include <stdio.h>
#include "pool.h"
#include "tsc_x86.h"

/**
 * set_up_data - Initialize values with hardcoded values from random b2b encounter from pooltool
 */
void set_up_data(double **rvw1, double **rvw2, float *R, float *M, float *u_s1, float *u_s2, float *u_b, float *e_b, float *deltaP, double **rvw1_result, double **rvw2_result)
{
    static double ball_1[9] = {
        0.883616,  // Pos X
        1.380029,  // Pos Y
        0.028575,  // Pos Z
        0.025551,  // Velocity X
        -0.009927, // Velocity Y
        0.000000,  // Velocity Z
        0.347411,  // Angular velocity X
        0.894168,  // Angular Velocity Y
        0.000000,  // Angular Velocity Z
    };
    static double ball_2[9] = {
        0.904487, // Pos X
        1.326826, // Pos Y
        0.028575, // Pos Z
        0.114539, // Velocity X
        0.251116, // Velocity Y
        0.000000, // Velocity Z
        8.787979, // Angular velocity X
        4.008375, // Angular Velocity Y
        0.000000, // Angular Velocity Z
    };
    *rvw1 = ball_1;
    *rvw2 = ball_2;
    *R = 0.028575;   // Radius
    *M = 0.170097;   // Mass
    *u_s1 = 0.2;     // Coefficient of sliding friction between ball 1 and the surface
    *u_s2 = 0.2;     // Coefficient of sliding friction between ball 1 and the surface
    *u_b = 0.088193; // Coefficient of friction between the balls during collision
    *e_b = 0.95;     // Coefficient of restitution between the balls
    *deltaP = 0.0;   // Normal impulse step size
    *rvw1_result = malloc(9 * sizeof(double));
    *rvw2_result = malloc(9 * sizeof(double));
}

double dummy_sink;

/**
 * warmup - Do busywork to get the CPU to raise its frequency to the base frequency
 *
 * @num_iterations: How many loops of busywork should be done
 */
void warmup(long long int num_iterations)
{
    volatile double a = 1.000001, b = 1.000002;

    for (int i = 0; i < num_iterations; i++)
    {
        a *= b;
    }
    dummy_sink = a;
}

int main()
{
    double *rvw1, *rvw2;
    float R, M, u_s1, u_s2, u_b, e_b, deltaP;
    int N = 1000;
    double *rvw1_result, *rvw2_result;

    FILE *csv_file = fopen("results.csv", "w");
    if (!csv_file)
    {
        perror("Failed to open CSV file");
        return 1;
    }

    fprintf(csv_file, "N,AverageCycles\n");

    set_up_data(&rvw1, &rvw2, &R, &M, &u_s1, &u_s2, &u_b, &e_b, &deltaP, &rvw1_result, &rvw2_result);

    warmup(1000000000);

    int exponents[] = {1, 2, 3, 4, 5, 6};
    int num_exponents = sizeof(exponents) / sizeof(exponents[0]);
    int num_runs = 30;

    for (int i = 0; i < num_exponents; i++)
    {
        int N = 1;
        for (int j = 0; j < exponents[i]; j++)
        {
            N *= 10;
        }

        unsigned long total_cycles = 0;

        for (int repeat = 0; repeat < num_runs; repeat++)
        {
            set_up_data(&rvw1, &rvw2, &R, &M, &u_s1, &u_s2, &u_b, &e_b, &deltaP, &rvw1_result, &rvw2_result);

            unsigned long start = start_tsc();
            collide_balls(rvw1, rvw2, R, M, u_s1, u_s2, u_b, e_b, deltaP, N, rvw1_result, rvw2_result);
            unsigned long end = stop_tsc(start);

            total_cycles += end;

            free(rvw1_result);
            free(rvw2_result);
        }

        double average_cycles = (double)total_cycles / num_runs;
        printf("N=%d, Average cycles = %.2f\n", N, average_cycles);
        fprintf(csv_file, "%d,%.2f\n", N, average_cycles);
    }

    fclose(csv_file);

    return 0;
}
