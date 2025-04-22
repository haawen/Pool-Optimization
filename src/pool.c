#include "pool.h"
#include "math_helper.h"

#include <stdio.h>
#include <math.h>

DLL_EXPORT void hello_world(const char* matrix_name, double* rvw) {

    printf("Received %s\n", matrix_name);
   for(int i = 0; i < 9; i++) {
       printf("%f ", rvw[i]);
       if((i + 1) % 3 == 0) printf("\n");
   }

}

DLL_EXPORT void collide_balls(double* rvw1, double* rvw2, float R, float M, float u_s1, float u_s2, float u_b, float e_b, float deltaP, int N, double* rvw1_result, double* rvw2_result) {
    double* translation_1 = get_displacement(rvw1);
    double* velocity_1 = get_velocity(rvw1);
    double* angular_velocity_1 = get_angular_velocity(rvw1);

    double* translation_2 = get_displacement(rvw2);
    double* velocity_2 = get_velocity(rvw2);
    double* angular_velocity_2 = get_angular_velocity(rvw2);

    // Print initial states
    printf("C Initial State - Ball 1:\n");
    printf("  Position: %.6f %.6f %.6f\n", translation_1[0], translation_1[1], translation_1[2]);
    printf("  Velocity: %.6f %.6f %.6f\n", velocity_1[0], velocity_1[1], velocity_1[2]);
    printf("  Angular:  %.6f %.6f %.6f\n", angular_velocity_1[0], angular_velocity_1[1], angular_velocity_1[2]);
    
    printf("C Initial State - Ball 2:\n");
    printf("  Position: %.6f %.6f %.6f\n", translation_2[0], translation_2[1], translation_2[2]);
    printf("  Velocity: %.6f %.6f %.6f\n", velocity_2[0], velocity_2[1], velocity_2[2]);
    printf("  Angular:  %.6f %.6f %.6f\n", angular_velocity_2[0], angular_velocity_2[1], angular_velocity_2[2]);

    double offset[3];
    subV3(translation_2, translation_1, offset);

    double offset_mag_sqrd = dotV3(offset, offset);
    double offset_mag = sqrt(offset_mag_sqrd);

    double forward[3]; // Forward from ball 1 to ball 2, normalized
    divV3(offset, offset_mag, forward);

    double Z_AXIS[3] = {0, 0, 1}; // Probably up axis?

    double right[3]; // Axis orthogonal to Z and forward
    crossV3(forward, Z_AXIS, right);

    printf("\nC Local Coordinate System:\n");
    printf("  x_loc (right): %.6f %.6f %.6f\n", right[0], right[1], right[2]);
    printf("  y_loc (forward): %.6f %.6f %.6f\n", forward[0], forward[1], forward[2]);

    // Transform velocities to local frame
    double v_ix = dotV3(velocity_1, right);
    double v_iy = dotV3(velocity_1, forward);
    double v_jx = dotV3(velocity_2, right);
    double v_jy = dotV3(velocity_2, forward);

    printf("\nC Initial Local Velocities:\n");
    printf("  Ball 1: v_ix = %.6f, v_iy = %.6f\n", v_ix, v_iy);
    printf("  Ball 2: v_jx = %.6f, v_jy = %.6f\n", v_jx, v_jy);

    // Main collision loop
    double v_ijy = v_jy - v_iy;
    if (deltaP == 0) {
        deltaP = 0.5 * (1 + e_b) * M * fabs(v_ijy) / N;
    }

    double W = 0;
    double W_f = INFINITY;
    double W_c = 0;
    int niters = 0;

    while (v_ijy < 0 || W < W_f) {
        // Basic collision response (simplified version)
        double deltaV_ix = deltaP / M;
        double deltaV_iy = -deltaP / M;
        double deltaV_jx = -deltaP / M;
        double deltaV_jy = deltaP / M;

        // Update velocities
        v_ix += deltaV_ix;
        v_iy += deltaV_iy;
        v_jx += deltaV_jx;
        v_jy += deltaV_jy;

        // Update work and check compression phase
        double v_ijy0 = v_ijy;
        v_ijy = v_jy - v_iy;
        W += 0.5 * deltaP * fabs(v_ijy0 + v_ijy);
        niters++;

        if (W_c == 0 && v_ijy > 0) {
            W_c = W;
            W_f = (1 + e_b * e_b) * W_c;
        }
    }

    printf("\nC Final Local Velocities:\n");
    printf("  Ball 1: v_ix = %.6f, v_iy = %.6f\n", v_ix, v_iy);
    printf("  Ball 2: v_jx = %.6f, v_jy = %.6f\n", v_jx, v_jy);

    // Transform back to global coordinates
    double v1_final[3], v2_final[3];
    for (int i = 0; i < 3; i++) {
        v1_final[i] = v_ix * right[i] + v_iy * forward[i];
        v2_final[i] = v_jx * right[i] + v_jy * forward[i];
    }

    printf("\nC Final Global Velocities:\n");
    printf("  Ball 1: %.6f %.6f %.6f\n", v1_final[0], v1_final[1], v1_final[2]);
    printf("  Ball 2: %.6f %.6f %.6f\n", v2_final[0], v2_final[1], v2_final[2]);

    // Copy results
    memcpy(&rvw1_result[3], v1_final, sizeof(double) * 3);
    memcpy(&rvw2_result[3], v2_final, sizeof(double) * 3);
    
    printf("\n=== End C Implementation ===\n");
}

/* Assuming rvw is row-major (passed from pooltool) */
double* get_displacement(double* rvw) {
    return rvw;
}

/* Assuming rvw is row-major (passed from pooltool) */
double* get_velocity(double* rvw) {
    return &rvw[3];
}

/* Assuming rvw is row-major (passed from pooltool) */
double* get_angular_velocity(double* rvw) {
    return &rvw[6];
}
