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

    // From here on, it is assumed that the x axis is the right axis and y axis is the forward axis
    // Transform velocities to local frame
    double local_velocity_x_1 = dotV3(velocity_1, right);
    double local_velocity_y_1 = dotV3(velocity_1, forward);
    double local_velocity_x_2 = dotV3(velocity_2, right);
    double local_velocity_y_2 = dotV3(velocity_2, forward);

    printf("\nC Initial Local Velocities:\n");
    printf("  Ball 1: v_ix = %.6f, v_iy = %.6f\n", local_velocity_x_1, local_velocity_y_1);
    printf("  Ball 2: v_jx = %.6f, v_jy = %.6f\n", local_velocity_x_2, local_velocity_y_2);

    // Main collision loop
    double velocity_diff_y = local_velocity_y_2 - local_velocity_y_1;

    // deltaP is most likely always 0?
    // ΔP represents the Impulse during a time of Δt
    if (deltaP == 0) {
        deltaP = 0.5f * (1.0f + e_b) * M * fabs(velocity_diff_y) / N;
    }

    double total_work = 0; // Work done due to impulse force
    double work_required = INFINITY; // Total amount of work required before collision handling is complete
    double work_compression = 0;

    while (velocity_diff_y < 0 || total_work < work_required) {
        // Basic collision response
        // Velocity changes
        double velocity_change_x_1 = deltaP / M;
        double velocity_change_y_1 = -deltaP / M;
        double velocity_change_x_2 = -deltaP / M;
        double velocity_change_y_2 = deltaP / M;

        // Update velocities
        local_velocity_x_1 += velocity_change_x_1;
        local_velocity_y_1 += velocity_change_y_1;
        local_velocity_x_2 += velocity_change_x_2;
        local_velocity_y_2 += velocity_change_y_2;

        // Update work and check compression phase
        double velocity_diff_y_temp = velocity_diff_y;
        velocity_diff_y = local_velocity_y_2 - local_velocity_y_1;
        total_work += 0.5f * deltaP * fabs(velocity_diff_y_temp + velocity_diff_y);

        if (work_compression == 0 && velocity_diff_y > 0) {
            work_compression = total_work;
            work_required = (1.0f + e_b * e_b) * work_compression;
        }
    }

    printf("\nC Final Local Velocities:\n");
    printf("  Ball 1: v_ix = %.6f, v_iy = %.6f\n", local_velocity_x_1, local_velocity_y_1);
    printf("  Ball 2: v_jx = %.6f, v_jy = %.6f\n", local_velocity_x_2, local_velocity_y_2);

    // Transform back to global coordinates
    for (int i = 0; i < 3; i++) {
        rvw1_result[i] = local_velocity_x_1 * right[i] + local_velocity_y_1 * forward[i];
        rvw2_result[i] = local_velocity_x_2 * right[i] + local_velocity_y_2 * forward[i];
    }

    printf("\nC Final Global Velocities:\n");
    printf("  Ball 1: %.6f %.6f %.6f\n", rvw1_result[0], rvw1_result[1], rvw1_result[2]);
    printf("  Ball 2: %.6f %.6f %.6f\n", rvw2_result[0], rvw2_result[1], rvw2_result[2]);

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
