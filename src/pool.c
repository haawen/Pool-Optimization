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
    /*
    printf("C Initial State - Ball 1:\n");
    printf("  Position: %.6f %.6f %.6f\n", translation_1[0], translation_1[1], translation_1[2]);
    printf("  Velocity: %.6f %.6f %.6f\n", velocity_1[0], velocity_1[1], velocity_1[2]);
    printf("  Angular:  %.6f %.6f %.6f\n", angular_velocity_1[0], angular_velocity_1[1], angular_velocity_1[2]);

    printf("C Initial State - Ball 2:\n");
    printf("  Position: %.6f %.6f %.6f\n", translation_2[0], translation_2[1], translation_2[2]);
    printf("  Velocity: %.6f %.6f %.6f\n", velocity_2[0], velocity_2[1], velocity_2[2]);
    printf("  Angular:  %.6f %.6f %.6f\n", angular_velocity_2[0], angular_velocity_2[1], angular_velocity_2[2]);
    */

    double offset[3];
    subV3(translation_2, translation_1, offset);

    double offset_mag_sqrd = dotV3(offset, offset);
    double offset_mag = sqrt(offset_mag_sqrd);

    double forward[3]; // Forward from ball 1 to ball 2, normalized
    divV3(offset, offset_mag, forward);

    double up[3] = {0, 0, 1}; // Probably up axis?

    double right[3]; // Axis orthogonal to Z and forward
    crossV3(forward, up, right);

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

    // Transform angular velocities into local frame
    double local_angular_velocity_x_1 = dotV3(angular_velocity_1, right);
    double local_angular_velocity_y_1 = dotV3(angular_velocity_1, forward);
    double local_angular_velocity_z_1 = dotV3(angular_velocity_1, up);
    double local_angular_velocity_x_2 = dotV3(angular_velocity_2, right);
    double local_angular_velocity_y_2 = dotV3(angular_velocity_2, forward);
    double local_angular_velocity_z_2 = dotV3(angular_velocity_2, up);

    printf("\nC Initial Angular Local Velocities:\n");
    printf("  Ball 1: v_ix = %.6f, v_iy = %.6f, v_iy = %.6f\n", local_angular_velocity_x_1, local_angular_velocity_y_1, local_angular_velocity_z_1);
    printf("  Ball 2: v_jx = %.6f, v_jy = %.6f, v_jy = %.6f\n", local_angular_velocity_x_2, local_angular_velocity_y_2, local_angular_velocity_z_2);

    // Calculate velocity at contact point
    // = Calculate ball-table slips?
    // Slip refers to relative motion between two surfaces in contact — here, the ball and the table.
    double velocity_at_contact_point_x_1 = local_velocity_x_1 + R * local_angular_velocity_y_1;
    double velocity_at_contact_point_y_1 = local_velocity_y_1 - R * local_angular_velocity_x_1;
    double velocity_at_contact_point_x_2 = local_velocity_x_2 + R * local_angular_velocity_y_2;
    double velocity_at_contact_point_y_2 = local_velocity_y_2 - R * local_angular_velocity_x_2;

    double contact_point_velocity_magnitude_1 = sqrt(velocity_at_contact_point_x_1 * velocity_at_contact_point_x_1 + velocity_at_contact_point_y_1 * velocity_at_contact_point_y_1);
    double contact_point_velocity_magnitude_2 = sqrt(velocity_at_contact_point_x_2 * velocity_at_contact_point_x_2 + velocity_at_contact_point_y_2 * velocity_at_contact_point_y_2);

    printf("\nC Contact Point Velocity Magnitude:\n");
    printf("  Ball 1: u_iR_xy_mag= %.6f\n", contact_point_velocity_magnitude_1);
    printf("  Ball 2: u_jR_xy_mag= %.6f\n", contact_point_velocity_magnitude_2);

    // Relative surface velocity in the x-direction at the point where the two balls are in contact.
    // TODO: better naming compared to velocity_at_contact_point, a bit confusing...
    // ball-ball slip
    double contact_point_sliding_velocity_x = local_velocity_x_1 - local_velocity_x_2 - R * (local_angular_velocity_z_1 + local_angular_velocity_z_2);
    double contact_point_spin_velocity_z = R * (local_angular_velocity_x_1 + local_angular_velocity_x_2);
    double contact_point_sliding_spin_magnitude = sqrt(contact_point_sliding_velocity_x * contact_point_sliding_velocity_x + contact_point_spin_velocity_z * contact_point_spin_velocity_z);
    printf("\nC Contact Point Slide, Spin:\n");
    printf("  Contact Point: u_ijC_xz_mag= %.6f\n", contact_point_sliding_spin_magnitude);

    // Main collision loop
    double velocity_diff_y = local_velocity_y_2 - local_velocity_y_1;

    // deltaP is most likely always 0?
    // ΔP represents the Impulse during a time of Δt
    if (deltaP == 0) {
        deltaP = 0.5 * (1.0 + e_b) * M * fabs(velocity_diff_y) / (double)(N);
    }

    double C = 5.0 / (2.0 * M * R);
    double total_work = 0; // Work done due to impulse force
    double work_required = INFINITY; // Total amount of work required before collision handling is complete
    double work_compression = 0;

    int niter = 0;

    // Delta impulse overall per ball?
    double deltaP_1 = deltaP;
    double deltaP_2 = deltaP;

    // Impulse per axis per ball
    double deltaP_x_1 = 0;
    double deltaP_y_1 = 0;
    double deltaP_x_2 = 0;
    double deltaP_y_2 = 0;

    while (velocity_diff_y < 0 || total_work < work_required) {

        // Impulse Calculation
        if (contact_point_sliding_spin_magnitude < 1e-16) {
            deltaP_1 = 0;
            deltaP_2 = 0;
            deltaP_x_1 = 0;
            deltaP_y_1 = 0;
            deltaP_x_2 = 0;
            deltaP_y_2 = 0;
        } else {
            deltaP_1 = -u_b * deltaP * contact_point_sliding_velocity_x / contact_point_sliding_spin_magnitude;
            if(fabs(contact_point_spin_velocity_z) < 1e-16) {
                deltaP_2 = 0;
                deltaP_x_1 = 0;
                deltaP_y_1 = 0;
                deltaP_x_2 = 0;
                deltaP_y_2 = 0;
            } else {
                deltaP_2 = -u_b * deltaP * contact_point_spin_velocity_z / contact_point_sliding_spin_magnitude;

                if(deltaP_2 > 0) {
                    deltaP_x_1 = 0;
                    deltaP_y_1 = 0;

                    // TODO: probably best to check for some tolerance
                    if(contact_point_velocity_magnitude_2 == 0.0) {
                        deltaP_x_2 = 0;
                        deltaP_y_2 = 0;
                    } else {
                        deltaP_x_2 = -u_s2 * (velocity_at_contact_point_x_2 / contact_point_velocity_magnitude_2) * deltaP_2;
                        deltaP_y_2 = -u_s2 * (velocity_at_contact_point_y_2 / contact_point_velocity_magnitude_2) * deltaP_2;
                    }
                } else {
                    deltaP_x_2 = 0;
                    deltaP_y_2 = 0;
                    if(contact_point_velocity_magnitude_1 == 0.0) {
                        deltaP_x_1 = 0;
                        deltaP_y_1 = 0;
                    } else {
                        // Why is it also deltaP_2 here? (Same in python)
                        deltaP_x_1 = u_s1 * (velocity_at_contact_point_x_1 / contact_point_velocity_magnitude_1) * deltaP_2;
                        deltaP_y_1 = u_s1 * (velocity_at_contact_point_y_1 / contact_point_velocity_magnitude_1) * deltaP_2;
                    }
                }

            }
        }

        // Velocity changes
        double velocity_change_x_1 = (deltaP_1 + deltaP_x_1) / M;
        double velocity_change_y_1 = (-deltaP + deltaP_y_1) / M;
        double velocity_change_x_2 = (-deltaP_1 + deltaP_x_2) / M;
        double velocity_change_y_2 = (deltaP + deltaP_y_2) / M;

        // Update velocities
        local_velocity_x_1 += velocity_change_x_1;
        local_velocity_y_1 += velocity_change_y_1;
        local_velocity_x_2 += velocity_change_x_2;
        local_velocity_y_2 += velocity_change_y_2;

        // Angular velocity changes
        double delta_angular_velocity_x_1 = C * (deltaP_2 + deltaP_y_1);
        double delta_angular_velocity_y_1 = C * (-deltaP_x_1);
        double delta_angular_velocity_z_1 = C * (-deltaP_1);
        double delta_angular_velocity_x_2 = C * (deltaP_2 + deltaP_y_2);
        double delta_angular_velocity_y_2 = C * (-deltaP_x_2);
        double delta_angular_velocity_z_2 = C * (-deltaP_1);

        // Update Angular Velocities
        local_angular_velocity_x_1 += delta_angular_velocity_x_1;
        local_angular_velocity_y_1 += delta_angular_velocity_y_1;
        local_angular_velocity_z_1 += delta_angular_velocity_z_1;

        local_angular_velocity_x_2 += delta_angular_velocity_x_2;
        local_angular_velocity_y_2 += delta_angular_velocity_y_2;
        local_angular_velocity_z_2 += delta_angular_velocity_z_2;

        // update ball-table slips
        velocity_at_contact_point_x_1 = local_velocity_x_1 + R * local_angular_velocity_y_1;
        velocity_at_contact_point_y_1 = local_velocity_y_1 - R * local_angular_velocity_x_1;
        velocity_at_contact_point_x_2 = local_velocity_x_2 + R * local_angular_velocity_y_2;
        velocity_at_contact_point_y_2 = local_velocity_y_2 - R * local_angular_velocity_x_2;

        contact_point_velocity_magnitude_1 = sqrt(velocity_at_contact_point_x_1 * velocity_at_contact_point_x_1 + velocity_at_contact_point_y_1 * velocity_at_contact_point_y_1);
        contact_point_velocity_magnitude_2 = sqrt(velocity_at_contact_point_x_2 * velocity_at_contact_point_x_2 + velocity_at_contact_point_y_2 * velocity_at_contact_point_y_2);

        // update ball-ball slip:
        contact_point_sliding_velocity_x = local_velocity_x_1 - local_velocity_x_2 - R * (local_angular_velocity_z_1 + local_angular_velocity_z_2);
        contact_point_spin_velocity_z = R * (local_angular_velocity_x_1 + local_angular_velocity_x_2);
        contact_point_sliding_spin_magnitude = sqrt(contact_point_sliding_velocity_x * contact_point_sliding_velocity_x + contact_point_spin_velocity_z * contact_point_spin_velocity_z);

        // Update work and check compression phase
        double velocity_diff_y_temp = velocity_diff_y;
        velocity_diff_y = local_velocity_y_2 - local_velocity_y_1;
        total_work += 0.5 * deltaP * fabs(velocity_diff_y_temp + velocity_diff_y);

        if (work_compression == 0 && velocity_diff_y > 0) {
            work_compression = total_work;
            work_required = (1.0 + e_b * e_b) * work_compression;
        }

        niter++;

        /*
        printf("\nC DELTA P:\n");
        printf("  deltaP_1= %.16f, deltaP_2=%.16f\n", deltaP_1, deltaP_2);
        printf("  deltaP_x_1= %.16f, deltaP_y_1=%.16f\n", deltaP_x_1, deltaP_y_1);
        printf("  deltaP_x_2= %.16f, deltaP_y_2=%.16f\n", deltaP_x_2, deltaP_y_2);


        if(niter > 10) {
            break;
        }
        */

    }

    printf("\nC Final Local Velocities:\n");
    printf("  Ball 1: v_ix = %.6f, v_iy = %.6f\n", local_velocity_x_1, local_velocity_y_1);
    printf("  Ball 2: v_jx = %.6f, v_jy = %.6f\n", local_velocity_x_2, local_velocity_y_2);


    printf("\nC Final local Angular Velocities:\n");
    printf("  Ball 1: %.6f %.6f %.6f\n", local_angular_velocity_x_1, local_angular_velocity_y_1, local_angular_velocity_z_1);
    printf("  Ball 2: %.6f %.6f %.6f\n", local_angular_velocity_x_2, local_angular_velocity_y_2, local_angular_velocity_z_2);


    // Transform back to global coordinates
    for (int i = 0; i < 3; i++) {
        rvw1_result[i + 3] = local_velocity_x_1 * right[i] + local_velocity_y_1 * forward[i];
        rvw2_result[i + 3] = local_velocity_x_2 * right[i] + local_velocity_y_2 * forward[i];

        if(i < 2) {
            rvw1_result[i + 6] = local_angular_velocity_x_1 * right[i] + local_angular_velocity_y_1 * forward[i];
            rvw2_result[i + 6] = local_angular_velocity_x_2 * right[i] + local_angular_velocity_y_2 * forward[i];
        }
        else {
            rvw1_result[i + 6] = local_angular_velocity_z_1;
            rvw2_result[i + 6] = local_angular_velocity_z_2;
        }
    }

    printf("\nC Final Global Velocities:\n");
    printf("  Ball 1: %.6f %.6f %.6f\n", rvw1_result[3], rvw1_result[4], rvw1_result[5]);
    printf("  Ball 2: %.6f %.6f %.6f\n", rvw2_result[3], rvw2_result[4], rvw2_result[5]);

    printf("\nC Final Global Angular Velocities:\n");
    printf("  Ball 1: %.6f %.6f %.6f\n", rvw1_result[6], rvw1_result[7], rvw1_result[8]);
    printf("  Ball 2: %.6f %.6f %.6f\n", rvw2_result[6], rvw2_result[7], rvw2_result[8]);

    printf("\n=== End C Implementation ===\n");

    fflush(stdout);
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
