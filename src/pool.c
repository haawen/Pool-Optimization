#include "pool.h"
#include "math_helper.h"

#include <stdio.h>
#include <math.h>
#include <time.h>
#include <string.h>


#ifdef _MSC_VER
    #include <intrin.h>
    #include <windows.h>
#else
    #include <x86intrin.h>
#endif


DLL_EXPORT void hello_world(const char* matrix_name, double* rvw) {

    printf("Received %s\n", matrix_name);
   for(int i = 0; i < 9; i++) {
       printf("%f ", rvw[i]);
       if((i + 1) % 3 == 0) printf("\n");
   }

}

typedef struct {
    unsigned long long cycle_start;
    #ifdef _MSC_VER
        LARGE_INTEGER freq, start_counter, end_counter;
    #else
        struct timespec start_ts, end_ts;
    #endif
} Profile;


void start_profiling_section(Profile* profile) {
    #ifdef _MSC_VER
        QueryPerformanceFrequency(&profile->freq);
        QueryPerformanceCounter(&profile->start_counter);
    #else
        clock_gettime(CLOCK_MONOTONIC, &profile->start_ts);
    #endif
    profile->cycle_start = __rdtsc();
}

void end_profiling_section(Profile* profile) {
    #ifdef _MSC_VER
        QueryPerformanceCounter(&profile->end_counter_init);
        // unsigned long long ns_init = (unsigned long long)(((end_counter_init.QuadPart - start_counter_init.QuadPart) * 1e9) / freq_init.QuadPart);
    #else
        clock_gettime(CLOCK_MONOTONIC, &profile->end_ts);
        //unsigned long long ns_init = (end_ts_init.tv_sec - start_ts_init.tv_sec) * 1000000000ULL
        //                            + (end_ts_init.tv_nsec - start_ts_init.tv_nsec);
    #endif
    // unsigned long long cycles_init = __rdtsc() - cycle_start_init; // assuming cycle_start_init was recorded with __rdtsc()
    // printf("\n== Profiling: Initial calculations took %llu ns (%llu cycles) ==\n", ns_init, cycles_init);
}

void summarize_profile(Profile* profile, const char* name) {

    unsigned long long ns = 0;
    #ifdef _MSC_VER
            ns = (unsigned long long)(((profile->end_counter.QuadPart - profile->start_counter.QuadPart) * 1e9) / profile->freq.QuadPart);
        #else
            ns = (profile->end_ts.tv_sec - profile->start_ts.tv_sec) * 1000000000ULL + (profile->end_ts.tv_nsec - profile->start_ts.tv_nsec);
        #endif

    unsigned long long cycles = __rdtsc() - profile->cycle_start;

   printf("\n=== %s Profile === \n", name);
   printf("\t%-12s %e\n", "Nanoseconds:", (double)ns);
   printf("\t%-12s %e\n", "Cycles:", (double)cycles);
   int total_length = 4 + strlen(name) + 13;
   for (int i = 0; i < total_length; i++) {
       putchar('=');
   }
   printf("\n");
}

DLL_EXPORT void collide_balls(double* rvw1, double* rvw2, float R, float M, float u_s1, float u_s2, float u_b, float e_b, float deltaP, int N, double* rvw1_result, double* rvw2_result) {


    #ifdef PROFILE
        Profile init_profile;
        Profile loop;
    #endif


    double* translation_1 = get_displacement(rvw1);
    double* velocity_1 = get_velocity(rvw1);
    double* angular_velocity_1 = get_angular_velocity(rvw1);

    double* translation_2 = get_displacement(rvw2);
    double* velocity_2 = get_velocity(rvw2);
    double* angular_velocity_2 = get_angular_velocity(rvw2);

   #ifdef PROFILE
        start_profiling_section(&init_profile);
    #endif

    double offset[3];
    subV3(translation_2, translation_1, offset);

    double offset_mag_sqrd = dotV3(offset, offset);
    double offset_mag = sqrt(offset_mag_sqrd);

    double forward[3]; // Forward from ball 1 to ball 2, normalized
    divV3(offset, offset_mag, forward);

    double up[3] = {0, 0, 1}; // Probably up axis?

    double right[3]; // Axis orthogonal to Z and forward
    crossV3(forward, up, right);

    // From here on, it is assumed that the x axis is the right axis and y axis is the forward axis
    // Transform velocities to local frame
    double local_velocity_x_1 = dotV3(velocity_1, right);
    double local_velocity_y_1 = dotV3(velocity_1, forward);
    double local_velocity_x_2 = dotV3(velocity_2, right);
    double local_velocity_y_2 = dotV3(velocity_2, forward);

    // Transform angular velocities into local frame
    double local_angular_velocity_x_1 = dotV3(angular_velocity_1, right);
    double local_angular_velocity_y_1 = dotV3(angular_velocity_1, forward);
    double local_angular_velocity_z_1 = dotV3(angular_velocity_1, up);
    double local_angular_velocity_x_2 = dotV3(angular_velocity_2, right);
    double local_angular_velocity_y_2 = dotV3(angular_velocity_2, forward);
    double local_angular_velocity_z_2 = dotV3(angular_velocity_2, up);

    // Calculate velocity at contact point
    // = Calculate ball-table slips?
    // Slip refers to relative motion between two surfaces in contact — here, the ball and the table.
    // Its the velocity at the contact point of the table and the ball
    double surface_velocity_x_1 = local_velocity_x_1 + R * local_angular_velocity_y_1;
    double surface_velocity_y_1 = local_velocity_y_1 - R * local_angular_velocity_x_1;
    double surface_velocity_x_2 = local_velocity_x_2 + R * local_angular_velocity_y_2;
    double surface_velocity_y_2 = local_velocity_y_2 - R * local_angular_velocity_x_2;

    double surface_velocity_magnitude_1 = sqrt(surface_velocity_x_1 * surface_velocity_x_1 + surface_velocity_y_1 * surface_velocity_y_1);
    double surface_velocity_magnitude_2 = sqrt(surface_velocity_x_2 * surface_velocity_x_2 + surface_velocity_y_2 * surface_velocity_y_2);

    // Relative surface velocity in the x-direction at the point where the two balls are in contact.
    // ball-ball slip
    double contact_point_velocity_x = local_velocity_x_1 - local_velocity_x_2 - R * (local_angular_velocity_z_1 + local_angular_velocity_z_2);
    double contact_point_velocity_z = R * (local_angular_velocity_x_1 + local_angular_velocity_x_2);
    double ball_ball_contact_point_magnitude = sqrt(contact_point_velocity_x * contact_point_velocity_x + contact_point_velocity_z * contact_point_velocity_z);
    //printf("\nC Contact Point Slide, Spin:\n");
    //printf("  Contact Point: u_ijC_xz_mag= %.6f\n", ball_ball_contact_point_magnitude);

    #ifdef PROFILE
        end_profiling_section(&init_profile);
    #endif



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

    // TODO: better naming for deltas
    // Delta impulse overall per ball?
    double deltaP_1 = deltaP;
    double deltaP_2 = deltaP;

    // Impulse per axis per ball
    double deltaP_x_1 = 0;
    double deltaP_y_1 = 0;
    double deltaP_x_2 = 0;
    double deltaP_y_2 = 0;

    #ifdef PROFILE
         start_profiling_section(&loop);
     #endif
    while (velocity_diff_y < 0 || total_work < work_required) {

        // Impulse Calculation
        if (ball_ball_contact_point_magnitude < 1e-16) {
            deltaP_1 = 0;
            deltaP_2 = 0;
            deltaP_x_1 = 0;
            deltaP_y_1 = 0;
            deltaP_x_2 = 0;
            deltaP_y_2 = 0;
        } else {
            deltaP_1 = -u_b * deltaP * contact_point_velocity_x / ball_ball_contact_point_magnitude;
            if(fabs(contact_point_velocity_z) < 1e-16) {
                deltaP_2 = 0;
                deltaP_x_1 = 0;
                deltaP_y_1 = 0;
                deltaP_x_2 = 0;
                deltaP_y_2 = 0;
            } else {
                deltaP_2 = -u_b * deltaP * contact_point_velocity_z / ball_ball_contact_point_magnitude;

                if(deltaP_2 > 0) {
                    deltaP_x_1 = 0;
                    deltaP_y_1 = 0;

                    // TODO: probably best to check for some tolerance
                    if(surface_velocity_magnitude_2 == 0.0) {
                        deltaP_x_2 = 0;
                        deltaP_y_2 = 0;
                    } else {
                        deltaP_x_2 = -u_s2 * (surface_velocity_x_2 / surface_velocity_magnitude_2) * deltaP_2;
                        deltaP_y_2 = -u_s2 * (surface_velocity_y_2 / surface_velocity_magnitude_2) * deltaP_2;
                    }
                } else {
                    deltaP_x_2 = 0;
                    deltaP_y_2 = 0;
                    if(surface_velocity_magnitude_1 == 0.0) {
                        deltaP_x_1 = 0;
                        deltaP_y_1 = 0;
                    } else {
                        deltaP_x_1 = u_s1 * (surface_velocity_x_1 / surface_velocity_magnitude_1) * deltaP_2;
                        deltaP_y_1 = u_s1 * (surface_velocity_y_1 / surface_velocity_magnitude_1) * deltaP_2;
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
        surface_velocity_x_1 = local_velocity_x_1 + R * local_angular_velocity_y_1;
        surface_velocity_y_1 = local_velocity_y_1 - R * local_angular_velocity_x_1;
        surface_velocity_x_2 = local_velocity_x_2 + R * local_angular_velocity_y_2;
        surface_velocity_y_2 = local_velocity_y_2 - R * local_angular_velocity_x_2;

        surface_velocity_magnitude_1 = sqrt(surface_velocity_x_1 * surface_velocity_x_1 + surface_velocity_y_1 * surface_velocity_y_1);
        surface_velocity_magnitude_2 = sqrt(surface_velocity_x_2 * surface_velocity_x_2 + surface_velocity_y_2 * surface_velocity_y_2);

        // update ball-ball slip:
        contact_point_velocity_x = local_velocity_x_1 - local_velocity_x_2 - R * (local_angular_velocity_z_1 + local_angular_velocity_z_2);
        contact_point_velocity_z = R * (local_angular_velocity_x_1 + local_angular_velocity_x_2);
        ball_ball_contact_point_magnitude = sqrt(contact_point_velocity_x * contact_point_velocity_x + contact_point_velocity_z * contact_point_velocity_z);

        // Update work and check compression phase
        double velocity_diff_y_temp = velocity_diff_y;
        velocity_diff_y = local_velocity_y_2 - local_velocity_y_1;
        total_work += 0.5 * deltaP * fabs(velocity_diff_y_temp + velocity_diff_y);

        if (work_compression == 0 && velocity_diff_y > 0) {
            work_compression = total_work;
            work_required = (1.0 + e_b * e_b) * work_compression;
        }
    }

    #ifdef PROFILE
        end_profiling_section(&loop);
    #endif

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

    #ifdef PROFILE
        summarize_profile(&init_profile, "Initialization");
        summarize_profile(&loop, "Loop");
    #endif
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
