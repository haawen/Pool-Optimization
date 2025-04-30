#include "pool.h"
#include "math_helper.h"

#include <stdio.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <stdbool.h>

#define PROFILE

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
    unsigned long long cycle_end;
    #ifdef _MSC_VER
        LARGE_INTEGER freq, start_counter, end_counter;
    #else
        struct timespec start_ts, end_ts;
    #endif
} Profile;


static inline void start_profiling_section(Profile* profile) {
    #ifdef _MSC_VER
        QueryPerformanceFrequency(&profile->freq);
        QueryPerformanceCounter(&profile->start_counter);
    #else
        clock_gettime(CLOCK_MONOTONIC, &profile->start_ts);
    #endif
    profile->cycle_start = __rdtsc();
}

static inline void end_profiling_section(Profile* profile) {
    #ifdef _MSC_VER
        QueryPerformanceCounter(&profile->end_counter);
        // unsigned long long ns_init = (unsigned long long)(((end_counter_init.QuadPart - start_counter_init.QuadPart) * 1e9) / freq_init.QuadPart);
    #else
        clock_gettime(CLOCK_MONOTONIC, &profile->end_ts);
        //unsigned long long ns_init = (end_ts_init.tv_sec - start_ts_init.tv_sec) * 1000000000ULL
        //                            + (end_ts_init.tv_nsec - start_ts_init.tv_nsec);
    #endif

    profile->cycle_end = __rdtsc();
}

void summarize_profile(Profile* profile, const char* name) {

    unsigned long long ns = 0;
    #ifdef _MSC_VER
            ns = (unsigned long long)(((profile->end_counter.QuadPart - profile->start_counter.QuadPart) * 1e9) / profile->freq.QuadPart);
        #else
            ns = (profile->end_ts.tv_sec - profile->start_ts.tv_sec) * 1000000000ULL + (profile->end_ts.tv_nsec - profile->start_ts.tv_nsec);
        #endif

    unsigned long long cycles = profile->cycle_end - profile->cycle_start;

   printf("\n=== %s Profile === \n", name);
   printf("\t%-12s %llu\n", "Nanoseconds:", ns);
   printf("\t%-12s %llu\n", "Cycles:", cycles);
   int total_length = 4 + strlen(name) + 13;
   for (int i = 0; i < total_length; i++) {
       putchar('=');
   }
   printf("\n");
}

DLL_EXPORT void collide_balls(double* rvw1, double* rvw2, float R, float M, float u_s1, float u_s2, float u_b, float e_b, float deltaP, int N, double* rvw1_result, double* rvw2_result) {


    #ifdef PROFILE
        Profile complete_function;
        Profile before_loop;
        Profile loop;
        Profile single_loop_iteration;
        Profile after_loop;
    #endif


    #ifdef FLOP_COUNT
    long int flops = 0;
    #endif

    #ifdef PROFILE
        start_profiling_section(&complete_function);
        start_profiling_section(&before_loop);
     #endif

     double* translation_1 = get_displacement(rvw1);
     double* velocity_1 = get_velocity(rvw1);
     double* angular_velocity_1 = get_angular_velocity(rvw1);

     double* translation_2 = get_displacement(rvw2);
     double* velocity_2 = get_velocity(rvw2);
     double* angular_velocity_2 = get_angular_velocity(rvw2);

    #ifdef FLOP_COUNT
        flops += 3;
    #endif
    double offset[3];
    subV3(translation_2, translation_1, offset);

    #ifdef FLOP_COUNT
        flops += 5;
    #endif
    double offset_mag_sqrd = dotV3(offset, offset);

    #ifdef FLOP_COUNT
        flops += 1;
    #endif
    double offset_mag = sqrt(offset_mag_sqrd);

    double forward[3]; // Forward from ball 1 to ball 2, normalized

    #ifdef FLOP_COUNT
        flops += 3;
    #endif
    divV3(offset, offset_mag, forward);

    double up[3] = {0, 0, 1}; // Probably up axis?

    double right[3]; // Axis orthogonal to Z and forward

    #ifdef FLOP_COUNT
        flops += 9;
    #endif
    crossV3(forward, up, right);

    // From here on, it is assumed that the x axis is the right axis and y axis is the forward axis
    // Transform velocities to local frame

    #ifdef FLOP_COUNT
        flops += 5 * 4;
    #endif
    double local_velocity_x_1 = dotV3(velocity_1, right);
    double local_velocity_y_1 = dotV3(velocity_1, forward);
    double local_velocity_x_2 = dotV3(velocity_2, right);
    double local_velocity_y_2 = dotV3(velocity_2, forward);

    // Transform angular velocities into local frame

    #ifdef FLOP_COUNT
        flops += 5 * 6;
    #endif
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
    #ifdef FLOP_COUNT
        flops += 8;
    #endif
    double surface_velocity_x_1 = local_velocity_x_1 + R * local_angular_velocity_y_1;
    double surface_velocity_y_1 = local_velocity_y_1 - R * local_angular_velocity_x_1;
    double surface_velocity_x_2 = local_velocity_x_2 + R * local_angular_velocity_y_2;
    double surface_velocity_y_2 = local_velocity_y_2 - R * local_angular_velocity_x_2;

    #ifdef FLOP_COUNT
        flops += 8;
    #endif
    double surface_velocity_magnitude_1 = sqrt(surface_velocity_x_1 * surface_velocity_x_1 + surface_velocity_y_1 * surface_velocity_y_1);
    double surface_velocity_magnitude_2 = sqrt(surface_velocity_x_2 * surface_velocity_x_2 + surface_velocity_y_2 * surface_velocity_y_2);

    // Relative surface velocity in the x-direction at the point where the two balls are in contact.
    // ball-ball slip
    #ifdef FLOP_COUNT
        flops += 10;
    #endif
    double contact_point_velocity_x = local_velocity_x_1 - local_velocity_x_2 - R * (local_angular_velocity_z_1 + local_angular_velocity_z_2);
    double contact_point_velocity_z = R * (local_angular_velocity_x_1 + local_angular_velocity_x_2);
    double ball_ball_contact_point_magnitude = sqrt(contact_point_velocity_x * contact_point_velocity_x + contact_point_velocity_z * contact_point_velocity_z);
    //printf("\nC Contact Point Slide, Spin:\n");
    //printf("  Contact Point: u_ijC_xz_mag= %.6f\n", ball_ball_contact_point_magnitude);




    #ifdef FLOP_COUNT
        flops += 1;
    #endif
    // Main collision loop
    double velocity_diff_y = local_velocity_y_2 - local_velocity_y_1;

    // deltaP is most likely always 0?
    // ΔP represents the Impulse during a time of Δt
    if (deltaP == 0) {
        #ifdef FLOP_COUNT
            flops += 5;
        #endif
        deltaP = 0.5 * (1.0 + e_b) * M * fabs(velocity_diff_y) / (double)(N);
    }

    #ifdef FLOP_COUNT
        flops += 3;
    #endif
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
        end_profiling_section(&before_loop);
         start_profiling_section(&loop);
         bool first_iter = true;
     #endif
    while (velocity_diff_y < 0 || total_work < work_required) {

        #ifdef PROFILE
            if(first_iter) {
                start_profiling_section(&single_loop_iteration);
            }
        #endif

        // Impulse Calculation
        if (ball_ball_contact_point_magnitude < 1e-16) {
            deltaP_1 = 0;
            deltaP_2 = 0;
            deltaP_x_1 = 0;
            deltaP_y_1 = 0;
            deltaP_x_2 = 0;
            deltaP_y_2 = 0;
        } else {
            #ifdef FLOP_COUNT
                flops += 4;
            #endif
            deltaP_1 = -u_b * deltaP * contact_point_velocity_x / ball_ball_contact_point_magnitude;
            if(fabs(contact_point_velocity_z) < 1e-16) {
                deltaP_2 = 0;
                deltaP_x_1 = 0;
                deltaP_y_1 = 0;
                deltaP_x_2 = 0;
                deltaP_y_2 = 0;
            } else {
                #ifdef FLOP_COUNT
                    flops += 4;
                #endif
                deltaP_2 = -u_b * deltaP * contact_point_velocity_z / ball_ball_contact_point_magnitude;

                if(deltaP_2 > 0) {
                    deltaP_x_1 = 0;
                    deltaP_y_1 = 0;

                    // TODO: probably best to check for some tolerance
                    if(surface_velocity_magnitude_2 == 0.0) {
                        deltaP_x_2 = 0;
                        deltaP_y_2 = 0;
                    } else {
                        #ifdef FLOP_COUNT
                            flops += 8;
                        #endif
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
                        #ifdef FLOP_COUNT
                            flops += 6;
                        #endif
                        deltaP_x_1 = u_s1 * (surface_velocity_x_1 / surface_velocity_magnitude_1) * deltaP_2;
                        deltaP_y_1 = u_s1 * (surface_velocity_y_1 / surface_velocity_magnitude_1) * deltaP_2;
                    }
                }

            }
        }

        #ifdef FLOP_COUNT
            flops += 10;
        #endif
        // Velocity changes
        double velocity_change_x_1 = (deltaP_1 + deltaP_x_1) / M;
        double velocity_change_y_1 = (-deltaP + deltaP_y_1) / M;
        double velocity_change_x_2 = (-deltaP_1 + deltaP_x_2) / M;
        double velocity_change_y_2 = (deltaP + deltaP_y_2) / M;

        #ifdef FLOP_COUNT
            flops += 4;
        #endif
        // Update velocities
        local_velocity_x_1 += velocity_change_x_1;
        local_velocity_y_1 += velocity_change_y_1;
        local_velocity_x_2 += velocity_change_x_2;
        local_velocity_y_2 += velocity_change_y_2;

        #ifdef FLOP_COUNT
            flops += 12;
        #endif
        // Angular velocity changes
        double delta_angular_velocity_x_1 = C * (deltaP_2 + deltaP_y_1);
        double delta_angular_velocity_y_1 = C * (-deltaP_x_1);
        double delta_angular_velocity_z_1 = C * (-deltaP_1);
        double delta_angular_velocity_x_2 = C * (deltaP_2 + deltaP_y_2);
        double delta_angular_velocity_y_2 = C * (-deltaP_x_2);
        double delta_angular_velocity_z_2 = C * (-deltaP_1);

        #ifdef FLOP_COUNT
            flops += 6;
        #endif
        // Update Angular Velocities
        local_angular_velocity_x_1 += delta_angular_velocity_x_1;
        local_angular_velocity_y_1 += delta_angular_velocity_y_1;
        local_angular_velocity_z_1 += delta_angular_velocity_z_1;

        local_angular_velocity_x_2 += delta_angular_velocity_x_2;
        local_angular_velocity_y_2 += delta_angular_velocity_y_2;
        local_angular_velocity_z_2 += delta_angular_velocity_z_2;

        #ifdef FLOP_COUNT
            flops += 8;
        #endif
        // update ball-table slips
        surface_velocity_x_1 = local_velocity_x_1 + R * local_angular_velocity_y_1;
        surface_velocity_y_1 = local_velocity_y_1 - R * local_angular_velocity_x_1;
        surface_velocity_x_2 = local_velocity_x_2 + R * local_angular_velocity_y_2;
        surface_velocity_y_2 = local_velocity_y_2 - R * local_angular_velocity_x_2;

        #ifdef FLOP_COUNT
            flops += 8;
        #endif
        surface_velocity_magnitude_1 = sqrt(surface_velocity_x_1 * surface_velocity_x_1 + surface_velocity_y_1 * surface_velocity_y_1);
        surface_velocity_magnitude_2 = sqrt(surface_velocity_x_2 * surface_velocity_x_2 + surface_velocity_y_2 * surface_velocity_y_2);

        #ifdef FLOP_COUNT
            flops += 10;
        #endif
        // update ball-ball slip:
        contact_point_velocity_x = local_velocity_x_1 - local_velocity_x_2 - R * (local_angular_velocity_z_1 + local_angular_velocity_z_2);
        contact_point_velocity_z = R * (local_angular_velocity_x_1 + local_angular_velocity_x_2);
        ball_ball_contact_point_magnitude = sqrt(contact_point_velocity_x * contact_point_velocity_x + contact_point_velocity_z * contact_point_velocity_z);

        #ifdef FLOP_COUNT
            flops += 5;
        #endif
        // Update work and check compression phase
        double velocity_diff_y_temp = velocity_diff_y;
        velocity_diff_y = local_velocity_y_2 - local_velocity_y_1;
        total_work += 0.5 * deltaP * fabs(velocity_diff_y_temp + velocity_diff_y);

        if (work_compression == 0 && velocity_diff_y > 0) {
            work_compression = total_work;
            #ifdef FLOP_COUNT
                flops += 3;
            #endif
            work_required = (1.0 + e_b * e_b) * work_compression;
        }

        #ifdef PROFILE
            if(first_iter) {
                first_iter = false;
                end_profiling_section(&single_loop_iteration);
            }
        #endif
    }

    #ifdef PROFILE
        end_profiling_section(&loop);
        start_profiling_section(&after_loop);
    #endif

    // Transform back to global coordinates
    for (int i = 0; i < 3; i++) {

        #ifdef FLOP_COUNT
             flops += 6;
         #endif
        rvw1_result[i + 3] = local_velocity_x_1 * right[i] + local_velocity_y_1 * forward[i];
        rvw2_result[i + 3] = local_velocity_x_2 * right[i] + local_velocity_y_2 * forward[i];

        if(i < 2) {
            #ifdef FLOP_COUNT
                flops += 6;
            #endif
            rvw1_result[i + 6] = local_angular_velocity_x_1 * right[i] + local_angular_velocity_y_1 * forward[i];
            rvw2_result[i + 6] = local_angular_velocity_x_2 * right[i] + local_angular_velocity_y_2 * forward[i];
        }
        else {
            rvw1_result[i + 6] = local_angular_velocity_z_1;
            rvw2_result[i + 6] = local_angular_velocity_z_2;
        }
    }

    #ifdef PROFILE
        end_profiling_section(&after_loop);
        end_profiling_section(&complete_function);
        summarize_profile(&complete_function, "collide_balls");
        summarize_profile(&before_loop, "Initialization");
        summarize_profile(&loop, "Loop");
        summarize_profile(&single_loop_iteration, "Single Loop Iteration");
        summarize_profile(&after_loop, "Transform to World Frame");
    #endif

    #ifdef FLOP_COUNT
        printf("\n\n\tExecuted %lu flops. \n\n", flops);
    #endif
}


DLL_EXPORT void code_motion_collide_balls(double* rvw1, double* rvw2, float R, float M, float u_s1, float u_s2, float u_b, float e_b, float deltaP, int N, double* rvw1_result, double* rvw2_result) {
    #ifdef PROFILE
        Profile complete_function;
        Profile before_loop;
        Profile loop;
        Profile single_loop_iteration;
        Profile after_loop;
    #endif

    #ifdef FLOP_COUNT
    long int flops = 0;
    #endif

    #ifdef PROFILE
        start_profiling_section(&complete_function);
        start_profiling_section(&before_loop);
     #endif

     // Get pointers into the state arrays
    double* translation_1 = &rvw1[0];
    double* velocity_1    = &rvw1[3];
    double* angular_1     = &rvw1[6];
    double* translation_2 = &rvw2[0];
    double* velocity_2    = &rvw2[3];
    double* angular_2     = &rvw2[6];



    #ifdef FLOP_COUNT
        flops += 3;
    #endif
    double offset[3];
    offset[0] = translation_2[0] - translation_1[0];
    offset[1] = translation_2[1] - translation_1[1];
    offset[2] = translation_2[2] - translation_1[2];

    #ifdef FLOP_COUNT
        flops += 5;
    #endif
    double offset_mag_sqrd = dotV3(offset, offset);
    offset_mag_sqrd = offset[0] * offset[0] +
    offset[1] * offset[1] +
    offset[2] * offset[2];

    #ifdef FLOP_COUNT
        flops += 1;
    #endif
    double offset_mag = sqrt(offset_mag_sqrd);

    double forward[3]; // Forward from ball 1 to ball 2, normalized

    #ifdef FLOP_COUNT
        flops += 3;
    #endif
    divV3(offset, offset_mag, forward);

    double up[3] = {0, 0, 1}; // Probably up axis?

    double right[3]; // Axis orthogonal to Z and forward

    #ifdef FLOP_COUNT
        flops += 9;
    #endif
    crossV3(forward, up, right);

    // From here on, it is assumed that the x axis is the right axis and y axis is the forward axis
    // Transform velocities to local frame

    #ifdef FLOP_COUNT
        flops += 5 * 4;
    #endif
    double local_vel_1[2] = { dotV3(velocity_1, right), dotV3(velocity_1, forward) };
    double local_vel_2[2] = { dotV3(velocity_2, right), dotV3(velocity_2, forward) };


    // Transform angular velocities into local frame

    #ifdef FLOP_COUNT
        flops += 5 * 6;
    #endif
    double local_ang_1[3] = { dotV3(angular_1, right), dotV3(angular_1, forward), dotV3(angular_1, up) };
    double local_ang_2[3] = { dotV3(angular_2, right), dotV3(angular_2, forward), dotV3(angular_2, up) };


    // Calculate velocity at contact point
    // = Calculate ball-table slips?
    // Slip refers to relative motion between two surfaces in contact — here, the ball and the table.
    // Its the velocity at the contact point of the table and the ball
    #ifdef FLOP_COUNT
        flops += 8;
    #endif
    double surf_vel_1[2] = { local_vel_1[0] + R * local_ang_1[1],
            local_vel_1[1] - R * local_ang_1[0] };
    double surf_vel_2[2] = { local_vel_2[0] + R * local_ang_2[1],
            local_vel_2[1] - R * local_ang_2[0] };

    #ifdef FLOP_COUNT
        flops += 8;
    #endif
    double surf_vel_mag_1 = sqrt(surf_vel_1[0]*surf_vel_1[0] + surf_vel_1[1]*surf_vel_1[1]);
    double surf_vel_mag_2 = sqrt(surf_vel_2[0]*surf_vel_2[0] + surf_vel_2[1]*surf_vel_2[1]);

    // Relative surface velocity in the x-direction at the point where the two balls are in contact.
    // ball-ball slip
    #ifdef FLOP_COUNT
        flops += 10;
    #endif
    double contact_vel[2] = { local_vel_1[0] - local_vel_2[0] - R * (local_ang_1[2] + local_ang_2[2]),
            R * (local_ang_1[0] + local_ang_2[0]) };
    double ball_ball_contact_mag = sqrt(contact_vel[0]*contact_vel[0] + contact_vel[1]*contact_vel[1]);

    #ifdef FLOP_COUNT
        flops += 1;
    #endif
    // Main collision loop
    double velocity_diff_y = local_vel_2[1] - local_vel_1[1];

    // deltaP is most likely always 0?
    // ΔP represents the Impulse during a time of Δt
    if (deltaP == 0) {
        #ifdef FLOP_COUNT
            flops += 5;
        #endif
        deltaP = 0.5 * (1.0 + e_b) * M * fabs(velocity_diff_y)/(double)N;
    }

    #ifdef FLOP_COUNT
        flops += 3;
    #endif
    double C = 5.0 / (2.0 * M * R);
    double total_work = 0; // Work done due to impulse force
    double work_required = INFINITY; // Total amount of work required before collision handling is complete
    double work_compression = 0;

    // TODO: better naming for deltas
    // Delta impulse overall per ball?
    // Impulse per axis per ball
    double deltaP_ball[2] = { 0, 0 };
    double deltaP_axis_1[2] = { 0, 0 };
    double deltaP_axis_2[2] = { 0, 0 };

    #ifdef PROFILE
        end_profiling_section(&before_loop);
         start_profiling_section(&loop);
         bool first_iter = true;
     #endif
    while (velocity_diff_y < 0 || total_work < work_required) {

        #ifdef PROFILE
            if(first_iter) {
                start_profiling_section(&single_loop_iteration);
            }
        #endif

        // Impulse Calculation
        if (ball_ball_contact_mag < 1e-16) {
            deltaP_ball[0] = deltaP_ball[1] = 0;
            deltaP_axis_1[0] = deltaP_axis_1[1] = 0;
            deltaP_axis_2[0] = deltaP_axis_2[1] = 0;
        } else {
            #ifdef FLOP_COUNT
                flops += 4;
            #endif
            deltaP_ball[0] = -u_b * deltaP * contact_vel[0] / ball_ball_contact_mag;
            if(fabs(contact_vel[1]) < 1e-16) {
                deltaP_ball[1] = 0;
                deltaP_axis_1[0] = deltaP_axis_1[1] = 0;
                deltaP_axis_2[0] = deltaP_axis_2[1] = 0;
            } else {
                #ifdef FLOP_COUNT
                    flops += 4;
                #endif
                deltaP_ball[1] = -u_b * deltaP * contact_vel[1] / ball_ball_contact_mag;
                if(deltaP_ball[1] > 0) {
                    deltaP_axis_1[0] = deltaP_axis_1[1] = 0;
                    if(surf_vel_mag_2 == 0.0) {
                        deltaP_axis_2[0] = deltaP_axis_2[1] = 0;
                    } else {
                        #ifdef FLOP_COUNT
                            flops += 8;
                        #endif
                        deltaP_axis_2[0] = -u_s2 * (surf_vel_2[0]/surf_vel_mag_2) * deltaP_ball[1];
                        deltaP_axis_2[1] = -u_s2 * (surf_vel_2[1]/surf_vel_mag_2) * deltaP_ball[1];
                    }
                } else {
                    deltaP_axis_2[0] = deltaP_axis_2[1] = 0;
                    if(surf_vel_mag_1 == 0.0) {
                        deltaP_axis_1[0] = deltaP_axis_1[1] = 0;
                    } else {
                        #ifdef FLOP_COUNT
                            flops += 6;
                        #endif
                        deltaP_axis_1[0] = u_s1 * (surf_vel_1[0]/surf_vel_mag_1) * deltaP_ball[1];
                        deltaP_axis_1[1] = u_s1 * (surf_vel_1[1]/surf_vel_mag_1) * deltaP_ball[1];
                    }
                }
            }
        }

        #ifdef FLOP_COUNT
            flops += 10;
        #endif
        // Velocity changes
        double v_change1[2] = { (deltaP_ball[0] + deltaP_axis_1[0]) / M,
                    (-deltaP + deltaP_axis_1[1]) / M };
        double v_change2[2] = { (-deltaP_ball[0] + deltaP_axis_2[0]) / M,
                    ( deltaP + deltaP_axis_2[1]) / M };

        #ifdef FLOP_COUNT
            flops += 4;
        #endif
        // Update velocities
        local_vel_1[0] += v_change1[0];
        local_vel_1[1] += v_change1[1];
        local_vel_2[0] += v_change2[0];
        local_vel_2[1] += v_change2[1];

        #ifdef FLOP_COUNT
            flops += 12;
        #endif
        // Angular velocity changes
        double ang_change1[3] = { C * (deltaP_ball[1] + deltaP_axis_1[1]),
                    C * (-deltaP_axis_1[0]),
                    C * (-deltaP_ball[0]) };
        double ang_change2[3] = { C * (deltaP_ball[1] + deltaP_axis_2[1]),
                    C * (-deltaP_axis_2[0]),
                    C * (-deltaP_ball[0]) };

        #ifdef FLOP_COUNT
            flops += 6;
        #endif
        // Update Angular Velocities
        local_ang_1[0] += ang_change1[0];
        local_ang_1[1] += ang_change1[1];
        local_ang_1[2] += ang_change1[2];

        local_ang_2[0] += ang_change2[0];
        local_ang_2[1] += ang_change2[1];
        local_ang_2[2] += ang_change2[2];

        #ifdef FLOP_COUNT
            flops += 8;
        #endif

        // Recalculate surface velocities using the updated arrays
        surf_vel_1[0] = local_vel_1[0] + R * local_ang_1[1];
        surf_vel_1[1] = local_vel_1[1] - R * local_ang_1[0];
        surf_vel_2[0] = local_vel_2[0] + R * local_ang_2[1];
        surf_vel_2[1] = local_vel_2[1] - R * local_ang_2[0];

        #ifdef FLOP_COUNT
            flops += 8;
        #endif
        surf_vel_mag_1 = sqrt(surf_vel_1[0]*surf_vel_1[0] + surf_vel_1[1]*surf_vel_1[1]);
        surf_vel_mag_2 = sqrt(surf_vel_2[0]*surf_vel_2[0] + surf_vel_2[1]*surf_vel_2[1]);

        #ifdef FLOP_COUNT
            flops += 10;
        #endif
        // update ball-ball slip:
        contact_vel[0] = local_vel_1[0] - local_vel_2[0] - R * (local_ang_1[2] + local_ang_2[2]);
        contact_vel[1] = R * (local_ang_1[0] + local_ang_2[0]);
        ball_ball_contact_mag = sqrt(contact_vel[0]*contact_vel[0] + contact_vel[1]*contact_vel[1]);

        #ifdef FLOP_COUNT
            flops += 5;
        #endif
        // Update work and check compression phase
        double prev_diff = velocity_diff_y;
        velocity_diff_y = local_vel_2[1] - local_vel_1[1];
        total_work += 0.5 * deltaP * fabs(prev_diff + velocity_diff_y);

        if (work_compression == 0 && velocity_diff_y > 0) {
            work_compression = total_work;
            #ifdef FLOP_COUNT
                flops += 3;
            #endif
            work_required = (1.0 + e_b * e_b) * work_compression;
        }

        #ifdef PROFILE
            if(first_iter) {
                first_iter = false;
                end_profiling_section(&single_loop_iteration);
            }
        #endif
    }

    #ifdef PROFILE
        end_profiling_section(&loop);
        start_profiling_section(&after_loop);
    #endif

    // Transform back to global coordinates
    for (int i = 0; i < 3; i++) {

        #ifdef FLOP_COUNT
             flops += 6;
        #endif
        rvw1_result[i+3] = local_vel_1[0] * right[i] + local_vel_1[1] * forward[i];
        rvw2_result[i+3] = local_vel_2[0] * right[i] + local_vel_2[1] * forward[i];
        if(i < 2) {
            #ifdef FLOP_COUNT
                flops += 6;
            #endif
            rvw1_result[i+6] = local_ang_1[0] * right[i] + local_ang_1[1] * forward[i];
            rvw2_result[i+6] = local_ang_2[0] * right[i] + local_ang_2[1] * forward[i];
        }
        else {
            rvw1_result[i+6] = local_ang_1[2];
            rvw2_result[i+6] = local_ang_2[2];
        }
    }

    #ifdef PROFILE
        end_profiling_section(&after_loop);
        end_profiling_section(&complete_function);
        summarize_profile(&complete_function, "collide_balls");
        summarize_profile(&before_loop, "Initialization");
        summarize_profile(&loop, "Loop");
        summarize_profile(&single_loop_iteration, "Single Loop Iteration");
        summarize_profile(&after_loop, "Transform to World Frame");
    #endif

    #ifdef FLOP_COUNT
        printf("\n\n\tExecuted %lu flops. \n\n", flops);
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
