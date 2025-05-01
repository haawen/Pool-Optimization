#include "pool.h"
#include "math_helper.h"

#include <stdio.h>
#include <math.h>
#include <stdbool.h>

#ifdef _MSC_VER
    #include <intrin.h>
    #include <windows.h>
#else
    #include <x86intrin.h>
#endif


#ifdef FLOP_COUNT

    #define FLOPS(count, ...) \
    do { \
        void *arr[] = { __VA_ARGS__ }; \
        for (size_t i = 0; i < sizeof(arr)/sizeof(void*); i++) { \
            ((Profile*)arr[i])->flops += (count); \
        } \
    } while(0)

    #define MEMORY(count, ...) \
    do { \
        void *arr[] = { __VA_ARGS__ }; \
        for (size_t i = 0; i < sizeof(arr)/sizeof(void*); i++) { \
            ((Profile*)arr[i])->memory += (count); \
        } \
    } while(0)

    #define FLOPS_SINGLE_LOOP(count) \
        if(first_iter) { \
        single_loop_iteration->flops += count; \
        }

    #define MEMORY_SINGLE_LOOP(count) \
        if(first_iter) { \
        single_loop_iteration->memory += count; \
        }
#else

#define FLOPS(count, ...)
#define MEMORY(count, ...)
#define FLOPS_SINGLE_LOOP(count)
#define MEMORY_SINGLE_LOOP(count)

#endif


DLL_EXPORT void hello_world(const char* matrix_name, double* rvw) {

    printf("Received %s\n", matrix_name);
   for(int i = 0; i < 9; i++) {
       printf("%f ", rvw[i]);
       if((i + 1) % 3 == 0) printf("\n");
   }

}



static inline void start_profiling_section(Profile* profile) {
    #ifdef _MSC_VER
        QueryPerformanceFrequency(&profile->freq);
        QueryPerformanceCounter(&profile->start_counter);
    #else
        clock_gettime(CLOCK_MONOTONIC, &profile->start_ts);
    #endif
    profile->cycle_start = __rdtsc();
    profile->flops = 0;
    profile->memory = 0;
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

#ifdef PROFILE

#define START_PROFILE(profile) start_profiling_section(profile)
#define END_PROFILE(profile) end_profiling_section(profile)

#else

#define START_PROFILE(profile)
#define END_PROFILE(profile)

#endif

DLL_EXPORT void collide_balls(double* rvw1, double* rvw2, float R, float M, float u_s1, float u_s2, float u_b, float e_b, float deltaP, int N, double* rvw1_result, double* rvw2_result, Profile* profiles) {

    #ifdef PROFILE
        Profile* complete_function = &profiles[0];
        Profile* before_loop = &profiles[1];
        Profile* loop = &profiles[2];
        Profile* single_loop_iteration = &profiles[3];
        Profile* after_loop = &profiles[4];
    #endif

    START_PROFILE(complete_function);
    START_PROFILE(before_loop);

    // Altough the memory is not really loaded here, assuming its only compulsary misses
    MEMORY(18, complete_function, before_loop);
    double* translation_1 = get_displacement(rvw1);
    double* velocity_1 = get_velocity(rvw1);
    double* angular_velocity_1 = get_angular_velocity(rvw1);

    double* translation_2 = get_displacement(rvw2);
    double* velocity_2 = get_velocity(rvw2);
    double* angular_velocity_2 = get_angular_velocity(rvw2);

    FLOPS(3, complete_function, before_loop);
    double offset[3];
    subV3(translation_2, translation_1, offset);

    FLOPS(5, complete_function, before_loop);
    double offset_mag_sqrd = dotV3(offset, offset);

    FLOPS(1, complete_function, before_loop);
    double offset_mag = sqrt(offset_mag_sqrd);

    double forward[3]; // Forward from ball 1 to ball 2, normalized

    FLOPS(3, complete_function, before_loop);
    divV3(offset, offset_mag, forward);

    double up[3] = {0, 0, 1}; // Probably up axis?

    double right[3]; // Axis orthogonal to Z and forward

    FLOPS(9, complete_function, before_loop);
    crossV3(forward, up, right);

    // From here on, it is assumed that the x axis is the right axis and y axis is the forward axis
    // Transform velocities to local frame

    FLOPS(5*4, complete_function, before_loop);
    double local_velocity_x_1 = dotV3(velocity_1, right);
    double local_velocity_y_1 = dotV3(velocity_1, forward);
    double local_velocity_x_2 = dotV3(velocity_2, right);
    double local_velocity_y_2 = dotV3(velocity_2, forward);

    // Transform angular velocities into local frame

    FLOPS(5*6, complete_function, before_loop);
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
    FLOPS(8, complete_function, before_loop);
    double surface_velocity_x_1 = local_velocity_x_1 + R * local_angular_velocity_y_1;
    double surface_velocity_y_1 = local_velocity_y_1 - R * local_angular_velocity_x_1;
    double surface_velocity_x_2 = local_velocity_x_2 + R * local_angular_velocity_y_2;
    double surface_velocity_y_2 = local_velocity_y_2 - R * local_angular_velocity_x_2;

    FLOPS(8, complete_function, before_loop);
    double surface_velocity_magnitude_1 = sqrt(surface_velocity_x_1 * surface_velocity_x_1 + surface_velocity_y_1 * surface_velocity_y_1);
    double surface_velocity_magnitude_2 = sqrt(surface_velocity_x_2 * surface_velocity_x_2 + surface_velocity_y_2 * surface_velocity_y_2);

    // Relative surface velocity in the x-direction at the point where the two balls are in contact.
    // ball-ball slip
    FLOPS(10, complete_function, before_loop);
    double contact_point_velocity_x = local_velocity_x_1 - local_velocity_x_2 - R * (local_angular_velocity_z_1 + local_angular_velocity_z_2);
    double contact_point_velocity_z = R * (local_angular_velocity_x_1 + local_angular_velocity_x_2);
    double ball_ball_contact_point_magnitude = sqrt(contact_point_velocity_x * contact_point_velocity_x + contact_point_velocity_z * contact_point_velocity_z);
    //printf("\nC Contact Point Slide, Spin:\n");
    //printf("  Contact Point: u_ijC_xz_mag= %.6f\n", ball_ball_contact_point_magnitude);

    // Main collision loop
    FLOPS(1, complete_function, before_loop);
    double velocity_diff_y = local_velocity_y_2 - local_velocity_y_1;

    // deltaP is most likely always 0?
    // ΔP represents the Impulse during a time of Δt
    if (deltaP == 0) {
        FLOPS(5, complete_function, before_loop);
        deltaP = 0.5 * (1.0 + e_b) * M * fabs(velocity_diff_y) / (double)(N);
    }

    FLOPS(3, complete_function, before_loop);
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

    END_PROFILE(before_loop);
    START_PROFILE(loop);

    #ifdef PROFILE
         bool first_iter = true;
    #endif

    while (velocity_diff_y < 0 || total_work < work_required) {

        #ifdef PROFILE
            if(first_iter) {
                START_PROFILE(single_loop_iteration);
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

            FLOPS(4, complete_function, loop);
            FLOPS_SINGLE_LOOP(4);
            deltaP_1 = -u_b * deltaP * contact_point_velocity_x / ball_ball_contact_point_magnitude;
            if(fabs(contact_point_velocity_z) < 1e-16) {
                deltaP_2 = 0;
                deltaP_x_1 = 0;
                deltaP_y_1 = 0;
                deltaP_x_2 = 0;
                deltaP_y_2 = 0;
            } else {
                FLOPS(4, complete_function, loop);
                FLOPS_SINGLE_LOOP(4);
                deltaP_2 = -u_b * deltaP * contact_point_velocity_z / ball_ball_contact_point_magnitude;

                if(deltaP_2 > 0) {
                    deltaP_x_1 = 0;
                    deltaP_y_1 = 0;

                    // TODO: probably best to check for some tolerance
                    if(surface_velocity_magnitude_2 == 0.0) {
                        deltaP_x_2 = 0;
                        deltaP_y_2 = 0;
                    } else {
                        FLOPS(8, complete_function, loop);
                        FLOPS_SINGLE_LOOP(8);
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
                        FLOPS(6, complete_function, loop);
                        FLOPS_SINGLE_LOOP(6);
                        deltaP_x_1 = u_s1 * (surface_velocity_x_1 / surface_velocity_magnitude_1) * deltaP_2;
                        deltaP_y_1 = u_s1 * (surface_velocity_y_1 / surface_velocity_magnitude_1) * deltaP_2;
                    }
                }

            }
        }

        FLOPS(10, complete_function, loop);
        FLOPS_SINGLE_LOOP(10);
        // Velocity changes
        double velocity_change_x_1 = (deltaP_1 + deltaP_x_1) / M;
        double velocity_change_y_1 = (-deltaP + deltaP_y_1) / M;
        double velocity_change_x_2 = (-deltaP_1 + deltaP_x_2) / M;
        double velocity_change_y_2 = (deltaP + deltaP_y_2) / M;

        FLOPS(4, complete_function, loop);
        FLOPS_SINGLE_LOOP(4);
        // Update velocities
        local_velocity_x_1 += velocity_change_x_1;
        local_velocity_y_1 += velocity_change_y_1;
        local_velocity_x_2 += velocity_change_x_2;
        local_velocity_y_2 += velocity_change_y_2;

        FLOPS(12, complete_function, loop);
        FLOPS_SINGLE_LOOP(12);
        // Angular velocity changes
        double delta_angular_velocity_x_1 = C * (deltaP_2 + deltaP_y_1);
        double delta_angular_velocity_y_1 = C * (-deltaP_x_1);
        double delta_angular_velocity_z_1 = C * (-deltaP_1);
        double delta_angular_velocity_x_2 = C * (deltaP_2 + deltaP_y_2);
        double delta_angular_velocity_y_2 = C * (-deltaP_x_2);
        double delta_angular_velocity_z_2 = C * (-deltaP_1);

        FLOPS(6, complete_function, loop);
        FLOPS_SINGLE_LOOP(6);
        // Update Angular Velocities
        local_angular_velocity_x_1 += delta_angular_velocity_x_1;
        local_angular_velocity_y_1 += delta_angular_velocity_y_1;
        local_angular_velocity_z_1 += delta_angular_velocity_z_1;

        local_angular_velocity_x_2 += delta_angular_velocity_x_2;
        local_angular_velocity_y_2 += delta_angular_velocity_y_2;
        local_angular_velocity_z_2 += delta_angular_velocity_z_2;

        FLOPS(8, complete_function, loop);
        FLOPS_SINGLE_LOOP(8);
        // update ball-table slips
        surface_velocity_x_1 = local_velocity_x_1 + R * local_angular_velocity_y_1;
        surface_velocity_y_1 = local_velocity_y_1 - R * local_angular_velocity_x_1;
        surface_velocity_x_2 = local_velocity_x_2 + R * local_angular_velocity_y_2;
        surface_velocity_y_2 = local_velocity_y_2 - R * local_angular_velocity_x_2;

        FLOPS(8, complete_function, loop);
        FLOPS_SINGLE_LOOP(8);
        surface_velocity_magnitude_1 = sqrt(surface_velocity_x_1 * surface_velocity_x_1 + surface_velocity_y_1 * surface_velocity_y_1);
        surface_velocity_magnitude_2 = sqrt(surface_velocity_x_2 * surface_velocity_x_2 + surface_velocity_y_2 * surface_velocity_y_2);

        FLOPS(10, complete_function, loop);
        FLOPS_SINGLE_LOOP(10);
        // update ball-ball slip:
        contact_point_velocity_x = local_velocity_x_1 - local_velocity_x_2 - R * (local_angular_velocity_z_1 + local_angular_velocity_z_2);
        contact_point_velocity_z = R * (local_angular_velocity_x_1 + local_angular_velocity_x_2);
        ball_ball_contact_point_magnitude = sqrt(contact_point_velocity_x * contact_point_velocity_x + contact_point_velocity_z * contact_point_velocity_z);

        FLOPS(5, complete_function, loop);
        FLOPS_SINGLE_LOOP(5);
        // Update work and check compression phase
        double velocity_diff_y_temp = velocity_diff_y;
        velocity_diff_y = local_velocity_y_2 - local_velocity_y_1;
        total_work += 0.5 * deltaP * fabs(velocity_diff_y_temp + velocity_diff_y);

        if (work_compression == 0 && velocity_diff_y > 0) {
            work_compression = total_work;
            FLOPS(3, complete_function, loop);
            FLOPS_SINGLE_LOOP(3);
            work_required = (1.0 + e_b * e_b) * work_compression;
        }

        #ifdef PROFILE
            if(first_iter) {
                first_iter = false;
                END_PROFILE(single_loop_iteration);
            }
        #endif
    }

    END_PROFILE(loop);
    START_PROFILE(after_loop);

    // Transform back to global coordinates
    for (int i = 0; i < 3; i++) {

        MEMORY(4, complete_function, after_loop);
        FLOPS(6, complete_function, after_loop);
        rvw1_result[i + 3] = local_velocity_x_1 * right[i] + local_velocity_y_1 * forward[i];
        rvw2_result[i + 3] = local_velocity_x_2 * right[i] + local_velocity_y_2 * forward[i];

        if(i < 2) {
            FLOPS(6, complete_function, after_loop);
            rvw1_result[i + 6] = local_angular_velocity_x_1 * right[i] + local_angular_velocity_y_1 * forward[i];
            rvw2_result[i + 6] = local_angular_velocity_x_2 * right[i] + local_angular_velocity_y_2 * forward[i];
        }
        else {
            rvw1_result[i + 6] = local_angular_velocity_z_1;
            rvw2_result[i + 6] = local_angular_velocity_z_2;
        }
    }

    END_PROFILE(after_loop);
    END_PROFILE(complete_function);
}


DLL_EXPORT void code_motion_collide_balls(double* rvw1, double* rvw2, float R, float M, float u_s1, float u_s2, float u_b, float e_b, float deltaP, int N, double* rvw1_result, double* rvw2_result, Profile* profiles) {

    #ifdef PROFILE
        Profile* complete_function = &profiles[0];
        Profile* before_loop = &profiles[1];
        Profile* loop = &profiles[2];
        Profile* single_loop_iteration = &profiles[3];
        Profile* after_loop = &profiles[4];
    #endif

    START_PROFILE(complete_function);
    START_PROFILE(before_loop);

     // Get pointers into the state arrays
    MEMORY(18, complete_function, before_loop);
    double* translation_1 = &rvw1[0];
    double* velocity_1    = &rvw1[3];
    double* angular_1     = &rvw1[6];
    double* translation_2 = &rvw2[0];
    double* velocity_2    = &rvw2[3];
    double* angular_2     = &rvw2[6];

    FLOPS(3, complete_function, before_loop);
    double offset[3];
    offset[0] = translation_2[0] - translation_1[0];
    offset[1] = translation_2[1] - translation_1[1];
    offset[2] = translation_2[2] - translation_1[2];

    FLOPS(5, complete_function, before_loop);
    double offset_mag_sqrd;
    offset_mag_sqrd = offset[0] * offset[0] +
    offset[1] * offset[1] +
    offset[2] * offset[2];

    FLOPS(1, complete_function, before_loop);
    double offset_mag = sqrt(offset_mag_sqrd);

    FLOPS(3, complete_function, before_loop);
    double forward[3]; // Forward from ball 1 to ball 2, normalized
    forward[0] = offset[0] / offset_mag;
    forward[1] = offset[1] / offset_mag;
    forward[2] = offset[2] / offset_mag;

    FLOPS(0, complete_function, before_loop);
    double right[3]; // Axis orthogonal to Z and forward
    right[0] = forward[1];
    right[1] = -forward[0];
    right[2] = 0;

    // From here on, it is assumed that the x axis is the right axis and y axis is the forward axis
    // Transform velocities to local frame

    FLOPS(5*4, complete_function, before_loop);
    double local_vel_1[2] = {
        velocity_1[0] * right[0] + velocity_1[1] * right[1] + velocity_1[2] * right[2],
        velocity_1[0] * forward[0] + velocity_1[1] * forward[1] + velocity_1[2] * forward[2]
    };

    double local_vel_2[2] = {
        velocity_2[0] * right[0] + velocity_2[1] * right[1] + velocity_2[2] * right[2],
        velocity_2[0] * forward[0] + velocity_2[1] * forward[1] + velocity_2[2] * forward[2]
    };

    // Transform angular velocities into local frame
    FLOPS(10, complete_function, before_loop);
    double local_ang_1[3] = {
        angular_1[0] * right[0] + angular_1[1] * right[1] + angular_1[2] * right[2],
        angular_1[0] * forward[0] + angular_1[1] * forward[1] + angular_1[2] * forward[2],
        angular_1[2]
    };

    FLOPS(10, complete_function, before_loop);
    double local_ang_2[3] = {
        angular_2[0] * right[0] + angular_2[1] * right[1] + angular_2[2] * right[2],
        angular_2[0] * forward[0] + angular_2[1] * forward[1] + angular_2[2] * forward[2],
        angular_2[2]
    };


    // Calculate velocity at contact point
    // = Calculate ball-table slips?
    // Slip refers to relative motion between two surfaces in contact — here, the ball and the table.
    // Its the velocity at the contact point of the table and the ball
    FLOPS(8, complete_function, before_loop);
    double surf_vel_1[2] = { local_vel_1[0] + R * local_ang_1[1],
            local_vel_1[1] - R * local_ang_1[0] };
    double surf_vel_2[2] = { local_vel_2[0] + R * local_ang_2[1],
            local_vel_2[1] - R * local_ang_2[0] };

    FLOPS(8, complete_function, before_loop);
    double surf_vel_mag_1 = sqrt(surf_vel_1[0]*surf_vel_1[0] + surf_vel_1[1]*surf_vel_1[1]);
    double surf_vel_mag_2 = sqrt(surf_vel_2[0]*surf_vel_2[0] + surf_vel_2[1]*surf_vel_2[1]);

    // Relative surface velocity in the x-direction at the point where the two balls are in contact.
    // ball-ball slip
    FLOPS(10, complete_function, before_loop);
    double contact_vel[2] = { local_vel_1[0] - local_vel_2[0] - R * (local_ang_1[2] + local_ang_2[2]),
            R * (local_ang_1[0] + local_ang_2[0]) };
    double ball_ball_contact_mag = sqrt(contact_vel[0]*contact_vel[0] + contact_vel[1]*contact_vel[1]);

    FLOPS(1, complete_function, before_loop);
    // Main collision loop
    double velocity_diff_y = local_vel_2[1] - local_vel_1[1];

    // deltaP is most likely always 0?
    // ΔP represents the Impulse during a time of Δt
    if (deltaP == 0) {
        FLOPS(5, complete_function, before_loop);
        deltaP = 0.5 * (1.0 + e_b) * M * fabs(velocity_diff_y)/(double)N;
    }

    FLOPS(3, complete_function, before_loop);
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

    END_PROFILE(before_loop);
    START_PROFILE(loop);

    #ifdef PROFILE
         bool first_iter = true;
    #endif
    while (velocity_diff_y < 0 || total_work < work_required) {

        #ifdef PROFILE
            if(first_iter) {
                START_PROFILE(single_loop_iteration);
            }
        #endif

        // Impulse Calculation
        if (ball_ball_contact_mag < 1e-16) {
            deltaP_ball[0] = deltaP_ball[1] = 0;
            deltaP_axis_1[0] = deltaP_axis_1[1] = 0;
            deltaP_axis_2[0] = deltaP_axis_2[1] = 0;
        } else {
            FLOPS(4, complete_function, loop);
            FLOPS_SINGLE_LOOP(4);
            deltaP_ball[0] = -u_b * deltaP * contact_vel[0] / ball_ball_contact_mag;
            if(fabs(contact_vel[1]) < 1e-16) {
                deltaP_ball[1] = 0;
                deltaP_axis_1[0] = deltaP_axis_1[1] = 0;
                deltaP_axis_2[0] = deltaP_axis_2[1] = 0;
            } else {
                FLOPS(4, complete_function, loop);
                FLOPS_SINGLE_LOOP(4);
                deltaP_ball[1] = -u_b * deltaP * contact_vel[1] / ball_ball_contact_mag;
                if(deltaP_ball[1] > 0) {
                    deltaP_axis_1[0] = deltaP_axis_1[1] = 0;
                    if(surf_vel_mag_2 == 0.0) {
                        deltaP_axis_2[0] = deltaP_axis_2[1] = 0;
                    } else {
                        FLOPS(8, complete_function, loop);
                        FLOPS_SINGLE_LOOP(8);
                        deltaP_axis_2[0] = -u_s2 * (surf_vel_2[0]/surf_vel_mag_2) * deltaP_ball[1];
                        deltaP_axis_2[1] = -u_s2 * (surf_vel_2[1]/surf_vel_mag_2) * deltaP_ball[1];
                    }
                } else {
                    deltaP_axis_2[0] = deltaP_axis_2[1] = 0;
                    if(surf_vel_mag_1 == 0.0) {
                        deltaP_axis_1[0] = deltaP_axis_1[1] = 0;
                    } else {
                        FLOPS(6, complete_function, loop);
                        FLOPS_SINGLE_LOOP(6);
                        deltaP_axis_1[0] = u_s1 * (surf_vel_1[0]/surf_vel_mag_1) * deltaP_ball[1];
                        deltaP_axis_1[1] = u_s1 * (surf_vel_1[1]/surf_vel_mag_1) * deltaP_ball[1];
                    }
                }
            }
        }

        FLOPS(10, complete_function, loop);
        FLOPS_SINGLE_LOOP(10);
        // Velocity changes
        double v_change1[2] = { (deltaP_ball[0] + deltaP_axis_1[0]) / M,
                    (-deltaP + deltaP_axis_1[1]) / M };
        double v_change2[2] = { (-deltaP_ball[0] + deltaP_axis_2[0]) / M,
                    ( deltaP + deltaP_axis_2[1]) / M };


        FLOPS(4, complete_function, loop);
        FLOPS_SINGLE_LOOP(4);
        // Update velocities
        local_vel_1[0] += v_change1[0];
        local_vel_1[1] += v_change1[1];
        local_vel_2[0] += v_change2[0];
        local_vel_2[1] += v_change2[1];

        FLOPS(12, complete_function, loop);
        FLOPS_SINGLE_LOOP(12);
        // Angular velocity changes
        double ang_change1[3] = { C * (deltaP_ball[1] + deltaP_axis_1[1]),
                    C * (-deltaP_axis_1[0]),
                    C * (-deltaP_ball[0]) };
        double ang_change2[3] = { C * (deltaP_ball[1] + deltaP_axis_2[1]),
                    C * (-deltaP_axis_2[0]),
                    C * (-deltaP_ball[0]) };

        FLOPS(6, complete_function, loop);
        FLOPS_SINGLE_LOOP(6);
        // Update Angular Velocities
        local_ang_1[0] += ang_change1[0];
        local_ang_1[1] += ang_change1[1];
        local_ang_1[2] += ang_change1[2];

        local_ang_2[0] += ang_change2[0];
        local_ang_2[1] += ang_change2[1];
        local_ang_2[2] += ang_change2[2];


        FLOPS(8, complete_function, loop);
        FLOPS_SINGLE_LOOP(8);
        // Recalculate surface velocities using the updated arrays
        surf_vel_1[0] = local_vel_1[0] + R * local_ang_1[1];
        surf_vel_1[1] = local_vel_1[1] - R * local_ang_1[0];
        surf_vel_2[0] = local_vel_2[0] + R * local_ang_2[1];
        surf_vel_2[1] = local_vel_2[1] - R * local_ang_2[0];

        FLOPS(8, complete_function, loop);
        FLOPS_SINGLE_LOOP(8);
        surf_vel_mag_1 = sqrt(surf_vel_1[0]*surf_vel_1[0] + surf_vel_1[1]*surf_vel_1[1]);
        surf_vel_mag_2 = sqrt(surf_vel_2[0]*surf_vel_2[0] + surf_vel_2[1]*surf_vel_2[1]);

        FLOPS(10, complete_function, loop);
        FLOPS_SINGLE_LOOP(10);
        // update ball-ball slip:
        contact_vel[0] = local_vel_1[0] - local_vel_2[0] - R * (local_ang_1[2] + local_ang_2[2]);
        contact_vel[1] = R * (local_ang_1[0] + local_ang_2[0]);
        ball_ball_contact_mag = sqrt(contact_vel[0]*contact_vel[0] + contact_vel[1]*contact_vel[1]);

        FLOPS(5, complete_function, loop);
        FLOPS_SINGLE_LOOP(5);
        // Update work and check compression phase
        double prev_diff = velocity_diff_y;
        velocity_diff_y = local_vel_2[1] - local_vel_1[1];
        total_work += 0.5 * deltaP * fabs(prev_diff + velocity_diff_y);

        if (work_compression == 0 && velocity_diff_y > 0) {
            work_compression = total_work;
            FLOPS(3, complete_function, loop);
            FLOPS_SINGLE_LOOP(3);
            work_required = (1.0 + e_b * e_b) * work_compression;
        }

        #ifdef PROFILE
            if(first_iter) {
                first_iter = false;
                END_PROFILE(single_loop_iteration);
            }
        #endif
    }

    END_PROFILE(loop);
    START_PROFILE(after_loop);

    // Transform back to global coordinates
    for (int i = 0; i < 3; i++) {
        MEMORY(4, complete_function, after_loop);

        FLOPS(6, complete_function, after_loop);
        rvw1_result[i+3] = local_vel_1[0] * right[i] + local_vel_1[1] * forward[i];
        rvw2_result[i+3] = local_vel_2[0] * right[i] + local_vel_2[1] * forward[i];
        if(i < 2) {
            FLOPS(6, complete_function, after_loop);
            rvw1_result[i+6] = local_ang_1[0] * right[i] + local_ang_1[1] * forward[i];
            rvw2_result[i+6] = local_ang_2[0] * right[i] + local_ang_2[1] * forward[i];
        }
        else {
            rvw1_result[i+6] = local_ang_1[2];
            rvw2_result[i+6] = local_ang_2[2];
        }
    }

    END_PROFILE(after_loop);
    END_PROFILE(complete_function);
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
