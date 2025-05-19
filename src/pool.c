#include "pool.h"
#include "math_helper.h"

#include <emmintrin.h>
#include <immintrin.h>
#include <smmintrin.h>
#include <stdio.h>
#include <math.h>
#include <stdbool.h>

#ifdef _MSC_VER
    #include <intrin.h>
    #include <windows.h>
#else
    #include <x86intrin.h>
#endif

#ifndef likely
#   define likely(x)   __builtin_expect(!!(x), 1)
#   define unlikely(x) __builtin_expect(!!(x), 0)
#endif

#ifdef FLOP_COUNT

    #define FLOPS(adds, muls, divs, sqrt, ...) \
    do { \
        void *arr[] = { __VA_ARGS__ }; \
        for (size_t i = 0; i < sizeof(arr)/sizeof(void*); i++) { \
            ((Profile*)arr[i])->flops += (adds + muls + divs + sqrt); \
            ((Profile*)arr[i])->ADDS += (adds); \
            ((Profile*)arr[i])->MULS += (muls); \
            ((Profile*)arr[i])->DIVS += (divs); \
            ((Profile*)arr[i])->SQRT += (sqrt); \
        } \
    } while(0)

    #define MEMORY(count, ...) \
    do { \
        void *arr[] = { __VA_ARGS__ }; \
        for (size_t i = 0; i < sizeof(arr)/sizeof(void*); i++) { \
            ((Profile*)arr[i])->memory += (count); \
        } \
    } while(0)

    #define BRANCH(i) branches[i].count++

#else

#define FLOPS(adds, muls, divs, sqrt, ...)
#define MEMORY(count, ...)
#define BRANCH(i)
#endif


DLL_EXPORT void hello_world(const char* matrix_name, double* rvw) {

    printf("Received %s\n", matrix_name);
   for(int i = 0; i < 9; i++) {
       printf("%f ", rvw[i]);
       if((i + 1) % 3 == 0) printf("\n");
   }

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

void init_profiling_section(Profile* profile) {
    profile->cycle_start = 0;
    profile->cycles_cumulative = 0;
    profile->flops = 0;
    profile->memory = 0;
    profile->ADDS = 0;
    profile->MULS = 0;
    profile->DIVS = 0;
    profile->SQRT = 0;
}

static inline void start_profiling_section(Profile* profile) {
    profile->cycle_start = start_tsc();
}

static inline void end_profiling_section(Profile* profile) {
    profile->cycles_cumulative += stop_tsc(profile->cycle_start);
}

#ifdef PROFILE

#define START_PROFILE(profile) start_profiling_section(profile)
#define END_PROFILE(profile) end_profiling_section(profile)

#else

#define START_PROFILE(profile)
#define END_PROFILE(profile)

#endif

DLL_EXPORT void collide_balls(double* rvw1, double* rvw2, float R, float M, float u_s1, float u_s2, float u_b, float e_b, float deltaP, int N, double* rvw1_result, double* rvw2_result, Profile* profiles, Branch* branches) {

    #ifdef PROFILE
        Profile* complete_function = &profiles[0];
        Profile* before_loop = &profiles[1];
        Profile* impulse = &profiles[2];
        Profile* delta = &profiles[3];
        Profile* velocity = &profiles[4];
        Profile* after_loop = &profiles[5];
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

    FLOPS(3, 0, 0, 0, complete_function, before_loop);
    double offset[3];
    subV3(translation_2, translation_1, offset);

    FLOPS(2, 3, 0, 0, complete_function, before_loop);
    double offset_mag_sqrd = dotV3(offset, offset);

    FLOPS(0, 0, 0, 1, complete_function, before_loop);
    double offset_mag = sqrt(offset_mag_sqrd);

    double forward[3]; // Forward from ball 1 to ball 2, normalized

    FLOPS(0, 0, 3, 0, complete_function, before_loop);
    divV3(offset, offset_mag, forward);

    double up[3] = {0, 0, 1}; // Probably up axis?

    double right[3]; // Axis orthogonal to Z and forward

    FLOPS(3, 6, 0, 0, complete_function, before_loop);
    crossV3(forward, up, right);

    // From here on, it is assumed that the x axis is the right axis and y axis is the forward axis
    // Transform velocities to local frame

    FLOPS(2 * 4, 3 * 4, 0, 0, complete_function, before_loop);
    double local_velocity_x_1 = dotV3(velocity_1, right);
    double local_velocity_y_1 = dotV3(velocity_1, forward);
    double local_velocity_x_2 = dotV3(velocity_2, right);
    double local_velocity_y_2 = dotV3(velocity_2, forward);

    // Transform angular velocities into local frame

    FLOPS(2*6, 3*6, 0, 0, complete_function, before_loop);
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
    FLOPS(4, 4, 0, 0, complete_function, before_loop);
    double surface_velocity_x_1 = local_velocity_x_1 + R * local_angular_velocity_y_1;
    double surface_velocity_y_1 = local_velocity_y_1 - R * local_angular_velocity_x_1;
    double surface_velocity_x_2 = local_velocity_x_2 + R * local_angular_velocity_y_2;
    double surface_velocity_y_2 = local_velocity_y_2 - R * local_angular_velocity_x_2;

    FLOPS(2, 4, 0, 2, complete_function, before_loop);
    double surface_velocity_magnitude_1 = sqrt(surface_velocity_x_1 * surface_velocity_x_1 + surface_velocity_y_1 * surface_velocity_y_1);
    double surface_velocity_magnitude_2 = sqrt(surface_velocity_x_2 * surface_velocity_x_2 + surface_velocity_y_2 * surface_velocity_y_2);

    // Relative surface velocity in the x-direction at the point where the two balls are in contact.
    // ball-ball slip
    FLOPS(5, 4, 0, 1, complete_function, before_loop);
    double contact_point_velocity_x = local_velocity_x_1 - local_velocity_x_2 - R * (local_angular_velocity_z_1 + local_angular_velocity_z_2);
    double contact_point_velocity_z = R * (local_angular_velocity_x_1 + local_angular_velocity_x_2);
    double ball_ball_contact_point_magnitude = sqrt(contact_point_velocity_x * contact_point_velocity_x + contact_point_velocity_z * contact_point_velocity_z);
    //printf("\nC Contact Point Slide, Spin:\n");
    //printf("  Contact Point: u_ijC_xz_mag= %.6f\n", ball_ball_contact_point_magnitude);

    // Main collision loop
    FLOPS(1, 0, 0, 0, complete_function, before_loop);
    double velocity_diff_y = local_velocity_y_2 - local_velocity_y_1;

    // deltaP is most likely always 0?
    // ΔP represents the Impulse during a time of Δt
    if (deltaP == 0) {
        FLOPS(0, 4, 1, 0, complete_function, before_loop);
        deltaP = 0.5 * (1.0 + e_b) * M * fabs(velocity_diff_y) / (double)(N);
    }

    FLOPS(0, 2, 1, 0, complete_function, before_loop);
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

    while (velocity_diff_y < 0 || total_work < work_required) {

        // printf("0 %lf %lf %lf\n", surface_velocity_magnitude_1, ball_ball_contact_point_magnitude, surface_velocity_magnitude_2);
        // printf("0 %lf %lf \n", contact_point_velocity_x, contact_point_velocity_z);
        //printf("1 %lf %lf %lf %lf \n", surface_velocity_x_1, surface_velocity_y_1, surface_velocity_x_2, surface_velocity_y_2);
        // printf("1 %lf %lf \n", local_angular_velocity_z_1, local_angular_velocity_z_2);
        START_PROFILE(impulse);

        // Impulse Calculation
        if (ball_ball_contact_point_magnitude < 1e-16) {
            BRANCH(0);
            deltaP_1 = 0;
            deltaP_2 = 0;
            deltaP_x_1 = 0;
            deltaP_y_1 = 0;
            deltaP_x_2 = 0;
            deltaP_y_2 = 0;
        } else {
            BRANCH(1);
            FLOPS(1, 2, 1, 0, complete_function, impulse);
            deltaP_1 = -u_b * deltaP * contact_point_velocity_x / ball_ball_contact_point_magnitude;
            if(fabs(contact_point_velocity_z) < 1e-16) {
                BRANCH(2);
                deltaP_2 = 0;
                deltaP_x_1 = 0;
                deltaP_y_1 = 0;
                deltaP_x_2 = 0;
                deltaP_y_2 = 0;
            } else {
                BRANCH(3);
                FLOPS(1, 2, 1, 0, complete_function, impulse);
                deltaP_2 = -u_b * deltaP * contact_point_velocity_z / ball_ball_contact_point_magnitude;

                if(deltaP_2 > 0) {
                    BRANCH(4);
                    deltaP_x_1 = 0;
                    deltaP_y_1 = 0;

                    // TODO: probably best to check for some tolerance
                    if(surface_velocity_magnitude_2 == 0.0) {
                        BRANCH(5);
                        deltaP_x_2 = 0;
                        deltaP_y_2 = 0;
                    } else {
                        BRANCH(6);
                        FLOPS(2, 4, 2, 0, complete_function, impulse);
                        deltaP_x_2 = -u_s2 * (surface_velocity_x_2 / surface_velocity_magnitude_2) * deltaP_2;
                        deltaP_y_2 = -u_s2 * (surface_velocity_y_2 / surface_velocity_magnitude_2) * deltaP_2;
                    }
                } else {
                    BRANCH(7);
                    deltaP_x_2 = 0;
                    deltaP_y_2 = 0;
                    if(surface_velocity_magnitude_1 == 0.0) {
                        BRANCH(8);
                        deltaP_x_1 = 0;
                        deltaP_y_1 = 0;
                    } else {
                        BRANCH(9);
                        FLOPS(0, 4, 2, 0, complete_function, impulse);
                        deltaP_x_1 = u_s1 * (surface_velocity_x_1 / surface_velocity_magnitude_1) * deltaP_2;
                        deltaP_y_1 = u_s1 * (surface_velocity_y_1 / surface_velocity_magnitude_1) * deltaP_2;
                    }
                }

            }
        }

        // printf("0 %f %lf %lf %lf %lf %lf %lf\n", deltaP, deltaP_1, deltaP_2, deltaP_x_1, deltaP_y_1, deltaP_x_2, deltaP_y_2);

        END_PROFILE(impulse);
        START_PROFILE(delta);

        FLOPS(6, 0, 4, 0, complete_function, delta);
        // Velocity changes
        double velocity_change_x_1 = (deltaP_1 + deltaP_x_1) / M;
        double velocity_change_y_1 = (-deltaP + deltaP_y_1) / M;
        double velocity_change_x_2 = (-deltaP_1 + deltaP_x_2) / M;
        double velocity_change_y_2 = (deltaP + deltaP_y_2) / M;

        FLOPS(4, 0, 0, 0, complete_function, delta);
        // Update velocities
        local_velocity_x_1 += velocity_change_x_1;
        local_velocity_y_1 += velocity_change_y_1;
        local_velocity_x_2 += velocity_change_x_2;
        local_velocity_y_2 += velocity_change_y_2;

        FLOPS(6, 6, 0, 0, complete_function, delta);
        // Angular velocity changes
        double delta_angular_velocity_x_1 = C * (deltaP_2 + deltaP_y_1);
        double delta_angular_velocity_y_1 = C * (-deltaP_x_1);
        double delta_angular_velocity_z_1 = C * (-deltaP_1);
        double delta_angular_velocity_x_2 = C * (deltaP_2 + deltaP_y_2);
        double delta_angular_velocity_y_2 = C * (-deltaP_x_2);
        double delta_angular_velocity_z_2 = C * (-deltaP_1);

        FLOPS(6, 0, 0, 0, complete_function, delta);
        // Update Angular Velocities
        local_angular_velocity_x_1 += delta_angular_velocity_x_1;
        local_angular_velocity_y_1 += delta_angular_velocity_y_1;
        local_angular_velocity_z_1 += delta_angular_velocity_z_1;

        local_angular_velocity_x_2 += delta_angular_velocity_x_2;
        local_angular_velocity_y_2 += delta_angular_velocity_y_2;
        local_angular_velocity_z_2 += delta_angular_velocity_z_2;

        END_PROFILE(delta);
        START_PROFILE(velocity);

        FLOPS(4, 4, 0, 0, complete_function, velocity);
        // update ball-table slips
        surface_velocity_x_1 = local_velocity_x_1 + R * local_angular_velocity_y_1;
        surface_velocity_y_1 = local_velocity_y_1 - R * local_angular_velocity_x_1;
        surface_velocity_x_2 = local_velocity_x_2 + R * local_angular_velocity_y_2;
        surface_velocity_y_2 = local_velocity_y_2 - R * local_angular_velocity_x_2;

        FLOPS(2, 4, 0, 2, complete_function, velocity);
        surface_velocity_magnitude_1 = sqrt(surface_velocity_x_1 * surface_velocity_x_1 + surface_velocity_y_1 * surface_velocity_y_1);
        surface_velocity_magnitude_2 = sqrt(surface_velocity_x_2 * surface_velocity_x_2 + surface_velocity_y_2 * surface_velocity_y_2);

        FLOPS(5, 4, 0, 1, complete_function, velocity);
        // update ball-ball slip:
        contact_point_velocity_x = local_velocity_x_1 - local_velocity_x_2 - R * (local_angular_velocity_z_1 + local_angular_velocity_z_2);
        contact_point_velocity_z = R * (local_angular_velocity_x_1 + local_angular_velocity_x_2);
        ball_ball_contact_point_magnitude = sqrt(contact_point_velocity_x * contact_point_velocity_x + contact_point_velocity_z * contact_point_velocity_z);

        FLOPS(3, 2, 0, 0, complete_function, velocity);
        // Update work and check compression phase
        double velocity_diff_y_temp = velocity_diff_y;
        velocity_diff_y = local_velocity_y_2 - local_velocity_y_1;
        total_work += 0.5 * deltaP * fabs(velocity_diff_y_temp + velocity_diff_y);

        if (work_compression == 0 && velocity_diff_y > 0) {
            work_compression = total_work;
            FLOPS(1, 2, 0, 0, complete_function, velocity);
            work_required = (1.0 + e_b * e_b) * work_compression;
        }

        END_PROFILE(velocity);
    }



    START_PROFILE(after_loop);

    // Transform back to global coordinates
    for (int i = 0; i < 3; i++) {

        MEMORY(4, complete_function, after_loop);
        FLOPS(2, 4, 0, 0, complete_function, after_loop);
        rvw1_result[i + 3] = local_velocity_x_1 * right[i] + local_velocity_y_1 * forward[i];
        rvw2_result[i + 3] = local_velocity_x_2 * right[i] + local_velocity_y_2 * forward[i];

        if(i < 2) {
            FLOPS(2, 4, 0, 0, complete_function, after_loop);
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


DLL_EXPORT void scalar_improvements(double* rvw1, double* rvw2, float R, float M, float u_s1, float u_s2, float u_b, float e_b, float deltaP, int N, double* rvw1_result, double* rvw2_result, Profile* profiles, Branch* branches) {
    #ifdef PROFILE
        Profile* complete_function = &profiles[0];
        Profile* before_loop       = &profiles[1];
        Profile* impulse           = &profiles[2];
        Profile* delta             = &profiles[3];
        Profile* velocity          = &profiles[4];
        Profile* after_loop        = &profiles[5];
    #endif
    START_PROFILE(complete_function);
    START_PROFILE(before_loop);

    MEMORY(18, complete_function, before_loop);
    double* translation_1     = get_displacement     (rvw1);
    double* velocity_1        = get_velocity         (rvw1);
    double* angular_velocity_1= get_angular_velocity (rvw1);

    double* translation_2     = get_displacement     (rvw2);
    double* velocity_2        = get_velocity         (rvw2);
    double* angular_velocity_2= get_angular_velocity (rvw2);

    /* ------------------------------------------------------------------ */
    /* ----------------------   scalar tweaks   -------------------------- */
    /* ------------------------------------------------------------------ */

    double invM     = 1.0 / M;           /* division → multiply   */
    double invR     = 1.0 / R;
    double C        = 5.0 * invM * invR * 0.5;   /* 5/(2MR) */

    double offset[3];
    subV3(translation_2, translation_1, offset);

    double offset_mag_sqrd = dotV3(offset, offset);
    double offset_inv_mag  = 1.0 / sqrt(offset_mag_sqrd);
    double forward[3];
    forward[0] = offset[0] * offset_inv_mag;
    forward[1] = offset[1] * offset_inv_mag;
    forward[2] = offset[2] * offset_inv_mag;

    double up[3] = {0.0, 0.0, 1.0};
    double right[3];
    crossV3(forward, up, right);

    /* ---------------- velocities to local frame ----------------------- */
    double local_velocity_x_1      = dotV3(velocity_1,  right);
    double local_velocity_y_1      = dotV3(velocity_1,  forward);
    double local_velocity_x_2      = dotV3(velocity_2,  right);
    double local_velocity_y_2      = dotV3(velocity_2,  forward);

    /* --------------- angular velocities to local frame ---------------- */
    double local_angular_velocity_x_1 = dotV3(angular_velocity_1, right);
    double local_angular_velocity_y_1 = dotV3(angular_velocity_1, forward);
    double local_angular_velocity_z_1 = dotV3(angular_velocity_1, up);

    double local_angular_velocity_x_2 = dotV3(angular_velocity_2, right);
    double local_angular_velocity_y_2 = dotV3(angular_velocity_2, forward);
    double local_angular_velocity_z_2 = dotV3(angular_velocity_2, up);

    /* ---------------- surface‑velocity helpers (use fma) -------------- */
    double surface_velocity_x_1 = fma(R, local_angular_velocity_y_1,  local_velocity_x_1);
    double surface_velocity_y_1 = fma(-R, local_angular_velocity_x_1, local_velocity_y_1);
    double surface_velocity_x_2 = fma(R, local_angular_velocity_y_2,  local_velocity_x_2);
    double surface_velocity_y_2 = fma(-R, local_angular_velocity_x_2, local_velocity_y_2);

    double surface_velocity_mag1_sq = surface_velocity_x_1*surface_velocity_x_1
                                    + surface_velocity_y_1*surface_velocity_y_1;
    double surface_velocity_mag2_sq = surface_velocity_x_2*surface_velocity_x_2
                                    + surface_velocity_y_2*surface_velocity_y_2;

    /* ---------------------- contact point slip ------------------------ */
    double contact_point_velocity_x =  local_velocity_x_1 - local_velocity_x_2
                                     - R*(local_angular_velocity_z_1 + local_angular_velocity_z_2);
    double contact_point_velocity_z =  R*(local_angular_velocity_x_1 + local_angular_velocity_x_2);
    double contact_inv_mag          = 1.0 / sqrt(contact_point_velocity_x*contact_point_velocity_x +
                                                 contact_point_velocity_z*contact_point_velocity_z);
    double ball_ball_contact_point_magnitude =
        1.0 / contact_inv_mag;  /* keep original scalar around for profiling */

    /* --------------------------- impulse step ------------------------- */
    double velocity_diff_y = local_velocity_y_2 - local_velocity_y_1;

    if (unlikely(deltaP == 0.0f)) {
        deltaP = 0.5 * (1.0 + e_b) * M * fabs(velocity_diff_y) / (double)N;
    }

    /* bookkeeping (unchanged) */
    double total_work      = 0.0;
    double work_required   = INFINITY;
    double work_compression= 0.0;

    double deltaP_1 = deltaP, deltaP_2 = deltaP;
    double deltaP_x_1 = 0, deltaP_y_1 = 0, deltaP_x_2 = 0, deltaP_y_2 = 0;

    END_PROFILE(before_loop);
    while (velocity_diff_y < 0.0 || total_work < work_required)
    {
        /* -------------------- impulse calculation -------------------- */
        START_PROFILE(impulse);

        if (unlikely(ball_ball_contact_point_magnitude < 1e-16)) {
            BRANCH(0);
            deltaP_1 = deltaP_2 = 0.0;
            deltaP_x_1 = deltaP_y_1 = deltaP_x_2 = deltaP_y_2 = 0.0;
        } else {
            BRANCH(1);
            double inv_mag = contact_inv_mag;          /* already computed */
            deltaP_1 = -u_b * deltaP * contact_point_velocity_x * inv_mag;

            if (unlikely(fabs(contact_point_velocity_z) < 1e-16)) {
                BRANCH(2);
                deltaP_2 = deltaP_x_1 = deltaP_y_1 = deltaP_x_2 = deltaP_y_2 = 0.0;
            } else {
                BRANCH(3);
                deltaP_2 = -u_b * deltaP * contact_point_velocity_z * inv_mag;

                if (deltaP_2 > 0.0) {
                    BRANCH(4);
                    deltaP_x_1 = deltaP_y_1 = 0.0;

                    if (unlikely(surface_velocity_mag2_sq == 0.0)) {
                        BRANCH(5);
                        deltaP_x_2 = deltaP_y_2 = 0.0;
                    } else {
                        BRANCH(6);
                        double inv_sv2 = 1.0 / sqrt(surface_velocity_mag2_sq);
                        deltaP_x_2 = -u_s2 * surface_velocity_x_2 * inv_sv2 * deltaP_2;
                        deltaP_y_2 = -u_s2 * surface_velocity_y_2 * inv_sv2 * deltaP_2;
                    }
                } else {
                    BRANCH(7);
                    deltaP_x_2 = deltaP_y_2 = 0.0;

                    if (unlikely(surface_velocity_mag1_sq == 0.0)) {
                        BRANCH(8);
                        deltaP_x_1 = deltaP_y_1 = 0.0;
                    } else {
                        BRANCH(9);
                        double inv_sv1 = 1.0 / sqrt(surface_velocity_mag1_sq);
                        deltaP_x_1 =  u_s1 * surface_velocity_x_1 * inv_sv1 * deltaP_2;
                        deltaP_y_1 =  u_s1 * surface_velocity_y_1 * inv_sv1 * deltaP_2;
                    }
                }
            }
        }
        END_PROFILE(impulse);

        /* ------------------ update linear + angular vel -------------- */
        START_PROFILE(delta);

        double velocity_change_x_1 = (deltaP_1 + deltaP_x_1) * invM;
        double velocity_change_y_1 = (-deltaP   + deltaP_y_1) * invM;
        double velocity_change_x_2 = (-deltaP_1 + deltaP_x_2) * invM;
        double velocity_change_y_2 = ( deltaP   + deltaP_y_2) * invM;

        local_velocity_x_1 += velocity_change_x_1;
        local_velocity_y_1 += velocity_change_y_1;
        local_velocity_x_2 += velocity_change_x_2;
        local_velocity_y_2 += velocity_change_y_2;

        /* angular */
        local_angular_velocity_x_1 += C * (deltaP_2 + deltaP_y_1);
        local_angular_velocity_y_1 += C * (-deltaP_x_1);
        local_angular_velocity_z_1 += C * (-deltaP_1);

        local_angular_velocity_x_2 += C * (deltaP_2 + deltaP_y_2);
        local_angular_velocity_y_2 += C * (-deltaP_x_2);
        local_angular_velocity_z_2 += C * (-deltaP_1);

        END_PROFILE(delta);

        /* ----------------- recompute helpers for next iter ----------- */
        START_PROFILE(velocity);

        surface_velocity_x_1 = fma(R,  local_angular_velocity_y_1,  local_velocity_x_1);
        surface_velocity_y_1 = fma(-R, local_angular_velocity_x_1,  local_velocity_y_1);
        surface_velocity_x_2 = fma(R,  local_angular_velocity_y_2,  local_velocity_x_2);
        surface_velocity_y_2 = fma(-R, local_angular_velocity_x_2,  local_velocity_y_2);

        surface_velocity_mag1_sq = surface_velocity_x_1*surface_velocity_x_1
                                + surface_velocity_y_1*surface_velocity_y_1;
        surface_velocity_mag2_sq = surface_velocity_x_2*surface_velocity_x_2
                                + surface_velocity_y_2*surface_velocity_y_2;

        contact_point_velocity_x = local_velocity_x_1 - local_velocity_x_2
                                 - (local_angular_velocity_z_1 + local_angular_velocity_z_2)*R;
        contact_point_velocity_z = (local_angular_velocity_x_1 + local_angular_velocity_x_2)*R;
        contact_inv_mag          = 1.0 / sqrt(contact_point_velocity_x*contact_point_velocity_x +
                                              contact_point_velocity_z*contact_point_velocity_z);
        ball_ball_contact_point_magnitude = 1.0 / contact_inv_mag;   /* for branch test */

        /* work / compression bookkeeping (unchanged) */
        double velocity_diff_y_prev = velocity_diff_y;
        velocity_diff_y   = local_velocity_y_2 - local_velocity_y_1;
        total_work       += 0.5 * deltaP * fabs(velocity_diff_y_prev + velocity_diff_y);

        if (work_compression == 0.0 && velocity_diff_y > 0.0) {
            work_compression = total_work;
            work_required    = (1.0 + e_b * e_b) * work_compression;
        }

        END_PROFILE(velocity);
    }
    /* ------------------------------------------------------------------ */
    /* ---------------------- epilogue – UNCHANGED ----------------------- */
    /* ------------------------------------------------------------------ */
    START_PROFILE(after_loop);
    for (int i = 0; i < 3; ++i) {
        rvw1_result[i+3] = local_velocity_x_1*right[i] + local_velocity_y_1*forward[i];
        rvw2_result[i+3] = local_velocity_x_2*right[i] + local_velocity_y_2*forward[i];

        if (i < 2) {
            rvw1_result[i+6] = local_angular_velocity_x_1*right[i] + local_angular_velocity_y_1*forward[i];
            rvw2_result[i+6] = local_angular_velocity_x_2*right[i] + local_angular_velocity_y_2*forward[i];
        } else {
            rvw1_result[i+6] = local_angular_velocity_z_1;
            rvw2_result[i+6] = local_angular_velocity_z_2;
        }
    }
    END_PROFILE(after_loop);
    END_PROFILE(complete_function);
}

DLL_EXPORT void scalar_less_sqrt(double* rvw1, double* rvw2, float R, float M, float u_s1, float u_s2, float u_b, float e_b, float deltaP, int N, double* rvw1_result, double* rvw2_result, Profile* profiles, Branch* branches) {
    #ifdef PROFILE
        Profile* complete_function = &profiles[0];
        Profile* before_loop       = &profiles[1];
        Profile* impulse           = &profiles[2];
        Profile* delta             = &profiles[3];
        Profile* velocity          = &profiles[4];
        Profile* after_loop        = &profiles[5];
    #endif
    START_PROFILE(complete_function);
    START_PROFILE(before_loop);

    MEMORY(18, complete_function, before_loop);
    double* translation_1     = get_displacement     (rvw1);
    double* velocity_1        = get_velocity         (rvw1);
    double* angular_velocity_1= get_angular_velocity (rvw1);

    double* translation_2     = get_displacement     (rvw2);
    double* velocity_2        = get_velocity         (rvw2);
    double* angular_velocity_2= get_angular_velocity (rvw2);

    /* ------------------------------------------------------------------ */
    /* ----------------------   scalar tweaks   -------------------------- */
    /* ------------------------------------------------------------------ */

    double invM     = 1.0 / M;           /* division → multiply   */
    double invR     = 1.0 / R;
    double C        = 5.0 * invM * invR * 0.5;   /* 5/(2MR) */

    double offset[3];
    subV3(translation_2, translation_1, offset);

    double offset_mag_sqrd = dotV3(offset, offset);
    double offset_inv_mag  = 1.0 / sqrt(offset_mag_sqrd);
    double forward[3];
    forward[0] = offset[0] * offset_inv_mag;
    forward[1] = offset[1] * offset_inv_mag;
    forward[2] = offset[2] * offset_inv_mag;

    double up[3] = {0.0, 0.0, 1.0};
    double right[3];
    crossV3(forward, up, right);

    /* ---------------- velocities to local frame ----------------------- */
    double local_velocity_x_1      = dotV3(velocity_1,  right);
    double local_velocity_y_1      = dotV3(velocity_1,  forward);
    double local_velocity_x_2      = dotV3(velocity_2,  right);
    double local_velocity_y_2      = dotV3(velocity_2,  forward);

    /* --------------- angular velocities to local frame ---------------- */
    double local_angular_velocity_x_1 = dotV3(angular_velocity_1, right);
    double local_angular_velocity_y_1 = dotV3(angular_velocity_1, forward);
    double local_angular_velocity_z_1 = dotV3(angular_velocity_1, up);

    double local_angular_velocity_x_2 = dotV3(angular_velocity_2, right);
    double local_angular_velocity_y_2 = dotV3(angular_velocity_2, forward);
    double local_angular_velocity_z_2 = dotV3(angular_velocity_2, up);

    /* ---------------- surface‑velocity helpers (use fma) -------------- */
    double surface_velocity_x_1 = fma(R, local_angular_velocity_y_1,  local_velocity_x_1);
    double surface_velocity_y_1 = fma(-R, local_angular_velocity_x_1, local_velocity_y_1);
    double surface_velocity_x_2 = fma(R, local_angular_velocity_y_2,  local_velocity_x_2);
    double surface_velocity_y_2 = fma(-R, local_angular_velocity_x_2, local_velocity_y_2);

    /*
    double surface_velocity_mag1_sq = surface_velocity_x_1*surface_velocity_x_1
                                    + surface_velocity_y_1*surface_velocity_y_1;
    double surface_velocity_mag2_sq = surface_velocity_x_2*surface_velocity_x_2
                                    + surface_velocity_y_2*surface_velocity_y_2;
    */

    /* ---------------------- contact point slip ------------------------ */
    double contact_point_velocity_x =  local_velocity_x_1 - local_velocity_x_2
                                     - R*(local_angular_velocity_z_1 + local_angular_velocity_z_2);
    double contact_point_velocity_z =  R*(local_angular_velocity_x_1 + local_angular_velocity_x_2);
    double contact_inv_mag          = 1.0 / sqrt(contact_point_velocity_x*contact_point_velocity_x +
                                                 contact_point_velocity_z*contact_point_velocity_z);
    double ball_ball_contact_point_magnitude =
        1.0 / contact_inv_mag;  /* keep original scalar around for profiling */

    /* --------------------------- impulse step ------------------------- */
    double velocity_diff_y = local_velocity_y_2 - local_velocity_y_1;

    if (unlikely(deltaP == 0.0f)) {
        deltaP = 0.5 * (1.0 + e_b) * M * fabs(velocity_diff_y) / (double)N;
    }

    /* bookkeeping (unchanged) */
    double total_work      = 0.0;
    double work_required   = INFINITY;
    double work_compression= 0.0;

    double deltaP_1 = deltaP, deltaP_2 = deltaP;
    double deltaP_x_1 = 0, deltaP_y_1 = 0, deltaP_x_2 = 0, deltaP_y_2 = 0;

    END_PROFILE(before_loop);
    while (velocity_diff_y < 0.0 || total_work < work_required)
    {
        /* -------------------- impulse calculation -------------------- */
        START_PROFILE(impulse);

        if (unlikely(ball_ball_contact_point_magnitude < 1e-16)) {
            BRANCH(0);
            deltaP_1 = deltaP_2 = 0.0;
            deltaP_x_1 = deltaP_y_1 = deltaP_x_2 = deltaP_y_2 = 0.0;
        } else {
            BRANCH(1);
            double inv_mag = contact_inv_mag;          /* already computed */
            deltaP_1 = -u_b * deltaP * contact_point_velocity_x * inv_mag;

            if (unlikely(fabs(contact_point_velocity_z) < 1e-16)) {
                BRANCH(2);
                deltaP_2 = deltaP_x_1 = deltaP_y_1 = deltaP_x_2 = deltaP_y_2 = 0.0;
            } else {
                BRANCH(3);
                deltaP_2 = -u_b * deltaP * contact_point_velocity_z * inv_mag;

                if (deltaP_2 > 0.0) {
                    BRANCH(4);
                    deltaP_x_1 = deltaP_y_1 = 0.0;

                    if (unlikely(surface_velocity_x_2 == 0.0 && surface_velocity_y_2 == 0)) {
                        BRANCH(5);
                        deltaP_x_2 = deltaP_y_2 = 0.0;
                    } else {
                        BRANCH(6);
                        double inv_sv2 = 1.0 / sqrt(surface_velocity_x_2 * surface_velocity_x_2 + surface_velocity_y_2 * surface_velocity_y_2);
                        deltaP_x_2 = -u_s2 * surface_velocity_x_2 * inv_sv2 * deltaP_2;
                        deltaP_y_2 = -u_s2 * surface_velocity_y_2 * inv_sv2 * deltaP_2;
                    }
                } else {
                    BRANCH(7);
                    deltaP_x_2 = deltaP_y_2 = 0.0;
                    if (unlikely(surface_velocity_x_1 == 0.0 && surface_velocity_y_1 == 0)) {
                        BRANCH(8);
                        deltaP_x_1 = deltaP_y_1 = 0.0;
                    } else {
                        BRANCH(9);
                        double inv_sv1 = 1.0 / sqrt(surface_velocity_x_1*surface_velocity_x_1 + surface_velocity_y_1*surface_velocity_y_1);
                        deltaP_x_1 = u_s1 * surface_velocity_x_1 * inv_sv1 * deltaP_2;
                        deltaP_y_1 = u_s1 * surface_velocity_y_1 * inv_sv1 * deltaP_2;
                    }
                }
            }
        }
        END_PROFILE(impulse);

        /* ------------------ update linear + angular vel -------------- */
        START_PROFILE(delta);

        double velocity_change_x_1 = (deltaP_1 + deltaP_x_1) * invM;
        double velocity_change_y_1 = (-deltaP   + deltaP_y_1) * invM;
        double velocity_change_x_2 = (-deltaP_1 + deltaP_x_2) * invM;
        double velocity_change_y_2 = ( deltaP   + deltaP_y_2) * invM;

        local_velocity_x_1 += velocity_change_x_1;
        local_velocity_y_1 += velocity_change_y_1;
        local_velocity_x_2 += velocity_change_x_2;
        local_velocity_y_2 += velocity_change_y_2;

        /* angular */
        local_angular_velocity_x_1 += C * (deltaP_2 + deltaP_y_1);
        local_angular_velocity_y_1 += C * (-deltaP_x_1);
        local_angular_velocity_z_1 += C * (-deltaP_1);

        local_angular_velocity_x_2 += C * (deltaP_2 + deltaP_y_2);
        local_angular_velocity_y_2 += C * (-deltaP_x_2);
        local_angular_velocity_z_2 += C * (-deltaP_1);

        END_PROFILE(delta);

        /* ----------------- recompute helpers for next iter ----------- */
        START_PROFILE(velocity);

        surface_velocity_x_1 = fma(R,  local_angular_velocity_y_1,  local_velocity_x_1);
        surface_velocity_y_1 = fma(-R, local_angular_velocity_x_1,  local_velocity_y_1);
        surface_velocity_x_2 = fma(R,  local_angular_velocity_y_2,  local_velocity_x_2);
        surface_velocity_y_2 = fma(-R, local_angular_velocity_x_2,  local_velocity_y_2);

        /*
        surface_velocity_mag1_sq = surface_velocity_x_1*surface_velocity_x_1
                                + surface_velocity_y_1*surface_velocity_y_1;
        surface_velocity_mag2_sq = surface_velocity_x_2*surface_velocity_x_2
                                + surface_velocity_y_2*surface_velocity_y_2;
        */

        contact_point_velocity_x = local_velocity_x_1 - local_velocity_x_2
                                 - (local_angular_velocity_z_1 + local_angular_velocity_z_2)*R;
        contact_point_velocity_z = (local_angular_velocity_x_1 + local_angular_velocity_x_2)*R;
        contact_inv_mag          = 1.0 / sqrt(contact_point_velocity_x*contact_point_velocity_x +
                                              contact_point_velocity_z*contact_point_velocity_z);
        ball_ball_contact_point_magnitude = 1.0 / contact_inv_mag;   /* for branch test */

        /* work / compression bookkeeping (unchanged) */
        double velocity_diff_y_prev = velocity_diff_y;
        velocity_diff_y   = local_velocity_y_2 - local_velocity_y_1;
        total_work       += 0.5 * deltaP * fabs(velocity_diff_y_prev + velocity_diff_y);

        if (work_compression == 0.0 && velocity_diff_y > 0.0) {
            work_compression = total_work;
            work_required    = (1.0 + e_b * e_b) * work_compression;
        }

        END_PROFILE(velocity);
    }
    /* ------------------------------------------------------------------ */
    /* ---------------------- epilogue – UNCHANGED ----------------------- */
    /* ------------------------------------------------------------------ */
    START_PROFILE(after_loop);
    for (int i = 0; i < 3; ++i) {
        rvw1_result[i+3] = local_velocity_x_1*right[i] + local_velocity_y_1*forward[i];
        rvw2_result[i+3] = local_velocity_x_2*right[i] + local_velocity_y_2*forward[i];

        if (i < 2) {
            rvw1_result[i+6] = local_angular_velocity_x_1*right[i] + local_angular_velocity_y_1*forward[i];
            rvw2_result[i+6] = local_angular_velocity_x_2*right[i] + local_angular_velocity_y_2*forward[i];
        } else {
            rvw1_result[i+6] = local_angular_velocity_z_1;
            rvw2_result[i+6] = local_angular_velocity_z_2;
        }
    }
    END_PROFILE(after_loop);
    END_PROFILE(complete_function);
}

DLL_EXPORT void approxsq_collide_balls(double* rvw1, double* rvw2, float R, float M, float u_s1, float u_s2, float u_b, float e_b, float deltaP, int N, double* rvw1_result, double* rvw2_result, Profile* profiles, Branch* branches)
{
    #ifdef PROFILE
        Profile* complete_function = &profiles[0];
        Profile* before_loop       = &profiles[1];
        Profile* impulse           = &profiles[2];
        Profile* delta             = &profiles[3];
        Profile* velocity          = &profiles[4];
        Profile* after_loop        = &profiles[5];
    #endif
    START_PROFILE(complete_function);
    START_PROFILE(before_loop);

    MEMORY(18, complete_function, before_loop);
    double* translation_1     = get_displacement     (rvw1);
    double* velocity_1        = get_velocity         (rvw1);
    double* angular_velocity_1= get_angular_velocity (rvw1);

    double* translation_2     = get_displacement     (rvw2);
    double* velocity_2        = get_velocity         (rvw2);
    double* angular_velocity_2= get_angular_velocity (rvw2);

    /* ------------------------------------------------------------------ */
    /* ----------------------   scalar tweaks   -------------------------- */
    /* ------------------------------------------------------------------ */

    double invM     = 1.0 / M;           /* division → multiply   */
    double invR     = 1.0 / R;
    double C        = 5.0 * invM * invR * 0.5;   /* 5/(2MR) */

    double offset[3];
    subV3(translation_2, translation_1, offset);

    double offset_mag_sqrd = dotV3(offset, offset);
    double offset_inv_mag  = 1.0 / sqrt(offset_mag_sqrd);
    double forward[3];
    forward[0] = offset[0] * offset_inv_mag;
    forward[1] = offset[1] * offset_inv_mag;
    forward[2] = offset[2] * offset_inv_mag;

    double up[3] = {0.0, 0.0, 1.0};
    double right[3];
    crossV3(forward, up, right);

    /* ---------------- velocities to local frame ----------------------- */
    double local_velocity_x_1      = dotV3(velocity_1,  right);
    double local_velocity_y_1      = dotV3(velocity_1,  forward);
    double local_velocity_x_2      = dotV3(velocity_2,  right);
    double local_velocity_y_2      = dotV3(velocity_2,  forward);

    /* --------------- angular velocities to local frame ---------------- */
    double local_angular_velocity_x_1 = dotV3(angular_velocity_1, right);
    double local_angular_velocity_y_1 = dotV3(angular_velocity_1, forward);
    double local_angular_velocity_z_1 = dotV3(angular_velocity_1, up);

    double local_angular_velocity_x_2 = dotV3(angular_velocity_2, right);
    double local_angular_velocity_y_2 = dotV3(angular_velocity_2, forward);
    double local_angular_velocity_z_2 = dotV3(angular_velocity_2, up);

    /* ---------------- surface‑velocity helpers (use fma) -------------- */
    double surface_velocity_x_1 = fma(R, local_angular_velocity_y_1,  local_velocity_x_1);
    double surface_velocity_y_1 = fma(-R, local_angular_velocity_x_1, local_velocity_y_1);
    double surface_velocity_x_2 = fma(R, local_angular_velocity_y_2,  local_velocity_x_2);
    double surface_velocity_y_2 = fma(-R, local_angular_velocity_x_2, local_velocity_y_2);

    /*
    double surface_velocity_mag1_sq = surface_velocity_x_1*surface_velocity_x_1
                                    + surface_velocity_y_1*surface_velocity_y_1;
    double surface_velocity_mag2_sq = surface_velocity_x_2*surface_velocity_x_2
                                    + surface_velocity_y_2*surface_velocity_y_2;
    */

    /* ---------------------- contact point slip ------------------------ */
    double contact_point_velocity_x =  local_velocity_x_1 - local_velocity_x_2
                                     - R*(local_angular_velocity_z_1 + local_angular_velocity_z_2);
    double contact_point_velocity_z =  R*(local_angular_velocity_x_1 + local_angular_velocity_x_2);
    double contact_inv_mag          = 1.0 / sqrt(contact_point_velocity_x*contact_point_velocity_x +
                                                 contact_point_velocity_z*contact_point_velocity_z);
    double ball_ball_contact_point_magnitude =
        1.0 / contact_inv_mag;  /* keep original scalar around for profiling */

    /* --------------------------- impulse step ------------------------- */
    double velocity_diff_y = local_velocity_y_2 - local_velocity_y_1;

    if (unlikely(deltaP == 0.0f)) {
        deltaP = 0.5 * (1.0 + e_b) * M * fabs(velocity_diff_y) / (double)N;
    }

    /* bookkeeping (unchanged) */
    double total_work      = 0.0;
    double work_required   = INFINITY;
    double work_compression= 0.0;

    double deltaP_1 = deltaP, deltaP_2 = deltaP;
    double deltaP_x_1 = 0, deltaP_y_1 = 0, deltaP_x_2 = 0, deltaP_y_2 = 0;

    END_PROFILE(before_loop);
    while (velocity_diff_y < 0.0 || total_work < work_required)
    {
        /* -------------------- impulse calculation -------------------- */
        START_PROFILE(impulse);

        if (unlikely(ball_ball_contact_point_magnitude < 1e-16)) {
            BRANCH(0);
            deltaP_1 = deltaP_2 = 0.0;
            deltaP_x_1 = deltaP_y_1 = deltaP_x_2 = deltaP_y_2 = 0.0;
        } else {
            BRANCH(1);
            double inv_mag = contact_inv_mag;          /* already computed */
            deltaP_1 = -u_b * deltaP * contact_point_velocity_x * inv_mag;

            if (unlikely(fabs(contact_point_velocity_z) < 1e-16)) {
                BRANCH(2);
                deltaP_2 = deltaP_x_1 = deltaP_y_1 = deltaP_x_2 = deltaP_y_2 = 0.0;
            } else {
                BRANCH(3);
                deltaP_2 = -u_b * deltaP * contact_point_velocity_z * inv_mag;

                if (deltaP_2 > 0.0) {
                    BRANCH(4);
                    deltaP_x_1 = deltaP_y_1 = 0.0;

                    if (unlikely(surface_velocity_x_2 == 0.0 && surface_velocity_y_2 == 0)) {
                        BRANCH(5);
                        deltaP_x_2 = deltaP_y_2 = 0.0;
                    } else {
                        BRANCH(6);
                        double inv_sv2 = 1.0 / sqrt(surface_velocity_x_2 * surface_velocity_x_2 + surface_velocity_y_2 * surface_velocity_y_2);
                        deltaP_x_2 = -u_s2 * surface_velocity_x_2 * inv_sv2 * deltaP_2;
                        deltaP_y_2 = -u_s2 * surface_velocity_y_2 * inv_sv2 * deltaP_2;
                    }
                } else {
                    BRANCH(7);
                    deltaP_x_2 = deltaP_y_2 = 0.0;
                    if (unlikely(surface_velocity_x_1 == 0.0 && surface_velocity_y_1 == 0)) {
                        BRANCH(8);
                        deltaP_x_1 = deltaP_y_1 = 0.0;
                    } else {
                        BRANCH(9);
                        double inv_sv1 = 1.0 / sqrt(surface_velocity_x_1*surface_velocity_x_1 + surface_velocity_y_1*surface_velocity_y_1);
                        deltaP_x_1 = u_s1 * surface_velocity_x_1 * inv_sv1 * deltaP_2;
                        deltaP_y_1 = u_s1 * surface_velocity_y_1 * inv_sv1 * deltaP_2;
                    }
                }
            }
        }
        END_PROFILE(impulse);

        /* ------------------ update linear + angular vel -------------- */
        START_PROFILE(delta);

        double velocity_change_x_1 = (deltaP_1 + deltaP_x_1) * invM;
        double velocity_change_y_1 = (-deltaP   + deltaP_y_1) * invM;
        double velocity_change_x_2 = (-deltaP_1 + deltaP_x_2) * invM;
        double velocity_change_y_2 = ( deltaP   + deltaP_y_2) * invM;

        local_velocity_x_1 += velocity_change_x_1;
        local_velocity_y_1 += velocity_change_y_1;
        local_velocity_x_2 += velocity_change_x_2;
        local_velocity_y_2 += velocity_change_y_2;

        /* angular */
        local_angular_velocity_x_1 += C * (deltaP_2 + deltaP_y_1);
        local_angular_velocity_y_1 += C * (-deltaP_x_1);
        local_angular_velocity_z_1 += C * (-deltaP_1);

        local_angular_velocity_x_2 += C * (deltaP_2 + deltaP_y_2);
        local_angular_velocity_y_2 += C * (-deltaP_x_2);
        local_angular_velocity_z_2 += C * (-deltaP_1);

        END_PROFILE(delta);

        /* ----------------- recompute helpers for next iter ----------- */
        START_PROFILE(velocity);

        surface_velocity_x_1 = fma(R,  local_angular_velocity_y_1,  local_velocity_x_1);
        surface_velocity_y_1 = fma(-R, local_angular_velocity_x_1,  local_velocity_y_1);
        surface_velocity_x_2 = fma(R,  local_angular_velocity_y_2,  local_velocity_x_2);
        surface_velocity_y_2 = fma(-R, local_angular_velocity_x_2,  local_velocity_y_2);

        /*
        surface_velocity_mag1_sq = surface_velocity_x_1*surface_velocity_x_1
                                + surface_velocity_y_1*surface_velocity_y_1;
        surface_velocity_mag2_sq = surface_velocity_x_2*surface_velocity_x_2
                                + surface_velocity_y_2*surface_velocity_y_2;
        */

        contact_point_velocity_x = local_velocity_x_1 - local_velocity_x_2
                                 - (local_angular_velocity_z_1 + local_angular_velocity_z_2)*R;
        contact_point_velocity_z = (local_angular_velocity_x_1 + local_angular_velocity_x_2)*R;
        contact_inv_mag          *= 0.5 * (3.0 - (contact_point_velocity_x * contact_point_velocity_x +
                               contact_point_velocity_z * contact_point_velocity_z) * contact_inv_mag * contact_inv_mag);
        ball_ball_contact_point_magnitude = 1.0 / contact_inv_mag;   /* for branch test */

        /* work / compression bookkeeping (unchanged) */
        double velocity_diff_y_prev = velocity_diff_y;
        velocity_diff_y   = local_velocity_y_2 - local_velocity_y_1;
        total_work       += 0.5 * deltaP * fabs(velocity_diff_y_prev + velocity_diff_y);

        if (work_compression == 0.0 && velocity_diff_y > 0.0) {
            work_compression = total_work;
            work_required    = (1.0 + e_b * e_b) * work_compression;
        }

        END_PROFILE(velocity);
    }
    /* ------------------------------------------------------------------ */
    /* ---------------------- epilogue – UNCHANGED ----------------------- */
    /* ------------------------------------------------------------------ */
    START_PROFILE(after_loop);
    for (int i = 0; i < 3; ++i) {
        rvw1_result[i+3] = local_velocity_x_1*right[i] + local_velocity_y_1*forward[i];
        rvw2_result[i+3] = local_velocity_x_2*right[i] + local_velocity_y_2*forward[i];

        if (i < 2) {
            rvw1_result[i+6] = local_angular_velocity_x_1*right[i] + local_angular_velocity_y_1*forward[i];
            rvw2_result[i+6] = local_angular_velocity_x_2*right[i] + local_angular_velocity_y_2*forward[i];
        } else {
            rvw1_result[i+6] = local_angular_velocity_z_1;
            rvw2_result[i+6] = local_angular_velocity_z_2;
        }
    }
    END_PROFILE(after_loop);
    END_PROFILE(complete_function);
}

DLL_EXPORT void approx_symmetry(double* rvw1, double* rvw2, float R, float M, float u_s1, float u_s2, float u_b, float e_b, float deltaP, int N, double* rvw1_result, double* rvw2_result, Profile* profiles, Branch* branches)
{
    #ifdef PROFILE
        Profile* complete_function = &profiles[0];
        Profile* before_loop       = &profiles[1];
        Profile* impulse           = &profiles[2];
        Profile* delta             = &profiles[3];
        Profile* velocity          = &profiles[4];
        Profile* after_loop        = &profiles[5];
    #endif
    START_PROFILE(complete_function);
    START_PROFILE(before_loop);

    MEMORY(18, complete_function, before_loop);
    double* translation_1     = get_displacement     (rvw1);
    double* velocity_1        = get_velocity         (rvw1);
    double* angular_velocity_1= get_angular_velocity (rvw1);

    double* translation_2     = get_displacement     (rvw2);
    double* velocity_2        = get_velocity         (rvw2);
    double* angular_velocity_2= get_angular_velocity (rvw2);

    /* ------------------------------------------------------------------ */
    /* ----------------------   scalar tweaks   -------------------------- */
    /* ------------------------------------------------------------------ */

    double invM     = 1.0 / M;           /* division → multiply   */
    double invR     = 1.0 / R;
    double C        = 5.0 * invM * invR * 0.5;   /* 5/(2MR) */

    double offset[3];
    subV3(translation_2, translation_1, offset);

    double offset_mag_sqrd = dotV3(offset, offset);
    double offset_inv_mag  = 1.0 / sqrt(offset_mag_sqrd);
    double forward[3];
    forward[0] = offset[0] * offset_inv_mag;
    forward[1] = offset[1] * offset_inv_mag;
    forward[2] = offset[2] * offset_inv_mag;

    double up[3] = {0.0, 0.0, 1.0};
    double right[3];
    crossV3(forward, up, right);

    /* ---------------- velocities to local frame ----------------------- */
    double local_velocity_x_1      = dotV3(velocity_1,  right);
    double local_velocity_y_1      = dotV3(velocity_1,  forward);
    double local_velocity_x_2      = dotV3(velocity_2,  right);
    double local_velocity_y_2      = dotV3(velocity_2,  forward);

    /* --------------- angular velocities to local frame ---------------- */
    double local_angular_velocity_x_1 = dotV3(angular_velocity_1, right);
    double local_angular_velocity_y_1 = dotV3(angular_velocity_1, forward);
    double local_angular_velocity_z_1 = dotV3(angular_velocity_1, up);

    double local_angular_velocity_x_2 = dotV3(angular_velocity_2, right);
    double local_angular_velocity_y_2 = dotV3(angular_velocity_2, forward);
    double local_angular_velocity_z_2 = dotV3(angular_velocity_2, up);

    /* ---------------- surface‑velocity helpers (use fma) -------------- */
    double surface_velocity_x_1 = fma(R, local_angular_velocity_y_1,  local_velocity_x_1);
    double surface_velocity_y_1 = fma(-R, local_angular_velocity_x_1, local_velocity_y_1);
    double surface_velocity_x_2 = fma(R, local_angular_velocity_y_2,  local_velocity_x_2);
    double surface_velocity_y_2 = fma(-R, local_angular_velocity_x_2, local_velocity_y_2);

    /*
    double surface_velocity_mag1_sq = surface_velocity_x_1*surface_velocity_x_1
                                    + surface_velocity_y_1*surface_velocity_y_1;
    double surface_velocity_mag2_sq = surface_velocity_x_2*surface_velocity_x_2
                                    + surface_velocity_y_2*surface_velocity_y_2;
    */

    /* ---------------------- contact point slip ------------------------ */
    double contact_point_velocity_x =  local_velocity_x_1 - local_velocity_x_2
                                     - R*(local_angular_velocity_z_1 + local_angular_velocity_z_2);
    double contact_point_velocity_z =  R*(local_angular_velocity_x_1 + local_angular_velocity_x_2);
    double contact_inv_mag          = 1.0 / sqrt(contact_point_velocity_x*contact_point_velocity_x +
                                                 contact_point_velocity_z*contact_point_velocity_z);
    double ball_ball_contact_point_magnitude =
        1.0 / contact_inv_mag;  /* keep original scalar around for profiling */

    /* --------------------------- impulse step ------------------------- */
    double velocity_diff_y = local_velocity_y_2 - local_velocity_y_1;

    if (unlikely(deltaP == 0.0f)) {
        deltaP = 0.5 * (1.0 + e_b) * M * fabs(velocity_diff_y) / (double)N;
    }

    /* bookkeeping (unchanged) */
    double total_work      = 0.0;
    double work_required   = INFINITY;
    double work_compression= 0.0;

    double deltaP_1 = deltaP, deltaP_2 = deltaP;
    double deltaP_x_1 = 0, deltaP_y_1 = 0, deltaP_x_2 = 0, deltaP_y_2 = 0;

    END_PROFILE(before_loop);
    while (velocity_diff_y < 0.0 || total_work < work_required)
    {
        /* -------------------- impulse calculation -------------------- */
        START_PROFILE(impulse);

        if (unlikely(ball_ball_contact_point_magnitude < 1e-16)) {
            BRANCH(0);
            deltaP_1 = deltaP_2 = 0.0;
            deltaP_x_1 = deltaP_y_1 = deltaP_x_2 = deltaP_y_2 = 0.0;
        } else {
            BRANCH(1);
            double inv_mag = contact_inv_mag;          /* already computed */
            deltaP_1 = -u_b * deltaP * contact_point_velocity_x * inv_mag;

            if (unlikely(fabs(contact_point_velocity_z) < 1e-16)) {
                BRANCH(2);
                deltaP_2 = deltaP_x_1 = deltaP_y_1 = deltaP_x_2 = deltaP_y_2 = 0.0;
            } else {
                BRANCH(3);
                deltaP_2 = -u_b * deltaP * contact_point_velocity_z * inv_mag;

                if (deltaP_2 > 0.0) {
                    BRANCH(4);
                    deltaP_x_1 = deltaP_y_1 = 0.0;

                    if (unlikely(surface_velocity_x_2 == 0.0 && surface_velocity_y_2 == 0)) {
                        BRANCH(5);
                        deltaP_x_2 = deltaP_y_2 = 0.0;
                    } else {
                        BRANCH(6);
                        double inv_sv2 = 1.0 / sqrt(surface_velocity_x_2 * surface_velocity_x_2 + surface_velocity_y_2 * surface_velocity_y_2);
                        deltaP_x_2 = -u_s2 * surface_velocity_x_2 * inv_sv2 * deltaP_2;
                        deltaP_y_2 = -u_s2 * surface_velocity_y_2 * inv_sv2 * deltaP_2;
                    }
                } else {
                    BRANCH(7);
                    deltaP_x_2 = deltaP_y_2 = 0.0;
                    if (unlikely(surface_velocity_x_1 == 0.0 && surface_velocity_y_1 == 0)) {
                        BRANCH(8);
                        deltaP_x_1 = deltaP_y_1 = 0.0;
                    } else {
                        BRANCH(9);
                        double inv_sv1 = 1.0 / sqrt(surface_velocity_x_1*surface_velocity_x_1 + surface_velocity_y_1*surface_velocity_y_1);
                        deltaP_x_1 = u_s1 * surface_velocity_x_1 * inv_sv1 * deltaP_2;
                        deltaP_y_1 = u_s1 * surface_velocity_y_1 * inv_sv1 * deltaP_2;
                    }
                }
            }
        }
        END_PROFILE(impulse);

        /* ------------------ update linear + angular vel -------------- */
        START_PROFILE(delta);

        double velocity_change_x_1 = (deltaP_1 + deltaP_x_1) * invM;
        double velocity_change_y_1 = (-deltaP   + deltaP_y_1) * invM;
        double velocity_change_x_2 = (-deltaP_1 + deltaP_x_2) * invM;
        double velocity_change_y_2 = ( deltaP   + deltaP_y_2) * invM;

        local_velocity_x_1 += velocity_change_x_1;
        local_velocity_y_1 += velocity_change_y_1;
        local_velocity_x_2 += velocity_change_x_2;
        local_velocity_y_2 += velocity_change_y_2;

        /* angular */
        local_angular_velocity_x_1 += C * (deltaP_2 + deltaP_y_1);
        local_angular_velocity_y_1 += C * (-deltaP_x_1);
        local_angular_velocity_z_1 += C * (-deltaP_1);

        local_angular_velocity_x_2 += C * (deltaP_2 + deltaP_y_2);
        local_angular_velocity_y_2 += C * (-deltaP_x_2);
        local_angular_velocity_z_2 += C * (-deltaP_1);

        END_PROFILE(delta);

        /* ----------------- recompute helpers for next iter ----------- */
        START_PROFILE(velocity);

        surface_velocity_x_1 = fma(R,  local_angular_velocity_y_1,  local_velocity_x_1);
        surface_velocity_y_1 = fma(-R, local_angular_velocity_x_1,  local_velocity_y_1);
        surface_velocity_x_2 = fma(R,  local_angular_velocity_y_2,  local_velocity_x_2);
        surface_velocity_y_2 = fma(-R, local_angular_velocity_x_2,  local_velocity_y_2);

        /*
        surface_velocity_mag1_sq = surface_velocity_x_1*surface_velocity_x_1
                                + surface_velocity_y_1*surface_velocity_y_1;
        surface_velocity_mag2_sq = surface_velocity_x_2*surface_velocity_x_2
                                + surface_velocity_y_2*surface_velocity_y_2;
        */

        contact_point_velocity_x = local_velocity_x_1 - local_velocity_x_2
                                 - (local_angular_velocity_z_1 + local_angular_velocity_z_2)*R;
        contact_point_velocity_z = (local_angular_velocity_x_1 + local_angular_velocity_x_2)*R;
        contact_inv_mag          *= 0.5 * (3.0 - (contact_point_velocity_x * contact_point_velocity_x +
                               contact_point_velocity_z * contact_point_velocity_z) * contact_inv_mag * contact_inv_mag);
        ball_ball_contact_point_magnitude = 1.0 / contact_inv_mag;   /* for branch test */

        /* work / compression bookkeeping (unchanged) */
        double velocity_diff_y_prev = velocity_diff_y;
        velocity_diff_y   = local_velocity_y_2 - local_velocity_y_1;
        total_work       += 0.5 * deltaP * fabs(velocity_diff_y_prev + velocity_diff_y);

        if (work_compression == 0.0 && velocity_diff_y > 0.0) {
            work_compression = total_work;
            work_required    = (1.0 + e_b * e_b) * work_compression;
        }

        END_PROFILE(velocity);
    }
    /* ------------------------------------------------------------------ */
    /* ---------------------- epilogue – UNCHANGED ----------------------- */
    /* ------------------------------------------------------------------ */
    START_PROFILE(after_loop);
    for (int i = 0; i < 3; ++i) {
        rvw1_result[i+3] = local_velocity_x_1*right[i] + local_velocity_y_1*forward[i];
        rvw2_result[i+3] = local_velocity_x_2*right[i] + local_velocity_y_2*forward[i];

        if (i < 2) {
            rvw1_result[i+6] = local_angular_velocity_x_1*right[i] + local_angular_velocity_y_1*forward[i];
            rvw2_result[i+6] = local_angular_velocity_x_2*right[i] + local_angular_velocity_y_2*forward[i];
        } else {
            rvw1_result[i+6] = local_angular_velocity_z_1;
            rvw2_result[i+6] = local_angular_velocity_z_2;
        }
    }
    END_PROFILE(after_loop);
    END_PROFILE(complete_function);
}

DLL_EXPORT void simple_precompute_cb(double* rvw1, double* rvw2, float Rf, float Mf, float u_s1f, float u_s2f, float u_bf, float e_bf, float deltaPf, int N, double* rvw1_result, double* rvw2_result, Profile* profiles, Branch* branches) {

    #ifdef PROFILE
        Profile* complete_function = &profiles[0];
        Profile* before_loop = &profiles[1];
        Profile* impulse = &profiles[2];
        Profile* delta = &profiles[3];
        Profile* velocity = &profiles[4];
        Profile* after_loop = &profiles[5];
    #endif

    START_PROFILE(complete_function);
    START_PROFILE(before_loop);

    FLOPS(3, 1, 1, 0, complete_function, before_loop);
    double R = (double)Rf;
    double M = (double)Mf;
    double u_s1 = (double)u_s1f;
    double u_s2 = -(double)u_s2f;
    double u_b = -(double)u_bf;
    double e_b = (double)e_bf;
    double e_b_sqrt_plus_1 = e_b * e_b + 1;
    double deltaP = (double)deltaPf;

    double M_rep = 1.0f / M;

    // Altough the memory is not really loaded here, assuming its only compulsary misses
    MEMORY(18, complete_function, before_loop);
    double* translation_1 = get_displacement(rvw1);
    double* velocity_1 = get_velocity(rvw1);
    double* angular_velocity_1 = get_angular_velocity(rvw1);

    double* translation_2 = get_displacement(rvw2);
    double* velocity_2 = get_velocity(rvw2);
    double* angular_velocity_2 = get_angular_velocity(rvw2);

    FLOPS(3, 0, 0, 0, complete_function, before_loop);
    double offset[3];
    subV3(translation_2, translation_1, offset);

    FLOPS(2, 3, 0, 0, complete_function, before_loop);
    double offset_mag_sqrd = dotV3(offset, offset);

    FLOPS(0, 0, 0, 1, complete_function, before_loop);
    double offset_mag = sqrt(offset_mag_sqrd);

    double forward[3]; // Forward from ball 1 to ball 2, normalized

    FLOPS(0, 0, 3, 0, complete_function, before_loop);
    divV3(offset, offset_mag, forward);

    double up[3] = {0, 0, 1}; // Probably up axis?

    double right[3]; // Axis orthogonal to Z and forward

    FLOPS(3, 6, 0, 0, complete_function, before_loop);
    crossV3(forward, up, right);

    // From here on, it is assumed that the x axis is the right axis and y axis is the forward axis
    // Transform velocities to local frame

    FLOPS(2 * 4, 3 * 4, 0, 0, complete_function, before_loop);
    double local_velocity_x_1 = dotV3(velocity_1, right);
    double local_velocity_y_1 = dotV3(velocity_1, forward);
    double local_velocity_x_2 = dotV3(velocity_2, right);
    double local_velocity_y_2 = dotV3(velocity_2, forward);

    // Transform angular velocities into local frame

    FLOPS(2*6, 3*6, 0, 0, complete_function, before_loop);
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
    FLOPS(4, 4, 0, 0, complete_function, before_loop);
    double surface_velocity_x_1 = local_velocity_x_1 + R * local_angular_velocity_y_1;
    double surface_velocity_y_1 = local_velocity_y_1 - R * local_angular_velocity_x_1;
    double surface_velocity_x_2 = local_velocity_x_2 + R * local_angular_velocity_y_2;
    double surface_velocity_y_2 = local_velocity_y_2 - R * local_angular_velocity_x_2;

    FLOPS(2, 4, 0, 2, complete_function, before_loop);
    double surface_velocity_magnitude_1 = sqrt(surface_velocity_x_1 * surface_velocity_x_1 + surface_velocity_y_1 * surface_velocity_y_1);
    double surface_velocity_magnitude_2 = sqrt(surface_velocity_x_2 * surface_velocity_x_2 + surface_velocity_y_2 * surface_velocity_y_2);

    // Relative surface velocity in the x-direction at the point where the two balls are in contact.
    // ball-ball slip
    FLOPS(5, 4, 0, 1, complete_function, before_loop);
    double contact_point_velocity_x = local_velocity_x_1 - local_velocity_x_2 - R * (local_angular_velocity_z_1 + local_angular_velocity_z_2);
    double contact_point_velocity_z = R * (local_angular_velocity_x_1 + local_angular_velocity_x_2);
    double ball_ball_contact_point_magnitude = sqrt(contact_point_velocity_x * contact_point_velocity_x + contact_point_velocity_z * contact_point_velocity_z);
    //printf("\nC Contact Point Slide, Spin:\n");
    //printf("  Contact Point: u_ijC_xz_mag= %.6f\n", ball_ball_contact_point_magnitude);

    // Main collision loop
    FLOPS(1, 0, 0, 0, complete_function, before_loop);
    double velocity_diff_y = local_velocity_y_2 - local_velocity_y_1;

    // deltaP is most likely always 0?
    // ΔP represents the Impulse during a time of Δt
    if (deltaP == 0) {
        FLOPS(0, 4, 1, 0, complete_function, before_loop);
        deltaP = 0.5 * (1.0 + e_b) * M * fabs(velocity_diff_y) / (double)(N);
    }

    FLOPS(0, 2, 1, 0, complete_function, before_loop);
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

    while (velocity_diff_y < 0 || total_work < work_required) {

        START_PROFILE(impulse);

        // Impulse Calculation
        if (ball_ball_contact_point_magnitude < 1e-16) {
            BRANCH(0);
            deltaP_1 = 0;
            deltaP_2 = 0;
            deltaP_x_1 = 0;
            deltaP_y_1 = 0;
            deltaP_x_2 = 0;
            deltaP_y_2 = 0;
        } else {
            BRANCH(1);
            FLOPS(0, 2, 1, 0, complete_function, impulse);
            deltaP_1 = u_b * deltaP * contact_point_velocity_x / ball_ball_contact_point_magnitude;
            if(fabs(contact_point_velocity_z) < 1e-16) {
                BRANCH(2);
                deltaP_2 = 0;
                deltaP_x_1 = 0;
                deltaP_y_1 = 0;
                deltaP_x_2 = 0;
                deltaP_y_2 = 0;
            } else {
                BRANCH(3);
                FLOPS(0, 2, 1, 0, complete_function, impulse);
                deltaP_2 = u_b * deltaP * contact_point_velocity_z / ball_ball_contact_point_magnitude;

                if(deltaP_2 > 0) {
                    BRANCH(4);
                    deltaP_x_1 = 0;
                    deltaP_y_1 = 0;

                    // TODO: probably best to check for some tolerance
                    if(surface_velocity_magnitude_2 == 0.0) {
                        BRANCH(5);
                        deltaP_x_2 = 0;
                        deltaP_y_2 = 0;
                    } else {
                        BRANCH(6);
                        FLOPS(0, 4, 2, 0, complete_function, impulse);
                        deltaP_x_2 = u_s2 * (surface_velocity_x_2 / surface_velocity_magnitude_2) * deltaP_2;
                        deltaP_y_2 = u_s2 * (surface_velocity_y_2 / surface_velocity_magnitude_2) * deltaP_2;
                    }
                } else {
                    BRANCH(7);
                    deltaP_x_2 = 0;
                    deltaP_y_2 = 0;
                    if(surface_velocity_magnitude_1 == 0.0) {
                        BRANCH(8);
                        deltaP_x_1 = 0;
                        deltaP_y_1 = 0;
                    } else {
                        BRANCH(9);
                        FLOPS(0, 4, 2, 0, complete_function, impulse);
                        deltaP_x_1 = u_s1 * (surface_velocity_x_1 / surface_velocity_magnitude_1) * deltaP_2;
                        deltaP_y_1 = u_s1 * (surface_velocity_y_1 / surface_velocity_magnitude_1) * deltaP_2;
                    }
                }

            }
        }

        END_PROFILE(impulse);
        START_PROFILE(delta);

        FLOPS(2, 3, 4, 0, complete_function, delta);
        // Velocity changes
        double velocity_change_x_1 = (deltaP_1 + deltaP_x_1) * M_rep;
        double velocity_change_y_1 = (-deltaP + deltaP_y_1) * M_rep;
        double velocity_change_x_2 = (-deltaP_1 + deltaP_x_2) * M_rep;
        double velocity_change_y_2 = (deltaP + deltaP_y_2) * M_rep;

        FLOPS(4, 0, 0, 0, complete_function, delta);
        // Update velocities
        local_velocity_x_1 += velocity_change_x_1;
        local_velocity_y_1 += velocity_change_y_1;
        local_velocity_x_2 += velocity_change_x_2;
        local_velocity_y_2 += velocity_change_y_2;

        FLOPS(6, 6, 0, 0, complete_function, delta);
        // Angular velocity changes
        double delta_angular_velocity_x_1 = C * (deltaP_2 + deltaP_y_1);
        double delta_angular_velocity_y_1 = C * (-deltaP_x_1);
        double delta_angular_velocity_z_1 = C * (-deltaP_1);
        double delta_angular_velocity_x_2 = C * (deltaP_2 + deltaP_y_2);
        double delta_angular_velocity_y_2 = C * (-deltaP_x_2);
        double delta_angular_velocity_z_2 = C * (-deltaP_1);

        FLOPS(6, 0, 0, 0, complete_function, delta);
        // Update Angular Velocities
        local_angular_velocity_x_1 += delta_angular_velocity_x_1;
        local_angular_velocity_y_1 += delta_angular_velocity_y_1;
        local_angular_velocity_z_1 += delta_angular_velocity_z_1;

        local_angular_velocity_x_2 += delta_angular_velocity_x_2;
        local_angular_velocity_y_2 += delta_angular_velocity_y_2;
        local_angular_velocity_z_2 += delta_angular_velocity_z_2;

        END_PROFILE(delta);
        START_PROFILE(velocity);

        FLOPS(4, 4, 0, 0, complete_function, velocity);
        // update ball-table slips
        surface_velocity_x_1 = local_velocity_x_1 + R * local_angular_velocity_y_1;
        surface_velocity_y_1 = local_velocity_y_1 - R * local_angular_velocity_x_1;
        surface_velocity_x_2 = local_velocity_x_2 + R * local_angular_velocity_y_2;
        surface_velocity_y_2 = local_velocity_y_2 - R * local_angular_velocity_x_2;

        FLOPS(2, 4, 0, 2, complete_function, velocity);
        surface_velocity_magnitude_1 = sqrt(surface_velocity_x_1 * surface_velocity_x_1 + surface_velocity_y_1 * surface_velocity_y_1);
        surface_velocity_magnitude_2 = sqrt(surface_velocity_x_2 * surface_velocity_x_2 + surface_velocity_y_2 * surface_velocity_y_2);

        FLOPS(5, 4, 0, 1, complete_function, velocity);
        // update ball-ball slip:
        contact_point_velocity_x = local_velocity_x_1 - local_velocity_x_2 - R * (local_angular_velocity_z_1 + local_angular_velocity_z_2);
        contact_point_velocity_z = R * (local_angular_velocity_x_1 + local_angular_velocity_x_2);
        ball_ball_contact_point_magnitude = sqrt(contact_point_velocity_x * contact_point_velocity_x + contact_point_velocity_z * contact_point_velocity_z);

        FLOPS(3, 2, 0, 0, complete_function, velocity);
        // Update work and check compression phase
        double velocity_diff_y_temp = velocity_diff_y;
        velocity_diff_y = local_velocity_y_2 - local_velocity_y_1;
        total_work += 0.5 * deltaP * fabs(velocity_diff_y_temp + velocity_diff_y);

        if (work_compression == 0 && velocity_diff_y > 0) {
            work_compression = total_work;
            FLOPS(0, 1, 0, 0, complete_function, velocity);
            work_required = e_b_sqrt_plus_1 * work_compression;
        }

        END_PROFILE(velocity);
    }

    START_PROFILE(after_loop);

    // Transform back to global coordinates
    for (int i = 0; i < 3; i++) {

        MEMORY(4, complete_function, after_loop);
        FLOPS(2, 4, 0, 0, complete_function, after_loop);
        rvw1_result[i + 3] = local_velocity_x_1 * right[i] + local_velocity_y_1 * forward[i];
        rvw2_result[i + 3] = local_velocity_x_2 * right[i] + local_velocity_y_2 * forward[i];

        if(i < 2) {
            FLOPS(2, 4, 0, 0, complete_function, after_loop);
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

DLL_EXPORT void less_sqrt_collide_balls(double* rvw1, double* rvw2, float R, float M, float u_s1, float u_s2, float u_b, float e_b, float deltaP, int N, double* rvw1_result, double* rvw2_result, Profile* profiles, Branch* branches) {

    #ifdef PROFILE
        Profile* complete_function = &profiles[0];
        Profile* before_loop = &profiles[1];
        Profile* impulse = &profiles[2];
        Profile* delta = &profiles[3];
        Profile* velocity = &profiles[4];
        Profile* after_loop = &profiles[5];
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

    FLOPS(3, 0, 0, 0, complete_function, before_loop);
    double offset[3];
    subV3(translation_2, translation_1, offset);

    FLOPS(2, 3, 0, 0, complete_function, before_loop);
    double offset_mag_sqrd = dotV3(offset, offset);

    FLOPS(0, 0, 0, 1, complete_function, before_loop);
    double offset_mag = sqrt(offset_mag_sqrd);

    double forward[3]; // Forward from ball 1 to ball 2, normalized

    FLOPS(0, 0, 3, 0, complete_function, before_loop);
    divV3(offset, offset_mag, forward);

    double up[3] = {0, 0, 1}; // Probably up axis?

    double right[3]; // Axis orthogonal to Z and forward

    FLOPS(3, 6, 0, 0, complete_function, before_loop);
    crossV3(forward, up, right);

    // From here on, it is assumed that the x axis is the right axis and y axis is the forward axis
    // Transform velocities to local frame

    FLOPS(2 * 4, 3 * 4, 0, 0, complete_function, before_loop);
    double local_velocity_x_1 = dotV3(velocity_1, right);
    double local_velocity_y_1 = dotV3(velocity_1, forward);
    double local_velocity_x_2 = dotV3(velocity_2, right);
    double local_velocity_y_2 = dotV3(velocity_2, forward);

    // Transform angular velocities into local frame

    FLOPS(2*6, 3*6, 0, 0, complete_function, before_loop);
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
    FLOPS(4, 4, 0, 0, complete_function, before_loop);
    double surface_velocity_x_1 = local_velocity_x_1 + R * local_angular_velocity_y_1;
    double surface_velocity_y_1 = local_velocity_y_1 - R * local_angular_velocity_x_1;
    double surface_velocity_x_2 = local_velocity_x_2 + R * local_angular_velocity_y_2;
    double surface_velocity_y_2 = local_velocity_y_2 - R * local_angular_velocity_x_2;

    FLOPS(2, 4, 0, 0, complete_function, before_loop);
    double surface_velocity_magnitude_1_sqrd = (surface_velocity_x_1 * surface_velocity_x_1 + surface_velocity_y_1 * surface_velocity_y_1);
    double surface_velocity_magnitude_2_sqrd = (surface_velocity_x_2 * surface_velocity_x_2 + surface_velocity_y_2 * surface_velocity_y_2);

    // Relative surface velocity in the x-direction at the point where the two balls are in contact.
    // ball-ball slip
    FLOPS(5, 4, 0, 1, complete_function, before_loop);
    double contact_point_velocity_x = local_velocity_x_1 - local_velocity_x_2 - R * (local_angular_velocity_z_1 + local_angular_velocity_z_2);
    double contact_point_velocity_z = R * (local_angular_velocity_x_1 + local_angular_velocity_x_2);
    double ball_ball_contact_point_magnitude = sqrt(contact_point_velocity_x * contact_point_velocity_x + contact_point_velocity_z * contact_point_velocity_z);
    //printf("\nC Contact Point Slide, Spin:\n");
    //printf("  Contact Point: u_ijC_xz_mag= %.6f\n", ball_ball_contact_point_magnitude);

    // Main collision loop
    FLOPS(1, 0, 0, 0, complete_function, before_loop);
    double velocity_diff_y = local_velocity_y_2 - local_velocity_y_1;

    // deltaP is most likely always 0?
    // ΔP represents the Impulse during a time of Δt
    if (deltaP == 0) {
        FLOPS(0, 4, 1, 0, complete_function, before_loop);
        deltaP = 0.5 * (1.0 + e_b) * M * fabs(velocity_diff_y) / (double)(N);
    }

    FLOPS(0, 2, 1, 0, complete_function, before_loop);
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

    while (velocity_diff_y < 0 || total_work < work_required) {

        START_PROFILE(impulse);

        // Impulse Calculation
        if (ball_ball_contact_point_magnitude < 1e-16) {
            BRANCH(0);
            deltaP_1 = 0;
            deltaP_2 = 0;
            deltaP_x_1 = 0;
            deltaP_y_1 = 0;
            deltaP_x_2 = 0;
            deltaP_y_2 = 0;
        } else {
            BRANCH(1);
            FLOPS(1, 2, 1, 0, complete_function, impulse);
            deltaP_1 = -u_b * deltaP * contact_point_velocity_x / ball_ball_contact_point_magnitude;
            if(fabs(contact_point_velocity_z) < 1e-16) {
                BRANCH(2);
                deltaP_2 = 0;
                deltaP_x_1 = 0;
                deltaP_y_1 = 0;
                deltaP_x_2 = 0;
                deltaP_y_2 = 0;
            } else {
                BRANCH(3);
                FLOPS(1, 2, 1, 0, complete_function, impulse);
                deltaP_2 = -u_b * deltaP * contact_point_velocity_z / ball_ball_contact_point_magnitude;

                if(deltaP_2 > 0) {
                    BRANCH(4);
                    deltaP_x_1 = 0;
                    deltaP_y_1 = 0;

                    // TODO: probably best to check for some tolerance
                    if(surface_velocity_magnitude_2_sqrd == 0.0) {
                        BRANCH(5);
                        deltaP_x_2 = 0;
                        deltaP_y_2 = 0;
                    } else {
                        BRANCH(6);
                        FLOPS(2, 4, 2, 1, complete_function, impulse);
                        double surface_velocity_magnitude_2 = sqrt(surface_velocity_magnitude_2_sqrd);
                        deltaP_x_2 = -u_s2 * (surface_velocity_x_2 / surface_velocity_magnitude_2) * deltaP_2;
                        deltaP_y_2 = -u_s2 * (surface_velocity_y_2 / surface_velocity_magnitude_2) * deltaP_2;
                    }
                } else {
                    BRANCH(7);
                    deltaP_x_2 = 0;
                    deltaP_y_2 = 0;
                    if(surface_velocity_magnitude_1_sqrd == 0.0) {
                        BRANCH(8);
                        deltaP_x_1 = 0;
                        deltaP_y_1 = 0;
                    } else {
                        BRANCH(9);
                        FLOPS(0, 4, 2, 1, complete_function, impulse);
                        double surface_velocity_magnitude_1 = sqrt(surface_velocity_magnitude_1_sqrd);
                        deltaP_x_1 = u_s1 * (surface_velocity_x_1 / surface_velocity_magnitude_1) * deltaP_2;
                        deltaP_y_1 = u_s1 * (surface_velocity_y_1 / surface_velocity_magnitude_1) * deltaP_2;
                    }
                }

            }
        }

        END_PROFILE(impulse);
        START_PROFILE(delta);

        FLOPS(6, 0, 4, 0, complete_function, delta);
        // Velocity changes
        double velocity_change_x_1 = (deltaP_1 + deltaP_x_1) / M;
        double velocity_change_y_1 = (-deltaP + deltaP_y_1) / M;
        double velocity_change_x_2 = (-deltaP_1 + deltaP_x_2) / M;
        double velocity_change_y_2 = (deltaP + deltaP_y_2) / M;

        FLOPS(4, 0, 0, 0, complete_function, delta);
        // Update velocities
        local_velocity_x_1 += velocity_change_x_1;
        local_velocity_y_1 += velocity_change_y_1;
        local_velocity_x_2 += velocity_change_x_2;
        local_velocity_y_2 += velocity_change_y_2;

        FLOPS(6, 6, 0, 0, complete_function, delta);
        // Angular velocity changes
        double delta_angular_velocity_x_1 = C * (deltaP_2 + deltaP_y_1);
        double delta_angular_velocity_y_1 = C * (-deltaP_x_1);
        double delta_angular_velocity_z_1 = C * (-deltaP_1);
        double delta_angular_velocity_x_2 = C * (deltaP_2 + deltaP_y_2);
        double delta_angular_velocity_y_2 = C * (-deltaP_x_2);
        double delta_angular_velocity_z_2 = C * (-deltaP_1);

        FLOPS(6, 0, 0, 0, complete_function, delta);
        // Update Angular Velocities
        local_angular_velocity_x_1 += delta_angular_velocity_x_1;
        local_angular_velocity_y_1 += delta_angular_velocity_y_1;
        local_angular_velocity_z_1 += delta_angular_velocity_z_1;

        local_angular_velocity_x_2 += delta_angular_velocity_x_2;
        local_angular_velocity_y_2 += delta_angular_velocity_y_2;
        local_angular_velocity_z_2 += delta_angular_velocity_z_2;

        END_PROFILE(delta);
        START_PROFILE(velocity);

        FLOPS(4, 4, 0, 0, complete_function, velocity);
        // update ball-table slips
        surface_velocity_x_1 = local_velocity_x_1 + R * local_angular_velocity_y_1;
        surface_velocity_y_1 = local_velocity_y_1 - R * local_angular_velocity_x_1;
        surface_velocity_x_2 = local_velocity_x_2 + R * local_angular_velocity_y_2;
        surface_velocity_y_2 = local_velocity_y_2 - R * local_angular_velocity_x_2;

        FLOPS(2, 4, 0, 0, complete_function, velocity);
        surface_velocity_magnitude_1_sqrd = (surface_velocity_x_1 * surface_velocity_x_1 + surface_velocity_y_1 * surface_velocity_y_1);
        surface_velocity_magnitude_2_sqrd = (surface_velocity_x_2 * surface_velocity_x_2 + surface_velocity_y_2 * surface_velocity_y_2);

        FLOPS(5, 4, 0, 1, complete_function, velocity);
        // update ball-ball slip:
        contact_point_velocity_x = local_velocity_x_1 - local_velocity_x_2 - R * (local_angular_velocity_z_1 + local_angular_velocity_z_2);
        contact_point_velocity_z = R * (local_angular_velocity_x_1 + local_angular_velocity_x_2);
        ball_ball_contact_point_magnitude = sqrt(contact_point_velocity_x * contact_point_velocity_x + contact_point_velocity_z * contact_point_velocity_z);

        FLOPS(3, 2, 0, 0, complete_function, velocity);
        // Update work and check compression phase
        double velocity_diff_y_temp = velocity_diff_y;
        velocity_diff_y = local_velocity_y_2 - local_velocity_y_1;
        total_work += 0.5 * deltaP * fabs(velocity_diff_y_temp + velocity_diff_y);

        if (work_compression == 0 && velocity_diff_y > 0) {
            work_compression = total_work;
            FLOPS(1, 2, 0, 0, complete_function, velocity);
            work_required = (1.0 + e_b * e_b) * work_compression;
        }

        END_PROFILE(velocity);
    }

    START_PROFILE(after_loop);

    // Transform back to global coordinates
    for (int i = 0; i < 3; i++) {

        MEMORY(4, complete_function, after_loop);
        FLOPS(2, 4, 0, 0, complete_function, after_loop);
        rvw1_result[i + 3] = local_velocity_x_1 * right[i] + local_velocity_y_1 * forward[i];
        rvw2_result[i + 3] = local_velocity_x_2 * right[i] + local_velocity_y_2 * forward[i];

        if(i < 2) {
            FLOPS(2, 4, 0, 0, complete_function, after_loop);
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

DLL_EXPORT void less_sqrt_collide_balls2(double* rvw1, double* rvw2, float R, float M, float u_s1, float u_s2, float u_b, float e_b, float deltaP, int N, double* rvw1_result, double* rvw2_result, Profile* profiles, Branch* branches) {

    #ifdef PROFILE
        Profile* complete_function = &profiles[0];
        Profile* before_loop = &profiles[1];
        Profile* impulse = &profiles[2];
        Profile* delta = &profiles[3];
        Profile* velocity = &profiles[4];
        Profile* after_loop = &profiles[5];
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

    FLOPS(3, 0, 0, 0, complete_function, before_loop);
    double offset[3];
    subV3(translation_2, translation_1, offset);

    FLOPS(2, 3, 0, 0, complete_function, before_loop);
    double offset_mag_sqrd = dotV3(offset, offset);

    FLOPS(0, 0, 0, 1, complete_function, before_loop);
    double offset_mag = sqrt(offset_mag_sqrd);

    double forward[3]; // Forward from ball 1 to ball 2, normalized

    FLOPS(0, 0, 3, 0, complete_function, before_loop);
    divV3(offset, offset_mag, forward);

    double up[3] = {0, 0, 1}; // Probably up axis?

    double right[3]; // Axis orthogonal to Z and forward

    FLOPS(3, 6, 0, 0, complete_function, before_loop);
    crossV3(forward, up, right);

    // From here on, it is assumed that the x axis is the right axis and y axis is the forward axis
    // Transform velocities to local frame

    FLOPS(2 * 4, 3 * 4, 0, 0, complete_function, before_loop);
    double local_velocity_x_1 = dotV3(velocity_1, right);
    double local_velocity_y_1 = dotV3(velocity_1, forward);
    double local_velocity_x_2 = dotV3(velocity_2, right);
    double local_velocity_y_2 = dotV3(velocity_2, forward);

    // Transform angular velocities into local frame

    FLOPS(2*6, 3*6, 0, 0, complete_function, before_loop);
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
    FLOPS(4, 4, 0, 0, complete_function, before_loop);
    double surface_velocity_x_1 = local_velocity_x_1 + R * local_angular_velocity_y_1;
    double surface_velocity_y_1 = local_velocity_y_1 - R * local_angular_velocity_x_1;
    double surface_velocity_x_2 = local_velocity_x_2 + R * local_angular_velocity_y_2;
    double surface_velocity_y_2 = local_velocity_y_2 - R * local_angular_velocity_x_2;

    /*
    FLOPS(2, 4, 0, 0, complete_function, before_loop);
    double surface_velocity_magnitude_1_sqrd = (surface_velocity_x_1 * surface_velocity_x_1 + surface_velocity_y_1 * surface_velocity_y_1);
    double surface_velocity_magnitude_2_sqrd = (surface_velocity_x_2 * surface_velocity_x_2 + surface_velocity_y_2 * surface_velocity_y_2);
    */

    // Relative surface velocity in the x-direction at the point where the two balls are in contact.
    // ball-ball slip
    FLOPS(5, 4, 0, 1, complete_function, before_loop);
    double contact_point_velocity_x = local_velocity_x_1 - local_velocity_x_2 - R * (local_angular_velocity_z_1 + local_angular_velocity_z_2);
    double contact_point_velocity_z = R * (local_angular_velocity_x_1 + local_angular_velocity_x_2);
    double ball_ball_contact_point_magnitude = sqrt(contact_point_velocity_x * contact_point_velocity_x + contact_point_velocity_z * contact_point_velocity_z);
    //printf("\nC Contact Point Slide, Spin:\n");
    //printf("  Contact Point: u_ijC_xz_mag= %.6f\n", ball_ball_contact_point_magnitude);

    // Main collision loop
    FLOPS(1, 0, 0, 0, complete_function, before_loop);
    double velocity_diff_y = local_velocity_y_2 - local_velocity_y_1;

    // deltaP is most likely always 0?
    // ΔP represents the Impulse during a time of Δt
    if (deltaP == 0) {
        FLOPS(0, 4, 1, 0, complete_function, before_loop);
        deltaP = 0.5 * (1.0 + e_b) * M * fabs(velocity_diff_y) / (double)(N);
    }

    FLOPS(0, 2, 1, 0, complete_function, before_loop);
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

    while (velocity_diff_y < 0 || total_work < work_required) {

        START_PROFILE(impulse);

        // Impulse Calculation
        if (ball_ball_contact_point_magnitude < 1e-16) {
            BRANCH(0);
            deltaP_1 = 0;
            deltaP_2 = 0;
            deltaP_x_1 = 0;
            deltaP_y_1 = 0;
            deltaP_x_2 = 0;
            deltaP_y_2 = 0;
        } else {
            BRANCH(1);
            FLOPS(1, 2, 1, 0, complete_function, impulse);
            deltaP_1 = -u_b * deltaP * contact_point_velocity_x / ball_ball_contact_point_magnitude;
            if(fabs(contact_point_velocity_z) < 1e-16) {
                BRANCH(2);
                deltaP_2 = 0;
                deltaP_x_1 = 0;
                deltaP_y_1 = 0;
                deltaP_x_2 = 0;
                deltaP_y_2 = 0;
            } else {
                BRANCH(3);
                FLOPS(1, 2, 1, 0, complete_function, impulse);
                deltaP_2 = -u_b * deltaP * contact_point_velocity_z / ball_ball_contact_point_magnitude;

                if(deltaP_2 > 0) {
                    BRANCH(4);
                    deltaP_x_1 = 0;
                    deltaP_y_1 = 0;

                    // TODO: probably best to check for some tolerance
                    if(surface_velocity_x_2 == 0.0 && surface_velocity_y_2 == 0) {
                        BRANCH(5);
                        deltaP_x_2 = 0;
                        deltaP_y_2 = 0;
                    } else {
                        BRANCH(6);
                        FLOPS(2, 4, 2, 1, complete_function, impulse);
                        double surface_velocity_magnitude_2 = sqrt(surface_velocity_x_2 * surface_velocity_x_2 + surface_velocity_y_2 * surface_velocity_y_2);
                        deltaP_x_2 = -u_s2 * (surface_velocity_x_2 / surface_velocity_magnitude_2) * deltaP_2;
                        deltaP_y_2 = -u_s2 * (surface_velocity_y_2 / surface_velocity_magnitude_2) * deltaP_2;
                    }
                } else {
                    BRANCH(7);
                    deltaP_x_2 = 0;
                    deltaP_y_2 = 0;
                    if(surface_velocity_x_1 == 0.0 && surface_velocity_y_1 == 0) {
                        BRANCH(8);
                        deltaP_x_1 = 0;
                        deltaP_y_1 = 0;
                    } else {
                        BRANCH(9);
                        FLOPS(0, 4, 2, 1, complete_function, impulse);
                        double surface_velocity_magnitude_1 = sqrt(surface_velocity_x_1 * surface_velocity_x_1 + surface_velocity_y_1 * surface_velocity_y_1);
                        deltaP_x_1 = u_s1 * (surface_velocity_x_1 / surface_velocity_magnitude_1) * deltaP_2;
                        deltaP_y_1 = u_s1 * (surface_velocity_y_1 / surface_velocity_magnitude_1) * deltaP_2;
                    }
                }

            }
        }

        END_PROFILE(impulse);
        START_PROFILE(delta);

        FLOPS(6, 0, 4, 0, complete_function, delta);
        // Velocity changes
        double velocity_change_x_1 = (deltaP_1 + deltaP_x_1) / M;
        double velocity_change_y_1 = (-deltaP + deltaP_y_1) / M;
        double velocity_change_x_2 = (-deltaP_1 + deltaP_x_2) / M;
        double velocity_change_y_2 = (deltaP + deltaP_y_2) / M;

        FLOPS(4, 0, 0, 0, complete_function, delta);
        // Update velocities
        local_velocity_x_1 += velocity_change_x_1;
        local_velocity_y_1 += velocity_change_y_1;
        local_velocity_x_2 += velocity_change_x_2;
        local_velocity_y_2 += velocity_change_y_2;

        FLOPS(6, 6, 0, 0, complete_function, delta);
        // Angular velocity changes
        double delta_angular_velocity_x_1 = C * (deltaP_2 + deltaP_y_1);
        double delta_angular_velocity_y_1 = C * (-deltaP_x_1);
        double delta_angular_velocity_z_1 = C * (-deltaP_1);
        double delta_angular_velocity_x_2 = C * (deltaP_2 + deltaP_y_2);
        double delta_angular_velocity_y_2 = C * (-deltaP_x_2);
        double delta_angular_velocity_z_2 = C * (-deltaP_1);

        FLOPS(6, 0, 0, 0, complete_function, delta);
        // Update Angular Velocities
        local_angular_velocity_x_1 += delta_angular_velocity_x_1;
        local_angular_velocity_y_1 += delta_angular_velocity_y_1;
        local_angular_velocity_z_1 += delta_angular_velocity_z_1;

        local_angular_velocity_x_2 += delta_angular_velocity_x_2;
        local_angular_velocity_y_2 += delta_angular_velocity_y_2;
        local_angular_velocity_z_2 += delta_angular_velocity_z_2;

        END_PROFILE(delta);
        START_PROFILE(velocity);

        FLOPS(4, 4, 0, 0, complete_function, velocity);
        // update ball-table slips
        surface_velocity_x_1 = local_velocity_x_1 + R * local_angular_velocity_y_1;
        surface_velocity_y_1 = local_velocity_y_1 - R * local_angular_velocity_x_1;
        surface_velocity_x_2 = local_velocity_x_2 + R * local_angular_velocity_y_2;
        surface_velocity_y_2 = local_velocity_y_2 - R * local_angular_velocity_x_2;

        /*
        FLOPS(2, 4, 0, 0, complete_function, velocity);
        surface_velocity_magnitude_1_sqrd = (surface_velocity_x_1 * surface_velocity_x_1 + surface_velocity_y_1 * surface_velocity_y_1);
        surface_velocity_magnitude_2_sqrd = (surface_velocity_x_2 * surface_velocity_x_2 + surface_velocity_y_2 * surface_velocity_y_2);
        */

        FLOPS(5, 4, 0, 1, complete_function, velocity);
        // update ball-ball slip:
        contact_point_velocity_x = local_velocity_x_1 - local_velocity_x_2 - R * (local_angular_velocity_z_1 + local_angular_velocity_z_2);
        contact_point_velocity_z = R * (local_angular_velocity_x_1 + local_angular_velocity_x_2);
        ball_ball_contact_point_magnitude = sqrt(contact_point_velocity_x * contact_point_velocity_x + contact_point_velocity_z * contact_point_velocity_z);

        FLOPS(3, 2, 0, 0, complete_function, velocity);
        // Update work and check compression phase
        double velocity_diff_y_temp = velocity_diff_y;
        velocity_diff_y = local_velocity_y_2 - local_velocity_y_1;
        total_work += 0.5 * deltaP * fabs(velocity_diff_y_temp + velocity_diff_y);

        if (work_compression == 0 && velocity_diff_y > 0) {
            work_compression = total_work;
            FLOPS(1, 2, 0, 0, complete_function, velocity);
            work_required = (1.0 + e_b * e_b) * work_compression;
        }

        END_PROFILE(velocity);
    }

    START_PROFILE(after_loop);

    // Transform back to global coordinates
    for (int i = 0; i < 3; i++) {

        MEMORY(4, complete_function, after_loop);
        FLOPS(2, 4, 0, 0, complete_function, after_loop);
        rvw1_result[i + 3] = local_velocity_x_1 * right[i] + local_velocity_y_1 * forward[i];
        rvw2_result[i + 3] = local_velocity_x_2 * right[i] + local_velocity_y_2 * forward[i];

        if(i < 2) {
            FLOPS(2, 4, 0, 0, complete_function, after_loop);
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


DLL_EXPORT void branch_prediction_collide_balls(double* rvw1, double* rvw2, float R, float M, float u_s1, float u_s2, float u_b, float e_b, float deltaP, int N, double* rvw1_result, double* rvw2_result, Profile* profiles, Branch* branches) {

    #ifdef PROFILE
        Profile* complete_function = &profiles[0];
        Profile* before_loop = &profiles[1];
        Profile* impulse = &profiles[2];
        Profile* delta = &profiles[3];
        Profile* velocity = &profiles[4];
        Profile* after_loop = &profiles[5];
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

    FLOPS(3, 0, 0, 0, complete_function, before_loop);
    double offset[3];
    subV3(translation_2, translation_1, offset);

    FLOPS(2, 3, 0, 0, complete_function, before_loop);
    double offset_mag_sqrd = dotV3(offset, offset);

    FLOPS(0, 0, 0, 1, complete_function, before_loop);
    double offset_mag = sqrt(offset_mag_sqrd);

    double forward[3]; // Forward from ball 1 to ball 2, normalized

    FLOPS(0, 0, 3, 0, complete_function, before_loop);
    divV3(offset, offset_mag, forward);

    double up[3] = {0, 0, 1}; // Probably up axis?

    double right[3]; // Axis orthogonal to Z and forward

    FLOPS(3, 6, 0, 0, complete_function, before_loop);
    crossV3(forward, up, right);

    // From here on, it is assumed that the x axis is the right axis and y axis is the forward axis
    // Transform velocities to local frame

    FLOPS(2 * 4, 3 * 4, 0, 0, complete_function, before_loop);
    double local_velocity_x_1 = dotV3(velocity_1, right);
    double local_velocity_y_1 = dotV3(velocity_1, forward);
    double local_velocity_x_2 = dotV3(velocity_2, right);
    double local_velocity_y_2 = dotV3(velocity_2, forward);

    // Transform angular velocities into local frame

    FLOPS(2*6, 3*6, 0, 0, complete_function, before_loop);
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
    FLOPS(4, 4, 0, 0, complete_function, before_loop);
    double surface_velocity_x_1 = local_velocity_x_1 + R * local_angular_velocity_y_1;
    double surface_velocity_y_1 = local_velocity_y_1 - R * local_angular_velocity_x_1;
    double surface_velocity_x_2 = local_velocity_x_2 + R * local_angular_velocity_y_2;
    double surface_velocity_y_2 = local_velocity_y_2 - R * local_angular_velocity_x_2;

    FLOPS(2, 4, 0, 2, complete_function, before_loop);
    double surface_velocity_magnitude_1 = sqrt(surface_velocity_x_1 * surface_velocity_x_1 + surface_velocity_y_1 * surface_velocity_y_1);
    double surface_velocity_magnitude_2 = sqrt(surface_velocity_x_2 * surface_velocity_x_2 + surface_velocity_y_2 * surface_velocity_y_2);

    // Relative surface velocity in the x-direction at the point where the two balls are in contact.
    // ball-ball slip
    FLOPS(5, 4, 0, 1, complete_function, before_loop);
    double contact_point_velocity_x = local_velocity_x_1 - local_velocity_x_2 - R * (local_angular_velocity_z_1 + local_angular_velocity_z_2);
    double contact_point_velocity_z = R * (local_angular_velocity_x_1 + local_angular_velocity_x_2);
    double ball_ball_contact_point_magnitude = sqrt(contact_point_velocity_x * contact_point_velocity_x + contact_point_velocity_z * contact_point_velocity_z);
    //printf("\nC Contact Point Slide, Spin:\n");
    //printf("  Contact Point: u_ijC_xz_mag= %.6f\n", ball_ball_contact_point_magnitude);

    // Main collision loop
    FLOPS(1, 0, 0, 0, complete_function, before_loop);
    double velocity_diff_y = local_velocity_y_2 - local_velocity_y_1;

    // deltaP is most likely always 0?
    // ΔP represents the Impulse during a time of Δt
    if (deltaP == 0) {
        FLOPS(0, 4, 1, 0, complete_function, before_loop);
        deltaP = 0.5 * (1.0 + e_b) * M * fabs(velocity_diff_y) / (double)(N);
    }

    FLOPS(0, 2, 1, 0, complete_function, before_loop);
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

    while (__builtin_expect(velocity_diff_y < 0 || total_work < work_required, true)) {

        START_PROFILE(impulse);

        // Impulse Calculation
        if (__builtin_expect(ball_ball_contact_point_magnitude < 1e-16, false)) { // Removing this line makes the code slowerr??
            BRANCH(0);
            deltaP_1 = 0;
            deltaP_2 = 0;
            deltaP_x_1 = 0;
            deltaP_y_1 = 0;
            deltaP_x_2 = 0;
            deltaP_y_2 = 0;
        } else {
            BRANCH(1);
            FLOPS(1, 2, 1, 0, complete_function, impulse);
            deltaP_1 = -u_b * deltaP * contact_point_velocity_x / ball_ball_contact_point_magnitude;
            if(__builtin_expect(fabs(contact_point_velocity_z) < 1e-16, false)) {
                BRANCH(2);
                deltaP_2 = 0;
                deltaP_x_1 = 0;
                deltaP_y_1 = 0;
                deltaP_x_2 = 0;
                deltaP_y_2 = 0;
            } else {
                BRANCH(3);
                FLOPS(1, 2, 1, 0, complete_function, impulse);
                deltaP_2 = -u_b * deltaP * contact_point_velocity_z / ball_ball_contact_point_magnitude;

                if(deltaP_2 > 0) { // 50/50, hard to optimise for branch prediction
                    BRANCH(4);
                    deltaP_x_1 = 0;
                    deltaP_y_1 = 0;

                    // TODO: probably best to check for some tolerance
                    if(__builtin_expect(surface_velocity_magnitude_2 == 0.0, false)) {
                        BRANCH(5);
                        deltaP_x_2 = 0;
                        deltaP_y_2 = 0;
                    } else {
                        BRANCH(6);
                        FLOPS(2, 4, 2, 0, complete_function, impulse);
                        deltaP_x_2 = -u_s2 * (surface_velocity_x_2 / surface_velocity_magnitude_2) * deltaP_2;
                        deltaP_y_2 = -u_s2 * (surface_velocity_y_2 / surface_velocity_magnitude_2) * deltaP_2;
                    }
                } else {
                    BRANCH(7);
                    deltaP_x_2 = 0;
                    deltaP_y_2 = 0;
                    if(__builtin_expect(surface_velocity_magnitude_1 == 0.0, false)) {
                        BRANCH(8);
                        deltaP_x_1 = 0;
                        deltaP_y_1 = 0;
                    } else {
                        BRANCH(9);
                        FLOPS(0, 4, 2, 0, complete_function, impulse);
                        deltaP_x_1 = u_s1 * (surface_velocity_x_1 / surface_velocity_magnitude_1) * deltaP_2;
                        deltaP_y_1 = u_s1 * (surface_velocity_y_1 / surface_velocity_magnitude_1) * deltaP_2;
                    }
                }

            }
        }

        END_PROFILE(impulse);
        START_PROFILE(delta);

        FLOPS(6, 0, 4, 0, complete_function, delta);
        // Velocity changes
        double velocity_change_x_1 = (deltaP_1 + deltaP_x_1) / M;
        double velocity_change_y_1 = (-deltaP + deltaP_y_1) / M;
        double velocity_change_x_2 = (-deltaP_1 + deltaP_x_2) / M;
        double velocity_change_y_2 = (deltaP + deltaP_y_2) / M;

        FLOPS(4, 0, 0, 0, complete_function, delta);
        // Update velocities
        local_velocity_x_1 += velocity_change_x_1;
        local_velocity_y_1 += velocity_change_y_1;
        local_velocity_x_2 += velocity_change_x_2;
        local_velocity_y_2 += velocity_change_y_2;

        FLOPS(6, 6, 0, 0, complete_function, delta);
        // Angular velocity changes
        double delta_angular_velocity_x_1 = C * (deltaP_2 + deltaP_y_1);
        double delta_angular_velocity_y_1 = C * (-deltaP_x_1);
        double delta_angular_velocity_z_1 = C * (-deltaP_1);
        double delta_angular_velocity_x_2 = C * (deltaP_2 + deltaP_y_2);
        double delta_angular_velocity_y_2 = C * (-deltaP_x_2);
        double delta_angular_velocity_z_2 = C * (-deltaP_1);

        FLOPS(6, 0, 0, 0, complete_function, delta);
        // Update Angular Velocities
        local_angular_velocity_x_1 += delta_angular_velocity_x_1;
        local_angular_velocity_y_1 += delta_angular_velocity_y_1;
        local_angular_velocity_z_1 += delta_angular_velocity_z_1;

        local_angular_velocity_x_2 += delta_angular_velocity_x_2;
        local_angular_velocity_y_2 += delta_angular_velocity_y_2;
        local_angular_velocity_z_2 += delta_angular_velocity_z_2;

        END_PROFILE(delta);
        START_PROFILE(velocity);

        FLOPS(4, 4, 0, 0, complete_function, velocity);
        // update ball-table slips
        surface_velocity_x_1 = local_velocity_x_1 + R * local_angular_velocity_y_1;
        surface_velocity_y_1 = local_velocity_y_1 - R * local_angular_velocity_x_1;
        surface_velocity_x_2 = local_velocity_x_2 + R * local_angular_velocity_y_2;
        surface_velocity_y_2 = local_velocity_y_2 - R * local_angular_velocity_x_2;

        FLOPS(2, 4, 0, 2, complete_function, velocity);
        surface_velocity_magnitude_1 = sqrt(surface_velocity_x_1 * surface_velocity_x_1 + surface_velocity_y_1 * surface_velocity_y_1);
        surface_velocity_magnitude_2 = sqrt(surface_velocity_x_2 * surface_velocity_x_2 + surface_velocity_y_2 * surface_velocity_y_2);

        FLOPS(5, 4, 0, 1, complete_function, velocity);
        // update ball-ball slip:
        contact_point_velocity_x = local_velocity_x_1 - local_velocity_x_2 - R * (local_angular_velocity_z_1 + local_angular_velocity_z_2);
        contact_point_velocity_z = R * (local_angular_velocity_x_1 + local_angular_velocity_x_2);
        ball_ball_contact_point_magnitude = sqrt(contact_point_velocity_x * contact_point_velocity_x + contact_point_velocity_z * contact_point_velocity_z);

        FLOPS(3, 2, 0, 0, complete_function, velocity);
        // Update work and check compression phase
        double velocity_diff_y_temp = velocity_diff_y;
        velocity_diff_y = local_velocity_y_2 - local_velocity_y_1;
        total_work += 0.5 * deltaP * fabs(velocity_diff_y_temp + velocity_diff_y);

        if (__builtin_expect(work_compression == 0 && velocity_diff_y > 0, false)) {
            BRANCH(10);
            work_compression = total_work;
            FLOPS(1, 2, 0, 0, complete_function, velocity);
            work_required = (1.0 + e_b * e_b) * work_compression;
        }

        END_PROFILE(velocity);
    }

    START_PROFILE(after_loop);

    // Transform back to global coordinates
    for (int i = 0; i < 3; i++) {

        MEMORY(4, complete_function, after_loop);
        FLOPS(2, 4, 0, 0, complete_function, after_loop);
        rvw1_result[i + 3] = local_velocity_x_1 * right[i] + local_velocity_y_1 * forward[i];
        rvw2_result[i + 3] = local_velocity_x_2 * right[i] + local_velocity_y_2 * forward[i];

        if(i < 2) {
            FLOPS(2, 4, 0, 0, complete_function, after_loop);
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


DLL_EXPORT void remove_unused_branches(double* rvw1, double* rvw2, float R, float M, float u_s1, float u_s2, float u_b, float e_b, float deltaP, int N, double* rvw1_result, double* rvw2_result, Profile* profiles, Branch* branches) {

    #ifdef PROFILE
        Profile* complete_function = &profiles[0];
        Profile* before_loop = &profiles[1];
        Profile* impulse = &profiles[2];
        Profile* delta = &profiles[3];
        Profile* velocity = &profiles[4];
        Profile* after_loop = &profiles[5];
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

    FLOPS(3, 0, 0, 0, complete_function, before_loop);
    double offset[3];
    subV3(translation_2, translation_1, offset);

    FLOPS(2, 3, 0, 0, complete_function, before_loop);
    double offset_mag_sqrd = dotV3(offset, offset);

    FLOPS(0, 0, 0, 1, complete_function, before_loop);
    double offset_mag = sqrt(offset_mag_sqrd);

    double forward[3]; // Forward from ball 1 to ball 2, normalized

    FLOPS(0, 0, 3, 0, complete_function, before_loop);
    divV3(offset, offset_mag, forward);

    double up[3] = {0, 0, 1}; // Probably up axis?

    double right[3]; // Axis orthogonal to Z and forward

    FLOPS(3, 6, 0, 0, complete_function, before_loop);
    crossV3(forward, up, right);

    // From here on, it is assumed that the x axis is the right axis and y axis is the forward axis
    // Transform velocities to local frame

    FLOPS(2 * 4, 3 * 4, 0, 0, complete_function, before_loop);
    double local_velocity_x_1 = dotV3(velocity_1, right);
    double local_velocity_y_1 = dotV3(velocity_1, forward);
    double local_velocity_x_2 = dotV3(velocity_2, right);
    double local_velocity_y_2 = dotV3(velocity_2, forward);

    // Transform angular velocities into local frame

    FLOPS(2*6, 3*6, 0, 0, complete_function, before_loop);
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
    FLOPS(4, 4, 0, 0, complete_function, before_loop);
    double surface_velocity_x_1 = local_velocity_x_1 + R * local_angular_velocity_y_1;
    double surface_velocity_y_1 = local_velocity_y_1 - R * local_angular_velocity_x_1;
    double surface_velocity_x_2 = local_velocity_x_2 + R * local_angular_velocity_y_2;
    double surface_velocity_y_2 = local_velocity_y_2 - R * local_angular_velocity_x_2;

    FLOPS(2, 4, 0, 2, complete_function, before_loop);
    double surface_velocity_magnitude_1 = sqrt(surface_velocity_x_1 * surface_velocity_x_1 + surface_velocity_y_1 * surface_velocity_y_1);
    double surface_velocity_magnitude_2 = sqrt(surface_velocity_x_2 * surface_velocity_x_2 + surface_velocity_y_2 * surface_velocity_y_2);

    // Relative surface velocity in the x-direction at the point where the two balls are in contact.
    // ball-ball slip
    FLOPS(5, 4, 0, 1, complete_function, before_loop);
    double contact_point_velocity_x = local_velocity_x_1 - local_velocity_x_2 - R * (local_angular_velocity_z_1 + local_angular_velocity_z_2);
    double contact_point_velocity_z = R * (local_angular_velocity_x_1 + local_angular_velocity_x_2);
    double ball_ball_contact_point_magnitude = sqrt(contact_point_velocity_x * contact_point_velocity_x + contact_point_velocity_z * contact_point_velocity_z);
    //printf("\nC Contact Point Slide, Spin:\n");
    //printf("  Contact Point: u_ijC_xz_mag= %.6f\n", ball_ball_contact_point_magnitude);

    // Main collision loop
    FLOPS(1, 0, 0, 0, complete_function, before_loop);
    double velocity_diff_y = local_velocity_y_2 - local_velocity_y_1;

    // deltaP is most likely always 0?
    // ΔP represents the Impulse during a time of Δt
    if (deltaP == 0) {
        FLOPS(0, 4, 1, 0, complete_function, before_loop);
        deltaP = 0.5 * (1.0 + e_b) * M * fabs(velocity_diff_y) / (double)(N);
    }

    FLOPS(0, 2, 1, 0, complete_function, before_loop);
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

    while (velocity_diff_y < 0 || total_work < work_required) {

        START_PROFILE(impulse);

            BRANCH(1);
            FLOPS(1, 2, 1, 0, complete_function, impulse);
            deltaP_1 = -u_b * deltaP * contact_point_velocity_x / ball_ball_contact_point_magnitude;
                BRANCH(3);
                FLOPS(1, 2, 1, 0, complete_function, impulse);
                deltaP_2 = -u_b * deltaP * contact_point_velocity_z / ball_ball_contact_point_magnitude;

                if(deltaP_2 > 0) { // 50/50, hard to optimise for branch prediction
                    BRANCH(4);
                    deltaP_x_1 = 0;
                    deltaP_y_1 = 0;

                    // TODO: probably best to check for some tolerance
                    if(surface_velocity_magnitude_2 == 0.0) {
                        BRANCH(5);
                        deltaP_x_2 = 0;
                        deltaP_y_2 = 0;
                    } else {
                        BRANCH(6);
                        FLOPS(2, 4, 2, 0, complete_function, impulse);
                        deltaP_x_2 = -u_s2 * (surface_velocity_x_2 / surface_velocity_magnitude_2) * deltaP_2;
                        deltaP_y_2 = -u_s2 * (surface_velocity_y_2 / surface_velocity_magnitude_2) * deltaP_2;
                    }
                } else {
                    BRANCH(7);
                    deltaP_x_2 = 0;
                    deltaP_y_2 = 0;
                    if(surface_velocity_magnitude_1 == 0.0) {
                        BRANCH(8);
                        deltaP_x_1 = 0;
                        deltaP_y_1 = 0;
                    } else {
                        BRANCH(9);
                        FLOPS(0, 4, 2, 0, complete_function, impulse);
                        deltaP_x_1 = u_s1 * (surface_velocity_x_1 / surface_velocity_magnitude_1) * deltaP_2;
                        deltaP_y_1 = u_s1 * (surface_velocity_y_1 / surface_velocity_magnitude_1) * deltaP_2;
                    }
                }


        END_PROFILE(impulse);
        START_PROFILE(delta);

        FLOPS(6, 0, 4, 0, complete_function, delta);
        // Velocity changes
        double velocity_change_x_1 = (deltaP_1 + deltaP_x_1) / M;
        double velocity_change_y_1 = (-deltaP + deltaP_y_1) / M;
        double velocity_change_x_2 = (-deltaP_1 + deltaP_x_2) / M;
        double velocity_change_y_2 = (deltaP + deltaP_y_2) / M;

        FLOPS(4, 0, 0, 0, complete_function, delta);
        // Update velocities
        local_velocity_x_1 += velocity_change_x_1;
        local_velocity_y_1 += velocity_change_y_1;
        local_velocity_x_2 += velocity_change_x_2;
        local_velocity_y_2 += velocity_change_y_2;

        FLOPS(6, 6, 0, 0, complete_function, delta);
        // Angular velocity changes
        double delta_angular_velocity_x_1 = C * (deltaP_2 + deltaP_y_1);
        double delta_angular_velocity_y_1 = C * (-deltaP_x_1);
        double delta_angular_velocity_z_1 = C * (-deltaP_1);
        double delta_angular_velocity_x_2 = C * (deltaP_2 + deltaP_y_2);
        double delta_angular_velocity_y_2 = C * (-deltaP_x_2);
        double delta_angular_velocity_z_2 = C * (-deltaP_1);

        FLOPS(6, 0, 0, 0, complete_function, delta);
        // Update Angular Velocities
        local_angular_velocity_x_1 += delta_angular_velocity_x_1;
        local_angular_velocity_y_1 += delta_angular_velocity_y_1;
        local_angular_velocity_z_1 += delta_angular_velocity_z_1;

        local_angular_velocity_x_2 += delta_angular_velocity_x_2;
        local_angular_velocity_y_2 += delta_angular_velocity_y_2;
        local_angular_velocity_z_2 += delta_angular_velocity_z_2;

        END_PROFILE(delta);
        START_PROFILE(velocity);

        FLOPS(4, 4, 0, 0, complete_function, velocity);
        // update ball-table slips
        surface_velocity_x_1 = local_velocity_x_1 + R * local_angular_velocity_y_1;
        surface_velocity_y_1 = local_velocity_y_1 - R * local_angular_velocity_x_1;
        surface_velocity_x_2 = local_velocity_x_2 + R * local_angular_velocity_y_2;
        surface_velocity_y_2 = local_velocity_y_2 - R * local_angular_velocity_x_2;

        FLOPS(2, 4, 0, 2, complete_function, velocity);
        surface_velocity_magnitude_1 = sqrt(surface_velocity_x_1 * surface_velocity_x_1 + surface_velocity_y_1 * surface_velocity_y_1);
        surface_velocity_magnitude_2 = sqrt(surface_velocity_x_2 * surface_velocity_x_2 + surface_velocity_y_2 * surface_velocity_y_2);

        FLOPS(5, 4, 0, 1, complete_function, velocity);
        // update ball-ball slip:
        contact_point_velocity_x = local_velocity_x_1 - local_velocity_x_2 - R * (local_angular_velocity_z_1 + local_angular_velocity_z_2);
        contact_point_velocity_z = R * (local_angular_velocity_x_1 + local_angular_velocity_x_2);
        ball_ball_contact_point_magnitude = sqrt(contact_point_velocity_x * contact_point_velocity_x + contact_point_velocity_z * contact_point_velocity_z);

        FLOPS(3, 2, 0, 0, complete_function, velocity);
        // Update work and check compression phase
        double velocity_diff_y_temp = velocity_diff_y;
        velocity_diff_y = local_velocity_y_2 - local_velocity_y_1;
        total_work += 0.5 * deltaP * fabs(velocity_diff_y_temp + velocity_diff_y);

        if (work_compression == 0 && velocity_diff_y > 0) {
            work_compression = total_work;
            FLOPS(1, 2, 0, 0, complete_function, velocity);
            work_required = (1.0 + e_b * e_b) * work_compression;
        }

        END_PROFILE(velocity);
    }

    START_PROFILE(after_loop);

    // Transform back to global coordinates
    for (int i = 0; i < 3; i++) {

        MEMORY(4, complete_function, after_loop);
        FLOPS(2, 4, 0, 0, complete_function, after_loop);
        rvw1_result[i + 3] = local_velocity_x_1 * right[i] + local_velocity_y_1 * forward[i];
        rvw2_result[i + 3] = local_velocity_x_2 * right[i] + local_velocity_y_2 * forward[i];

        if(i < 2) {
            FLOPS(2, 4, 0, 0, complete_function, after_loop);
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

DLL_EXPORT void code_motion_collide_balls(double* rvw1, double* rvw2, float Rf, float Mf, float u_s1f, float u_s2f, float u_bf, float e_bf, float deltaPf, int N, double* rvw1_result, double* rvw2_result, Profile* profiles, Branch* branches) {

    #ifdef PROFILE
        Profile* complete_function = &profiles[0];
        Profile* before_loop = &profiles[1];
        Profile* impulse = &profiles[2];
        Profile* delta = &profiles[3];
        Profile* velocity = &profiles[4];
        Profile* after_loop = &profiles[5];
    #endif

    START_PROFILE(complete_function);
    START_PROFILE(before_loop);

    FLOPS(2, 0, 1, 0, complete_function, before_loop);

    double R = (double)Rf;
    double M = (double)Mf;
    double u_s1 = (double)u_s1f;
    double u_s2 = -(double)u_s2f;
    double u_b = -(double)u_bf;
    double e_b = (double)e_bf;
    double deltaP = (double)deltaPf;

    double M_rep = 1.0f / M;

     // Get pointers into the state arrays
    MEMORY(18, complete_function, before_loop);
    double* translation_1 = &rvw1[0];
    double* velocity_1    = &rvw1[3];
    double* angular_1     = &rvw1[6];
    double* translation_2 = &rvw2[0];
    double* velocity_2    = &rvw2[3];
    double* angular_2     = &rvw2[6];

    FLOPS(3, 0, 0, 0, complete_function, before_loop);
    double offset[3];
    offset[0] = translation_2[0] - translation_1[0];
    offset[1] = translation_2[1] - translation_1[1];
    offset[2] = translation_2[2] - translation_1[2];

    FLOPS(2, 3, 1, 1, complete_function, before_loop);
    double offset_mag = 1.0 / sqrt(offset[0] * offset[0] + offset[1] * offset[1] + offset[2] * offset[2]);

    FLOPS(0, 3, 0, 0, complete_function, before_loop);
    double forward[3]; // Forward from ball 1 to ball 2, normalized
    forward[0] = offset[0] * offset_mag;
    forward[1] = offset[1] * offset_mag;
    forward[2] = offset[2] * offset_mag;

    FLOPS(1, 0, 0, 0, complete_function, before_loop);
    double right[3]; // Axis orthogonal to Z and forward
    right[0] = forward[1];
    right[1] = -forward[0];
    right[2] = 0;

    // From here on, it is assumed that the x axis is the right axis and y axis is the forward axis
    // Transform velocities to local frame

    FLOPS(4, 6, 0, 0, complete_function, before_loop);
    double local_vel_1[2] = {
        velocity_1[0] * right[0] + velocity_1[1] * right[1] + velocity_1[2] * right[2],
        velocity_1[0] * forward[0] + velocity_1[1] * forward[1] + velocity_1[2] * forward[2]
    };

    FLOPS(4, 6, 0, 0, complete_function, before_loop);
    double local_ang_1[3] = {
        angular_1[0] * right[0] + angular_1[1] * right[1] + angular_1[2] * right[2],
        angular_1[0] * forward[0] + angular_1[1] * forward[1] + angular_1[2] * forward[2],
        angular_1[2]
    };

    FLOPS(4, 6, 0, 0, complete_function, before_loop);
    double local_vel_2[2] = {
        velocity_2[0] * right[0] + velocity_2[1] * right[1] + velocity_2[2] * right[2],
        velocity_2[0] * forward[0] + velocity_2[1] * forward[1] + velocity_2[2] * forward[2]
    };

    FLOPS(4, 6, 0, 0, complete_function, before_loop);
    double local_ang_2[3] = {
        angular_2[0] * right[0] + angular_2[1] * right[1] + angular_2[2] * right[2],
        angular_2[0] * forward[0] + angular_2[1] * forward[1] + angular_2[2] * forward[2],
        angular_2[2]
    };


    // Calculate velocity at contact point
    // = Calculate ball-table slips?
    // Slip refers to relative motion between two surfaces in contact — here, the ball and the table.
    // Its the velocity at the contact point of the table and the ball
    FLOPS(4, 4, 0, 0, complete_function, before_loop);
    double surf_vel_1[2] = { local_vel_1[0] + R * local_ang_1[1],
            local_vel_1[1] - R * local_ang_1[0] };
    double surf_vel_2[2] = { local_vel_2[0] + R * local_ang_2[1],
            local_vel_2[1] - R * local_ang_2[0] };

    FLOPS(2, 4, 0, 0, complete_function, before_loop);
    double surf_vel_mag_1_sqrd = surf_vel_1[0]*surf_vel_1[0] + surf_vel_1[1]*surf_vel_1[1];
    double surf_vel_mag_2_sqrd = surf_vel_2[0]*surf_vel_2[0] + surf_vel_2[1]*surf_vel_2[1];

    // Relative surface velocity in the x-direction at the point where the two balls are in contact.
    // ball-ball slip
    FLOPS(5, 4, 0, 0, complete_function, before_loop);
    double contact_vel[2] = { local_vel_1[0] - local_vel_2[0] - R * (local_ang_1[2] + local_ang_2[2]),
            R * (local_ang_1[0] + local_ang_2[0]) };
    double ball_ball_contact_mag_sqrd = contact_vel[0]*contact_vel[0] + contact_vel[1]*contact_vel[1];

    FLOPS(1, 0, 0, 0, complete_function, before_loop);
    // Main collision loop
    double velocity_diff_y = local_vel_2[1] - local_vel_1[1];

    // deltaP is most likely always 0?
    // ΔP represents the Impulse during a time of Δt
    if (deltaP == 0) {
        FLOPS(1, 3, 1, 0, complete_function, before_loop);
        deltaP = 0.5 * (1.0 + e_b) * M * fabs(velocity_diff_y)/(double)N;
    }

    FLOPS(0, 2, 0, 0, complete_function, before_loop);
    float half_deltaP = 0.5 * deltaP;
    float e_b_sqrd_plus_1 = e_b * e_b + 1;

    FLOPS(0, 1, 1, 0, complete_function, before_loop);
    double C = 2.5 * M_rep / R;
    double total_work = 0; // Work done due to impulse force
    double work_required = INFINITY; // Total amount of work required before collision handling is complete
    double work_compression = 0;

    // TODO: better naming for deltas
    // Delta impulse overall per ball?
    // Impulse per axis per ball

    // Initialization not needed, will be overwritten anyways.
    double deltaP_ball[2];
    double deltaP_ball_C[2]; // Multiplied by C

    double deltaP_axis_1[2];
    double deltaP_axis_1_C[2]; // Multiplied by C

    double deltaP_axis_2[2];
    double deltaP_axis_2_C[2]; // Multiplied by C

    double ball_ball_contact_mag;
    double delta_ball_precomp;
    double surf_vel_mag_1;
    double surf_vel_mag_2;
    double surf_vel_precomp;
    double local_ang_1_0_R;
    double local_ang_2_0_R;
    double prev_diff;

    END_PROFILE(before_loop);

    while (velocity_diff_y < 0 || total_work < work_required) {

        START_PROFILE(impulse);

        // Impulse Calculation
        // TODO: I think in 99% of cases this will be true, maybe its faster to just skip this if and do it in the end
        // if (ball_ball_contact_mag_sqrd >= 1e-32) { // Always executed
            BRANCH(0);

            FLOPS(0, 3, 1, 1, complete_function, impulse);
            // TODO: Could be optimized by using reciprocal sqrt, but intrinsics only support floats
            ball_ball_contact_mag = sqrt(ball_ball_contact_mag_sqrd);
            delta_ball_precomp = u_b * deltaP / ball_ball_contact_mag;
            deltaP_ball[0] = delta_ball_precomp * contact_vel[0];
            deltaP_ball_C[0] = C * deltaP_ball[0];

            // if(fabs(contact_vel[1]) >= 1e-16) { // Executed every time
                BRANCH(2);
                FLOPS(0, 3, 1, 0, complete_function, impulse);
                deltaP_ball[1] = delta_ball_precomp * contact_vel[1];
                deltaP_ball_C[1] = C * deltaP_ball[1];
                if(deltaP_ball[1] > 0) {
                    BRANCH(4); // 50 % of executions
                    deltaP_axis_1[0] = deltaP_axis_1[1] = 0;
                    deltaP_axis_1_C[0] = deltaP_axis_1_C[1] = 0;
                    if(surf_vel_mag_2_sqrd != 0.0) {
                        BRANCH(6);
                        FLOPS(0, 6, 2, 1, complete_function, impulse);
                        surf_vel_mag_2 = sqrt(surf_vel_mag_2_sqrd);
                        surf_vel_precomp = u_s2 * deltaP_ball[1] / surf_vel_mag_2;
                        deltaP_axis_2[0] = surf_vel_precomp * surf_vel_2[0];
                        deltaP_axis_2[1] = surf_vel_precomp * surf_vel_2[1];
                        deltaP_axis_2_C[0] = C * deltaP_axis_2[0];
                        deltaP_axis_2_C[1] = C * deltaP_axis_2[1];
                    } else {
                        BRANCH(7); // Executed Once or 0 times
                        deltaP_axis_2[0] = deltaP_axis_2[1] = 0;
                        deltaP_axis_2_C[0] = deltaP_axis_2_C[1] = 0;
                    }
                } else {
                    BRANCH(5); // 50 % of executions
                    deltaP_axis_2[0] = deltaP_axis_2[1] = 0;
                    deltaP_axis_2_C[0] = deltaP_axis_2_C[1] = 0;
                    if(surf_vel_mag_1_sqrd != 0.0) {
                        BRANCH(8);
                        FLOPS(0, 5, 1, 1, complete_function, impulse);
                        surf_vel_mag_1 = sqrt(surf_vel_mag_1_sqrd);
                        surf_vel_precomp = u_s1 * deltaP_ball[1] / surf_vel_mag_1;
                        deltaP_axis_1[0] = surf_vel_precomp * surf_vel_1[0];
                        deltaP_axis_1[1] = surf_vel_precomp * surf_vel_1[1];
                        deltaP_axis_1_C[0] = C * deltaP_axis_1[0];
                        deltaP_axis_1_C[1] = C * deltaP_axis_1[1];
                    } else {
                        BRANCH(9);
                        deltaP_axis_1[0] = deltaP_axis_1[1] = 0;
                        deltaP_axis_1_C[0] = deltaP_axis_1_C[1] = 0;
                    }
                }
                /*  } else { // In all five testcases, not once executed
                BRANCH(3);
                deltaP_ball[1] = 0;
                deltaP_ball_C[1] = 0;
                deltaP_axis_1[0] = deltaP_axis_1[1] = 0;
                deltaP_axis_1_C[0] = deltaP_axis_1_C[1] = 0;
                deltaP_axis_2[0] = deltaP_axis_2[1] = 0;
                deltaP_axis_2_C[0] = deltaP_axis_2_C[1] = 0;
            }
            */
        /* } else { // Never executed
            BRANCH(1);
            deltaP_ball[0] = deltaP_ball[1] = 0;
            deltaP_ball_C[0] = deltaP_ball_C[1] = 0;
            deltaP_axis_1[0] = deltaP_axis_1[1] = 0;
            deltaP_axis_1_C[0] = deltaP_axis_1_C[1] = 0;
            deltaP_axis_2[0] = deltaP_axis_2[1] = 0;
            deltaP_axis_2_C[0] = deltaP_axis_2_C[1] = 0;
        } */

        END_PROFILE(impulse);

        START_PROFILE(delta);

        FLOPS(10, 4, 0, 0, complete_function, delta);
        // Update velocities
        local_vel_1[0] += (deltaP_ball[0] + deltaP_axis_1[0]) * M_rep;
        local_vel_1[1] += (-deltaP + deltaP_axis_1[1]) * M_rep;
        local_vel_2[0] += (-deltaP_ball[0] + deltaP_axis_2[0]) * M_rep;
        local_vel_2[1] += (deltaP + deltaP_axis_2[1]) * M_rep;

        FLOPS(8, 0, 0, 0, complete_function, delta);
        // Update Angular Velocities
        local_ang_1[0] += (deltaP_ball_C[1] + deltaP_axis_1_C[1]);
        local_ang_1[1] -= deltaP_axis_1_C[0];
        local_ang_1[2] -= deltaP_ball_C[0];

        local_ang_2[0] += (deltaP_ball_C[1] + deltaP_axis_2_C[1]);
        local_ang_2[1] -= deltaP_axis_2_C[0];
        local_ang_2[2] -= deltaP_ball_C[0];

        FLOPS(0, 2, 0, 0, complete_function, delta);
        local_ang_1_0_R = R * local_ang_1[0];
        local_ang_2_0_R = R * local_ang_2[0];
        END_PROFILE(delta);
        START_PROFILE(velocity);

        // Recalculate surface velocities using the updated arrays
        FLOPS(4, 2, 0, 0, complete_function, velocity);
        surf_vel_1[0] = local_vel_1[0] + R * local_ang_1[1];
        surf_vel_1[1] = local_vel_1[1] - local_ang_1_0_R;
        surf_vel_2[0] = local_vel_2[0] + R * local_ang_2[1];
        surf_vel_2[1] = local_vel_2[1] - local_ang_2_0_R;

        FLOPS(2, 4, 0, 0, complete_function, velocity);
        surf_vel_mag_1_sqrd = surf_vel_1[0]*surf_vel_1[0] + surf_vel_1[1]*surf_vel_1[1];
        surf_vel_mag_2_sqrd = surf_vel_2[0]*surf_vel_2[0] + surf_vel_2[1]*surf_vel_2[1];

        FLOPS(5, 2, 0, 0, complete_function, velocity);
        // update ball-ball slip:
        contact_vel[0] = local_vel_1[0] - local_vel_2[0] - R * (local_ang_1[2] + local_ang_2[2]);
        contact_vel[1] = local_ang_1_0_R + local_ang_2_0_R;
        ball_ball_contact_mag_sqrd = contact_vel[0]*contact_vel[0] + contact_vel[1]*contact_vel[1];

        FLOPS(3, 1, 0, 0, complete_function, velocity);
        // Update work and check compression phase
        prev_diff = velocity_diff_y;
        velocity_diff_y = local_vel_2[1] - local_vel_1[1];
        total_work += half_deltaP * fabs(prev_diff + velocity_diff_y);

        if (work_compression == 0 && velocity_diff_y > 0) {
            work_compression = total_work;
            FLOPS(0, 1, 0, 0, complete_function, velocity);
            work_required = e_b_sqrd_plus_1 * work_compression;
        }

        END_PROFILE(velocity);
    }

    START_PROFILE(after_loop);

    // Transform back to global coordinates
    for (int i = 0; i < 3; i++) {
        MEMORY(4, complete_function, after_loop);

        FLOPS(2, 4, 0, 0, complete_function, after_loop);
        rvw1_result[i+3] = local_vel_1[0] * right[i] + local_vel_1[1] * forward[i];
        rvw2_result[i+3] = local_vel_2[0] * right[i] + local_vel_2[1] * forward[i];
        if(i < 2) {
            FLOPS(2, 4, 0, 0, complete_function, after_loop);
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

DLL_EXPORT void code_motion_collide_balls2(double* rvw1, double* rvw2, float R, float M, float u_s1, float u_s2, float u_b, float e_b, float deltaP, int N, double* rvw1_result, double* rvw2_result, Profile* profiles, Branch* branches) {

    #ifdef PROFILE
        Profile* complete_function = &profiles[0];
        Profile* before_loop = &profiles[1];
        Profile* impulse = &profiles[2];
        Profile* delta = &profiles[3];
        Profile* velocity = &profiles[4];
        Profile* after_loop = &profiles[5];
    #endif

    START_PROFILE(complete_function);
    START_PROFILE(before_loop);

    // Altough the memory is not really loaded here, assuming its only compulsary misses
    MEMORY(18, complete_function, before_loop);
    double* translation_1 = rvw1;
    double* velocity_1 = &rvw1[3];
    double* angular_velocity_1 = &rvw1[6];

    double* translation_2 = rvw2;
    double* velocity_2 = &rvw2[3];
    double* angular_velocity_2 = &rvw2[6];

    FLOPS(3, 0, 0, 0, complete_function, before_loop);
    double forward[3]; // Forward from ball 1 to ball 2, normalized, forard[2] will always be zero
    forward[0] = translation_2[0] - translation_1[0];
    forward[1] = translation_2[1] - translation_1[1];

    FLOPS(2, 3, 0, 0, complete_function, before_loop);
    double offset_mag = forward[0] * forward[0] + forward[1] * forward[1] ;

    FLOPS(0, 0, 0, 1, complete_function, before_loop);
    offset_mag = sqrt(offset_mag);

    FLOPS(0, 0, 3, 0, complete_function, before_loop);
    forward[0] = forward[0] / offset_mag;
    forward[1] = forward[1] / offset_mag;

    FLOPS(1, 0, 0, 0, complete_function, before_loop);
    forward[2] = -forward[0]; // This is the same as right[1]

    // From here on, it is assumed that the x axis is the right axis and y axis is the forward axis
    // Transform velocities to local frame

    FLOPS(2, 4, 0, 0, complete_function, before_loop);
    double local_velocity_x_1 = velocity_1[0] * forward[1] + velocity_1[1] * forward[2];
    double local_velocity_x_2 = velocity_2[0] * forward[1] + velocity_2[1] * forward[2];

    FLOPS(2, 4, 0, 0, complete_function, before_loop);
    double local_velocity_y_1 = velocity_1[0] * forward[0] + velocity_1[1] * forward[1];
    double local_velocity_y_2 = velocity_2[0] * forward[0] + velocity_2[1] * forward[1];

    // Transform angular velocities into local frame

    FLOPS(2, 4, 0, 0, complete_function, before_loop);
    double local_angular_velocity_x_1 = angular_velocity_1[0] * forward[1] + angular_velocity_1[1] * forward[2];
    double local_angular_velocity_x_2 = angular_velocity_2[0] * forward[1] + angular_velocity_2[1] * forward[2];

    FLOPS(2, 4, 0, 0, complete_function, before_loop);
    double local_angular_velocity_y_1 = angular_velocity_1[0] * forward[0] + angular_velocity_1[1] * forward[1];
    double local_angular_velocity_y_2 = angular_velocity_2[0] * forward[0] + angular_velocity_2[1] * forward[1];

    double local_angular_velocity_z_1 = angular_velocity_1[2];
    double local_angular_velocity_z_2 = angular_velocity_2[2];

    // Calculate velocity at contact point
    // = Calculate ball-table slips?
    // Slip refers to relative motion between two surfaces in contact — here, the ball and the table.
    // Its the velocity at the contact point of the table and the ball
    FLOPS(4, 4, 0, 0, complete_function, before_loop);
    double surface_velocity_x_1 = local_velocity_x_1 + R * local_angular_velocity_y_1;
    double surface_velocity_y_1 = local_velocity_y_1 - R * local_angular_velocity_x_1;
    double surface_velocity_x_2 = local_velocity_x_2 + R * local_angular_velocity_y_2;
    double surface_velocity_y_2 = local_velocity_y_2 - R * local_angular_velocity_x_2;

    FLOPS(2, 4, 0, 0, complete_function, before_loop);
    double surface_velocity_magnitude_1_sqrd = (surface_velocity_x_1 * surface_velocity_x_1 + surface_velocity_y_1 * surface_velocity_y_1);
    double surface_velocity_magnitude_2_sqrd = (surface_velocity_x_2 * surface_velocity_x_2 + surface_velocity_y_2 * surface_velocity_y_2);

    // Relative surface velocity in the x-direction at the point where the two balls are in contact.
    // ball-ball slip
    FLOPS(5, 4, 0, 1, complete_function, before_loop);
    double contact_point_velocity_x = local_velocity_x_1 - local_velocity_x_2 - R * (local_angular_velocity_z_1 + local_angular_velocity_z_2);
    double contact_point_velocity_z = R * (local_angular_velocity_x_1 + local_angular_velocity_x_2);
    double ball_ball_contact_point_magnitude = sqrt(contact_point_velocity_x * contact_point_velocity_x + contact_point_velocity_z * contact_point_velocity_z);
    //printf("\nC Contact Point Slide, Spin:\n");
    //printf("  Contact Point: u_ijC_xz_mag= %.6f\n", ball_ball_contact_point_magnitude);

    // Main collision loop
    FLOPS(1, 0, 0, 0, complete_function, before_loop);
    double velocity_diff_y = local_velocity_y_2 - local_velocity_y_1;

    // deltaP is most likely always 0?
    // ΔP represents the Impulse during a time of Δt
    if (deltaP == 0) {
        FLOPS(0, 4, 1, 0, complete_function, before_loop);
        deltaP = 0.5 * (1.0 + e_b) * M * fabs(velocity_diff_y) / (double)(N);
    }

    FLOPS(0, 2, 1, 0, complete_function, before_loop);
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

    while (__builtin_expect(velocity_diff_y < 0 || total_work < work_required, true)) {
        BRANCH(11);
        START_PROFILE(impulse);

        // Impulse Calculation
        if (__builtin_expect(ball_ball_contact_point_magnitude < 1e-16, false)) {
            BRANCH(0);
            deltaP_1 = 0;
            deltaP_2 = 0;
            deltaP_x_1 = 0;
            deltaP_y_1 = 0;
            deltaP_x_2 = 0;
            deltaP_y_2 = 0;
        } else {
            BRANCH(1);
            FLOPS(1, 2, 1, 0, complete_function, impulse);
            deltaP_1 = -u_b * deltaP * contact_point_velocity_x / ball_ball_contact_point_magnitude;
            if(__builtin_expect(fabs(contact_point_velocity_z) < 1e-16, false)) {
                BRANCH(2);
                deltaP_2 = 0;
                deltaP_x_1 = 0;
                deltaP_y_1 = 0;
                deltaP_x_2 = 0;
                deltaP_y_2 = 0;
            } else {
                BRANCH(3);
                FLOPS(1, 2, 1, 0, complete_function, impulse);
                deltaP_2 = -u_b * deltaP * contact_point_velocity_z / ball_ball_contact_point_magnitude;

                if(deltaP_2 > 0) {
                    BRANCH(4);
                    deltaP_x_1 = 0;
                    deltaP_y_1 = 0;

                    // TODO: probably best to check for some tolerance
                    if(__builtin_expect(surface_velocity_magnitude_2_sqrd == 0.0, false)) {
                        BRANCH(5);
                        deltaP_x_2 = 0;
                        deltaP_y_2 = 0;
                    } else {
                        BRANCH(6);
                        FLOPS(2, 4, 2, 1, complete_function, impulse);
                        double surface_velocity_magnitude_2 = sqrt(surface_velocity_magnitude_2_sqrd);
                        deltaP_x_2 = -u_s2 * (surface_velocity_x_2 / surface_velocity_magnitude_2) * deltaP_2;
                        deltaP_y_2 = -u_s2 * (surface_velocity_y_2 / surface_velocity_magnitude_2) * deltaP_2;
                    }
                } else {
                    BRANCH(7);
                    deltaP_x_2 = 0;
                    deltaP_y_2 = 0;
                    if(__builtin_expect(surface_velocity_magnitude_1_sqrd == 0.0, false)) {
                        BRANCH(8);
                        deltaP_x_1 = 0;
                        deltaP_y_1 = 0;
                    } else {
                        BRANCH(9);
                        FLOPS(0, 4, 2, 1, complete_function, impulse);
                        double surface_velocity_magnitude_1 = sqrt(surface_velocity_magnitude_1_sqrd);
                        deltaP_x_1 = u_s1 * (surface_velocity_x_1 / surface_velocity_magnitude_1) * deltaP_2;
                        deltaP_y_1 = u_s1 * (surface_velocity_y_1 / surface_velocity_magnitude_1) * deltaP_2;
                    }
                }

            }
        }

        END_PROFILE(impulse);
        START_PROFILE(delta);

        FLOPS(6, 0, 4, 0, complete_function, delta);
        // Velocity changes
        double velocity_change_x_1 = (deltaP_1 + deltaP_x_1) / M;
        double velocity_change_y_1 = (-deltaP + deltaP_y_1) / M;
        double velocity_change_x_2 = (-deltaP_1 + deltaP_x_2) / M;
        double velocity_change_y_2 = (deltaP + deltaP_y_2) / M;

        FLOPS(4, 0, 0, 0, complete_function, delta);
        // Update velocities
        local_velocity_x_1 += velocity_change_x_1;
        local_velocity_y_1 += velocity_change_y_1;
        local_velocity_x_2 += velocity_change_x_2;
        local_velocity_y_2 += velocity_change_y_2;

        FLOPS(6, 6, 0, 0, complete_function, delta);
        // Angular velocity changes
        double delta_angular_velocity_x_1 = C * (deltaP_2 + deltaP_y_1);
        double delta_angular_velocity_y_1 = C * (-deltaP_x_1);
        double delta_angular_velocity_z_1 = C * (-deltaP_1);
        double delta_angular_velocity_x_2 = C * (deltaP_2 + deltaP_y_2);
        double delta_angular_velocity_y_2 = C * (-deltaP_x_2);
        double delta_angular_velocity_z_2 = C * (-deltaP_1);

        FLOPS(6, 0, 0, 0, complete_function, delta);
        // Update Angular Velocities
        local_angular_velocity_x_1 += delta_angular_velocity_x_1;
        local_angular_velocity_y_1 += delta_angular_velocity_y_1;
        local_angular_velocity_z_1 += delta_angular_velocity_z_1;

        local_angular_velocity_x_2 += delta_angular_velocity_x_2;
        local_angular_velocity_y_2 += delta_angular_velocity_y_2;
        local_angular_velocity_z_2 += delta_angular_velocity_z_2;

        END_PROFILE(delta);
        START_PROFILE(velocity);

        FLOPS(4, 4, 0, 0, complete_function, velocity);
        // update ball-table slips
        surface_velocity_x_1 = local_velocity_x_1 + R * local_angular_velocity_y_1;
        surface_velocity_y_1 = local_velocity_y_1 - R * local_angular_velocity_x_1;
        surface_velocity_x_2 = local_velocity_x_2 + R * local_angular_velocity_y_2;
        surface_velocity_y_2 = local_velocity_y_2 - R * local_angular_velocity_x_2;

        FLOPS(2, 4, 0, 0, complete_function, velocity);
        surface_velocity_magnitude_1_sqrd = (surface_velocity_x_1 * surface_velocity_x_1 + surface_velocity_y_1 * surface_velocity_y_1);
        surface_velocity_magnitude_2_sqrd = (surface_velocity_x_2 * surface_velocity_x_2 + surface_velocity_y_2 * surface_velocity_y_2);

        FLOPS(5, 4, 0, 1, complete_function, velocity);
        // update ball-ball slip:
        contact_point_velocity_x = local_velocity_x_1 - local_velocity_x_2 - R * (local_angular_velocity_z_1 + local_angular_velocity_z_2);
        contact_point_velocity_z = R * (local_angular_velocity_x_1 + local_angular_velocity_x_2);
        ball_ball_contact_point_magnitude = sqrt(contact_point_velocity_x * contact_point_velocity_x + contact_point_velocity_z * contact_point_velocity_z);

        FLOPS(3, 2, 0, 0, complete_function, velocity);
        // Update work and check compression phase
        double velocity_diff_y_temp = velocity_diff_y;
        velocity_diff_y = local_velocity_y_2 - local_velocity_y_1;
        total_work += 0.5 * deltaP * fabs(velocity_diff_y_temp + velocity_diff_y);

        if (__builtin_expect(work_compression == 0 && velocity_diff_y > 0, false)) {
            BRANCH(10);
            work_compression = total_work;
            FLOPS(1, 2, 0, 0, complete_function, velocity);
            work_required = (1.0 + e_b * e_b) * work_compression;
        }

        END_PROFILE(velocity);
    }

    START_PROFILE(after_loop);

    MEMORY(4, complete_function, after_loop);
    FLOPS(2, 4, 0, 0, complete_function, after_loop);
    rvw1_result[3] = local_velocity_x_1 * forward[1] + local_velocity_y_1 * forward[0];
    rvw2_result[3] = local_velocity_x_2 * forward[1] + local_velocity_y_2 * forward[0];
    FLOPS(2, 4, 0, 0, complete_function, after_loop);
    rvw1_result[6] = local_angular_velocity_x_1 * forward[1] + local_angular_velocity_y_1 * forward[0];
    rvw2_result[6] = local_angular_velocity_x_2 * forward[1] + local_angular_velocity_y_2 * forward[0];

    MEMORY(4, complete_function, after_loop);
    FLOPS(2, 4, 0, 0, complete_function, after_loop);
    rvw1_result[4] = local_velocity_x_1 * forward[2] + local_velocity_y_1 * forward[1];
    rvw2_result[4] = local_velocity_x_2 * forward[2] + local_velocity_y_2 * forward[1];
    FLOPS(2, 4, 0, 0, complete_function, after_loop);
    rvw1_result[7] = local_angular_velocity_x_1 * forward[2] + local_angular_velocity_y_1 * forward[1];
    rvw2_result[7] = local_angular_velocity_x_2 * forward[2] + local_angular_velocity_y_2 * forward[1];

    MEMORY(4, complete_function, after_loop);
    FLOPS(0, 2, 0, 0, complete_function, after_loop);
    rvw1_result[5] = 0.0;
    rvw2_result[5] = 0.0;
    rvw1_result[8] = local_angular_velocity_z_1;
    rvw2_result[8] = local_angular_velocity_z_2;

    END_PROFILE(after_loop);
    END_PROFILE(complete_function);
}

// https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html
DLL_EXPORT void simd_collide_balls(double* rvw1, double* rvw2, float R, float M, float u_s1, float u_s2, float u_b, float e_b, float deltaP, int N, double* rvw1_result, double* rvw2_result, Profile* profiles, Branch* branches) {

    #ifdef PROFILE
        Profile* complete_function = &profiles[0];
        Profile* before_loop = &profiles[1];
        Profile* impulse = &profiles[2];
        Profile* delta = &profiles[3];
        Profile* velocity = &profiles[4];
        Profile* after_loop = &profiles[5];
    #endif

    START_PROFILE(complete_function);
    START_PROFILE(before_loop);

    // Altough the memory is not really loaded here, assuming its only compulsary misses
    MEMORY(18, complete_function, before_loop);
    double* translation_1 = rvw1;
    double* velocity_1 = &rvw1[3];
    double* angular_velocity_1 = &rvw1[6];

    double* translation_2 = rvw2;
    double* velocity_2 = &rvw2[3];
    double* angular_velocity_2 = &rvw2[6];

    FLOPS(3, 0, 0, 0, complete_function, before_loop);
    double forward[3]; // Forward from ball 1 to ball 2, normalized, forard[2] will always be zero
    forward[0] = translation_2[0] - translation_1[0];
    forward[1] = translation_2[1] - translation_1[1];

    FLOPS(2, 3, 0, 0, complete_function, before_loop);
    double offset_mag = forward[0] * forward[0] + forward[1] * forward[1] ;

    FLOPS(0, 0, 0, 1, complete_function, before_loop);
    offset_mag = sqrt(offset_mag);

    FLOPS(0, 0, 3, 0, complete_function, before_loop);
    forward[0] = forward[0] / offset_mag;
    forward[1] = forward[1] / offset_mag;

    FLOPS(1, 0, 0, 0, complete_function, before_loop);
    forward[2] = -forward[0]; // This is the same as right[1]

    // From here on, it is assumed that the x axis is the right axis and y axis is the forward axis
    // Transform velocities to local frame
    FLOPS(2, 4, 0, 0, complete_function, before_loop);
    double _local_velocity_x_1 = velocity_1[0] * forward[1] + velocity_1[1] * forward[2];
    double _local_velocity_x_2 = velocity_2[0] * forward[1] + velocity_2[1] * forward[2];

    FLOPS(2, 4, 0, 0, complete_function, before_loop);
    double _local_velocity_y_1 = velocity_1[0] * forward[0] + velocity_1[1] * forward[1];
    double _local_velocity_y_2 = velocity_2[0] * forward[0] + velocity_2[1] * forward[1];


    // Transform angular velocities into local frame

    FLOPS(2, 4, 0, 0, complete_function, before_loop);
    double _local_angular_velocity_x_1 = angular_velocity_1[0] * forward[1] + angular_velocity_1[1] * forward[2];
    double _local_angular_velocity_x_2 = angular_velocity_2[0] * forward[1] + angular_velocity_2[1] * forward[2];

    FLOPS(2, 4, 0, 0, complete_function, before_loop);
    double _local_angular_velocity_y_1 = angular_velocity_1[0] * forward[0] + angular_velocity_1[1] * forward[1];
    double _local_angular_velocity_y_2 = angular_velocity_2[0] * forward[0] + angular_velocity_2[1] * forward[1];

    double _local_angular_velocity_z_1 = angular_velocity_1[2];
    double _local_angular_velocity_z_2 = angular_velocity_2[2];



    // SIMD Prep
    // [x1, y1, x2, y2]
    __m256d velocities = _mm256_set_pd(_local_velocity_y_2, _local_velocity_x_2, _local_velocity_y_1, _local_velocity_x_1);

    // [y1, x1, y2, x2] !!! ! Y is first such that we can skip reorderring before surface calculation
    __m256d angular = _mm256_set_pd(_local_angular_velocity_x_2, _local_angular_velocity_y_2, _local_angular_velocity_x_1, _local_angular_velocity_y_1);
    // [0, 0, wz1, wz2]
    __m256d angular_z = _mm256_set_pd(_local_angular_velocity_z_2, _local_angular_velocity_z_1, 0, 0);

    __m256d R_ALTERNATE_4 = _mm256_set_pd((double)(-R), (double)R, (double)(-R), R); // TODO: could use fm_addsub instead of this?
    __m256d R4 = _mm256_set1_pd((double)R);
    __m256d M4 = _mm256_set1_pd((double)M);
    // [x1, y1, x2, y2]
    __m256d surface_velocities = _mm256_fmadd_pd(R_ALTERNATE_4, angular, velocities);

    FLOPS(5, 4, 0, 1, complete_function, before_loop);


    //printf("\nC Contact Point Slide, Spin:\n");
    //printf("  Contact Point: u_ijC_xz_mag= %.6f\n", ball_ball_contact_point_magnitude);

    // deltaP is most likely always 0?
    // ΔP represents the Impulse during a time of Δt
    double velocity_diff_y = _local_velocity_y_2 - _local_velocity_y_1;
    if (deltaP == 0) {
        deltaP = 0.5 * (1.0 + e_b) * M * fabs(velocity_diff_y) / (double)(N);
    }

    __m256d deltaP4 = _mm256_set1_pd(deltaP);
    __m256d nub4 = _mm256_set1_pd(-u_b);

    double C = 5.0 / (2.0 * M * R);
    __m256d C4 = _mm256_set1_pd(C);
    __m256d NC4 = _mm256_set1_pd(-C);

    double total_work = 0; // Work done due to impulse force
    double work_required = INFINITY; // Total amount of work required before collision handling is complete
    double work_compression = 0;

    END_PROFILE(before_loop);

    // while (__builtin_expect(velocity_diff_y < 0 || total_work < work_required, true)) {

    while (__builtin_expect(velocity_diff_y < 0 || total_work < work_required, true)) {

        //[x1, wx1, x2, wx2]
        __m256d lower = _mm256_blend_pd(velocities, angular,  0b1010);
        // [x2, wx2, undefined, undefined]
        __m256d upper = _mm256_castpd128_pd256(_mm256_extractf128_pd(lower, 1));

        //[x1, wx1, undef, wz1]
        lower = _mm256_blend_pd(lower, _mm256_shuffle_pd(angular_z, angular_z, 0b0010),  0b1000);
        //[x2, wx2, undef, wz2]
        upper = _mm256_blend_pd(upper, angular_z,  0b1000);

        // [x1 - x2, wx1 + wx2, undefined, wz1 + wz2]
        __m256d sums = _mm256_addsub_pd(lower, upper);

        // [undefined, contact point z (velocity / angular?), undefiend, undefined]
        __m256d contact_point_z = _mm256_mul_pd(R4, sums);

        // [wz1 + wz2, undef, undef, undef]
        __m256d contact_point_x_prep = _mm256_permute_pd(_mm256_castpd128_pd256(_mm256_extractf128_pd(sums, 1)), 0b1011);
        // [contact_point_x, undef, undef, undef]
        __m256d contact_point_x = _mm256_fnmadd_pd(R4, contact_point_x_prep, sums);

        // [contact_point_x, contact_point_z, undef, undef]
        __m256d contact_point = _mm256_blend_pd(contact_point_x, contact_point_z, 0b1010);

        // double contact_point_velocity_x = _local_velocity_x_1 - _local_velocity_x_2 - R * (_local_angular_velocity_z_1 + local_angular_velocity_z_2);
        // double contact_point_velocity_z = R * (_local_angular_velocity_x_1 + _local_angular_velocity_x_2);

        __m256d surf_sqrd = _mm256_mul_pd(surface_velocities, surface_velocities);
        __m256d contact_point_sqrd = _mm256_mul_pd(contact_point, contact_point);

        // [surfx1 * surfx1, contz * contz, surfx2*surfx2, undef]
        __m256d sqrd_lhs = _mm256_blend_pd(surf_sqrd, contact_point_sqrd, 0b0010);

        // [surfy1 * surfy1, undef, surfy2*surfy2, undef]
        __m256d sqrd_rhs = _mm256_shuffle_pd(surf_sqrd, surf_sqrd, 0b1111);

        // [surfy1 * surfy1, contx * contx, surfy2*surfy2, undef]
        sqrd_rhs = _mm256_blend_pd(sqrd_rhs, _mm256_shuffle_pd(contact_point_sqrd, contact_point_sqrd, 0b1000), 0b0010);

        // [surfx1 * surfx1 + surfy1*surfy1, contx*contx+contz*contz, surfx2 * surfx2 + surfy2*surfy2, undef]
        surf_sqrd = _mm256_add_pd(sqrd_lhs, sqrd_rhs);

        // [surf 1 magnitude, contact point magnitude, surf 2 magnitude]
        __m256d sqrts = _mm256_sqrt_pd(surf_sqrd);

        BRANCH(11);
        START_PROFILE(impulse);

        __m256d deltaP_12 = _mm256_div_pd(contact_point, _mm256_shuffle_pd(sqrts, sqrts, 0b1011));

        // [deltaP_1, deltaP_2, undef, undef]
        deltaP_12 =  _mm256_mul_pd(nub4, _mm256_mul_pd(deltaP4, deltaP_12));

        __m256d surf_norm = _mm256_div_pd(surface_velocities, _mm256_shuffle_pd(sqrts, sqrts, 0b0000));

        __m256d u_s4 = _mm256_set_pd(-u_s2, -u_s2, u_s1, u_s1);

        // [deltaP_2 * 4]
        __m256d deltaP_2_4 = _mm256_permute4x64_pd(deltaP_12, 0b01010101);
        // [deltaP_x_1, deltaP_y_1, deltaP_x_2, deltaP_y_2]
        __m256d deltaP_xy12 = _mm256_mul_pd(u_s4, _mm256_mul_pd(surf_norm, deltaP_2_4));

        // _mm256_set_pd(0.0, deltaP_2_4, fabs(contact_point_velocity_z), ball_ball_contact_point_magnitude);
        __m256d impulse_rhs = _mm256_setzero_pd();
        impulse_rhs = _mm256_blend_pd(impulse_rhs, deltaP_2_4, 0b0100);

        // fabs
        __m256d abs_mask = _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFFFFFFFFFFFFFF));
        __m256d abs_contact = _mm256_and_pd(contact_point, abs_mask);
        impulse_rhs = _mm256_blend_pd(impulse_rhs, abs_contact, 0b0010);
        impulse_rhs = _mm256_blend_pd(impulse_rhs, _mm256_shuffle_pd(sqrts, sqrts, 0b0011), 0b0001);


        // _mm256_set_pd(0.0, 0.0, 1e-16, 1e-16);
        __m256d impulse_lhs = _mm256_set_pd(0.0, 0.0, 1e-16, 1e-16);

        // [ball_ball_contact_point_magnitude > 1e-16, fabs(contact_point_velocity_z) > 1e-16, deltaP_2 > 0, undef]
        __m256d impulse_mask = _mm256_cmp_pd(impulse_lhs, impulse_rhs, _CMP_LT_OQ);
        // [surface_vel_mag1 != 0, undef, surface_vel_mag2 != 0, undef]
        __m256d impulse_mask_2 = _mm256_cmp_pd(sqrts, _mm256_setzero_pd(), _CMP_NEQ_OQ);
        // [surface_vel_mag1 != 0 ** 2, surface_vel_mag2 != 0**2]
        impulse_mask_2 = _mm256_shuffle_pd(impulse_mask_2, impulse_mask_2, 0b0000);

        // Expensive part? 9 cycles...
        __m256d contact_mag_mask = _mm256_permute4x64_pd(impulse_mask, 0b00000000);
        __m256d contact_z_mask = _mm256_permute4x64_pd(impulse_mask, 0b01010101);
        __m256d deltaP_2_mask = _mm256_permute4x64_pd(impulse_mask, 0b10101010);

        __m256d flip_mask = _mm256_castsi256_pd(_mm256_set_epi64x(0, 0, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF));
        // [deltaP_2 <= 0, deltaP_2 <= 0, deltaP_2 > 0, deltaP_2 > 0]
        deltaP_2_mask = _mm256_xor_pd(deltaP_2_mask, flip_mask);

        __m256d deltaP_xy_12_mask = _mm256_and_pd(_mm256_and_pd(deltaP_2_mask, impulse_mask_2), _mm256_and_pd(contact_mag_mask, contact_z_mask)) ;
        deltaP_xy12 = _mm256_and_pd(deltaP_xy12, deltaP_xy_12_mask);

        deltaP_12 = _mm256_and_pd(deltaP_12, contact_mag_mask);
        __m256d deltaP_12_mask = _mm256_castsi256_pd(_mm256_set_epi64x(0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0, 0xFFFFFFFFFFFFFFFF));
        deltaP_12 = _mm256_and_pd(deltaP_12, _mm256_or_pd(contact_z_mask, deltaP_12_mask));

        END_PROFILE(impulse);
        START_PROFILE(delta);

        FLOPS(6, 0, 4, 0, complete_function, delta);

        // Velocity changes
        __m256d deltaP_1_4 = _mm256_permute4x64_pd(deltaP_12, 0);
        // [deltaP_1, deltaP_1, deltaP_2 deltaP_2]
        __m256d deltaP1_deltaP_x1 = _mm256_blend_pd(deltaP4, deltaP_1_4, 0b0101);
        __m256d delta_velocites = _mm256_fmadd_pd(_mm256_set_pd(1.0, -1.0, -1.0, 1.0), deltaP1_deltaP_x1, deltaP_xy12);
        velocities = _mm256_add_pd(velocities, _mm256_div_pd(delta_velocites, M4));

        FLOPS(6, 6, 0, 0, complete_function, delta);

        // [y1, x1, y2, x2] !!! ! Y is first such that we can skip reorderring before surface calculation
        // cant reuse previous, because it may have been set to 0 in rare cases
        deltaP_2_4 = _mm256_permute4x64_pd(deltaP_12, 0b01010101);
        __m256d deltaP_2_0 = _mm256_blend_pd(deltaP_2_4, _mm256_setzero_pd(), 0b0101);
        __m256d delta_angular = _mm256_fmadd_pd(_mm256_set_pd(1.0, -1.0, 1.0, -1.0), deltaP_xy12, deltaP_2_0);
        // __m256d delta_angular = _mm256_set_pd(deltaP_2 + deltaP_y_2, -deltaP_x_2, deltaP_2 + deltaP_y_1, -deltaP_x_1);
        angular = _mm256_fmadd_pd(C4, delta_angular, angular);

        FLOPS(6, 0, 0, 0, complete_function, delta);
        angular_z = _mm256_fmadd_pd(NC4, deltaP_1_4, angular_z);

        END_PROFILE(delta);
        START_PROFILE(velocity);

        FLOPS(4, 4, 0, 0, complete_function, velocity);
        surface_velocities = _mm256_fmadd_pd(R_ALTERNATE_4, angular, velocities);

        double _local_velocity_y_1 = ((double*)&velocities)[1];
        double _local_velocity_y_2 = ((double*)&velocities)[3];

        FLOPS(3, 2, 0, 0, complete_function, velocity);
        // Update work and check compression phase
        double velocity_diff_y_temp = velocity_diff_y;
        velocity_diff_y = _local_velocity_y_2 - _local_velocity_y_1;
        total_work += 0.5 * deltaP * fabs(velocity_diff_y_temp + velocity_diff_y);

        if (__builtin_expect(work_compression == 0 && velocity_diff_y > 0, false)) {
            BRANCH(10);
            work_compression = total_work;
            FLOPS(1, 2, 0, 0, complete_function, velocity);
            work_required = (1.0 + e_b * e_b) * work_compression;
        }

        END_PROFILE(velocity);
    }

    START_PROFILE(after_loop);

    double test[4];
    _mm256_storeu_pd(test, velocities);
    _local_velocity_x_1 = test[0];
    _local_velocity_y_1 = test[1];
    _local_velocity_x_2 = test[2];
    _local_velocity_y_2 = test[3];

    double test_a[4];
    _mm256_storeu_pd(test, angular);
    _local_angular_velocity_y_1 = test[0];
    _local_angular_velocity_x_1 = test[1];
    _local_angular_velocity_y_2 = test[2];
    _local_angular_velocity_x_2 = test[3];

    MEMORY(4, complete_function, after_loop);
    FLOPS(2, 4, 0, 0, complete_function, after_loop);
    rvw1_result[3] = _local_velocity_x_1 * forward[1] + _local_velocity_y_1 * forward[0];
    rvw2_result[3] = _local_velocity_x_2 * forward[1] + _local_velocity_y_2 * forward[0];
    FLOPS(2, 4, 0, 0, complete_function, after_loop);
    rvw1_result[6] = _local_angular_velocity_x_1 * forward[1] + _local_angular_velocity_y_1 * forward[0];
    rvw2_result[6] = _local_angular_velocity_x_2 * forward[1] + _local_angular_velocity_y_2 * forward[0];

    MEMORY(4, complete_function, after_loop);
    FLOPS(2, 4, 0, 0, complete_function, after_loop);
    rvw1_result[4] = _local_velocity_x_1 * forward[2] + _local_velocity_y_1 * forward[1];
    rvw2_result[4] = _local_velocity_x_2 * forward[2] + _local_velocity_y_2 * forward[1];
    FLOPS(2, 4, 0, 0, complete_function, after_loop);
    rvw1_result[7] = _local_angular_velocity_x_1 * forward[2] + _local_angular_velocity_y_1 * forward[1];
    rvw2_result[7] = _local_angular_velocity_x_2 * forward[2] + _local_angular_velocity_y_2 * forward[1];

    MEMORY(4, complete_function, after_loop);
    FLOPS(0, 2, 0, 0, complete_function, after_loop);
    rvw1_result[5] = 0.0;
    rvw2_result[5] = 0.0;


    double test_x[4];
    _mm256_storeu_pd(test, angular_z);
    _local_angular_velocity_z_1 = test[2];
    _local_angular_velocity_z_2 = test[3];

    rvw1_result[8] = _local_angular_velocity_z_1;
    rvw2_result[8] = _local_angular_velocity_z_2;

    END_PROFILE(after_loop);
    END_PROFILE(complete_function);
}


DLL_EXPORT void improved_symmetry_collide_balls(double* rvw1, double* rvw2, float R, float M, float u_s1, float u_s2, float u_b, float e_b, float deltaP, int N, double* rvw1_result, double* rvw2_result, Profile* profiles, Branch* branches) {
    // Keep the original structure but optimize within it
    #ifdef PROFILE
        Profile* complete_function = &profiles[0];
        Profile* before_loop       = &profiles[1];
        Profile* impulse           = &profiles[2];
        Profile* delta             = &profiles[3];
        Profile* velocity          = &profiles[4];
        Profile* after_loop        = &profiles[5];
    #endif
    START_PROFILE(complete_function);
    START_PROFILE(before_loop);

    MEMORY(18, complete_function, before_loop);
    double* translation_1     = get_displacement(rvw1);
    double* velocity_1        = get_velocity(rvw1);
    double* angular_velocity_1= get_angular_velocity(rvw1);

    double* translation_2     = get_displacement(rvw2);
    double* velocity_2        = get_velocity(rvw2);
    double* angular_velocity_2= get_angular_velocity(rvw2);

    // Keep the same scalar calculations
    double invM = 1.0 / M;
    double invR = 1.0 / R;
    double C = 5.0 * invM * invR * 0.5;

    // Calculate coordinate system exactly as before
    double offset[3];
    subV3(translation_2, translation_1, offset);

    double offset_mag_sqrd = dotV3(offset, offset);
    double offset_inv_mag = 1.0 / sqrt(offset_mag_sqrd);
    double forward[3];
    forward[0] = offset[0] * offset_inv_mag;
    forward[1] = offset[1] * offset_inv_mag;
    forward[2] = offset[2] * offset_inv_mag;

    double up[3] = {0.0, 0.0, 1.0};
    double right[3];
    crossV3(forward, up, right);

    // Keep separate velocities but optimize calculations
    double local_velocity_x_1 = dotV3(velocity_1, right);
    double local_velocity_y_1 = dotV3(velocity_1, forward);
    // Exploit symmetry: compute once and reuse for local coordinate calculations
    double velocities_2_dot_right = dotV3(velocity_2, right);
    double velocities_2_dot_forward = dotV3(velocity_2, forward);
    double local_velocity_x_2 = velocities_2_dot_right;
    double local_velocity_y_2 = velocities_2_dot_forward;

    // Same approach for angular velocities
    double angular_1_dot_right = dotV3(angular_velocity_1, right);
    double angular_1_dot_forward = dotV3(angular_velocity_1, forward);
    double angular_1_dot_up = dotV3(angular_velocity_1, up);
    
    double local_angular_velocity_x_1 = angular_1_dot_right;
    double local_angular_velocity_y_1 = angular_1_dot_forward;
    double local_angular_velocity_z_1 = angular_1_dot_up;

    double angular_2_dot_right = dotV3(angular_velocity_2, right);
    double angular_2_dot_forward = dotV3(angular_velocity_2, forward);  
    double angular_2_dot_up = dotV3(angular_velocity_2, up);
    
    double local_angular_velocity_x_2 = angular_2_dot_right;
    double local_angular_velocity_y_2 = angular_2_dot_forward;
    double local_angular_velocity_z_2 = angular_2_dot_up;

    // Keep FMA operations for surface velocities
    double surface_velocity_x_1 = fma(R, local_angular_velocity_y_1, local_velocity_x_1);
    double surface_velocity_y_1 = fma(-R, local_angular_velocity_x_1, local_velocity_y_1);
    double surface_velocity_x_2 = fma(R, local_angular_velocity_y_2, local_velocity_x_2);
    double surface_velocity_y_2 = fma(-R, local_angular_velocity_x_2, local_velocity_y_2);

    // Calculate contact point slip as before
    double contact_point_velocity_x = local_velocity_x_1 - local_velocity_x_2
                                   - R*(local_angular_velocity_z_1 + local_angular_velocity_z_2);
    double contact_point_velocity_z = R*(local_angular_velocity_x_1 + local_angular_velocity_x_2);
    double contact_inv_mag = 1.0 / sqrt(contact_point_velocity_x*contact_point_velocity_x +
                                        contact_point_velocity_z*contact_point_velocity_z);
    double ball_ball_contact_point_magnitude = 1.0 / contact_inv_mag;

    // Use original impulse calculation
    double velocity_diff_y = local_velocity_y_2 - local_velocity_y_1;
    if (unlikely(deltaP == 0.0f)) {
        deltaP = 0.5 * (1.0 + e_b) * M * fabs(velocity_diff_y) / (double)N;
    }

    // Keep original bookkeeping variables
    double total_work = 0.0;
    double work_required = INFINITY;
    double work_compression = 0.0;
    double deltaP_1 = deltaP, deltaP_2 = deltaP;
    double deltaP_x_1 = 0, deltaP_y_1 = 0, deltaP_x_2 = 0, deltaP_y_2 = 0;

    END_PROFILE(before_loop);
    
    // Use original loop with branching
    while (velocity_diff_y < 0.0 || total_work < work_required) {
        START_PROFILE(impulse);

        // Keep the same branching logic
        if (unlikely(ball_ball_contact_point_magnitude < 1e-16)) {
            BRANCH(0);
            deltaP_1 = deltaP_2 = 0.0;
            deltaP_x_1 = deltaP_y_1 = deltaP_x_2 = deltaP_y_2 = 0.0;
        } else {
            BRANCH(1);
            double inv_mag = contact_inv_mag;
            deltaP_1 = -u_b * deltaP * contact_point_velocity_x * inv_mag;

            if (unlikely(fabs(contact_point_velocity_z) < 1e-16)) {
                BRANCH(2);
                deltaP_2 = deltaP_x_1 = deltaP_y_1 = deltaP_x_2 = deltaP_y_2 = 0.0;
            } else {
                BRANCH(3);
                deltaP_2 = -u_b * deltaP * contact_point_velocity_z * inv_mag;

                if (deltaP_2 > 0.0) {
                    BRANCH(4);
                    deltaP_x_1 = deltaP_y_1 = 0.0;

                    if (unlikely(surface_velocity_x_2 == 0.0 && surface_velocity_y_2 == 0)) {
                        BRANCH(5);
                        deltaP_x_2 = deltaP_y_2 = 0.0;
                    } else {
                        BRANCH(6);
                        double inv_sv2 = 1.0 / sqrt(surface_velocity_x_2 * surface_velocity_x_2 + 
                                                  surface_velocity_y_2 * surface_velocity_y_2);
                        deltaP_x_2 = -u_s2 * surface_velocity_x_2 * inv_sv2 * deltaP_2;
                        deltaP_y_2 = -u_s2 * surface_velocity_y_2 * inv_sv2 * deltaP_2;
                    }
                } else {
                    BRANCH(7);
                    deltaP_x_2 = deltaP_y_2 = 0.0;
                    if (unlikely(surface_velocity_x_1 == 0.0 && surface_velocity_y_1 == 0)) {
                        BRANCH(8);
                        deltaP_x_1 = deltaP_y_1 = 0.0;
                    } else {
                        BRANCH(9);
                        double inv_sv1 = 1.0 / sqrt(surface_velocity_x_1*surface_velocity_x_1 + 
                                                  surface_velocity_y_1*surface_velocity_y_1);
                        deltaP_x_1 = u_s1 * surface_velocity_x_1 * inv_sv1 * deltaP_2;
                        deltaP_y_1 = u_s1 * surface_velocity_y_1 * inv_sv1 * deltaP_2;
                    }
                }
            }
        }
        END_PROFILE(impulse);

        // Keep the same velocity update logic
        START_PROFILE(delta);
        
        // Calculate velocity changes
        double velocity_change_x_1 = (deltaP_1 + deltaP_x_1) * invM;
        double velocity_change_y_1 = (-deltaP + deltaP_y_1) * invM;
        double velocity_change_x_2 = (-deltaP_1 + deltaP_x_2) * invM;
        double velocity_change_y_2 = (deltaP + deltaP_y_2) * invM;

        // Update velocities
        local_velocity_x_1 += velocity_change_x_1;
        local_velocity_y_1 += velocity_change_y_1;
        local_velocity_x_2 += velocity_change_x_2;
        local_velocity_y_2 += velocity_change_y_2;

        // Update angular velocities
        local_angular_velocity_x_1 += C * (deltaP_2 + deltaP_y_1);
        local_angular_velocity_y_1 += C * (-deltaP_x_1);
        local_angular_velocity_z_1 += C * (-deltaP_1);

        local_angular_velocity_x_2 += C * (deltaP_2 + deltaP_y_2);
        local_angular_velocity_y_2 += C * (-deltaP_x_2);
        local_angular_velocity_z_2 += C * (-deltaP_1);

        END_PROFILE(delta);

        // Recompute surface velocities
        START_PROFILE(velocity);
        
        surface_velocity_x_1 = fma(R, local_angular_velocity_y_1, local_velocity_x_1);
        surface_velocity_y_1 = fma(-R, local_angular_velocity_x_1, local_velocity_y_1);
        surface_velocity_x_2 = fma(R, local_angular_velocity_y_2, local_velocity_x_2);
        surface_velocity_y_2 = fma(-R, local_angular_velocity_x_2, local_velocity_y_2);

        // Update contact point velocity
        contact_point_velocity_x = local_velocity_x_1 - local_velocity_x_2
                                 - (local_angular_velocity_z_1 + local_angular_velocity_z_2)*R;
        contact_point_velocity_z = (local_angular_velocity_x_1 + local_angular_velocity_x_2)*R;
        
        // Use original approximation formula for consistency
        contact_inv_mag *= 0.5 * (3.0 - (contact_point_velocity_x * contact_point_velocity_x +
                           contact_point_velocity_z * contact_point_velocity_z) * 
                           contact_inv_mag * contact_inv_mag);
        ball_ball_contact_point_magnitude = 1.0 / contact_inv_mag;

        // Work calculation
        double velocity_diff_y_prev = velocity_diff_y;
        velocity_diff_y = local_velocity_y_2 - local_velocity_y_1;
        total_work += 0.5 * deltaP * fabs(velocity_diff_y_prev + velocity_diff_y);

        if (work_compression == 0.0 && velocity_diff_y > 0.0) {
            work_compression = total_work;
            work_required = (1.0 + e_b * e_b) * work_compression;
        }

        END_PROFILE(velocity);
    }

    // Transform back to global coordinates
    START_PROFILE(after_loop);
    for (int i = 0; i < 3; ++i) {
        rvw1_result[i+3] = local_velocity_x_1*right[i] + local_velocity_y_1*forward[i];
        rvw2_result[i+3] = local_velocity_x_2*right[i] + local_velocity_y_2*forward[i];

        if (i < 2) {
            rvw1_result[i+6] = local_angular_velocity_x_1*right[i] + local_angular_velocity_y_1*forward[i];
            rvw2_result[i+6] = local_angular_velocity_x_2*right[i] + local_angular_velocity_y_2*forward[i];
        } else {
            rvw1_result[i+6] = local_angular_velocity_z_1;
            rvw2_result[i+6] = local_angular_velocity_z_2;
        }
    }
    END_PROFILE(after_loop);
    END_PROFILE(complete_function);
}

// https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html
DLL_EXPORT void simd_collide_ball_2(double* rvw1, double* rvw2, float R, float M, float u_s1, float u_s2, float u_b, float e_b, float deltaP, int N, double* rvw1_result, double* rvw2_result, Profile* profiles, Branch* branches) {

    #ifdef PROFILE
        Profile* complete_function = &profiles[0];
        Profile* before_loop = &profiles[1];
        Profile* impulse = &profiles[2];
        Profile* delta = &profiles[3];
        Profile* velocity = &profiles[4];
        Profile* after_loop = &profiles[5];
    #endif

    START_PROFILE(complete_function);
    START_PROFILE(before_loop);

    // Altough the memory is not really loaded here, assuming its only compulsary misses
    MEMORY(18, complete_function, before_loop);
    double* translation_1 = rvw1;
    double* velocity_1 = &rvw1[3];
    double* angular_velocity_1 = &rvw1[6];

    double* translation_2 = rvw2;
    double* velocity_2 = &rvw2[3];
    double* angular_velocity_2 = &rvw2[6];

    FLOPS(3, 0, 0, 0, complete_function, before_loop);
    double forward[3]; // Forward from ball 1 to ball 2, normalized, forard[2] will always be zero
    forward[0] = translation_2[0] - translation_1[0];
    forward[1] = translation_2[1] - translation_1[1];

    FLOPS(2, 3, 0, 0, complete_function, before_loop);
    double offset_mag = forward[0] * forward[0] + forward[1] * forward[1] ;

    FLOPS(0, 0, 0, 1, complete_function, before_loop);
    offset_mag = sqrt(offset_mag);

    FLOPS(0, 0, 3, 0, complete_function, before_loop);
    forward[0] = forward[0] / offset_mag;
    forward[1] = forward[1] / offset_mag;

    FLOPS(1, 0, 0, 0, complete_function, before_loop);
    forward[2] = -forward[0]; // This is the same as right[1]

    // From here on, it is assumed that the x axis is the right axis and y axis is the forward axis
    // Transform velocities to local frame
    FLOPS(2, 4, 0, 0, complete_function, before_loop);
    double _local_velocity_x_1 = velocity_1[0] * forward[1] + velocity_1[1] * forward[2];
    double _local_velocity_x_2 = velocity_2[0] * forward[1] + velocity_2[1] * forward[2];

    FLOPS(2, 4, 0, 0, complete_function, before_loop);
    double _local_velocity_y_1 = velocity_1[0] * forward[0] + velocity_1[1] * forward[1];
    double _local_velocity_y_2 = velocity_2[0] * forward[0] + velocity_2[1] * forward[1];


    // Transform angular velocities into local frame

    FLOPS(2, 4, 0, 0, complete_function, before_loop);
    double _local_angular_velocity_x_1 = angular_velocity_1[0] * forward[1] + angular_velocity_1[1] * forward[2];
    double _local_angular_velocity_x_2 = angular_velocity_2[0] * forward[1] + angular_velocity_2[1] * forward[2];

    FLOPS(2, 4, 0, 0, complete_function, before_loop);
    double _local_angular_velocity_y_1 = angular_velocity_1[0] * forward[0] + angular_velocity_1[1] * forward[1];
    double _local_angular_velocity_y_2 = angular_velocity_2[0] * forward[0] + angular_velocity_2[1] * forward[1];

    double _local_angular_velocity_z_1 = angular_velocity_1[2];
    double _local_angular_velocity_z_2 = angular_velocity_2[2];



    // SIMD Prep
    // [x1, y1, x2, y2]
    __m256d velocities = _mm256_set_pd(_local_velocity_y_2, _local_velocity_x_2, _local_velocity_y_1, _local_velocity_x_1);

    // [y1, x1, y2, x2] !!! ! Y is first such that we can skip reorderring before surface calculation
    __m256d angular = _mm256_set_pd(_local_angular_velocity_x_2, _local_angular_velocity_y_2, _local_angular_velocity_x_1, _local_angular_velocity_y_1);
    // [0, 0, wz1, wz2]
    __m256d angular_z = _mm256_set_pd(_local_angular_velocity_z_2, _local_angular_velocity_z_1, 0, 0);

    __m256d R_ALTERNATE_4 = _mm256_set_pd((double)(-R), (double)R, (double)(-R), R); // TODO: could use fm_addsub instead of this?
    __m256d R4 = _mm256_set1_pd((double)R);
    __m256d M4 = _mm256_set1_pd((double)M);
    // [x1, y1, x2, y2]
    __m256d surface_velocities = _mm256_fmadd_pd(R_ALTERNATE_4, angular, velocities);

    FLOPS(5, 4, 0, 1, complete_function, before_loop);


    //printf("\nC Contact Point Slide, Spin:\n");
    //printf("  Contact Point: u_ijC_xz_mag= %.6f\n", ball_ball_contact_point_magnitude);

    // deltaP is most likely always 0?
    // ΔP represents the Impulse during a time of Δt
    double velocity_diff_y = _local_velocity_y_2 - _local_velocity_y_1;
    if (deltaP == 0) {
        deltaP = 0.5 * (1.0 + e_b) * M * fabs(velocity_diff_y) / (double)(N);
    }

    __m256d deltaP4 = _mm256_set1_pd(deltaP);
    __m256d nub4 = _mm256_set1_pd(-u_b);

    double C = 5.0 / (2.0 * M * R);
    __m256d C4 = _mm256_set1_pd(C);
    __m256d NC4 = _mm256_set1_pd(-C);

    double total_work = 0; // Work done due to impulse force
    double work_required = INFINITY; // Total amount of work required before collision handling is complete
    double work_compression = 0;

    END_PROFILE(before_loop);

    // while (__builtin_expect(velocity_diff_y < 0 || total_work < work_required, true)) {

    while (__builtin_expect(velocity_diff_y < 0 || total_work < work_required, true)) {

        //[x1, wx1, x2, wx2]
        __m256d lower = _mm256_blend_pd(velocities, angular,  0b1010);
        // [x2, wx2, undefined, undefined]
        __m256d upper = _mm256_castpd128_pd256(_mm256_extractf128_pd(lower, 1));

        //[x1, wx1, undef, wz1]
        lower = _mm256_blend_pd(lower, _mm256_shuffle_pd(angular_z, angular_z, 0b0010),  0b1000);
        //[x2, wx2, undef, wz2]
        upper = _mm256_blend_pd(upper, angular_z,  0b1000);

        // [x1 - x2, wx1 + wx2, undefined, wz1 + wz2]
        __m256d sums = _mm256_addsub_pd(lower, upper);

        // [undefined, contact point z (velocity / angular?), undefiend, undefined]
        __m256d contact_point_z = _mm256_mul_pd(R4, sums);

        // [wz1 + wz2, undef, undef, undef]
        __m256d contact_point_x_prep = _mm256_permute_pd(_mm256_castpd128_pd256(_mm256_extractf128_pd(sums, 1)), 0b1011);
        // [contact_point_x, undef, undef, undef]
        __m256d contact_point_x = _mm256_fnmadd_pd(R4, contact_point_x_prep, sums);

        // [contact_point_x, contact_point_z, undef, undef]
        __m256d contact_point = _mm256_blend_pd(contact_point_x, contact_point_z, 0b1010);

        // double contact_point_velocity_x = _local_velocity_x_1 - _local_velocity_x_2 - R * (_local_angular_velocity_z_1 + local_angular_velocity_z_2);
        // double contact_point_velocity_z = R * (_local_angular_velocity_x_1 + _local_angular_velocity_x_2);

        __m256d surf_sqrd = _mm256_mul_pd(surface_velocities, surface_velocities);
        __m256d contact_point_sqrd = _mm256_mul_pd(contact_point, contact_point);

        // [surfx1 * surfx1, contz * contz, surfx2*surfx2, undef]
        __m256d sqrd_lhs = _mm256_blend_pd(surf_sqrd, contact_point_sqrd, 0b0010);

        // [surfy1 * surfy1, undef, surfy2*surfy2, undef]
        __m256d sqrd_rhs = _mm256_shuffle_pd(surf_sqrd, surf_sqrd, 0b1111);

        // [surfy1 * surfy1, contx * contx, surfy2*surfy2, undef]
        sqrd_rhs = _mm256_blend_pd(sqrd_rhs, _mm256_shuffle_pd(contact_point_sqrd, contact_point_sqrd, 0b1000), 0b0010);

        // [surfx1 * surfx1 + surfy1*surfy1, contx*contx+contz*contz, surfx2 * surfx2 + surfy2*surfy2, undef]
        __m256d final_surf_sqrd = _mm256_add_pd(sqrd_lhs, sqrd_rhs);

        // [surf 1 magnitude, contact point magnitude, surf 2 magnitude]
        __m256d sqrts = _mm256_sqrt_pd(final_surf_sqrd);

        BRANCH(11);
        START_PROFILE(impulse);

        __m256d deltaP_12 = _mm256_div_pd(contact_point, _mm256_shuffle_pd(sqrts, sqrts, 0b1011));

        // [deltaP_1, deltaP_2, undef, undef]
        deltaP_12 =  _mm256_mul_pd(nub4, _mm256_mul_pd(deltaP4, deltaP_12));

        __m256d surf_norm = _mm256_div_pd(surface_velocities, _mm256_shuffle_pd(sqrts, sqrts, 0b0000));

        __m256d u_s4 = _mm256_set_pd(-u_s2, -u_s2, u_s1, u_s1);

        // [deltaP_2 * 4]
        __m256d deltaP_2_4 = _mm256_permute4x64_pd(deltaP_12, 0b01010101);
        // [deltaP_x_1, deltaP_y_1, deltaP_x_2, deltaP_y_2]
        __m256d deltaP_xy12 = _mm256_mul_pd(u_s4, _mm256_mul_pd(surf_norm, deltaP_2_4));

        // _mm256_set_pd(0.0, deltaP_2_4, fabs(contact_point_velocity_z), ball_ball_contact_point_magnitude);
        __m256d impulse_rhs = _mm256_setzero_pd();
        impulse_rhs = _mm256_blend_pd(impulse_rhs, deltaP_2_4, 0b0100);

        // fabs
        __m256d abs_mask = _mm256_castsi256_pd(_mm256_set1_epi64x(0x7FFFFFFFFFFFFFFF));
        __m256d abs_contact = _mm256_and_pd(contact_point, abs_mask);
        impulse_rhs = _mm256_blend_pd(impulse_rhs, abs_contact, 0b0010);
        impulse_rhs = _mm256_blend_pd(impulse_rhs, _mm256_shuffle_pd(sqrts, sqrts, 0b0011), 0b0001);


        // _mm256_set_pd(0.0, 0.0, 1e-16, 1e-16);
        __m256d impulse_lhs = _mm256_set_pd(0.0, 0.0, 1e-16, 1e-16);

        // [ball_ball_contact_point_magnitude > 1e-16, fabs(contact_point_velocity_z) > 1e-16, deltaP_2 > 0, undef]
        __m256d impulse_mask = _mm256_cmp_pd(impulse_lhs, impulse_rhs, _CMP_LT_OQ);
        // [surface_vel_mag1 != 0, undef, surface_vel_mag2 != 0, undef]
        __m256d impulse_mask_2 = _mm256_cmp_pd(sqrts, _mm256_setzero_pd(), _CMP_NEQ_OQ);
        // [surface_vel_mag1 != 0 ** 2, surface_vel_mag2 != 0**2]
        impulse_mask_2 = _mm256_shuffle_pd(impulse_mask_2, impulse_mask_2, 0b0000);

        // Expensive part? 9 cycles...
        __m256d contact_mag_mask = _mm256_permute4x64_pd(impulse_mask, 0b00000000);
        __m256d contact_z_mask = _mm256_permute4x64_pd(impulse_mask, 0b01010101);
        __m256d deltaP_2_mask = _mm256_permute4x64_pd(impulse_mask, 0b10101010);

        __m256d flip_mask = _mm256_castsi256_pd(_mm256_set_epi64x(0, 0, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF));
        // [deltaP_2 <= 0, deltaP_2 <= 0, deltaP_2 > 0, deltaP_2 > 0]
        deltaP_2_mask = _mm256_xor_pd(deltaP_2_mask, flip_mask);

        __m256d deltaP_xy_12_mask = _mm256_and_pd(_mm256_and_pd(deltaP_2_mask, impulse_mask_2), _mm256_and_pd(contact_mag_mask, contact_z_mask)) ;
        deltaP_xy12 = _mm256_and_pd(deltaP_xy12, deltaP_xy_12_mask);

        deltaP_12 = _mm256_and_pd(deltaP_12, contact_mag_mask);
        __m256d deltaP_12_mask = _mm256_castsi256_pd(_mm256_set_epi64x(0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0, 0xFFFFFFFFFFFFFFFF));
        deltaP_12 = _mm256_and_pd(deltaP_12, _mm256_or_pd(contact_z_mask, deltaP_12_mask));

        END_PROFILE(impulse);
        START_PROFILE(delta);

        FLOPS(6, 0, 4, 0, complete_function, delta);

        // Velocity changes
        __m256d deltaP_1_4 = _mm256_permute4x64_pd(deltaP_12, 0);
        // [deltaP_1, deltaP_1, deltaP_2 deltaP_2]
        __m256d deltaP1_deltaP_x1 = _mm256_blend_pd(deltaP4, deltaP_1_4, 0b0101);
        __m256d delta_velocites = _mm256_fmadd_pd(_mm256_set_pd(1.0, -1.0, -1.0, 1.0), deltaP1_deltaP_x1, deltaP_xy12);
        velocities = _mm256_add_pd(velocities, _mm256_div_pd(delta_velocites, M4));

        FLOPS(6, 6, 0, 0, complete_function, delta);

        // [y1, x1, y2, x2] !!! ! Y is first such that we can skip reorderring before surface calculation
        // cant reuse previous, because it may have been set to 0 in rare cases
        deltaP_2_4 = _mm256_permute4x64_pd(deltaP_12, 0b01010101);
        __m256d deltaP_2_0 = _mm256_blend_pd(deltaP_2_4, _mm256_setzero_pd(), 0b0101);
        __m256d delta_angular = _mm256_fmadd_pd(_mm256_set_pd(1.0, -1.0, 1.0, -1.0), deltaP_xy12, deltaP_2_0);
        // __m256d delta_angular = _mm256_set_pd(deltaP_2 + deltaP_y_2, -deltaP_x_2, deltaP_2 + deltaP_y_1, -deltaP_x_1);
        angular = _mm256_fmadd_pd(C4, delta_angular, angular);

        FLOPS(6, 0, 0, 0, complete_function, delta);
        angular_z = _mm256_fmadd_pd(NC4, deltaP_1_4, angular_z);

        END_PROFILE(delta);
        START_PROFILE(velocity);

        FLOPS(4, 4, 0, 0, complete_function, velocity);
        surface_velocities = _mm256_fmadd_pd(R_ALTERNATE_4, angular, velocities);

        // Extract y-components without spilling the register to memory
        // lanes 0-1: x1 | y1   (low 128)   lanes 2-3: x2 | y2 (high 128)
        __m128d low128  = _mm256_castpd256_pd128(velocities);          // x1 | y1
        __m128d high128 = _mm256_extractf128_pd(velocities, 1);        // x2 | y2

        double _local_velocity_y_1 = _mm_cvtsd_f64(_mm_unpackhi_pd(low128,  low128));
        double _local_velocity_y_2 = _mm_cvtsd_f64(_mm_unpackhi_pd(high128, high128));

        FLOPS(3, 2, 0, 0, complete_function, velocity);
        // Update work and check compression phase
        double velocity_diff_y_temp = velocity_diff_y;
        velocity_diff_y = _local_velocity_y_2 - _local_velocity_y_1;
        total_work += 0.5 * deltaP * fabs(velocity_diff_y_temp + velocity_diff_y);

        if (__builtin_expect(work_compression == 0 && velocity_diff_y > 0, false)) {
            BRANCH(10);
            work_compression = total_work;
            FLOPS(1, 2, 0, 0, complete_function, velocity);
            work_required = (1.0 + e_b * e_b) * work_compression;
        }

        END_PROFILE(velocity);
    }

    START_PROFILE(after_loop);

    double test[4];
    _mm256_storeu_pd(test, velocities);
    _local_velocity_x_1 = test[0];
    _local_velocity_y_1 = test[1];
    _local_velocity_x_2 = test[2];
    _local_velocity_y_2 = test[3];

    double test_a[4];
    _mm256_storeu_pd(test, angular);
    _local_angular_velocity_y_1 = test[0];
    _local_angular_velocity_x_1 = test[1];
    _local_angular_velocity_y_2 = test[2];
    _local_angular_velocity_x_2 = test[3];

    MEMORY(4, complete_function, after_loop);
    FLOPS(2, 4, 0, 0, complete_function, after_loop);
    rvw1_result[3] = _local_velocity_x_1 * forward[1] + _local_velocity_y_1 * forward[0];
    rvw2_result[3] = _local_velocity_x_2 * forward[1] + _local_velocity_y_2 * forward[0];
    FLOPS(2, 4, 0, 0, complete_function, after_loop);
    rvw1_result[6] = _local_angular_velocity_x_1 * forward[1] + _local_angular_velocity_y_1 * forward[0];
    rvw2_result[6] = _local_angular_velocity_x_2 * forward[1] + _local_angular_velocity_y_2 * forward[0];

    MEMORY(4, complete_function, after_loop);
    FLOPS(2, 4, 0, 0, complete_function, after_loop);
    rvw1_result[4] = _local_velocity_x_1 * forward[2] + _local_velocity_y_1 * forward[1];
    rvw2_result[4] = _local_velocity_x_2 * forward[2] + _local_velocity_y_2 * forward[1];
    FLOPS(2, 4, 0, 0, complete_function, after_loop);
    rvw1_result[7] = _local_angular_velocity_x_1 * forward[2] + _local_angular_velocity_y_1 * forward[1];
    rvw2_result[7] = _local_angular_velocity_x_2 * forward[2] + _local_angular_velocity_y_2 * forward[1];

    MEMORY(4, complete_function, after_loop);
    FLOPS(0, 2, 0, 0, complete_function, after_loop);
    rvw1_result[5] = 0.0;
    rvw2_result[5] = 0.0;


    double test_x[4];
    _mm256_storeu_pd(test, angular_z);
    _local_angular_velocity_z_1 = test[2];
    _local_angular_velocity_z_2 = test[3];

    rvw1_result[8] = _local_angular_velocity_z_1;
    rvw2_result[8] = _local_angular_velocity_z_2;

    END_PROFILE(after_loop);
    END_PROFILE(complete_function);
}