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

    double offset[3];
    subV3(translation_2, translation_1, offset);

    double offset_mag_sqrd = dotV3(offset, offset);
    double offset_mag = sqrt(offset_mag_sqrd);

    double forward[3]; // Forward from ball 1 to ball 2, normalized
    divV3(offset, offset_mag, forward);

    double Z_AXIS[3] = {0, 0, 1}; // Probably up axis?

    double right[3]; // Axis orthogonal to Z and forward
    crossV3(forward, Z_AXIS, right);

    printf("C %f %f %f\n", right[0], right[1], right[2]);
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
