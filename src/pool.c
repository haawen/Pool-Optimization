#include "pool.h"

#include <stdio.h>

void hello_world(const char* matrix_name, double* rvw) {

    printf("Received %s\n", matrix_name);
   for(int i = 0; i < 9; i++) {
       printf("%f ", rvw[i]);
       if((i + 1) % 3 == 0) printf("\n");
   }

}

/* Assuming rvw is row-major (passed from pooltool) */
float* get_displacement(BallState* ballState) {
    return ballState->rvw;
}

/* Assuming rvw is row-major (passed from pooltool) */
float* get_velocity(BallState* ballState) {
    return &ballState->rvw[3];
}

/* Assuming rvw is row-major (passed from pooltool) */
float* get_angular_velocity(BallState* ballState) {
    return &ballState->rvw[6];
}
