#ifndef POOL_H
#define POOL_H

#include <stdint.h>


// Class Ball defined
// https://github.com/ekiefl/pooltool/blob/main/pooltool/objects/ball/datatypes.py#L321
//
// Class BallState
// https://github.com/ekiefl/pooltool/blob/main/pooltool/objects/ball/datatypes.py#L70

enum MotionState{
    stationary = 0,
    spinning = 1,
    sliding = 2,
    rolling = 3,
    pocketed = 4
};

typedef struct BallState {
    float rvw[9];
    enum MotionState state;
    float time;
} BallState;

void hello_world(const char* matrix_name, double* rvw);

/* Assuming rvw is row-major (passed from pooltool) */
float* get_displacement(BallState* ballState);

/* Assuming rvw is row-major (passed from pooltool) */
float* get_velocity(BallState* ballState);

/* Assuming rvw is row-major (passed from pooltool) */
float* get_angular_velocity(BallState* ballState);

void collide_balls();


#endif
