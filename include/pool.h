#ifndef POOL_H
#define POOL_H

#ifdef _WIN32
    #define DLL_EXPORT __declspec(dllexport)
#else
    #define DLL_EXPORT
#endif


#include <time.h>
    #include <stdint.h>

// Class Ball defined
// https://github.com/ekiefl/pooltool/blob/main/pooltool/objects/ball/datatypes.py#L321
//
// Class BallState
// https://github.com/ekiefl/pooltool/blob/main/pooltool/objects/ball/datatypes.py#L70

DLL_EXPORT void hello_world(const char* matrix_name, double* rvw);

/*
def collide_balls(
    rvw1: NDArray[np.float64],
    rvw2: NDArray[np.float64],
    R: float,
    M: float,
    u_s1: float = 0.21,
    u_s2: float = 0.21,
    u_b: float = 0.05,
    e_b: float = 0.89,
    deltaP: Optional[float] = None,
    N: int = 1000,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
 */

 /*
    rvw1:
        Kinematic state of ball 1 (see
        :class:`pooltool.objects.ball.datatypes.BallState`).
    rvw2:
        Kinematic state of ball 2 (see
        :class:`pooltool.objects.ball.datatypes.BallState`).
    R: Radius of the balls.
    M: Mass of the balls.
    u_s1: Coefficient of sliding friction between ball 1 and the surface.
    u_s2: Coefficient of sliding friction between ball 2 and the surface.
    u_b: Coefficient of friction between the balls during collision.
    e_b: Coefficient of restitution between the balls.
    deltaP:
        Normal impulse step size. If not passed, automatically selected according to
        Equation 14 in the reference.
    N:
        If deltaP is not specified, it is calculated such that approximately this
        number of iterations are performed (see Equation 14 in reference). If deltaP
        is not None, this does nothing.
 */


 typedef struct {
     unsigned long long cycle_start;
     unsigned long long cycle_end;
     #ifdef _MSC_VER
         LARGE_INTEGER freq, start_counter, end_counter;
     #else
         struct timespec start_ts, end_ts;
     #endif
     long int flops; // W
     long int memory; // Q -> loads and stores
 } Profile;

DLL_EXPORT void collide_balls(double* rvw1, double* rvw2, float R, float M, float u_s1, float u_s2, float u_b, float e_b, float deltaP, int N, double* rvw1_result, double* rvw2_result, Profile* profiles);

DLL_EXPORT void code_motion_collide_balls(double* rvw1, double* rvw2, float R, float M, float u_s1, float u_s2, float u_b, float e_b, float deltaP, int N, double* rvw1_result, double* rvw2_result, Profile* profiles);

/* Assuming rvw is row-major (passed from pooltool) */
double* get_displacement(double* rvw);

/* Assuming rvw is row-major (passed from pooltool) */
double* get_velocity(double* rvw);

/* Assuming rvw is row-major (passed from pooltool) */
double* get_angular_velocity(double* rvw);


#endif
