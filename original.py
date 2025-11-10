from typing import Optional, Tuple

import numpy as np
from numba import jit
from numpy import array, dot, sqrt
from numpy.typing import NDArray

INF = float("inf")
Z_LOC = array([0, 0, 1], dtype=np.float64)


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
    """Simulates the frictional collision between two balls.

    Args:
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

    Returns:
        Tuple[NDArray[np.float64], NDArray[np.float64]]:
            The updated rvw arrays for balls 1 and 2.
    """
    r_i, v_i, w_i = rvw1.copy()
    r_j, v_j, w_j = rvw2.copy()

    v_i1, w_i1, v_j1, w_j1 = _collide_balls(
        r_i, v_i, w_i, r_j, v_j, w_j, R, M, u_s1, u_s2, u_b, e_b, deltaP, N
    )

    return rvw1, rvw2


@jit(nopython=True, cache=True)
def _collide_balls(
    r_i: NDArray[np.float64],
    v_i: NDArray[np.float64],
    w_i: NDArray[np.float64],
    r_j: NDArray[np.float64],
    v_j: NDArray[np.float64],
    w_j: NDArray[np.float64],
    R: float,
    M: float,
    u_s1: float = 0.21,
    u_s2: float = 0.21,
    u_b: float = 0.05,
    e_b: float = 0.89,
    deltaP: Optional[float] = None,
    N: int = 1000,
) -> Tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
]:
    """The numba-compiled delegate for :func:`collide_balls`.

    Returns:
        Tuple[NDArray[np.float64], NDArray[np.float64]]:
            A tuple containing:

            - Updated velocity vector of ball 1 after collision in global coordinates.
            - Updated angular velocity vector of ball 1 after collision in global
              coordinates.
            - Updated velocity vector of ball 2 after collision in global coordinates.
            - Updated angular velocity vector of ball 2 after collision in global
              coordinates.
    """
    r_ij = r_j - r_i
    r_ij_mag_sqrd = dot(r_ij, r_ij)
    r_ij_mag = sqrt(r_ij_mag_sqrd)
    y_loc = r_ij / r_ij_mag
    x_loc = np.cross(y_loc, Z_LOC)

    G = np.vstack((x_loc, y_loc, Z_LOC))
    v_ix, v_iy = dot(v_i, x_loc), dot(v_i, y_loc)
    v_jx, v_jy = dot(v_j, x_loc), dot(v_j, y_loc)

    w_ix, w_iy, w_iz = dot(G, w_i)
    w_jx, w_jy, w_jz = dot(G, w_j)

    u_iR_x, u_iR_y = v_ix + R * w_iy, v_iy - R * w_ix
    u_jR_x, u_jR_y = v_jx + R * w_jy, v_jy - R * w_jx
    u_iR_xy_mag = sqrt(u_iR_x**2 + u_iR_y**2)
    u_jR_xy_mag = sqrt(u_jR_x**2 + u_jR_y**2)

    u_ijC_x = v_ix - v_jx - R * (w_iz + w_jz)
    u_ijC_z = R * (w_ix + w_jx)
    u_ijC_xz_mag = sqrt(u_ijC_x**2 + u_ijC_z**2)

    v_ijy = v_jy - v_iy
    if deltaP is None:
        deltaP = 0.5 * (1 + e_b) * M * abs(v_ijy) / N
    assert deltaP is not None
    C = 5 / (2 * M * R)
    W_f = INF
    W_c = None
    W = 0
    niters = 0
    while v_ijy < 0 or W < W_f:
        # determine impulse deltas:
        if u_ijC_xz_mag < 1e-16:
            deltaP_1 = deltaP_2 = 0
            deltaP_ix = deltaP_iy = deltaP_jx = deltaP_jy = 0
        else:
            deltaP_1 = -u_b * deltaP * u_ijC_x / u_ijC_xz_mag
            if abs(u_ijC_z) < 1e-16:
                deltaP_2 = 0
                deltaP_ix = deltaP_iy = deltaP_jx = deltaP_jy = 0
            else:
                deltaP_2 = -u_b * deltaP * u_ijC_z / u_ijC_xz_mag
                if deltaP_2 > 0:
                    deltaP_ix = deltaP_iy = 0
                    if u_jR_xy_mag == 0:
                        deltaP_jx = deltaP_jy = 0
                    else:
                        deltaP_jx = -u_s2 * (u_jR_x / u_jR_xy_mag) * deltaP_2
                        deltaP_jy = -u_s2 * (u_jR_y / u_jR_xy_mag) * deltaP_2
                else:
                    deltaP_jx = deltaP_jy = 0
                    if u_iR_xy_mag == 0:
                        deltaP_ix = deltaP_iy = 0
                    else:
                        deltaP_ix = u_s1 * (u_iR_x / u_iR_xy_mag) * deltaP_2
                        deltaP_iy = u_s1 * (u_iR_y / u_iR_xy_mag) * deltaP_2

        # calc velocity changes:
        deltaV_ix = (deltaP_1 + deltaP_ix) / M
        deltaV_iy = (-deltaP + deltaP_iy) / M
        deltaV_jx = (-deltaP_1 + deltaP_jx) / M
        deltaV_jy = (deltaP + deltaP_jy) / M
        # calc angular velocity changes:
        deltaOm_ix = C * (deltaP_2 + deltaP_iy)
        deltaOm_iy = C * (-deltaP_ix)
        deltaOm_iz = C * (-deltaP_1)
        deltaOm_j = C * array([(deltaP_2 + deltaP_jy), (-deltaP_jx), (-deltaP_1)])
        # update velocities:
        v_ix += deltaV_ix
        v_jx += deltaV_jx
        v_iy += deltaV_iy
        v_jy += deltaV_jy
        # update angular velocities:
        w_ix += deltaOm_ix
        w_iy += deltaOm_iy
        w_iz += deltaOm_iz
        w_jx += deltaOm_j[0]
        w_jy += deltaOm_j[1]
        w_jz += deltaOm_j[2]
        # update ball-table slips:
        u_iR_x, u_iR_y = v_ix + R * w_iy, v_iy - R * w_ix
        u_jR_x, u_jR_y = v_jx + R * w_jy, v_jy - R * w_jx
        u_iR_xy_mag = sqrt(u_iR_x**2 + u_iR_y**2)
        u_jR_xy_mag = sqrt(u_jR_x**2 + u_jR_y**2)
        # update ball-ball slip:
        u_ijC_x = v_ix - v_jx - R * (w_iz + w_jz)
        u_ijC_z = R * (w_ix + w_jx)
        u_ijC_xz_mag = sqrt(u_ijC_x**2 + u_ijC_z**2)
        # increment work:
        v_ijy0 = v_ijy
        v_ijy = v_jy - v_iy
        W += 0.5 * deltaP * abs(v_ijy0 + v_ijy)
        niters += 1
        if W_c is None and v_ijy > 0:
            W_c = W
            W_f = (1 + e_b**2) * W_c
            # niters_c = niters
            # _logger.debug('''
            # END OF COMPRESSION PHASE
            # W_c = %s
            # W_f = %s
            # niters_c = %s
            # ''', W_c, W_f, niters_c)
    # _logger.debug('''
    # END OF RESTITUTION PHASE
    # niters = %d
    # ''', niters)

    v_i = array((v_ix, v_iy, 0))
    v_j = array((v_jx, v_jy, 0))
    w_i = array((w_ix, w_iy, w_iz))
    w_j = array((w_jx, w_jy, w_jz))
    G_T = G.T

    result_v_i = dot(G_T, v_i)
    result_v_j = dot(G_T, v_j)
    result_w_i = dot(G_T, w_i)
    result_w_j = dot(G_T, w_j)

    return result_v_i, result_w_i, result_v_j, result_w_j


if __name__ == "__main__":
    from dataclasses import dataclass
    from typing import List

    @dataclass
    class BallData:
        velocity: List[float]
        angular: List[float]

    @dataclass
    class CollisionData:
        R: float
        M: float
        u_s1: float
        u_s2: float
        u_b: float
        e_b: float
        N: int
        rvw1: np.ndarray
        rvw2: np.ndarray
        ball1: BallData
        ball2: BallData

    reference = [
        CollisionData(
            R=0.028575,
            M=0.170097,
            u_s1=0.2,
            u_s2=0.2,
            u_b=0.08734384602561387,
            e_b=0.95,
            N=1000,
            rvw1=np.array(
                [
                    0.6992984838308823,
                    1.6730222005156081,
                    0.028575,
                    0.1698748188477139,
                    -0.1697055263453331,
                    0.0,
                    5.938951053205008,
                    5.944875550226208,
                    0.0,
                ]
            ).reshape(3, 3),
            rvw2=np.array(
                [
                    0.7094873834407238,
                    1.616787789884389,
                    0.028575,
                    0.0606759657524449,
                    0.0319282034361416,
                    0.0,
                    -1.117347451833478,
                    2.123393377163425,
                    0.0,
                ]
            ).reshape(3, 3),
            ball1=BallData(
                velocity=[0.12563856764795983, 0.03837893810480299, 0.0],
                angular=[4.439313076347657, 5.673161832134484, -0.5625137907000655],
            ),
            ball2=BallData(
                velocity=[0.10160950039505853, -0.17583062081979212, 0.0],
                angular=[-2.5884954729422778, 2.1406312388740973, -0.5625137907000655],
            ),
        ),
        CollisionData(
            R=0.028575,
            M=0.170097,
            u_s1=0.2,
            u_s2=0.2,
            u_b=0.051263960345564547,
            e_b=0.95,
            N=1000,
            rvw1=np.array(
                [
                    0.6625385097859544,
                    1.7115842755232573,
                    0.028575,
                    0.0446139693964031,
                    0.0828311545852449,
                    0.0,
                    -2.8987280694748887,
                    1.5612937671532132,
                    0.0,
                ]
            ).reshape(3, 3),
            rvw2=np.array(
                [
                    0.6388535708475327,
                    1.763595296295621,
                    0.028575,
                    0.4584680433236995,
                    -0.2025096965252413,
                    0.0,
                    7.086953509194799,
                    16.044375969333316,
                    15.419428906285734,
                ]
            ).reshape(3, 3),
            ball1=BallData(
                velocity=[0.23522139367796607, -0.2921400324639589, 0.0],
                angular=[-3.6328276540227558, 1.3688037962335442, -1.7097016805921317],
            ),
            ball2=BallData(
                velocity=[0.2662732674008842, 0.17238793191918897, 0.0],
                angular=[6.3592894981178825, 15.713010089292721, 13.709727225693594],
            ),
        ),
        CollisionData(
            R=0.028575,
            M=0.170097,
            u_s1=0.2,
            u_s2=0.2,
            u_b=0.08312916406140705,
            e_b=0.95,
            N=1000,
            rvw1=np.array(
                [
                    0.3959469351985615,
                    0.4161515777741728,
                    0.028575,
                    0.1217519974181435,
                    -0.326307055387703,
                    0.0,
                    11.41931952362915,
                    4.260787311221119,
                    23.191742437842027,
                ]
            ).reshape(3, 3),
            rvw2=np.array(
                [
                    0.3463452334341006,
                    0.3877650704798705,
                    0.028575,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            ).reshape(3, 3),
            ball1=BallData(
                velocity=[0.16726244740417917, -0.2949975214713052, 0.0],
                angular=[11.408127004317645, 4.280344772064272, 22.79199711500019],
            ),
            ball2=BallData(
                velocity=[-0.04546808169752021, -0.0312803252741715, 0.0],
                angular=[
                    -0.008637082546362508,
                    0.01585069894318658,
                    -0.3997453228418319,
                ],
            ),
        ),
        CollisionData(
            R=0.028575,
            M=0.170097,
            u_s1=0.2,
            u_s2=0.2,
            u_b=0.034911633671085254,
            e_b=0.95,
            N=1000,
            rvw1=np.array(
                [
                    0.5302955807259006,
                    1.6078116068699646,
                    0.028575,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            ).reshape(3, 3),
            rvw2=np.array(
                [
                    0.5326951841566092,
                    1.550712005140396,
                    0.028575,
                    -0.2808347525858405,
                    0.6822451488164933,
                    0.0,
                    -23.87559575910738,
                    -9.82798784202416,
                    -22.204439444348584,
                ]
            ).reshape(3, 3),
            ball1=BallData(
                velocity=[-0.017305554796558674, 0.6744899058197328, 0.0],
                angular=[1.4640303990030437, 0.06811652077476217, 0.9715947428011615],
            ),
            ball2=BallData(
                velocity=[-0.2634288078123872, 0.003573833954956038, 0.0],
                angular=[-22.045737797390846, -9.751088296149208, -21.232844701547382],
            ),
        ),
        CollisionData(
            R=0.028575,
            M=0.170097,
            u_s1=0.2,
            u_s2=0.2,
            u_b=0.01927792191606111,
            e_b=0.95,
            N=1000,
            rvw1=np.array(
                [
                    0.7259229146341827,
                    0.5061287584781619,
                    0.028575,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
            ).reshape(3, 3),
            rvw2=np.array(
                [
                    0.6823057617760474,
                    0.543056615048102,
                    0.028575,
                    2.4511368257402206,
                    0.6259593056098344,
                    0.0,
                    -19.324926113047344,
                    75.67271166973282,
                    0.0,
                ]
            ).reshape(3, 3),
            ball1=BallData(
                velocity=[1.1060149417261151, -0.9059349187641589, 0.0],
                angular=[-0.6741763817472055, -0.7895903199908095, -2.0380303552667263],
            ),
            ball2=BallData(
                velocity=[1.3428287942353496, 1.5337707065147121, 0.0],
                angular=[-20.163274160647937, 74.68250107163708, -2.0380303552667263],
            ),
        ),
    ]

    import ctypes

    lib = ctypes.CDLL("./build/libpool_shared.dll")
    lib.python_start_tsc.argtypes = []
    lib.python_start_tsc.restype = ctypes.c_uint64

    lib.python_stop_tsc.argtypes = (ctypes.c_uint64,)
    lib.python_stop_tsc.restype = ctypes.c_uint64

    collide_balls(
        reference[0].rvw1,
        reference[0].rvw2,
        reference[0].R,
        reference[0].M,
        reference[0].u_s1,
        reference[0].u_s2,
        reference[0].u_b,
        reference[0].e_b,
        None,
        reference[0].N,
    )

    data = [["Function", "Test Case", "Iteration", "Cycles"]]
    iters = 10000
    for j in range(len(reference)):
        for i in range(iters):
            start = lib.python_start_tsc()
            collide_balls(
                reference[j].rvw1,
                reference[j].rvw2,
                reference[j].R,
                reference[j].M,
                reference[j].u_s1,
                reference[j].u_s2,
                reference[j].u_b,
                reference[j].e_b,
                None,
                reference[j].N,
            )
            end = lib.python_stop_tsc(start)
            data.append(["Original", j, i, end])

    import csv

    with open("./build/original.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(data)
