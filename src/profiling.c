#include "pool.h"
#include <stdbool.h>
#include <math.h>
#include <stdio.h>

int main() {

    double rvw1[9] = {
                        0.6992984838308823, 1.6730222005156081, 0.028575,
                        0.1698748188477139, -0.1697055263453331, 0.0,
                        5.938951053205008, 5.944875550226208, 0.0
                    };
    double rvw2[9] = {
                        0.7094873834407238, 1.616787789884389, 0.028575,
                        0.0606759657524449, 0.0319282034361416, 0.0,
                        -1.117347451833478, 2.123393377163425, 0.0
                    };

    double rvw1_result[9];
    double rvw2_result[9];
    collide_balls(rvw1, rvw2, 0.028575f, 0.170097, 0.2, 0.2, 0.08734384602561387, 0.95, 0.0, 1000, rvw1_result, rvw2_result);

    double tolerance = 1e-6;
    bool correct = true;

    correct &= (fabs(rvw1_result[3] - 0.12563856764795983) <= tolerance);
    correct &= (fabs(rvw1_result[4] - 0.03837893810480299) <= tolerance);
    correct &= (fabs(rvw1_result[5] - 0.0) <= tolerance);

    correct &= (fabs(rvw1_result[6] - 4.439313076347657) <= tolerance);
    correct &= (fabs(rvw1_result[7] - 5.673161832134484) <= tolerance);
    correct &= (fabs(rvw1_result[8] - -0.5625137907000655) <= tolerance);

    correct &= (fabs(rvw2_result[3] - 0.10160950039505853) <= tolerance);
    correct &= (fabs(rvw2_result[4] - -0.17583062081979212) <= tolerance);
    correct &= (fabs(rvw2_result[5] - 0.0) <= tolerance);

    correct &= (fabs(rvw2_result[6] - -2.5884954729422778) <= tolerance);
    correct &= (fabs(rvw2_result[7] -  2.1406312388740973) <= tolerance);
    correct &= (fabs(rvw2_result[8] - -0.5625137907000655) <= tolerance);

    if(!correct) {
        printf("\nWarning, your code is no longer within 1e-6 tolerance.\n");
    }

    return 0;
}
