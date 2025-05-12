#include <stdio.h>

#include "pool.h"

typedef struct {
    float R;          // Ball radius
    float M;          // Ball mass
    float u_s1;       // Coefficient of sliding friction (ball 1)
    float u_s2;       // Coefficient of sliding friction (ball 2)
    float u_b;        // Coefficient of ball-ball friction
    float e_b;        // Coefficient of restitution
    int    N;          // Number of iterations (deltaP is None, so we skip it)

    // Initial rvw (position, velocity, angular velocity)
    double rvw1[9];
    double rvw2[9];

    struct {
        double velocity [3];
        double angular[3];
    } ball1, ball2;
} CollisionData;

#define TEST_CASES 5
CollisionData reference[TEST_CASES];

void setUp(void) {

    reference[0] = (CollisionData){
            .R = 0.028575,
            .M = 0.170097,
            .u_s1 = 0.2,
            .u_s2 = 0.2,
            .u_b = 0.08734384602561387,
            .e_b = 0.95,
            .N = 1000,
            .rvw1 = {
                0.6992984838308823, 1.6730222005156081, 0.028575,
                0.1698748188477139, -0.1697055263453331, 0.0,
                5.938951053205008, 5.944875550226208, 0.0
            },
            .rvw2 = {
                0.7094873834407238, 1.616787789884389, 0.028575,
                0.0606759657524449, 0.0319282034361416, 0.0,
                -1.117347451833478, 2.123393377163425, 0.0
            },
            .ball1 = {
                .velocity = { 0.12563856764795983, 0.03837893810480299, 0.0 },
                .angular = { 4.439313076347657, 5.673161832134484, -0.5625137907000655 }
            },
            .ball2 = {
                .velocity = { 0.10160950039505853, -0.17583062081979212, 0.0 },
                .angular = { -2.5884954729422778, 2.1406312388740973, -0.5625137907000655 }
            }
        };

    reference[1] = (CollisionData){
        .R = 0.028575,
        .M = 0.170097,
        .u_s1 = 0.2,
        .u_s2 = 0.2,
        .u_b = 0.051263960345564547,
        .e_b = 0.95,
        .N = 1000,
        .rvw1 = {
            0.6625385097859544, 1.7115842755232573, 0.028575,
            0.0446139693964031, 0.0828311545852449, 0.0,
            -2.8987280694748887, 1.5612937671532132, 0.0
        },
        .rvw2 = {
            0.6388535708475327, 1.763595296295621, 0.028575,
            0.4584680433236995, -0.2025096965252413, 0.0,
            7.086953509194799, 16.044375969333316, 15.419428906285734
        },
        .ball1 = {
            .velocity = { 0.23522139367796607, -0.2921400324639589, 0.0 },
            .angular = { -3.6328276540227558, 1.3688037962335442, -1.7097016805921317 }
        },
        .ball2 = {
            .velocity = { 0.2662732674008842, 0.17238793191918897, 0.0 },
            .angular = { 6.3592894981178825, 15.713010089292721, 13.709727225693594 }
        }
    };


    reference[2] = (CollisionData) {
        .R = 0.028575,
        .M = 0.170097,
        .u_s1 = 0.2,
        .u_s2 = 0.2,
        .u_b = 0.08312916406140705,
        .e_b = 0.95,
        .N = 1000,
        .rvw1 = {
            0.3959469351985615, 0.4161515777741728, 0.028575,
            0.1217519974181435, -0.326307055387703, 0.0,
            11.41931952362915, 4.260787311221119, 23.191742437842027
        },
        .rvw2 = {
            0.3463452334341006, 0.3877650704798705, 0.028575,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0
        },
        .ball1 = {
            .velocity = { 0.16726244740417917, -0.2949975214713052, 0.0 },
            .angular = { 11.408127004317645, 4.280344772064272, 22.79199711500019 }
        },
        .ball2 = {
            .velocity = { -0.04546808169752021, -0.0312803252741715, 0.0 },
            .angular = { -0.008637082546362508, 0.01585069894318658, -0.3997453228418319 }
        }
    };

    reference[3] = (CollisionData){
        .R = 0.028575,
        .M = 0.170097,
        .u_s1 = 0.2,
        .u_s2 = 0.2,
        .u_b = 0.034911633671085254,
        .e_b = 0.95,
        .N = 1000,
        .rvw1 = {
            0.5302955807259006, 1.6078116068699646, 0.028575,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0
        },
        .rvw2 = {
            0.5326951841566092, 1.550712005140396, 0.028575,
            -0.2808347525858405, 0.6822451488164933, 0.0,
            -23.87559575910738, -9.82798784202416, -22.204439444348584
        },
        .ball1 = {
            .velocity = { -0.017305554796558674, 0.6744899058197328, 0.0 },
            .angular = { 1.4640303990030437, 0.06811652077476217, 0.9715947428011615 }
        },
        .ball2 = {
            .velocity = { -0.2634288078123872, 0.003573833954956038, 0.0 },
            .angular = { -22.045737797390846, -9.751088296149208, -21.232844701547382 }
        }
    };


    reference[4] = (CollisionData){
        .R = 0.028575,
        .M = 0.170097,
        .u_s1 = 0.2,
        .u_s2 = 0.2,
        .u_b = 0.01927792191606111,
        .e_b = 0.95,
        .N = 1000,
        .rvw1 = {
            0.7259229146341827, 0.5061287584781619, 0.028575,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0
        },
        .rvw2 = {
            0.6823057617760474, 0.543056615048102, 0.028575,
            2.4511368257402206, 0.6259593056098344, 0.0,
            -19.324926113047344, 75.67271166973282, 0.0
        },
        .ball1 = {
            .velocity = { 1.1060149417261151, -0.9059349187641589, 0.0 },
            .angular = { -0.6741763817472055, -0.7895903199908095, -2.0380303552667263 }
        },
        .ball2 = {
            .velocity = { 1.3428287942353496, 1.5337707065147121, 0.0 },
            .angular = { -20.163274160647937, 74.68250107163708, -2.0380303552667263 }
        }
    };
}

typedef void (*CollideBallsFn)(double*, double*, float, float, float, float, float, float, float, int, double*, double*, Profile*, Branch*);

void call_function(const char* name, CollideBallsFn collide_fn) {

    FILE* csv = fopen("flops.csv", "a");
    if (csv == NULL) {
        perror("Failed to open CSV file");
        return;
    }

    Profile profiles[6];
    Branch branches[11] = {0};

    for(int i = 0; i < TEST_CASES; i++) {
        double rvw1_result[9];
        double rvw2_result[9];

        for (int p = 0; p < 6; p++) {
            init_profiling_section(&profiles[p]);
        }

        collide_fn(
            reference[i].rvw1,
            reference[i].rvw2,
            reference[i].R,
            reference[i].M,
            reference[i].u_s1,
            reference[i].u_s2,
            reference[i].u_b,
            reference[i].e_b,
            0.0f,           // deltaP
            reference[i].N,
            rvw1_result,
            rvw2_result,
            profiles,
            branches
        );

        fprintf(csv, "%s,%s,%d,%ld,%ld,%ld,%ld,%ld,%ld\n", name, "collide_balls", i, profiles[0].flops, profiles[0].memory * sizeof(double), profiles[0].ADDS, profiles[0].MULS, profiles[0].DIVS, profiles[0].SQRT);
        fprintf(csv, "%s,%s,%d,%ld,%ld,%ld,%ld,%ld,%ld\n", name, "Initialization", i, profiles[1].flops, profiles[1].memory* sizeof(double), profiles[1].ADDS, profiles[1].MULS, profiles[1].DIVS, profiles[1].SQRT);
        fprintf(csv, "%s,%s,%d,%ld,%ld,%ld,%ld,%ld,%ld\n", name, "Impulse", i, profiles[2].flops, profiles[2].memory* sizeof(double), profiles[2].ADDS, profiles[2].MULS, profiles[2].DIVS, profiles[2].SQRT);
        fprintf(csv, "%s,%s,%d,%ld,%ld,%ld,%ld,%ld,%ld\n", name, "Delta", i, profiles[3].flops, profiles[3].memory* sizeof(double), profiles[3].ADDS, profiles[3].MULS, profiles[3].DIVS, profiles[3].SQRT);
        fprintf(csv, "%s,%s,%d,%ld,%ld,%ld,%ld,%ld,%ld\n", name, "Velocity", i, profiles[4].flops, profiles[4].memory* sizeof(double), profiles[4].ADDS, profiles[4].MULS, profiles[4].DIVS, profiles[4].SQRT);
        fprintf(csv, "%s,%s,%d,%ld,%ld,%ld,%ld,%ld,%ld\n", name, "Transform to World Frame", i, profiles[5].flops, profiles[5].memory* sizeof(double), profiles[5].ADDS, profiles[5].MULS, profiles[5].DIVS, profiles[5].SQRT);

        printf("=== BRANCH PREDICTION ANALYSIS %s TC %d === \n", name, i);
        for(int b = 0; b < 11; b++) {
            printf("Executed Branch %d %lu times.\n", b, branches[b].count);
        }
    }

    fclose(csv);
}

int main() {
    FILE* csv = fopen("flops.csv", "w");
    if (csv == NULL) {
        perror("Failed to open CSV file");
        return 0;
    }
    fprintf(csv, "Function,Section,Test Case,Flops,Memory,ADDS,MULS,DIVS,SQRT\n");
    fclose(csv);

    setUp();

    call_function("Basic Implementation", collide_balls);
    call_function("Less SQRT", less_sqrt_collide_balls);
    call_function("Precompute", simple_precompute_cb);
    call_function("Branch Pred", branch_prediction_collide_balls);
    call_function("Removed Unused Branches", remove_unused_branches);
    call_function("Code Motion", code_motion_collide_balls);
    call_function("SIMD", simd_collide_balls);

    return 0;
}
