#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#ifdef _WIN32
#include <windows.h>
#endif
#include "unity.h"
#include "pool.h"
#include "pool_spmd.h"
#include "profiling.h"
#include "profiling_spmd.h"

#ifdef PROFILE
const char *file_name = "profiling_spmd.csv";
#else
const char *file_name = "benchmark_spmd.csv";
#endif

CollisionDataSPMD reference;

void setUp(void)
{
    reference = (CollisionDataSPMD){
        .R = 0.028575,
        .M = 0.170097,
        .u_s1 = 0.2,
        .u_s2 = 0.2,
        .e_b = 0.95,
        .N = 1000,

        .col1_u_b = 0.08734384602561387,
        .col2_u_b = 0.051263960345564547,
        .col3_u_b = 0.08312916406140705,
        .col4_u_b = 0.034911633671085254,

        .col1_rvw1 = {0.6992984838308823, 1.6730222005156081, 0.028575, 0.1698748188477139, -0.1697055263453331, 0.0, 5.938951053205008, 5.944875550226208, 0.0},
        .col1_rvw2 = {0.7094873834407238, 1.616787789884389, 0.028575, 0.0606759657524449, 0.0319282034361416, 0.0, -1.117347451833478, 2.123393377163425, 0.0},
        .col2_rvw1 = {0.6625385097859544, 1.7115842755232573, 0.028575, 0.0446139693964031, 0.0828311545852449, 0.0, -2.8987280694748887, 1.5612937671532132, 0.0},
        .col2_rvw2 = {0.6388535708475327, 1.763595296295621, 0.028575, 0.4584680433236995, -0.2025096965252413, 0.0, 7.086953509194799, 16.044375969333316, 15.419428906285734},
        .col3_rvw1 = {0.3959469351985615, 0.4161515777741728, 0.028575, 0.1217519974181435, -0.326307055387703, 0.0, 11.41931952362915, 4.260787311221119, 23.191742437842027},
        .col3_rvw2 = {0.3463452334341006, 0.3877650704798705, 0.028575, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        .col4_rvw1 = {0.5302955807259006, 1.6078116068699646, 0.028575, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        .col4_rvw2 = {0.5326951841566092, 1.550712005140396, 0.028575, -0.2808347525858405, 0.6822451488164933, 0.0, -23.87559575910738, -9.82798784202416, -22.204439444348584},

        .col1_ball1 = {.velocity = {0.12563856764795983, 0.03837893810480299, 0.0}, .angular = {4.439313076347657, 5.673161832134484, -0.5625137907000655}},
        .col1_ball2 = {.velocity = {0.10160950039505853, -0.17583062081979212, 0.0}, .angular = {-2.5884954729422778, 2.1406312388740973, -0.5625137907000655}},
        .col2_ball1 = {.velocity = {0.23522139367796607, -0.2921400324639589, 0.0}, .angular = {-3.6328276540227558, 1.3688037962335442, -1.7097016805921317}},
        .col2_ball2 = {.velocity = {0.2662732674008842, 0.17238793191918897, 0.0}, .angular = {6.3592894981178825, 15.713010089292721, 13.709727225693594}},
        .col3_ball1 = {.velocity = {0.16726244740417917, -0.2949975214713052, 0.0}, .angular = {11.408127004317645, 4.280344772064272, 22.79199711500019}},
        .col3_ball2 = {.velocity = {-0.04546808169752021, -0.0312803252741715, 0.0}, .angular = {-0.008637082546362508, 0.01585069894318658, -0.3997453228418319}},
        .col4_ball1 = {.velocity = {-0.017305554796558674, 0.6744899058197328, 0.0}, .angular = {1.4640303990030437, 0.06811652077476217, 0.9715947428011615}},
        .col4_ball2 = {.velocity = {-0.2634288078123872, 0.003573833954956038, 0.0}, .angular = {-22.045737797390846, -9.751088296149208, -21.232844701547382}}};
}

void tearDown(void) {}

void summarize_profile(Profile *profile, const char *func_name, const char *part_name, int iteration, FILE *file)
{
    fprintf(file, "%s,%s,%d,%llu\n", func_name, part_name, iteration, profile->cycles_cumulative);
}

void flush_cache(void)
{
    volatile char *buf = (volatile char *)malloc(FLUSH_SIZE);
    if (buf == NULL)
    {
        return;
    }
    for (size_t i = 0; i < FLUSH_SIZE; i++)
    {
        buf[i] = (char)i; // Write data to each location
    }
    for (size_t i = 0; i < FLUSH_SIZE; i++)
    {
        (void)buf[i]; // Read back the data
    }
    free((void *)buf);
}

void call_function(const char *name, CollideBallsFnSPMD collide_fn)
{

    FILE *csv = fopen(file_name, "a");
    if (csv == NULL)
    {
        perror("Failed to open CSV file");
        return;
    }

    Profile warmup_profiles[6];
    double col1_rvw1_result[9];
    double col1_rvw2_result[9];
    double col2_rvw1_result[9];
    double col2_rvw2_result[9];
    double col3_rvw1_result[9];
    double col3_rvw2_result[9];
    double col4_rvw1_result[9];
    double col4_rvw2_result[9];

    for (int i = 0; i < WARMUP; i++)
    {
        collide_fn(
            reference.col1_rvw1,
            reference.col1_rvw2,
            reference.col2_rvw1,
            reference.col2_rvw2,
            reference.col3_rvw1,
            reference.col3_rvw2,
            reference.col4_rvw1,
            reference.col4_rvw2,
            reference.R,
            reference.M,
            reference.u_s1,
            reference.u_s2,
            reference.col1_u_b,
            reference.col2_u_b,
            reference.col3_u_b,
            reference.col4_u_b,
            reference.e_b,
            0.0f, // deltaP
            reference.N,
            col1_rvw1_result,
            col1_rvw2_result,
            col2_rvw1_result,
            col2_rvw2_result,
            col3_rvw1_result,
            col3_rvw2_result,
            col4_rvw1_result,
            col4_rvw2_result,
            warmup_profiles,
            NULL);
    }

    Profile profiles[6];
    // flush_cache(); // Flush the cache before each iteration
    for (int j = 0; j < ITERATIONS; j++)
    {
        // flush_cache(); // Flush the cache before each iteration
        // for (int i = 0; i < TEST_CASES; i++)
        // {
        double col1_rvw1_result[9];
        double col1_rvw2_result[9];
        double col2_rvw1_result[9];
        double col2_rvw2_result[9];
        double col3_rvw1_result[9];
        double col3_rvw2_result[9];
        double col4_rvw1_result[9];
        double col4_rvw2_result[9];
        for (int p = 0; p < 6; p++)
        {
            init_profiling_section(&profiles[p]);
        }
// flush_cache(); // Flush the cache before each iteration
#ifndef PROFILE
        myInt64 start = start_tsc();
#endif

        collide_fn(
            reference.col1_rvw1,
            reference.col1_rvw2,
            reference.col2_rvw1,
            reference.col2_rvw2,
            reference.col3_rvw1,
            reference.col3_rvw2,
            reference.col4_rvw1,
            reference.col4_rvw2,
            reference.R,
            reference.M,
            reference.u_s1,
            reference.u_s2,
            reference.col1_u_b,
            reference.col2_u_b,
            reference.col3_u_b,
            reference.col4_u_b,
            reference.e_b,
            0.0f, // deltaP
            reference.N,
            col1_rvw1_result,
            col1_rvw2_result,
            col2_rvw1_result,
            col2_rvw2_result,
            col3_rvw1_result,
            col3_rvw2_result,
            col4_rvw1_result,
            col4_rvw2_result,
            profiles,
            NULL);

#ifdef PROFILE
        summarize_profile(&profiles[0], name, "collide_balls", j, csv);
        summarize_profile(&profiles[1], name, "Initialization", j, csv);
        summarize_profile(&profiles[2], name, "Impulse", j, csv);
        summarize_profile(&profiles[3], name, "Delta", j, csv);
        summarize_profile(&profiles[4], name, "Velocity", j, csv);
        summarize_profile(&profiles[5], name, "Transform to World Frame", j, csv);
#else
        myInt64 cycles = stop_tsc(start);
        fprintf(csv, "%s,%d,%llu\n", name, j, cycles);
#endif

        TEST_ASSERT_DOUBLE_WITHIN_MESSAGE(tolerance, reference.col1_ball1.velocity[0], col1_rvw1_result[3], "Collision 1: Ball 1 Velocity X not within tolerance!");
        TEST_ASSERT_DOUBLE_WITHIN_MESSAGE(tolerance, reference.col1_ball1.velocity[1], col1_rvw1_result[4], "Collision 1: Ball 1 Velocity Y not within tolerance!");
        TEST_ASSERT_DOUBLE_WITHIN_MESSAGE(tolerance, reference.col1_ball1.velocity[2], col1_rvw1_result[5], "Collision 1: Ball 1 Velocity Z not within tolerance!");

        TEST_ASSERT_DOUBLE_WITHIN_MESSAGE(tolerance, reference.col1_ball1.angular[0], col1_rvw1_result[6], "Collision 1: Ball 1 Angular Velocity X not within tolerance!");
        TEST_ASSERT_DOUBLE_WITHIN_MESSAGE(tolerance, reference.col1_ball1.angular[1], col1_rvw1_result[7], "Collision 1: Ball 1 Angular Velocity Y not within tolerance!");
        TEST_ASSERT_DOUBLE_WITHIN_MESSAGE(tolerance, reference.col1_ball1.angular[2], col1_rvw1_result[8], "Collision 1: Ball 1 Angular Velocity Z not within tolerance!");

        TEST_ASSERT_DOUBLE_WITHIN_MESSAGE(tolerance, reference.col1_ball2.velocity[0], col1_rvw2_result[3], "Collision 1: Ball 2 Velocity X not within tolerance!");
        TEST_ASSERT_DOUBLE_WITHIN_MESSAGE(tolerance, reference.col1_ball2.velocity[1], col1_rvw2_result[4], "Collision 1: Ball 2 Velocity Y not within tolerance!");
        TEST_ASSERT_DOUBLE_WITHIN_MESSAGE(tolerance, reference.col1_ball2.velocity[2], col1_rvw2_result[5], "Collision 1: Ball 2 Velocity Z not within tolerance!");

        TEST_ASSERT_DOUBLE_WITHIN_MESSAGE(tolerance, reference.col1_ball2.angular[0], col1_rvw2_result[6], "Collision 1: Ball 2 Angular Velocity X not within tolerance!");
        TEST_ASSERT_DOUBLE_WITHIN_MESSAGE(tolerance, reference.col1_ball2.angular[1], col1_rvw2_result[7], "Collision 1: Ball 2 Angular Velocity Y not within tolerance!");
        TEST_ASSERT_DOUBLE_WITHIN_MESSAGE(tolerance, reference.col1_ball2.angular[2], col1_rvw2_result[8], "Collision 1: Ball 2 Angular Velocity Z not within tolerance!");

        TEST_ASSERT_DOUBLE_WITHIN_MESSAGE(tolerance, reference.col2_ball1.velocity[0], col2_rvw1_result[3], "Collision 2: Ball 1 Velocity X not within tolerance!");
        TEST_ASSERT_DOUBLE_WITHIN_MESSAGE(tolerance, reference.col2_ball1.velocity[1], col2_rvw1_result[4], "Collision 2: Ball 1 Velocity Y not within tolerance!");
        TEST_ASSERT_DOUBLE_WITHIN_MESSAGE(tolerance, reference.col2_ball1.velocity[2], col2_rvw1_result[5], "Collision 2: Ball 1 Velocity Z not within tolerance!");

        TEST_ASSERT_DOUBLE_WITHIN_MESSAGE(tolerance, reference.col2_ball1.angular[0], col2_rvw1_result[6], "Collision 2: Ball 1 Angular Velocity X not within tolerance!");
        TEST_ASSERT_DOUBLE_WITHIN_MESSAGE(tolerance, reference.col2_ball1.angular[1], col2_rvw1_result[7], "Collision 2: Ball 1 Angular Velocity Y not within tolerance!");
        TEST_ASSERT_DOUBLE_WITHIN_MESSAGE(tolerance, reference.col2_ball1.angular[2], col2_rvw1_result[8], "Collision 2: Ball 1 Angular Velocity Z not within tolerance!");

        TEST_ASSERT_DOUBLE_WITHIN_MESSAGE(tolerance, reference.col2_ball2.velocity[0], col2_rvw2_result[3], "Collision 2: Ball 2 Velocity X not within tolerance!");
        TEST_ASSERT_DOUBLE_WITHIN_MESSAGE(tolerance, reference.col2_ball2.velocity[1], col2_rvw2_result[4], "Collision 2: Ball 2 Velocity Y not within tolerance!");
        TEST_ASSERT_DOUBLE_WITHIN_MESSAGE(tolerance, reference.col2_ball2.velocity[2], col2_rvw2_result[5], "Collision 2: Ball 2 Velocity Z not within tolerance!");

        TEST_ASSERT_DOUBLE_WITHIN_MESSAGE(tolerance, reference.col2_ball2.angular[0], col2_rvw2_result[6], "Collision 2: Ball 2 Angular Velocity X not within tolerance!");
        TEST_ASSERT_DOUBLE_WITHIN_MESSAGE(tolerance, reference.col2_ball2.angular[1], col2_rvw2_result[7], "Collision 2: Ball 2 Angular Velocity Y not within tolerance!");
        TEST_ASSERT_DOUBLE_WITHIN_MESSAGE(tolerance, reference.col2_ball2.angular[2], col2_rvw2_result[8], "Collision 2: Ball 2 Angular Velocity Z not within tolerance!");

        TEST_ASSERT_DOUBLE_WITHIN_MESSAGE(tolerance, reference.col3_ball1.velocity[0], col3_rvw1_result[3], "Collision 3: Ball 1 Velocity X not within tolerance!");
        TEST_ASSERT_DOUBLE_WITHIN_MESSAGE(tolerance, reference.col3_ball1.velocity[1], col3_rvw1_result[4], "Collision 3: Ball 1 Velocity Y not within tolerance!");
        TEST_ASSERT_DOUBLE_WITHIN_MESSAGE(tolerance, reference.col3_ball1.velocity[2], col3_rvw1_result[5], "Collision 3: Ball 1 Velocity Z not within tolerance!");

        TEST_ASSERT_DOUBLE_WITHIN_MESSAGE(tolerance, reference.col3_ball1.angular[0], col3_rvw1_result[6], "Collision 3: Ball 1 Angular Velocity X not within tolerance!");
        TEST_ASSERT_DOUBLE_WITHIN_MESSAGE(tolerance, reference.col3_ball1.angular[1], col3_rvw1_result[7], "Collision 3: Ball 1 Angular Velocity Y not within tolerance!");
        TEST_ASSERT_DOUBLE_WITHIN_MESSAGE(tolerance, reference.col3_ball1.angular[2], col3_rvw1_result[8], "Collision 3: Ball 1 Angular Velocity Z not within tolerance!");

        TEST_ASSERT_DOUBLE_WITHIN_MESSAGE(tolerance, reference.col3_ball2.velocity[0], col3_rvw2_result[3], "Collision 3: Ball 2 Velocity X not within tolerance!");
        TEST_ASSERT_DOUBLE_WITHIN_MESSAGE(tolerance, reference.col3_ball2.velocity[1], col3_rvw2_result[4], "Collision 3: Ball 2 Velocity Y not within tolerance!");
        TEST_ASSERT_DOUBLE_WITHIN_MESSAGE(tolerance, reference.col3_ball2.velocity[2], col3_rvw2_result[5], "Collision 3: Ball 2 Velocity Z not within tolerance!");

        TEST_ASSERT_DOUBLE_WITHIN_MESSAGE(tolerance, reference.col3_ball2.angular[0], col3_rvw2_result[6], "Collision 3: Ball 2 Angular Velocity X not within tolerance!");
        TEST_ASSERT_DOUBLE_WITHIN_MESSAGE(tolerance, reference.col3_ball2.angular[1], col3_rvw2_result[7], "Collision 3: Ball 2 Angular Velocity Y not within tolerance!");
        TEST_ASSERT_DOUBLE_WITHIN_MESSAGE(tolerance, reference.col3_ball2.angular[2], col3_rvw2_result[8], "Collision 3: Ball 2 Angular Velocity Z not within tolerance!");

        TEST_ASSERT_DOUBLE_WITHIN_MESSAGE(tolerance, reference.col4_ball1.velocity[0], col4_rvw1_result[3], "Collision 4: Ball 1 Velocity X not within tolerance!");
        TEST_ASSERT_DOUBLE_WITHIN_MESSAGE(tolerance, reference.col4_ball1.velocity[1], col4_rvw1_result[4], "Collision 4: Ball 1 Velocity Y not within tolerance!");
        TEST_ASSERT_DOUBLE_WITHIN_MESSAGE(tolerance, reference.col4_ball1.velocity[2], col4_rvw1_result[5], "Collision 4: Ball 1 Velocity Z not within tolerance!");

        TEST_ASSERT_DOUBLE_WITHIN_MESSAGE(tolerance, reference.col4_ball1.angular[0], col4_rvw1_result[6], "Collision 4: Ball 1 Angular Velocity X not within tolerance!");
        TEST_ASSERT_DOUBLE_WITHIN_MESSAGE(tolerance, reference.col4_ball1.angular[1], col4_rvw1_result[7], "Collision 4: Ball 1 Angular Velocity Y not within tolerance!");
        TEST_ASSERT_DOUBLE_WITHIN_MESSAGE(tolerance, reference.col4_ball1.angular[2], col4_rvw1_result[8], "Collision 4: Ball 1 Angular Velocity Z not within tolerance!");

        TEST_ASSERT_DOUBLE_WITHIN_MESSAGE(tolerance, reference.col4_ball2.velocity[0], col4_rvw2_result[3], "Collision 4: Ball 2 Velocity X not within tolerance!");
        TEST_ASSERT_DOUBLE_WITHIN_MESSAGE(tolerance, reference.col4_ball2.velocity[1], col4_rvw2_result[4], "Collision 4: Ball 2 Velocity Y not within tolerance!");
        TEST_ASSERT_DOUBLE_WITHIN_MESSAGE(tolerance, reference.col4_ball2.velocity[2], col4_rvw2_result[5], "Collision 4: Ball 2 Velocity Z not within tolerance!");

        TEST_ASSERT_DOUBLE_WITHIN_MESSAGE(tolerance, reference.col4_ball2.angular[0], col4_rvw2_result[6], "Collision 4: Ball 2 Angular Velocity X not within tolerance!");
        TEST_ASSERT_DOUBLE_WITHIN_MESSAGE(tolerance, reference.col4_ball2.angular[1], col4_rvw2_result[7], "Collision 4: Ball 2 Angular Velocity Y not within tolerance!");
        TEST_ASSERT_DOUBLE_WITHIN_MESSAGE(tolerance, reference.col4_ball2.angular[2], col4_rvw2_result[8], "Collision 4: Ball 2 Angular Velocity Z not within tolerance!");
        // }
    }

    fclose(csv);
}

void test_collide_balls_basic(void)
{
    call_function("4x Basic Implementation", spmd_4x_linear);
}
void test_collide_balls_recip_sqrt(void)
{
    call_function("4x Recip Sqrt Implementation", spmd_4x_recip_sqrt);
}

void test_spmd_basic(void)
{
    call_function("SPMD Basic Implementation", spmd_basic_collide_balls);
}

void test_spmd2_FMA(void)
{
    call_function("SPMD 2: FMA", spmd2_FMA);
}

void test_spmd3_recip_sqrt(void)
{
    call_function("SPMD 3: Recip Sqrt", spmd3_Recip_Sqrt);
}

int main()
{
#ifdef _WIN32
    // Pin to first processor
    SetProcessAffinityMask(GetCurrentProcess(), 1);
#endif

    FILE *csv = fopen(file_name, "w");
    if (csv == NULL)
    {
        perror("Failed to open CSV file");
        return 0;
    }
#ifdef PROFILE
    fprintf(csv, "Function,Section,Iteration,Cycles\n");
#else
    fprintf(csv, "Function,Iteration,Cycles\n");
#endif
    fclose(csv);

    srand((unsigned int)time(NULL));

    void (*tests[])(void) = {
        test_collide_balls_basic,
        test_collide_balls_recip_sqrt,
        test_spmd_basic,
        test_spmd2_FMA,
        test_spmd3_recip_sqrt,
    };

    const char *function_names[] = {
        "4x Basic Implementation",
        "4x Recip Sqrt Implementation",
        "SPMD: Basic Implementation",
        "SPMD 2: FMA",
        "SPMD 3: Recip Sqrt",
    };

#define NUM_FUNCTIONS (sizeof(tests) / sizeof(tests[0]))

    UNITY_BEGIN();

    for (int i = 0; i < TEST_RUNNER_ITERATIONS; i++)
    {

        // Fisher–Yates shuffle
        for (int i = NUM_FUNCTIONS - 1; i > 0; i--)
        {
            int j = rand() % (i + 1);
            void (*tmp)(void) = tests[i];
            tests[i] = tests[j];
            tests[j] = tmp;
            const char *tmp_name = function_names[i];
            function_names[i] = function_names[j];
            function_names[j] = tmp_name;
        }

        for (int i = 0; i < NUM_FUNCTIONS; i++)
        {
            printf("\n--- Running test %d: %s ---\n", i, function_names[i]);
            RUN_TEST(tests[i]);
        }

        printf("\n=== Finished Test Iteration %d/%d ===\n", i + 1, TEST_RUNNER_ITERATIONS);
    }

    int result = UNITY_END();

    return result;
}
