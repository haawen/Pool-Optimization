#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "../include/pool.h"

// Function to get high-resolution time in seconds
double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// Enum for collision types
typedef enum {
    SCENARIO_RANDOM,
    SCENARIO_HEAD_ON,
    SCENARIO_ANGLED_30,
    SCENARIO_ANGLED_45,
    SCENARIO_GLANCING,
    SCENARIO_SPIN_DOMINANT,
    SCENARIO_COUNT
} CollisionScenario;

// Generate specific collision scenario
void generate_collision_scenario(double* rvw1, double* rvw2, CollisionScenario scenario) {
    // Default values
    double radius = 0.0571; // Standard pool ball radius in meters
    
    // Reset arrays
    for (int i = 0; i < 9; i++) {
        rvw1[i] = 0.0;
        rvw2[i] = 0.0;
    }
    
    // Set scenario-specific values
    switch (scenario) {
        case SCENARIO_RANDOM:
            // Random position
            for (int i = 0; i < 3; i++) {
                rvw1[i] = (double)rand() / RAND_MAX * 10.0 - 5.0;  // -5.0 to 5.0
                rvw2[i] = (double)rand() / RAND_MAX * 10.0 - 5.0;
            }
            
            // Keep balls close enough to collide
            double dx = rvw2[0] - rvw1[0];
            double dy = rvw2[1] - rvw1[1];
            double dist = sqrt(dx*dx + dy*dy);
            if (dist > 0.1) {
                rvw2[0] = rvw1[0] + 0.1 * dx / dist;
                rvw2[1] = rvw1[1] + 0.1 * dy / dist;
            }
            
            // Random velocity
            for (int i = 3; i < 6; i++) {
                rvw1[i] = (double)rand() / RAND_MAX * 4.0 - 2.0;  // -2.0 to 2.0
                rvw2[i] = (double)rand() / RAND_MAX * 4.0 - 2.0;
            }
            
            // Random angular velocity
            for (int i = 6; i < 9; i++) {
                rvw1[i] = (double)rand() / RAND_MAX * 10.0 - 5.0;  // -5.0 to 5.0
                rvw2[i] = (double)rand() / RAND_MAX * 10.0 - 5.0;
            }
            break;
            
        case SCENARIO_HEAD_ON:
            // Ball 1 at origin, Ball 2 at (2*radius, 0, 0) - just touching
            rvw1[0] = 0.0; rvw1[1] = 0.0; rvw1[2] = 0.0;
            rvw2[0] = 2.01*radius; rvw2[1] = 0.0; rvw2[2] = 0.0;
            
            // Velocity: Ball 1 stationary, Ball 2 moving directly toward Ball 1
            rvw1[3] = 0.0; rvw1[4] = 0.0; rvw1[5] = 0.0;
            rvw2[3] = -1.0; rvw2[4] = 0.0; rvw2[5] = 0.0;
            
            // No initial spin
            rvw1[6] = 0.0; rvw1[7] = 0.0; rvw1[8] = 0.0;
            rvw2[6] = 0.0; rvw2[7] = 0.0; rvw2[8] = 0.0;
            break;
            
        case SCENARIO_ANGLED_30:
            // Ball 1 at origin, Ball 2 at an angle
            rvw1[0] = 0.0; rvw1[1] = 0.0; rvw1[2] = 0.0;
            rvw2[0] = 2.01*radius*cos(30.0*M_PI/180.0); 
            rvw2[1] = 2.01*radius*sin(30.0*M_PI/180.0); 
            rvw2[2] = 0.0;
            
            // Velocity: Ball 1 stationary, Ball 2 moving toward Ball 1
            rvw1[3] = 0.0; rvw1[4] = 0.0; rvw1[5] = 0.0;
            rvw2[3] = -cos(30.0*M_PI/180.0); 
            rvw2[4] = -sin(30.0*M_PI/180.0); 
            rvw2[5] = 0.0;
            
            // Minimal spin
            rvw1[6] = 0.0; rvw1[7] = 0.0; rvw1[8] = 0.0;
            rvw2[6] = 0.0; rvw2[7] = 0.0; rvw2[8] = 0.0;
            break;
            
        case SCENARIO_ANGLED_45:
            // 45-degree angle collision
            rvw1[0] = 0.0; rvw1[1] = 0.0; rvw1[2] = 0.0;
            rvw2[0] = 2.01*radius*cos(45.0*M_PI/180.0); 
            rvw2[1] = 2.01*radius*sin(45.0*M_PI/180.0); 
            rvw2[2] = 0.0;
            
            // Velocity: Ball 1 stationary, Ball 2 moving toward Ball 1
            rvw1[3] = 0.0; rvw1[4] = 0.0; rvw1[5] = 0.0;
            rvw2[3] = -cos(45.0*M_PI/180.0); 
            rvw2[4] = -sin(45.0*M_PI/180.0); 
            rvw2[5] = 0.0;
            
            // Minimal spin
            rvw1[6] = 0.0; rvw1[7] = 0.0; rvw1[8] = 0.0;
            rvw2[6] = 0.0; rvw2[7] = 0.0; rvw2[8] = 0.0;
            break;
            
        case SCENARIO_GLANCING:
            // Glancing hit (near miss adjusted to just touch)
            rvw1[0] = 0.0; rvw1[1] = 0.0; rvw1[2] = 0.0;
            rvw2[0] = 2.01 * radius; rvw2[1] = 1.95*radius; rvw2[2] = 0.0;
            
            // Velocity: Both balls moving
            rvw1[3] = 0.5; rvw1[4] = 0.0; rvw1[5] = 0.0;
            rvw2[3] = -0.5; rvw2[4] = 0.0; rvw2[5] = 0.0;
            
            // Some initial spin
            rvw1[6] = 0.0; rvw1[7] = 0.0; rvw1[8] = 2.0;
            rvw2[6] = 0.0; rvw2[7] = 0.0; rvw2[8] = -2.0;
            break;
            
        case SCENARIO_SPIN_DOMINANT:
            // Ball 1 at origin, Ball 2 nearby
            rvw1[0] = 0.0; rvw1[1] = 0.0; rvw1[2] = 0.0;
            rvw2[0] = 2.01*radius; rvw2[1] = 0.0; rvw2[2] = 0.0;
            
            // Low velocity
            rvw1[3] = 0.1; rvw1[4] = 0.0; rvw1[5] = 0.0;
            rvw2[3] = -0.1; rvw2[4] = 0.0; rvw2[5] = 0.0;
            
            // High spin
            rvw1[6] = 0.0; rvw1[7] = 0.0; rvw1[8] = 5.0;
            rvw2[6] = 0.0; rvw2[7] = 0.0; rvw2[8] = -5.0;
            break;
            
        default:
            fprintf(stderr, "Unknown scenario: %d\n", scenario);
            break;
    }
}

// Get scenario name
const char* get_scenario_name(CollisionScenario scenario) {
    switch (scenario) {
        case SCENARIO_RANDOM: return "Random";
        case SCENARIO_HEAD_ON: return "Head-on";
        case SCENARIO_ANGLED_30: return "30deg Angle";
        case SCENARIO_ANGLED_45: return "45deg Angle";
        case SCENARIO_GLANCING: return "Glancing";
        case SCENARIO_SPIN_DOMINANT: return "Spin-dominant";
        default: return "Unknown";
    }
}

// Run a single benchmark with specific parameters
double run_benchmark(int iterations, int collision_iterations, float deltaP_value, CollisionScenario scenario) {
    double* rvw1 = (double*)malloc(9 * sizeof(double));
    double* rvw2 = (double*)malloc(9 * sizeof(double));
    double* rvw1_result = (double*)malloc(9 * sizeof(double));
    double* rvw2_result = (double*)malloc(9 * sizeof(double));
    
    // Parameters for collision
    float radius = 0.0571; // Standard pool ball radius in meters
    float mass = 0.17;     // Standard pool ball mass in kg
    float u_s1 = 0.1;      // Sliding friction coefficient for ball 1
    float u_s2 = 0.1;      // Sliding friction coefficient for ball 2
    float u_b = 0.01;      // Ball-ball friction coefficient
    float e_b = 0.95;      // Ball-ball restitution coefficient
    float deltaP = deltaP_value;
    
    // Timing variables
    double start, end, total_time = 0.0;
    
    // Run warmup iterations (not counted in timing)
    for (int i = 0; i < 5; i++) {
        generate_collision_scenario(rvw1, rvw2, scenario);
        memcpy(rvw1_result, rvw1, 9 * sizeof(double));
        memcpy(rvw2_result, rvw2, 9 * sizeof(double));
        collide_balls(rvw1, rvw2, radius, mass, u_s1, u_s2, u_b, e_b, deltaP, 
                      collision_iterations, rvw1_result, rvw2_result);
    }
    
    // Run timed iterations
    for (int i = 0; i < iterations; i++) {
        generate_collision_scenario(rvw1, rvw2, scenario);
        memcpy(rvw1_result, rvw1, 9 * sizeof(double));
        memcpy(rvw2_result, rvw2, 9 * sizeof(double));
        
        start = get_time();
        collide_balls(rvw1, rvw2, radius, mass, u_s1, u_s2, u_b, e_b, deltaP, 
                      collision_iterations, rvw1_result, rvw2_result);
        end = get_time();
        
        total_time += (end - start);
    }
    
    free(rvw1);
    free(rvw2);
    free(rvw1_result);
    free(rvw2_result);
    
    return total_time / iterations; // Return average time per iteration
}

int main(int argc, char* argv[]) {
    // Parse command line arguments
    int iterations = 100;
    int collision_iterations = 10;
    int run_scenario = -1;  // -1 means run all scenarios
    float test_deltaP = 0.0;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) {
            iterations = atoi(argv[i+1]);
            i++;
        } else if (strcmp(argv[i], "--collision-iterations") == 0 && i + 1 < argc) {
            collision_iterations = atoi(argv[i+1]);
            i++;
        } else if (strcmp(argv[i], "--scenario") == 0 && i + 1 < argc) {
            run_scenario = atoi(argv[i+1]);
            i++;
        } else if (strcmp(argv[i], "--deltaP") == 0 && i + 1 < argc) {
            test_deltaP = atof(argv[i+1]);
            i++;
        } else if (i == 1) {
            iterations = atoi(argv[i]);
        } else if (i == 2) {
            collision_iterations = atoi(argv[i]);
        }
    }
    
    // Seed random number generator
    srand(time(NULL));
    
    printf("Running benchmark with %d iterations, %d collision iterations per call\n", 
           iterations, collision_iterations);
    
    // Define scenarios to run
    CollisionScenario scenarios[SCENARIO_COUNT] = {
        SCENARIO_RANDOM,
        SCENARIO_HEAD_ON,
        SCENARIO_ANGLED_30,
        SCENARIO_ANGLED_45,
        SCENARIO_GLANCING,
        SCENARIO_SPIN_DOMINANT
    };
    
    // ======= SCENARIO BENCHMARKS =======
    if (run_scenario == -1) {
        printf("\n# Collision Scenario Benchmarks,deltaP=%.4f\n", test_deltaP);
        printf("Scenario,AverageTime(s)\n");
        
        for (int i = 0; i < SCENARIO_COUNT; i++) {
            double avg_time = run_benchmark(iterations, collision_iterations, test_deltaP, scenarios[i]);
            printf("%s,%.9f\n", get_scenario_name(scenarios[i]), avg_time);
        }
    } else if (run_scenario >= 0 && run_scenario < SCENARIO_COUNT) {
        printf("\n=== Benchmarking Scenario: %s (deltaP = %.4f) ===\n", 
               get_scenario_name(scenarios[run_scenario]), test_deltaP);
        double avg_time = run_benchmark(iterations, collision_iterations, test_deltaP, scenarios[run_scenario]);
        printf("Average Time: %.9f seconds\n", avg_time);
    }
    
    // ======= COLLISION ITERATIONS BENCHMARKS =======
    printf("\n# Performance vs Collision Iterations,Scenario=%s,deltaP=%.4f\n",
           get_scenario_name(run_scenario >= 0 ? scenarios[run_scenario] : SCENARIO_RANDOM), test_deltaP);
    printf("CollisionIterations,AverageTime(s)\n");
    
    int collision_iter_values[] = {5, 10, 20, 50, 100};
    for (int i = 0; i < sizeof(collision_iter_values)/sizeof(int); i++) {
        int n = collision_iter_values[i];
        double avg_time = run_benchmark(iterations, n, test_deltaP, 
                                        run_scenario >= 0 ? scenarios[run_scenario] : SCENARIO_RANDOM);
        printf("%d,%.9f\n", n, avg_time);
    }
    
    // ======= DELTA P BENCHMARKS =======
    printf("\n# Performance vs deltaP Values,Scenario=%s,N=%d\n",
           get_scenario_name(run_scenario >= 0 ? scenarios[run_scenario] : SCENARIO_HEAD_ON), 
           collision_iterations);
    printf("deltaP,AverageTime(s)\n");
    
    float deltaP_values[] = {0.0, 0.001, 0.005, 0.01, 0.05};
    for (int i = 0; i < sizeof(deltaP_values)/sizeof(float); i++) {
        float dp = deltaP_values[i];
        double avg_time = run_benchmark(iterations, collision_iterations, dp, 
                                        run_scenario >= 0 ? scenarios[run_scenario] : SCENARIO_HEAD_ON);
        printf("%.4f,%.9f\n", dp, avg_time);
    }
    
    // ======= CONSISTENCY TEST =======
    printf("\n# Consistency Test,Scenario=%s,N=%d,deltaP=%.4f\n",
           get_scenario_name(run_scenario >= 0 ? scenarios[run_scenario] : SCENARIO_HEAD_ON), 
           collision_iterations, test_deltaP);
    printf("Run,AverageTime(s)\n");
    
    double total = 0.0;
    double min_time = 1e10;
    double max_time = 0.0;
    
    for (int i = 0; i < 5; i++) {
        double avg_time = run_benchmark(iterations, collision_iterations, test_deltaP, 
                                        run_scenario >= 0 ? scenarios[run_scenario] : SCENARIO_HEAD_ON);
        total += avg_time;
        if (avg_time < min_time) min_time = avg_time;
        if (avg_time > max_time) max_time = avg_time;
        printf("%d,%.9f\n", i+1, avg_time);
    }
    
    printf("\nSummary:\n");
    printf("  Average time: %.9f seconds\n", total / 5);
    printf("  Min time: %.9f seconds\n", min_time);
    printf("  Max time: %.9f seconds\n", max_time);
    
    return 0;
}