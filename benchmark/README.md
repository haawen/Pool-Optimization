# Benchmarking Suite

This directory contains the benchmarking tools for the pool physics simulation.

## Building the Benchmarks

From the benchmark directory:

```bash
# Create and enter build directory
mkdir -p build
cd build

# Configure and build
cmake ..
make
```

## Running Benchmarks

After building, you can run the benchmarks in several ways:

1. Run all benchmarks and generate plots:
```bash
make run_benchmarks
```

2. Run specific benchmark scenarios:
```bash
./pool_benchmark --scenario 0  # Random collision
./pool_benchmark --scenario 1  # Head-on collision
./pool_benchmark --scenario 2  # 30° angle collision
./pool_benchmark --scenario 3  # 45° angle collision
./pool_benchmark --scenario 4  # Glancing collision
./pool_benchmark --scenario 5  # Spin-dominant collision
```

3. Run with specific parameters:
```bash
./pool_benchmark --iterations 1000 --collision-iterations 20 --deltaP 0.001
```

## Output

The benchmarks generate:
1. JSON output file with detailed results
2. PNG plots showing:
   - Performance across different scenarios
   - Scaling with collision iterations
   - Impact of deltaP parameter
   - Consistency test results

## Requirements

- Python 3.x with matplotlib and numpy
- CMake 3.10 or higher
- C compiler with optimization support 