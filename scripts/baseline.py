#!/usr/bin/env python3

import os
import sys
from utils import benchmark_program, save_results_to_csv, create_performance_plot


def main():
    # Configuration
    BENCHMARKS_DIR = "../benchmarks"
    POLYBENCH_DIR = os.path.join(BENCHMARKS_DIR, "polybench-c-4.2.1-beta")
    UTILITIES_DIR = os.path.join(POLYBENCH_DIR, "utilities")

    # PolyBench programs to benchmark
    programs = [
        {
            "name": "2mm",
            "path": os.path.join(POLYBENCH_DIR, "linear-algebra/kernels/2mm/2mm.c"),
        },
        {
            "name": "gemm",
            "path": os.path.join(POLYBENCH_DIR, "linear-algebra/kernels/gemm/gemm.c"),
        },
        {
            "name": "syr2k",
            "path": os.path.join(POLYBENCH_DIR, "linear-algebra/kernels/syr2k/syr2k.c"),
        },
    ]

    # Optimization flags to test
    optimization_flags = ["-O2", "-O3"]

    # Output files
    csv_output = "baseline.csv"
    plot_output = "baseline_performance.png"

    print("=== PolyBench Baseline Benchmarking ===\n")

    # Remove existing CSV to start fresh
    if os.path.exists(csv_output):
        os.remove(csv_output)

    # Benchmark each program
    for program in programs:
        print(f"Benchmarking {program['name']}...")

        if not os.path.exists(program["path"]):
            print(f"  Warning: {program['path']} not found, skipping...")
            continue

        try:
            results = benchmark_program(
                source_file=program["path"],
                optimization_flags=optimization_flags,
                include_path=UTILITIES_DIR,
                runs=3,
            )

            save_results_to_csv(results, csv_output, program["name"])
            print(f"  ✅ Completed {program['name']}\n")

        except Exception as e:
            print(f"  ❌ Error benchmarking {program['name']}: {e}\n")

    # Create performance plots
    if os.path.exists(csv_output):
        print("Creating performance plots...")
        try:
            create_performance_plot(csv_output, plot_output)
            print("✅ Baseline benchmarking complete!")
            print(f"📊 Results saved to: {csv_output}")
            print(f"📈 Plot saved to: {plot_output}")
        except Exception as e:
            print(f"❌ Error creating plots: {e}")
    else:
        print("❌ No results to plot")


if __name__ == "__main__":
    main()
