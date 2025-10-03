import subprocess
import time
import os
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple

def compile_polybench_program(source_file: str, optimization_flag: str, include_path: str = None) -> str:
    """
    Compile a PolyBench program with specified optimization flag.
    
    Args:
        source_file: Path to the C source file
        optimization_flag: Optimization flag (e.g., '-O2', '-O3')
        include_path: Path to polybench headers
    
    Returns:
        Path to compiled executable
    """
    if not os.path.exists(source_file):
        raise FileNotFoundError(f"Source file not found: {source_file}")
    
    # Generate executable name
    base_name = os.path.splitext(os.path.basename(source_file))[0]
    executable = f"{base_name}_{optimization_flag.replace('-', '')}"
    
    # Build compilation command
    cmd = ["gcc"]
    
    # Add include path if provided
    if include_path:
        cmd.extend(["-I", include_path])
    
    # Add optimization flag
    cmd.append(optimization_flag)
    
    # Add source file and output
    cmd.extend([source_file, "-o", executable])
    
    # Add math library for PolyBench
    cmd.append("-lm")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return executable
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Compilation failed: {e.stderr}")

def run_and_time_program(executable: str, runs: int = 3) -> float:
    """
    Run executable multiple times and return average execution time.
    
    Args:
        executable: Path to executable
        runs: Number of runs for averaging
    
    Returns:
        Average execution time in seconds
    """
    if not os.path.exists(executable):
        raise FileNotFoundError(f"Executable not found: {executable}")
    
    times = []
    for _ in range(runs):
        start_time = time.time()
        try:
            subprocess.run([f"./{executable}"], check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Execution failed: {e}")
        end_time = time.time()
        times.append(end_time - start_time)
    
    return sum(times) / len(times)

def benchmark_program(source_file: str, optimization_flags: List[str], 
                     include_path: str = None, runs: int = 3) -> List[Tuple[str, float]]:
    """
    Benchmark a program with different optimization flags.
    
    Args:
        source_file: Path to C source file
        optimization_flags: List of optimization flags to test
        include_path: Path to header files
        runs: Number of runs per optimization level
    
    Returns:
        List of (optimization_flag, execution_time) tuples
    """
    results = []
    
    for flag in optimization_flags:
        print(f"Benchmarking with {flag}...")
        
        # Compile
        executable = compile_polybench_program(source_file, flag, include_path)
        
        try:
            # Run and time
            avg_time = run_and_time_program(executable, runs)
            results.append((flag, avg_time))
            print(f"  Average time: {avg_time:.6f} seconds")
        finally:
            # Cleanup executable
            if os.path.exists(executable):
                os.remove(executable)
    
    return results

def save_results_to_csv(results: List[Tuple[str, float]], output_file: str, program_name: str = None):
    """
    Save benchmark results to CSV file.
    
    Args:
        results: List of (optimization_flag, execution_time) tuples
        output_file: Path to output CSV file
        program_name: Name of the benchmarked program
    """
    data = {
        'program': [program_name or 'unknown'] * len(results),
        'optimization_flag': [flag for flag, _ in results],
        'execution_time': [time for _, time in results]
    }
    
    df = pd.DataFrame(data)
    
    # Append to existing file or create new one
    if os.path.exists(output_file):
        existing_df = pd.read_csv(output_file)
        df = pd.concat([existing_df, df], ignore_index=True)
    
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

def create_performance_plot(csv_file: str, output_plot: str = None):
    """
    Create performance comparison plots from CSV data.
    
    Args:
        csv_file: Path to CSV file with results
        output_plot: Path to save plot (optional)
    """
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file not found: {csv_file}")
    
    df = pd.read_csv(csv_file)
    
    if df.empty:
        raise ValueError("CSV file is empty")
    
    # Create plot
    plt.figure(figsize=(10, 6))
    
    if 'program' in df.columns:
        # Group by program if multiple programs
        for program in df['program'].unique():
            program_data = df[df['program'] == program]
            plt.plot(program_data['optimization_flag'], program_data['execution_time'], 
                    marker='o', label=program, linewidth=2, markersize=8)
    else:
        plt.plot(df['optimization_flag'], df['execution_time'], 
                marker='o', linewidth=2, markersize=8)
    
    plt.xlabel('Optimization Flag')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Performance Comparison: Different Optimization Levels')
    plt.grid(True, alpha=0.3)
    
    if 'program' in df.columns and len(df['program'].unique()) > 1:
        plt.legend()
    
    plt.tight_layout()
    
    if output_plot:
        plt.savefig(output_plot, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_plot}")
    
    plt.show()