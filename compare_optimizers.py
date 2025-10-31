#!/usr/bin/env python3
"""
Comparison script for FOGA vs HBRF optimization approaches
Runs both optimizers and provides detailed comparative analysis
"""

import subprocess
import time
import json
import sys
import os
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


class OptimizerComparison:
    def __init__(self, source_file, test_input_file=None):
        self.source_file = source_file
        self.test_input_file = test_input_file
        self.results = {
            'FOGA': {},
            'HBRF': {},
            'baseline': {}
        }

    def run_baseline_benchmarks(self):
        """Run baseline -O1, -O2, -O3 benchmarks"""
        print("=" * 80)
        print("🎯 RUNNING BASELINE BENCHMARKS (-O1, -O2, -O3)")
        print("=" * 80)

        compiler = 'gcc' if self.source_file.endswith('.c') else 'g++'
        baseline_times = {}

        for opt_level in ['-O1', '-O2', '-O3']:
            binary = f"baseline{opt_level}"
            compile_cmd = f"{compiler} {opt_level} {self.source_file} -o {binary}"

            try:
                result = subprocess.run(compile_cmd, shell=True, capture_output=True, timeout=30)
                if result.returncode != 0:
                    print(f"  {opt_level}: Compilation failed")
                    baseline_times[opt_level] = float('inf')
                    continue

                test_input = None
                if self.test_input_file and os.path.exists(self.test_input_file):
                    with open(self.test_input_file, 'r') as f:
                        test_input = f.read()

                start = time.time()
                exec_result = subprocess.run(
                    f"./{binary}",
                    shell=True,
                    input=test_input,
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                exec_time = time.time() - start

                if exec_result.returncode == 0:
                    baseline_times[opt_level] = exec_time
                    print(f"  {opt_level}: {exec_time:.6f}s")
                else:
                    baseline_times[opt_level] = float('inf')
                    print(f"  {opt_level}: Execution failed")

                if os.path.exists(binary):
                    os.remove(binary)

            except Exception as e:
                print(f"  {opt_level}: Error - {e}")
                baseline_times[opt_level] = float('inf')

        self.results['baseline'] = baseline_times
        return baseline_times

    def run_foga(self):
        """Run FOGA optimizer (streaming output live)"""
        print("\n" + "=" * 80)
        print("🧬 RUNNING FOGA (Genetic Algorithm)")
        print("=" * 80)

        cmd = ['python3', '-u', 'foga.py', self.source_file]
        if self.test_input_file:
            cmd.append(self.test_input_file)

        start_time = time.time()
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            output_lines = []
            best_time = float('inf')
            evaluations = 0

            for line in process.stdout:
                line = line.rstrip()
                print(line, flush=True)
                output_lines.append(line)

                if 'Best Execution Time:' in line:
                    try:
                        best_time = float(line.split(':')[1].strip().split()[0])
                    except:
                        pass
                if 'Gen' in line and 'Best:' in line:
                    evaluations += 1

            process.wait(timeout=3600)
            total_time = time.time() - start_time
            output = "\n".join(output_lines)

            self.results['FOGA'] = {
                'best_time': best_time,
                'total_time': total_time,
                'evaluations': evaluations,
                'output': output
            }

            print(f"\n✅ FOGA Complete: {best_time:.6f}s in {total_time:.2f}s")
            return best_time, total_time

        except subprocess.TimeoutExpired:
            process.kill()
            print("❌ FOGA timed out after 1 hour")
            return float('inf'), 3600
        except Exception as e:
            print(f"❌ FOGA failed: {e}")
            return float('inf'), 0

    def run_hbrf(self):
        """Run HBRF optimizer (streaming output live)"""
        print("\n" + "=" * 80)
        print("🔬 RUNNING HBRF (Hybrid Bayesian-RF)")
        print("=" * 80)

        cmd = ['python3', '-u', 'hbrf_optimizer.py', self.source_file]
        if self.test_input_file:
            cmd.append(self.test_input_file)

        start_time = time.time()
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            output_lines = []
            best_time = float('inf')
            evaluations = 0

            for line in process.stdout:
                line = line.rstrip()
                print(line, flush=True)
                output_lines.append(line)

                if 'Best Execution Time:' in line:
                    try:
                        best_time = float(line.split(':')[1].strip().split()[0])
                    except:
                        pass
                if 'Total Evaluations:' in line:
                    try:
                        evaluations = int(line.split(':')[1].strip())
                    except:
                        pass

            process.wait(timeout=3600)
            total_time = time.time() - start_time
            output = "\n".join(output_lines)

            if os.path.exists('hbrf_results.json'):
                with open('hbrf_results.json', 'r') as f:
                    hbrf_data = json.load(f)
                    best_time = hbrf_data.get('best_time', best_time)
                    evaluations = hbrf_data.get('total_evaluations', evaluations)

            self.results['HBRF'] = {
                'best_time': best_time,
                'total_time': total_time,
                'evaluations': evaluations,
                'output': output
            }

            print(f"\n✅ HBRF Complete: {best_time:.6f}s in {total_time:.2f}s")
            return best_time, total_time

        except subprocess.TimeoutExpired:
            process.kill()
            print("❌ HBRF timed out after 1 hour")
            return float('inf'), 3600
        except Exception as e:
            print(f"❌ HBRF failed: {e}")
            return float('inf'), 0

    def generate_comparison_report(self):
        """Generate comprehensive comparison report"""
        print("\n" + "=" * 80)
        print("📊 COMPREHENSIVE COMPARISON REPORT")
        print("=" * 80)

        baseline = self.results['baseline']
        foga = self.results['FOGA']
        hbrf = self.results['HBRF']

        print("\n1️⃣  EXECUTION TIME COMPARISON")
        print("-" * 80)
        print(f"{'Method':<20} | {'Time (s)':<15} | {'Speedup vs -O3':<20} | {'Rank':<10}")
        print("-" * 80)

        o3_time = baseline.get('-O3', float('inf'))

        times_dict = {
            '-O1': baseline.get('-O1', float('inf')),
            '-O2': baseline.get('-O2', float('inf')),
            '-O3': o3_time,
            'FOGA': foga.get('best_time', float('inf')),
            'HBRF': hbrf.get('best_time', float('inf'))
        }

        ranked = sorted(times_dict.items(), key=lambda x: x[1])

        for rank, (method, exec_time) in enumerate(ranked, 1):
            if exec_time == float('inf'):
                speedup_str = "Failed"
            elif o3_time == float('inf') or o3_time == 0:
                speedup_str = "N/A"
            else:
                speedup = ((o3_time - exec_time) / o3_time) * 100
                speedup_str = f"{speedup:+.2f}%"

            time_str = f"{exec_time:.6f}" if exec_time != float('inf') else "Failed"
            print(f"{method:<20} | {time_str:<15} | {speedup_str:<20} | #{rank}")

        print("\n2️⃣  OPTIMIZATION TIME COMPARISON")
        print("-" * 80)
        print(f"{'Method':<20} | {'Opt. Time (s)':<15} | {'Evaluations':<15} | {'Eval/sec':<10}")
        print("-" * 80)

        foga_time = foga.get('total_time', 0)
        hbrf_time = hbrf.get('total_time', 0)
        foga_evals = foga.get('evaluations', 0)
        hbrf_evals = hbrf.get('evaluations', 0)

        foga_eps = foga_evals / foga_time if foga_time > 0 else 0
        hbrf_eps = hbrf_evals / hbrf_time if hbrf_time > 0 else 0

        print(f"{'FOGA':<20} | {foga_time:<15.2f} | {foga_evals:<15} | {foga_eps:<10.2f}")
        print(f"{'HBRF':<20} | {hbrf_time:<15.2f} | {hbrf_evals:<15} | {hbrf_eps:<10.2f}")

        print("\n3️⃣  EFFICIENCY METRICS")
        print("-" * 80)

        if foga.get('best_time', float('inf')) != float('inf') and o3_time != float('inf'):
            foga_improvement = ((o3_time - foga['best_time']) / o3_time) * 100
        else:
            foga_improvement = 0

        if hbrf.get('best_time', float('inf')) != float('inf') and o3_time != float('inf'):
            hbrf_improvement = ((o3_time - hbrf['best_time']) / o3_time) * 100
        else:
            hbrf_improvement = 0

        print(f"FOGA Improvement over -O3: {foga_improvement:+.2f}%")
        print(f"HBRF Improvement over -O3: {hbrf_improvement:+.2f}%")

        # if foga_evals > 0 and hbrf_evals > 0:
        #     eval_ratio = foga_evals / hbrf_evals
        #     print(f"\nHBRF used {eval_ratio:.2f}x fewer evaluations than FOGA")

        # if foga_time > 0 and hbrf_time > 0:
        #     time_ratio = foga_time / hbrf_time
        #     print(f"HBRF was {time_ratio:.2f}x faster in optimization time")

        print("\n4️⃣  WINNER ANALYSIS")
        print("-" * 80)

        foga_best = foga.get('best_time', float('inf'))
        hbrf_best = hbrf.get('best_time', float('inf'))

        if foga_best < hbrf_best:
            winner = "FOGA"
            margin = ((hbrf_best - foga_best) / hbrf_best) * 100
            print(f"🏆 WINNER: FOGA")
            print(f"   FOGA achieved {margin:.2f}% better execution time")
        elif hbrf_best < foga_best:
            winner = "HBRF"
            margin = ((foga_best - hbrf_best) / foga_best) * 100
            print(f"🏆 WINNER: HBRF")
            print(f"   HBRF achieved {margin:.2f}% better execution time")
        else:
            winner = "TIE"
            print(f"🤝 RESULT: TIE")
            print(f"   Both methods achieved the same execution time")

        # if hbrf_time < foga_time:
        #     time_saved = foga_time - hbrf_time
        #     print(f"\n⚡ HBRF saved {time_saved:.2f}s in optimization time")
        #     print(f"   ({((foga_time - hbrf_time) / foga_time * 100):.1f}% faster)")

        comparison_data = {
            'timestamp': datetime.now().isoformat(),
            'source_file': self.source_file,
            'baseline': baseline,
            'FOGA': {
                'best_time': foga_best,
                'total_time': foga_time,
                'evaluations': foga_evals
            },
            'HBRF': {
                'best_time': hbrf_best,
                'total_time': hbrf_time,
                'evaluations': hbrf_evals
            },
            'winner': winner,
            'improvements': {
                'FOGA_vs_O3': foga_improvement,
                'HBRF_vs_O3': hbrf_improvement
            }
        }

        with open('comparison_results.json', 'w') as f:
            json.dump(comparison_data, f, indent=2)

        print("\n📄 Comparison results saved to: comparison_results.json")
        self.generate_visualizations(times_dict, foga_time, hbrf_time)

    def generate_visualizations(self, times_dict, foga_opt_time, hbrf_opt_time):
        """Generate comparison charts"""
        try:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            methods = list(times_dict.keys())
            times = [times_dict[m] if times_dict[m] != float('inf') else 0 for m in methods]
            colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6']

            axes[0].bar(methods, times, color=colors)
            axes[0].set_ylabel('Execution Time (seconds)')
            axes[0].set_title('Execution Time Comparison')
            axes[0].grid(axis='y', alpha=0.3)

            opt_methods = ['FOGA', 'HBRF']
            opt_times = [foga_opt_time, hbrf_opt_time]

            axes[1].bar(opt_methods, opt_times, color=['#e74c3c', '#9b59b6'])
            axes[1].set_ylabel('Optimization Time (seconds)')
            axes[1].set_title('Optimization Time Comparison')
            axes[1].grid(axis='y', alpha=0.3)

            plt.tight_layout()
            plt.savefig('comparison_chart.png', dpi=150, bbox_inches='tight')
            print("📊 Visualization saved to: comparison_chart.png")

        except Exception as e:
            print(f"⚠ Could not generate visualizations: {e}")

    def run_full_comparison(self):
        """Run complete comparison pipeline"""
        print("=" * 80)
        print("🚀 STARTING COMPREHENSIVE OPTIMIZATION COMPARISON")
        print("=" * 80)
        print(f"Source file: {self.source_file}")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

        self.run_baseline_benchmarks()
        self.run_foga()
        self.run_hbrf()
        self.generate_comparison_report()

        print("\n" + "=" * 80)
        print("✅ COMPARISON COMPLETE")
        print("=" * 80)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python compare_optimizers.py <source_file> [test_input_file]")
        sys.exit(1)

    source_file = sys.argv[1]
    test_input = sys.argv[2] if len(sys.argv) > 2 else None

    if not os.path.exists(source_file):
        print(f"❌ Error: Source file '{source_file}' not found")
        sys.exit(1)

    comparison = OptimizerComparison(source_file, test_input)
    comparison.run_full_comparison()
