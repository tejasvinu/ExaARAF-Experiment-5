#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demonstration of HPC Monte Carlo Experiment Optimization
========================================================

This script demonstrates the autonomous optimization capabilities
of the Monte Carlo π estimation experiment.
"""

import subprocess
import sys
import time
import json
from pathlib import Path


def run_experiment(config_args, description=""):
    """Run a single experiment with given configuration."""
    print(f"\n--- {description} ---")
    
    cmd = [sys.executable, "main.py", "--single-run"] + config_args
    
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    end_time = time.time()
    
    if result.returncode == 0:
        # Extract results
        lines = result.stdout.strip().split('\n')
        final_result = {}
        for line in lines:
            if "PI estimate:" in line:
                final_result['pi_estimate'] = float(line.split(':')[1].strip())
            elif "Error:" in line:
                final_result['error'] = float(line.split(':')[1].strip())
            elif "Execution time:" in line:
                final_result['exec_time'] = float(line.split(':')[1].replace('s', '').strip())
            elif "Samples/sec:" in line:
                final_result['samples_per_sec'] = float(line.split(':')[1].strip())
        
        print(f"Results: PI={final_result.get('pi_estimate', 0):.6f}, "
              f"Error={final_result.get('error', 0):.6f}, "
              f"Time={final_result.get('exec_time', 0):.3f}s, "
              f"Rate={final_result.get('samples_per_sec', 0):.0f} samples/s")
        
        return final_result
    else:
        print(f"Experiment failed: {result.stderr}")
        return None


def demonstrate_baseline():
    """Demonstrate baseline performance."""
    config = [
        "--total-samples", "1000000",
        "--num-workers", "4",
        "--parallel-strategy", "multiprocessing"
    ]
    return run_experiment(config, "Baseline Configuration")


def demonstrate_optimization_dimensions():
    """Demonstrate different optimization dimensions."""
    
    print("\n🔬 OPTIMIZATION DIMENSION EXPLORATION")
    print("=" * 50)
    
    results = []
    
    # 1. Parallelization strategy comparison
    print("\n1. Parallelization Strategy Comparison:")
    strategies = ["thread", "multiprocessing", "vectorized"]
    
    for strategy in strategies:
        config = [
            "--total-samples", "500000",
            "--num-workers", "4",
            "--parallel-strategy", strategy
        ]
        result = run_experiment(config, f"{strategy.title()} Strategy")
        if result:
            result['strategy'] = strategy
            results.append(result)
    
    # 2. Worker count scaling
    print("\n2. Worker Count Scaling:")
    worker_counts = [1, 2, 4, 8]
    
    for workers in worker_counts:
        config = [
            "--total-samples", "500000",
            "--num-workers", str(workers),
            "--parallel-strategy", "multiprocessing"
        ]
        result = run_experiment(config, f"{workers} Workers")
        if result:
            result['workers'] = workers
            results.append(result)
    
    # 3. Precision comparison
    print("\n3. Precision Comparison:")
    precisions = ["float32", "float64"]
    
    for precision in precisions:
        config = [
            "--total-samples", "500000",
            "--num-workers", "4",
            "--precision", precision
        ]
        result = run_experiment(config, f"{precision.upper()} Precision")
        if result:
            result['precision'] = precision
            results.append(result)
    
    # 4. Sample size scaling
    print("\n4. Sample Size Scaling:")
    sample_sizes = [100000, 500000, 1000000, 2000000]
    
    for samples in sample_sizes:
        config = [
            "--total-samples", str(samples),
            "--num-workers", "4"
        ]
        result = run_experiment(config, f"{samples:,} Samples")
        if result:
            result['samples'] = samples
            results.append(result)
    
    return results


def demonstrate_autonomous_optimization():
    """Demonstrate simplified autonomous optimization."""
    
    print("\n🤖 AUTONOMOUS OPTIMIZATION SIMULATION")
    print("=" * 50)
    
    # Simplified optimization search
    search_space = [
        {"workers": 2, "batch": 5000, "strategy": "thread"},
        {"workers": 4, "batch": 10000, "strategy": "multiprocessing"},
        {"workers": 8, "batch": 20000, "strategy": "multiprocessing"},
        {"workers": 4, "batch": 50000, "strategy": "vectorized"},
    ]
    
    best_score = float('inf')
    best_config = None
    best_result = None
    
    print("Searching optimization space...")
    
    for i, config in enumerate(search_space, 1):
        print(f"\nIteration {i}: Testing {config}")
        
        cmd_args = [
            "--total-samples", "1000000",
            "--num-workers", str(config['workers']),
            "--batch-size", str(config['batch']),
            "--parallel-strategy", config['strategy']
        ]
        
        result = run_experiment(cmd_args, f"Optimization Trial {i}")
        
        if result:
            # Calculate optimization score (minimize time while maintaining accuracy)
            if result.get('error', 1) < 0.01:  # Accuracy threshold
                score = result.get('exec_time', float('inf'))
            else:
                score = result.get('exec_time', float('inf')) * 10  # Penalty for poor accuracy
            
            print(f"Score: {score:.4f}")
            
            if score < best_score:
                best_score = score
                best_config = config
                best_result = result
                print("🏆 New best configuration found!")
    
    print(f"\n🎯 OPTIMIZATION COMPLETE")
    print(f"Best Configuration: {best_config}")
    print(f"Best Score: {best_score:.4f}")
    if best_result:
        print(f"Best Result: PI={best_result.get('pi_estimate', 0):.6f}, "
              f"Error={best_result.get('error', 0):.6f}, "
              f"Time={best_result.get('exec_time', 0):.3f}s")
    
    return best_config, best_result


def demonstrate_advanced_optimization():
    """Demonstrate using the optimizer.py script."""
    
    print("\n⚡ ADVANCED OPTIMIZATION WITH AGENT")
    print("=" * 50)
    
    print("Running automated optimization agent...")
    print("(This simulates what an autonomous agent could do)")
    
    # Create a simple input file for the optimizer
    with open("optimizer_input.txt", "w") as f:
        f.write("grid\n")  # Choose grid search
    
    try:
        cmd = [sys.executable, "optimizer.py"]
        result = subprocess.run(
            cmd, 
            input="grid\n", 
            text=True, 
            capture_output=True, 
            timeout=120
        )
        
        if result.returncode == 0:
            print("✅ Automated optimization completed successfully!")
            print("\nOptimization output (last 10 lines):")
            lines = result.stdout.strip().split('\n')
            for line in lines[-10:]:
                print(f"  {line}")
        else:
            print(f"❌ Optimization failed: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        print("⏰ Optimization timed out (this is normal for demo)")
    except Exception as e:
        print(f"❌ Error running optimizer: {e}")
    finally:
        # Clean up
        if Path("optimizer_input.txt").exists():
            Path("optimizer_input.txt").unlink()


def main():
    """Main demonstration routine."""
    print("🧮 HPC Monte Carlo Experiment Optimization Demo")
    print("=" * 60)
    print("This demonstration shows how an autonomous agent can")
    print("optimize HPC tasks through configuration parameter tuning.")
    
    # 1. Baseline demonstration
    baseline = demonstrate_baseline()
    
    # 2. Optimization dimensions
    optimization_results = demonstrate_optimization_dimensions()
    
    # 3. Autonomous optimization simulation
    best_config, best_result = demonstrate_autonomous_optimization()
    
    # 4. Advanced optimization (optional)
    print("\n" + "=" * 60)
    response = input("Run advanced optimization demo? (y/n): ").strip().lower()
    if response == 'y':
        demonstrate_advanced_optimization()
    
    # Summary
    print(f"\n📊 DEMONSTRATION SUMMARY")
    print("=" * 60)
    print("The experiment demonstrates multiple optimization opportunities:")
    print("1. ✅ Parallelization strategy selection")
    print("2. ✅ Worker count optimization") 
    print("3. ✅ Precision/accuracy trade-offs")
    print("4. ✅ Sample size scaling analysis")
    print("5. ✅ Autonomous optimization search")
    print("6. ✅ Performance metric tracking")
    
    print(f"\n🎯 Key Insights:")
    print("- Different configurations provide different performance trade-offs")
    print("- Agents can systematically explore the optimization space")
    print("- Clear metrics enable objective performance comparison")
    print("- The task provides rich opportunities for autonomous improvement")
    
    print(f"\n📁 Results saved in: results/")
    print("✨ Experiment ready for autonomous agent evaluation!")


if __name__ == "__main__":
    main() 