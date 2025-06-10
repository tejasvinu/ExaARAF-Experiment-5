#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HPC Optimization Experiment: Monte Carlo PI Estimation
======================================================

This experiment evaluates an agent's ability to autonomously optimize
a Monte Carlo PI estimation task across multiple dimensions:
- Algorithm variants (uniform sampling, importance sampling, quasi-random)
- Parallelization strategies (threading, multiprocessing, vectorization)
- Memory management (batch sizes, chunk processing)
- Numerical precision and convergence criteria

The task provides clear performance metrics and optimization opportunities.
"""

import argparse
import json
import time
import numpy as np
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Dict, Any, List, Tuple
import logging
import sys
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class ExperimentConfig:
    """Configuration for the Monte Carlo PI estimation experiment."""
    
    # Core algorithm parameters
    total_samples = 1_000_000
    algorithm = "uniform"  # uniform, sobol, halton, importance
    
    # Parallelization parameters
    num_workers = mp.cpu_count()
    parallel_strategy = "multiprocessing"  # thread, multiprocessing, vectorized
    batch_size = 10_000
    
    # Performance optimization parameters
    use_numpy_vectorization = True
    precision = "float64"  # float32, float64
    memory_efficient = False
    
    # Convergence and accuracy parameters
    target_accuracy = 1e-4
    max_iterations = 100
    convergence_window = 10
    
    # Output and logging
    output_dir = "results"
    log_level = "INFO"
    save_intermediate = True
    
    # Experimental parameters for optimization
    random_seed = 42
    profile_performance = True


class MonteCarloEstimator:
    """Monte Carlo PI estimator with multiple algorithm variants."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.rng = np.random.default_rng(config.random_seed)
        
        # Set numpy precision
        self.dtype = getattr(np, config.precision)
        
        # Algorithm mapping
        self.algorithms = {
            "uniform": self._uniform_sampling,
            "sobol": self._sobol_sampling,
            "halton": self._halton_sampling,
            "importance": self._importance_sampling
        }
    
    def _uniform_sampling(self, n_samples: int) -> np.ndarray:
        """Standard uniform random sampling."""
        if self.config.use_numpy_vectorization:
            x = self.rng.uniform(-1, 1, n_samples).astype(self.dtype)
            y = self.rng.uniform(-1, 1, n_samples).astype(self.dtype)
            return (x * x + y * y <= 1).astype(self.dtype)
        else:
            # Slower non-vectorized version for comparison
            hits = 0
            for _ in range(n_samples):
                x, y = self.rng.uniform(-1, 1, 2)
                if x * x + y * y <= 1:
                    hits += 1
            return np.array([hits / n_samples * 4], dtype=self.dtype)
    
    def _sobol_sampling(self, n_samples: int) -> np.ndarray:
        """Quasi-random Sobol sequence sampling."""
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=2, scramble=True, seed=self.config.random_seed)
            samples = sampler.random(n_samples)
            # Scale from [0,1] to [-1,1]
            x = (samples[:, 0] * 2 - 1).astype(self.dtype)
            y = (samples[:, 1] * 2 - 1).astype(self.dtype)
            return (x * x + y * y <= 1).astype(self.dtype)
        except ImportError:
            logging.warning("scipy not available for Sobol sampling, falling back to uniform")
            return self._uniform_sampling(n_samples)
    
    def _halton_sampling(self, n_samples: int) -> np.ndarray:
        """Quasi-random Halton sequence sampling."""
        try:
            from scipy.stats import qmc
            sampler = qmc.Halton(d=2, scramble=True, seed=self.config.random_seed)
            samples = sampler.random(n_samples)
            # Scale from [0,1] to [-1,1]
            x = (samples[:, 0] * 2 - 1).astype(self.dtype)
            y = (samples[:, 1] * 2 - 1).astype(self.dtype)
            return (x * x + y * y <= 1).astype(self.dtype)
        except ImportError:
            logging.warning("scipy not available for Halton sampling, falling back to uniform")
            return self._uniform_sampling(n_samples)
    
    def _importance_sampling(self, n_samples: int) -> np.ndarray:
        """Importance sampling with optimized distribution for circle estimation."""
        # Use a distribution that concentrates samples near the unit circle boundary
        # where the indicator function changes most rapidly
        try:
            # Sample radii with bias towards the circle boundary
            r = np.sqrt(self.rng.beta(0.8, 0.8, n_samples)).astype(self.dtype)
            theta = self.rng.uniform(0, 2 * np.pi, n_samples).astype(self.dtype)
            
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            
            # Importance weights (inverse of sampling density)
            # For beta(0.8, 0.8) transformed to sqrt, the density is roughly proportional to r^(-0.4)
            weights = r ** 0.4
            
            # Indicator function
            inside_circle = (x * x + y * y <= 1).astype(self.dtype)
            
            # Apply importance weights
            weighted_estimates = inside_circle * weights
            
            # Normalize by average weight to get unbiased estimate
            avg_weight = np.mean(weights)
            return weighted_estimates / avg_weight
            
        except Exception as e:
            logging.warning(f"Importance sampling failed ({e}), falling back to uniform")
            return self._uniform_sampling(n_samples)
    
    def estimate_batch(self, batch_size: int) -> float:
        """Estimate PI using a single batch."""
        algorithm = self.algorithms[self.config.algorithm]
        hits = algorithm(batch_size)
        
        if self.config.use_numpy_vectorization:
            return np.sum(hits) / len(hits) * 4
        else:
            return hits[0]


class ParallelEstimator:
    """Handles parallel execution strategies."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.estimator = MonteCarloEstimator(config)
    
    def run_parallel(self) -> Tuple[float, Dict[str, float]]:
        """Run the estimation with the configured parallel strategy."""
        start_time = time.perf_counter()
        
        if self.config.parallel_strategy == "multiprocessing":
            pi_estimate = self._multiprocessing_strategy()
        elif self.config.parallel_strategy == "thread":
            pi_estimate = self._threading_strategy()
        elif self.config.parallel_strategy == "vectorized":
            pi_estimate = self._vectorized_strategy()
        else:
            raise ValueError(f"Unknown parallel strategy: {self.config.parallel_strategy}")
        
        end_time = time.perf_counter()
        
        metrics = {
            "execution_time": end_time - start_time,
            "samples_per_second": self.config.total_samples / (end_time - start_time),
            "parallel_efficiency": self._calculate_efficiency(),
        }
        
        return pi_estimate, metrics
    
    def _multiprocessing_strategy(self) -> float:
        """Use multiprocessing for parallel estimation."""
        batch_size = self.config.total_samples // self.config.num_workers
        
        with ProcessPoolExecutor(max_workers=self.config.num_workers) as executor:
            futures = [
                executor.submit(self.estimator.estimate_batch, batch_size)
                for _ in range(self.config.num_workers)
            ]
            results = [future.result() for future in futures]
        
        return np.mean(results)
    
    def _threading_strategy(self) -> float:
        """Use threading for parallel estimation."""
        batch_size = self.config.total_samples // self.config.num_workers
        
        with ThreadPoolExecutor(max_workers=self.config.num_workers) as executor:
            futures = [
                executor.submit(self.estimator.estimate_batch, batch_size)
                for _ in range(self.config.num_workers)
            ]
            results = [future.result() for future in futures]
        
        return np.mean(results)
    
    def _vectorized_strategy(self) -> float:
        """Use pure vectorization without explicit parallelization."""
        return self.estimator.estimate_batch(self.config.total_samples)
    
    def _calculate_efficiency(self) -> float:
        """Calculate parallel efficiency metric."""
        # Simplified efficiency calculation
        return min(1.0, 1.0 / self.config.num_workers)


class ExperimentRunner:
    """Main experiment runner with optimization tracking."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.setup_logging()
        self.setup_output_dir()
        self.results_history: List[Dict[str, Any]] = []
    
    def setup_logging(self):
        """Configure logging for the experiment."""
        # Ensure output directory exists before setting up file logging
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{self.config.output_dir}/experiment.log"),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    def setup_output_dir(self):
        """Create output directory for results."""
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
    
    def run_single_experiment(self) -> Dict[str, Any]:
        """Run a single experiment iteration."""
        logging.info(f"Starting experiment with config: {asdict(self.config)}")
        
        # Initialize estimator
        parallel_estimator = ParallelEstimator(self.config)
        
        # Run estimation
        start_time = time.perf_counter()
        pi_estimate, metrics = parallel_estimator.run_parallel()
        total_time = time.perf_counter() - start_time
        
        # Calculate accuracy metrics
        error = abs(pi_estimate - np.pi)
        relative_error = error / np.pi
        
        result = {
            "timestamp": time.time(),
            "config": asdict(self.config),
            "pi_estimate": float(pi_estimate),
            "actual_pi": float(np.pi),
            "absolute_error": float(error),
            "relative_error": float(relative_error),
            "total_execution_time": total_time,
            "performance_metrics": metrics,
            "converged": relative_error < self.config.target_accuracy,
        }
        
        # Log results
        logging.info(f"PI estimate: {pi_estimate:.8f} (error: {error:.8f})")
        logging.info(f"Execution time: {total_time:.4f}s")
        logging.info(f"Samples/sec: {metrics['samples_per_second']:.2f}")
        
        # Save results
        self.results_history.append(result)
        if self.config.save_intermediate:
            self.save_results()
        
        return result
    
    def run_optimization_experiment(self) -> Dict[str, Any]:
        """Run multiple iterations for optimization evaluation."""
        logging.info("Starting optimization experiment...")
        
        best_result = None
        best_score = float('inf')
        
        for iteration in range(self.config.max_iterations):
            logging.info(f"Optimization iteration {iteration + 1}/{self.config.max_iterations}")
            
            result = self.run_single_experiment()
            
            # Define optimization score (minimize execution time while maintaining accuracy)
            if result['converged']:
                score = result['total_execution_time']
            else:
                score = result['total_execution_time'] * (1 + result['relative_error'] * 100)
            
            if score < best_score:
                best_score = score
                best_result = result
                logging.info(f"New best result found! Score: {score:.4f}")
            
            # Early stopping if target accuracy achieved consistently
            if self._check_convergence():
                logging.info("Convergence criteria met, stopping optimization")
                break
        
        summary = {
            "total_iterations": len(self.results_history),
            "best_result": best_result,
            "best_score": best_score,
            "optimization_progress": self.results_history,
            "final_config": asdict(self.config)
        }
        
        self.save_summary(summary)
        return summary
    
    def _check_convergence(self) -> bool:
        """Check if optimization has converged."""
        if len(self.results_history) < self.config.convergence_window:
            return False
        
        recent_results = self.results_history[-self.config.convergence_window:]
        return all(r['converged'] for r in recent_results)
    
    def save_results(self):
        """Save current results to file."""
        results_file = Path(self.config.output_dir) / "results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results_history, f, indent=2, default=self._json_serializer)
    
    def _json_serializer(self, obj):
        """Custom JSON serializer for numpy types."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return str(obj)
    
    def save_summary(self, summary: Dict[str, Any]):
        """Save experiment summary."""
        summary_file = Path(self.config.output_dir) / "summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=self._json_serializer)


def create_config_from_args() -> ExperimentConfig:
    """Create configuration from command line arguments."""
    parser = argparse.ArgumentParser(
        description="HPC Monte Carlo PI Estimation Optimization Experiment"
    )
    
    # Core algorithm parameters
    parser.add_argument("--total-samples", type=int, default=1_000_000,
                       help="Total number of Monte Carlo samples")
    parser.add_argument("--algorithm", choices=["uniform", "sobol", "halton", "importance"],
                       default="uniform", help="Sampling algorithm")
    
    # Parallelization parameters
    parser.add_argument("--num-workers", type=int, default=mp.cpu_count(),
                       help="Number of parallel workers")
    parser.add_argument("--parallel-strategy", 
                       choices=["thread", "multiprocessing", "vectorized"],
                       default="multiprocessing", help="Parallelization strategy")
    parser.add_argument("--batch-size", type=int, default=10_000,
                       help="Batch size for processing")
    
    # Performance parameters
    parser.add_argument("--no-vectorization", action="store_true",
                       help="Disable numpy vectorization")
    parser.add_argument("--precision", choices=["float32", "float64"],
                       default="float64", help="Numerical precision")
    parser.add_argument("--memory-efficient", action="store_true",
                       help="Enable memory-efficient processing")
    
    # Convergence parameters
    parser.add_argument("--target-accuracy", type=float, default=1e-4,
                       help="Target relative accuracy for π estimation")
    parser.add_argument("--max-iterations", type=int, default=100,
                       help="Maximum optimization iterations")
    parser.add_argument("--convergence-window", type=int, default=10,
                       help="Window size for convergence checking")
    
    # Output parameters
    parser.add_argument("--output-dir", default="results",
                       help="Output directory for results")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       default="INFO", help="Logging level")
    parser.add_argument("--no-save-intermediate", action="store_true",
                       help="Disable saving intermediate results")
    
    # Experimental parameters
    parser.add_argument("--random-seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--no-profiling", action="store_true",
                       help="Disable performance profiling")
    
    # Execution mode
    parser.add_argument("--single-run", action="store_true",
                       help="Run single experiment instead of optimization loop")
    
    args = parser.parse_args()
    
    return ExperimentConfig(
        total_samples=args.total_samples,
        algorithm=args.algorithm,
        num_workers=args.num_workers,
        parallel_strategy=args.parallel_strategy,
        batch_size=args.batch_size,
        use_numpy_vectorization=not args.no_vectorization,
        precision=args.precision,
        memory_efficient=args.memory_efficient,
        target_accuracy=args.target_accuracy,
        max_iterations=args.max_iterations,
        convergence_window=args.convergence_window,
        output_dir=args.output_dir,
        log_level=args.log_level,
        save_intermediate=not args.no_save_intermediate,
        random_seed=args.random_seed,
        profile_performance=not args.no_profiling,
    )


def main():
    """Main experiment entry point."""
    config = create_config_from_args()
    runner = ExperimentRunner(config)
    
    # Parse command line for execution mode
    parser = argparse.ArgumentParser()
    parser.add_argument("--single-run", action="store_true")
    args, _ = parser.parse_known_args()
    
    if args.single_run:
        result = runner.run_single_experiment()
        print(f"\nFinal Result:")
        print(f"PI estimate: {result['pi_estimate']:.8f}")
        print(f"Error: {result['absolute_error']:.8f}")
        print(f"Execution time: {result['total_execution_time']:.4f}s")
        print(f"Samples/sec: {result['performance_metrics']['samples_per_second']:.2f}")
    else:
        summary = runner.run_optimization_experiment()
        print(f"\nOptimization Complete!")
        print(f"Best PI estimate: {summary['best_result']['pi_estimate']:.8f}")
        print(f"Best score: {summary['best_score']:.4f}")
        print(f"Total iterations: {summary['total_iterations']}")


if __name__ == "__main__":
    main()