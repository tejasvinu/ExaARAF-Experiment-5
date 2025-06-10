#!/usr/bin/env python3
"""
Example Optimization Agent for HPC Monte Carlo Experiment
=========================================================

This script demonstrates how an autonomous agent could optimize the
Monte Carlo π estimation experiment by systematically exploring
different configurations and learning from results.
"""

import json
import subprocess
import itertools
import numpy as np
from typing import Dict, List, Any, Tuple
from pathlib import Path
import time


class HyperparameterOptimizer:
    """Automated hyperparameter optimization for the Monte Carlo experiment."""
    
    def __init__(self, base_config: Dict[str, Any] = None):
        self.base_config = base_config or {}
        self.optimization_history = []
        self.best_score = float('inf')
        self.best_config = None
        
        # Define optimization search space
        self.search_space = {
            'num_workers': [1, 2, 4, 8, 16],
            'parallel_strategy': ['thread', 'multiprocessing', 'vectorized'],
            'batch_size': [1000, 5000, 10000, 50000, 100000],
            'total_samples': [100000, 500000, 1000000, 5000000],
            'precision': ['float32', 'float64'],
            'algorithm': ['uniform'],  # Start with uniform, agent can expand
        }
    
    def grid_search_optimization(self, max_evaluations: int = 50) -> Dict[str, Any]:
        """Perform grid search optimization with early stopping."""
        print("Starting grid search optimization...")
        
        # Generate parameter combinations
        param_names = list(self.search_space.keys())
        param_values = list(self.search_space.values())
        
        # Use itertools to create all combinations, but limit evaluations
        combinations = list(itertools.product(*param_values))
        np.random.shuffle(combinations)  # Randomize order
        
        evaluated = 0
        for combination in combinations[:max_evaluations]:
            if evaluated >= max_evaluations:
                break
                
            config = dict(zip(param_names, combination))
            score, result = self.evaluate_configuration(config)
            
            if score < self.best_score:
                self.best_score = score
                self.best_config = config.copy()
                print(f"New best configuration found! Score: {score:.4f}")
                print(f"Config: {config}")
            
            evaluated += 1
            print(f"Evaluated {evaluated}/{max_evaluations} configurations")
        
        return {
            'best_config': self.best_config,
            'best_score': self.best_score,
            'total_evaluations': evaluated,
            'optimization_history': self.optimization_history
        }
    
    def bayesian_optimization(self, n_iterations: int = 20) -> Dict[str, Any]:
        """Simplified Bayesian optimization approach."""
        print("Starting Bayesian-style optimization...")
        
        # Start with random sampling
        for i in range(min(5, n_iterations)):
            config = self.sample_random_config()
            score, result = self.evaluate_configuration(config)
            
            if score < self.best_score:
                self.best_score = score
                self.best_config = config.copy()
        
        # Then focus on promising regions
        for i in range(5, n_iterations):
            if self.best_config:
                config = self.sample_near_best(self.best_config)
            else:
                config = self.sample_random_config()
                
            score, result = self.evaluate_configuration(config)
            
            if score < self.best_score:
                self.best_score = score
                self.best_config = config.copy()
                print(f"Iteration {i+1}: New best score {score:.4f}")
        
        return {
            'best_config': self.best_config,
            'best_score': self.best_score,
            'total_evaluations': n_iterations,
            'optimization_history': self.optimization_history
        }
    
    def sample_random_config(self) -> Dict[str, Any]:
        """Sample a random configuration from the search space."""
        config = {}
        for param, values in self.search_space.items():
            config[param] = np.random.choice(values)
        return config
    
    def sample_near_best(self, best_config: Dict[str, Any]) -> Dict[str, Any]:
        """Sample a configuration near the current best."""
        config = best_config.copy()
        
        # Randomly modify 1-2 parameters
        params_to_modify = np.random.choice(
            list(self.search_space.keys()), 
            size=np.random.randint(1, 3),
            replace=False
        )
        
        for param in params_to_modify:
            values = self.search_space[param]
            current_idx = values.index(config[param]) if config[param] in values else 0
            
            # Sample from nearby values
            nearby_indices = [
                max(0, current_idx - 1),
                current_idx,
                min(len(values) - 1, current_idx + 1)
            ]
            nearby_idx = np.random.choice(nearby_indices)
            config[param] = values[nearby_idx]
        
        return config
    
    def evaluate_configuration(self, config: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Evaluate a single configuration by running the experiment."""
        print(f"Evaluating config: {config}")
        
        # Build command line arguments
        cmd = ['python', 'main.py', '--single-run']
        
        for key, value in config.items():
            cmd.extend([f'--{key.replace("_", "-")}', str(value)])
        
        try:
            # Run the experiment
            start_time = time.time()
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode != 0:
                print(f"Experiment failed: {result.stderr}")
                return float('inf'), {}
            
            # Parse results from output or results file
            results_file = Path('results/results.json')
            if results_file.exists():
                with open(results_file, 'r') as f:
                    results_data = json.load(f)
                    if results_data:
                        latest_result = results_data[-1]
                        
                        # Calculate optimization score
                        if latest_result['converged']:
                            score = latest_result['total_execution_time']
                        else:
                            # Penalize non-converged results
                            score = latest_result['total_execution_time'] * (1 + latest_result['relative_error'] * 100)
                        
                        # Store result
                        optimization_entry = {
                            'config': config,
                            'score': score,
                            'result': latest_result,
                            'timestamp': time.time()
                        }
                        self.optimization_history.append(optimization_entry)
                        
                        return score, latest_result
            
            return float('inf'), {}
            
        except subprocess.TimeoutExpired:
            print("Experiment timed out")
            return float('inf'), {}
        except Exception as e:
            print(f"Error running experiment: {e}")
            return float('inf'), {}
    
    def save_optimization_results(self, results: Dict[str, Any], filename: str = "optimization_results.json"):
        """Save optimization results to file."""
        Path('results').mkdir(exist_ok=True)
        with open(f'results/{filename}', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Optimization results saved to results/{filename}")


def main():
    """Main optimization routine."""
    print("HPC Monte Carlo Experiment Optimizer")
    print("=====================================")
    
    optimizer = HyperparameterOptimizer()
    
    # Choose optimization strategy
    optimization_method = input("Choose optimization method (grid/bayesian/both): ").strip().lower()
    
    if optimization_method in ['grid', 'both']:
        print("\n--- Grid Search Optimization ---")
        grid_results = optimizer.grid_search_optimization(max_evaluations=20)
        optimizer.save_optimization_results(grid_results, "grid_search_results.json")
        
        print(f"\nGrid Search Results:")
        print(f"Best Score: {grid_results['best_score']:.4f}")
        print(f"Best Config: {grid_results['best_config']}")
    
    if optimization_method in ['bayesian', 'both']:
        print("\n--- Bayesian Optimization ---")
        bayesian_results = optimizer.bayesian_optimization(n_iterations=15)
        optimizer.save_optimization_results(bayesian_results, "bayesian_optimization_results.json")
        
        print(f"\nBayesian Optimization Results:")
        print(f"Best Score: {bayesian_results['best_score']:.4f}")
        print(f"Best Config: {bayesian_results['best_config']}")
    
    # Test the best configuration
    if optimizer.best_config:
        print(f"\n--- Testing Best Configuration ---")
        print(f"Best config: {optimizer.best_config}")
        
        final_score, final_result = optimizer.evaluate_configuration(optimizer.best_config)
        print(f"Final validation score: {final_score:.4f}")
        
        if final_result:
            print(f"PI estimate: {final_result['pi_estimate']:.8f}")
            print(f"Error: {final_result['absolute_error']:.8f}")
            print(f"Execution time: {final_result['total_execution_time']:.4f}s")


if __name__ == "__main__":
    main() 