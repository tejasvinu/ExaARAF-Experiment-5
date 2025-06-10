# HPC Monte Carlo π Estimation Optimization Experiment

This experiment is designed to evaluate an autonomous agent's ability to optimize High Performance Computing (HPC) tasks through configuration parameter tuning. The task focuses on Monte Carlo π estimation with multiple optimization dimensions.

## Overview

The experiment provides a Monte Carlo π estimation task with the following optimization opportunities:

- **Algorithm Selection**: Different sampling strategies (uniform, Sobol, Halton, importance sampling)
- **Parallelization**: Various parallel execution strategies (threading, multiprocessing, vectorization)
- **Performance Tuning**: Memory management, numerical precision, batch sizes
- **Convergence Control**: Accuracy targets, iteration limits, convergence windows

## Quick Start

### Installation

1. Install Python 3.8 or higher
2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Basic Usage

Run a single experiment:
```bash
python main.py --single-run
```

Run optimization loop:
```bash
python main.py
```

### Example Configurations

**High Performance Configuration:**
```bash
python main.py --single-run --total-samples 5000000 --num-workers 8 --parallel-strategy multiprocessing --precision float64
```

**Memory Efficient Configuration:**
```bash
python main.py --single-run --batch-size 1000 --memory-efficient --precision float32
```

**High Accuracy Configuration:**
```bash
python main.py --single-run --target-accuracy 1e-6 --max-iterations 200
```

## Configuration Parameters

### Core Algorithm Parameters
- `--total-samples`: Number of Monte Carlo samples (default: 1,000,000)
- `--algorithm`: Sampling algorithm (uniform, sobol, halton, importance)

### Parallelization Parameters
- `--num-workers`: Number of parallel workers (default: CPU count)
- `--parallel-strategy`: Parallelization strategy (thread, multiprocessing, vectorized)
- `--batch-size`: Batch size for processing (default: 10,000)

### Performance Parameters
- `--no-vectorization`: Disable numpy vectorization
- `--precision`: Numerical precision (float32, float64)
- `--memory-efficient`: Enable memory-efficient processing

### Convergence Parameters
- `--target-accuracy`: Target relative accuracy (default: 1e-4)
- `--max-iterations`: Maximum optimization iterations (default: 100)
- `--convergence-window`: Window size for convergence checking (default: 10)

### Output Parameters
- `--output-dir`: Output directory for results (default: "results")
- `--log-level`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `--no-save-intermediate`: Disable saving intermediate results

## Optimization Agent Example

The repository includes an example optimization agent (`optimizer.py`) that demonstrates automated hyperparameter optimization:

```bash
python optimizer.py
```

This agent implements:
- **Grid Search**: Systematic exploration of parameter combinations
- **Bayesian Optimization**: Intelligent search focusing on promising regions
- **Performance Tracking**: Automatic evaluation and comparison of configurations

## Evaluation Metrics

The experiment tracks multiple performance metrics:

- **Accuracy**: Absolute and relative error in π estimation
- **Performance**: Execution time and samples per second
- **Convergence**: Whether target accuracy was achieved
- **Efficiency**: Parallel processing efficiency
- **Optimization Score**: Combined metric balancing accuracy and performance

## Output Files

Results are saved in the `results/` directory:

- `results.json`: Individual experiment results
- `summary.json`: Optimization experiment summary
- `experiment.log`: Detailed execution logs
- `optimization_results.json`: Hyperparameter optimization results

## Agent Optimization Opportunities

An autonomous agent can optimize this task by:

1. **Parameter Tuning**: Systematically exploring the configuration space
2. **Algorithm Enhancement**: Implementing better sampling algorithms (Sobol, Halton)
3. **Performance Optimization**: Optimizing memory usage and parallel efficiency
4. **Adaptive Strategies**: Adjusting parameters based on system characteristics
5. **Code Optimization**: Improving the implementation for better performance

## Example Agent Workflow

```python
# 1. Initial baseline evaluation
baseline_config = {...}
baseline_score = evaluate_configuration(baseline_config)

# 2. Systematic parameter exploration
for config in generate_configurations():
    score = evaluate_configuration(config)
    if score < best_score:
        best_score = score
        best_config = config

# 3. Algorithm improvements
implement_sobol_sampling()
implement_importance_sampling()

# 4. Performance optimization
optimize_memory_usage()
tune_parallel_efficiency()

# 5. Final validation
final_score = evaluate_configuration(best_config)
```

## Advanced Usage

### Custom Algorithm Implementation

Agents can implement new sampling algorithms by extending the `MonteCarloEstimator` class:

```python
def _custom_sampling(self, n_samples: int) -> np.ndarray:
    # Implement custom sampling strategy
    pass
```

### Performance Profiling

Enable detailed performance profiling:
```bash
python main.py --single-run --profile-performance
```

### Large Scale Experiments

For large-scale optimization experiments:
```bash
python main.py --total-samples 10000000 --max-iterations 1000 --target-accuracy 1e-6
```

## Performance Expectations

Typical performance on modern hardware:
- **1M samples**: 0.1-1.0 seconds
- **10M samples**: 1-10 seconds
- **100M samples**: 10-100 seconds

Optimization targets:
- **Accuracy**: < 1e-4 relative error
- **Performance**: > 1M samples/second
- **Efficiency**: > 80% parallel efficiency

## Troubleshooting

**Memory Issues**:
- Reduce `--total-samples` or `--batch-size`
- Use `--precision float32`
- Enable `--memory-efficient`

**Performance Issues**:
- Adjust `--num-workers` based on CPU cores
- Try different `--parallel-strategy` options
- Optimize `--batch-size` for your system

**Convergence Issues**:
- Increase `--total-samples`
- Adjust `--target-accuracy`
- Check `--max-iterations` limit

## Contributing

To extend the experiment:

1. Add new sampling algorithms in `MonteCarloEstimator`
2. Implement new parallel strategies in `ParallelEstimator`
3. Extend optimization search space in `optimizer.py`
4. Add new performance metrics in `ExperimentRunner`

## License

This experiment is designed for research and educational purposes. 