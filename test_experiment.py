#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for HPC Monte Carlo Experiment
==========================================

This script runs basic tests to verify the experiment setup is working correctly.
"""

import subprocess
import sys
import time
import json
from pathlib import Path


def test_basic_functionality():
    """Test basic experiment functionality."""
    print("Testing basic experiment functionality...")
    
    # Test single run with minimal parameters
    cmd = [
        sys.executable, "main.py", 
        "--single-run", 
        "--total-samples", "10000",
        "--log-level", "WARNING"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✓ Basic functionality test passed")
            return True
        else:
            print(f"✗ Basic functionality test failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("✗ Basic functionality test timed out")
        return False
    except Exception as e:
        print(f"✗ Basic functionality test error: {e}")
        return False


def test_parallel_strategies():
    """Test different parallelization strategies."""
    print("Testing parallel strategies...")
    
    strategies = ["thread", "multiprocessing", "vectorized"]
    
    for strategy in strategies:
        print(f"  Testing {strategy} strategy...")
        
        cmd = [
            sys.executable, "main.py",
            "--single-run",
            "--total-samples", "10000",
            "--parallel-strategy", strategy,
            "--num-workers", "2",
            "--log-level", "ERROR"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                print(f"  ✓ {strategy} strategy works")
            else:
                print(f"  ✗ {strategy} strategy failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print(f"  ✗ {strategy} strategy timed out")
            return False
        except Exception as e:
            print(f"  ✗ {strategy} strategy error: {e}")
            return False
    
    print("✓ All parallel strategies test passed")
    return True


def test_precision_modes():
    """Test different precision modes."""
    print("Testing precision modes...")
    
    precisions = ["float32", "float64"]
    
    for precision in precisions:
        print(f"  Testing {precision} precision...")
        
        cmd = [
            sys.executable, "main.py",
            "--single-run",
            "--total-samples", "10000",
            "--precision", precision,
            "--log-level", "ERROR"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                print(f"  ✓ {precision} precision works")
            else:
                print(f"  ✗ {precision} precision failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print(f"  ✗ {precision} precision timed out")
            return False
        except Exception as e:
            print(f"  ✗ {precision} precision error: {e}")
            return False
    
    print("✓ All precision modes test passed")
    return True


def test_output_generation():
    """Test that outputs are generated correctly."""
    print("Testing output generation...")
    
    # Clean up any existing results
    results_dir = Path("results")
    if results_dir.exists():
        import shutil
        shutil.rmtree(results_dir)
    
    cmd = [
        sys.executable, "main.py",
        "--single-run",
        "--total-samples", "10000",
        "--output-dir", "results",
        "--log-level", "INFO"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            print(f"✗ Output generation test failed: {result.stderr}")
            return False
        
        # Check if results files are created
        results_file = results_dir / "results.json"
        log_file = results_dir / "experiment.log"
        
        if not results_file.exists():
            print("✗ Results file not created")
            return False
        
        if not log_file.exists():
            print("✗ Log file not created")
            return False
        
        # Check if results file contains valid JSON
        try:
            with open(results_file, 'r') as f:
                data = json.load(f)
                if not data or len(data) == 0:
                    print("✗ Results file is empty")
                    return False
        except json.JSONDecodeError:
            print("✗ Results file contains invalid JSON")
            return False
        
        print("✓ Output generation test passed")
        return True
        
    except subprocess.TimeoutExpired:
        print("✗ Output generation test timed out")
        return False
    except Exception as e:
        print(f"✗ Output generation test error: {e}")
        return False


def test_accuracy_range():
    """Test that π estimates are in reasonable range."""
    print("Testing accuracy range...")
    
    cmd = [
        sys.executable, "main.py",
        "--single-run",
        "--total-samples", "100000",
        "--log-level", "ERROR"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode != 0:
            print(f"✗ Accuracy test failed: {result.stderr}")
            return False
        
        # Check results
        results_file = Path("results/results.json")
        if results_file.exists():
            with open(results_file, 'r') as f:
                data = json.load(f)
                if data:
                    pi_estimate = data[-1]['pi_estimate']
                    error = data[-1]['absolute_error']
                    
                    # PI should be between 3.0 and 3.2 with reasonable sample size
                    if 3.0 <= pi_estimate <= 3.2:
                        print(f"✓ Accuracy test passed (PI ≈ {pi_estimate:.4f}, error: {error:.6f})")
                        return True
                    else:
                        print(f"✗ PI estimate out of range: {pi_estimate}")
                        return False
        
        print("✗ Could not verify accuracy")
        return False
        
    except subprocess.TimeoutExpired:
        print("✗ Accuracy test timed out")
        return False
    except Exception as e:
        print(f"✗ Accuracy test error: {e}")
        return False


def run_all_tests():
    """Run all tests."""
    print("HPC Monte Carlo Experiment Test Suite")
    print("=" * 40)
    
    tests = [
        test_basic_functionality,
        test_parallel_strategies,
        test_precision_modes,
        test_output_generation,
        test_accuracy_range,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()  # Add spacing between tests
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            print()
    
    print("=" * 40)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! The experiment is ready to use.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 