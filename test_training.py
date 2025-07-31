#!/usr/bin/env python3
"""
Test script for COMET Quality Estimation training.

This script verifies that the training script can be imported and basic
functionality works without requiring GPU resources or long training times.
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        # Test basic imports
        import torch
        import pandas as pd
        import numpy as np
        from scipy.stats import pearsonr, spearmanr
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        print("✓ Basic ML libraries imported successfully")
        
        # Test COMET import
        from comet.models import download_model, load_from_checkpoint
        print("✓ COMET libraries imported successfully")
        
        # Test training script imports
        sys.path.append('.')
        from train_comet_qe import QualityEstimationDataset, COMETQualityEstimator, collate_fn, compute_correlations
        print("✓ Training script modules imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def test_dataset_loading():
    """Test dataset loading functionality."""
    print("\nTesting dataset loading...")
    
    try:
        from train_comet_qe import QualityEstimationDataset
        
        # Test with sample data
        if os.path.exists("sample_train_data.tsv"):
            dataset = QualityEstimationDataset("sample_train_data.tsv")
            print(f"✓ Dataset loaded with {len(dataset)} samples")
            
            # Test __getitem__
            sample = dataset[0]
            required_keys = ['source', 'mt', 'reference', 'score']
            if all(key in sample for key in required_keys):
                print("✓ Dataset samples have correct format")
            else:
                print("✗ Dataset samples missing required keys")
                return False
        else:
            print("✗ Sample data file not found")
            return False
            
        return True
        
    except Exception as e:
        print(f"✗ Dataset loading error: {e}")
        return False


def test_correlation_computation():
    """Test correlation computation functionality."""
    print("\nTesting correlation computation...")
    
    try:
        from train_comet_qe import compute_correlations
        import numpy as np
        
        # Create test data
        predictions = np.array([0.8, 0.9, 0.7, 0.85, 0.95])
        targets = np.array([0.82, 0.88, 0.72, 0.83, 0.92])
        
        correlations = compute_correlations(predictions, targets)
        
        required_metrics = ['pearson', 'spearman', 'mse', 'mae']
        if all(metric in correlations for metric in required_metrics):
            print("✓ Correlation computation works correctly")
            print(f"  Pearson: {correlations['pearson']:.3f}")
            print(f"  Spearman: {correlations['spearman']:.3f}")
            print(f"  MSE: {correlations['mse']:.6f}")
            print(f"  MAE: {correlations['mae']:.6f}")
        else:
            print("✗ Missing required correlation metrics")
            return False
            
        return True
        
    except Exception as e:
        print(f"✗ Correlation computation error: {e}")
        return False


def test_data_collation():
    """Test data collation functionality."""
    print("\nTesting data collation...")
    
    try:
        from train_comet_qe import collate_fn
        
        # Create sample batch
        batch = [
            {'source': 'Hello world', 'mt': 'Hola mundo', 'reference': 'Hola mundo', 'score': 1.0},
            {'source': 'Good morning', 'mt': 'Buenos días', 'reference': 'Buenos días', 'score': 0.9}
        ]
        
        collated = collate_fn(batch)
        
        expected_keys = ['sources', 'mts', 'references', 'scores']
        if all(key in collated for key in expected_keys):
            print("✓ Data collation works correctly")
            print(f"  Batch size: {len(collated['sources'])}")
        else:
            print("✗ Collated data missing required keys")
            return False
            
        return True
        
    except Exception as e:
        print(f"✗ Data collation error: {e}")
        return False


def test_argument_parsing():
    """Test command line argument parsing."""
    print("\nTesting argument parsing...")
    
    try:
        import subprocess
        import sys
        
        # Test help command
        result = subprocess.run([
            sys.executable, "train_comet_qe.py", "--help"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0 and "COMET Quality Estimation Model" in result.stdout:
            print("✓ Argument parsing works correctly")
            return True
        else:
            print("✗ Argument parsing failed")
            print(f"Return code: {result.returncode}")
            print(f"Stderr: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"✗ Argument parsing test error: {e}")
        return False


def test_model_saving_structure():
    """Test model saving directory structure."""
    print("\nTesting model saving structure...")
    
    try:
        from train_comet_qe import save_model
        import tempfile
        import json
        
        # Create a mock model object
        class MockModel:
            def __init__(self):
                self.model_name = "test-model"
                self.device = "cpu"
        
        mock_model = MockModel()
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = os.path.join(temp_dir, "model_artifacts")
            
            # Test saving
            save_model(mock_model, model_dir, epoch=1, loss=0.1, metrics={'pearson': 0.9})
            
            # Check if files were created
            expected_files = ['model_epoch_1.pt', 'training_config.json', 'model.pt']
            created_files = os.listdir(model_dir)
            
            if all(f in created_files for f in expected_files):
                print("✓ Model saving creates correct file structure")
                
                # Test config file content
                with open(os.path.join(model_dir, 'training_config.json'), 'r') as f:
                    config = json.load(f)
                    if 'model_name' in config and 'epoch' in config:
                        print("✓ Training config saved correctly")
                    else:
                        print("✗ Training config missing required fields")
                        return False
            else:
                print(f"✗ Missing expected files. Created: {created_files}")
                return False
                
        return True
        
    except Exception as e:
        print(f"✗ Model saving test error: {e}")
        return False


def run_all_tests():
    """Run all tests and report results."""
    print("=" * 60)
    print("COMET Quality Estimation Training Script Tests")
    print("=" * 60)
    
    tests = [
        ("Import Tests", test_imports),
        ("Dataset Loading", test_dataset_loading),
        ("Correlation Computation", test_correlation_computation),
        ("Data Collation", test_data_collation),
        ("Argument Parsing", test_argument_parsing),
        ("Model Saving Structure", test_model_saving_structure),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 40)
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status:<8} {test_name}")
    
    print("-" * 60)
    print(f"TOTAL: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The training script is ready to use.")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the issues above.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)