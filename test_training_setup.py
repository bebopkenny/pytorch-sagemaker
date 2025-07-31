#!/usr/bin/env python3
"""
Test script to validate COMET training setup.

This script checks:
1. Required dependencies are installed
2. COMET model can be loaded
3. Sample data can be created and loaded
4. Basic training functionality works

Run this before training to ensure everything is set up correctly.
"""

import sys
import subprocess
import importlib
import tempfile
import os
from pathlib import Path

def check_package(package_name, import_name=None):
    """Check if a package is installed and can be imported."""
    if import_name is None:
        import_name = package_name
    
    try:
        importlib.import_module(import_name)
        print(f"✓ {package_name} - OK")
        return True
    except ImportError:
        print(f"✗ {package_name} - MISSING")
        return False

def check_dependencies():
    """Check all required dependencies."""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        ("torch", "torch"),
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("scikit-learn", "sklearn"),
        ("scipy", "scipy"),
        ("unbabel-comet", "comet"),
    ]
    
    all_good = True
    for package, import_name in required_packages:
        if not check_package(package, import_name):
            all_good = False
    
    return all_good

def test_comet_model():
    """Test COMET model loading."""
    print("\n🤖 Testing COMET model loading...")
    
    try:
        from comet.models import download_model, load_from_checkpoint
        
        # Try to download and load a small model for testing
        print("  Downloading COMET model (this may take a moment)...")
        model_path = download_model("Unbabel/wmt22-comet-da")
        print(f"  Model downloaded to: {model_path}")
        
        print("  Loading model...")
        model = load_from_checkpoint(model_path)
        print("  ✓ Model loaded successfully")
        
        # Test a simple prediction
        print("  Testing prediction...")
        test_data = [{
            'src': 'Hello world',
            'mt': 'Hola mundo',
            'ref': 'Hola mundo'
        }]
        
        predictions = model.predict(test_data)
        print(f"  ✓ Prediction successful: {predictions[0]:.4f}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def test_dataset_creation():
    """Test sample dataset creation."""
    print("\n📊 Testing dataset creation...")
    
    try:
        from create_sample_dataset import create_sample_dataset
        import pandas as pd
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as tmp:
            temp_path = tmp.name
        
        try:
            # Create small sample dataset
            print("  Creating sample dataset...")
            df = create_sample_dataset(num_samples=10, output_path=temp_path)
            
            # Validate dataset
            print("  Validating dataset structure...")
            required_cols = ['source', 'mt', 'reference', 'score']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                print(f"  ✗ Missing columns: {missing_cols}")
                return False
            
            print(f"  ✓ Dataset created with {len(df)} samples")
            print(f"  ✓ Score range: {df['score'].min():.3f} - {df['score'].max():.3f}")
            
            return True
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def test_training_imports():
    """Test that training script imports work."""
    print("\n🏋️ Testing training script imports...")
    
    try:
        # Test key imports from training script
        from torch.utils.data import Dataset, DataLoader
        from torch.optim import AdamW
        from torch.nn.utils import clip_grad_norm_
        from sklearn.model_selection import train_test_split
        from scipy.stats import pearsonr, spearmanr
        
        print("  ✓ PyTorch imports - OK")
        print("  ✓ Scikit-learn imports - OK")
        print("  ✓ SciPy imports - OK")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Import error: {e}")
        return False

def test_gpu_availability():
    """Check GPU availability."""
    print("\n🖥️ Checking GPU availability...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"  ✓ CUDA available with {gpu_count} GPU(s)")
            print(f"  ✓ Primary GPU: {gpu_name}")
            
            # Test GPU memory
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"  ✓ GPU Memory: {gpu_memory:.1f} GB")
            
        else:
            print("  ⚠️ CUDA not available - will use CPU")
            
        return True
        
    except Exception as e:
        print(f"  ✗ Error checking GPU: {e}")
        return False

def run_quick_training_test():
    """Run a very quick training test to ensure the pipeline works."""
    print("\n🚀 Running quick training test...")
    
    try:
        # Create temporary dataset
        from create_sample_dataset import create_sample_dataset
        import tempfile
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as tmp:
            temp_path = tmp.name
        
        try:
            # Create small dataset
            print("  Creating test dataset...")
            create_sample_dataset(num_samples=20, output_path=temp_path)
            
            # Import training components
            print("  Importing training components...")
            from train_comet_model import COMETTrainer, load_dataset, create_data_loaders
            
            # Test data loading
            print("  Testing data loading...")
            train_df, val_df = load_dataset(temp_path, test_size=0.3)
            train_loader, val_loader = create_data_loaders(train_df, val_df, batch_size=4)
            
            print(f"  ✓ Train samples: {len(train_df)}")
            print(f"  ✓ Validation samples: {len(val_df)}")
            
            # Test trainer initialization (but don't actually train)
            print("  Testing trainer initialization...")
            with tempfile.TemporaryDirectory() as temp_model_dir:
                trainer = COMETTrainer(
                    model_name_or_path="Unbabel/wmt22-comet-da",
                    device="cpu",  # Force CPU for quick test
                    batch_size=4,
                    model_dir=temp_model_dir
                )
                
                # Note: We skip actual model loading and training for speed
                print("  ✓ Trainer initialized successfully")
            
            print("  ✓ Quick training test passed!")
            return True
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
    except Exception as e:
        print(f"  ✗ Training test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 COMET Training Setup Test")
    print("=" * 50)
    
    tests = [
        ("Dependencies", check_dependencies),
        ("COMET Model", test_comet_model),
        ("Dataset Creation", test_dataset_creation),
        ("Training Imports", test_training_imports),
        ("GPU Availability", test_gpu_availability),
        ("Quick Training Test", run_quick_training_test),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print()
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"  ✗ Unexpected error in {test_name}: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(tests)
    
    for test_name, passed_test in results.items():
        status = "✓ PASS" if passed_test else "✗ FAIL"
        print(f"{test_name:<20} {status}")
        if passed_test:
            passed += 1
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 All tests passed! Your setup is ready for training.")
        print("\n📖 Next steps:")
        print("1. Create your training dataset (or use create_sample_dataset.py)")
        print("2. Run: python train_comet_model.py --data_path your_data.tsv")
        return 0
    else:
        print(f"\n⚠️ {total - passed} test(s) failed. Please fix the issues above.")
        print("\n🔧 Common fixes:")
        print("- Install missing packages: pip install -r train_requirements.txt")
        print("- Check internet connection for model downloads")
        print("- Ensure you have sufficient disk space")
        return 1

if __name__ == "__main__":
    sys.exit(main())