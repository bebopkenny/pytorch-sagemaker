#!/usr/bin/env python3
"""
Script to download and set up the COMET model artifacts for SageMaker deployment.
"""

import os
import subprocess
import sys
import wget
import tarfile
import shutil

def run_command(command, description):
    """Run a shell command and handle errors"""
    print(f"\n{description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error during {description}:")
        print(e.stderr)
        return False

def main():
    print("Setting up COMET model for SageMaker deployment...")
    
    # Step 1: Install required packages
    print("\n1. Installing required packages...")
    packages = [
        "pip install --upgrade pip",
        "pip install unbabel-comet==2.2.2 -q",
        "pip install protobuf==3.20.* -q",
        "pip install wget"
    ]
    
    for package in packages:
        if not run_command(package, f"Installing {package.split()[-2]}"):
            sys.exit(1)
    
    # Step 2: Download model artifacts
    print("\n2. Downloading COMET model artifacts...")
    model_url = "https://unbabel-experimental-models.s3.amazonaws.com/comet/wmt20/wmt20-comet-qe-da.tar.gz"
    model_file = "wmt20-comet-qe-da.tar.gz"
    
    if not os.path.exists(model_file):
        try:
            wget.download(model_url, model_file)
            print(f"\n✓ Downloaded {model_file}")
        except Exception as e:
            print(f"\n✗ Error downloading model: {e}")
            sys.exit(1)
    else:
        print(f"✓ {model_file} already exists")
    
    # Step 3: Extract model artifacts
    print("\n3. Extracting model artifacts...")
    try:
        with tarfile.open(model_file, 'r:gz') as tar:
            tar.extractall('.')
        print("✓ Model artifacts extracted successfully")
    except Exception as e:
        print(f"✗ Error extracting model: {e}")
        sys.exit(1)
    
    # Step 4: Move extracted files to inference directory
    print("\n4. Setting up directory structure...")
    source_dir = "wmt20-comet-qe-da"
    target_dir = "inference/wmt20-comet-qe-da"
    
    if os.path.exists(source_dir):
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
        shutil.move(source_dir, target_dir)
        print(f"✓ Moved {source_dir} to {target_dir}")
    else:
        print(f"✗ Source directory {source_dir} not found")
        sys.exit(1)
    
    # Step 5: Create model.tar.gz
    print("\n5. Creating model.tar.gz for SageMaker deployment...")
    current_dir = os.getcwd()
    try:
        os.chdir('inference')
        if not run_command("tar -czvf ../model.tar.gz .", "Creating model.tar.gz"):
            sys.exit(1)
        os.chdir(current_dir)
    except Exception as e:
        print(f"✗ Error creating model.tar.gz: {e}")
        os.chdir(current_dir)
        sys.exit(1)
    
    print("\n✓ Setup completed successfully!")
    print("\nDirectory structure:")
    print("├── inference/")
    print("│   ├── code/")
    print("│   │   ├── inference.py")
    print("│   │   └── requirements.txt")
    print("│   └── wmt20-comet-qe-da/")
    print("│       ├── checkpoints/")
    print("│       │   └── model.ckpt")
    print("│       └── hparams.yaml")
    print("├── model.tar.gz")
    print("└── wmt20-comet-qe-da.tar.gz")

if __name__ == "__main__":
    main()