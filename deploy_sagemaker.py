#!/usr/bin/env python3
"""
SageMaker deployment script for the COMET model.
This script handles uploading the model to S3 and deploying it as a SageMaker endpoint.
"""

import sagemaker
from sagemaker import get_execution_role
from sagemaker.pytorch import PyTorchModel
import boto3
import os
import sys

def setup_sagemaker():
    """Set up SageMaker session, role, and S3 configuration"""
    
    print("Setting up SageMaker environment...")
    
    try:
        # Set up session, role and S3
        sagemaker_session = sagemaker.Session()
        role = get_execution_role()
        
        bucket = sagemaker_session.default_bucket()
        prefix = "sagemaker/comet/inference"
        
        print(f"✓ Role: {role}")
        print(f"✓ S3 Bucket: s3://{bucket}/{prefix}")
        
        # Get region
        session = boto3.session.Session()
        print(f"✓ Configured Region: {session.region_name}")
        
        return sagemaker_session, role, bucket, prefix
        
    except Exception as e:
        print(f"✗ Error setting up SageMaker: {e}")
        print("Make sure you are running this in a SageMaker environment or have proper AWS credentials configured.")
        sys.exit(1)

def upload_model_to_s3(sagemaker_session, bucket, prefix):
    """Upload the model.tar.gz to S3"""
    
    print("\nUploading model artifacts to S3...")
    
    model_file = 'model.tar.gz'
    
    if not os.path.exists(model_file):
        print(f"✗ {model_file} not found!")
        print("Please run setup_model.py first to create the model artifacts.")
        sys.exit(1)
    
    try:
        model_artifact = sagemaker_session.upload_data(
            path=model_file, 
            bucket=bucket, 
            key_prefix=prefix
        )
        print(f"✓ Model artifacts uploaded to: {model_artifact}")
        return model_artifact
        
    except Exception as e:
        print(f"✗ Error uploading model to S3: {e}")
        sys.exit(1)

def deploy_model(model_artifact, role, instance_type='ml.p3.2xlarge', wait=True):
    """Deploy the model to a SageMaker endpoint"""
    
    print(f"\nDeploying model to SageMaker endpoint...")
    print(f"Instance type: {instance_type}")
    
    try:
        # Create a PyTorch model
        pytorch_model = PyTorchModel(
            model_data=model_artifact,
            role=role,
            framework_version="2.0.0",
            py_version='py310',
            entry_point='inference.py',
            source_dir='inference/code'
        )
        
        # Deploy the model
        print("Starting deployment (this may take several minutes)...")
        predictor = pytorch_model.deploy(
            initial_instance_count=1, 
            instance_type=instance_type,
            wait=wait
        )
        
        if wait:
            print(f"✓ Model deployed successfully!")
            print(f"✓ Endpoint name: {predictor.endpoint_name}")
        else:
            print(f"✓ Deployment started successfully!")
            print(f"✓ Endpoint name: {predictor.endpoint_name}")
            print("Note: Deployment is running in the background. Check AWS console for status.")
        
        return predictor
        
    except Exception as e:
        print(f"✗ Error deploying model: {e}")
        sys.exit(1)

def main():
    """Main deployment function"""
    
    print("🚀 Starting SageMaker deployment process...")
    
    # Check if model.tar.gz exists
    if not os.path.exists('model.tar.gz'):
        print("✗ model.tar.gz not found!")
        print("Please run setup_model.py first to create the model artifacts.")
        sys.exit(1)
    
    # Set up SageMaker
    sagemaker_session, role, bucket, prefix = setup_sagemaker()
    
    # Upload model to S3
    model_artifact = upload_model_to_s3(sagemaker_session, bucket, prefix)
    
    # Deploy the model
    predictor = deploy_model(model_artifact, role, wait=False)
    
    print("\n🎉 Deployment process completed!")
    print("\nNext steps:")
    print("1. Wait for the endpoint to be 'InService' (check AWS Console or use check_endpoint_status.py)")
    print("2. Use test_endpoint.py to test the deployed endpoint")
    print(f"3. Endpoint name: {predictor.endpoint_name}")

if __name__ == "__main__":
    main()