#!/usr/bin/env python3
"""
Local testing script for the COMET model inference functions.
This script tests the inference.py file locally before deploying to SageMaker.
"""

import json
import sys
import os

# Add the inference code directory to Python path
sys.path.append('./inference/code')

try:
    from inference import input_fn, output_fn, model_fn, predict_fn
except ImportError as e:
    print(f"Error importing inference functions: {e}")
    print("Make sure you have run setup_model.py first and installed the required packages.")
    sys.exit(1)

def test_model_locally():
    """Test the model inference functions locally"""
    
    print("Starting local model testing...")
    
    # Sample data for testing
    sample_data = {
        "src": "This is a sample text to be translated.",
        "mt": "Dies ist ein zu übersetzender Beispieltext."
    }
    
    # Convert the dictionary to a JSON string
    sample_json_string = json.dumps(sample_data)
    
    model_dir = './inference/'
    
    try:
        # Test input_fn
        print("\n1. Testing input_fn...")
        deserialized_data = input_fn(sample_json_string, 'application/json')
        print(f"✓ Deserialized Data: {deserialized_data}")
        
        # Test model_fn
        print("\n2. Testing model_fn...")
        model = model_fn(model_dir)
        print("✓ Model loaded successfully")
        
        # Test predict_fn
        print("\n3. Testing predict_fn...")
        prediction = predict_fn(deserialized_data, model)
        print(f"✓ Prediction: {prediction}")
        
        # Test output_fn
        print("\n4. Testing output_fn...")
        serialized_data = output_fn(prediction, 'application/json')
        print(f"✓ Serialized Data: {serialized_data}")
        print(f"✓ Type: {type(serialized_data)}")
        
        print(f"\n✓ All tests passed successfully!")
        print(f"✓ COMET score: {prediction['predictions']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function to run the tests"""
    
    # Check if model directory exists
    if not os.path.exists('./inference/wmt20-comet-qe-da'):
        print("✗ Model directory not found!")
        print("Please run setup_model.py first to download and set up the model.")
        sys.exit(1)
    
    # Run the tests
    success = test_model_locally()
    
    if success:
        print("\n🎉 Local testing completed successfully!")
        print("The model is ready for SageMaker deployment.")
    else:
        print("\n❌ Local testing failed!")
        print("Please fix the issues before deploying to SageMaker.")
        sys.exit(1)

if __name__ == "__main__":
    main()