#!/usr/bin/env python3
"""
Script to test the deployed SageMaker endpoint with sample data.
"""

from sagemaker.predictor import Predictor
import json
import time
import boto3
import sys

def get_available_endpoints():
    """Get list of available InService endpoints"""
    
    try:
        sagemaker_client = boto3.client('sagemaker')
        response = sagemaker_client.list_endpoints(MaxResults=20)
        endpoints = response['Endpoints']
        
        # Filter for InService endpoints
        active_endpoints = [ep for ep in endpoints if ep['EndpointStatus'] == 'InService']
        
        return active_endpoints
        
    except Exception as e:
        print(f"Error retrieving endpoints: {e}")
        return []

def test_endpoint(endpoint_name, sample_data=None):
    """Test the endpoint with sample data"""
    
    if sample_data is None:
        # Default sample data (same as in the article)
        sample_data = {
            "src": "This is a sample text to be translated.",
            "mt": "Dies ist ein zu übersetzender Beispieltext."
        }
    
    try:
        # Start tracking time
        start_time = time.time()
        
        # Convert the dictionary to a JSON string
        sample_input = json.dumps(sample_data)
        
        print(f"Testing endpoint: {endpoint_name}")
        print(f"Input data:")
        print(f"  Source: {sample_data['src']}")
        print(f"  Translation: {sample_data['mt']}")
        print("\nSending request to endpoint...")
        
        # Initialize the Predictor
        predictor = Predictor(endpoint_name=endpoint_name)
        
        # Invoke the endpoint
        response = predictor.predict(
            sample_input, 
            initial_args={"ContentType": "application/json"}
        )
        
        # Deserialize the response (assuming it's JSON)
        if isinstance(response, bytes):
            result = json.loads(response.decode('utf-8'))
        else:
            result = response
        
        # Stop tracking time
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Display the results
        print(f"\n✓ Response received successfully!")
        print(f"Result: {result}")
        print(f"COMET Score: {result.get('predictions', 'N/A')}")
        print(f"Response time: {elapsed_time:.2f} seconds")
        
        return True, result
        
    except Exception as e:
        print(f"✗ Error testing endpoint: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def run_multiple_tests(endpoint_name):
    """Run multiple tests with different sample data"""
    
    test_cases = [
        {
            "name": "Original Example",
            "data": {
                "src": "This is a sample text to be translated.",
                "mt": "Dies ist ein zu übersetzender Beispieltext."
            }
        },
        {
            "name": "Simple Greeting",
            "data": {
                "src": "Hello, how are you?",
                "mt": "Hola, ¿cómo estás?"
            }
        },
        {
            "name": "Technical Text",
            "data": {
                "src": "The machine learning model processes the input data efficiently.",
                "mt": "El modelo de aprendizaje automático procesa los datos de entrada de manera eficiente."
            }
        }
    ]
    
    print(f"\nRunning {len(test_cases)} test cases...")
    print("=" * 80)
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {test_case['name']}")
        print("-" * 40)
        
        success, result = test_endpoint(endpoint_name, test_case['data'])
        results.append({
            'name': test_case['name'],
            'success': success,
            'result': result
        })
        
        if i < len(test_cases):
            print("\nWaiting 2 seconds before next test...")
            time.sleep(2)
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    successful_tests = sum(1 for r in results if r['success'])
    
    for i, result in enumerate(results, 1):
        status = "✓ PASSED" if result['success'] else "✗ FAILED"
        score = result['result'].get('predictions', 'N/A') if result['result'] else 'N/A'
        print(f"Test {i}: {result['name']} - {status} (Score: {score})")
    
    print(f"\nOverall: {successful_tests}/{len(test_cases)} tests passed")
    
    return results

def main():
    """Main function"""
    
    print("SageMaker Endpoint Tester")
    print("=" * 50)
    
    # Get available endpoints
    endpoints = get_available_endpoints()
    
    if not endpoints:
        print("No InService endpoints found!")
        print("Please deploy an endpoint first using deploy_sagemaker.py")
        sys.exit(1)
    
    print(f"Found {len(endpoints)} InService endpoint(s):")
    for i, endpoint in enumerate(endpoints, 1):
        print(f"{i}. {endpoint['EndpointName']}")
    
    # Select endpoint
    if len(endpoints) == 1:
        endpoint_name = endpoints[0]['EndpointName']
        print(f"\nUsing the only available endpoint: {endpoint_name}")
    else:
        print("\nEnter the endpoint name you want to test:")
        endpoint_name = input().strip()
        
        # Validate endpoint name
        valid_names = [ep['EndpointName'] for ep in endpoints]
        if endpoint_name not in valid_names:
            print(f"✗ Invalid endpoint name. Available endpoints: {', '.join(valid_names)}")
            sys.exit(1)
    
    # Ask for test type
    print(f"\nSelect test type:")
    print("1. Single test (original example)")
    print("2. Multiple test cases")
    print("3. Custom input")
    
    choice = input("Enter your choice (1-3): ").strip()
    
    if choice == "1":
        print(f"\nTesting endpoint with original example...")
        success, result = test_endpoint(endpoint_name)
        if success:
            print(f"\n🎉 Test completed successfully!")
        else:
            print(f"\n❌ Test failed!")
            
    elif choice == "2":
        results = run_multiple_tests(endpoint_name)
        successful_tests = sum(1 for r in results if r['success'])
        if successful_tests == len(results):
            print(f"\n🎉 All tests passed!")
        else:
            print(f"\n⚠️  {len(results) - successful_tests} test(s) failed!")
            
    elif choice == "3":
        print("\nEnter source text:")
        src_text = input().strip()
        print("Enter translated text:")
        mt_text = input().strip()
        
        custom_data = {"src": src_text, "mt": mt_text}
        success, result = test_endpoint(endpoint_name, custom_data)
        if success:
            print(f"\n🎉 Custom test completed successfully!")
        else:
            print(f"\n❌ Custom test failed!")
    
    else:
        print("Invalid choice. Exiting.")
        sys.exit(1)

if __name__ == "__main__":
    main()