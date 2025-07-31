#!/usr/bin/env python3
"""
Script to check the status of SageMaker endpoints and wait for them to be ready.
"""

import boto3
import time
import sys
from datetime import datetime

def list_endpoints():
    """List all available SageMaker endpoints"""
    
    print("Retrieving SageMaker endpoints...")
    
    try:
        sagemaker_client = boto3.client('sagemaker')
        
        response = sagemaker_client.list_endpoints(MaxResults=20)
        endpoints = response['Endpoints']
        
        if not endpoints:
            print("No endpoints found in your account.")
            return []
        
        print(f"\nFound {len(endpoints)} endpoint(s):")
        print("-" * 80)
        
        for i, endpoint in enumerate(endpoints, 1):
            print(f"{i}. Endpoint name: {endpoint['EndpointName']}")
            print(f"   Status: {endpoint['EndpointStatus']}")
            print(f"   Created: {endpoint['CreationTime']}")
            print(f"   Modified: {endpoint['LastModifiedTime']}")
            print("-" * 80)
        
        return endpoints
        
    except Exception as e:
        print(f"Error retrieving endpoints: {e}")
        return []

def check_endpoint_status(endpoint_name, wait_for_service=False):
    """Check the status of a specific endpoint"""
    
    sagemaker_client = boto3.client('sagemaker')
    
    try:
        while True:
            response = sagemaker_client.list_endpoints(MaxResults=20)
            endpoints = response['Endpoints']
            
            endpoint_found = False
            for endpoint in endpoints:
                if endpoint['EndpointName'] == endpoint_name:
                    endpoint_found = True
                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    print(f"Time: {current_time}")
                    print(f"Endpoint name: {endpoint['EndpointName']}")
                    print(f"Endpoint status: {endpoint['EndpointStatus']}")
                    print(f"Creation time: {endpoint['CreationTime']}")
                    print(f"Last modified time: {endpoint['LastModifiedTime']}")
                    print("-" * 60)
                    
                    if endpoint['EndpointStatus'] == 'InService':
                        print(f"✓ Endpoint {endpoint_name} is ready for inference!")
                        return True
                    elif endpoint['EndpointStatus'] == 'Failed':
                        print(f"✗ Endpoint {endpoint_name} deployment failed!")
                        return False
                    elif not wait_for_service:
                        return endpoint['EndpointStatus'] == 'InService'
            
            if not endpoint_found:
                print(f"✗ Endpoint {endpoint_name} not found!")
                return False
            
            if not wait_for_service:
                break
                
            print("Waiting 30 seconds before next check...")
            time.sleep(30)
            
    except Exception as e:
        print(f"Error checking endpoint status: {e}")
        return False

def main():
    """Main function"""
    
    print("SageMaker Endpoint Status Checker")
    print("=" * 50)
    
    # List all endpoints
    endpoints = list_endpoints()
    
    if not endpoints:
        sys.exit(0)
    
    # If only one endpoint, use it automatically
    if len(endpoints) == 1:
        endpoint_name = endpoints[0]['EndpointName']
        print(f"\nUsing the only available endpoint: {endpoint_name}")
    else:
        # Ask user to select an endpoint
        print("\nEnter the endpoint name you want to check:")
        endpoint_name = input().strip()
        
        # Validate endpoint name
        valid_names = [ep['EndpointName'] for ep in endpoints]
        if endpoint_name not in valid_names:
            print(f"✗ Invalid endpoint name. Available endpoints: {', '.join(valid_names)}")
            sys.exit(1)
    
    # Ask if user wants to wait
    print(f"\nDo you want to wait for endpoint '{endpoint_name}' to be InService? (y/n): ")
    wait_choice = input().strip().lower()
    wait_for_service = wait_choice in ['y', 'yes']
    
    if wait_for_service:
        print(f"\nMonitoring endpoint '{endpoint_name}' (checking every 30 seconds)...")
        print("Press Ctrl+C to stop monitoring")
        print("-" * 60)
        
        try:
            success = check_endpoint_status(endpoint_name, wait_for_service=True)
            if success:
                print(f"\n🎉 Endpoint '{endpoint_name}' is ready!")
            else:
                print(f"\n❌ Endpoint '{endpoint_name}' deployment failed!")
        except KeyboardInterrupt:
            print(f"\n\nMonitoring stopped by user.")
    else:
        print(f"\nChecking current status of endpoint '{endpoint_name}'...")
        check_endpoint_status(endpoint_name, wait_for_service=False)

if __name__ == "__main__":
    main()