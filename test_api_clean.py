"""
API Testing Script for LLM Data Factory
Tests all endpoints of the FastAPI server
"""

import requests
import json
import time

# Configuration
BASE_URL = "http://localhost:8000"

def wait_for_server(max_attempts=30, delay=2):
    """Wait for the server to be ready"""
    for attempt in range(max_attempts):
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=5)
            if response.status_code == 200:
                print(f"Server is ready after {attempt + 1} attempts")
                return True
        except requests.exceptions.RequestException:
            if attempt < max_attempts - 1:
                print(f"Attempt {attempt + 1}: Server not ready, waiting {delay}s...")
                time.sleep(delay)
    
    print(f"Server failed to start after {max_attempts} attempts")
    return False

def test_health():
    """Test the health endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"Health check passed: {data['status']}")
            return True
        else:
            print(f"Health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"Health check failed: {e}")
        return False

def test_prediction():
    """Test the prediction endpoint"""
    url = f"{BASE_URL}/predict"
    
    # Test data for prediction
    test_ticket = {
        "text": "My order hasn't arrived yet and it's been 2 weeks. When will it be delivered?"
    }
    
    try:
        response = requests.post(url, json=test_ticket, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            print(f"Prediction successful:")
            print(f"  Text: {test_ticket['text']}")
            print(f"  Category: {data['category']}")
            print(f"  Confidence: {data['confidence']:.3f}")
            return True
        else:
            print(f"Prediction failed: {response.status_code}")
            print(response.text)
            return False
            
    except Exception as e:
        print(f"Prediction failed: {e}")
        return False

def test_model_info():
    """Test the model info endpoint"""
    url = f"{BASE_URL}/model-info"
    
    try:
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print(f"Model info retrieved:")
            print(f"  Model: {data['model_name']}")
            print(f"  Type: {data['model_type']}")
            print(f"  Status: {data['status']}")
            print(f"  Categories: {data['categories']}")
            return True
        else:
            print(f"Model info failed: {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"Model info failed: {e}")
        return False

def test_batch_prediction():
    """Test batch prediction endpoint"""
    url = f"{BASE_URL}/predict-batch"
    
    # Test data for batch prediction
    test_tickets = {
        "tickets": [
            {"text": "I want to return my purchase"},
            {"text": "How do I reset my password?"},
            {"text": "My payment was declined"}
        ]
    }
    
    try:
        response = requests.post(url, json=test_tickets, timeout=60)
        
        if response.status_code == 200:
            data = response.json()
            print(f"Batch prediction successful:")
            print(f"  Processed {len(data['predictions'])} tickets")
            for i, pred in enumerate(data['predictions']):
                print(f"  {i+1}. {pred['category']} (confidence: {pred['confidence']:.3f})")
            return True
        else:
            print(f"Batch prediction failed: {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"Batch prediction failed: {e}")
        return False

def run_all_tests():
    """Run all API tests"""
    tests = [
        ("Health Check", test_health),
        ("Single Prediction", test_prediction),
        ("Model Info", test_model_info),
        ("Batch Prediction", test_batch_prediction)
    ]
    
    results = []
    
    print("Test Results:")
    print("=" * 50)
    
    for test_name, test_func in tests:
        print(f"\nRunning {test_name}...")
        try:
            success = test_func()
            results.append((test_name, success))
            status = "PASS" if success else "FAIL"
        except Exception as e:
            print(f"Test failed: {e}")
            results.append((test_name, False))
            status = "FAIL"
        
        print(f"{test_name}: {status}")
    
    # Summary
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    print("\n" + "=" * 50)
    if passed == total:
        print("All tests passed! Your API is working correctly.")
    else:
        print(f"Warning: {len(results) - passed} test(s) failed. Check the API server and model setup.")
    
    return results

if __name__ == "__main__":
    print("LLM Data Factory API Test Suite")
    print("Waiting for server to start...")
    
    if wait_for_server():
        run_all_tests()
    else:
        print("Could not connect to API server. Make sure it's running on http://localhost:8000")
