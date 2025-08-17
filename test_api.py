#!/usr/bin/env python3
"""
Test script for the LLM Data Factory API

This script tests the FastAPI backend endpoints to ensure they're working correctly.
Run this after starting the API server to verify everything is set up properly.
"""

import requests
import json
import time
from typing import Dict, Any

BASE_URL = "http://localhost:8000"

def test_health_check() -> bool:
    """Test the health check endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data['status']}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_prediction() -> bool:
    """Test the prediction endpoint"""
    test_ticket = "My application keeps crashing when I try to upload large files. This is urgent!"
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json={"ticket_text": test_ticket},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Prediction successful:")
            print(f"   Category: {data['predicted_category']}")
            print(f"   Confidence: {data['confidence']:.2%}")
            print(f"   Processing time: {data['processing_time']:.3f}s")
            return True
        else:
            print(f"❌ Prediction failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Prediction failed: {e}")
        return False

def test_model_info() -> bool:
    """Test the model info endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/model-info", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Model info retrieved:")
            print(f"   Model: {data['model_name']}")
            print(f"   Size: {data['model_size']}")
            print(f"   Fine-tuned: {data['fine_tuned']}")
            print(f"   Categories: {', '.join(data['categories'])}")
            return True
        else:
            print(f"❌ Model info failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Model info failed: {e}")
        return False

def test_batch_prediction() -> bool:
    """Test the batch prediction endpoint"""
    test_tickets = [
        "How do I reset my password?",
        "The system is down and customers can't access our service!",
        "I'd like to request a new feature for bulk exports"
    ]
    
    try:
        response = requests.post(
            f"{BASE_URL}/batch-predict",
            json={"tickets": test_tickets},
            timeout=60
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Batch prediction successful:")
            for i, result in enumerate(data):
                print(f"   Ticket {i+1}: {result['predicted_category']} ({result['confidence']:.2%})")
            return True
        else:
            print(f"❌ Batch prediction failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Batch prediction failed: {e}")
        return False

def main():
    """Run all API tests"""
    print("🧪 Testing LLM Data Factory API\n")
    
    tests = [
        ("Health Check", test_health_check),
        ("Model Info", test_model_info),
        ("Single Prediction", test_prediction),
        ("Batch Prediction", test_batch_prediction),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"🔍 Running {test_name}...")
        success = test_func()
        results.append((test_name, success))
        print()
    
    # Summary
    print("📊 Test Results:")
    print("-" * 40)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:20} {status}")
        if success:
            passed += 1
    
    print("-" * 40)
    print(f"Total: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All tests passed! Your API is working correctly.")
        return 0
    else:
        print(f"\n⚠️  {len(results) - passed} test(s) failed. Check the API server and model setup.")
        return 1

if __name__ == "__main__":
    exit(main())
