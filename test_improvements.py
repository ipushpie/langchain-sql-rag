#!/usr/bin/env python3
"""
Test script to verify the chat service improvements.
This script tests the new customer_id functionality and improved prompts.
"""

import json
import sys
from app.services.chat_services import ask_question

def test_basic_question():
    """Test basic question without customer_id"""
    print("=== Testing Basic Question (No Customer ID) ===")
    
    question = "list all the data breaches"
    navigation_routes = ["/security/breaches", "/compliance/incidents"]
    
    try:
        result = ask_question(question, navigation_routes)
        print(f"Question: {question}")
        print(f"SQL Generated: {result.get('sql_query')}")
        print(f"Results Count: {len(result.get('results', []))}")
        print(f"Answer: {json.dumps(result.get('answer'), indent=2)}")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_customer_filtered_question():
    """Test question with customer_id filtering"""
    print("\n=== Testing Customer-Filtered Question ===")
    
    question = "show me recent requests"
    navigation_routes = ["/requests", "/customer/requests"]
    customer_id = 123
    
    try:
        result = ask_question(question, navigation_routes, customer_id)
        print(f"Question: {question}")
        print(f"Customer ID: {customer_id}")
        print(f"SQL Generated: {result.get('sql_query')}")
        print(f"Results Count: {len(result.get('results', []))}")
        print(f"Answer: {json.dumps(result.get('answer'), indent=2)}")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_greeting():
    """Test greeting functionality"""
    print("\n=== Testing Greeting ===")
    
    question = "hello"
    navigation_routes = []
    customer_id = 456
    
    try:
        result = ask_question(question, navigation_routes, customer_id)
        print(f"Question: {question}")
        print(f"Customer ID: {customer_id}")
        print(f"Answer: {json.dumps(result.get('answer'), indent=2)}")
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Chat Service Improvements")
    print("=" * 50)
    
    tests = [
        test_basic_question,
        test_customer_filtered_question,
        test_greeting
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
                print("✅ Test passed")
            else:
                print("❌ Test failed")
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
        print("-" * 30)
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        sys.exit(0)
    else:
        print("⚠️ Some tests failed")
        sys.exit(1)