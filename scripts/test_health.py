#!/usr/bin/env python3
"""
Health check script for the Private Trustworthy RAG Assistant API.
Verifies that the backend is running and the core endpoints are responsive.
"""

import sys
import json
import argparse
from typing import Dict, Any

import requests


def check_health(base_url: str, timeout: int = 10) -> Dict[str, Any]:
    """Check the /api/v1/health endpoint."""
    url = f"{base_url}/api/v1/health"
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Health check failed: {str(e)}"}


def check_query(base_url: str, query: str, timeout: int = 180) -> Dict[str, Any]:
    """Test the /api/v1/query endpoint with a simple question."""
    url = f"{base_url}/api/v1/query"
    payload = {
        "query": query,
        "top_k": 2
    }
    try:
        response = requests.post(url, json=payload, timeout=timeout)
        # If we get a 422, we want to see the validation details
        if response.status_code == 422:
            try:
                error_detail = response.json()
                return {"error": f"Validation error: {json.dumps(error_detail, indent=2)}"}
            except json.JSONDecodeError:
                return {"error": f"Validation error (status 422): {response.text}"}
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Query test failed: {str(e)}"}


def main():
    parser = argparse.ArgumentParser(description="Test the RAG API health and functionality")
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Base URL of the API (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=10,
        help="Timeout for HTTP requests in seconds (default: 10)"
    )
    parser.add_argument(
        "--query-timeout",
        type=int,
        default=180,
        help="Timeout for query request in seconds (default: 30)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only output errors and final status"
    )
    args = parser.parse_args()

    if not args.quiet:
        print(f"Checking API at {args.url}")
        print("-" * 50)

    # Health check
    health_result = check_health(args.url, args.timeout)
    if "error" in health_result:
        print(f"❌ Health check failed: {health_result['error']}")
        sys.exit(1)
    else:
        if not args.quiet:
            print(f"✅ Health check passed: {json.dumps(health_result, indent=2)}")

    # Query test
    test_question = "What is the purpose of this system?"
    query_result = check_query(args.url, test_question, args.query_timeout)
    if "error" in query_result:
        print(f"❌ Query test failed: {query_result['error']}")
        sys.exit(1)
    else:
        if not args.quiet:
            print(f"✅ Query test succeeded")
            print(f"   Question: {test_question}")
            answer = query_result.get("answer", "")
            print(f"   Answer: {answer[:100]}{'...' if len(answer) > 100 else ''}")
            sources = query_result.get("sources", [])
            print(f"   Sources found: {len(sources)}")
            if sources:
                # Show first source briefly
                first = sources[0]
                print(f"   First source: {first.get('content', '')[:50]}... (score: {first.get('score', 0):.2f})")

    print("\n🎉 All checks passed!")
    sys.exit(0)


if __name__ == "__main__":
    main()