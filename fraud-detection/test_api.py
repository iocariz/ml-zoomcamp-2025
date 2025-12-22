import requests
import numpy as np
import json
import time

def test_api():
    # API endpoint
    url = "http://localhost:8000/predict"
    
    # Generate random features (88 features as per config.json)
    # Using random normal distribution to simulate scaled features
    features = np.random.randn(42).tolist()
    
    # Prepare payload
    payload = {
        "features": features,
        "strategy": "weighted"
    }
    
    print(f"🚀 Sending request to {url}...")
    print(f"📦 Payload size: {len(features)} features")
    
    try:
        start_time = time.time()
        response = requests.post(url, json=payload)
        latency = (time.time() - start_time) * 1000
        
        if response.status_code == 200:
            result = response.json()
            print("\n✅ Success!")
            print(f"⏱️ Latency: {latency:.2f}ms")
            print("\nResponse:")
            print(json.dumps(result, indent=2))
        else:
            print(f"\n❌ Error: {response.status_code}")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print("\n❌ Could not connect to API. Is it running?")
        print("Run: uv run app.py")

if __name__ == "__main__":
    test_api()
