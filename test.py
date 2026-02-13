import requests
import time
import sys

BASE_URL = "http://0.0.0.0:10006"

# 1. Health check
print("=== Health Check ===")
try:
    resp = requests.get(f"{BASE_URL}/health", timeout=5)
    print(f"Status: {resp.status_code}, Body: {resp.json()}")
except requests.exceptions.ConnectionError:
    print("ERROR: Cannot connect to server. Is it running on port 10006?")
    sys.exit(1)

# 2. Generate endpoint
print("\n=== Generate ===")
image_path = "7f691c0b893a85f92a4ff3d2fa6d05db141af1b1938ba570fdb474a688ead51e.png"
warmup_image_path = "warmup_image.png"
with open(warmup_image_path, "rb") as img:
    files = {"prompt_image_file": img}
    data = {"seed": 42}

    start = time.time()
    response = requests.post(f"{BASE_URL}/generate", files=files, data=data)
    elapsed = time.time() - start

print(f"Status: {response.status_code}")
print(f"Time: {elapsed:.2f}s")
print(f"Content-Type: {response.headers.get('content-type')}")
print(f"Response size: {len(response.content)} bytes ({len(response.content) / 1024 / 1024:.2f} MB)")

if response.status_code == 200:
    output_path = "sample_model.glb"
    with open(output_path, "wb") as f:
        f.write(response.content)
    print(f"Saved to {output_path}")
else:
    print(f"Error: {response.text[:500]}")
