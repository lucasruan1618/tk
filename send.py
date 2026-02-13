import os
import time
import json
import random
import requests

# ================= CONFIG =================
API_URL = "http://0.0.0.0:10006/generate"
IMAGE_DIR = "heavy_voxel"
OUTPUT_DIR = "outputs_heavy_2"
LOG_FILE = "timing_2.json"

ALLOWED_EXT = {".png", ".jpg", ".jpeg", ".webp"}
# =========================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

timing_log = {}

for filename in sorted(os.listdir(IMAGE_DIR)):
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXT:
        continue

    image_path = os.path.join(IMAGE_DIR, filename)
    output_name = os.path.splitext(filename)[0] + ".glb"
    output_path = os.path.join(OUTPUT_DIR, output_name)

    seed = random.randint(0, 2**31 - 1)

    print(f"Processing {filename} (seed={seed}) ...")

    start_time = time.time()

    with open(image_path, "rb") as f:
        response = requests.post(
            API_URL,
            files={"prompt_image_file": f},
            data={"seed": seed}   # 👈 FIX 422
        )

    elapsed_time = time.time() - start_time

    if response.status_code == 200:
        with open(output_path, "wb") as out:
            out.write(response.content)

        timing_log[filename] = {
            "status": "success",
            "seed": seed,
            "time_seconds": round(elapsed_time, 4),
            "output": output_name
        }
        print(f"  ✔ Done in {elapsed_time:.2f}s")

    else:
        timing_log[filename] = {
            "status": "failed",
            "seed": seed,
            "time_seconds": round(elapsed_time, 4),
            "error_code": response.status_code,
            "error_message": response.text
        }
        print(f"  ✖ Failed ({response.status_code})")

# Save timing log
with open(LOG_FILE, "w", encoding="utf-8") as f:
    json.dump(timing_log, f, indent=2, ensure_ascii=False)

print(f"\nTiming log saved to {LOG_FILE}")

# ================= AVERAGE TIME =================
success_times = [
    v["time_seconds"]
    for v in timing_log.values()
    if v["status"] == "success"
]

total_requests = len(timing_log)
success_count = len(success_times)
failed_count = total_requests - success_count

if success_count > 0:
    avg_time = sum(success_times) / success_count
    print("\n========== SUMMARY ==========")
    print(f"Total requests : {total_requests}")
    print(f"Success        : {success_count}")
    print(f"Failed         : {failed_count}")
    print(f"Average time   : {avg_time:.4f} seconds")
else:
    print("\nNo successful requests to calculate average time.")