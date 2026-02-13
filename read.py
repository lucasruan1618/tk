import json
import sys

# ===== CONFIG =====
LOG_FILE = "timing_2.json"   # đổi nếu cần
# ==================

def main():
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            timing_log = json.load(f)
    except FileNotFoundError:
        print(f"File not found: {LOG_FILE}")
        return
    except json.JSONDecodeError:
        print("Invalid JSON format.")
        return

    total_requests = len(timing_log)

    success_times = [
        v["time_seconds"]
        for v in timing_log.values()
        if v.get("status") == "success"
    ]

    success_count = len(success_times)
    failed_count = total_requests - success_count

    if success_count == 0:
        print("No successful requests found.")
        return

    avg_time = sum(success_times) / success_count
    min_time = min(success_times)
    max_time = max(success_times)

    print("========== SUMMARY ==========")
    print(f"Total requests : {total_requests}")
    print(f"Success        : {success_count}")
    print(f"Failed         : {failed_count}")
    print(f"Average time   : {avg_time:.4f} seconds")
    print(f"Min time       : {min_time:.4f} seconds")
    print(f"Max time       : {max_time:.4f} seconds")


if __name__ == "__main__":
    main()
