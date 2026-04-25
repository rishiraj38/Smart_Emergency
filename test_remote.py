import requests
import json

# ── CONFIG ──
SPACE_URL = "https://rishi38-smartcity-emergency-dispatch.hf.space"

def test_remote_rewards():
    print(f"📡 Connecting to remote environment: {SPACE_URL}")
    
    # 1. Reset
    print("\n--- RESETTING ---")
    resp = requests.post(f"{SPACE_URL}/reset", json={"task_id": 1})
    if resp.status_code != 200:
        print(f"❌ Reset failed: {resp.text}")
        return
    
    obs = resp.json()
    call_id = obs.get("current_call_id")
    print(f"✅ Reset success. Initial call: {call_id}")
    print(f"Transcript: {obs.get('transcript')}")

    # 2. Step: Classify (Correctly)
    print("\n--- STEP: CLASSIFY ---")
    action = {
        "command": "CLASSIFY",
        "event_id": call_id,
        "severity": "CRITICAL"
    }
    resp = requests.post(f"{SPACE_URL}/step", json=action)
    if resp.status_code != 200:
        print(f"❌ Step failed: {resp.text}")
        return
    
    result = resp.json()
    print(f"✅ Step success.")
    print(f"Reward: {result['reward']}")
    print(f"Breakdown: {json.dumps(result['info']['reward_breakdown'], indent=2)}")

    # 3. Step: Dispatch (Correctly)
    print("\n--- STEP: DISPATCH ---")
    # Find the newly created event and a fire truck
    event_id = result["observation"]["active_events"][0]["event_id"]
    vehicle_id = "FIRE-01"
    
    action = {
        "command": "DISPATCH",
        "vehicle_id": vehicle_id,
        "event_id": event_id
    }
    resp = requests.post(f"{SPACE_URL}/step", json=action)
    result = resp.json()
    
    print(f"✅ Step success.")
    print(f"Reward: {result['reward']}")
    print(f"Breakdown: {json.dumps(result['info']['reward_breakdown'], indent=2)}")

if __name__ == "__main__":
    test_remote_rewards()
