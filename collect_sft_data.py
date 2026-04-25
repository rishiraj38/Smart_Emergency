import json
import os
from client import EmergencyDispatchClient

def heuristic_policy(obs):
    """
    A smart rule-based agent to generate "Gold Standard" SFT data.
    """
    # 1. Triage current call
    call_id = obs.get("current_call_id")
    transcript = obs.get("transcript", "").lower()
    
    if call_id:
        severity = "NORMAL"
        if any(w in transcript for w in ["fire", "explosion", "critical", "trapped"]):
            severity = "CRITICAL"
        elif any(w in transcript for w in ["accident", "bleeding", "injured", "hurt"]):
            severity = "SEMI_CRITICAL"
            
        return {"command": "CLASSIFY", "event_id": call_id, "severity": severity}

    # 2. Dispatch to high-priority events
    events = obs.get("active_events", [])
    resources = obs.get("resources", [])
    
    # Sort events by severity
    priority_map = {"CRITICAL": 0, "SEMI_CRITICAL": 1, "NORMAL": 2}
    unserved_events = [e for e in events if e["status"] == "unserved"]
    unserved_events.sort(key=lambda x: priority_map.get(x["severity"], 3))

    available_vehicles = [v for v in resources if v["status"] == "available"]

    for event in unserved_events:
        # Find matching vehicle
        for vehicle in available_vehicles:
            if vehicle["type"] == event["resource_needed"]:
                return {
                    "command": "DISPATCH", 
                    "vehicle_id": vehicle["vehicle_id"], 
                    "event_id": event["event_id"]
                }

    return {"command": "WAIT"}

def collect_data(num_episodes=5, output_file="sft_data.jsonl"):
    print(f"Connecting to environment and collecting {num_episodes} episodes...")
    
    dataset = []
    
    # Use local server (assumed running)
    with EmergencyDispatchClient("http://localhost:8000") as client:
        for ep in range(num_episodes):
            print(f"  Episode {ep+1}/{num_episodes}")
            task_id = (ep % 2) + 1 # Alternate Task 1 and 2
            obs = client.reset(task_id=task_id)
            
            while not obs.get("done", False):
                action = heuristic_policy(obs)
                
                # Save the pair
                dataset.append({
                    "instruction": "You are an emergency dispatch AI. Manage the city resources based on the following state.",
                    "input": json.dumps(obs),
                    "output": json.dumps(action)
                })
                
                # Execute in env
                result = client.step(**action)
                obs = result["observation"]

    with open(output_file, "w") as f:
        for entry in dataset:
            f.write(json.dumps(entry) + "\n")
            
    print(f"Successfully collected {len(dataset)} SFT samples in {output_file}")

if __name__ == "__main__":
    # Ensure server is running before executing this
    try:
        collect_data()
    except Exception as e:
        print(f"Error: {e}")
        print("Tip: Make sure you started the server with 'python -m server.app'")
