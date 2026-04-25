"""
collect_sft_data.py — Generate SFT training data from the baseline agent.

Runs the rule-based baseline agent across all tasks and saves
(system, observation, action) tuples as JSONL for TRL fine-tuning.

Usage:
    python collect_sft_data.py                      # default: 5 episodes per task
    python collect_sft_data.py --episodes 20        # more data
    python collect_sft_data.py --output data.jsonl   # custom output
"""

import json
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server.environment import EmergencyDispatchEnvironment
from models import EmergencyAction

SYSTEM_PROMPT = """You are an expert emergency dispatch agent for a smart city.
You receive 911 call transcripts and must triage them, then dispatch the correct emergency vehicles.

Available commands:
- CLASSIFY: Classify an incoming call with a severity (CRITICAL, SEMI_CRITICAL, NORMAL)
- DISPATCH: Send a vehicle to an event
- MERGE: Merge a duplicate call into an existing event
- ESCALATE: Change the severity of an existing event
- DISCARD: Discard a non-emergency call
- HOLD: Put an event on hold
- RECALL: Recall a dispatched vehicle
- REROUTE: Redirect a vehicle to a different event
- WAIT: Do nothing this step

Respond with a JSON action object. Example:
{"command": "CLASSIFY", "severity": "CRITICAL"}
{"command": "DISPATCH", "vehicle_id": "FIRE-01", "event_id": "E001"}
{"command": "WAIT"}"""


def classify_transcript(transcript: str) -> str:
    t = (transcript or "").lower()
    if any(w in t for w in ["fire", "flames", "smoke", "explosion", "gas", "burning"]):
        return "CRITICAL"
    if any(w in t for w in ["dying", "dead", "not breathing", "heart", "blood",
                             "unconscious", "screaming"]):
        return "CRITICAL"
    if any(w in t for w in ["hurt", "injured", "crash", "accident", "pain",
                             "trapped", "serious"]):
        return "SEMI_CRITICAL"
    return "NORMAL"


def baseline_agent(obs: dict) -> dict:
    """Rule-based agent: dispatch first, then classify."""
    transcript = obs.get("transcript") or ""
    current_call_id = obs.get("current_call_id")
    events = obs.get("active_events", [])
    resources = obs.get("resources", [])
    prefix_map = {"ambulance": "AMB", "fire_truck": "FIRE", "police": "POL"}

    # Priority 1: Dispatch to unserved events (critical first)
    priority = {"CRITICAL": 0, "SEMI_CRITICAL": 1, "NORMAL": 2}
    for event in sorted(events, key=lambda e: priority.get(e.get("severity", "NORMAL"), 2)):
        if event.get("status") != "unserved":
            continue
        prefix = prefix_map.get(event.get("resource_needed", "ambulance"), "AMB")
        for v in resources:
            if v["status"] == "available" and v["vehicle_id"].startswith(prefix):
                return {
                    "command": "DISPATCH",
                    "vehicle_id": v["vehicle_id"],
                    "event_id": event["event_id"],
                }

    # Priority 2: Classify incoming call
    if current_call_id and transcript:
        return {
            "command": "CLASSIFY",
            "severity": classify_transcript(transcript),
        }

    return {"command": "WAIT"}


def format_observation(obs: dict) -> str:
    """Format observation as a concise prompt for the LLM."""
    parts = []

    if obs.get("current_call_id"):
        parts.append(f"📞 Incoming Call: {obs['current_call_id']}")
        parts.append(f"   Transcript: \"{obs['transcript']}\"")

    events = obs.get("active_events", [])
    if events:
        parts.append(f"\n🚨 Active Events ({len(events)}):")
        for e in events:
            parts.append(f"   {e['event_id']}: {e['severity']} | {e['resource_needed']} "
                        f"| status={e['status']} | waiting={e['steps_waiting']} steps")

    resources = obs.get("resources", [])
    available = [v for v in resources if v["status"] == "available"]
    dispatched = [v for v in resources if v["status"] != "available"]

    parts.append(f"\n🚑 Available Vehicles ({len(available)}):")
    for v in available:
        parts.append(f"   {v['vehicle_id']} ({v['type']}) at {v['current_location']}")

    if dispatched:
        parts.append(f"\n🚗 Dispatched Vehicles ({len(dispatched)}):")
        for v in dispatched:
            parts.append(f"   {v['vehicle_id']} → {v['destination']} "
                        f"(status={v['status']}, ETA={v['steps_to_arrival']})")

    parts.append(f"\n📊 Step: {obs['step']} | Calls remaining: {obs['calls_remaining']} "
                f"| Resolved: {obs['cases_resolved']} | Reward: {obs['cumulative_reward']}")

    return "\n".join(parts)


def format_action(action_dict: dict) -> str:
    """Format action as clean JSON string."""
    clean = {k: v for k, v in action_dict.items() if v is not None}
    return json.dumps(clean)


def collect_data(episodes_per_task: int = 5, output_path: str = "sft_data.jsonl"):
    """Collect SFT data from baseline agent runs."""
    samples = []
    total_resolved = 0
    total_steps = 0

    for task_id in [1, 2, 3]:
        for ep in range(episodes_per_task):
            seed = task_id * 1000 + ep
            env = EmergencyDispatchEnvironment(seed=seed)
            obs = env.reset(task_id=task_id, seed=seed).model_dump()

            steps = 0
            while not obs.get("done", False) and steps < 500:
                action_dict = baseline_agent(obs)

                # Record the (observation, action) pair
                user_msg = format_observation(obs)
                assistant_msg = format_action(action_dict)

                sample = {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                        {"role": "assistant", "content": assistant_msg},
                    ]
                }
                samples.append(sample)

                # Step
                try:
                    result = env.step(EmergencyAction(**action_dict)).model_dump()
                    obs = result["observation"]
                except Exception as e:
                    print(f"  Error at step {steps}: {e}")
                    break
                steps += 1

            total_resolved += obs.get("cases_resolved", 0)
            total_steps += steps
            print(f"  Task {task_id} ep {ep+1}: {steps} steps, "
                  f"{obs.get('cases_resolved', 0)} resolved, "
                  f"reward={obs.get('cumulative_reward', 0):.1f}")

    # Save
    with open(output_path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    print(f"\n{'='*50}")
    print(f"Collected {len(samples)} training samples")
    print(f"Total episodes: {episodes_per_task * 3}")
    print(f"Total steps: {total_steps}")
    print(f"Total resolved: {total_resolved}")
    print(f"Saved to: {output_path}")
    print(f"{'='*50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect SFT data from baseline agent")
    parser.add_argument("--episodes", type=int, default=5, help="Episodes per task")
    parser.add_argument("--output", type=str, default="sft_data.jsonl", help="Output path")
    args = parser.parse_args()

    collect_data(episodes_per_task=args.episodes, output_path=args.output)
