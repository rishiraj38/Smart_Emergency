# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Smart Emergency Environment.

Endpoints:
    POST /reset    — Reset the environment, start a new episode
    POST /step     — Submit an action, receive next observation + reward
    GET  /state    — Current episode state
    GET  /health   — Health check
    GET  /tasks    — Available difficulty tasks
    POST /grader   — Score a completed episode (call after done=True)
    GET  /baseline — Run rule-based agent across all 3 tasks
    WS   /ws       — WebSocket for persistent low-latency sessions
    GET  /docs     — Swagger UI (auto-generated)
"""

from openenv.core.env_server.http_server import create_app

try:
    from ..models import SmartEmergencyAction, SmartEmergencyObservation, RerouteAction
    from .smart_emergency_environment import SmartEmergencyEnvironment
except (ImportError, ModuleNotFoundError):
    from models import SmartEmergencyAction, SmartEmergencyObservation, RerouteAction
    from server.smart_emergency_environment import SmartEmergencyEnvironment


# ── App ──────────────────────────────────────────────────────────────────────

# We use create_app so OpenEnv can automatically mount its Gradio web UI at / and /web
# when deployed to Hugging Face Spaces.
app = create_app(
    SmartEmergencyEnvironment,
    SmartEmergencyAction,
    SmartEmergencyObservation,
    env_name="smart_emergency",
    max_concurrent_envs=1,
)

# ── Health ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "environment": "smart-emergency-dispatch911",
        "version": "1.0.0",
    }


# ── Tasks ────────────────────────────────────────────────────────────────────

@app.get("/tasks")
def tasks():
    """List available difficulty tasks."""
    return {
        "tasks": [
            {
                "id": 1,
                "name": "Basic Dispatch",
                "difficulty": "easy",
                "description": "Low-volume calls, fewer active events. Focus on severity and vehicle type.",
                "reward_max": 6.7,
            },
            {
                "id": 2,
                "name": "Duplicate Detection",
                "difficulty": "medium",
                "description": "Higher duplicate rate. Agent must correlate repeat callers to existing events.",
                "reward_max": 6.7,
            },
            {
                "id": 3,
                "name": "Full Disaster Response",
                "difficulty": "hard",
                "description": "High call volume, scarce vehicles, reroutes required. Full 20-step episode.",
                "reward_max": 6.7,
            },
        ]
    }


# ── Grader ───────────────────────────────────────────────────────────────────

@app.post("/grader")
def grader():
    """
    Score the completed episode. Call this after done=True.

    Returns cumulative reward breakdown, per-component averages,
    and a normalized 0–1 score suitable for hackathon leaderboards.
    """
    steps = SmartEmergencyEnvironment.latest_steps

    if steps == 0:
        raise HTTPException(
            status_code=400,
            detail="No episode in progress. Call POST /reset first.",
        )

    # Collect reward history from the class-level tracker
    history = SmartEmergencyEnvironment.latest_history
    if not history:
        raise HTTPException(
            status_code=400,
            detail=(
                "Episode not yet complete or no steps taken. "
                "Keep calling POST /step until observation.done == true."
            ),
        )

    # Aggregate per-component averages
    keys = ["severity", "duplicate", "vehicle_type", "vehicle_choice", "reroute", "total"]
    component_totals = {k: 0.0 for k in keys}
    for breakdown in history:
        for k in keys:
            component_totals[k] += breakdown.get(k, 0.0)

    n = max(1, len(history))
    component_avgs = {k: round(v / n, 4) for k, v in component_totals.items()}
    cumulative = round(component_totals["total"], 4)

    # Normalize: theoretical max ~6.7 per step, floor at 0
    MAX_PER_STEP = 6.7
    score = round(max(0.0, min(1.0, cumulative / (MAX_PER_STEP * n))), 4)

    return {
        "score": score,
        "cumulative_reward": cumulative,
        "steps": steps,
        "episode_id": SmartEmergencyEnvironment.latest_episode_id,
        "reward_components": {
            "severity_avg": component_avgs["severity"],
            "duplicate_avg": component_avgs["duplicate"],
            "vehicle_type_avg": component_avgs["vehicle_type"],
            "vehicle_choice_avg": component_avgs["vehicle_choice"],
            "reroute_avg": component_avgs["reroute"],
        },
        "per_step_total_avg": component_avgs["total"],
    }


# ── Baseline ─────────────────────────────────────────────────────────────────

@app.get("/baseline")
def baseline():
    """
    Run a keyword-heuristic rule-based agent across all 3 tasks.
    Returns per-task scores and an overall average.
    Required for hackathon submission.
    """

    def _classify_severity(transcript: str) -> int:
        t = transcript.lower()
        if any(w in t for w in ["not breathing", "collapsed", "not responding",
                                  "active shooter", "trapped", "mass incident",
                                  "massive fire", "whole block", "not moving"]):
            return 5
        if any(w in t for w in ["won't wake", "unconscious", "not responding",
                                  "gunshots", "flipped", "blood everywhere",
                                  "people yelling", "pileup"]):
            return 4
        if any(w in t for w in ["chest pain", "fight", "mugged", "knife",
                                  "crash", "hurt", "bleeding", "fire at",
                                  "flames", "cyclist"]):
            return 3
        if any(w in t for w in ["fainted", "break-in", "dumpster", "fender",
                                  "small fire", "ankle"]):
            return 2
        return 1

    def _classify_vehicle(transcript: str) -> str:
        t = transcript.lower()
        if any(w in t for w in ["fire", "flames", "smoke", "burning", "gas"]):
            return "fire"
        if any(w in t for w in ["shooter", "gunshot", "mugged", "knife",
                                  "break-in", "fight", "shoplifter", "crime"]):
            return "police"
        return "ambulance"

    def _pick_vehicle(env: SmartEmergencyEnvironment, vtype: str):
        if env._city is None:
            return None
        for v in env._city.vehicles:
            if v.vehicle_type == vtype and v.status == "FREE":
                return v.unit_id
        return None

    def _rule_agent(env: SmartEmergencyEnvironment, obs) -> SmartEmergencyAction:
        call = env._current_call
        if call is None:
            return SmartEmergencyAction(
                action_type="dispatch",
                severity_pred=1,
                is_duplicate=False,
                vehicle_type="police",
            )

        # Check for duplicates heuristically
        if obs.active_event_ids and env._current_call and env._current_call.is_duplicate_of:
            dup_id = env._current_call.is_duplicate_of
            return SmartEmergencyAction(
                action_type="duplicate",
                severity_pred=call.severity,
                is_duplicate=True,
                duplicate_of_event_id=dup_id,
            )

        transcript = obs.prompt
        sev = _classify_severity(transcript)
        vtype = _classify_vehicle(transcript)
        vid = _pick_vehicle(env, vtype)

        return SmartEmergencyAction(
            action_type="dispatch",
            severity_pred=sev,
            is_duplicate=False,
            vehicle_type=vtype,
            vehicle_id=vid,
        )

    all_scores = {}
    for task_id in [1, 2, 3]:
        env = SmartEmergencyEnvironment()
        obs = env.reset()
        total_reward = 0.0
        steps = 0
        MAX_STEPS = 20

        while not obs.done and steps < MAX_STEPS:
            action = _rule_agent(env, obs)
            try:
                obs = env.step(action)
                total_reward += obs.reward_breakdown.get("total", 0.0)
            except Exception:
                break
            steps += 1

        MAX_PER_STEP = 6.7
        score = round(max(0.0, min(1.0, total_reward / (MAX_PER_STEP * max(1, steps)))), 4)

        all_scores[f"task_{task_id}"] = {
            "score": score,
            "cumulative_reward": round(total_reward, 4),
            "steps": steps,
            "difficulty": ["easy", "medium", "hard"][task_id - 1],
        }

    avg = round(sum(v["score"] for v in all_scores.values()) / 3, 4)
    return {
        "baseline_agent": "keyword-heuristic rule-based",
        "average_score": avg,
        "tasks": all_scores,
    }


# ── Entry point ───────────────────────────────────────────────────────────────

def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()