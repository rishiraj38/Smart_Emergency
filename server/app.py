import os
import sys
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Ensure root directory is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.responses import RedirectResponse
from models import EmergencyAction, ResetRequest
from server.environment import EmergencyDispatchEnvironment

app = FastAPI(
    title="Emergency Dispatch — OpenEnv Environment",
    description="A disaster management RL environment for dispatching emergency resources.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global environment instance
_env = EmergencyDispatchEnvironment()

@app.get("/", include_in_schema=False)
@app.get("/web", include_in_schema=False)
def root():
    """Redirect to documentation."""
    return RedirectResponse(url="/docs")

@app.get("/health")
def health():
    return {"status": "healthy", "environment": "emergency-dispatch", "version": "1.0.0"}

@app.post("/reset")
def reset(request: ResetRequest = ResetRequest()):
    try:
        obs = _env.reset(task_id=request.task_id, seed=request.seed)
        return obs.model_dump()
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

@app.post("/step")
def step(action: EmergencyAction):
    if _env.done:
        raise HTTPException(status_code=400, detail="Episode is complete. Call /reset.")
    try:
        result = _env.step(action)
        return result.model_dump()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/state")
def state():
    return _env.state().model_dump()

@app.get("/tasks")
def tasks():
    return {
        "tasks": [
            {
                "id": 1,
                "name": "Basic Triage",
                "difficulty": "easy",
                "description": "Classify incoming calls correctly.",
            },
            {
                "id": 2,
                "name": "Resource Management",
                "difficulty": "medium",
                "description": "Triage and dispatch vehicles to events.",
            },
            {
                "id": 3,
                "name": "Disaster Response",
                "difficulty": "hard",
                "description": "Full-scale disaster management under pressure.",
            }
        ]
    }
@app.post("/grader")
def grader():
    """Grade the completed episode. Call after done=True."""
    if not _env.done:
        active = sum(1 for e in _env.event_queue if e["status"] != "resolved")
        raise HTTPException(
            status_code=400,
            detail=(
                f"Episode not complete. "
                f"Active events: {active}, "
                f"Calls remaining: {len(_env.calls) - _env.call_index}. "
                "Keep calling POST /step until obs.done == true."
            ),
        )

    total_events = max(1, len(_env.event_queue))
    fix_rate = round(_env.cases_resolved / total_events, 4)
    score = round(max(0.0, min(1.0, fix_rate - (_env.critical_failures * 0.2))), 4)

    return {
        "score": score,
        "fix_rate": fix_rate,
        "cases_resolved": _env.cases_resolved,
        "total_events": total_events,
        "critical_failures": _env.critical_failures,
        "total_steps": _env.step_count,
        "cumulative_reward": round(_env.cumulative_reward, 4),
        "episode_id": _env.episode_id,
        "task_id": _env.task_id,
    }

@app.get("/baseline")
def baseline():
    """Run rule-based agent on all 3 tasks. Required for hackathon."""
    from server.environment import EmergencyDispatchEnvironment
    from models import EmergencyAction

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

    def rule_agent(obs: dict) -> dict:
        transcript = obs.get("transcript") or ""
        current_call_id = obs.get("current_call_id")
        events = obs.get("active_events", [])
        resources = obs.get("resources", [])
        prefix_map = {"ambulance": "AMB", "fire_truck": "FIRE", "police": "POL"}

        # Priority 1: Dispatch to any unserved event (especially critical ones first)
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

    all_scores = {}
    for task_id in [1, 2, 3]:
        env = EmergencyDispatchEnvironment()
        obs = env.reset(task_id=task_id).model_dump()
        steps = 0
        while not obs.get("done", False) and steps < 500:
            action_dict = rule_agent(obs)
            try:
                result = env.step(EmergencyAction(**action_dict)).model_dump()
                obs = result["observation"]
            except Exception:
                break
            steps += 1

        total = max(1, len(env.event_queue))
        fix_rate = round(env.cases_resolved / total, 4)
        score = round(max(0.0, min(1.0, fix_rate - env.critical_failures * 0.2)), 4)

        all_scores[f"task_{task_id}"] = {
            "score": score,
            "fix_rate": fix_rate,
            "cases_resolved": env.cases_resolved,
            "critical_failures": env.critical_failures,
            "steps": steps,
            "difficulty": ["easy", "medium", "hard"][task_id - 1],
        }

    avg = round(sum(v["score"] for v in all_scores.values()) / 3, 4)
    return {
        "baseline_agent": "rule-based keyword heuristics",
        "average_score": avg,
        "tasks": all_scores,
    }


def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, log_level="info")

if __name__ == "__main__":
    main()
