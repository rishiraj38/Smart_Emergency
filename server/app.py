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
    if not _env.done:
        remaining = len(_env.calls) - _env.call_index
        raise HTTPException(400, 
            f"Episode not complete. {remaining} calls remaining.")
    return {
        "score": round(max(0.0, min(1.0, 
            _env.cumulative_reward / max(1, _env.step_count * 10))), 4),
        "cases_resolved": _env.cases_resolved,
        "critical_failures": _env.critical_failures,
        "total_steps": _env.step_count,
        "cumulative_reward": _env.cumulative_reward,
    }

@app.get("/baseline")
def baseline():
    """Run rule-based agent against all 3 tasks."""
    from environment import EmergencyDispatchEnvironment
    
    def rule_agent(obs: dict) -> dict:
        transcript = (obs.get("transcript") or "").lower()
        current_call_id = obs.get("current_call_id")
        active_events = obs.get("active_events", [])
        resources = obs.get("resources", [])
        
        # If there's a call to classify
        if current_call_id and transcript:
            if any(w in transcript for w in ["fire", "gas", "explosion", "smoke"]):
                severity = "CRITICAL"
            elif any(w in transcript for w in ["dying", "blood", "accident", "heart"]):
                severity = "CRITICAL"
            elif any(w in transcript for w in ["hurt", "injured", "crash"]):
                severity = "SEMI_CRITICAL"
            else:
                severity = "NORMAL"
            return {"command": "CLASSIFY", "severity": severity}
        
        # If there are unserved events, dispatch
        for event in active_events:
            if event["status"] == "unserved":
                eid = event["event_id"]
                needed = event.get("resource_needed", "ambulance")
                prefix_map = {
                    "ambulance": "AMB", 
                    "fire_truck": "FIRE", 
                    "police": "POL"
                }
                prefix = prefix_map.get(needed, "AMB")
                for v in resources:
                    if v["status"] == "available" and v["vehicle_id"].startswith(prefix):
                        return {
                            "command": "DISPATCH",
                            "vehicle_id": v["vehicle_id"],
                            "event_id": eid
                        }
        return {"command": "WAIT"}
    
    env = EmergencyDispatchEnvironment()
    scores = {}
    
    for task_id in [1, 2, 3]:
        obs = env.reset(task_id=task_id).model_dump()
        total_reward = 0
        steps = 0
        while not obs.get("done", False) and steps < 200:
            action_dict = rule_agent(obs)
            from models import EmergencyAction
            action = EmergencyAction(**action_dict)
            result = env.step(action).model_dump()
            obs = result["observation"]
            total_reward += result["reward"]
            steps += 1
        
        scores[f"task_{task_id}"] = {
            "score": round(env.cases_resolved / max(1, steps) * 10, 4),
            "cases_resolved": env.cases_resolved,
            "steps": steps,
            "difficulty": ["easy", "medium", "hard"][task_id - 1]
        }
    
    avg = sum(v["score"] for v in scores.values()) / 3
    return {
        "baseline_agent": "rule-based keyword heuristics",
        "average_score": round(avg, 4),
        "tasks": scores
    }

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000, log_level="info")

if __name__ == "__main__":
    main()
