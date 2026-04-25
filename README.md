---
title: Smart Emergency Environment Server
emoji: 🚨
colorFrom: pink
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Smart Emergency — Dispatch911 RL Environment

A disaster management reinforcement learning environment where an agent acts as an emergency dispatcher. Each episode, the agent receives live 911 call transcripts and must triage severity, detect duplicate calls, and dispatch the right vehicle (police / ambulance / fire) from a procedurally generated city graph.

Built on [OpenEnv](https://github.com/meta-pytorch/OpenEnv) — a standard interface for RL environments exposed over HTTP/WebSocket, compatible with TRL + Unsloth training pipelines.

---

## Environment Overview

| Property | Value |
|---|---|
| **Task** | Emergency dispatch (triage + routing) |
| **Episode length** | 20 steps |
| **Action space** | `dispatch` or `duplicate` with structured fields |
| **Observation** | Rich text prompt (call transcript + active events + fleet status + city map) |
| **Reward** | 5-component shaped reward (severity, duplicate detection, vehicle type, vehicle choice, reroute) |
| **Duplicate call rate** | 30% |

---

## Quick Start

```python
from smart_emergency import SmartEmergencyAction, SmartEmergencyEnv

with SmartEmergencyEnv(base_url="http://localhost:8000") as env:
    result = env.reset()
    print(result.observation.prompt)

    # Dispatch an ambulance to the incident
    action = SmartEmergencyAction(
        action_type="dispatch",
        severity_pred=3,
        is_duplicate=False,
        vehicle_type="ambulance",
        vehicle_id="ambulance_0",
    )
    result = env.step(action)
    print(result.observation.reward_breakdown)
    # → {'severity': 1.0, 'duplicate': 1.0, 'vehicle_type': 1.5, 'vehicle_choice': 0.5, 'reroute': 0.0, 'total': 4.0}
```

---

## Action Space

**`SmartEmergencyAction`** — the agent's structured response to each incoming 911 call.

| Field | Type | Required | Description |
|---|---|---|---|
| `action_type` | `str` | ✅ | `"dispatch"` or `"duplicate"` |
| `severity_pred` | `int` (1–5) | ✅ | Predicted severity (1=minor, 5=catastrophic) |
| `is_duplicate` | `bool` | ✅ | Whether this call is a repeat of an existing event |
| `duplicate_of_event_id` | `str` | if duplicate | EVT-NNNN of the event this duplicates |
| `vehicle_type` | `str` | if dispatch | `"police"`, `"ambulance"`, or `"fire"` |
| `vehicle_id` | `str` | if dispatch | Specific unit ID (e.g. `"ambulance_0"`) |
| `reroute` | `RerouteAction` | optional | Redirect an in-flight vehicle to the new event |

**`RerouteAction`** sub-action:

| Field | Type | Description |
|---|---|---|
| `vehicle_to_reroute` | `str` | Unit ID of the vehicle to redirect |
| `from_event_id` | `str` | EVT-NNNN the vehicle is currently heading to |
| `replacement_vehicle_id` | `str` | Optional free unit to cover the abandoned event |

---

## Observation Space

**`SmartEmergencyObservation`** — what the agent sees each step.

| Field | Type | Description |
|---|---|---|
| `prompt` | `str` | Full text observation for the LLM (see format below) |
| `step` | `int` | Current step number (0–20) |
| `call_id` | `str` | ID of the incoming call (e.g. `CALL-0001`) |
| `reward_breakdown` | `dict` | Per-component reward from the previous action |
| `active_event_ids` | `list[str]` | Currently active event IDs (EVT-NNNN) |
| `fleet_utilisation` | `float` | Fraction of fleet currently busy (0.0–1.0) |

### Prompt Format

```
=== INCOMING CALL [CALL-0003] ===
Bad crash on Oak Avenue! Car flipped near Riverside Market. Driver trapped, not responding!

=== ACTIVE EVENTS ===
EVT-0001 | fire       | Engine House No. 1             | sev 3 | fire_2 ETA 2 min | opened step 1
EVT-0002 | medical    | Oakwood Apartments             | sev 2 | UNASSIGNED       | opened step 2

=== UNIT STATUS ===
police_0        | police     | Central Police Station        | FREE
ambulance_1     | ambulance  | Riverside General Hospital    | DISPATCHED → EVT-0001
fire_2          | fire       | Central Fire Station          | DISPATCHED → EVT-0001

=== CITY REFERENCE ===
Riverside General Hospital (hospital) → Oakwood Apartments [3 min], Central Plaza [5 min]
...

=== DISPATCHER NOTES ===
Step 1: CALL-0001 → fire fire_2
Step 2: CALL-0002 → Duplicate of EVT-0001
```

---

## Reward Design

5 independent reward components returned as `reward_breakdown`:

| Component | Max | Min | Description |
|---|---|---|---|
| `severity` | +1.0 | -0.5 | Accuracy of severity prediction (graded, ±0 to ±4 off) |
| `duplicate` | +1.5 | -1.0 | Correct duplicate detection and event ID matching |
| `vehicle_type` | +1.5 | -1.5 | Correct vehicle type (police / ambulance / fire) |
| `vehicle_choice` | +1.0 | -2.0 | Vehicle availability, type match, and proximity bonus |
| `reroute` | +1.7 | -1.0 | Quality of optional reroute instruction |
| **`total`** | **~6.7** | **~-6.0** | Sum of all components |

Parse failure (malformed action): **-2.0** flat penalty.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/reset` | Start a new episode |
| `POST` | `/step` | Submit an action, get next observation |
| `GET` | `/state` | Current episode state |
| `GET` | `/tasks` | List available tasks / difficulty levels |
| `POST` | `/grader` | Score a completed episode (call after `done=True`) |
| `GET` | `/baseline` | Run rule-based agent across all tasks |
| `GET` | `/docs` | Interactive Swagger UI |
| `WS` | `/ws` | WebSocket for persistent low-latency sessions |

---

## Running Locally

### Option 1: uv (fastest)

```bash
uv sync
uv run uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
```

Or via the Makefile:

```bash
make serve      # uv run, with hot-reload
make build      # build Docker image
make start      # run Docker container
```

### Option 2: Docker

```bash
make build
make start
```

Then open http://localhost:8000/docs

---

## Connecting to a Running Server

```python
from smart_emergency import SmartEmergencyEnv

env = SmartEmergencyEnv(base_url="http://localhost:8000")
result = env.reset()
print(result.observation.prompt)
```

Or use the deployed HF Space directly:

```python
env = SmartEmergencyEnv(base_url="https://rishi38-eme-enviro.hf.space")
```

---

## Grading a Completed Episode

After the episode ends (`done=True`), call `/grader`:

```bash
curl -X POST http://localhost:8000/grader
```

```json
{
  "score": 0.82,
  "reward_components": {
    "severity_accuracy": 0.91,
    "duplicate_f1": 0.75,
    "dispatch_accuracy": 0.88,
    "vehicle_efficiency": 0.74
  },
  "steps": 20,
  "episode_id": "abc-123"
}
```

---

## Baseline Agent

Run the built-in rule-based agent to get a reference score:

```bash
curl http://localhost:8000/baseline
```

```json
{
  "baseline_agent": "keyword-heuristic rule-based",
  "average_score": 0.61,
  "tasks": {
    "task_1": {"score": 0.72, "difficulty": "easy", "steps": 20},
    "task_2": {"score": 0.63, "difficulty": "medium", "steps": 20},
    "task_3": {"score": 0.48, "difficulty": "hard", "steps": 20}
  }
}
```

---

## Project Structure

```
smart_emergency/
├── README.md                        # This file (HF Space config + docs)
├── openenv.yaml                     # OpenEnv manifest
├── pyproject.toml                   # Package metadata & dependencies
├── Dockerfile                       # Container build
├── Makefile                         # Dev commands (build, start, serve)
├── uv.lock                          # Locked dependencies
├── __init__.py                      # Package exports
├── models.py                        # SmartEmergencyAction + Observation
├── client.py                        # SmartEmergencyEnv HTTP/WS client
└── server/
    ├── __init__.py
    ├── app.py                       # FastAPI app via openenv create_app
    ├── smart_emergency_environment.py  # Core reset/step/reward logic
    ├── city.py                      # Procedural city graph + Dijkstra
    ├── calls.py                     # 911 call generator (25 templates)
    └── reward.py                    # 5-component decomposed reward
```
