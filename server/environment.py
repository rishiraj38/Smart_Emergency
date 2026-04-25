import random
import uuid
import time
from typing import Dict, List, Optional, Any
import copy

from models import (
    EmergencyAction,
    EmergencyObservation,
    EmergencyReward,
    EmergencyState,
    StepResult,
)
from city_graph import CityGraph
from call_generator import CallGenerator
from reward import RewardCalculator
from config import REWARD, EPISODE, CITY

# Map event types to the resource they need
EVENT_TYPE_TO_RESOURCE = {
    "Building Fire":      "fire_truck",
    "Car Accident":       "ambulance",
    "Medical Emergency":  "ambulance",
    "Gas Leak":           "fire_truck",
    "Violent Incident":   "police",
    "Noise Complaint":    "police",
}

VEHICLE_TYPE_MAP = {
    "AMB":  "ambulance",
    "FIRE": "fire_truck",
    "POL":  "police",
}

TASK_META = {
    1: {"name": "Basic Triage",        "difficulty": "easy"},
    2: {"name": "Resource Management", "difficulty": "medium"},
    3: {"name": "Disaster Response",   "difficulty": "hard"},
}

class EmergencyDispatchEnvironment:
    """
    Smart City Emergency Dispatch Environment — OpenEnv compliant.
    Following Hackathon best practices:
    - Multi-component rewards
    - Step limits and timeouts
    - Resource constraints
    """

    def __init__(self, seed: int = 42):
        self.episode_id: Optional[str] = None
        self.task_id: int = 1
        self.seed = seed
        self.rng = random.Random(seed)

        # Simulation components
        self.graph = None
        self.reward_calc = RewardCalculator()
        self.call_gen = CallGenerator(seed=seed)

        # Episode state
        self.event_queue = []
        self.resources = []
        self.calls = []
        self.call_index = 0
        self.current_call = None
        self.step_count = 0
        self.done = True
        self.cumulative_reward = 0.0
        self.cases_resolved = 0
        self.critical_failures = 0

        # Bookkeeping
        self._ground_truth = {}
        self._dispatched_events = {}
        self._classified_calls = set()
        self._step_start_time = 0

    def reset(self, task_id: int = 1, seed: Optional[int] = None) -> EmergencyObservation:
        """Start a new episode."""
        if task_id not in TASK_META:
            raise ValueError(f"task_id must be 1, 2, or 3 — got {task_id}")

        self.episode_id = str(uuid.uuid4())
        self.task_id = task_id
        if seed is not None:
            self.seed = seed
        
        self.rng = random.Random(self.seed)
        random.seed(self.seed)

        # Build city and vehicles
        self.graph = CityGraph(seed=self.seed)
        self._init_vehicles()

        # Generate calls
        self.calls = self.call_gen.generate_episode_calls(self.graph.nodes)
        self.call_index = 0

        # Reset state
        self.event_queue = []
        self.step_count = 0
        self.done = False
        self.cumulative_reward = 0.0
        self.cases_resolved = 0
        self.critical_failures = 0
        self._ground_truth = {}
        self._dispatched_events = {}
        self._classified_calls = set()
        self.reward_calc.reset()

        # Load first calls
        for _ in range(min(3, len(self.calls))):
            self._advance_call()

        return self._make_observation(reward=0.0)

    def step(self, action: EmergencyAction) -> StepResult:
        """Execute one step in the environment."""
        if self.done:
            raise ValueError("Episode is done. Call reset().")

        self._step_start_time = time.time()
        self.step_count += 1
        
        # 1. Execute the action (with safety checks)
        self._execute_action(action)

        # 2. Advance simulation
        self._tick_vehicles()
        self._check_resolutions()
        self._advance_call()
        self._tick_waiting_steps()

        # 3. Calculate rewards (multi-component)
        reward_breakdown = self.reward_calc.calculate_step_reward()
        step_reward = reward_breakdown["total"]
        self.cumulative_reward += step_reward

        # 4. Check termination
        self._check_termination()

        obs = self._make_observation(reward=step_reward)
        
        # Include breakdown in info for better monitoring (Hackathon Guide Step 15)
        return StepResult(
            observation=obs,
            reward=step_reward,
            done=self.done,
            info={
                "reward_breakdown": reward_breakdown,
                "cases_resolved": self.cases_resolved,
                "critical_failures": self.critical_failures,
                "events_active": len([e for e in self.event_queue if e["status"] != "resolved"]),
                "step_latency_ms": (time.time() - self._step_start_time) * 1000
            }
        )

    def state(self) -> EmergencyState:
        """Return current metadata."""
        return EmergencyState(
            episode_id=self.episode_id or "",
            task_id=self.task_id,
            step_count=self.step_count,
            cumulative_reward=round(self.cumulative_reward, 4),
            done=self.done
        )

    # ──────────────────────────────────────────────────────────────
    # PRIVATE HELPERS
    # ──────────────────────────────────────────────────────────────

    def _init_vehicles(self):
        self.resources = []
        stations = ["STATION_1", "STATION_2", "STATION_3"]
        counts = {
            "AMB": CITY["AMBULANCES"],
            "FIRE": CITY["FIRE_TRUCKS"],
            "POL": CITY["POLICE"]
        }
        
        for prefix, count in counts.items():
            v_type = VEHICLE_TYPE_MAP[prefix]
            for i in range(count):
                self.resources.append({
                    "vehicle_id": f"{prefix}-{i+1:02d}",
                    "type": v_type,
                    "status": "available",
                    "current_location": stations[i % len(stations)],
                    "destination": None,
                    "destination_event": None,
                    "path": [],
                    "steps_to_arrival": 0,
                })

    def _execute_action(self, action: EmergencyAction):
        cmd = action.command
        
        # Anti-hacking: Reject actions if step time exceeded (theoretical here)
        if time.time() - self._step_start_time > 1.0: # 1 second limit
             self.reward_calc.record_signal("penalties", -1.0)
             return

        # Mapping EmergencyAction to internal handlers
        if cmd == "CLASSIFY":
            self._handle_classify(action.event_id, action.severity)
        elif cmd == "MERGE":
            self._handle_merge(action.call_id or action.event_id, action.into_event_id)
        elif cmd == "ESCALATE":
            self._handle_escalate(action.event_id, action.severity)
        elif cmd == "DISCARD":
            self._handle_discard(action.call_id)
        elif cmd == "DISPATCH":
            self._handle_dispatch(action.vehicle_id, action.event_id)
        elif cmd == "REROUTE":
            self._handle_reroute(action.vehicle_id, action.event_id)
        elif cmd == "HOLD":
            self._handle_hold(action.event_id)
        elif cmd == "RECALL":
            self._handle_recall(action.vehicle_id)
        elif cmd == "WAIT":
            self.reward_calc.record_signal("efficiency", -0.05) # Small penalty for doing nothing
        else:
            self.reward_calc.record_signal("penalties", -1.0)

    def _make_observation(self, reward: float) -> EmergencyObservation:
        return EmergencyObservation(
            current_call_id=self.current_call["call_id"] if self.current_call else None,
            transcript=self.current_call["transcript"] if self.current_call else None,
            active_events=copy.deepcopy(self.event_queue),
            resources=copy.deepcopy(self.resources),
            step=self.step_count,
            calls_remaining=len(self.calls) - self.call_index,
            cases_resolved=self.cases_resolved,
            reward=reward,
            cumulative_reward=round(self.cumulative_reward, 4),
            done=self.done
        )

    def _handle_classify(self, event_id, severity):
        if not self.current_call or not severity:
            self.reward_calc.record_signal("penalties", -0.5)
            return
        
        call = self.current_call
        gt_event = call["ground_truth_event"]
        gt_severity = call["ground_truth_severity"]
        
        if event_id and event_id not in {call["call_id"], gt_event}:
             self.reward_calc.record_signal("penalties", -0.5)
             return

        resource_needed = EVENT_TYPE_TO_RESOURCE.get(call.get("ground_truth_type", ""), "ambulance")
        
        # Create or update event
        existing = [e for e in self.event_queue if e["event_id"] == gt_event]
        if not existing:
            self.event_queue.append({
                "event_id": gt_event,
                "severity": severity,
                "location": call["ground_truth_node"],
                "resource_needed": resource_needed,
                "steps_waiting": 0,
                "status": "unserved",
            })
            self.reward_calc.add_waiting_event(gt_event, severity)
            self._ground_truth[gt_event] = {
                "severity": gt_severity,
                "node": call["ground_truth_node"],
            }

        # Scoring (Triage signal)
        if severity == gt_severity:
            self.reward_calc.record_signal("triage", 1.0)
        elif self._severity_rank(severity) < self._severity_rank(gt_severity):
            self.reward_calc.record_signal("triage", -0.5) # Penalty for underestimating
        else:
            self.reward_calc.record_signal("triage", 0.2) # Small positive for overestimating (safety first)

        self._classified_calls.add(call["call_id"])
        self.current_call = None

    def _handle_dispatch(self, vehicle_id, event_id):
        vehicle = self._find_vehicle(vehicle_id)
        event = self._find_event(event_id)

        if not vehicle or not event or event["status"] == "resolved" or vehicle["status"] != "available":
            self.reward_calc.record_signal("penalties", -0.5)
            return

        # Double dispatch penalty (Anti-hacking/Optimization)
        if event_id in self._dispatched_events and len(self._dispatched_events[event_id]) > 0:
            self.reward_calc.record_signal("penalties", -1.0)
            return

        # Vehicle type match
        v_prefix = vehicle_id.split("-")[0]
        v_type = VEHICLE_TYPE_MAP.get(v_prefix, "")
        if v_type != event["resource_needed"]:
            self.reward_calc.record_signal("dispatch", -0.5)
        else:
            self.reward_calc.record_signal("dispatch", 0.5)

        # Route vehicle
        path = self.graph.shortest_path(vehicle["current_location"], event["location"])
        if path:
            # Distance penalty (Efficiency signal)
            dist_penalty = (len(path) - 1) * -0.05
            self.reward_calc.record_signal("efficiency", dist_penalty)
            
            vehicle["status"] = "en_route"
            vehicle["destination"] = event["location"]
            vehicle["destination_event"] = event_id
            vehicle["path"] = path[1:]
            vehicle["steps_to_arrival"] = len(path) - 1
            if not vehicle["path"]:
                vehicle["status"] = "on_scene"
            self._dispatched_events.setdefault(event_id, []).append(vehicle_id)
        else:
            self.reward_calc.record_signal("penalties", -0.5)

    def _handle_merge(self, call_id, into_event_id):
        if not self.current_call: return
        if into_event_id == self.current_call["ground_truth_event"]:
            self.reward_calc.record_signal("triage", 0.8)
        else:
            self.reward_calc.record_signal("triage", -0.8)
        self.current_call = None

    def _handle_discard(self, call_id):
        if self.current_call: 
            # Was it a real event? (Anti-hacking: penalize discarding real emergencies)
            if self.current_call.get("ground_truth_severity") != "NONE":
                 self.reward_calc.record_signal("penalties", -2.0)
            self.current_call = None

    def _handle_escalate(self, event_id, new_severity):
        event = self._find_event(event_id)
        if event and new_severity:
            event["severity"] = new_severity
            self.reward_calc.add_waiting_event(event_id, new_severity)
            self.reward_calc.record_signal("triage", 0.1)

    def _handle_reroute(self, vehicle_id, new_event_id):
        self._handle_recall(vehicle_id)
        self._handle_dispatch(vehicle_id, new_event_id)
        self.reward_calc.record_signal("efficiency", -0.2) # Small penalty for changing mind

    def _handle_hold(self, event_id):
        event = self._find_event(event_id)
        if event:
            if event["severity"] == "CRITICAL":
                self.reward_calc.record_signal("penalties", -1.0)
            else:
                self.reward_calc.record_signal("efficiency", -0.1)

    def _handle_recall(self, vehicle_id):
        vehicle = self._find_vehicle(vehicle_id)
        if vehicle and vehicle["status"] in ("en_route", "on_scene"):
            old_event = vehicle.get("destination_event")
            if old_event in self._dispatched_events:
                if vehicle_id in self._dispatched_events[old_event]:
                    self._dispatched_events[old_event].remove(vehicle_id)
            vehicle["status"] = "available"
            vehicle["destination"] = None
            vehicle["destination_event"] = None
            vehicle["path"] = []
            vehicle["steps_to_arrival"] = 0
            self.reward_calc.record_signal("efficiency", -0.3)

    def _tick_vehicles(self):
        for v in self.resources:
            if v["status"] == "en_route" and v["path"]:
                v["current_location"] = v["path"].pop(0)
                v["steps_to_arrival"] = len(v["path"])
                if not v["path"]:
                    v["status"] = "on_scene"

    def _check_resolutions(self):
        for event in self.event_queue:
            if event["status"] == "resolved": continue
            eid = event["event_id"]
            for v in self.resources:
                if v["status"] == "on_scene" and v["destination_event"] == eid:
                    if v["type"] == event["resource_needed"]:
                        event["status"] = "resolved"
                        self.cases_resolved += 1
                        self.reward_calc.remove_waiting_event(eid)
                        v["status"] = "available"
                        # Success reward (Triage/Dispatch combo)
                        bonus = {"CRITICAL": 5.0, "SEMI_CRITICAL": 3.0, "NORMAL": 1.0}.get(event['severity'], 0.0)
                        self.reward_calc.record_signal("triage", bonus)
                        break

    def _advance_call(self):
        if self.current_call is None and self.call_index < len(self.calls):
            self.current_call = self.calls[self.call_index]
            self.call_index += 1

    def _tick_waiting_steps(self):
        for event in self.event_queue:
            if event["status"] != "resolved":
                event["steps_waiting"] += 1

    def _check_termination(self):
        # Termination conditions (Anti-loop)
        for event in self.event_queue:
            if event["severity"] == "CRITICAL" and event["steps_waiting"] > EPISODE["CRITICAL_FAILURE_WAIT"]:
                self.done = True
                self.critical_failures += 1
                self.reward_calc.record_signal("penalties", -10.0) # Huge penalty for critical failure
        
        if self.step_count >= EPISODE["MAX_STEPS"]:
            self.done = True
            
        if self.call_index >= len(self.calls) and self.current_call is None and all(e["status"] == "resolved" for e in self.event_queue):
            self.done = True

    def _find_event(self, eid):
        for e in self.event_queue:
            if e["event_id"] == eid: return e
        return None

    def _find_vehicle(self, vid):
        for v in self.resources:
            if v["vehicle_id"] == vid: return v
        return None

    @staticmethod
    def _severity_rank(severity: str) -> int:
        return {"CRITICAL": 3, "SEMI_CRITICAL": 2, "NORMAL": 1}.get(severity, 0)
