"""
Dispatch911 Environment — OpenEnv-compatible Gym environment.

Handles reset/step loop, vehicle lifecycle, event registry,
observation formatting, and reward integration.
"""

import random
from typing import Dict, List, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import SmartEmergencyAction, SmartEmergencyObservation
except ImportError:
    from models import SmartEmergencyAction, SmartEmergencyObservation

from .city import City, Destination, Vehicle, dijkstra, generate_city
from .calls import Call, generate_call
from .reward import PARSE_FAILURE_PENALTY, compute_reward

# ── Config defaults ──────────────────────────────────────────────────────────

MAX_STEPS = 20
DUPLICATE_PROB = 0.30
ON_SCENE_STEPS = 2
RETURN_STEPS = 2


class SmartEmergencyEnvironment(Environment):
    """
    Dispatch911 RL environment.

    Each episode = one procedurally generated city.
    Each step = one incoming 911 call.
    The agent outputs a structured JSON action; the environment
    evaluates it against hidden ground truth and returns a shaped reward.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._city: Optional[City] = None
        self._rng = random.Random()
        self._active_events: Dict[str, dict] = {}
        self._event_counter = 1
        self._current_call: Optional[Call] = None
        self._dispatcher_notes: List[str] = []
        self._seed = 0

    # ── Reset ────────────────────────────────────────────────────────────

    def reset(self) -> SmartEmergencyObservation:
        self._seed = random.randint(0, 999999)
        self._rng = random.Random(self._seed)
        self._city = generate_city(self._seed)
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._active_events = {}
        self._event_counter = 1
        self._dispatcher_notes = []

        # Generate first call
        self._current_call, self._event_counter = generate_call(
            self._city, 1, self._active_events,
            DUPLICATE_PROB, self._rng, self._event_counter,
        )

        obs_text = self._build_observation()
        return SmartEmergencyObservation(
            prompt=obs_text,
            step=0,
            call_id=self._current_call.call_id,
            reward_breakdown={},
            active_event_ids=list(self._active_events.keys()),
            fleet_utilisation=self._fleet_util(),
            done=False,
            reward=0.0,
        )

    # ── Step ─────────────────────────────────────────────────────────────

    def step(self, action: SmartEmergencyAction) -> SmartEmergencyObservation:
        # Auto-reset if step is called before reset
        if self._current_call is None or self._city is None:
            self.reset()

        self._state.step_count += 1
        call = self._current_call
        city = self._city
        assert call is not None and city is not None

        # ── Evaluate action ──────────────────────────────────────────────
        reward_kwargs = self._evaluate_action(action, call)
        breakdown = compute_reward(**reward_kwargs)

        # ── Update state ─────────────────────────────────────────────────
        self._apply_action(action, call)

        # ── Advance simulation clock ─────────────────────────────────────
        self._tick_vehicles()

        # ── Log dispatcher note ──────────────────────────────────────────
        note = f"Step {self._state.step_count}: {call.call_id}"
        if action.is_duplicate:
            note += f" → Duplicate of {action.duplicate_of_event_id or '?'}"
        elif action.action_type == "hold":
            note += f" → HOLD ({action.vehicle_type}, waiting for {action.vehicle_id or '?'})"
        elif action.action_type == "dispatch":
            note += f" → {action.vehicle_type} {action.vehicle_id or '?'}"
        self._dispatcher_notes.append(note)
        if len(self._dispatcher_notes) > 3:
            self._dispatcher_notes = self._dispatcher_notes[-3:]

        # ── Check done ───────────────────────────────────────────────────
        done = self._state.step_count >= MAX_STEPS

        # ── Generate next call ───────────────────────────────────────────
        if not done:
            self._current_call, self._event_counter = generate_call(
                city, self._state.step_count + 1,
                self._active_events, DUPLICATE_PROB,
                self._rng, self._event_counter,
            )
        obs_text = self._build_observation() if not done else "Episode complete."

        return SmartEmergencyObservation(
            prompt=obs_text,
            step=self._state.step_count,
            call_id=call.call_id,
            reward_breakdown=breakdown,
            active_event_ids=list(self._active_events.keys()),
            fleet_utilisation=self._fleet_util(),
            done=done,
            reward=breakdown.get("total", 0.0),
            metadata={
                "ground_truth": {
                    "severity": call.severity,
                    "emergency_type": call.emergency_type,
                    "is_duplicate": call.is_duplicate_of is not None,
                    "required_vehicle_type": call.required_vehicle_type,
                },
                "city_seed": self._seed,
            },
        )

    # ── Evaluate ─────────────────────────────────────────────────────────

    def _evaluate_action(self, action: SmartEmergencyAction, call: Call) -> dict:
        """Build kwargs for compute_reward."""
        city = self._city
        assert city is not None

        gt_is_dup = call.is_duplicate_of is not None
        gt_eid = call.is_duplicate_of

        # Vehicle checks
        v_exists = True
        v_free = True
        v_type_match = True
        travel = 0.0
        is_nearest = False

        # Hold checks
        hold_is_action = action.action_type == "hold"
        hold_free_exists = False
        hold_min_busy_sev = 0
        hold_vehicle_soonest = False

        if hold_is_action:
            # Check if the named vehicle exists and its state
            if action.vehicle_id:
                veh = self._find_vehicle(action.vehicle_id)
                if veh is None:
                    v_exists = False
                else:
                    v_free = veh.status == "FREE"
                    v_type_match = veh.vehicle_type == (action.vehicle_type or "")
            else:
                v_exists = False

            # Check if any free unit of the correct type exists
            vtype = action.vehicle_type or call.required_vehicle_type
            free_of_type = [
                v for v in city.vehicles
                if v.status == "FREE" and v.vehicle_type == vtype
            ]
            hold_free_exists = len(free_of_type) > 0

            # Find min severity among busy units of this type
            busy_of_type = [
                v for v in city.vehicles
                if v.status != "FREE" and v.vehicle_type == vtype
                and v.assigned_event is not None
            ]
            if busy_of_type:
                busy_sevs = []
                for bv in busy_of_type:
                    evt = self._active_events.get(bv.assigned_event, {})
                    busy_sevs.append(evt.get("severity", 5))
                hold_min_busy_sev = min(busy_sevs)

                # Check if named vehicle is the soonest to free
                if v_exists and not v_free and action.vehicle_id:
                    veh = self._find_vehicle(action.vehicle_id)
                    if veh and veh.eta is not None:
                        min_eta = min(
                            (bv.eta for bv in busy_of_type if bv.eta is not None),
                            default=999,
                        )
                        hold_vehicle_soonest = veh.eta <= min_eta

        elif not action.is_duplicate and action.vehicle_id:
            veh = self._find_vehicle(action.vehicle_id)
            if veh is None:
                v_exists = False
            else:
                v_free = veh.status == "FREE"
                v_type_match = veh.vehicle_type == action.vehicle_type
                if v_exists and v_free:
                    travel, _ = dijkstra(city, veh.current_node, call.origin_node_id)
                    # Check if nearest
                    free_same = [
                        v for v in city.vehicles
                        if v.status == "FREE" and v.vehicle_type == call.required_vehicle_type
                    ]
                    if free_same:
                        min_t = min(dijkstra(city, v.current_node, call.origin_node_id)[0] for v in free_same)
                        is_nearest = abs(travel - min_t) < 0.1

        # Reroute checks
        reroute_attempted = action.reroute is not None and not hold_is_action
        reroute_valid = False
        reroute_sev_delta = 0
        reroute_faster = False
        replacement_valid = None

        if reroute_attempted and action.reroute is not None:
            rv = self._find_vehicle(action.reroute.vehicle_to_reroute)
            if rv and rv.status == "DISPATCHED" and rv.assigned_event == action.reroute.from_event_id:
                reroute_valid = True
                old_evt = self._active_events.get(action.reroute.from_event_id, {})
                reroute_sev_delta = call.severity - old_evt.get("severity", call.severity)
                if action.reroute.replacement_vehicle_id:
                    rep = self._find_vehicle(action.reroute.replacement_vehicle_id)
                    replacement_valid = (
                        rep is not None and rep.status == "FREE"
                        and rep.vehicle_type == old_evt.get("vehicle", "")
                    )

        return dict(
            gt_severity=call.severity,
            gt_is_duplicate=gt_is_dup,
            gt_event_id=gt_eid,
            gt_vehicle_type=call.required_vehicle_type,
            gt_origin_node=call.origin_node_id,
            severity_pred=action.severity_pred,
            is_duplicate_pred=action.is_duplicate,
            duplicate_of_event_id=action.duplicate_of_event_id,
            vehicle_type_pred=action.vehicle_type,
            vehicle_id_pred=action.vehicle_id,
            vehicle_exists=v_exists,
            vehicle_is_free=v_free,
            vehicle_type_matches=v_type_match,
            travel_time=travel,
            is_nearest=is_nearest,
            reroute_attempted=reroute_attempted,
            reroute_valid=reroute_valid,
            reroute_severity_delta=reroute_sev_delta,
            reroute_faster=reroute_faster,
            replacement_valid=replacement_valid,
            hold_is_action=hold_is_action,
            hold_free_unit_exists=hold_free_exists,
            hold_min_busy_severity=hold_min_busy_sev,
            hold_vehicle_is_soonest=hold_vehicle_soonest,
        )

    # ── Apply action to state ────────────────────────────────────────────

    def _apply_action(self, action: SmartEmergencyAction, call: Call):
        city = self._city
        assert city is not None

        if action.is_duplicate:
            # Link call to existing event
            eid = action.duplicate_of_event_id or call.event_id
            if eid in self._active_events:
                self._active_events[eid].setdefault("calls", []).append(call.call_id)
            return

        # Register new event
        eid = call.event_id
        self._active_events[eid] = {
            "type": call.emergency_type,
            "severity": call.severity,
            "vehicle": call.required_vehicle_type,
            "node_id": call.origin_node_id,
            "node_name": call.origin_node_name,
            "assigned_unit": None,
            "unit_eta": None,
            "held_for_unit": None,
            "step_opened": self._state.step_count,
            "calls": [call.call_id],
        }

        # ── Hold action ──────────────────────────────────────────────────
        if action.action_type == "hold" and action.vehicle_id:
            veh = self._find_vehicle(action.vehicle_id)
            if veh is not None and veh.status != "FREE":
                # Queue this event as a future destination for the vehicle
                veh.destinations.append(
                    Destination(node_id=call.origin_node_id, event_id=eid)
                )
                self._active_events[eid]["held_for_unit"] = action.vehicle_id
            return

        # Handle reroute
        if action.reroute is not None:
            rv = self._find_vehicle(action.reroute.vehicle_to_reroute)
            if rv and rv.status == "DISPATCHED":
                # Unassign from old event
                old_eid = action.reroute.from_event_id
                if old_eid in self._active_events:
                    self._active_events[old_eid]["assigned_unit"] = None
                    self._active_events[old_eid]["unit_eta"] = None
                # Dispatch rerouted vehicle to new event
                travel, path = dijkstra(city, rv.current_node, call.origin_node_id)
                rv.status = "DISPATCHED"
                rv.assigned_event = eid
                rv.eta = max(1, int(travel))
                rv.path = path
                self._active_events[eid]["assigned_unit"] = rv.unit_id
                self._active_events[eid]["unit_eta"] = rv.eta
                # Handle replacement
                if action.reroute.replacement_vehicle_id:
                    rep = self._find_vehicle(action.reroute.replacement_vehicle_id)
                    if rep and rep.status == "FREE" and old_eid in self._active_events:
                        old_node = self._active_events[old_eid]["node_id"]
                        t, p = dijkstra(city, rep.current_node, old_node)
                        rep.status = "DISPATCHED"
                        rep.assigned_event = old_eid
                        rep.eta = max(1, int(t))
                        rep.path = p
                        self._active_events[old_eid]["assigned_unit"] = rep.unit_id
                        self._active_events[old_eid]["unit_eta"] = rep.eta
                return

        # Normal dispatch
        if action.vehicle_id:
            veh = self._find_vehicle(action.vehicle_id)
            if veh and veh.status == "FREE":
                travel, path = dijkstra(city, veh.current_node, call.origin_node_id)
                veh.status = "DISPATCHED"
                veh.assigned_event = eid
                veh.eta = max(1, int(travel))
                veh.path = path
                self._active_events[eid]["assigned_unit"] = veh.unit_id
                self._active_events[eid]["unit_eta"] = veh.eta

    # ── Vehicle tick ─────────────────────────────────────────────────────

    def _tick_vehicles(self):
        city = self._city
        assert city is not None
        resolved = []

        for v in city.vehicles:
            if v.status == "DISPATCHED":
                v.eta -= 1
                if v.eta <= 0:
                    v.status = "ON_SCENE"
                    v.on_scene_remaining = ON_SCENE_STEPS
                    if v.path:
                        v.current_node = v.path[-1]
            elif v.status == "ON_SCENE":
                v.on_scene_remaining -= 1
                if v.on_scene_remaining <= 0:
                    v.status = "RETURNING"
                    v.return_remaining = RETURN_STEPS
                    # Mark event resolved
                    if v.assigned_event and v.assigned_event in self._active_events:
                        resolved.append(v.assigned_event)
            elif v.status == "RETURNING":
                v.return_remaining -= 1
                if v.return_remaining <= 0:
                    v.status = "FREE"
                    v.current_node = v.home_node
                    v.assigned_event = None
                    # Auto-dispatch to next queued destination (from hold)
                    self._dispatch_next_destination(v)

        for eid in resolved:
            self._active_events.pop(eid, None)

    def _dispatch_next_destination(self, v: Vehicle):
        """If the vehicle has queued destinations, pop the first and dispatch."""
        city = self._city
        assert city is not None

        while v.destinations:
            dest = v.destinations.pop(0)
            # Only dispatch if the event is still active and unassigned
            evt = self._active_events.get(dest.event_id)
            if evt is not None and evt.get("assigned_unit") is None:
                travel, path = dijkstra(city, v.current_node, dest.node_id)
                v.status = "DISPATCHED"
                v.assigned_event = dest.event_id
                v.eta = max(1, int(travel))
                v.path = path
                evt["assigned_unit"] = v.unit_id
                evt["unit_eta"] = v.eta
                return
        # No valid destinations left — vehicle stays FREE

    # ── Observation builder ──────────────────────────────────────────────

    def _build_observation(self) -> str:
        call = self._current_call
        city = self._city
        if call is None or city is None:
            return ""

        parts = []

        # 1. Incoming call
        parts.append(f"=== INCOMING CALL [{call.call_id}] ===")
        parts.append(call.transcript)
        parts.append("")

        # 2. Active events
        parts.append("=== ACTIVE EVENTS ===")
        if self._active_events:
            for eid, evt in self._active_events.items():
                unit = evt.get("assigned_unit")
                held = evt.get("held_for_unit")
                eta = evt.get("unit_eta")
                if unit:
                    eta_str = f"ETA {eta} min" if eta else "ON SCENE"
                    status_str = f"{unit} {eta_str}"
                elif held:
                    status_str = f"HELD → {held}"
                else:
                    status_str = "UNASSIGNED"
                sev = evt.get("severity", "?")
                parts.append(
                    f"{eid} | {evt['type']:10s} | {evt['node_name']:30s} | "
                    f"sev {sev} | {status_str} | opened step {evt['step_opened']}"
                )
        else:
            parts.append("(none)")
        parts.append("")

        # 3. Unit status
        parts.append("=== UNIT STATUS ===")
        for v in city.vehicles:
            loc = city.nodes[v.current_node].name if v.current_node in city.nodes else v.current_node
            status = v.status
            if v.assigned_event:
                status += f" → {v.assigned_event}"
            parts.append(f"{v.unit_id:15s} | {v.vehicle_type:10s} | {loc:30s} | {status}")
        parts.append("")

        # 4. City reference (compact adjacency)
        parts.append("=== CITY REFERENCE ===")
        for nid, node in city.nodes.items():
            neighbours = []
            for oid, w in city.edges.get(nid, {}).items():
                oname = city.nodes[oid].name
                neighbours.append(f"{oname} [{w:.0f} min]")
            parts.append(f"{node.name} ({node.node_type}) → {', '.join(neighbours)}")
        parts.append("")

        # 5. Dispatcher notes
        parts.append("=== DISPATCHER NOTES ===")
        if self._dispatcher_notes:
            for n in self._dispatcher_notes:
                parts.append(n)
        else:
            parts.append("(first call)")
        parts.append("")

        return "\n".join(parts)

    # ── Helpers ──────────────────────────────────────────────────────────

    def _find_vehicle(self, unit_id: str) -> Optional[Vehicle]:
        if self._city is None:
            return None
        for v in self._city.vehicles:
            if v.unit_id == unit_id:
                return v
        return None

    def _fleet_util(self) -> float:
        if self._city is None or not self._city.vehicles:
            return 0.0
        busy = sum(1 for v in self._city.vehicles if v.status != "FREE")
        return busy / len(self._city.vehicles)

    @property
    def state(self) -> State:
        return self._state
