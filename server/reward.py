"""Decomposed reward computation for Dispatch911 (5 components)."""

from typing import Dict, Optional


# ── Default reward config ────────────────────────────────────────────────────

SEVERITY_REWARDS = {0: 1.0, 1: 0.6, 2: 0.2, 3: -0.2, 4: -0.5}
PARSE_FAILURE_PENALTY = -2.0
MAX_TRAVEL_TIME = 15.0


def compute_reward(
    *,
    # ground truth
    gt_severity: int,
    gt_is_duplicate: bool,
    gt_event_id: Optional[str],
    gt_vehicle_type: str,
    gt_origin_node: str,
    # agent predictions
    severity_pred: int,
    is_duplicate_pred: bool,
    duplicate_of_event_id: Optional[str],
    vehicle_type_pred: Optional[str],
    vehicle_id_pred: Optional[str],
    # vehicle context
    vehicle_exists: bool = True,
    vehicle_is_free: bool = True,
    vehicle_type_matches: bool = True,
    travel_time: float = 0.0,
    is_nearest: bool = False,
    # reroute context
    reroute_attempted: bool = False,
    reroute_valid: bool = False,
    reroute_severity_delta: int = 0,
    reroute_faster: bool = False,
    replacement_valid: Optional[bool] = None,
) -> Dict[str, float]:
    """Return per-component reward breakdown + total."""

    breakdown: Dict[str, float] = {}

    # ── 1. Severity ──────────────────────────────────────────────────────
    err = abs(severity_pred - gt_severity)
    breakdown["severity"] = SEVERITY_REWARDS.get(err, -0.5)

    # ── 2. Duplicate detection ───────────────────────────────────────────
    if not is_duplicate_pred and not gt_is_duplicate:
        breakdown["duplicate"] = 1.0
    elif not is_duplicate_pred and gt_is_duplicate:
        breakdown["duplicate"] = -1.0
    elif is_duplicate_pred and not gt_is_duplicate:
        breakdown["duplicate"] = -0.8
    elif is_duplicate_pred and gt_is_duplicate:
        if duplicate_of_event_id is None:
            breakdown["duplicate"] = 0.0
        elif duplicate_of_event_id == gt_event_id:
            breakdown["duplicate"] = 1.5
        else:
            breakdown["duplicate"] = 0.3

    # ── 3. Vehicle type ──────────────────────────────────────────────────
    if is_duplicate_pred:
        breakdown["vehicle_type"] = 0.0
    elif vehicle_type_pred == gt_vehicle_type:
        breakdown["vehicle_type"] = 1.5
    else:
        breakdown["vehicle_type"] = -1.5

    # ── 4. Vehicle choice ────────────────────────────────────────────────
    if is_duplicate_pred:
        breakdown["vehicle_choice"] = 0.0
    elif not vehicle_exists:
        breakdown["vehicle_choice"] = -2.0
    elif not vehicle_is_free:
        breakdown["vehicle_choice"] = -1.0
    elif not vehicle_type_matches:
        breakdown["vehicle_choice"] = -0.5
    else:
        prox = max(0.0, 1.0 - travel_time / MAX_TRAVEL_TIME)
        mult = 1.0 if is_nearest else 0.5
        breakdown["vehicle_choice"] = prox * mult

    # ── 5. Reroute ───────────────────────────────────────────────────────
    if not reroute_attempted:
        breakdown["reroute"] = 0.0
    elif not reroute_valid:
        breakdown["reroute"] = -1.0
    else:
        r = 0.0
        if reroute_severity_delta <= 0:
            r = -0.5
        elif reroute_severity_delta == 1:
            r = 0.3
        else:
            r = 0.8
        if reroute_faster:
            r += 0.4
        if replacement_valid is True:
            r += 0.5
        elif replacement_valid is False:
            r -= 0.3
        breakdown["reroute"] = r

    breakdown["total"] = sum(breakdown.values())
    return breakdown
