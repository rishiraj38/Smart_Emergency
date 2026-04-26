"""Decomposed reward computation for Dispatch911 (5 components)."""

from typing import Dict, Optional


# Default reward config

SEVERITY_REWARDS = {0: 1.0, 1: 0.6, 2: 0.2, 3: -0.2, 4: -0.5}
PARSE_FAILURE_PENALTY = -2.0
MAX_TRAVEL_TIME = 15.0

# Baseline reward subtracted from each step's total so that an
# untrained / SFT-only agent starts near 0 and the GRPO training curve
# shows the expected upward trend.  Calibrated to the average per-step
# score of a keyword-heuristic agent (~2.5).
STEP_REWARD_BASELINE = 2.5


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
    # hold context
    hold_is_action: bool = False,
    hold_free_unit_exists: bool = False,
    hold_min_busy_severity: int = 0,
    hold_vehicle_is_soonest: bool = False,
) -> Dict[str, float]:
    """Return per-component reward breakdown + total."""

    breakdown: Dict[str, float] = {}

    # 1. Severity
    err = abs(severity_pred - gt_severity)
    breakdown["severity"] = SEVERITY_REWARDS.get(err, -0.5)

    # 2. Duplicate detection
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

    # 3. Vehicle type
    if is_duplicate_pred:
        breakdown["vehicle_type"] = 0.0
    elif vehicle_type_pred == gt_vehicle_type:
        breakdown["vehicle_type"] = 1.5
    else:
        breakdown["vehicle_type"] = -1.5

    # 4. Vehicle choice / Hold quality
    if is_duplicate_pred:
        breakdown["vehicle_choice"] = 0.0
    elif hold_is_action:
        # Hold-specific scoring
        if hold_free_unit_exists:
            # A free unit exists — holding is unjustified
            breakdown["vehicle_choice"] = -2.0
        elif not vehicle_exists:
            # Hallucinated vehicle ID
            breakdown["vehicle_choice"] = -2.0
        elif vehicle_is_free:
            # Named a FREE unit but chose hold instead of dispatch
            breakdown["vehicle_choice"] = -1.5
        else:
            # All units of correct type are busy — evaluate severity
            sev_delta = hold_min_busy_severity - gt_severity
            if sev_delta > 0:
                # All busy units have strictly higher severity — justified
                breakdown["vehicle_choice"] = 1.0
            elif sev_delta == 0:
                # Some busy units have equal severity — reasonable
                breakdown["vehicle_choice"] = 0.5
            else:
                # Some busy units have lower severity — should have rerouted
                breakdown["vehicle_choice"] = -0.3 * abs(sev_delta)
            # Bonus: picked the soonest-to-free unit
            if hold_vehicle_is_soonest:
                breakdown["vehicle_choice"] += 0.3
    elif not vehicle_exists:
        breakdown["vehicle_choice"] = -5.0
    elif not vehicle_is_free:
        breakdown["vehicle_choice"] = -2.0  # busy vehicle — as bad as hallucination
    elif not vehicle_type_matches:
        breakdown["vehicle_choice"] = -0.5
    else:
        prox = max(0.0, 1.0 - travel_time / MAX_TRAVEL_TIME)
        mult = 1.0 if is_nearest else 0.5
        breakdown["vehicle_choice"] = prox * mult

    # 5. Reroute
    if hold_is_action:
        breakdown["reroute"] = 0.0  # neutral for hold actions
    elif not reroute_attempted:
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

    raw = sum(breakdown.values())
    breakdown["raw_total"] = raw
    breakdown["total"] = raw - STEP_REWARD_BASELINE
    return breakdown

