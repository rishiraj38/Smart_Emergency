# config.py — Balanced rewards + task-based curriculum

REWARD = {
    # Waiting penalties (per event per step) — scaled down to not overwhelm
    "CRITICAL_WAIT":        -2.0,
    "SEMI_CRITICAL_WAIT":   -1.0,
    "NORMAL_WAIT":          -0.2,

    # Triage action rewards
    "CORRECT_CLASSIFY":     +3.0,
    "MISCLASSIFY_DOWN":     -5.0,   # underestimate severity (dangerous)
    "MISCLASSIFY_UP":       -1.0,   # overestimate (safety-first, small penalty)
    "CORRECT_MERGE":        +4.0,
    "FALSE_MERGE":          -8.0,

    # Dispatch action rewards
    "CORRECT_DISPATCH":     +2.0,
    "WRONG_VEHICLE":        -3.0,
    "DOUBLE_DISPATCH":      -5.0,
    "DISTANCE_PER_HOP":     -0.1,   # per-hop travel cost

    # Hold rewards
    "HOLD_CRITICAL":        -3.0,
    "HOLD_NORMAL":          -0.2,

    # Resolution rewards (the big carrot)
    "CRITICAL_RESOLVED":    +30.0,
    "SEMI_RESOLVED":        +15.0,
    "NORMAL_RESOLVED":      +5.0,

    # Penalties
    "INVALID_ACTION":       -2.0,
    "CRITICAL_FAILURE":     -20.0,  # critical event timed out
    "DISCARD_REAL":         -5.0,   # discarding a real emergency
    "WAIT_PENALTY":         -0.1,   # doing nothing
    "RECALL_PENALTY":       -1.0,
    "REROUTE_PENALTY":      -0.5,
    "ESCALATE_BONUS":       +0.5,
    "TIME_EXCEEDED":        -2.0,
}

# Task-based curriculum — tasks 1/2/3 are actually different
TASKS = {
    1: {
        "name": "Basic Triage",
        "difficulty": "easy",
        "unique_events": 5,
        "calls_per_episode": 15,
        "max_steps": 200,
        "critical_failure_wait": 50,
        "ambulances": 4,
        "fire_trucks": 3,
        "police": 3,
    },
    2: {
        "name": "Resource Management",
        "difficulty": "medium",
        "unique_events": 10,
        "calls_per_episode": 40,
        "max_steps": 500,
        "critical_failure_wait": 30,
        "ambulances": 3,
        "fire_trucks": 2,
        "police": 2,
    },
    3: {
        "name": "Disaster Response",
        "difficulty": "hard",
        "unique_events": 15,
        "calls_per_episode": 60,
        "max_steps": 500,
        "critical_failure_wait": 15,
        "ambulances": 2,
        "fire_trucks": 1,
        "police": 1,
    },
}

CITY = {
    "NUM_NODES": 20,
}
