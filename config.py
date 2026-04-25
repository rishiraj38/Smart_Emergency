# config.py

REWARD = {
    # Waiting penalties per step
    "CRITICAL_WAIT":        -10,
    "SEMI_CRITICAL_WAIT":   -5,
    "NORMAL_WAIT":          -1,

    # Triage action rewards
    "CORRECT_CLASSIFY":     +5,
    "MISCLASSIFY_DOWN":     -20,  # critical -> normal
    "MISCLASSIFY_UP":       -3,   # normal -> critical
    "CORRECT_MERGE":        +8,
    "FALSE_MERGE":          -25,

    # Dispatch action rewards
    "CORRECT_DISPATCH":     +10,
    "WRONG_VEHICLE":        -15,
    "DOUBLE_DISPATCH":      -20,
    "HOLD_CRITICAL":        -15,
    "HOLD_NORMAL":          +2,

    # Resolution rewards
    "CRITICAL_RESOLVED":    +50,
    "SEMI_RESOLVED":        +20,
    "NORMAL_RESOLVED":      +5,

    # Parser penalty
    "INVALID_ACTION":       -5
}

EPISODE = {
    "MAX_STEPS":              500,
    "CRITICAL_FAILURE_WAIT":  30,
    "CALLS_PER_EPISODE":      40,
    "UNIQUE_EVENTS":          10,
    "DUPLICATION_RATE":       0.5
}

CITY = {
    "NUM_NODES":    20,
    "AMBULANCES":   3,
    "FIRE_TRUCKS":  2,
    "POLICE":       2
}
