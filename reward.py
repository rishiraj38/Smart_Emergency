from typing import Dict, List
from config import REWARD

class RewardCalculator:
    """
    Multi-component reward calculator for the Emergency Dispatch Environment.
    Follows OpenEnv hackathon best practices for multi-signal verification.
    """
    def __init__(self):
        self.waiting_events = {} # {event_id: severity}
        # Discrete events in the current step
        self.step_signals = {
            "triage": 0.0,
            "dispatch": 0.0,
            "waiting": 0.0,
            "efficiency": 0.0,
            "penalties": 0.0
        }
        
    def reset(self):
        self.waiting_events.clear()
        self._clear_signals()
        
    def _clear_signals(self):
        for key in self.step_signals:
            self.step_signals[key] = 0.0

    def add_waiting_event(self, event_id: str, severity: str):
        self.waiting_events[event_id] = severity
        
    def remove_waiting_event(self, event_id: str):
        if event_id in self.waiting_events:
            del self.waiting_events[event_id]
            
    def record_signal(self, component: str, value: float):
        """Record a reward signal for a specific component."""
        if component in self.step_signals:
            self.step_signals[component] += value

    def calculate_step_reward(self) -> Dict[str, float]:
        """
        Calculate total reward and return breakdown.
        """
        # Calculate waiting penalties
        waiting_penalty = 0.0
        for event_id, severity in self.waiting_events.items():
            if severity == "CRITICAL":
                waiting_penalty += REWARD["CRITICAL_WAIT"]
            elif severity == "SEMI_CRITICAL":
                waiting_penalty += REWARD["SEMI_CRITICAL_WAIT"]
            elif severity == "NORMAL":
                waiting_penalty += REWARD["NORMAL_WAIT"]
        
        self.step_signals["waiting"] = waiting_penalty
        
        # Calculate total
        total = sum(self.step_signals.values())
        
        # Normalized or raw? Hackathon guide suggests verifiable rewards.
        # We'll return the breakdown and the total.
        result = {
            "total": total,
            **self.step_signals
        }
        
        # Reset for next step
        self._clear_signals()
        
        return result
