from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator

# ─────────────────────────────────────────────────────────────────────────────
# ACTION — what the agent sends on every step
# ─────────────────────────────────────────────────────────────────────────────

VALID_ACTIONS = {
    "CLASSIFY", "MERGE", "ESCALATE", "DISCARD", 
    "DISPATCH", "REROUTE", "HOLD", "RECALL", "WAIT"
}
VALID_SEVERITIES = {"NORMAL", "SEMI_CRITICAL", "CRITICAL"}

class EmergencyAction(BaseModel):
    """
    The action an agent takes to manage emergency dispatch.
    """
    command: str = Field(
        ..., 
        description="The command to execute (CLASSIFY, DISPATCH, etc.)",
        examples=["CLASSIFY", "DISPATCH", "WAIT"]
    )
    event_id: Optional[str] = Field(None, description="Target event ID")
    vehicle_id: Optional[str] = Field(None, description="Target vehicle ID")
    severity: Optional[str] = Field(None, description="Severity level for CLASSIFY or ESCALATE")
    into_event_id: Optional[str] = Field(None, description="Target event for MERGE")
    call_id: Optional[str] = Field(None, description="Specific call ID for DISCARD")
    
    @field_validator("command")
    @classmethod
    def validate_command(cls, v: str) -> str:
        v = v.upper().strip()
        if v not in VALID_ACTIONS:
            raise ValueError(f"command must be one of {VALID_ACTIONS}")
        return v

    @field_validator("severity")
    @classmethod
    def validate_severity(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        v = v.upper().strip()
        if v not in VALID_SEVERITIES:
            raise ValueError(f"severity must be one of {VALID_SEVERITIES}")
        return v

# ─────────────────────────────────────────────────────────────────────────────
# OBSERVATION — what the agent receives each step
# ─────────────────────────────────────────────────────────────────────────────

class EmergencyObservation(BaseModel):
    """
    Everything the agent can see after a reset() or step() call.
    """
    # Incoming call content
    current_call_id: Optional[str] = Field(None, description="ID of the incoming call")
    transcript: Optional[str] = Field(None, description="Transcript of the 911 call")
    
    # World state
    active_events: List[Dict[str, Any]] = Field(default_factory=list, description="List of active events")
    resources: List[Dict[str, Any]] = Field(default_factory=list, description="Status of all emergency vehicles")
    
    # Episode progress
    step: int = Field(..., description="Current step number")
    calls_remaining: int = Field(..., description="Number of calls left in queue")
    cases_resolved: int = Field(..., description="Number of successfully resolved events")
    
    # Reward signal
    reward: float = Field(0.0, description="Reward for the last action")
    cumulative_reward: float = Field(0.0, description="Total reward so far")
    
    # Terminal flag
    done: bool = Field(False, description="True when the episode ends")

# ─────────────────────────────────────────────────────────────────────────────
# REWARD — detailed breakdown
# ─────────────────────────────────────────────────────────────────────────────

class EmergencyReward(BaseModel):
    value: float = Field(..., description="Total reward for this step")
    feedback: str = Field("", description="Explanation of the reward/penalty")
    penalties: List[str] = Field(default_factory=list, description="List of penalties applied")

# ─────────────────────────────────────────────────────────────────────────────
# STATE — episode-level metadata
# ─────────────────────────────────────────────────────────────────────────────

class EmergencyState(BaseModel):
    episode_id: str = Field(..., description="Unique ID for this episode")
    task_id: int = Field(..., description="Current task ID")
    step_count: int = Field(..., description="Total steps taken")
    cumulative_reward: float = Field(..., description="Total reward")
    done: bool = Field(..., description="Is the episode finished?")

# ─────────────────────────────────────────────────────────────────────────────
# STEP RESULT — full response from step()
# ─────────────────────────────────────────────────────────────────────────────

class StepResult(BaseModel):
    observation: EmergencyObservation
    reward: float
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)

# ─────────────────────────────────────────────────────────────────────────────
# RESET REQUEST
# ─────────────────────────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: int = Field(default=1, ge=1, le=3, description="Task difficulty level")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")