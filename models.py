# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Dispatch911 Environment.

Action: the agent's structured dispatch decision per incoming 911 call.
Observation: the text-based observation the agent receives each step.
"""

from typing import Dict, List, Literal, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


# ── Reroute sub-action ──────────────────────────────────────────────────────

class RerouteAction(Action):
    """Optional reroute block inside a dispatch action."""

    vehicle_to_reroute: str = Field(..., description="Unit ID of vehicle to redirect")
    from_event_id: str = Field(..., description="EVT-NNNN the vehicle is pulled from")
    replacement_vehicle_id: Optional[str] = Field(
        None, description="Free unit to cover the abandoned event"
    )


# ── Agent action ─────────────────────────────────────────────────────────────

class SmartEmergencyAction(Action):
    """
    The agent's response to an incoming 911 call.

    Three modes:
      - action_type='dispatch': handle a new emergency
      - action_type='duplicate': flag as repeat of an existing event
      - action_type='hold': queue event for a busy vehicle to handle after it frees
    """

    action_type: Literal["dispatch", "duplicate", "hold"] = Field(
        ..., description="'dispatch', 'duplicate', or 'hold'"
    )
    severity_pred: int = Field(
        ..., ge=1, le=5, description="Predicted severity 1-5"
    )
    is_duplicate: bool = Field(
        False, description="Whether the agent believes this is a repeat call"
    )
    duplicate_of_event_id: Optional[str] = Field(
        None, description="EVT-NNNN of the event this duplicates (required if is_duplicate)"
    )
    vehicle_type: Optional[str] = Field(
        None, description="'police', 'ambulance', or 'fire' (required if dispatch or hold)"
    )
    vehicle_id: Optional[str] = Field(
        None, description="Unit to dispatch now (dispatch) or busy unit to queue for (hold)"
    )
    reroute: Optional[RerouteAction] = Field(
        None, description="Optional reroute instruction"
    )


# ── Observation ──────────────────────────────────────────────────────────────

class SmartEmergencyObservation(Observation):
    """
    Observation returned to the agent each step.

    Contains the full text prompt (transcript + active events + unit status +
    city reference + dispatcher notes) and structured metadata for logging.
    """

    prompt: str = Field(default="", description="Full text observation for the LLM")
    step: int = Field(default=0, description="Current step number")
    call_id: str = Field(default="", description="ID of the incoming call")
    reward_breakdown: Dict[str, float] = Field(
        default_factory=dict, description="Per-component reward breakdown"
    )
    active_event_ids: List[str] = Field(
        default_factory=list, description="Currently active event IDs"
    )
    fleet_utilisation: float = Field(
        default=0.0, description="Fraction of fleet currently busy"
    )
    ground_truth: Dict = Field(
        default_factory=dict,
        description="Hidden ground truth for the current call (populated after step)",
    )
