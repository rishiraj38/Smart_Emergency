# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Dispatch911 Environment Client."""

from typing import Dict, Optional

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import SmartEmergencyAction, SmartEmergencyObservation, RerouteAction


class SmartEmergencyEnv(
    EnvClient[SmartEmergencyAction, SmartEmergencyObservation, State]
):
    """
    Client for the Dispatch911 Environment.

    Example:
        >>> with SmartEmergencyEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.prompt)
        ...
        ...     action = SmartEmergencyAction(
        ...         action_type="dispatch",
        ...         severity_pred=3,
        ...         is_duplicate=False,
        ...         vehicle_type="ambulance",
        ...         vehicle_id="ambulance_0",
        ...     )
        ...     result = client.step(action)
        ...     print(result.observation.reward_breakdown)
    """

    def _step_payload(self, action: SmartEmergencyAction) -> Dict:
        """Convert SmartEmergencyAction to JSON payload."""
        payload: Dict = {
            "action_type": action.action_type,
            "severity_pred": action.severity_pred,
            "is_duplicate": action.is_duplicate,
        }
        if action.duplicate_of_event_id is not None:
            payload["duplicate_of_event_id"] = action.duplicate_of_event_id
        if action.vehicle_type is not None:
            payload["vehicle_type"] = action.vehicle_type
        if action.vehicle_id is not None:
            payload["vehicle_id"] = action.vehicle_id
        if action.reroute is not None:
            payload["reroute"] = {
                "vehicle_to_reroute": action.reroute.vehicle_to_reroute,
                "from_event_id": action.reroute.from_event_id,
                "replacement_vehicle_id": action.reroute.replacement_vehicle_id,
            }
        return payload

    def _parse_result(self, payload: Dict) -> StepResult[SmartEmergencyObservation]:
        """Parse server response into StepResult.

        Note: OpenEnv's serialize_observation() intentionally strips 'metadata',
        'done', and 'reward' from the nested observation dict and promotes them
        to the top level. ground_truth is now a first-class field on the
        observation model so it survives serialization.
        """
        obs_data = payload.get("observation", {})
        # metadata is stripped by the framework; ground_truth is now a dedicated field
        metadata = payload.get("metadata", obs_data.get("metadata", {}))
        # Support both the new dedicated ground_truth field and the legacy metadata path
        gt = obs_data.get("ground_truth") or metadata.get("ground_truth", {})
        if gt:
            metadata = dict(metadata)
            metadata["ground_truth"] = gt
        observation = SmartEmergencyObservation(
            prompt=obs_data.get("prompt", ""),
            step=obs_data.get("step", 0),
            call_id=obs_data.get("call_id", ""),
            reward_breakdown=obs_data.get("reward_breakdown", {}),
            active_event_ids=obs_data.get("active_event_ids", []),
            fleet_utilisation=obs_data.get("fleet_utilisation", 0.0),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            ground_truth=gt or {},
            metadata=metadata,
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """Parse server response into State."""
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
