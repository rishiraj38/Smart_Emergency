# client.py — FIXED
import requests
from typing import Optional

class EmergencyDispatchClient:
    def __init__(self, base_url: str = "http://localhost:7860"):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def health(self): 
        return self.session.get(f"{self.base_url}/health").json()

    def reset(self, task_id: int = 1, seed: Optional[int] = None):
        return self.session.post(f"{self.base_url}/reset",
            json={"task_id": task_id, "seed": seed}).json()

    def step(self, command: str, event_id=None, vehicle_id=None, 
             severity=None, into_event_id=None, call_id=None):
        return self.session.post(f"{self.base_url}/step", json={
            "command": command, "event_id": event_id,
            "vehicle_id": vehicle_id, "severity": severity,
            "into_event_id": into_event_id, "call_id": call_id
        }).json()

    def state(self): 
        return self.session.get(f"{self.base_url}/state").json()

    def grader(self): 
        return self.session.post(f"{self.base_url}/grader").json()

    def baseline(self): 
        return self.session.get(f"{self.base_url}/baseline").json()

    def __enter__(self): return self
    def __exit__(self, *args): self.session.close()