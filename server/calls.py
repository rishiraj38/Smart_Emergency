"""911 call transcript generator for Dispatch911."""

import random
from dataclasses import dataclass
from typing import List, Optional

from .city import City

# ── Call templates ────────────────────────────────────────────────────────────

TEMPLATES = [
    # ── FIRE ──────────────────────────────────────────────────────────────
    {"type": "fire", "sev": 1, "vehicle": "fire",
     "text": "Hi, I think I see some smoke coming from behind {landmark}. It might be nothing but thought I should call."},
    {"type": "fire", "sev": 2, "vehicle": "fire",
     "text": "Yeah there's a small fire in a dumpster near {landmark} on {street}. It's not spreading but it's pretty smoky."},
    {"type": "fire", "sev": 3, "vehicle": "fire",
     "text": "There's a fire at {address}! Flames coming out a window on the second floor. I don't think anyone's inside but I'm not sure."},
    {"type": "fire", "sev": 4, "vehicle": "fire",
     "text": "Oh god, the whole kitchen is on fire at {address}! My kids are upstairs — please send someone NOW!"},
    {"type": "fire", "sev": 4, "vehicle": "fire",
     "text": "Building's on fire on {street} near {landmark}! People are yelling from the windows, please hurry!"},
    {"type": "fire", "sev": 5, "vehicle": "fire",
     "text": "There's a massive fire — the whole block near {landmark} is burning. Multiple buildings involved, I can see people trapped. Send everything you've got!"},
    # ── MEDICAL ───────────────────────────────────────────────────────────
    {"type": "medical", "sev": 1, "vehicle": "ambulance",
     "text": "Hello, my neighbor fell and hurt her ankle at {address}. She's conscious and talking but can't walk."},
    {"type": "medical", "sev": 2, "vehicle": "ambulance",
     "text": "Someone fainted at {landmark}. They're breathing okay now but look really pale. We're on {street}."},
    {"type": "medical", "sev": 3, "vehicle": "ambulance",
     "text": "There's a man having chest pains at {address}. He's sweating a lot and says his arm feels numb."},
    {"type": "medical", "sev": 4, "vehicle": "ambulance",
     "text": "My husband just collapsed and he won't wake up! He's breathing weird. We're at {address}, please hurry!"},
    {"type": "medical", "sev": 4, "vehicle": "ambulance",
     "text": "Someone's not breathing at {landmark}! A bystander is doing CPR. Please send an ambulance to {street} immediately!"},
    {"type": "medical", "sev": 5, "vehicle": "ambulance",
     "text": "There's been some kind of mass incident at {landmark} — multiple people down, some not moving. We need everything, {street} entrance."},
    # ── CRIME ─────────────────────────────────────────────────────────────
    {"type": "crime", "sev": 1, "vehicle": "police",
     "text": "I'd like to report a shoplifter at {landmark} on {street}. They already left but I got a good look."},
    {"type": "crime", "sev": 2, "vehicle": "police",
     "text": "There's a break-in happening right now at {address}. I can see someone climbing through a window from across the street."},
    {"type": "crime", "sev": 3, "vehicle": "police",
     "text": "There's a fight outside {landmark} on {street}. Looks like 3-4 people involved, getting pretty violent."},
    {"type": "crime", "sev": 3, "vehicle": "police",
     "text": "I just got mugged near {landmark}! The guy ran towards {cross_street}. He had a knife."},
    {"type": "crime", "sev": 4, "vehicle": "police",
     "text": "I think I heard gunshots near {address}! People are running. I'm hiding inside {landmark}, please send help!"},
    {"type": "crime", "sev": 5, "vehicle": "police",
     "text": "Active shooter at {landmark}! Multiple shots fired, people running everywhere. Send everyone NOW!"},
    # ── ACCIDENT ──────────────────────────────────────────────────────────
    {"type": "accident", "sev": 2, "vehicle": "ambulance",
     "text": "Fender bender on {street} near {landmark}. No injuries but the cars are blocking the road."},
    {"type": "accident", "sev": 3, "vehicle": "ambulance",
     "text": "Car accident at {street} and {cross_street}. One driver looks hurt, holding their neck. Other car's smoking."},
    {"type": "accident", "sev": 3, "vehicle": "ambulance",
     "text": "A cyclist got hit by a car near {landmark}. They're on the ground, conscious but bleeding from the head."},
    {"type": "accident", "sev": 4, "vehicle": "ambulance",
     "text": "Bad crash on {street}! Car flipped over near {landmark}. Driver's trapped inside, not responding!"},
    {"type": "accident", "sev": 4, "vehicle": "ambulance",
     "text": "Pedestrian hit by a truck at {cross_street} near {landmark}. They're not moving. There's blood everywhere."},
    {"type": "accident", "sev": 5, "vehicle": "ambulance",
     "text": "Multi-car pileup on {street} near {landmark}! At least 5 cars, people screaming, I can smell gas leaking. Send fire too!"},
]


@dataclass
class Call:
    """A single incoming 911 call with hidden ground truth."""
    call_id: str
    event_id: str
    origin_node_id: str
    origin_node_name: str
    emergency_type: str
    severity: int
    required_vehicle_type: str
    is_duplicate_of: Optional[str]
    transcript: str


def generate_call(
    city: City,
    call_number: int,
    active_events: dict,
    duplicate_prob: float,
    rng: random.Random,
    next_event_counter: int,
) -> tuple:
    """
    Generate one 911 call.

    Returns (Call, new_event_counter).
    """
    node_ids = list(city.nodes.keys())

    # ── Decide if duplicate ──────────────────────────────────────────────
    is_dup = False
    dup_event_id = None
    dup_event = None
    if active_events and rng.random() < duplicate_prob:
        dup_event_id = rng.choice(list(active_events.keys()))
        dup_event = active_events[dup_event_id]
        is_dup = True

    if is_dup and dup_event is not None:
        etype = dup_event["type"]
        sev = dup_event["severity"]
        vtype = dup_event["vehicle"]
        origin = dup_event["node_id"]
        event_id = dup_event_id
    else:
        # Pick a random template
        tmpl = rng.choice(TEMPLATES)
        etype = tmpl["type"]
        sev = tmpl["sev"] + rng.choice([-1, 0, 0, 0, 1])
        sev = max(1, min(5, sev))
        vtype = tmpl["vehicle"]
        # Pick origin node (prefer residential/commercial)
        preferred = [n for n in node_ids if city.nodes[n].node_type in ("residential", "commercial")]
        origin = rng.choice(preferred) if preferred else rng.choice(node_ids)
        event_id = f"EVT-{next_event_counter:04d}"
        next_event_counter += 1

    # ── Build transcript ─────────────────────────────────────────────────
    node = city.nodes[origin]
    neighbours = list(city.edges.get(origin, {}).keys())
    cross = city.nodes[rng.choice(neighbours)].street if neighbours else "unknown road"

    # Pick a template matching the type
    matching = [t for t in TEMPLATES if t["type"] == etype]
    tmpl = rng.choice(matching)
    address = f"{rng.randint(100, 999)} {node.street}"

    text = tmpl["text"].format(
        landmark=node.name,
        street=node.street,
        address=address,
        cross_street=cross,
    )

    call = Call(
        call_id=f"CALL-{call_number:04d}",
        event_id=event_id,
        origin_node_id=origin,
        origin_node_name=node.name,
        emergency_type=etype,
        severity=sev,
        required_vehicle_type=vtype,
        is_duplicate_of=dup_event_id if is_dup else None,
        transcript=text,
    )
    print(call)
    return call, next_event_counter
