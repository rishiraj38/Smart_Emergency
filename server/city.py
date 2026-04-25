"""Procedural city graph builder for Dispatch911."""

import heapq
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ── Name pools ───────────────────────────────────────────────────────────────

STREET_NAMES = [
    "Oak", "Maple", "Cedar", "Elm", "Pine", "River", "Lake", "Hill",
    "Park", "Main", "First", "Second", "Third", "Spring", "Sunset",
]
SUFFIXES = ["Street", "Avenue", "Road", "Drive", "Lane", "Boulevard"]
LANDMARKS = {
    "hospital": ["Riverside General Hospital", "St. Mary's Medical Center",
                 "City Central Hospital"],
    "fire_station": ["Engine House No. 1", "Central Fire Station",
                     "Westside Fire Department"],
    "police_station": ["Central Police Station", "Metro Police HQ",
                       "Downtown Precinct"],
    "residential": ["Oakwood Apartments", "Maple Heights", "Pinecrest Homes",
                    "Riverside Condos", "Cedar Park Village", "Elmwood Terrace",
                    "Lakeview Residences", "Hilltop Manor", "Sunset Villas",
                    "Spring Meadow Estates", "Willow Creek Homes",
                    "Birchwood Place", "Magnolia Gardens", "Aspen Ridge"],
    "commercial": ["Downtown Mall", "Oak Avenue Shops", "Riverside Market",
                   "Central Plaza", "Parkside Shopping Center"],
    "road_junction": ["Highway 9 Interchange", "Central Crossroads",
                      "Northside Junction", "Eastgate Roundabout",
                      "Southbound Overpass", "Westway Intersection"],
}


@dataclass
class Node:
    node_id: str
    node_type: str
    name: str
    street: str
    x: float = 0.0
    y: float = 0.0


@dataclass
class Vehicle:
    unit_id: str
    vehicle_type: str  # police / ambulance / fire
    home_node: str
    current_node: str
    status: str = "FREE"          # FREE / DISPATCHED / ON_SCENE / RETURNING
    assigned_event: Optional[str] = None
    eta: int = 0
    on_scene_remaining: int = 0
    return_remaining: int = 0
    path: List[str] = field(default_factory=list)
    transit_progress: float = 0.0  # 0..1 along current path


@dataclass
class City:
    nodes: Dict[str, Node] = field(default_factory=dict)
    edges: Dict[str, Dict[str, float]] = field(default_factory=dict)  # adj list
    vehicles: List[Vehicle] = field(default_factory=list)
    seed: int = 0


def _distance(a: Node, b: Node) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


def _make_street(rng: random.Random) -> str:
    return f"{rng.choice(STREET_NAMES)} {rng.choice(SUFFIXES)}"


def generate_city(seed: int) -> City:
    """Build a random city graph, spawn vehicles, return City."""
    rng = random.Random(seed)
    city = City(seed=seed)

    # ── 1. Create nodes ──────────────────────────────────────────────────
    node_specs: List[Tuple[str, int]] = [
        ("hospital", 1),
        ("fire_station", 1),
        ("police_station", 1),
        ("residential", rng.randint(6, 10)),
        ("commercial", rng.randint(2, 4)),
        ("road_junction", rng.randint(2, 4)),
    ]
    idx = 0
    for ntype, count in node_specs:
        pool = list(LANDMARKS.get(ntype, []))
        rng.shuffle(pool)
        for i in range(count):
            nid = f"{ntype}_{idx}"
            name = pool[i] if i < len(pool) else f"{ntype.title()} {idx}"
            node = Node(
                node_id=nid, node_type=ntype, name=name,
                street=_make_street(rng),
                x=rng.uniform(0, 1), y=rng.uniform(0, 1),
            )
            city.nodes[nid] = node
            city.edges[nid] = {}
            idx += 1

    # ── 2. Build edges (proximity-biased) ────────────────────────────────
    node_ids = list(city.nodes.keys())
    for nid in node_ids:
        n = city.nodes[nid]
        others = sorted(
            [oid for oid in node_ids if oid != nid],
            key=lambda oid: _distance(n, city.nodes[oid]),
        )
        k = rng.randint(2, 4)
        neighbours = others[:k]
        # add 0-1 long-range edges
        for _ in range(rng.randint(0, 1)):
            far = rng.choice(others[k:]) if len(others) > k else None
            if far:
                neighbours.append(far)
        for oid in neighbours:
            if oid not in city.edges[nid]:
                dist = _distance(n, city.nodes[oid])
                travel = max(1.0, dist * 15 + rng.uniform(-1, 2))
                travel = round(travel, 1)
                city.edges[nid][oid] = travel
                city.edges[oid][nid] = travel

    # ── 3. Ensure connectivity ───────────────────────────────────────────
    visited = set()
    stack = [node_ids[0]]
    while stack:
        cur = stack.pop()
        if cur in visited:
            continue
        visited.add(cur)
        stack.extend(city.edges[cur].keys())
    if len(visited) < len(node_ids):
        unvisited = [n for n in node_ids if n not in visited]
        for uid in unvisited:
            closest = min(visited, key=lambda v: _distance(city.nodes[uid], city.nodes[v]))
            d = round(max(1.0, _distance(city.nodes[uid], city.nodes[closest]) * 15), 1)
            city.edges[uid][closest] = d
            city.edges[closest][uid] = d
            visited.add(uid)

    # ── 4. Spawn vehicles ────────────────────────────────────────────────
    def _find_node(ntype: str) -> str:
        for nid, n in city.nodes.items():
            if n.node_type == ntype:
                return nid
        return node_ids[0]

    vid = 0
    for vtype, home_type, count in [
        ("police", "police_station", rng.randint(2, 3)),
        ("ambulance", "hospital", rng.randint(2, 3)),
        ("fire", "fire_station", rng.randint(2, 3)),
    ]:
        home = _find_node(home_type)
        for _ in range(count):
            city.vehicles.append(Vehicle(
                unit_id=f"{vtype}_{vid}",
                vehicle_type=vtype,
                home_node=home,
                current_node=home,
            ))
            vid += 1

    return city


def dijkstra(city: City, src: str, dst: str) -> Tuple[float, List[str]]:
    """Shortest path (travel time) between two nodes. Returns (time, path)."""
    dist: Dict[str, float] = {src: 0.0}
    prev: Dict[str, Optional[str]] = {src: None}
    heap = [(0.0, src)]
    while heap:
        d, u = heapq.heappop(heap)
        if u == dst:
            break
        if d > dist.get(u, float("inf")):
            continue
        for v, w in city.edges.get(u, {}).items():
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, v))
    if dst not in dist:
        return float("inf"), []
    path = []
    cur: Optional[str] = dst
    while cur is not None:
        path.append(cur)
        cur = prev.get(cur)
    path.reverse()
    return dist[dst], path
