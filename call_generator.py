import random
import string

LOCATION_ALIASES = {
    "NODE_INT1": ["1st and broadway", "the north intersection", "near the big billboard"],
    "NODE_INT2": ["2nd street crossing", "outside the mall", "by the metro station"],
    "NODE_INT3": ["3rd and oak", "the busy corner", "near the coffee shop"],
    "NODE_INT4": ["4th ave", "the south square", "by the fountain"],
    "NODE_INT5": [
        "5th and main",
        "near the downtown crossing",
        "outside the old bank building",
        "main street intersection",
        "by the traffic light on 5th"
    ],
    "NODE_INT6": ["6th and pine", "the west intersection", "near the post office"],
    "NODE_INT7": ["7th ave block", "near the library", "by the statue"],
    "NODE_INT8": ["8th and maple", "the east intersection", "near the cinema"],
    "NODE_INT9": ["9th street", "near the park entrance", "by the old church"],
    "NODE_INT10": ["10th and cedar", "the industrial crossing", "near the warehouse"],
    
    "NODE_H1": [
        "near the hospital",
        "outside central hospital",
        "h1 district",
        "by the ER entrance",
        "hospital road"
    ],
    "NODE_H2": ["westside clinic", "near the medical center", "h2 area", "second hospital"],
    
    "NODE_RES1": ["residential block 1", "near the suburbs", "the east neighborhood"],
    "NODE_RES2": ["residential block 2", "the north neighborhood", "near the apartments"],
    "NODE_RES3": [
        "residential block 3",
        "the quiet neighborhood",
        "near elm street",
        "behind the school",
        "res area downtown"
    ],
    "NODE_RES4": ["residential block 4", "the south neighborhood", "near the villas"],
    "NODE_RES5": ["residential block 5", "the west neighborhood", "near the condos"],
    
    "STATION_1": ["fire station 1", "central station", "near the main depot"],
    "STATION_2": ["police station 2", "west station", "near the precinct"],
    "STATION_3": ["ambulance bay 3", "east station", "near the garage"]
}

SEVERITY_NOISE = {
    "CRITICAL": [
        "someone is dying",
        "its really bad",
        "people are screaming",
        "there is blood everywhere",
        "minor scratch"          # conflicting caller
    ],
    "SEMI_CRITICAL": [
        "they seem hurt but conscious",
        "not life threatening i think",
        "pretty serious",
        "maybe needs a doctor"
    ],
    "NORMAL": [
        "no injuries",
        "just a fender bender",
        "everything seems fine",
        "not sure if this is an emergency"
    ]
}

def add_noise(text: str, rng: random.Random = None) -> str:
    """Add various types of noise to simulate real 911 calls."""
    if rng is None:
        rng = random.Random()
    noisy_text = text
    
    # Randomly insert filler
    fillers = ["um", "uh", "i dont know", "oh god", "listen"]
    if rng.random() < 0.3:
        words = noisy_text.split()
        if words:
            insert_idx = rng.randint(0, len(words))
            words.insert(insert_idx, rng.choice(fillers))
            noisy_text = " ".join(words)
            
    # Random single char swap to simulate typos
    if rng.random() < 0.2 and len(noisy_text) > 5:
        idx = rng.randint(0, len(noisy_text) - 1)
        random_char = rng.choice(string.ascii_lowercase)
        noisy_text = noisy_text[:idx] + random_char + noisy_text[idx+1:]
        
    # Random truncation to simulate caller hanging up
    if rng.random() < 0.1:
        trunc_len = max(5, int(len(noisy_text) * 0.7))
        noisy_text = noisy_text[:trunc_len] + "..."
        
    return noisy_text

def generate_911_call(
    event_id: str,
    call_number: int,
    event_type: str,
    ground_truth_node: str,
    ground_truth_severity: str,
    rng: random.Random = None
) -> dict:
    if rng is None:
        rng = random.Random()
    
    # 30% of time drop location detail if we want (or partially drop)
    if rng.random() < 0.3:
        location_str = rng.choice(LOCATION_ALIASES[ground_truth_node]).split()[0] # just first word
    else:
        location_str = rng.choice(LOCATION_ALIASES[ground_truth_node])
        
    severity_str = rng.choice(SEVERITY_NOISE[ground_truth_severity])
    
    raw_text = f"{event_type}. {severity_str} at {location_str}"
    
    return {
        "call_id": f"CALL-{event_id}-{call_number}",
        "transcript": add_noise(raw_text, rng=rng),
        "ground_truth_event": event_id,     # hidden from agent
        "ground_truth_type": event_type,    # hidden
        "ground_truth_severity": ground_truth_severity,  # hidden
        "ground_truth_node": ground_truth_node           # hidden
    }

class CallGenerator:
    def __init__(self, seed: int = None):
        self.rng = random.Random(seed)  # instance-only, doesn't pollute global
            
        self.event_types = ["Building Fire", "Car Accident", "Medical Emergency", "Gas Leak", "Violent Incident", "Noise Complaint"]
        
    def generate_episode_calls(self, nodes: list, num_events: int = 10, num_calls: int = 40) -> list:
        """Generate episode calls with configurable event/call counts for curriculum."""
        events = {}
        for i in range(num_events):
            event_id = f"E{i+1:03d}"
            severity = self.rng.choice(["CRITICAL", "SEMI_CRITICAL", "NORMAL"])
            node = self.rng.choice(nodes)
            etype = self.rng.choice(self.event_types)
            events[event_id] = {
                "severity": severity,
                "node": node,
                "type": etype
            }
            
        calls = []
        event_ids = list(events.keys())
        for i in range(num_calls):
            e_id = self.rng.choice(event_ids)
            event = events[e_id]
            call = generate_911_call(
                event_id=e_id,
                call_number=i+1,
                event_type=event["type"],
                ground_truth_node=event["node"],
                ground_truth_severity=event["severity"],
                rng=self.rng
            )
            calls.append(call)
            
        return calls
