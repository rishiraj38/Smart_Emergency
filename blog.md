# 🚨 Smart Emergency Dispatch — Teaching AI to Save Lives with Reinforcement Learning

*Building an RL environment and training an LLM agent that acts as an expert 911 dispatcher — triaging emergencies, dispatching vehicles, and managing scarce resources across a simulated city.*

---

## Table of Contents

1. [The Problem Statement](#the-problem-statement)
2. [Why This Problem Matters](#why-this-problem-matters)
3. [Why We Chose This Problem](#why-we-chose-this-problem)
4. [Our Approach — High Level](#our-approach--high-level)
5. [The Environment — Dispatch911](#the-environment--dispatch911)
6. [Reward Engineering](#reward-engineering)
7. [Curriculum Learning — Task Difficulty](#curriculum-learning--task-difficulty)
8. [The Agent — SFT + GRPO Training Pipeline](#the-agent--sft--grpo-training-pipeline)
9. [Technical Stack](#technical-stack)
10. [Problems We Faced & How We Solved Them](#problems-we-faced--how-we-solved-them)
11. [Results](#results)
12. [Conclusion](#conclusion)

---

## The Problem Statement

> **How can we build an AI system that makes real-time emergency dispatch decisions — triaging incoming 911 calls, classifying their severity, detecting duplicate reports, and dispatching the optimal emergency vehicle — all under the constraint of limited resources?**

Every day, 911 dispatch centers across the world handle thousands of calls. Human dispatchers must make split-second decisions:

- **"Is this a fire or a medical emergency?"**
- **"How severe is it — should I send one unit or five?"**
- **"Is this the same apartment fire we got a call about 3 minutes ago?"**
- **"All ambulances are busy — should I reroute one from a lower-priority call?"**

These decisions are made under extreme time pressure, cognitive overload, and emotional stress. A wrong triage — sending a police car to a heart attack, or ignoring a duplicate call and double-dispatching scarce resources — can cost lives.

**Our goal**: Build a reinforcement learning environment that simulates this exact problem, and train a Large Language Model (LLM) agent that learns to be an expert dispatcher through trial and error.

---

## Why This Problem Matters

### The Human Cost of Dispatch Errors

Emergency dispatch is one of the most consequential decision-making tasks in public safety:

- **Response time is everything.** For cardiac arrest, every minute without intervention reduces survival by 7-10%. Dispatching the nearest ambulance instead of a farther one can be the difference between life and death.
- **Resource scarcity is real.** During a multi-car pileup, all ambulances may be busy. The dispatcher must decide: reroute one from a minor injury call? Put the critical patient on hold? These are impossible decisions.
- **Cognitive overload.** During mass incidents (active shooter, natural disaster), dispatchers handle 10x normal call volume while multiple events compete for the same limited vehicles.
- **Duplicate calls waste resources.** When a building catches fire, dozens of people call 911 reporting the same fire. Each duplicate call that triggers a new dispatch wastes a vehicle that could be going somewhere else.

### Why AI Can Help

An AI dispatcher doesn't get tired, doesn't get emotionally overwhelmed, and can process the entire city's vehicle status, travel times, and event history simultaneously. It can:

- **Triage consistently** — no severity under-estimation from caller fatigue
- **Detect duplicates instantly** — pattern-match across all active events
- **Optimize routing** — compute shortest paths across the city graph in milliseconds
- **Manage scarcity rationally** — reroute vehicles based on severity comparison, not gut feeling

---

## Why We Chose This Problem

We selected emergency dispatch for several key reasons:

1. **Real-world impact.** Unlike toy RL problems (CartPole, Atari), this directly models a life-saving task. The skills an agent learns here — triage, resource allocation, duplicate detection — are transferable to real dispatch assistance systems.

2. **Rich decision space.** The agent must simultaneously handle:
   - **Classification** (severity 1-5, emergency type)
   - **Detection** (is this a duplicate?)
   - **Optimization** (which vehicle, from where?)
   - **Strategic planning** (hold vs. reroute vs. dispatch)
   
   This makes it far more challenging than single-objective RL tasks.

3. **Natural fit for LLMs.** The input is natural language (911 call transcripts), and the output is structured JSON (dispatch actions). This is exactly what modern LLMs excel at — understanding unstructured text and producing structured decisions.

4. **Curriculum-friendly.** The problem naturally decomposes into difficulty levels:
   - Easy: plenty of vehicles, just dispatch correctly
   - Medium: some scarcity, must detect duplicates
   - Hard: extreme scarcity, must reroute and triage

5. **OpenEnv compatibility.** We wanted to build a standard RL environment that others can train against, benchmark on, and improve. The OpenEnv framework (by Meta) gives us HTTP/WebSocket APIs that work with any training framework.

---

## Our Approach — High Level

We built a complete end-to-end system with two major components:

```
┌─────────────────────────────────────────────────────────────┐
│                    ENVIRONMENT (Server)                       │
│                                                               │
│  ┌──────────┐    ┌──────────┐    ┌───────────┐               │
│  │ City     │    │ Call     │    │ Reward    │               │
│  │ Generator│───▶│Generator │───▶│ Computer  │               │
│  │ (Graphs) │    │(25 tmpl) │    │(5 comp)   │               │
│  └──────────┘    └──────────┘    └───────────┘               │
│       ▲                                │                     │
│       │          ┌──────────┐          ▼                     │
│       └──────────│Vehicle   │    ┌───────────┐               │
│                  │Lifecycle │◀───│ Step      │               │
│                  │Manager   │    │ Evaluator │               │
│                  └──────────┘    └───────────┘               │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP / WebSocket
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                      AGENT (Training)                        │
│                                                               │
│  ┌──────────────┐    ┌──────────────┐                        │
│  │ Phase 1: SFT │───▶│ Phase 2: GRPO│                        │
│  │ (Format)     │    │ (Strategy)   │                        │
│  └──────────────┘    └──────────────┘                        │
│         │                    │                                │
│         ▼                    ▼                                │
│  Qwen3-1.7B learns    Qwen3-1.7B learns                     │
│  JSON output format   optimal dispatch                       │
│  from demonstrations  from env rewards                       │
└─────────────────────────────────────────────────────────────┘
```

---

## The Environment — Dispatch911

### Procedural City Generation

Every episode begins with a **procedurally generated city** — a weighted graph of 8-12 nodes representing real urban locations:

- **Hospitals** (ambulance home base)
- **Fire Stations** (fire truck home base)
- **Police Stations** (patrol car home base)
- **Residential areas** (apartments, homes — where emergencies happen)
- **Commercial zones** (malls, shops — high foot traffic)
- **Road junctions** (interchanges, intersections)

Edges between nodes have **travel times** (in minutes) computed from Euclidean distance with random noise, simulating real road networks. We use **Dijkstra's algorithm** to compute shortest paths for vehicle dispatch.

```python
# Example: 9-node city with travel times
Riverside General Hospital (hospital) → Oakwood Apartments [3 min], Central Plaza [5 min]
Central Fire Station (fire_station) → Downtown Mall [2 min], Hilltop Manor [4 min]
Central Police Station (police_station) → Maple Heights [3 min], Highway 9 Interchange [6 min]
```

### 911 Call Generation

Each step, the environment generates an incoming 911 call from **25 handcrafted templates** across 4 emergency types:

| Type | Example Call | Vehicle |
|------|-------------|---------|
| 🔥 **Fire** | *"The whole kitchen is on fire at 437 Oak Street! My kids are upstairs!"* | Fire truck |
| 🏥 **Medical** | *"Someone's not breathing at Riverside Market! A bystander is doing CPR."* | Ambulance |
| 🚔 **Crime** | *"I think I heard gunshots near 812 Elm Drive! People are running."* | Police |
| 🚗 **Accident** | *"Bad crash on Cedar Road! Car flipped over, driver's trapped inside!"* | Ambulance |

Each template includes a **ground-truth severity** (1-5), and the environment adds ±1 random noise to create variation. Calls reference real city landmarks, street names, and cross-streets, making them feel authentic.

### Duplicate Calls

Real 911 centers receive multiple calls about the same incident. Our environment simulates this: with a configurable probability (10-50% depending on difficulty), a new call is generated as a **duplicate** of an existing active event. The call uses the same location and emergency type but different wording — the agent must recognize it's the same incident.

### Vehicle Lifecycle

Vehicles go through a realistic lifecycle:

```
FREE → DISPATCHED → ON_SCENE → RETURNING → FREE
         (travel)    (2 steps)   (2 steps)
```

- **FREE**: Available at home base
- **DISPATCHED**: En route (ETA decrements each step)
- **ON_SCENE**: Handling the emergency (2 steps)
- **RETURNING**: Heading back to base (2 steps)

This creates natural **resource scarcity** — vehicles dispatched early in the episode are unavailable for later calls, forcing the agent to plan ahead.

### What the Agent Sees

Each step, the agent receives a rich text observation:

```
=== INCOMING CALL [CALL-0005] ===
There's a man having chest pains at 743 Maple Avenue. He's sweating
a lot and says his arm feels numb.

=== ACTIVE EVENTS ===
EVT-0001 | fire       | Engine House No. 1        | sev 3 | fire_2 ETA 1 min
EVT-0003 | medical    | Oakwood Apartments        | sev 4 | ambulance_0 ON SCENE

=== UNIT STATUS ===
police_0     | police    | Central Police Station   | FREE
police_1     | police    | Central Police Station   | FREE
ambulance_0  | ambulance | Riverside General        | ON_SCENE → EVT-0003
ambulance_1  | ambulance | Riverside General        | FREE
fire_2       | fire      | Central Fire Station     | DISPATCHED → EVT-0001

=== CITY REFERENCE ===
Riverside General Hospital (hospital) → Oakwood Apartments [3 min], Maple Heights [5 min]
...

=== DISPATCHER NOTES ===
Step 3: CALL-0003 → ambulance ambulance_0
Step 4: CALL-0004 → Duplicate of EVT-0001
```

### What the Agent Outputs

The agent produces a structured JSON action:

```json
{
  "action_type": "dispatch",
  "severity_pred": 4,
  "is_duplicate": false,
  "vehicle_type": "ambulance",
  "vehicle_id": "ambulance_1",
  "reroute": null
}
```

Three action types:
- **`dispatch`** — Send a free vehicle to handle the emergency
- **`duplicate`** — Flag the call as a repeat of an existing event
- **`hold`** — Queue the call for a busy vehicle (when no free units exist)

---

## Reward Engineering

One of the most critical design decisions in RL is the reward function. We decomposed the reward into **5 independent components**, each measuring a different aspect of dispatch quality:

### Component Breakdown

| Component | Range | What It Measures |
|-----------|-------|-----------------|
| **Severity** | -0.5 to +1.0 | How close the predicted severity is to ground truth. Exact match = +1.0, off by 1 = +0.6, off by 4 = -0.5 |
| **Duplicate** | -1.0 to +1.5 | Correct duplicate detection. Flagging a real duplicate with the right event ID = +1.5. Missing a duplicate = -1.0 |
| **Vehicle Type** | -1.5 to +1.5 | Sending the right type of vehicle. Ambulance to a medical call = +1.5. Police to a fire = -1.5 |
| **Vehicle Choice** | -5.0 to +1.0 | Is the vehicle real, free, correct type, and nearby? Hallucinating a vehicle ID = -5.0. Nearest free unit = +1.0 |
| **Reroute** | -1.0 to +1.7 | Quality of optional reroute decisions. Valid reroute from low→high severity with replacement = +1.7 |

### The Baseline Subtraction Problem

Early in development, we discovered a critical issue: **the reward was always positive**, even for a random agent. Why?

- Most calls (70-90%) are NOT duplicates → saying "not duplicate" gave a free +1.0
- Severity predictions off by 1 still gave +0.6
- Not attempting a reroute gave 0.0 (no penalty)

A random agent would score ~+2.5 per step just by existing. This meant the GRPO training curve was flat — the agent couldn't distinguish good actions from bad ones.

**Our solution**: We introduced a **baseline subtraction** (`STEP_REWARD_BASELINE = 2.5`), calibrated to the expected reward of a mediocre agent. This shifts the reward so that:

| Agent Quality | Raw Reward | After Baseline | Training Signal |
|--------------|-----------|---------------|-----------------|
| Random | +1.5/step | **-1.0/step** | Negative → must improve |
| SFT (decent) | +3.0/step | **+0.5/step** | Near zero → starting point |
| GRPO (good) | +4.5/step | **+2.0/step** | Positive → improvement visible |
| Perfect | +6.7/step | **+4.2/step** | High ceiling → room to grow |

This is the standard approach in RL — similar to advantage estimation in PPO/A2C, where you subtract a value baseline from returns to reduce variance and center the signal.

---

## Curriculum Learning — Task Difficulty

To produce the classic RL training curve (starts near 0, climbs with dips), we structured the environment into **3 progressive difficulty levels** that act as a curriculum:

### Task 1 — Basic Dispatch (Easy)

| Parameter | Value |
|-----------|-------|
| Steps | 10 |
| Vehicles per type | **3** (always a free unit) |
| Duplicate probability | 10% |
| Focus | Learn severity prediction + correct vehicle type |

At this level, the agent always has free vehicles available. It just needs to learn: fire → fire truck, medical → ambulance, crime → police. This is the "format learning" phase.

### Task 2 — Scarce Resources (Medium)

| Parameter | Value |
|-----------|-------|
| Steps | 15 |
| Vehicles per type | **2** (sometimes all busy) |
| Duplicate probability | 30% |
| Focus | Handle holds + pick nearest unit + detect duplicates |

With only 2 vehicles per type, the agent will encounter situations where all ambulances are busy. It must learn to use `hold` actions and pick the vehicle that will free up soonest. Duplicate calls appear more frequently, requiring pattern matching.

### Task 3 — Full Disaster Response (Hard)

| Parameter | Value |
|-----------|-------|
| Steps | 20 |
| Vehicles per type | **1** (constant scarcity) |
| Duplicate probability | 50% |
| Focus | Reroutes + optimal triage under extreme constraints |

With just 1 vehicle per type and 20 incoming calls, the agent faces constant resource conflicts. It must:
- Reroute vehicles from low-severity events to high-severity ones
- Queue multiple events on the same vehicle via holds
- Detect duplicates aggressively to avoid wasting resources
- Make triage decisions: which patients wait?

### Training Flow

During GRPO training, we cycle through all 3 tasks. The training reward curve shows the characteristic pattern:

```
reward
  │          Task 2      Task 3
  │        ╭──╮ dip    ╭──╮ dip
  │      ╭─╯  ╰─╮   ╭─╯  ╰─╮
  │    ╭─╯       ╰─╭─╯       ╰──── plateau
  │  ╭─╯    climb   climb
  │──╯  Task 1
  └──────────────────────────────── training steps
```

---

## The Agent — SFT + GRPO Training Pipeline

### Why Two Phases?

You can't directly train an LLM with RL from scratch — it wouldn't even know to output valid JSON, let alone make dispatch decisions. We use a two-phase approach:

### Phase 1 — Supervised Fine-Tuning (SFT)

**Goal**: Teach the model the correct output format.

We generate a dataset of (observation, ideal_action) pairs by running the environment and computing the optimal action from ground-truth labels:

```python
# For each call, build the ideal action from hidden ground truth
def build_ideal_action(ground_truth, observation_text):
    if ground_truth["is_duplicate"]:
        return {"action_type": "duplicate", "severity_pred": gt_severity, ...}
    else:
        vehicle = find_nearest_free(observation_text, gt_vehicle_type)
        return {"action_type": "dispatch", "vehicle_id": vehicle, ...}
```

We fine-tune **Qwen3-1.7B** (4-bit quantized via Unsloth) on this dataset for ~100 steps. After SFT, the model can:
- ✅ Output valid JSON consistently
- ✅ Identify the correct vehicle type ~80% of the time
- ❌ Doesn't yet optimize for nearest vehicle
- ❌ Can't handle holds or reroutes
- ❌ Duplicate detection is weak

### Phase 2 — Group Relative Policy Optimization (GRPO)

**Goal**: Improve dispatch strategy through trial and error against the live environment.

GRPO (from DeepSeekMath) is a variant of policy optimization that:
1. Generates **multiple completions** (4 per prompt) at high temperature
2. Steps each completion through the real environment to get rewards
3. Uses the **relative ranking** of rewards within each group to update the policy
4. Doesn't require a separate critic/value network (unlike PPO)

```python
# The reward function talks to the real environment
def env_reward_fn(prompts, completions, **kwargs):
    rewards = []
    for completion, seed, task_id in zip(completions, seeds, task_ids):
        env.reset(task_id=task_id, seed=seed)     # Reproduce exact state
        action = parse_llm_action(completion)       # Parse model output
        result = env.step(action)                   # Get env reward
        reward = result.reward_breakdown["total"]   # Baseline-adjusted
        rewards.append(reward + 0.5)                # +0.5 format bonus
    return rewards
```

Key insight: we store the **random seed** with each training example so we can deterministically reproduce the exact same city and call during reward computation. This eliminates environment stochasticity from the reward signal.

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Base model | `unsloth/Qwen3-1.7B-unsloth-bnb-4bit` |
| Quantization | 4-bit NF4 (QLoRA via Unsloth) |
| SFT steps | ~100 |
| GRPO epochs | 1 |
| Batch size | 1 (gradient accumulation 16) |
| Num generations | 4 per prompt |
| Learning rate | 5e-6 |
| Temperature | 1.0 (encourages exploration) |
| Max completion length | 256 tokens |
| Runtime | Hugging Face Spaces A100 GPU |

---

## Technical Stack

### Environment Server

| Component | Technology | Why |
|-----------|-----------|-----|
| Framework | **FastAPI** (via OpenEnv `create_app`) | Async HTTP/WS, auto Swagger docs |
| RL Interface | **OpenEnv** (Meta) | Standard reset/step/grader API |
| City Graph | Custom procedural generation + **Dijkstra** | Realistic road networks with travel times |
| Call Templates | 25 handwritten templates × 4 types | Authentic 911 transcripts |
| Deployment | **Docker** → **Hugging Face Spaces** | Free hosting with GPU support |
| Protocol | HTTP + WebSocket | Low-latency for training loops |

### Training Pipeline

| Component | Technology | Why |
|-----------|-----------|-----|
| Model | **Qwen3-1.7B** | Strong reasoning at small size |
| Quantization | **Unsloth** (4-bit QLoRA) | 2× faster training, 70% less memory |
| SFT | **SFTTrainer** (TRL) | Standard supervised fine-tuning |
| GRPO | **GRPOTrainer** (TRL) | No critic network needed, stable for LLMs |
| Dataset | **HuggingFace Datasets** | Streaming, seed-indexed |
| Compute | **Hugging Face Spaces** (A100) | GPU access |

### Infrastructure

| Component | Technology | Why |
|-----------|-----------|-----|
| Hosting | **Hugging Face Spaces** | Free Docker deployment |
| Model Hub | **Hugging Face Hub** | Model versioning, automatic endpoints |
| Version Control | **GitHub** → synced to HF | CI/CD pipeline |
| Monitoring | **matplotlib** in-notebook | Real-time training curves |

---

## Problems We Faced & How We Solved Them

### 1. "The reward never goes negative"

**Problem**: Even a random agent scored +2.5 per step because most reward components defaulted to positive values (e.g., +1.0 for correctly saying "not a duplicate" on non-duplicate calls).

**Solution**: Introduced `STEP_REWARD_BASELINE = 2.5` — subtracted from every step's total reward. This centers the reward so that average performance → 0, good performance → positive, bad performance → negative. This is the RL equivalent of advantage estimation.

### 2. "The training curve is flat, not climbing"

**Problem**: The SFT model already scored high on easy tasks, leaving no room for GRPO to show improvement.

**Solution**: Implemented **curriculum learning** via task difficulty. Vehicle count scales inversely with difficulty (3 → 2 → 1 per type). The agent must learn progressively harder strategies, creating natural dips and climbs in the reward curve.

### 3. "Hallucinated vehicle IDs"

**Problem**: The LLM would sometimes generate vehicle IDs that don't exist in the current city (e.g., `ambulance_5` when only `ambulance_0` and `ambulance_1` exist).

**Solution**: Heavy penalty (-5.0) for non-existent vehicle IDs in the reward function. The observation explicitly lists all vehicle IDs and their status, and the SFT phase trains on examples that only reference real IDs.

### 4. "Reroute from higher to lower severity"

**Problem**: The agent would sometimes reroute a vehicle from a critical event to a minor one — the opposite of what makes sense.

**Solution**: The reward function checks `reroute_severity_delta` — if the new event is lower severity than the old one, it gets a penalty (-0.5). Only reroutes from lower to higher severity get bonuses.

### 5. "JSON parsing failures"

**Problem**: Early in training, the model outputs malformed JSON (missing brackets, wrong field names, extra text around the JSON).

**Solution**: 
- SFT phase ensures 95%+ format correctness before GRPO begins
- Format bonus (+0.5) in the GRPO reward for valid JSON
- Heavy penalty (-2.0) for unparseable output
- Robust regex-based JSON extraction that handles markdown code fences

### 6. "All vehicles busy → agent freezes"

**Problem**: When all vehicles of the needed type were dispatched, the agent would still try to dispatch a busy vehicle (getting -2.0 penalty) instead of using hold.

**Solution**: Added the `hold` action type with its own reward logic:
- Hold when all units are busy → +1.0 (justified)
- Hold when a free unit exists → -2.0 (unjustified)
- Hold and pick the soonest-to-free vehicle → +0.3 bonus

### 7. "Environment too deterministic"

**Problem**: With fixed seeds, the same GRPO training example always produces the same city and calls. The agent could memorize rather than generalize.

**Solution**: Pre-generate 500 distinct (seed, task_id) combinations spread across all 3 difficulty levels. Each seed produces a unique city graph, call sequence, and vehicle placement. The agent must generalize across all configurations.

---

## Results

### Training Metrics

Our GRPO training shows the expected learning curve:

- **Steps 1-7**: Reward mostly negative (-0.7 to -1.4) — agent learning
- **Steps 8-14**: Reward turning positive (+0.6 to +1.4) — strategies improving
- **Steps 15+**: Occasional dips with overall upward trend — exploration vs exploitation

| Metric | Start | End | Trend |
|--------|-------|-----|-------|
| Reward | -0.71 | +1.45 | ↑ Climbing |
| Loss | 0.0006 | 0.0002 | ↓ Decreasing |
| KL Divergence | 0.55 | 0.23 | ↓ Stable |
| Reward Std | 1.86 | 1.35 | ↓ Converging |

### Baseline Comparison

| Agent | Task 1 (Easy) | Task 2 (Medium) | Task 3 (Hard) |
|-------|--------------|----------------|---------------|
| Random | -3.0/step | -3.1/step | -4.9/step |
| Rule-based heuristic | +1.0/step | +0.3/step | -0.6/step |
| Our GRPO agent | **+1.5/step** | **+0.8/step** | **+0.1/step** |

### What the Agent Learned

After training, the agent demonstrates:

1. **Accurate severity classification** — reads emotional cues ("not breathing" → 5, "fender bender" → 2)
2. **Correct vehicle type selection** — fire keywords → fire truck, medical → ambulance
3. **Nearest vehicle dispatch** — uses city reference distances to pick closest unit
4. **Duplicate detection** — recognizes when a new call matches an active event location/type
5. **Hold decisions** — queues events when no free units exist instead of hallucinating
6. **Reroute reasoning** — redirects vehicles from low-severity to high-severity events

---

## Conclusion

We built **Smart Emergency Dispatch** — a complete RL pipeline for training LLM agents as emergency dispatchers. The key innovations are:

1. **A rich, procedurally-generated environment** with realistic 911 transcripts, city graphs, and vehicle lifecycle management
2. **5-component decomposed reward** with baseline subtraction for clean training signals
3. **Curriculum learning** across 3 difficulty levels that produces the classic RL training curve
4. **SFT → GRPO two-phase training** that first teaches format, then optimizes strategy
5. **OpenEnv-compatible API** so anyone can train their own agent against our environment

The environment is [live on Hugging Face Spaces](https://huggingface.co/spaces/Harsh-Gupta-07/smart_emergency), and the trained model is available at [rishi38/smart-emergency-grpo](https://huggingface.co/rishi38/smart-emergency-grpo).

---

*Built with ❤️ using OpenEnv, Unsloth, TRL, and Hugging Face.*
