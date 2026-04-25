# %% [markdown]
# # 🚨 Smart Emergency Dispatch — SFT → GRPO Training (Colab + Unsloth)
#
# Fine-tunes **Qwen3-1.7B** as an emergency 911 dispatcher using **Unsloth** for 2× faster training:
# 1. **Phase 1 — SFT**: Teach the model the JSON output format
# 2. **Phase 2 — GRPO**: Improve dispatch strategy via RL against the live HF Space environment
#
# **Runtime**: Google Colab with T4 or A100 GPU

# %% [markdown]
# ## 0 · Install Dependencies

# %%
!pip install -Uq unsloth vllm
!pip install -Uq git+https://github.com/huggingface/trl.git
!pip install -Uq git+https://github.com/meta-pytorch/OpenEnv.git
!pip install -Uq git+https://github.com/rishiraj38/Smart_Emergency.git datasets requests

# %%
from huggingface_hub import notebook_login
notebook_login()

# %% [markdown]
# ## 1 · Configuration

# %%
import os, json, re, random, requests, time
from collections import defaultdict

MODEL_NAME = "unsloth/Qwen3-1.7B-unsloth-bnb-4bit"
SFT_OUTPUT_DIR = "smart-emergency-sft"
GRPO_OUTPUT_DIR = "smart-emergency-grpo"
MAX_SEQ_LENGTH = 3072

# HuggingFace Space URL for the environment server
HF_SPACE_URL = "https://rishi38-eme-enviro.hf.space"

# %% [markdown]
# ## 2 · Connect to Environment
#
# Wake the HF Space if sleeping, then connect directly using `SmartEmergencyEnv`.

# %%
import requests, time
from smart_emergency import SmartEmergencyEnv, SmartEmergencyAction

# Ping the Space health endpoint until it wakes up (free Spaces sleep after inactivity)
print("⏳ Waking up HF Space (may take 30-60s if sleeping) …")
for _attempt in range(60):
    try:
        r = requests.get(f"{HF_SPACE_URL}/health", timeout=5)
        if r.status_code == 200:
            print(f"✅ Space awake at {HF_SPACE_URL}")
            break
    except Exception:
        pass
    time.sleep(2)
else:
    raise RuntimeError("HF Space did not respond after 2 minutes. Check the URL.")

# Direct WebSocket connection via the official client
env = SmartEmergencyEnv(base_url=HF_SPACE_URL).sync()
_test = env.reset()
print(f"✅ Connected — first call: {_test.observation.call_id}")

# %% [markdown]
# ## 3 · System Prompt

# %%
SYSTEM_PROMPT = """\
You are an expert 911 emergency dispatcher. You receive incoming calls and must make rapid, structured dispatch decisions.

## RULES
1. Each step you see: an incoming call transcript, active events, unit status, and a city map.
2. You must respond with a single JSON object — nothing else.

## ACTION TYPES
You have three action types: `dispatch`, `duplicate`, and `hold`.

### 1. dispatch — Handle a new emergency
Use when a FREE vehicle of the correct type is available.
```json
{
  "action_type": "dispatch",
  "severity_pred": <int 1-5>,
  "is_duplicate": false,
  "duplicate_of_event_id": null,
  "vehicle_type": "police" | "ambulance" | "fire",
  "vehicle_id": "<unit_id of a FREE vehicle>",
  "reroute": null
}
```

### 2. duplicate — Flag a repeat call
Use when the incoming call matches an existing active event (same location/type).
```json
{
  "action_type": "duplicate",
  "severity_pred": <int 1-5>,
  "is_duplicate": true,
  "duplicate_of_event_id": "<EVT-NNNN>",
  "vehicle_type": null,
  "vehicle_id": null,
  "reroute": null
}
```

### 3. hold — Queue for a busy vehicle
Use ONLY when ALL vehicles of the required type are busy (none are FREE).
```json
{
  "action_type": "hold",
  "severity_pred": <int 1-5>,
  "is_duplicate": false,
  "duplicate_of_event_id": null,
  "vehicle_type": "police" | "ambulance" | "fire",
  "vehicle_id": "<unit_id of a BUSY vehicle to queue behind>",
  "reroute": null
}
```
**Hold rules:** NEVER hold if a free unit exists. Pick the vehicle with the lowest ETA.

## REROUTE (optional, only with dispatch)
Redirect an in-flight vehicle from a LOWER-severity event to this HIGHER-severity one:
```json
"reroute": {
  "vehicle_to_reroute": "<DISPATCHED unit_id>",
  "from_event_id": "<EVT-NNNN>",
  "replacement_vehicle_id": "<FREE unit or null>"
}
```
Only reroute DISPATCHED vehicles. Only reroute from lower to higher severity.

## SEVERITY GUIDE
1=minor, 2=moderate, 3=serious, 4=critical, 5=catastrophic

## VEHICLE GUIDE
- **fire** → fire, smoke, flames, gas leak
- **police** → shooting, robbery, fight, break-in
- **ambulance** → medical, crash, accident, injury, collapse

## STRATEGY
- Pick the nearest FREE vehicle (use CITY REFERENCE distances).
- If call matches an ACTIVE EVENT, flag as duplicate.
- No free units → use `hold`. Higher severity than busy units → consider `reroute`.
"""

# %% [markdown]
# ---
# # Phase 1 — Supervised Fine-Tuning (SFT)

# %% [markdown]
# ### Observation Parsing Helpers

# %%
def parse_free_vehicles(obs_text: str) -> dict:
    """Return {unit_id: vehicle_type} for FREE vehicles."""
    vehicles = {}
    in_section = False
    for line in obs_text.split("\n"):
        if "=== UNIT STATUS ===" in line:
            in_section = True; continue
        if in_section and line.startswith("==="):
            break
        if in_section and "|" in line and "FREE" in line:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 2:
                vehicles[parts[0]] = parts[1]
    return vehicles


def parse_all_vehicles(obs_text: str) -> list:
    """Return list of {id, type, status} for ALL vehicles."""
    vehicles = []
    in_section = False
    for line in obs_text.split("\n"):
        if "=== UNIT STATUS ===" in line:
            in_section = True; continue
        if in_section and line.startswith("==="):
            break
        if in_section and "|" in line:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 4:
                status = parts[3].split()[0] if parts[3] else "UNKNOWN"
                vehicles.append({"id": parts[0], "type": parts[1], "status": status})
    return vehicles


def parse_active_events(obs_text: str) -> dict:
    events = {}
    in_section = False
    for line in obs_text.split("\n"):
        if "=== ACTIVE EVENTS ===" in line:
            in_section = True; continue
        if in_section and line.startswith("==="):
            break
        if in_section and "|" in line and "EVT-" in line:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 2:
                events[parts[0]] = parts[1]
    return events


TYPE_TO_VEHICLE = {"fire": "fire", "medical": "ambulance", "crime": "police", "accident": "ambulance"}

SEV_KW = {
    5: ["not breathing", "active shooter", "trapped", "mass incident", "whole block", "pileup", "send everything"],
    4: ["won't wake", "gunshots", "flipped", "blood everywhere", "kids are upstairs", "not responding"],
    3: ["chest pain", "fight", "mugged", "knife", "crash", "bleeding", "fire at", "flames", "cyclist"],
    2: ["fainted", "break-in", "dumpster", "fender", "small fire", "ankle", "shoplifter"],
}


def heuristic_severity(text):
    t = text.lower()
    for sev in [5, 4, 3, 2]:
        if any(kw in t for kw in SEV_KW[sev]):
            return sev
    return 1


def heuristic_vehicle_type(text):
    t = text.lower()
    if any(w in t for w in ["fire", "flames", "smoke", "burning", "gas leak"]):
        return "fire"
    if any(w in t for w in ["shooter", "gunshot", "mugged", "knife", "break-in", "fight", "shoplifter"]):
        return "police"
    return "ambulance"


def pick_free(free_vehicles, vtype):
    for vid, vt in free_vehicles.items():
        if vt == vtype:
            return vid
    return None


def pick_busy(all_vehicles, vtype):
    for v in all_vehicles:
        if v["type"] == vtype and v["status"] != "FREE":
            return v["id"]
    return None

# %% [markdown]
# ### Generate SFT Dataset

# %%
def build_ideal_action(gt, obs_text):
    """Build ideal JSON action dict from ground truth + observation."""
    sev = gt.get("severity", 1)
    vtype = gt.get("required_vehicle_type", "ambulance")
    is_dup = gt.get("is_duplicate", False)

    if is_dup:
        active = parse_active_events(obs_text)
        etype = gt.get("emergency_type", "")
        dup_eid = None
        for eid, et in active.items():
            if et.strip() == etype:
                dup_eid = eid; break
        if dup_eid is None and active:
            dup_eid = list(active.keys())[0]
        return {"action_type": "duplicate", "severity_pred": sev, "is_duplicate": True,
                "duplicate_of_event_id": dup_eid, "vehicle_type": None, "vehicle_id": None, "reroute": None}

    free = parse_free_vehicles(obs_text)
    vid = pick_free(free, vtype)
    if vid:
        return {"action_type": "dispatch", "severity_pred": sev, "is_duplicate": False,
                "duplicate_of_event_id": None, "vehicle_type": vtype, "vehicle_id": vid, "reroute": None}

    busy_vid = pick_busy(parse_all_vehicles(obs_text), vtype)
    if busy_vid:
        return {"action_type": "hold", "severity_pred": sev, "is_duplicate": False,
                "duplicate_of_event_id": None, "vehicle_type": vtype, "vehicle_id": busy_vid, "reroute": None}

    return {"action_type": "dispatch", "severity_pred": sev, "is_duplicate": False,
            "duplicate_of_event_id": None, "vehicle_type": vtype, "vehicle_id": f"{vtype}_0", "reroute": None}


def generate_sft_data(env, num_episodes=60):
    examples = []
    for ep in range(num_episodes):
        task_id = (ep % 3) + 1
        result = env.reset(task_id=task_id)
        prev_obs = result.observation.prompt

        while not result.done:
            free = parse_free_vehicles(prev_obs)
            vtype = heuristic_vehicle_type(prev_obs)
            vid = pick_free(free, vtype)
            action = SmartEmergencyAction(
                action_type="dispatch",
                severity_pred=heuristic_severity(prev_obs),
                is_duplicate=False,
                vehicle_type=vtype,
                vehicle_id=vid,
            )

            result = env.step(action)
            # ground_truth is now a first-class field on the observation;
            # fall back to metadata for backward compatibility with older servers.
            gt = result.observation.ground_truth or result.observation.metadata.get("ground_truth")
            if gt:
                ideal = build_ideal_action(gt, prev_obs)
                examples.append({
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prev_obs},
                        {"role": "assistant", "content": json.dumps(ideal)},
                    ]
                })
            prev_obs = result.observation.prompt

        if (ep + 1) % 10 == 0:
            print(f"  Episodes: {ep+1}/{num_episodes} | examples: {len(examples)}")
    return examples


print("📝 Generating SFT data …")
sft_examples = generate_sft_data(env, num_episodes=60)
print(f"✅ Collected {len(sft_examples)} SFT examples")

# %%
from datasets import Dataset
sft_dataset = Dataset.from_list(sft_examples)
print(sft_dataset)

# %% [markdown]
# ### SFT Training with Unsloth

# %%
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth",
)

sft_config = SFTConfig(
    output_dir=SFT_OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    logging_steps=5,
    save_steps=50,
    max_seq_length=MAX_SEQ_LENGTH,
    bf16=True,
    report_to="none",
)

sft_trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=sft_dataset,
    args=sft_config,
)

# %%
print("🏋️ Starting SFT training …")
sft_trainer.train()
print("✅ SFT complete")

# %%
sft_trainer.save_model(SFT_OUTPUT_DIR)
tokenizer.save_pretrained(SFT_OUTPUT_DIR)
print(f"✅ SFT model saved to {SFT_OUTPUT_DIR}/")

# Free memory
import torch, gc
del model, sft_trainer
gc.collect()
torch.cuda.empty_cache()

# %% [markdown]
# ---
# # Phase 2 — GRPO with Unsloth

# %% [markdown]
# ### Action Parsing

# %%
def parse_llm_action(text):
    """Extract action dict from LLM output."""
    m = re.search(r"```json\s*(.*?)```", text, re.DOTALL)
    if m:
        text = m.group(1)
    else:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            text = m.group(0)
    try:
        d = json.loads(text)
        # Validate required fields
        assert d.get("action_type") in ("dispatch", "duplicate", "hold")
        assert 1 <= int(d.get("severity_pred", 0)) <= 5
        return d
    except Exception:
        return None


def fallback_action(obs_text):
    free = parse_free_vehicles(obs_text)
    vtype = heuristic_vehicle_type(obs_text)
    vid = pick_free(free, vtype)
    if vid:
        return {"action_type": "dispatch", "severity_pred": heuristic_severity(obs_text),
                "is_duplicate": False, "vehicle_type": vtype, "vehicle_id": vid}
    busy_vid = pick_busy(parse_all_vehicles(obs_text), vtype)
    return {"action_type": "hold" if busy_vid else "dispatch",
            "severity_pred": heuristic_severity(obs_text), "is_duplicate": False,
            "vehicle_type": vtype, "vehicle_id": busy_vid or f"{vtype}_0"}

# %% [markdown]
# ### Rollout Functions

# %%
from unsloth import FastLanguageModel, PatchFastRL
from trl import GRPOConfig, GRPOTrainer

# Patch TRL for Unsloth compatibility
PatchFastRL("GRPO", FastLanguageModel)

# Load the SFT model for GRPO with fast inference (vLLM)
grpo_model, grpo_tokenizer = FastLanguageModel.from_pretrained(
    model_name=SFT_OUTPUT_DIR,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
    fast_inference=True,  # enables vLLM for GRPO generation
)

grpo_model = FastLanguageModel.get_peft_model(
    grpo_model,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth",
)

# %%
def make_user_prompt(obs_text):
    return f"You are the dispatcher. Read the situation and respond with a single JSON action.\n\n{obs_text}\n\nRespond ONLY with a JSON object."


def action_dict_to_obj(d):
    """Convert a plain dict action to SmartEmergencyAction."""
    from smart_emergency import RerouteAction
    reroute = None
    if d.get("reroute") and isinstance(d["reroute"], dict):
        rd = d["reroute"]
        reroute = RerouteAction(
            vehicle_to_reroute=rd["vehicle_to_reroute"],
            from_event_id=rd["from_event_id"],
            replacement_vehicle_id=rd.get("replacement_vehicle_id"),
        )
    return SmartEmergencyAction(
        action_type=d.get("action_type", "dispatch"),
        severity_pred=int(d.get("severity_pred", 1)),
        is_duplicate=bool(d.get("is_duplicate", False)),
        duplicate_of_event_id=d.get("duplicate_of_event_id"),
        vehicle_type=d.get("vehicle_type"),
        vehicle_id=d.get("vehicle_id"),
        reroute=reroute,
    )


def rollout_once(trainer, env, tokenizer, system_prompt, max_turns=15):
    """Run one full episode."""
    from trl.experimental.openenv import generate_rollout_completions

    result = env.reset()
    prompt_ids, completion_ids, logprobs = [], [], []
    rewards = {k: [] for k in ["severity", "duplicate", "vehicle_type", "vehicle_choice", "reroute", "format"]}

    for _ in range(max_turns):
        if result.done:
            break

        obs_text = result.observation.prompt
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": make_user_prompt(obs_text)},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, enable_thinking=False,
        )

        out = generate_rollout_completions(trainer, [prompt_text])[0]
        prompt_ids.extend(out["prompt_ids"])
        completion_ids.extend(out["completion_ids"])
        logprobs.extend(out["logprobs"])

        comp_text = out.get("text") or tokenizer.decode(out["completion_ids"], skip_special_tokens=True)
        action_d = parse_llm_action(comp_text)
        parse_ok = action_d is not None
        if action_d is None:
            action_d = fallback_action(obs_text)
        action = action_dict_to_obj(action_d)

        result = env.step(action)
        bd = result.observation.reward_breakdown

        rewards["severity"].append(bd.get("severity", 0.0))
        rewards["duplicate"].append(bd.get("duplicate", 0.0))
        rewards["vehicle_type"].append(bd.get("vehicle_type", 0.0))
        rewards["vehicle_choice"].append(bd.get("vehicle_choice", 0.0))
        rewards["reroute"].append(bd.get("reroute", 0.0))
        rewards["format"].append(1.0 if parse_ok else -2.0)

    return {
        "prompt_ids": prompt_ids,
        "completion_ids": completion_ids,
        "logprobs": logprobs,
        **{f"{k}_reward": v[-1] if v else 0.0 for k, v in rewards.items()},
    }


def rollout_func(prompts, trainer=None):
    """GRPO rollout — called by GRPOTrainer each step."""
    results = {k: [] for k in ["prompt_ids", "completion_ids", "logprobs",
               "severity_reward", "duplicate_reward", "vehicle_type_reward",
               "vehicle_choice_reward", "reroute_reward", "format_reward"]}

    for _ in prompts:
        ep = rollout_once(trainer, env, grpo_tokenizer, SYSTEM_PROMPT)
        for k in results:
            results[k].append(ep[k])
    return results

# %% [markdown]
# ### Reward Wrappers & Config

# %%
def _make_reward_fn(key):
    def fn(completions, **kwargs):
        r = kwargs.get(key)
        return [float(x) for x in r] if r else [0.0] * len(completions)
    fn.__name__ = f"reward_{key.replace('_reward', '')}"
    return fn

reward_fns = [_make_reward_fn(k) for k in
              ["severity_reward", "duplicate_reward", "vehicle_type_reward",
               "vehicle_choice_reward", "reroute_reward", "format_reward"]]

# %%
grpo_dataset = Dataset.from_dict({
    "prompt": ["Dispatch emergency services for incoming 911 calls."] * 500
})

grpo_config = GRPOConfig(
    num_train_epochs=1,
    learning_rate=5e-6,
    gradient_accumulation_steps=32,
    per_device_train_batch_size=1,
    warmup_steps=10,
    num_generations=4,
    max_completion_length=128,
    max_prompt_length=MAX_SEQ_LENGTH,
    use_vllm=True,
    output_dir=GRPO_OUTPUT_DIR,
    logging_steps=1,
    save_steps=10,
    push_to_hub=True,
)

# %% [markdown]
# ### Train GRPO

# %%
grpo_trainer = GRPOTrainer(
    model=grpo_model,
    processing_class=grpo_tokenizer,
    reward_funcs=reward_fns,
    train_dataset=grpo_dataset,
    args=grpo_config,
    rollout_func=rollout_func,
)

import torch
gpu = torch.cuda.get_device_properties(0)
print(f"GPU: {gpu.name} | {round(gpu.total_memory/1024**3, 1)} GB")
print(f"Reserved: {round(torch.cuda.max_memory_reserved()/1024**3, 2)} GB")

# %%
print("🏋️ Starting GRPO training …")
stats = grpo_trainer.train()
print("✅ GRPO complete")

# %%
peak = round(torch.cuda.max_memory_reserved() / 1024**3, 2)
print(f"Peak memory: {peak} GB | Time: {round(stats.metrics['train_runtime']/60, 1)} min")

grpo_trainer.save_model(GRPO_OUTPUT_DIR)
grpo_trainer.push_to_hub()
print(f"✅ Model saved & pushed to Hub")

# %% [markdown]
# ---
# # Phase 3 — Inference & Evaluation

# %%
from unsloth import FastLanguageModel as FLM

inf_model, inf_tokenizer = FLM.from_pretrained(
    model_name=GRPO_OUTPUT_DIR, max_seq_length=MAX_SEQ_LENGTH, load_in_4bit=True,
)
FLM.for_inference(inf_model)

def run_episode(env, model, tokenizer, task_id=1):
    result = env.reset(task_id=task_id)
    total_reward = 0.0

    for step in range(20):
        if result.done:
            break
        obs_text = result.observation.prompt
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": make_user_prompt(obs_text)},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, enable_thinking=False,
        )
        inputs = tokenizer([prompt_text], return_tensors="pt").to(model.device)
        gen = model.generate(**inputs, max_new_tokens=256, temperature=0.1)
        output = tokenizer.decode(gen[0][len(inputs.input_ids[0]):], skip_special_tokens=True)

        action_d = parse_llm_action(output) or fallback_action(obs_text)
        action = action_dict_to_obj(action_d)
        tag = "✅" if parse_llm_action(output) else "⚠️"
        print(f"  Step {step}: {tag} {action_d.get('action_type')} sev={action_d.get('severity_pred')}")

        result = env.step(action)
        total_reward += result.observation.reward_breakdown.get("total", 0.0)

    print(f"\n  Done — reward: {total_reward:.2f} over {step+1} steps")
    return total_reward

# %%
print("=" * 50)
print("Evaluation — Task 1 (Easy)")
print("=" * 50)
run_episode(env, inf_model, inf_tokenizer, task_id=1)
env.close()
