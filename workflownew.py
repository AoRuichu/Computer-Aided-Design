from google import genai
from google.genai import types
from utils.save_response import writeMd
from dotenv import load_dotenv
from datetime import datetime
import numpy as np
import json
import os
import yaml
from toolify import Args, TD3Runner, CircuitEvaluator
import utils.plotting

env_path = "./local.env"
load_dotenv(dotenv_path=env_path)

client = genai.Client()

# =====================================================
# --- Utility Functions ---
# =====================================================
def sanitize_numpy(obj):
    """Convert NumPy types to floats recursively."""
    if isinstance(obj, dict):
        return {k: sanitize_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_numpy(x) for x in obj]
    elif isinstance(obj, np.generic):
        return float(obj)
    else:
        return obj


def compute_spec_delta(current_specs: dict, target_specs: dict) -> dict:
    """Compute difference between current and target specs."""
    delta = {}
    for k, v in target_specs.items():
        if k not in current_specs:
            continue
        try:
            delta[k] = float(current_specs[k]) - float(v)
        except Exception:
            delta[k] = None
    return delta


def compact_yaml(d: dict, keep_keys=None, precision=6):
    """Compact YAML formatting for prompts."""
    if keep_keys is None:
        keep_keys = list(d.keys())
    slim = {}
    for k in keep_keys:
        v = d.get(k, None)
        if isinstance(v, float):
            slim[k] = float(f"{v:.{precision}g}")
        else:
            slim[k] = v
    import yaml
    return yaml.dump(slim, sort_keys=False)


# =====================================================
# --- AGENT DEFINITIONS ---
# =====================================================
def start_agent(circuit_image, specs_before, target_specs):
    """Agent 1: Analyze circuit and propose optimization strategy."""
    prompt = f"""
You are the **Starter Agent**, an experienced analog IC designer.

Your task is to analyze the Two-Stage Miller op-amp topology and 
form an initial optimization direction to improve its performance.

**Given specs:**
{yaml.dump(sanitize_numpy(specs_before), sort_keys=False)}

**Target specs:**
{yaml.dump(sanitize_numpy(target_specs), sort_keys=False)}

Explain:
1. The likely performance bottlenecks (e.g., gain, phase margin, UGBW).
2. Which transistor groups (input pair, load, bias) most influence each spec.
3. A prioritized plan (max 5 steps) for improving performance.

Be concise and technical. Output plain markdown only.
    """
    contents = [types.Part.from_text(text=prompt)]
    if circuit_image and os.path.exists(circuit_image):
        with open(circuit_image, "rb") as f:
            img_bytes = f.read()
        contents.append(types.Part.from_bytes(data=img_bytes, mime_type="image/png"))

    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=contents,
        config=types.GenerateContentConfig(
            temperature=0.4,
            max_output_tokens=2000,
            system_instruction="You are an analog IC expert explaining design reasoning clearly.",
        ),
    )
    writeMd(response, "llm_start_agent.md")
    return getattr(response, "text", "")


def spec_critic_agent(specs_before, target_specs):
    """Agent 2: Compare measured vs target specs and identify priorities."""
    delta_specs = compute_spec_delta(specs_before, target_specs)
    prompt = f"""
You are the **Spec Critic Agent**.

Compare the measured and target specifications of the op-amp,
and identify which metrics are the critical bottlenecks.

**Measured specs:**
{yaml.dump(sanitize_numpy(specs_before), sort_keys=False)}

**Target specs:**
{yaml.dump(sanitize_numpy(target_specs), sort_keys=False)}

**Delta (current − target):**
{yaml.dump(delta_specs, sort_keys=False)}

Output a short markdown table:
| Metric | Current | Target | Delta | Priority (HARD/SOFT) |
Then write 3–4 sentences explaining what limits each HARD constraint.
"""
    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=[types.Part.from_text(text=prompt)],
        config=types.GenerateContentConfig(
            temperature=0.3,
            max_output_tokens=2000,
            system_instruction="You are an analog design reviewer analyzing performance gaps.",
        ),
    )
    writeMd(response, "llm_spec_critic.md")
    text = getattr(response, "text", "")
    return text, delta_specs

def refinement_agent(best_params, specs_before, target_specs, param_ranges, iteration, critic_feedback=None):
    """Agent 3: Suggest new transistor sizes (JSON)."""
    delta_specs = compute_spec_delta(specs_before, target_specs)
    range_summary = {
        k: {"min": float(v["min"]), "max": float(v["max"])} for k, v in param_ranges.items()
    }

    prompt = f"""
You are the **Parameter Refiner Agent**.

Given the current specs, the parameter ranges, and the analysis from the Spec Critic Agent below,
propose updated transistor sizes to move performance toward the targets.

---
**Spec Critic Analysis:**
{critic_feedback or "N/A"}

**Iteration:** {iteration}

**Parameters (before):**
{yaml.dump(best_params, sort_keys=False)}

**Specs (before):**
{yaml.dump(sanitize_numpy(specs_before), sort_keys=False)}

**Delta (current − target):**
{yaml.dump(delta_specs, sort_keys=False)}

**Parameter ranges:**
{yaml.dump(range_summary, sort_keys=False)}

Rules:
- Keep L ≥ 45 nm.
- Adjust each parameter ≤ ±20% unless necessary.
- If UGBW = 0, redesign to restore amplification.
- Avoid extreme bias currents.
- Return exactly one JSON object under key "optimized_parameters". No extra text.
"""
    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=[types.Part.from_text(text=prompt)],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.3,
            max_output_tokens=8000,
            system_instruction="You output JSON only with transistor sizing.",
        ),
    )
    writeMd(response, f"llm_refiner_iter{iteration}.md")
    text = getattr(response, "text", "")
    try:
        start, end = text.find("{"), text.rfind("}") + 1
        return json.loads(text[start:end])["optimized_parameters"]
    except Exception:
        return None


def coordinator_agent(specs_before, target_specs):
    """Agent 4: Decide whether to call RL or continue refinement."""
    delta = compute_spec_delta(specs_before, target_specs)
    prompt = f"""
You are the **Coordinator Agent**.

Based on how close the circuit is to its target specs,
decide whether to:
- (a) Refine directly, or
- (b) Call the RL sizer tool for more exploration.
- the steps to run for updating the RL memory based on your confidence level. The maximum steps is 1000.

Delta to targets:
{yaml.dump(delta, sort_keys=False)}

Output JSON only:
{{"decision": "refine"|"call_rl", "rl_training_budget": int (if applicable)}}
    """
    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=[types.Part.from_text(text=prompt)],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.2,
            max_output_tokens=1000,
            system_instruction="You choose the optimization strategy and return JSON only.",
        ),
    )
    writeMd(response, "llm_coordinator.md")
    text = getattr(response, "text", "")
    try:
        start, end = text.find("{"), text.rfind("}") + 1
        return json.loads(text[start:end])
    except Exception:
        return {"decision": "refine"}



def main(args=None):
    if args is None:
        args = Args(seed = 42, 
                    noise_sigma = 0.01,
                    gamma = 0.99,
                    tau = 0.005,
                    target_update_interval = 1,
                    actor_update_interval = 2,
                    hidden_size = 256,
                    batch_size = 256,
                    w = 200,
                    pi_lr = 1e-5,
                    q_lr = 1e-3,
                    replay_size = 1_000_000,
                    cuda = False,
                    no_discretize = False,
                    run_id = datetime.now().strftime('%Y-%m-%d--%H-%M-%S'))
    td3_runner = TD3Runner(args)
    circuit_evaluator = td3_runner.env

    # print("\n=== Starter Agent: Analyzing Circuit ===")
    # start_text = start_agent("circuit_image.png", circuit_evaluator.real_specs, circuit_evaluator.dict_targets)
    # print(start_text)

    print("\nStarting warmup exploration...")
    td3_runner.warmup_exploration()
    total_steps= 0
    last_llm_specs = None
    while total_steps < 5000: # 
        print(f"\n========== Current steps {total_steps} ==========")
        best = td3_runner.find_best_design_in_pool()
        best_params, specs_before = best["params"], best["specs"]

        critic_text = spec_critic_agent(specs_before, circuit_evaluator.dict_targets)
        llm_parameters = refinement_agent(best_params, specs_before, circuit_evaluator.dict_targets,
                                          circuit_evaluator.param_ranges, i)

        if llm_parameters:
            print("LLM suggested new parameters, simulating...")
            real_specs_after = circuit_evaluator.simulate(llm_parameters)
            td3_runner.add_llm_experience_to_RL_memory(specs_before, llm_parameters, real_specs_after)
            last_llm_specs = real_specs_after

        decision = coordinator_agent(specs_before, circuit_evaluator.dict_targets)
        if decision.get("decision") == "call_rl":
            steps = int(decision.get("rl_training_budget", 500))
            total_steps += steps
            print(f"Calling RL Sizer for {steps} steps...")
            td3_runner.train_after_llm_guidance(steps)

    print("\nTraining completed.")
    print("Final specs:", td3_runner.env.real_specs)
    print("Final params:", td3_runner.env.param_values)

    csv_fname='./solutions/solutions.csv'
    utils.plotting.solutions2pareto(csv_fname, args.run_id)


if __name__ == "__main__":
    main()
