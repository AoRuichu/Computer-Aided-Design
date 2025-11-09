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

# ---------- SAFETY HELPERS (add once near your utilities) ----------
def safe_generate(model: str, contents, config: types.GenerateContentConfig):
    """Wrap client.models.generate_content with try/except."""
    try:
        return client.models.generate_content(model=model, contents=contents, config=config)
    except Exception as e:
        print(f"⚠️ Gemini API call failed: {e}")
        return None

def extract_text_from_response(response) -> str:
    """Best-effort text extraction across JSON/text responses."""
    if response is None:
        return ""
    # Preferred: direct .text when present
    if hasattr(response, "text") and response.text:
        return response.text
    # Fallback: assemble text from parts, if any
    try:
        if hasattr(response, "candidates") and response.candidates:
            content = response.candidates[0].content
            if content and hasattr(content, "parts") and content.parts:
                return "\n".join(getattr(p, "text", "") for p in content.parts if hasattr(p, "text"))
    except Exception:
        pass
    return ""

def write_md_if_any(response, filename: str):
    """Call writeMd only if parts exist; otherwise skip safely."""
    try:
        if (
            response is not None
            and hasattr(response, "candidates")
            and response.candidates
            and response.candidates[0].content
            and hasattr(response.candidates[0].content, "parts")
            and response.candidates[0].content.parts
        ):
            writeMd(response, filename)
        else:
            print(f"⚠️ No content.parts for {filename}; skipping writeMd().")
    except Exception as e:
        print(f"⚠️ writeMd failed for {filename}: {e}")

def parse_first_json_blob(text: str):
    """Find first {...} blob and parse; return dict or None."""
    if not text:
        return None
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
    except Exception:
        pass
    return None


# =====================================================
# --- AGENT DEFINITIONS ---
# =====================================================

def spec_critic_agent(specs_before, target_specs):
    
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
    response = safe_generate(
        model="gemini-2.5-pro",
        contents=[types.Part.from_text(text=prompt)],
        config=types.GenerateContentConfig(
            temperature=0.3,
            max_output_tokens=2000,
            system_instruction="You are an analog design reviewer analyzing performance gaps.",
        ),
    )

    # Only write markdown if content.parts exist
    write_md_if_any(response, "llm_spec_critic.md")

    text = extract_text_from_response(response)
    return text, delta_specs

def refinement_agent(best_params, specs_before, target_specs, param_ranges, critic_feedback=None):

    delta_specs = compute_spec_delta(specs_before, target_specs)
    range_summary = {k: {"min": float(v["min"]), "max": float(v["max"])} for k, v in param_ranges.items()}

    prompt = f"""
        You are the **Parameter Refiner Agent**.

        Given the current specs, the parameter ranges, and the analysis from the Spec Critic Agent below,
        propose updated transistor sizes to move performance toward the targets.

        ---
        **Spec Critic Analysis:**
        {critic_feedback or "N/A"}

        **Parameters (before):**
        {yaml.dump(best_params, sort_keys=False)}

        **Specs (before):**
        {yaml.dump(sanitize_numpy(specs_before), sort_keys=False)}

        **Delta (current - target):**
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
    response = safe_generate(
        model="gemini-2.5-pro",
        contents=[types.Part.from_text(text=prompt)],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.3,
            max_output_tokens=8000,
            system_instruction="You output JSON only with transistor sizing.",
        ),
    )

    # For JSON-mode, response.content.parts is often None → skip writeMd safely
    write_md_if_any(response, "llm_refiner.md")

    text = extract_text_from_response(response)
    obj = parse_first_json_blob(text)
    if isinstance(obj, dict) and "optimized_parameters" in obj:
        return obj["optimized_parameters"]
    return None

def coordinator_agent(specs_before, target_specs):
    
    delta = compute_spec_delta(specs_before, target_specs)
    prompt = f"""
        You are the **Coordinator Agent**.

        Based on how close the circuit is to its target specs,
        decide whether to:
        - (a) Refine directly, or
        - (b) Call the RL sizer tool for more exploration.
        - Choose steps (max 1000) for updating RL memory based on confidence.

        Delta to targets:
        {yaml.dump(delta, sort_keys=False)}

        Output JSON only:
        {{"decision": "refine"|"call_rl", "rl_training_budget": int (if applicable)}}
            """

    response = safe_generate(
        model="gemini-2.5-pro",
        contents=[types.Part.from_text(text=prompt)],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.2,
            max_output_tokens=2000,
            system_instruction="You choose the optimization strategy and return JSON only.",
        ),
    )
    # Log if possible
    write_md_if_any(response, "llm_coordinator.md")

    text = extract_text_from_response(response)
    obj = parse_first_json_blob(text)
    if isinstance(obj, dict) and "decision" in obj:
        # enforce default budget when call_rl has no budget
        if obj.get("decision") == "call_rl":
            obj.setdefault("rl_training_budget", 500)
        return obj

    print("⚠️ Coordinator returned no valid JSON; defaulting to refine.")
    return {"decision": "refine", "rl_training_budget": 500}




def main(args=None):
    if args is None:
        args = Args(
            seed=42,
            noise_sigma=0.1,
            gamma=0.99,
            tau=0.005,
            target_update_interval=1,
            actor_update_interval=2,
            hidden_size=256,
            batch_size=256,
            w=300,                
            pi_lr=1e-4,
            q_lr=1e-3,
            replay_size=1_000_000,
            cuda=False,
            no_discretize=False,
            run_id=datetime.now().strftime('%Y-%m-%d--%H-%M-%S'),
        )

    td3_runner = TD3Runner(args)
    circuit_evaluator = td3_runner.env

    
    print("\nStarting warmup exploration...")
    td3_runner.warmup_exploration()
    print(f"Warmup finished. total_steps = {td3_runner.total_steps}")

  
    TARGET_TOTAL_STEPS = 5000

    last_llm_specs = None
    while td3_runner.total_steps < TARGET_TOTAL_STEPS:
        print(f"\n========== RL completed steps so far: {td3_runner.total_steps} ==========")

      
        best = td3_runner.find_best_design_in_pool()
        best_params, specs_before = best["params"], best["specs"]

      
        critic_feedback, delta_specs = spec_critic_agent(
            specs_before, circuit_evaluator.dict_targets
        )
        print("\n=== Spec Critic Agent Analysis ===")

        llm_parameters = refinement_agent(
            best_params,
            specs_before,
            circuit_evaluator.dict_targets,
            circuit_evaluator.param_ranges,
            critic_feedback=critic_feedback,
        )

        if llm_parameters:
            print("LLM suggested new parameters, simulating...")
            real_specs_after = circuit_evaluator.simulate(llm_parameters)
            td3_runner.add_llm_experience_to_RL_memory(
                specs_before, llm_parameters, real_specs_after
            )
            last_llm_specs = real_specs_after

     
        decision = coordinator_agent(specs_before, circuit_evaluator.dict_targets)
        if decision.get("decision") == "call_rl":
           
            budget_increment = int(decision.get("rl_training_budget", 500))
            prev_steps = td3_runner.total_steps
            absolute_target = min(prev_steps + budget_increment, TARGET_TOTAL_STEPS)

            print(
                f"Calling RL Sizer for {budget_increment} steps "
                f"(from {prev_steps} -> target {absolute_target})..."
            )

            
            td3_runner.train_after_llm_guidance(absolute_target)

            actual = td3_runner.total_steps - prev_steps
            print(f"RL finished: ran {actual} steps (total = {td3_runner.total_steps})")

            
            if td3_runner.total_steps >= TARGET_TOTAL_STEPS:
                break
        else:
            
            print("Coordinator chose not to train RL this round.")
            
            pass

    print("\nTraining completed.")
    print("Final specs:", td3_runner.env.real_specs)
    print("Final params:", td3_runner.env.param_values)

    # Pareto
    try:
        result = utils.plotting.solutions2pareto(
            "./solutions/solutions.csv", run_id="2025-11-09--test"
        )
        
        if isinstance(result, tuple) and len(result) == 2:
            pareto_df, pareto_csv = result
            print(pareto_df[["Specs", "Reward"]])
            print("Pareto CSV:", pareto_csv)
    except Exception as e:
        print("solutions2pareto failed:", repr(e))


if __name__ == "__main__":
    main()

