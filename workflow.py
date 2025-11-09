from google import genai
from google.genai import types
from utils.save_response import writeMd
from dotenv import load_dotenv
from pydantic import BaseModel, create_model, Field
from datetime import datetime
import numpy as np
import json
import os
import yaml
from toolify import Args, TD3Runner, CircuitEvaluator

env_path = "./local.env"
load_dotenv(dotenv_path=env_path)

# The client gets the API key from the environment variable `GEMINI_API_KEY`.
client = genai.Client()

def compute_spec_delta(current_specs: dict, target_specs: dict) -> dict:
    """compute the delta between current specs and target specs, used for feedback to LLM"""
    delta = {}
    for k, v in target_specs.items():
        if k not in current_specs:
            continue
        cur = current_specs[k]
        try:
            delta[k] = float(cur) - float(v)
        except Exception:
            delta[k] = None
    return delta


def compact_yaml(d: dict, keep_keys=None, precision=6):
    # keep only keys you care about and round small numbers
    if keep_keys is None: keep_keys = list(d.keys())
    slim = {}
    for k in keep_keys:
        v = d.get(k, None)
        if isinstance(v, float):
            slim[k] = float(f"{v:.{precision}g}")
        else:
            slim[k] = v
    import yaml
    return yaml.dump(slim, sort_keys=False)

def sanitize_numpy(obj):
    """Convert NumPy types in a dict to plain Python floats/ints recursively."""
    if isinstance(obj, dict):
        return {k: sanitize_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_numpy(x) for x in obj]
    elif isinstance(obj, np.generic):  # NumPy scalar types
        return float(obj)
    else:
        return obj


## Complete the code!
def ask_gemini_to_optimize_with_feedback(
    best_params: dict,
    specs_before: dict,
    target_specs: dict,
    param_ranges: dict,
    iteration: int,
    image_path: str = None,
    last_llm_specs: dict = None,
):
    # --- Compute deltas ---
    delta_specs = compute_spec_delta(specs_before, target_specs)

    if last_llm_specs:
        last_delta = compute_spec_delta(last_llm_specs, target_specs)
        last_feedback = yaml.dump(last_delta, sort_keys=False)
    else:
        last_feedback = "N/A"

    # --- Range ---
    range_summary = {
        k: {
            "min": float(v["min"]),
            "max": float(v["max"]),
        }
        for k, v in param_ranges.items()
    }
    
    
    prompt = f"""
        You are an expert analog circuit designer optimizing a Two-Stage Miller op-amp.

        Analyze the current circuit performance and parameters.
        Suggest a refined set of 20 transistor sizing parameters that move performance closer to the target,
        while keeping parameter values within their specified bounds. The suggested parameters will be used in a TD3 reinforcement
        learning environment.

        Respond with a short explanation, then output ONLY a valid JSON object
        with the 20 updated parameters under key "optimized_parameters".All the parameters could not be none and follow the physical rules.
        You need to keep iterate until the all hard constraints are meet before you give me the final json-formed parameters.
        
        You can decide how long the step need to run for updating the RL memory based on your confidence level.
        ---
        **Previous improvement feedback:**
        {last_feedback}

        **Current parameters:**
        {yaml.dump(best_params, sort_keys=False)}

        **Measured specs (before):**
        {yaml.dump(sanitize_numpy(specs_before), sort_keys=False)}

        **Delta to targets (current - target):**
        {yaml.dump(delta_specs, sort_keys=False)}

        **Target specs:**
        {yaml.dump(target_specs, sort_keys=False)}

        **Parameter ranges (must stay within):**
        {yaml.dump(range_summary, sort_keys=False)}

        ---

        Rules:
        - Adjust each parameter by no more than ±20% unless necessary.
        - Use 45nm PDK, Keep all transistor lengths above 45 nm.
        - If The UGBW is found 0, meaning that the circuit is not amplifier. Re-design the circuit based on the targets.
        - Avoid extreme bias currents.
        - Output both reasoning and final JSON.
        """

   # --- Build multimodal content ---
    contents = [types.Part.from_text(text=prompt)]
    if image_path and os.path.exists(image_path):
        with open(image_path, "rb") as f:
            img_bytes = f.read()
        contents.append(types.Part.from_bytes(data=img_bytes, mime_type="image/png"))

    # --- Call Gemini ---
    try:
        response = client.models.generate_content(
            model="gemini-2.5-pro",
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                thinking_config=types.ThinkingConfig(
                    thinking_budget=20_000, include_thoughts=True
                ),
                temperature=0.3,
                max_output_tokens=40_000,
                seed=42,
                candidate_count=1,
                system_instruction=(
                    "You are a senior analog IC designer. "
                    "You optimize op-amp sizing parameters under process constraints "
                    "and should output both reasoning and valid JSON in the field 'optimized_parameters'."
                ),
            ),
            contents=contents,
        )
    except Exception as e:
        print(f"⚠️ Gemini API call failed: {e}")
        return None
    
    

    # ---  Markdown  ---
    try:
        writeMd(response, filename=f"llm_response_iter{iteration}.md")
    except Exception as e:
        print(f"⚠️ Failed to write markdown: {e}")

    # --- extract json ---
    text = getattr(response, "text", "")
    refined_params = None
    if text:
        try:
            
            start = text.find("{")
            end = text.rfind("}") + 1
            json_str = text[start:end]
            json_obj = json.loads(json_str)
            refined_params = (
                json_obj.get("optimized_parameters", json_obj)
                if isinstance(json_obj, dict)
                else None
            )
        except Exception as e:
            print(f"⚠️ JSON parse failed: {e}")

    
    log_dir = "./backup/llm_logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"iteration_{iteration}.yaml")

    with open(log_file, "w", encoding="utf-8") as f:
        yaml.dump(
            {
                "iteration": iteration,
                "prompt": prompt,
                "response_text": text[:3000] if text else "N/A",
                "refined_params": refined_params,
                "specs_before": sanitize_numpy(specs_before),
                "target_specs": target_specs,
                "delta_specs": delta_specs,
            },
            f,
            sort_keys=False,
        )

    print(f"✅ LLM guidance saved to {log_file}")

    return refined_params


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
                    w = 500,
                    pi_lr = 1e-4,
                    q_lr = 1e-3,
                    replay_size = 1_000_000,
                    cuda = False,
                    no_discretize = False,
                    run_id = datetime.now().strftime('%Y-%m-%d--%H-%M-%S'))
    td3_runner = TD3Runner(args)
    circuit_evaluator_llm = CircuitEvaluator(circuit_name='TwoStage', run_id='llm_workflow', simulator='ngspice')
    circuit_evaluator = td3_runner.env

    # Warmup exploration
    

    for i in range(3): # 5 iterations of LLM-guided optimization
        print("\nStarting warmup exploration...")
        td3_runner.warmup_exploration()
        print(f"\nWarmup exploration completed with {td3_runner.total_steps} steps.")
        
        print("\n==================================")
        print(f"\nAsk LLM for guide, itr = {i}")
        print("\n==================================")
        best_design = td3_runner.find_best_design_in_pool()
        best_params = best_design['params']
        specs_before = best_design['specs']
        last_llm_specs = {}

        llm_parameters = ask_gemini_to_optimize_with_feedback(
            best_params=best_params,
            specs_before=specs_before,
            target_specs=circuit_evaluator.dict_targets,
            param_ranges=circuit_evaluator.param_ranges,
            iteration=3 ,
            image_path="circuit_image.png",
            last_llm_specs=last_llm_specs
        )

        if llm_parameters:
            real_specs_after = circuit_evaluator.simulate(llm_parameters)
            td3_runner.add_llm_experience_to_RL_memory(specs_before, llm_parameters, real_specs_after)
            last_llm_specs = real_specs_after  # used for feedback in next iteration
       
        # Train after llm guideance
        budget_steps = 1000  # Example budget steps
        print(f"Starting training for {budget_steps} budget steps...")  
        td3_runner.train_after_llm_guidance(budget_steps)

    print("\nTraining completed.")
    print("Final evaluation of the trained model:")
    print(f"\n real spec: {td3_runner.env.real_specs}")
    print(f"\n real param values: {td3_runner.env.param_values}")
    print(f"\nSaving the trained models in ./models/td3_{args.run_id}/")
    if not os.path.exists(f'./models/td3_{args.run_id}/'):
        os.makedirs(f'./models/td3_{args.run_id}/')
    td3_runner.agent.save(f'./models/td3_{args.run_id}/td3_model')
    print("Models saved.")
    print("Exiting now.")



if __name__ == "__main__":
    main()