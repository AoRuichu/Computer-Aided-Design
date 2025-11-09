import matplotlib.pyplot as plt 
import numpy as np
from typing import List, Dict
import pandas as pd
import os
import matplotlib

def plotLearning(scores, run_id):
    filename = f'./output_figs/{run_id}/average_score.png'
    N = len(scores)
    # Calculate the running average of scores
    running_avg = np.zeros(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t-N):(t+1)])

    # change the font size
    plt.rcParams.update({'font.size': 18})
    plt.ylabel('Episode Scores')       
    plt.xlabel('# Episode')                     
    plt.plot(range(N), running_avg, label='Running Avg')
    plt.plot(range(N), scores, alpha=0.5, label='Scores')  # Optionally overlay the raw scores
    plt.legend()
    plt.grid('on')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_running_maximum(data, run_id):
    file_name = f'./output_figs/{run_id}/max_reward.png'
    running_max = float('-inf')  # Initialize running maximum to negative infinity
    running_max_values = []

    for value in data:
        if value > running_max:
            running_max = value
        running_max_values.append(running_max)

    # change the font size
    plt.rcParams.update({'font.size': 18})
    plt.plot(running_max_values)
    plt.xlabel('# Simulation')
    plt.ylabel('Maximum FoM Reached')
    # plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()
    


def is_pareto_efficient(costs: np.ndarray) -> np.ndarray:
    """Return a boolean mask of Pareto-efficient points."""
    if costs.size == 0:
        return np.array([], dtype=bool)
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            # Mark dominated points
            is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)
            is_efficient[i] = True
    return is_efficient


def plot_pareto_front(solutions: List[Dict[str, float]], fname: str, show_all=False):
    """Plot Pareto front and return Pareto mask (True/False per solution)."""
    if not solutions:
        print("⚠️ No solutions found. Skipping plot.")
        return np.array([], dtype=bool)

    # Extract objectives and scale
    current_scaled = [sol['current'] * 1e6 for sol in solutions]   # μA
    area_scaled = [sol['area'] * 1e12 for sol in solutions]        # μm²

    combined = np.vstack((area_scaled, current_scaled)).T
    pareto_mask = is_pareto_efficient(combined)

    pareto_points = combined[pareto_mask]
    pareto_area = pareto_points[:, 0]
    pareto_current = pareto_points[:, 1]

    # --- Plot ---
    plt.rcParams.update({'font.size': 18})
    fig, ax = plt.subplots(figsize=(8, 6))
    if show_all:
        ax.scatter(area_scaled, current_scaled, c='gray', alpha=0.5, s=60, label='All Solutions')
    ax.scatter(pareto_area, pareto_current, c='blue', s=100, label='Pareto Front')
    sorted_idx = np.argsort(pareto_area)
    ax.plot(pareto_area[sorted_idx], pareto_current[sorted_idx], 'b--', linewidth=2)

    ax.set_xlabel('Active Area (μm²)', fontsize=18)
    ax.set_ylabel('Total Current (μA)', fontsize=18)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)

    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.tight_layout()
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()

    return pareto_mask


def solutions2pareto(csv_fname, run_id, show_all=True):
    """Compute Pareto front, plot it, and return Pareto subset safely."""
    plot_fname = f'./output_figs/{run_id}/pareto.png'
    os.makedirs(os.path.dirname(plot_fname), exist_ok=True)

    if not os.path.exists(csv_fname):
        raise FileNotFoundError(f"❌ CSV not found: {csv_fname}")

    df = pd.read_csv(csv_fname)
    if 'Specs' not in df.columns:
        raise ValueError("❌ CSV must contain a 'Specs' column with circuit metrics.")

    # Parse 'Specs' as dicts
    solutions = []
    for x in df['Specs']:
        try:
            solutions.append(eval(x))
        except Exception:
            solutions.append({})

    pareto_mask = plot_pareto_front(solutions, plot_fname, show_all)

    # Ensure the mask matches dataframe length
    if len(pareto_mask) != len(df):
        print("⚠️ Pareto mask length mismatch; skipping filter.")
        pareto_mask = np.zeros(len(df), dtype=bool)

    pareto_df = df[pareto_mask].copy().reset_index(drop=True)

    # Save only Pareto rows
    pareto_csv = f'./solutions/{run_id}/pareto_solutions.csv'
    pareto_df.to_csv(pareto_csv, index=False)

    print(f"✅ Pareto front plotted: {plot_fname}")
    print(f"✅ Pareto solutions saved: {pareto_csv}")
    print(f"✅ Number of Pareto-optimal designs: {len(pareto_df)}")

    return pareto_df, pareto_csv