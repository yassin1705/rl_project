"""
Evaluation and Comparison Script for CPU Scheduling with DVFS
Compares PPO agent against baseline policies with detailed metrics:
- Mean waiting time per task class
- Maximum waiting time
- Total energy consumption
- Task service distribution
- Constraint violation rate
"""

import sys
import os
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch
import config
from envs.cpu_scheduler_env import CPUSchedulerEnv
from agents.ppo_agent import PPOAgent
from baselines.fifo_fixed_freq import (
    FIFOFixedFreqAgent,
    FIFOMaxFreqAgent,
    FIFOMinFreqAgent
)


def evaluate_agent_detailed(
    agent,
    env,
    num_episodes: int,
    agent_name: str,
    verbose: bool = False
) -> Dict[str, float]:
    """
    Evaluate an agent with detailed metrics tracking
    
    Args:
        agent: Agent to evaluate
        env: Environment to evaluate on
        num_episodes: Number of evaluation episodes
        agent_name: Name of the agent for reporting
        verbose: Whether to print per-episode info
    
    Returns:
        Dictionary of detailed evaluation metrics
    """
    episode_rewards = []
    episode_lengths = []
    
    # Aggregate metrics across episodes
    all_waiting_times = {cat: [] for cat in config.TASK_CATEGORIES}
    all_max_waiting_times = []
    all_total_energy = []
    all_tasks_served = {cat: 0 for cat in config.TASK_CATEGORIES}
    all_starvation_violations = 0
    total_episodes = 0
    
    for ep in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0.0
        episode_length = 0
        
        while not done:
            action_mask = env.get_action_mask()
            
            # Get action from agent
            if hasattr(agent, 'select_action'):
                action, _, _ = agent.select_action(state, action_mask, deterministic=True)
            else:
                action = agent.get_action(state, action_mask)
            
            state, reward, done, info = env.step(action)
            episode_reward += reward
            episode_length += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Get detailed metrics from environment
        metrics = env.get_episode_metrics()
        
        # Collect waiting times per category
        for cat in config.TASK_CATEGORIES:
            cat_waiting = env.metrics_tracker.waiting_times.get(cat, [])
            all_waiting_times[cat].extend(cat_waiting)
        
        # Max waiting time for this episode
        if env.metrics_tracker.all_waiting_times:
            all_max_waiting_times.append(max(env.metrics_tracker.all_waiting_times))
        
        # Total energy for this episode
        all_total_energy.append(sum(env.metrics_tracker.energy_consumptions))
        
        # Tasks served per category
        for cat in config.TASK_CATEGORIES:
            all_tasks_served[cat] += env.metrics_tracker.tasks_served.get(cat, 0)
        
        # Starvation violations
        all_starvation_violations += env.metrics_tracker.starvation_violations
        total_episodes += 1
        
        if verbose:
            print(f"  Episode {ep + 1}/{num_episodes}: "
                  f"Reward = {episode_reward:.2f}, Length = {episode_length}")
    
    # Calculate aggregated results
    results = {
        'agent_name': agent_name,
        'num_episodes': num_episodes,
        
        # Reward metrics
        'mean_reward': float(np.mean(episode_rewards)),
        'std_reward': float(np.std(episode_rewards)),
        'min_reward': float(np.min(episode_rewards)),
        'max_reward': float(np.max(episode_rewards)),
        
        # Episode length
        'mean_length': float(np.mean(episode_lengths)),
    }
    
    # Mean waiting time per task class
    for cat in config.TASK_CATEGORIES:
        if all_waiting_times[cat]:
            results[f'mean_wait_{cat}'] = float(np.mean(all_waiting_times[cat]))
            results[f'max_wait_{cat}'] = float(np.max(all_waiting_times[cat]))
        else:
            results[f'mean_wait_{cat}'] = 0.0
            results[f'max_wait_{cat}'] = 0.0
    
    # Overall waiting time
    all_waits = []
    for cat in all_waiting_times:
        all_waits.extend(all_waiting_times[cat])
    
    if all_waits:
        results['mean_waiting_time'] = float(np.mean(all_waits))
        results['max_waiting_time'] = float(np.max(all_waits))
    else:
        results['mean_waiting_time'] = 0.0
        results['max_waiting_time'] = 0.0
    
    # Maximum waiting time (worst case across episodes)
    if all_max_waiting_times:
        results['global_max_waiting_time'] = float(max(all_max_waiting_times))
        results['avg_max_waiting_time'] = float(np.mean(all_max_waiting_times))
    else:
        results['global_max_waiting_time'] = 0.0
        results['avg_max_waiting_time'] = 0.0
    
    # Total energy consumption
    results['total_energy'] = float(sum(all_total_energy))
    results['mean_energy_per_episode'] = float(np.mean(all_total_energy))
    
    # Task service distribution
    total_tasks = sum(all_tasks_served.values())
    results['total_tasks_served'] = total_tasks
    
    for cat in config.TASK_CATEGORIES:
        results[f'tasks_served_{cat}'] = all_tasks_served[cat]
        if total_tasks > 0:
            results[f'service_ratio_{cat}'] = float(all_tasks_served[cat] / total_tasks)
        else:
            results[f'service_ratio_{cat}'] = 0.0
    
    # Constraint violation rate
    results['starvation_violations'] = all_starvation_violations
    results['violation_rate'] = float(all_starvation_violations / total_episodes) if total_episodes > 0 else 0.0
    
    return results


def load_ppo_agent(model_path: str, device: str = 'auto') -> PPOAgent:
    """Load a trained PPO agent from checkpoint"""
    agent = PPOAgent(device=device)
    agent.load(model_path)
    return agent


def compare_agents(
    num_episodes: int = 50,
    ppo_model_path: str = None,
    seed: int = None,
    device: str = 'auto',
    verbose: int = 1,
    save_results: bool = True
) -> Dict[str, Dict]:
    """
    Compare PPO agent against baseline agents with detailed metrics
    """
    if seed is None:
        seed = config.RANDOM_SEED
    
    # Create evaluation environment
    env = CPUSchedulerEnv(seed=seed)
    
    # Initialize agents
    agents = {}
    
    # Baseline agents
    agents['FIFO_Fixed_2.5GHz'] = FIFOFixedFreqAgent(frequency_index=2)
    agents['FIFO_Max_3.1GHz'] = FIFOMaxFreqAgent()
    agents['FIFO_Min_1.5GHz'] = FIFOMinFreqAgent()
    
    # PPO agent
    if ppo_model_path is None:
        default_paths = [
            os.path.join(config.MODEL_DIR, 'ppo_newconfig_best.pt'),
            os.path.join(config.MODEL_DIR, 'test_run_best.pt'),
            os.path.join(config.MODEL_DIR, 'ppo_best.pt'),
        ]
        for path in default_paths:
            if os.path.exists(path):
                ppo_model_path = path
                break
    
    if ppo_model_path and os.path.exists(ppo_model_path):
        try:
            ppo_agent = load_ppo_agent(ppo_model_path, device)
            agents['PPO'] = ppo_agent
            if verbose >= 1:
                print(f"Loaded PPO model from: {ppo_model_path}")
        except Exception as e:
            print(f"Warning: Could not load PPO model: {e}")
    else:
        if verbose >= 1:
            print("No PPO model found. Training a new one first...")
            print("Run: python experiments/train_ppo.py")
    
    # Evaluate all agents
    all_results = {}
    
    print("\n" + "=" * 80)
    print("DETAILED EVALUATION COMPARISON")
    print("=" * 80)
    print(f"Episodes per agent: {num_episodes}")
    print(f"Environment seed: {seed}")
    print(f"Max waiting time limit: {config.MAX_WAITING_TIME}")
    print("-" * 80)
    
    for name, agent in agents.items():
        if verbose >= 1:
            print(f"\nEvaluating: {name}...")
        
        results = evaluate_agent_detailed(
            agent=agent,
            env=env,
            num_episodes=num_episodes,
            agent_name=name,
            verbose=(verbose >= 2)
        )
        
        all_results[name] = results
        
        if verbose >= 1:
            print(f"  Mean Reward: {results['mean_reward']:8.2f} ± {results['std_reward']:.2f}")
    
    # Print detailed comparison
    print_comparison_table(all_results)
    
    # Save results
    if save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = os.path.join(config.RESULTS_DIR, f"detailed_comparison_{timestamp}.json")
        
        Path(config.RESULTS_DIR).mkdir(parents=True, exist_ok=True)
        
        with open(results_path, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'num_episodes': num_episodes,
                'seed': seed,
                'config': {
                    'workload_sizes': config.WORKLOAD_SIZES,
                    'frequency_levels': config.FREQUENCY_LEVELS,
                    'task_arrival_rate': config.TASK_ARRIVAL_RATE,
                    'max_waiting_time': config.MAX_WAITING_TIME,
                    'beta': config.BETA
                },
                'results': all_results
            }, f, indent=2)
        
        if verbose >= 1:
            print(f"\nResults saved to: {results_path}")
    
    return all_results


def print_comparison_table(results: Dict[str, Dict]):
    """Print formatted comparison tables"""
    agents = list(results.keys())
    
    # Header
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    
    # 1. Reward Comparison
    print("\n┌" + "─" * 78 + "┐")
    print("│ REWARD METRICS" + " " * 63 + "│")
    print("├" + "─" * 25 + "┬" + "─" * 52 + "┤")
    header = "│ {:23s} │".format("Agent")
    for a in agents:
        header += " {:>12s}".format(a[:12])
    print(header + " │")
    print("├" + "─" * 25 + "┼" + "─" * 52 + "┤")
    
    print("│ {:23s} │".format("Mean Reward") + 
          "".join(" {:>12.2f}".format(results[a]['mean_reward']) for a in agents) + " │")
    print("│ {:23s} │".format("Std Reward") + 
          "".join(" {:>12.2f}".format(results[a]['std_reward']) for a in agents) + " │")
    print("└" + "─" * 25 + "┴" + "─" * 52 + "┘")
    
    # 2. Waiting Time Metrics
    print("\n┌" + "─" * 78 + "┐")
    print("│ WAITING TIME METRICS (per task class)" + " " * 39 + "│")
    print("├" + "─" * 25 + "┬" + "─" * 52 + "┤")
    header = "│ {:23s} │".format("Metric")
    for a in agents:
        header += " {:>12s}".format(a[:12])
    print(header + " │")
    print("├" + "─" * 25 + "┼" + "─" * 52 + "┤")
    
    print("│ {:23s} │".format("Mean Wait (Overall)") + 
          "".join(" {:>12.2f}".format(results[a]['mean_waiting_time']) for a in agents) + " │")
    
    for cat in config.TASK_CATEGORIES:
        print("│ {:23s} │".format(f"Mean Wait ({cat})") + 
              "".join(" {:>12.2f}".format(results[a].get(f'mean_wait_{cat}', 0)) for a in agents) + " │")
    
    print("├" + "─" * 25 + "┼" + "─" * 52 + "┤")
    print("│ {:23s} │".format("Max Waiting Time") + 
          "".join(" {:>12.2f}".format(results[a]['max_waiting_time']) for a in agents) + " │")
    print("│ {:23s} │".format("Global Max Wait") + 
          "".join(" {:>12.2f}".format(results[a]['global_max_waiting_time']) for a in agents) + " │")
    print("└" + "─" * 25 + "┴" + "─" * 52 + "┘")
    
    # 3. Energy Consumption
    print("\n┌" + "─" * 78 + "┐")
    print("│ ENERGY CONSUMPTION" + " " * 58 + "│")
    print("├" + "─" * 25 + "┬" + "─" * 52 + "┤")
    header = "│ {:23s} │".format("Metric")
    for a in agents:
        header += " {:>12s}".format(a[:12])
    print(header + " │")
    print("├" + "─" * 25 + "┼" + "─" * 52 + "┤")
    
    print("│ {:23s} │".format("Total Energy") + 
          "".join(" {:>12.2f}".format(results[a]['total_energy']) for a in agents) + " │")
    print("│ {:23s} │".format("Mean Energy/Episode") + 
          "".join(" {:>12.2f}".format(results[a]['mean_energy_per_episode']) for a in agents) + " │")
    print("└" + "─" * 25 + "┴" + "─" * 52 + "┘")
    
    # 4. Task Service Distribution
    print("\n┌" + "─" * 78 + "┐")
    print("│ TASK SERVICE DISTRIBUTION" + " " * 51 + "│")
    print("├" + "─" * 25 + "┬" + "─" * 52 + "┤")
    header = "│ {:23s} │".format("Task Class")
    for a in agents:
        header += " {:>12s}".format(a[:12])
    print(header + " │")
    print("├" + "─" * 25 + "┼" + "─" * 52 + "┤")
    
    print("│ {:23s} │".format("Total Tasks Served") + 
          "".join(" {:>12d}".format(results[a]['total_tasks_served']) for a in agents) + " │")
    
    for cat in config.TASK_CATEGORIES:
        ratio_str = "".join(" {:>11.1%}".format(results[a].get(f'service_ratio_{cat}', 0)) for a in agents)
        print("│ {:23s} │".format(f"{cat} Ratio") + ratio_str + "  │")
    
    print("└" + "─" * 25 + "┴" + "─" * 52 + "┘")
    
    # 5. Constraint Violations
    print("\n┌" + "─" * 78 + "┐")
    print("│ CONSTRAINT VIOLATIONS" + " " * 55 + "│")
    print("├" + "─" * 25 + "┬" + "─" * 52 + "┤")
    header = "│ {:23s} │".format("Metric")
    for a in agents:
        header += " {:>12s}".format(a[:12])
    print(header + " │")
    print("├" + "─" * 25 + "┼" + "─" * 52 + "┤")
    
    print("│ {:23s} │".format("Starvation Violations") + 
          "".join(" {:>12d}".format(results[a]['starvation_violations']) for a in agents) + " │")
    print("│ {:23s} │".format("Violation Rate") + 
          "".join(" {:>11.1%}".format(results[a]['violation_rate']) for a in agents) + "  │")
    print("└" + "─" * 25 + "┴" + "─" * 52 + "┘")
    
    # Summary ranking
    print("\n" + "=" * 80)
    print("RANKING (by Mean Reward - higher is better)")
    print("=" * 80)
    
    sorted_agents = sorted(results.items(), key=lambda x: x[1]['mean_reward'], reverse=True)
    
    print(f"\n{'Rank':<6}{'Agent':<25}{'Mean Reward':<15}{'Violations':<12}{'Energy':<15}")
    print("-" * 75)
    
    for rank, (name, r) in enumerate(sorted_agents, 1):
        print(f"{rank:<6}{name:<25}{r['mean_reward']:>12.2f}   "
              f"{r['starvation_violations']:>10}   {r['total_energy']:>12.2f}")
    
    # PPO improvement
    if 'PPO' in results:
        print("\n" + "-" * 75)
        print("PPO Improvement over Baselines:")
        ppo = results['PPO']
        
        for name, r in results.items():
            if name != 'PPO':
                reward_imp = ppo['mean_reward'] - r['mean_reward']
                energy_diff = ppo['total_energy'] - r['total_energy']
                wait_diff = ppo['mean_waiting_time'] - r['mean_waiting_time']
                
                pct = (reward_imp / abs(r['mean_reward'])) * 100 if r['mean_reward'] != 0 else 0
                
                print(f"  vs {name}:")
                print(f"      Reward: {reward_imp:+.2f} ({pct:+.1f}%)")
                print(f"      Energy: {energy_diff:+.2f}")
                print(f"      Wait Time: {wait_diff:+.2f}")
    
    print("=" * 80)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Evaluate and compare agents with detailed metrics'
    )
    
    parser.add_argument('--episodes', type=int, default=50,
                        help='Number of evaluation episodes per agent')
    parser.add_argument('--ppo-model', type=str, default=None,
                        help='Path to trained PPO model')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for evaluation')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['cpu', 'cuda', 'auto'])
    parser.add_argument('--verbose', type=int, default=1,
                        choices=[0, 1, 2])
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save results to file')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    results = compare_agents(
        num_episodes=args.episodes,
        ppo_model_path=args.ppo_model,
        seed=args.seed,
        device=args.device,
        verbose=args.verbose,
        save_results=not args.no_save
    )
    
    print("\n✓ Evaluation completed!")
