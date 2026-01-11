"""
FIFO Fixed Frequency Baseline Agent
Implements a simple First-In-First-Out scheduling policy with a fixed frequency level.
Used as a baseline for comparison with RL-based approaches.
"""

import sys
import os
import numpy as np
from typing import Dict, Tuple, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class FIFOFixedFreqAgent:
    """
    FIFO Scheduling with Fixed Frequency Baseline
    
    Policy:
    - Always selects the queue with the oldest waiting task (FIFO across queues)
    - Always uses a fixed frequency level (configurable)
    
    This represents a simple, non-adaptive scheduling approach commonly used
    in traditional operating systems.
    """
    
    def __init__(
        self,
        frequency_index: int = None,
        priority_order: list = None,
        seed: int = None
    ):
        """
        Initialize FIFO Fixed Frequency Agent
        
        Args:
            frequency_index: Index of frequency level to use (default: middle frequency)
            priority_order: Priority order for queues when ages are equal
                           (default: ['Heavy', 'Medium', 'Low'])
            seed: Random seed (not used, for API compatibility)
        """
        if frequency_index is None:
            frequency_index = config.FIFO_FIXED_FREQ_INDEX
        
        if priority_order is None:
            # Default: prioritize heavier tasks when ages are equal
            priority_order = ['Heavy', 'Medium', 'Low']
        
        self.frequency_index = frequency_index
        self.frequency = config.FREQUENCY_LEVELS[frequency_index]
        self.priority_order = priority_order
        self.name = f"FIFO_FixedFreq_{self.frequency}GHz"
        
        # Statistics
        self.total_decisions = 0
        self.category_counts = {cat: 0 for cat in config.TASK_CATEGORIES}
        
        print(f"FIFO Fixed Frequency Agent initialized")
        print(f"  Frequency: {self.frequency} GHz (index {frequency_index})")
        print(f"  Priority: {priority_order}")
    
    def reset_stats(self):
        """Reset agent statistics"""
        self.total_decisions = 0
        self.category_counts = {cat: 0 for cat in config.TASK_CATEGORIES}
    
    def select_action(
        self,
        state: np.ndarray,
        action_mask: np.ndarray = None,
        deterministic: bool = True
    ) -> Tuple[int, float, float]:
        """
        Select action using FIFO policy with fixed frequency
        
        The agent selects the queue with the oldest task (highest head age).
        Ties are broken by priority order (Heavy > Medium > Low by default).
        
        Args:
            state: Current state (normalized)
            action_mask: Valid action mask (1=valid, 0=invalid)
            deterministic: Ignored (always deterministic)
        
        Returns:
            action: Selected action index
            log_prob: Always 0.0 (deterministic policy)
            value: Always 0.0 (no value estimation)
        """
        # Parse state to get queue ages
        # State format: [Q_Low_len, Q_Low_age, Q_Med_len, Q_Med_age, 
        #                Q_Heavy_len, Q_Heavy_age, FreeCores, AvgRemain, MaxRemain]
        
        queue_info = {
            'Low': {'length': state[0], 'age': state[1]},
            'Medium': {'length': state[2], 'age': state[3]},
            'Heavy': {'length': state[4], 'age': state[5]}
        }
        
        # Find valid queues (non-empty based on action mask)
        valid_categories = []
        for task_idx, category in enumerate(config.TASK_CATEGORIES):
            # Check if any action for this category is valid
            for freq_idx in range(config.NUM_FREQUENCIES):
                action = config.get_action_from_indices(task_idx, freq_idx)
                if action_mask is not None and action_mask[action] > 0:
                    valid_categories.append(category)
                    break
        
        if not valid_categories:
            # No valid actions - shouldn't happen in normal operation
            # Return first action as fallback
            return 0, 0.0, 0.0
        
        # Select category with oldest task (FIFO)
        # Use priority order to break ties
        best_category = None
        best_age = -1
        
        for category in self.priority_order:
            if category in valid_categories:
                age = queue_info[category]['age']
                if age > best_age:
                    best_age = age
                    best_category = category
        
        # If no category selected (shouldn't happen), use first valid
        if best_category is None:
            best_category = valid_categories[0]
        
        # Get action index
        task_idx = config.TASK_CATEGORIES.index(best_category)
        action = config.get_action_from_indices(task_idx, self.frequency_index)
        
        # Update statistics
        self.total_decisions += 1
        self.category_counts[best_category] += 1
        
        # Return action with dummy log_prob and value
        return action, 0.0, 0.0
    
    def evaluate(
        self,
        env,
        num_episodes: int = None,
        verbose: bool = False
    ) -> Dict[str, float]:
        """
        Evaluate agent on environment
        
        Args:
            env: Environment to evaluate on
            num_episodes: Number of evaluation episodes
            verbose: Whether to print per-episode info
        
        Returns:
            Evaluation statistics
        """
        if num_episodes is None:
            num_episodes = config.NUM_EVAL_EPISODES
        
        self.reset_stats()
        
        episode_rewards = []
        episode_lengths = []
        episode_metrics = []
        
        for ep in range(num_episodes):
            state = env.reset()
            done = False
            episode_reward = 0.0
            episode_length = 0
            
            while not done:
                action_mask = env.get_action_mask()
                action, _, _ = self.select_action(state, action_mask)
                state, reward, done, info = env.step(action)
                episode_reward += reward
                episode_length += 1
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            # Get detailed metrics
            metrics = env.get_episode_metrics()
            episode_metrics.append(metrics)
            
            if verbose:
                print(f"Episode {ep + 1}: Reward = {episode_reward:.2f}, Length = {episode_length}")
        
        # Aggregate metrics
        results = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'total_decisions': self.total_decisions,
            'agent_name': self.name
        }
        
        # Add category distribution
        total = sum(self.category_counts.values())
        if total > 0:
            for cat in config.TASK_CATEGORIES:
                results[f'{cat}_ratio'] = self.category_counts[cat] / total
        
        # Aggregate episode metrics
        if episode_metrics:
            for key in episode_metrics[0].keys():
                if isinstance(episode_metrics[0][key], (int, float)):
                    values = [m[key] for m in episode_metrics]
                    results[f'mean_{key}'] = np.mean(values)
        
        return results
    
    def get_stats(self) -> Dict:
        """Get agent statistics"""
        return {
            'name': self.name,
            'frequency': self.frequency,
            'total_decisions': self.total_decisions,
            'category_counts': self.category_counts.copy()
        }


class FIFOMaxFreqAgent(FIFOFixedFreqAgent):
    """
    FIFO with Maximum Frequency (Race-to-Halt)
    
    Uses the highest available frequency to complete tasks as fast as possible.
    This is a "race-to-halt" strategy that prioritizes latency over energy.
    """
    
    def __init__(self, seed: int = None):
        # Use highest frequency index
        max_freq_idx = len(config.FREQUENCY_LEVELS) - 1
        super().__init__(frequency_index=max_freq_idx, seed=seed)
        self.name = f"FIFO_MaxFreq_{self.frequency}GHz"


class FIFOMinFreqAgent(FIFOFixedFreqAgent):
    """
    FIFO with Minimum Frequency (Energy Saver)
    
    Uses the lowest available frequency to minimize energy consumption.
    This prioritizes energy over latency.
    """
    
    def __init__(self, seed: int = None):
        # Use lowest frequency index
        super().__init__(frequency_index=0, seed=seed)
        self.name = f"FIFO_MinFreq_{self.frequency}GHz"


if __name__ == "__main__":
    print("Testing FIFO Fixed Frequency Baseline...")
    
    from envs.cpu_scheduler_env import CPUSchedulerEnv
    
    # Create environment
    env = CPUSchedulerEnv(seed=42)
    
    # Test different FIFO variants
    agents = [
        FIFOFixedFreqAgent(frequency_index=2),  # Middle frequency
        FIFOMaxFreqAgent(),  # Max frequency (race-to-halt)
        FIFOMinFreqAgent(),  # Min frequency (energy saver)
    ]
    
    print("\n" + "=" * 60)
    print("Evaluating FIFO Baseline Agents")
    print("=" * 60)
    
    for agent in agents:
        print(f"\nEvaluating: {agent.name}")
        results = agent.evaluate(env, num_episodes=10)
        
        print(f"  Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"  Mean Length: {results['mean_length']:.1f}")
        print(f"  Reward Range: [{results['min_reward']:.2f}, {results['max_reward']:.2f}]")
        
        # Print category distribution
        print(f"  Category Distribution:")
        for cat in config.TASK_CATEGORIES:
            ratio = results.get(f'{cat}_ratio', 0)
            print(f"    {cat}: {ratio:.1%}")
    
    print("\n✓ FIFO baseline tests completed!")
