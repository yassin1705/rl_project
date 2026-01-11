"""
Reward Calculation Utilities for CPU Scheduling with DVFS
Implements the reward function and related utilities
"""

import numpy as np
from typing import Dict, Tuple
import config


class RewardCalculator:
    """Calculates rewards for scheduling decisions"""

    def __init__(self, beta: float = None):
        """
        Initialize reward calculator

        Args:
            beta: Energy-latency trade-off parameter
        """
        self.beta = beta if beta is not None else config.BETA
        self.anti_starvation_weights = config.ANTI_STARVATION_WEIGHTS

        # Statistics tracking
        self.total_rewards = []
        self.latency_rewards = []
        self.energy_penalties = []

    def calculate_reward(self,
                         chosen_category: str,
                         task_age: int,
                         frequency: float,
                         workload: float) -> Tuple[float, Dict[str, float]]:
        """
        Calculate reward for a scheduling decision

        Reward formula: R = w_c * A_c - beta * E_chosen
        Where:
            - w_c: anti-starvation weight for category c
            - A_c: age of the head task in chosen queue
            - beta: energy-latency trade-off parameter
            - E_chosen: energy consumption = f^2 * W_c

        Args:
            chosen_category: Selected task category
            task_age: Age (waiting time) of the selected task
            frequency: Selected frequency level
            workload: Workload size of the task

        Returns:
            Tuple of (total_reward, reward_components_dict)
        """
        # Latency reward component (positive, rewards serving older tasks)
        weight = self.anti_starvation_weights[chosen_category]
        latency_reward = weight * task_age

        # Energy consumption (quadratic in frequency)
        energy_consumption = (frequency ** 2) * workload

        # Energy penalty component (negative)
        energy_penalty = self.beta * energy_consumption

        # Total reward
        total_reward = latency_reward - energy_penalty

        # Store components for analysis
        reward_components = {
            'latency_reward': latency_reward,
            'energy_penalty': energy_penalty,
            'energy_consumption': energy_consumption,
            'total_reward': total_reward,
            'task_age': task_age,
            'frequency': frequency,
            'category': chosen_category
        }

        # Track statistics
        self.total_rewards.append(total_reward)
        self.latency_rewards.append(latency_reward)
        self.energy_penalties.append(energy_penalty)

        return total_reward, reward_components

    def calculate_starvation_penalty(self) -> float:
        """Return the starvation penalty for constraint violation"""
        return config.STARVATION_PENALTY

    def get_statistics(self) -> Dict[str, float]:
        """Get reward statistics"""
        if not self.total_rewards:
            return {
                'mean_reward': 0.0,
                'mean_latency_reward': 0.0,
                'mean_energy_penalty': 0.0,
                'total_episodes': 0
            }

        return {
            'mean_reward': np.mean(self.total_rewards),
            'std_reward': np.std(self.total_rewards),
            'mean_latency_reward': np.mean(self.latency_rewards),
            'mean_energy_penalty': np.mean(self.energy_penalties),
            'total_episodes': len(self.total_rewards)
        }

    def reset_statistics(self):
        """Reset accumulated statistics"""
        self.total_rewards.clear()
        self.latency_rewards.clear()
        self.energy_penalties.clear()


class RewardShaper:
    """Applies reward shaping techniques for better learning"""

    def __init__(self, shaping_type: str = 'none'):
        """
        Initialize reward shaper

        Args:
            shaping_type: Type of shaping ('none', 'normalize', 'clip')
        """
        self.shaping_type = shaping_type
        self.reward_history = []
        self.running_mean = 0.0
        self.running_std = 1.0
        self.alpha = 0.01  # Update rate for running statistics

    def shape(self, reward: float) -> float:
        """
        Apply reward shaping

        Args:
            reward: Raw reward value

        Returns:
            Shaped reward
        """
        self.reward_history.append(reward)

        if self.shaping_type == 'none':
            return reward

        elif self.shaping_type == 'normalize':
            # Update running statistics
            if len(self.reward_history) > 1:
                self.running_mean = (1 - self.alpha) * self.running_mean + self.alpha * reward
                self.running_std = (1 - self.alpha) * self.running_std + \
                                   self.alpha * abs(reward - self.running_mean)

            # Normalize
            if self.running_std > 1e-8:
                return (reward - self.running_mean) / (self.running_std + 1e-8)
            return reward

        elif self.shaping_type == 'clip':
            # Clip to reasonable range
            return np.clip(reward, -100, 100)

        else:
            raise ValueError(f"Unknown shaping type: {self.shaping_type}")

    def reset(self):
        """Reset shaper statistics"""
        self.reward_history.clear()
        self.running_mean = 0.0
        self.running_std = 1.0


class MetricsTracker:
    """Tracks evaluation metrics for the scheduling policy"""

    def __init__(self):
        """Initialize metrics tracker"""
        self.reset()

    def reset(self):
        """Reset all tracked metrics"""
        # Waiting time metrics per category
        self.waiting_times = {cat: [] for cat in config.TASK_CATEGORIES}
        self.all_waiting_times = []

        # Energy metrics
        self.energy_consumptions = []

        # Task service distribution
        self.tasks_served = {cat: 0 for cat in config.TASK_CATEGORIES}

        # Frequency usage
        self.frequency_usage = {freq: 0 for freq in config.FREQUENCY_LEVELS}

        # Constraint violations
        self.starvation_violations = 0

    def record_task_completion(self,
                               category: str,
                               waiting_time: float,
                               energy: float,
                               frequency: float):
        """
        Record a completed task

        Args:
            category: Task category
            waiting_time: Time task spent waiting
            energy: Energy consumed
            frequency: Frequency used
        """
        self.waiting_times[category].append(waiting_time)
        self.all_waiting_times.append(waiting_time)
        self.energy_consumptions.append(energy)
        self.tasks_served[category] += 1
        self.frequency_usage[frequency] = self.frequency_usage.get(frequency, 0) + 1

    def record_starvation_violation(self):
        """Record a starvation constraint violation"""
        self.starvation_violations += 1

    def get_metrics(self) -> Dict[str, float]:
        """
        Get comprehensive metrics

        Returns:
            Dictionary of metrics
        """
        metrics = {}

        # Overall waiting time metrics
        if self.all_waiting_times:
            metrics['mean_waiting_time'] = np.mean(self.all_waiting_times)
            metrics['max_waiting_time'] = np.max(self.all_waiting_times)
            metrics['std_waiting_time'] = np.std(self.all_waiting_times)
        else:
            metrics['mean_waiting_time'] = 0.0
            metrics['max_waiting_time'] = 0.0
            metrics['std_waiting_time'] = 0.0

        # Per-category waiting times
        for cat in config.TASK_CATEGORIES:
            if self.waiting_times[cat]:
                metrics[f'mean_waiting_time_{cat}'] = np.mean(self.waiting_times[cat])
            else:
                metrics[f'mean_waiting_time_{cat}'] = 0.0

        # Energy metrics
        if self.energy_consumptions:
            metrics['total_energy'] = np.sum(self.energy_consumptions)
            metrics['mean_energy_per_task'] = np.mean(self.energy_consumptions)
        else:
            metrics['total_energy'] = 0.0
            metrics['mean_energy_per_task'] = 0.0

        # Task service distribution
        total_served = sum(self.tasks_served.values())
        for cat in config.TASK_CATEGORIES:
            if total_served > 0:
                metrics[f'service_ratio_{cat}'] = self.tasks_served[cat] / total_served
            else:
                metrics[f'service_ratio_{cat}'] = 0.0

        metrics['total_tasks_served'] = total_served

        # Constraint violations
        metrics['starvation_violations'] = self.starvation_violations

        return metrics

    def print_summary(self):
        """Print a formatted summary of metrics"""
        metrics = self.get_metrics()

        print("\n" + "=" * 60)
        print("PERFORMANCE METRICS SUMMARY")
        print("=" * 60)

        print("\nWaiting Time Metrics:")
        print(f"  Mean Waiting Time: {metrics['mean_waiting_time']:.2f}")
        print(f"  Max Waiting Time:  {metrics['max_waiting_time']:.2f}")
        print(f"  Std Waiting Time:  {metrics['std_waiting_time']:.2f}")

        print("\nPer-Category Waiting Times:")
        for cat in config.TASK_CATEGORIES:
            print(f"  {cat:8s}: {metrics[f'mean_waiting_time_{cat}']:.2f}")

        print("\nEnergy Metrics:")
        print(f"  Total Energy:      {metrics['total_energy']:.2f}")
        print(f"  Mean Energy/Task:  {metrics['mean_energy_per_task']:.2f}")

        print("\nTask Service Distribution:")
        for cat in config.TASK_CATEGORIES:
            ratio = metrics[f'service_ratio_{cat}']
            print(f"  {cat:8s}: {ratio:.2%} ({self.tasks_served[cat]} tasks)")

        print(f"\nTotal Tasks Served: {metrics['total_tasks_served']}")
        print(f"Starvation Violations: {metrics['starvation_violations']}")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    print("Testing Reward Calculator...")

    # Initialize calculator
    calc = RewardCalculator(beta=0.1)

    # Test reward calculation
    print("\nTest 1: Low workload, old task, low frequency")
    reward, components = calc.calculate_reward(
        chosen_category='Low',
        task_age=20,
        frequency=1.5,
        workload=100
    )
    print(f"Reward: {reward:.2f}")
    print(f"Components: {components}")

    print("\nTest 2: Heavy workload, new task, high frequency")
    reward, components = calc.calculate_reward(
        chosen_category='Heavy',
        task_age=5,
        frequency=3.0,
        workload=600
    )
    print(f"Reward: {reward:.2f}")
    print(f"Components: {components}")

    # Test metrics tracker
    print("\n\nTesting Metrics Tracker...")
    tracker = MetricsTracker()

    # Simulate some task completions
    tracker.record_task_completion('Low', 10, 225, 1.5)
    tracker.record_task_completion('Medium', 25, 1350, 3.0)
    tracker.record_task_completion('Heavy', 15, 900, 2.0)

    tracker.print_summary()