"""
CPU Scheduler Environment with DVFS
Reinforcement Learning Environment for CPU Scheduling with Dynamic Voltage and Frequency Scaling
"""

import numpy as np
from typing import Tuple, Dict, Optional, List
import config
from utils.task_generator import TaskGenerator, MultiQueueSystem, Task
from utils.state_utils import StateBuilder, CoreManager
from utils.reward_utils import RewardCalculator, MetricsTracker


class CPUSchedulerEnv:
    """
    RL Environment for CPU Scheduling with DVFS

    State: [Q_Low_len, Q_Low_age, Q_Med_len, Q_Med_age, Q_Heavy_len, Q_Heavy_age,
            FreeCores, AvgRemain, MaxRemain]

    Action: (Task Category, Frequency Level) flattened to single integer

    Reward: w_c * A_c - beta * E_chosen
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize CPU Scheduler Environment

        Args:
            seed: Random seed for reproducibility
        """
        # Set random seed
        if seed is not None:
            np.random.seed(seed)
            self.seed = seed
        else:
            self.seed = config.RANDOM_SEED

        # Initialize components
        self.task_generator = TaskGenerator(seed=self.seed)
        self.queue_system = MultiQueueSystem()
        self.core_manager = CoreManager(num_cores=config.NUM_CORES)
        self.state_builder = StateBuilder()
        self.reward_calculator = RewardCalculator()
        self.metrics_tracker = MetricsTracker()

        # Environment configuration
        self.state_dim = config.STATE_DIM
        self.action_dim = config.ACTION_DIM
        self.max_decisions = config.MAX_SCHEDULING_DECISIONS
        self.max_waiting_time = config.MAX_WAITING_TIME

        # Normalization constants (simple division by max values)
        self.norm_queue_length = 20.0  # Max queue length
        self.norm_age = float(config.MAX_WAITING_TIME)  # Max waiting time
        self.norm_cores = float(config.NUM_CORES)  # Max free cores
        self.norm_time = 20.0  # Normalization for remaining times

        # Episode tracking
        self.current_time = 0
        self.decision_count = 0
        self.episode_reward = 0.0
        self.done = False

        # Task tracking for metrics
        self.completed_tasks = []
        self.task_completion_info = []  # (task, waiting_time, energy, frequency)

        print(f"CPUSchedulerEnv initialized with {config.NUM_CORES} cores")
        print(f"State dim: {self.state_dim}, Action dim: {self.action_dim}")

    def reset(self) -> np.ndarray:
        """
        Reset environment to initial state

        Returns:
            Initial normalized state
        """
        # Reset all components
        self.task_generator.reset()
        self.queue_system.clear_all()
        self.core_manager.reset_all()
        self.metrics_tracker.reset()

        # Reset episode tracking
        self.current_time = 0
        self.decision_count = 0
        self.episode_reward = 0.0
        self.done = False
        self.completed_tasks = []
        self.task_completion_info = []

        # Generate initial tasks
        initial_tasks = self.task_generator.generate_tasks(self.current_time)
        self.queue_system.add_tasks(initial_tasks)

        # Get initial state
        state = self._get_state()

        return state

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one scheduling decision

        Args:
            action: Action index (task_category * num_frequencies + frequency_level)

        Returns:
            Tuple of (next_state, reward, done, info)
        """
        if self.done:
            raise RuntimeError("Episode is done. Call reset() to start new episode.")

        # Decode action
        task_idx, freq_idx = config.get_indices_from_action(action)
        chosen_category = config.TASK_CATEGORIES[task_idx]
        chosen_frequency = config.FREQUENCY_LEVELS[freq_idx]

        # Get the queue for chosen category
        queue = self.queue_system.get_queue(chosen_category)

        # Check if queue has tasks
        if queue.is_empty():
            # Invalid action - queue is empty, give small penalty
            reward = -1.0
            self.episode_reward += reward

            # Still need to advance time and check for new tasks
            self._advance_time()
            next_state = self._get_state()

            # Check termination
            self.decision_count += 1
            self.done = self._check_done()

            info = {
                'invalid_action': True,
                'chosen_category': chosen_category,
                'message': 'Queue was empty'
            }

            return next_state, reward, self.done, info

        # Valid action - dequeue task and assign to free core
        task = queue.dequeue()
        free_core = self.core_manager.get_free_core()

        if free_core is None:
            raise RuntimeError("No free core available but step was called")

        # Assign task to core
        free_core.assign_task(task, chosen_frequency)

        # Calculate reward
        task_age = task.get_age(self.current_time)
        reward, reward_components = self.reward_calculator.calculate_reward(
            chosen_category=chosen_category,
            task_age=task_age,
            frequency=chosen_frequency,
            workload=task.workload
        )

        self.episode_reward += reward

        # Advance time until next decision point
        self._advance_time()

        # Get next state
        next_state = self._get_state()

        # Check termination conditions
        self.decision_count += 1
        self.done = self._check_done()

        # Prepare info dict
        info = {
            'invalid_action': False,
            'chosen_category': chosen_category,
            'chosen_frequency': chosen_frequency,
            'task_age': task_age,
            'reward_components': reward_components,
            'decision_count': self.decision_count,
            'current_time': self.current_time
        }

        return next_state, reward, self.done, info

    def _advance_time(self):
        """
        Advance time until next decision point
        A decision point occurs when: at least one core is free AND at least one queue is non-empty
        """
        while True:
            self.current_time += 1

            # Generate new tasks
            new_tasks = self.task_generator.generate_tasks(self.current_time)
            self.queue_system.add_tasks(new_tasks)

            # Update all cores (decrement remaining time)
            completed = self.core_manager.update_all(time_elapsed=1.0)

            # Track completed tasks for metrics
            for core_id, task, frequency in completed:
                waiting_time = task.get_age(self.current_time)
                energy = task.get_energy_consumption(frequency)

                self.completed_tasks.append(task)
                self.task_completion_info.append((task, waiting_time, energy, frequency))

                # Record in metrics tracker
                self.metrics_tracker.record_task_completion(
                    category=task.category,
                    waiting_time=waiting_time,
                    energy=energy,
                    frequency=frequency
                )

            # Check if we've reached a decision point
            has_free_core = self.core_manager.has_free_core()
            has_tasks = self.queue_system.has_tasks()

            if has_free_core and has_tasks:
                # Decision point reached
                break

            # Check if episode should end (no more work to do)
            if not has_tasks and self.core_manager.get_num_free_cores() == config.NUM_CORES:
                # All queues empty and all cores idle - episode naturally ends
                self.done = True
                break

            # Safety check: don't run forever
            if self.current_time > self.max_decisions * 100:
                self.done = True
                break

    def _get_state(self) -> np.ndarray:
        """
        Construct and normalize state vector

        Returns:
            Normalized state vector
        """
        # Get queue information
        queue_info = {}
        for category in config.TASK_CATEGORIES:
            queue = self.queue_system.get_queue(category)
            queue_len = queue.size()
            head_age = queue.get_head_age(self.current_time)
            queue_info[category] = (queue_len, head_age)

        # Get core information
        core_info = self.core_manager.get_aggregated_info()

        # Build raw state
        state = self.state_builder.build_state(queue_info, core_info)

        # Normalize state (simple division by max values)
        normalized_state = self._normalize_state(state)

        return normalized_state

    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        """
        Normalize state by dividing by maximum values

        State format: [Q_Low_len, Q_Low_age, Q_Med_len, Q_Med_age, Q_Heavy_len, Q_Heavy_age,
                       FreeCores, AvgRemain, MaxRemain]

        Args:
            state: Raw state vector

        Returns:
            Normalized state vector
        """
        normalized = np.zeros_like(state, dtype=np.float32)

        # Normalize queue lengths (indices 0, 2, 4)
        normalized[0] = state[0] / self.norm_queue_length
        normalized[2] = state[2] / self.norm_queue_length
        normalized[4] = state[4] / self.norm_queue_length

        # Normalize ages (indices 1, 3, 5)
        normalized[1] = state[1] / self.norm_age
        normalized[3] = state[3] / self.norm_age
        normalized[5] = state[5] / self.norm_age

        # Normalize free cores (index 6)
        normalized[6] = state[6] / self.norm_cores

        # Normalize remaining times (indices 7, 8)
        normalized[7] = state[7] / self.norm_time
        normalized[8] = state[8] / self.norm_time

        return normalized

    def _check_done(self) -> bool:
        """
        Check if episode should terminate

        Returns:
            True if episode should end, False otherwise
        """
        # Check decision count limit
        if self.decision_count >= self.max_decisions:
            return True

        # Check starvation constraint
        max_age = self.queue_system.get_max_age(self.current_time)
        if max_age > self.max_waiting_time:
            # Starvation violation - apply penalty and terminate
            starvation_penalty = self.reward_calculator.calculate_starvation_penalty()
            self.episode_reward += starvation_penalty
            self.metrics_tracker.record_starvation_violation()
            return True

        # Already set to done in _advance_time if no more work
        return self.done

    def get_episode_metrics(self) -> Dict[str, float]:
        """
        Get metrics for the completed episode

        Returns:
            Dictionary of episode metrics
        """
        metrics = self.metrics_tracker.get_metrics()  # existing metrics

        # Add total reward and decision info
        metrics['total_reward'] = self.episode_reward
        metrics['decision_count'] = self.decision_count
        metrics['episode_length'] = self.current_time

        # Compute total energy consumed and mean waiting time
        if self.task_completion_info:
            energies = [info[2] for info in self.task_completion_info]  # energy
            waiting_times = [info[1] for info in self.task_completion_info]  # waiting time
            metrics['total_energy'] = sum(energies)
            metrics['mean_waiting_time'] = np.mean(waiting_times)
            metrics['num_completed_tasks'] = len(self.task_completion_info)
        else:
            metrics['total_energy'] = 0.0
            metrics['mean_waiting_time'] = 0.0
            metrics['num_completed_tasks'] = 0

        return metrics

    def get_action_mask(self) -> np.ndarray:
        """
        Get mask for valid actions (1 = valid, 0 = invalid)

        Returns:
            Binary mask array of shape (action_dim,)
        """
        mask = np.zeros(self.action_dim, dtype=np.float32)

        # Check which queues have tasks
        for task_idx, category in enumerate(config.TASK_CATEGORIES):
            queue = self.queue_system.get_queue(category)
            if not queue.is_empty():
                # All frequency levels are valid for this category
                for freq_idx in range(config.NUM_FREQUENCIES):
                    action = config.get_action_from_indices(task_idx, freq_idx)
                    mask[action] = 1.0

        return mask

    def render(self, mode: str = 'human'):
        """
        Render current environment state

        Args:
            mode: Rendering mode ('human' for console output)
        """
        if mode == 'human':
            print(f"\n{'=' * 60}")
            print(f"Time: {self.current_time}, Decisions: {self.decision_count}")
            print(f"{'=' * 60}")

            # Queue status
            print("\nQueue Status:")
            for category in config.TASK_CATEGORIES:
                queue = self.queue_system.get_queue(category)
                head_age = queue.get_head_age(self.current_time)
                print(f"  {category:8s}: {queue.size():3d} tasks, head age: {head_age:3d}")

            # Core status
            print(f"\nCore Status: {self.core_manager}")
            free = self.core_manager.get_num_free_cores()
            busy = config.NUM_CORES - free
            print(f"  Free: {free}, Busy: {busy}")

            # Episode stats
            print(f"\nEpisode Reward: {self.episode_reward:.2f}")
            print(f"Tasks Completed: {len(self.completed_tasks)}")
            print(f"{'=' * 60}\n")

    def close(self):
        """Clean up environment resources"""
        pass

    @property
    def observation_space_shape(self) -> Tuple[int]:
        """Get observation space shape"""
        return (self.state_dim,)

    @property
    def action_space_size(self) -> int:
        """Get action space size"""
        return self.action_dim


if __name__ == "__main__":
    print("Testing CPUSchedulerEnv...")

    # Create environment
    env = CPUSchedulerEnv(seed=42)

    # Test reset
    print("\n1. Testing reset:")
    state = env.reset()
    print(f"Initial state shape: {state.shape}")
    print(f"Initial state: {state}")

    # Render initial state
    env.render()

    # Test action mask
    print("\n2. Testing action mask:")
    mask = env.get_action_mask()
    valid_actions = np.where(mask > 0)[0]
    print(f"Valid actions: {valid_actions[:10]}...")  # Show first 10
    print(f"Total valid actions: {np.sum(mask)}")

    # Run a few steps
    print("\n3. Running episode steps:")
    for i in range(5):
        # Get valid actions
        mask = env.get_action_mask()
        valid_actions = np.where(mask > 0)[0]

        if len(valid_actions) == 0:
            print(f"Step {i}: No valid actions available")
            break

        # Choose random valid action
        action = np.random.choice(valid_actions)

        # Take step
        next_state, reward, done, info = env.step(action)

        print(f"\nStep {i + 1}:")
        print(f"  Action: {action} -> Category: {info['chosen_category']}, Freq: {info.get('chosen_frequency', 'N/A')}")
        print(f"  Reward: {reward:.2f}")
        print(f"  Done: {done}")

        if done:
            print("Episode finished!")
            break

    # Get episode metrics
    print("\n4. Episode metrics:")
    metrics = env.get_episode_metrics()
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")

    print("\nEnvironment test completed!")