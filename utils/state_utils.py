"""
State Representation Utilities for CPU Scheduling with DVFS
Constructs and manages the state vector for RL agent
"""

import numpy as np
from typing import Dict, List, Tuple
import config


class StateBuilder:
    """
    Builds state representation from queue and core information

    State vector format:
    [Q_Low_len, Q_Low_age, Q_Med_len, Q_Med_age, Q_Heavy_len, Q_Heavy_age,
     FreeCores, AvgRemain, MaxRemain]
    """

    def __init__(self):
        """Initialize state builder"""
        self.state_dim = config.STATE_DIM

    def build_state(self,
                    queue_info: Dict[str, Tuple[int, int]],
                    core_info: Dict[str, float]) -> np.ndarray:
        """
        Build state vector from queue and core information

        Args:
            queue_info: Dictionary mapping category to (length, head_age)
            core_info: Dictionary with 'free_cores', 'avg_remain', 'max_remain'

        Returns:
            State vector as numpy array
        """
        state = []

        # Queue-level features (length and head age for each category)
        for category in config.TASK_CATEGORIES:
            queue_len, head_age = queue_info.get(category, (0, 0))
            state.append(queue_len)
            state.append(head_age)

        # Core-level aggregated features
        state.append(core_info.get('free_cores', 0))
        state.append(core_info.get('avg_remain', 0.0))
        state.append(core_info.get('max_remain', 0.0))

        return np.array(state, dtype=np.float32)

    def extract_queue_info(self, queue_system, current_time: int) -> Dict[str, Tuple[int, int]]:
        """
        Extract queue information for state construction

        Args:
            queue_system: MultiQueueSystem object
            current_time: Current simulation time

        Returns:
            Dictionary mapping category to (queue_length, head_task_age)
        """
        queue_info = {}

        for category in config.TASK_CATEGORIES:
            queue = queue_system.get_queue(category)
            queue_len = queue.size()
            head_age = queue.get_head_age(current_time)
            queue_info[category] = (queue_len, head_age)

        return queue_info

    def extract_core_info(self, cores: List) -> Dict[str, float]:
        """
        Extract aggregated core information for state construction

        Args:
            cores: List of Core objects

        Returns:
            Dictionary with aggregated core statistics
        """
        free_cores = sum(1 for core in cores if core.is_free())

        # Calculate remaining times for busy cores
        remaining_times = [core.get_remaining_time() for core in cores if not core.is_free()]

        if remaining_times:
            avg_remain = np.mean(remaining_times)
            max_remain = np.max(remaining_times)
        else:
            avg_remain = 0.0
            max_remain = 0.0

        return {
            'free_cores': free_cores,
            'avg_remain': avg_remain,
            'max_remain': max_remain
        }

    def get_state_dim(self) -> int:
        """Get state dimension"""
        return self.state_dim


class Core:
    """
    Represents a single CPU core
    Tracks current task and remaining execution time
    """

    def __init__(self, core_id: int):
        """
        Initialize core

        Args:
            core_id: Unique identifier for the core
        """
        self.core_id = core_id
        self.current_task = None
        self.remaining_time = 0.0
        self.frequency = 0.0

        # Statistics
        self.total_tasks_executed = 0
        self.total_busy_time = 0.0
        self.total_idle_time = 0.0

    def assign_task(self, task, frequency: float):
        """
        Assign a task to this core

        Args:
            task: Task object to execute
            frequency: Frequency level for execution
        """
        if not self.is_free():
            raise ValueError(f"Core {self.core_id} is already busy")

        self.current_task = task
        self.frequency = frequency
        self.remaining_time = task.get_execution_time(frequency)

    def update(self, time_elapsed: float = 1.0) -> bool:
        """
        Update core state (decrement remaining time)

        Args:
            time_elapsed: Amount of time that has passed

        Returns:
            True if task completed, False otherwise
        """
        if self.is_free():
            self.total_idle_time += time_elapsed
            return False

        self.remaining_time -= time_elapsed
        self.total_busy_time += time_elapsed

        if self.remaining_time <= 0:
            # Task completed
            self.total_tasks_executed += 1
            completed_task = self.current_task
            self.current_task = None
            self.remaining_time = 0.0
            self.frequency = 0.0
            return True

        return False

    def is_free(self) -> bool:
        """Check if core is free"""
        return self.current_task is None

    def get_remaining_time(self) -> float:
        """Get remaining execution time for current task"""
        return self.remaining_time

    def get_current_task(self):
        """Get currently executing task"""
        return self.current_task

    def get_stats(self) -> Dict:
        """Get core statistics"""
        total_time = self.total_busy_time + self.total_idle_time
        utilization = self.total_busy_time / total_time if total_time > 0 else 0.0

        return {
            'core_id': self.core_id,
            'tasks_executed': self.total_tasks_executed,
            'busy_time': self.total_busy_time,
            'idle_time': self.total_idle_time,
            'utilization': utilization
        }

    def reset(self):
        """Reset core state"""
        self.current_task = None
        self.remaining_time = 0.0
        self.frequency = 0.0

    def __repr__(self):
        if self.is_free():
            return f"Core({self.core_id}, FREE)"
        else:
            return f"Core({self.core_id}, BUSY, remaining={self.remaining_time:.2f})"


class CoreManager:
    """
    Manages multiple CPU cores
    Provides aggregated view and core allocation
    """

    def __init__(self, num_cores: int = None):
        """
        Initialize core manager

        Args:
            num_cores: Number of cores to manage
        """
        self.num_cores = num_cores if num_cores else config.NUM_CORES
        self.cores = [Core(i) for i in range(self.num_cores)]

    def get_free_core(self):
        """
        Get a free core if available

        Returns:
            Free Core object or None if all busy
        """
        for core in self.cores:
            if core.is_free():
                return core
        return None

    def has_free_core(self) -> bool:
        """Check if any core is free"""
        return any(core.is_free() for core in self.cores)

    def get_num_free_cores(self) -> int:
        """Get number of free cores"""
        return sum(1 for core in self.cores if core.is_free())

    def update_all(self, time_elapsed: float = 1.0) -> List[Tuple]:
        """
        Update all cores and collect completed tasks

        Args:
            time_elapsed: Amount of time that has passed

        Returns:
            List of (core_id, completed_task, frequency) tuples
        """
        completed = []

        for core in self.cores:
            # Capture task and frequency BEFORE update (update clears them on completion)
            task = core.get_current_task()
            freq = core.frequency
            task_completed = core.update(time_elapsed)
            if task_completed and task is not None:
                completed.append((core.core_id, task, freq))

        return completed

    def get_aggregated_info(self) -> Dict[str, float]:
        """
        Get aggregated core information for state construction

        Returns:
            Dictionary with free_cores, avg_remain, max_remain
        """
        free_cores = self.get_num_free_cores()

        remaining_times = [
            core.get_remaining_time()
            for core in self.cores
            if not core.is_free()
        ]

        if remaining_times:
            avg_remain = np.mean(remaining_times)
            max_remain = np.max(remaining_times)
        else:
            avg_remain = 0.0
            max_remain = 0.0

        return {
            'free_cores': free_cores,
            'avg_remain': avg_remain,
            'max_remain': max_remain
        }

    def get_all_stats(self) -> Dict:
        """Get statistics for all cores"""
        stats = {
            'num_cores': self.num_cores,
            'cores': [core.get_stats() for core in self.cores]
        }

        # Calculate aggregate statistics
        total_tasks = sum(core.total_tasks_executed for core in self.cores)
        avg_utilization = np.mean([
            core.get_stats()['utilization']
            for core in self.cores
        ])

        stats['total_tasks_executed'] = total_tasks
        stats['avg_utilization'] = avg_utilization

        return stats

    def reset_all(self):
        """Reset all cores"""
        for core in self.cores:
            core.reset()

    def __repr__(self):
        free = self.get_num_free_cores()
        busy = self.num_cores - free
        return f"CoreManager(cores={self.num_cores}, free={free}, busy={busy})"


if __name__ == "__main__":
    print("Testing State Utilities...")

    # Test Core
    print("\n1. Testing Core:")
    core = Core(0)
    print(f"Initial state: {core}")
    print(f"Is free: {core.is_free()}")


    # Create a mock task
    class MockTask:
        def __init__(self, workload):
            self.workload = workload
            self.category = 'Low'

        def get_execution_time(self, freq):
            return self.workload / freq


    task = MockTask(100)
    core.assign_task(task, 2.0)
    print(f"After assignment: {core}")
    print(f"Remaining time: {core.get_remaining_time()}")

    # Update core
    completed = core.update(25.0)
    print(f"After update (25s): {core}, completed={completed}")
    completed = core.update(30.0)
    print(f"After update (30s): {core}, completed={completed}")

    # Test CoreManager
    print("\n2. Testing CoreManager:")
    manager = CoreManager(num_cores=4)
    print(f"Initial state: {manager}")

    # Assign tasks to cores
    for i in range(3):
        free_core = manager.get_free_core()
        if free_core:
            free_core.assign_task(MockTask(100 * (i + 1)), 2.0)
            print(f"Assigned task to core {free_core.core_id}")

    print(f"\nAfter assignments: {manager}")
    print(f"Aggregated info: {manager.get_aggregated_info()}")

    # Update all cores
    completed = manager.update_all(30.0)
    print(f"\nAfter update: {manager}")
    print(f"Completed tasks: {len(completed)}")

    # Test StateBuilder
    print("\n3. Testing StateBuilder:")
    builder = StateBuilder()

    # Mock queue info
    queue_info = {
        'Low': (3, 10),
        'Medium': (2, 15),
        'Heavy': (1, 20)
    }

    # Get core info from manager
    core_info = manager.get_aggregated_info()

    # Build state
    state = builder.build_state(queue_info, core_info)
    print(f"State vector shape: {state.shape}")
    print(f"State vector: {state}")
    print(f"\nState interpretation:")
    print(f"  Q_Low: length={state[0]}, age={state[1]}")
    print(f"  Q_Med: length={state[2]}, age={state[3]}")
    print(f"  Q_Heavy: length={state[4]}, age={state[5]}")
    print(f"  Free cores: {state[6]}")
    print(f"  Avg remaining: {state[7]:.2f}")
    print(f"  Max remaining: {state[8]:.2f}")