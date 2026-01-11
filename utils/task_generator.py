"""
Task Generator for CPU Scheduling with DVFS
Generates stochastic task arrivals based on Poisson process
"""

import numpy as np
from typing import List, Tuple, Dict
import config


class Task:
    """Represents a single task"""

    def __init__(self, task_id: int, category: str, arrival_time: int):
        """
        Initialize a task

        Args:
            task_id: Unique identifier for the task
            category: Task category ('Low', 'Medium', 'Heavy')
            arrival_time: Time step when task arrived
        """
        self.task_id = task_id
        self.category = category
        self.arrival_time = arrival_time
        self.workload = config.WORKLOAD_SIZES[category]

    def get_execution_time(self, frequency: float) -> float:
        """Calculate execution time at given frequency"""
        return self.workload / frequency

    def get_energy_consumption(self, frequency: float) -> float:
        """Calculate energy consumption at given frequency (quadratic)"""
        return (frequency ** 2) * self.workload

    def get_age(self, current_time: int) -> int:
        """Get the age (waiting time) of the task"""
        return current_time - self.arrival_time

    def __repr__(self):
        return f"Task({self.task_id}, {self.category}, t={self.arrival_time})"


class TaskGenerator:
    """Generates tasks according to a Poisson arrival process"""

    def __init__(self, arrival_rate: float = None,
                 category_probs: Dict[str, float] = None,
                 seed: int = None):
        """
        Initialize task generator

        Args:
            arrival_rate: Lambda parameter for Poisson process (tasks per time step)
            category_probs: Probability distribution over task categories
            seed: Random seed for reproducibility
        """
        self.arrival_rate = arrival_rate if arrival_rate else config.TASK_ARRIVAL_RATE
        self.category_probs = category_probs if category_probs else config.TASK_CATEGORY_PROBS

        # Validate probabilities
        prob_sum = sum(self.category_probs.values())
        assert abs(prob_sum - 1.0) < 1e-6, f"Category probabilities must sum to 1.0, got {prob_sum}"

        # Set random seed
        if seed is not None:
            np.random.seed(seed)

        # Task counter for unique IDs
        self.task_counter = 0

        # Extract categories and probabilities
        self.categories = list(self.category_probs.keys())
        self.probs = list(self.category_probs.values())

    def generate_tasks(self, current_time: int) -> List[Task]:
        """
        Generate tasks for the current time step using Poisson process

        Args:
            current_time: Current simulation time step

        Returns:
            List of newly generated tasks
        """
        # Sample number of tasks arriving at this time step from Poisson distribution
        num_tasks = np.random.poisson(self.arrival_rate)

        tasks = []
        for _ in range(num_tasks):
            # Sample task category according to probability distribution
            category = np.random.choice(self.categories, p=self.probs)

            # Create task with unique ID
            task = Task(
                task_id=self.task_counter,
                category=category,
                arrival_time=current_time
            )
            tasks.append(task)
            self.task_counter += 1

        return tasks

    def reset(self):
        """Reset the task generator"""
        self.task_counter = 0

    def get_stats(self) -> Dict[str, int]:
        """Get statistics about generated tasks"""
        return {
            'total_tasks_generated': self.task_counter
        }


class TaskQueue:
    """FIFO queue for tasks of a specific category"""

    def __init__(self, category: str):
        """
        Initialize task queue

        Args:
            category: Task category this queue handles
        """
        self.category = category
        self.queue: List[Task] = []
        self.total_enqueued = 0
        self.total_dequeued = 0

    def enqueue(self, task: Task):
        """Add task to the queue"""
        assert task.category == self.category, \
            f"Task category {task.category} doesn't match queue category {self.category}"
        self.queue.append(task)
        self.total_enqueued += 1

    def dequeue(self) -> Task:
        """Remove and return the head task from the queue"""
        if self.is_empty():
            raise ValueError(f"Cannot dequeue from empty {self.category} queue")
        task = self.queue.pop(0)
        self.total_dequeued += 1
        return task

    def peek(self) -> Task:
        """Return the head task without removing it"""
        if self.is_empty():
            return None
        return self.queue[0]

    def is_empty(self) -> bool:
        """Check if queue is empty"""
        return len(self.queue) == 0

    def size(self) -> int:
        """Return current queue length"""
        return len(self.queue)

    def get_head_age(self, current_time: int) -> int:
        """Get age of the head task, 0 if empty"""
        if self.is_empty():
            return 0
        return self.queue[0].get_age(current_time)

    def get_max_age(self, current_time: int) -> int:
        """Get maximum age among all tasks in queue"""
        if self.is_empty():
            return 0
        return max(task.get_age(current_time) for task in self.queue)

    def clear(self):
        """Clear all tasks from the queue"""
        self.queue.clear()

    def get_stats(self) -> Dict[str, int]:
        """Get queue statistics"""
        return {
            'category': self.category,
            'current_size': len(self.queue),
            'total_enqueued': self.total_enqueued,
            'total_dequeued': self.total_dequeued
        }

    def __repr__(self):
        return f"Queue({self.category}, size={len(self.queue)})"


class MultiQueueSystem:
    """System of multiple FIFO queues, one per task category"""

    def __init__(self):
        """Initialize queue system with one queue per category"""
        self.queues = {
            category: TaskQueue(category)
            for category in config.TASK_CATEGORIES
        }

    def add_tasks(self, tasks: List[Task]):
        """Add multiple tasks to their respective queues"""
        for task in tasks:
            self.queues[task.category].enqueue(task)

    def add_task(self, task: Task):
        """Add a single task to its queue"""
        self.queues[task.category].enqueue(task)

    def get_queue(self, category: str) -> TaskQueue:
        """Get queue for a specific category"""
        return self.queues[category]

    def has_tasks(self) -> bool:
        """Check if any queue has tasks"""
        return any(not q.is_empty() for q in self.queues.values())

    def get_total_tasks(self) -> int:
        """Get total number of tasks across all queues"""
        return sum(q.size() for q in self.queues.values())

    def get_max_age(self, current_time: int) -> int:
        """Get maximum age across all queues"""
        if not self.has_tasks():
            return 0
        return max(q.get_head_age(current_time) for q in self.queues.values() if not q.is_empty())

    def clear_all(self):
        """Clear all queues"""
        for q in self.queues.values():
            q.clear()

    def get_stats(self) -> Dict[str, Dict]:
        """Get statistics for all queues"""
        return {
            category: queue.get_stats()
            for category, queue in self.queues.items()
        }

    def __repr__(self):
        queue_sizes = {cat: q.size() for cat, q in self.queues.items()}
        return f"MultiQueueSystem({queue_sizes})"


if __name__ == "__main__":
    # Test task generator
    print("Testing Task Generator...")
    generator = TaskGenerator(seed=42)

    print("\nGenerating tasks for 5 time steps:")
    for t in range(5):
        tasks = generator.generate_tasks(t)
        print(f"Time {t}: Generated {len(tasks)} tasks - {[task.category for task in tasks]}")

    # Test queue system
    print("\n\nTesting Queue System...")
    queue_system = MultiQueueSystem()

    # Generate and add tasks
    generator.reset()
    for t in range(3):
        tasks = generator.generate_tasks(t)
        queue_system.add_tasks(tasks)

    print(f"\nQueue system: {queue_system}")
    print(f"Total tasks: {queue_system.get_total_tasks()}")
    print(f"\nQueue statistics:")
    for cat, stats in queue_system.get_stats().items():
        print(f"  {cat}: {stats}")

    # Test dequeue
    print("\n\nTesting dequeue:")
    if not queue_system.get_queue('Low').is_empty():
        task = queue_system.get_queue('Low').dequeue()
        print(f"Dequeued: {task}")
        print(f"Queue system after dequeue: {queue_system}")