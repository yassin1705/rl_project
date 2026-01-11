"""
Logger for CPU Scheduling with DVFS RL Project
Handles logging, metrics tracking, and results saving
"""

import os
import json
import csv
import numpy as np
from typing import Dict, List, Any
from datetime import datetime
import config


class Logger:
    """
    Main logger for training and evaluation
    Tracks metrics, saves results, and provides visualization data
    """

    def __init__(self, experiment_name: str = None, log_dir: str = None):
        """
        Initialize logger

        Args:
            experiment_name: Name of the experiment
            log_dir: Directory to save logs
        """
        self.experiment_name = experiment_name or f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.log_dir = log_dir or config.LOG_DIR

        # Create directories
        self.experiment_dir = os.path.join(self.log_dir, self.experiment_name)
        os.makedirs(self.experiment_dir, exist_ok=True)

        # Initialize storage
        self.training_logs = []
        self.evaluation_logs = []
        self.episode_logs = []
        self.decision_logs = []

        # Metrics history
        self.metrics_history = {
            'episode': [],
            'total_reward': [],
            'mean_waiting_time': [],
            'total_energy': [],
            'starvation_violations': []
        }

        # Best metrics tracking
        self.best_reward = -float('inf')
        self.best_waiting_time = float('inf')
        self.best_energy = float('inf')

        print(f"Logger initialized: {self.experiment_dir}")

    def log_episode(self, episode: int, episode_data: Dict[str, Any]):
        """
        Log data from a training episode

        Args:
            episode: Episode number
            episode_data: Dictionary with episode metrics
        """
        log_entry = {
            'episode': episode,
            'timestamp': datetime.now().isoformat(),
            **episode_data
        }

        self.episode_logs.append(log_entry)

        # Update metrics history
        self.metrics_history['episode'].append(episode)
        self.metrics_history['total_reward'].append(episode_data.get('total_reward', 0))
        self.metrics_history['mean_waiting_time'].append(episode_data.get('mean_waiting_time', 0))
        self.metrics_history['total_energy'].append(episode_data.get('total_energy', 0))
        self.metrics_history['starvation_violations'].append(episode_data.get('starvation_violations', 0))

        # Update best metrics
        if episode_data.get('total_reward', -float('inf')) > self.best_reward:
            self.best_reward = episode_data['total_reward']
        if episode_data.get('mean_waiting_time', float('inf')) < self.best_waiting_time:
            self.best_waiting_time = episode_data['mean_waiting_time']
        if episode_data.get('total_energy', float('inf')) < self.best_energy:
            self.best_energy = episode_data['total_energy']

        # Print progress
        if config.VERBOSE >= 1 and episode % config.LOG_INTERVAL == 0:
            self._print_episode_summary(episode, episode_data)

    def log_decision(self, decision_data: Dict[str, Any]):
        """
        Log a single scheduling decision (for detailed analysis)

        Args:
            decision_data: Dictionary with decision details
        """
        if config.DEBUG_MODE:
            self.decision_logs.append(decision_data)

    def log_evaluation(self, episode: int, eval_metrics: Dict[str, Any]):
        """
        Log evaluation results

        Args:
            episode: Episode number when evaluation occurred
            eval_metrics: Dictionary with evaluation metrics
        """
        eval_entry = {
            'episode': episode,
            'timestamp': datetime.now().isoformat(),
            **eval_metrics
        }

        self.evaluation_logs.append(eval_entry)

        if config.VERBOSE >= 1:
            self._print_evaluation_summary(episode, eval_metrics)

    def log_training_step(self, step_data: Dict[str, Any]):
        """
        Log PPO training step metrics

        Args:
            step_data: Dictionary with training metrics (loss, etc.)
        """
        self.training_logs.append(step_data)

    def _print_episode_summary(self, episode: int, data: Dict[str, Any]):
        """Print episode summary to console"""
        print(f"\n{'=' * 70}")
        print(f"Episode {episode}")
        print(f"{'=' * 70}")
        print(f"Total Reward:        {data.get('total_reward', 0):.2f}")
        print(f"Mean Waiting Time:   {data.get('mean_waiting_time', 0):.2f}")
        print(f"Total Energy:        {data.get('total_energy', 0):.2f}")
        print(f"Tasks Served:        {data.get('total_tasks_served', 0)}")
        print(f"Starvation Violations: {data.get('starvation_violations', 0)}")

        # Per-category waiting times
        if 'mean_waiting_time_Low' in data:
            print(f"\nPer-Category Waiting Times:")
            for cat in config.TASK_CATEGORIES:
                wt = data.get(f'mean_waiting_time_{cat}', 0)
                print(f"  {cat:8s}: {wt:.2f}")

        print(f"{'=' * 70}\n")

    def _print_evaluation_summary(self, episode: int, eval_metrics: Dict[str, Any]):
        """Print evaluation summary to console"""
        print(f"\n{'*' * 70}")
        print(f"EVALUATION at Episode {episode}")
        print(f"{'*' * 70}")
        print(f"Avg Reward:          {eval_metrics.get('avg_reward', 0):.2f}")
        print(f"Avg Waiting Time:    {eval_metrics.get('avg_waiting_time', 0):.2f}")
        print(f"Avg Energy:          {eval_metrics.get('avg_energy', 0):.2f}")
        print(f"Success Rate:        {eval_metrics.get('success_rate', 0):.2%}")
        print(f"{'*' * 70}\n")

    def save_logs(self):
        """Save all logs to files"""
        # Save episode logs
        if self.episode_logs:
            episode_file = os.path.join(self.experiment_dir, 'episodes.json')
            with open(episode_file, 'w') as f:
                json.dump(self.episode_logs, f, indent=2)

        # Save evaluation logs
        if self.evaluation_logs:
            eval_file = os.path.join(self.experiment_dir, 'evaluations.json')
            with open(eval_file, 'w') as f:
                json.dump(self.evaluation_logs, f, indent=2)

        # Save training logs
        if self.training_logs:
            training_file = os.path.join(self.experiment_dir, 'training.json')
            with open(training_file, 'w') as f:
                json.dump(self.training_logs, f, indent=2)

        # Save decision logs if in debug mode
        if self.decision_logs:
            decision_file = os.path.join(self.experiment_dir, 'decisions.json')
            with open(decision_file, 'w') as f:
                json.dump(self.decision_logs, f, indent=2)

        # Save metrics history as CSV for easy plotting
        self._save_metrics_csv()

        print(f"Logs saved to: {self.experiment_dir}")

    def _save_metrics_csv(self):
        """Save metrics history to CSV file"""
        csv_file = os.path.join(self.experiment_dir, 'metrics_history.csv')

        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)

            # Write header
            header = ['episode', 'total_reward', 'mean_waiting_time',
                      'total_energy', 'starvation_violations']
            writer.writerow(header)

            # Write data
            for i in range(len(self.metrics_history['episode'])):
                row = [
                    self.metrics_history['episode'][i],
                    self.metrics_history['total_reward'][i],
                    self.metrics_history['mean_waiting_time'][i],
                    self.metrics_history['total_energy'][i],
                    self.metrics_history['starvation_violations'][i]
                ]
                writer.writerow(row)

    def save_config(self):
        """Save configuration to file"""
        config_file = os.path.join(self.experiment_dir, 'config.json')

        config_dict = {
            'NUM_CORES': config.NUM_CORES,
            'FREQUENCY_LEVELS': config.FREQUENCY_LEVELS,
            'TASK_CATEGORIES': config.TASK_CATEGORIES,
            'WORKLOAD_SIZES': config.WORKLOAD_SIZES,
            'TASK_ARRIVAL_RATE': config.TASK_ARRIVAL_RATE,
            'MAX_SCHEDULING_DECISIONS': config.MAX_SCHEDULING_DECISIONS,
            'MAX_WAITING_TIME': config.MAX_WAITING_TIME,
            'BETA': config.BETA,
            'ANTI_STARVATION_WEIGHTS': config.ANTI_STARVATION_WEIGHTS,
            'LEARNING_RATE_ACTOR': config.LEARNING_RATE_ACTOR,
            'LEARNING_RATE_CRITIC': config.LEARNING_RATE_CRITIC,
            'GAMMA': config.GAMMA,
            'CLIP_EPSILON': config.CLIP_EPSILON,
            'STATE_DIM': config.STATE_DIM,
            'ACTION_DIM': config.ACTION_DIM
        }

        with open(config_file, 'w') as f:
            json.dump(config_dict, f, indent=2)

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics"""
        if not self.episode_logs:
            return {}

        summary = {
            'total_episodes': len(self.episode_logs),
            'best_reward': self.best_reward,
            'best_waiting_time': self.best_waiting_time,
            'best_energy': self.best_energy,
            'final_reward': self.metrics_history['total_reward'][-1] if self.metrics_history['total_reward'] else 0,
            'final_waiting_time': self.metrics_history['mean_waiting_time'][-1] if self.metrics_history[
                'mean_waiting_time'] else 0,
            'final_energy': self.metrics_history['total_energy'][-1] if self.metrics_history['total_energy'] else 0
        }

        # Calculate moving averages for last 100 episodes
        if len(self.episode_logs) >= 100:
            recent_rewards = self.metrics_history['total_reward'][-100:]
            recent_waiting = self.metrics_history['mean_waiting_time'][-100:]
            recent_energy = self.metrics_history['total_energy'][-100:]

            summary['avg_reward_last_100'] = np.mean(recent_rewards)
            summary['avg_waiting_time_last_100'] = np.mean(recent_waiting)
            summary['avg_energy_last_100'] = np.mean(recent_energy)

        return summary

    def print_final_summary(self):
        """Print final training summary"""
        summary = self.get_metrics_summary()

        print("\n" + "=" * 70)
        print("TRAINING SUMMARY")
        print("=" * 70)
        print(f"Total Episodes:      {summary.get('total_episodes', 0)}")
        print(f"\nBest Metrics:")
        print(f"  Best Reward:       {summary.get('best_reward', 0):.2f}")
        print(f"  Best Waiting Time: {summary.get('best_waiting_time', 0):.2f}")
        print(f"  Best Energy:       {summary.get('best_energy', 0):.2f}")
        print(f"\nFinal Metrics:")
        print(f"  Final Reward:      {summary.get('final_reward', 0):.2f}")
        print(f"  Final Waiting Time: {summary.get('final_waiting_time', 0):.2f}")
        print(f"  Final Energy:      {summary.get('final_energy', 0):.2f}")

        if 'avg_reward_last_100' in summary:
            print(f"\nLast 100 Episodes Average:")
            print(f"  Avg Reward:        {summary['avg_reward_last_100']:.2f}")
            print(f"  Avg Waiting Time:  {summary['avg_waiting_time_last_100']:.2f}")
            print(f"  Avg Energy:        {summary['avg_energy_last_100']:.2f}")

        print("=" * 70 + "\n")


class TensorboardLogger:
    """
    Logger for TensorBoard (optional)
    Requires tensorboard package
    """

    def __init__(self, log_dir: str = None):
        """
        Initialize TensorBoard logger

        Args:
            log_dir: Directory for TensorBoard logs
        """
        self.log_dir = log_dir or config.LOG_DIR
        self.writer = None

        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(self.log_dir)
            print(f"TensorBoard logger initialized: {self.log_dir}")
        except ImportError:
            print("TensorBoard not available. Install with: pip install tensorboard")

    def log_scalar(self, tag: str, value: float, step: int):
        """Log scalar value"""
        if self.writer:
            self.writer.add_scalar(tag, value, step)

    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], step: int):
        """Log multiple scalar values"""
        if self.writer:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)

    def close(self):
        """Close TensorBoard writer"""
        if self.writer:
            self.writer.close()


if __name__ == "__main__":
    print("Testing Logger...")

    # Create logger
    logger = Logger(experiment_name="test_experiment")

    # Save configuration
    logger.save_config()

    # Simulate some episodes
    print("\nSimulating episodes...")
    for ep in range(1, 6):
        episode_data = {
            'total_reward': 100 + ep * 10 + np.random.randn() * 5,
            'mean_waiting_time': 20 - ep * 0.5 + np.random.randn(),
            'total_energy': 5000 - ep * 50 + np.random.randn() * 100,
            'total_tasks_served': 150 + ep * 5,
            'starvation_violations': max(0, 5 - ep),
            'mean_waiting_time_Low': 15 + np.random.randn(),
            'mean_waiting_time_Medium': 20 + np.random.randn(),
            'mean_waiting_time_Heavy': 25 + np.random.randn()
        }

        logger.log_episode(ep, episode_data)

    # Simulate evaluation
    print("\nSimulating evaluation...")
    eval_metrics = {
        'avg_reward': 150.5,
        'avg_waiting_time': 18.3,
        'avg_energy': 4500.0,
        'success_rate': 0.95
    }
    logger.log_evaluation(5, eval_metrics)

    # Print final summary
    logger.print_final_summary()

    # Save logs
    logger.save_logs()

    print("\nLogger test completed!")