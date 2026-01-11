"""
Training Script for PPO Agent on CPU Scheduling with DVFS Environment
Implements a complete training pipeline with:
- Training loop with configurable epochs
- Periodic evaluation and logging
- Model checkpointing
- TensorBoard logging (optional)
- Early stopping based on performance
"""

import sys
import os
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
import torch

import config
from envs.cpu_scheduler_env import CPUSchedulerEnv
from agents.ppo_agent import PPOAgent


def create_directories():
    """Create necessary directories for logs, models, and results"""
    directories = [config.LOG_DIR, config.MODEL_DIR, config.RESULTS_DIR]
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    return directories


def setup_logging(run_name: str = None):
    """Setup logging configuration"""
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"ppo_run_{timestamp}"
    
    log_path = os.path.join(config.LOG_DIR, f"{run_name}.log")
    return run_name, log_path


def train_ppo(
    num_epochs: int = None,
    seed: int = None,
    device: str = 'auto',
    log_interval: int = None,
    eval_interval: int = None,
    save_interval: int = None,
    run_name: str = None,
    verbose: int = None,
    early_stopping_patience: int = 50,
    target_reward: float = None
):
    """
    Main training function for PPO agent
    
    Args:
        num_epochs: Number of training epochs (default: from config)
        seed: Random seed for reproducibility
        device: Device to use ('cpu', 'cuda', 'auto')
        log_interval: How often to log progress
        eval_interval: How often to evaluate the agent
        save_interval: How often to save model checkpoints
        run_name: Name for this training run
        verbose: Verbosity level (0, 1, 2)
        early_stopping_patience: Stop training if no improvement for this many epochs
        target_reward: Stop training when this reward is achieved
        
    Returns:
        Dictionary containing training history and final evaluation results
    """
    # Setup defaults from config
    if num_epochs is None:
        num_epochs = config.NUM_EPOCHS
    if seed is None:
        seed = config.RANDOM_SEED
    if log_interval is None:
        log_interval = config.LOG_INTERVAL
    if eval_interval is None:
        eval_interval = config.EVAL_INTERVAL
    if save_interval is None:
        save_interval = config.SAVE_INTERVAL
    if verbose is None:
        verbose = config.VERBOSE
    
    # Create directories
    create_directories()
    
    # Setup run name and logging
    run_name, log_path = setup_logging(run_name)
    
    print(f"\n{'='*60}")
    print(f"PPO Training - {run_name}")
    print(f"{'='*60}")
    
    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Create environments
    print(f"\nCreating environment...")
    train_env = CPUSchedulerEnv(seed=seed)
    eval_env = CPUSchedulerEnv(seed=seed + 1000)  # Different seed for eval
    
    # Create agent
    print(f"Creating PPO agent...")
    agent = PPOAgent(
        state_dim=train_env.state_dim,
        action_dim=train_env.action_dim,
        device=device,
        seed=seed
    )
    
    # Training statistics
    training_history = []
    eval_history = []
    best_eval_reward = float('-inf')
    best_epoch = 0
    epochs_without_improvement = 0
    
    # Model save paths
    best_model_path = os.path.join(config.MODEL_DIR, f"{run_name}_best.pt")
    latest_model_path = os.path.join(config.MODEL_DIR, f"{run_name}_latest.pt")
    
    print(f"\nTraining Configuration:")
    print(f"  Epochs: {num_epochs}")
    print(f"  Buffer Size: {agent.buffer_size}")
    print(f"  Batch Size: {agent.batch_size}")
    print(f"  PPO Epochs: {agent.ppo_epochs}")
    print(f"  Learning Rate: {config.LEARNING_RATE_ACTOR}")
    print(f"  Device: {agent.device}")
    print(f"  Early Stopping Patience: {early_stopping_patience}")
    print(f"{'='*60}\n")
    
    # Training loop
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Collect rollouts and update
        train_stats = agent.train_step(train_env)
        train_stats['epoch'] = epoch
        train_stats['elapsed_time'] = time.time() - start_time
        training_history.append(train_stats)
        
        # Periodic logging
        if verbose >= 1 and (epoch + 1) % log_interval == 0:
            epoch_time = time.time() - epoch_start_time
            print(f"Epoch {epoch + 1:4d}/{num_epochs}")
            print(f"  Episodes: {train_stats['episodes_completed']:3d} | "
                  f"Mean Reward: {train_stats['mean_reward']:8.2f} ± {train_stats['std_reward']:.2f}")
            print(f"  Policy Loss: {train_stats['policy_loss']:.4f} | "
                  f"Value Loss: {train_stats['value_loss']:.4f} | "
                  f"Entropy: {train_stats['entropy']:.4f}")
            if verbose >= 2:
                print(f"  Time: {epoch_time:.2f}s | "
                      f"Total: {train_stats['elapsed_time']/60:.1f}min")
        
        # Periodic evaluation
        if (epoch + 1) % eval_interval == 0:
            eval_stats = agent.evaluate(eval_env, num_episodes=config.NUM_EVAL_EPISODES)
            eval_stats['epoch'] = epoch
            eval_history.append(eval_stats)
            
            if verbose >= 1:
                print(f"  [EVAL] Mean Reward: {eval_stats['mean_reward']:.2f} ± {eval_stats['std_reward']:.2f}")
            
            # Check for best model
            if eval_stats['mean_reward'] > best_eval_reward:
                best_eval_reward = eval_stats['mean_reward']
                best_epoch = epoch
                epochs_without_improvement = 0
                
                # Save best model
                agent.save(best_model_path)
                if verbose >= 1:
                    print(f"  [NEW BEST] Saved to {best_model_path}")
            else:
                epochs_without_improvement += 1
            
            # Check target reward
            if target_reward is not None and eval_stats['mean_reward'] >= target_reward:
                print(f"\nTarget reward {target_reward} achieved at epoch {epoch + 1}!")
                break
            
            # Early stopping
            if epochs_without_improvement >= early_stopping_patience:
                print(f"\nEarly stopping: No improvement for {early_stopping_patience} epochs")
                break
        
        # Periodic model saving
        if (epoch + 1) % save_interval == 0:
            agent.save(latest_model_path)
    
    # Training completed
    total_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"Training Completed!")
    print(f"{'='*60}")
    print(f"Total Time: {total_time/60:.2f} minutes")
    print(f"Best Eval Reward: {best_eval_reward:.2f} at epoch {best_epoch + 1}")
    print(f"Total Episodes: {agent.episode_count}")
    print(f"Total Training Steps: {agent.training_step}")
    
    # Final evaluation
    print(f"\nFinal Evaluation ({config.NUM_EVAL_EPISODES * 2} episodes)...")
    final_eval = agent.evaluate(eval_env, num_episodes=config.NUM_EVAL_EPISODES * 2)
    print(f"  Mean Reward: {final_eval['mean_reward']:.2f} ± {final_eval['std_reward']:.2f}")
    print(f"  Mean Episode Length: {final_eval['mean_length']:.1f}")
    print(f"  Min/Max Reward: {final_eval['min_reward']:.2f} / {final_eval['max_reward']:.2f}")
    
    # Save final model
    final_model_path = os.path.join(config.MODEL_DIR, f"{run_name}_final.pt")
    agent.save(final_model_path)
    
    # Save training results
    results = {
        'run_name': run_name,
        'seed': seed,
        'num_epochs': epoch + 1,
        'total_time_minutes': total_time / 60,
        'best_eval_reward': best_eval_reward,
        'best_epoch': best_epoch,
        'final_eval': final_eval,
        'config': {
            'num_cores': config.NUM_CORES,
            'frequency_levels': config.FREQUENCY_LEVELS,
            'state_dim': config.STATE_DIM,
            'action_dim': config.ACTION_DIM,
            'buffer_size': config.BUFFER_SIZE,
            'batch_size': config.BATCH_SIZE,
            'ppo_epochs': config.PPO_EPOCHS,
            'learning_rate': config.LEARNING_RATE_ACTOR,
            'gamma': config.GAMMA,
            'gae_lambda': config.GAE_LAMBDA,
            'clip_epsilon': config.CLIP_EPSILON
        }
    }
    
    results_path = os.path.join(config.RESULTS_DIR, f"{run_name}_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    
    # Save training history
    history_path = os.path.join(config.RESULTS_DIR, f"{run_name}_history.json")
    history_data = {
        'training': training_history,
        'evaluation': eval_history
    }
    with open(history_path, 'w') as f:
        json.dump(history_data, f, indent=2)
    print(f"History saved to {history_path}")
    
    return {
        'agent': agent,
        'training_history': training_history,
        'eval_history': eval_history,
        'final_eval': final_eval,
        'results': results
    }


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Train PPO agent for CPU Scheduling with DVFS'
    )
    
    parser.add_argument('--epochs', type=int, default=None,
                        help=f'Number of training epochs (default: {config.NUM_EPOCHS})')
    parser.add_argument('--seed', type=int, default=None,
                        help=f'Random seed (default: {config.RANDOM_SEED})')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['cpu', 'cuda', 'auto'],
                        help='Device to use (default: auto)')
    parser.add_argument('--log-interval', type=int, default=None,
                        help=f'Logging interval (default: {config.LOG_INTERVAL})')
    parser.add_argument('--eval-interval', type=int, default=None,
                        help=f'Evaluation interval (default: {config.EVAL_INTERVAL})')
    parser.add_argument('--save-interval', type=int, default=None,
                        help=f'Model save interval (default: {config.SAVE_INTERVAL})')
    parser.add_argument('--run-name', type=str, default=None,
                        help='Name for this training run')
    parser.add_argument('--verbose', type=int, default=None,
                        choices=[0, 1, 2],
                        help='Verbosity level (default: 1)')
    parser.add_argument('--early-stopping', type=int, default=50,
                        help='Early stopping patience (default: 50)')
    parser.add_argument('--target-reward', type=float, default=None,
                        help='Target reward to achieve (default: None)')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    results = train_ppo(
        num_epochs=args.epochs,
        seed=args.seed,
        device=args.device,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        run_name=args.run_name,
        verbose=args.verbose,
        early_stopping_patience=args.early_stopping,
        target_reward=args.target_reward
    )
    
    print("\n✓ Training script completed successfully!")
