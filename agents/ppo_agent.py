"""
PPO Agent for CPU Scheduling with DVFS
Implements Proximal Policy Optimization with:
- Trajectory collection
- Generalized Advantage Estimation (GAE)
- Policy & value updates with clipping
- Action masking for invalid actions (empty queues)
"""

import sys
import os

# Add parent directory to path for imports when running directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import Tuple, Dict, List, Optional
import config


class ActorCritic(nn.Module):
    """
    Combined Actor-Critic Network for PPO
    
    Actor: Outputs action logits (before mask & softmax)
    Critic: Outputs state value estimate
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_sizes: List[int] = None):
        super(ActorCritic, self).__init__()
        
        if hidden_sizes is None:
            hidden_sizes = config.HIDDEN_SIZES
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Build shared feature extractor
        layers = []
        input_size = state_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(self._get_activation())
            input_size = hidden_size
        
        self.shared_net = nn.Sequential(*layers)
        
        # Actor head (outputs logits for each action)
        self.actor_head = nn.Linear(hidden_sizes[-1], action_dim)
        
        # Critic head (outputs single value)
        self.critic_head = nn.Linear(hidden_sizes[-1], 1)
        
        # Initialize weights
        self._init_weights()
    
    def _get_activation(self) -> nn.Module:
        """Get activation function based on config"""
        if config.ACTIVATION == 'relu':
            return nn.ReLU()
        elif config.ACTIVATION == 'tanh':
            return nn.Tanh()
        elif config.ACTIVATION == 'elu':
            return nn.ELU()
        else:
            return nn.ReLU()
    
    def _init_weights(self):
        """Initialize network weights using orthogonal initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
        
        # Smaller initialization for output layers
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)
    
    def forward(self, state: torch.Tensor, action_mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through actor-critic network
        
        Args:
            state: State tensor of shape (batch, state_dim)
            action_mask: Optional mask of shape (batch, action_dim), 1=valid, 0=invalid
        
        Returns:
            action_logits: Masked logits for actions
            value: State value estimate
        """
        # Shared features
        features = self.shared_net(state)
        
        # Actor output (logits)
        logits = self.actor_head(features)
        
        # Apply action mask if provided
        if action_mask is not None:
            # Ensure mask is boolean or can be compared
            mask_bool = action_mask > 0.5
            
            # Check if mask has any valid actions
            if mask_bool.any():
                # Set invalid actions to very large negative value
                logits = logits.masked_fill(~mask_bool, -1e8)
            else:
                # No valid actions - keep original logits (uniform distribution)
                # This shouldn't happen in normal operation but prevents NaN
                pass
        
        # Critic output
        value = self.critic_head(features)
        
        return logits, value.squeeze(-1)
    
    def get_action(self, state: torch.Tensor, action_mask: torch.Tensor = None, 
                   deterministic: bool = False) -> Tuple[int, float, float]:
        """
        Sample action from policy
        
        Args:
            state: State tensor
            action_mask: Action validity mask
            deterministic: If True, return most likely action
        
        Returns:
            action: Selected action index
            log_prob: Log probability of selected action
            value: State value estimate
        """
        logits, value = self.forward(state, action_mask)
        
        # Handle edge case where all logits might be very negative
        # Add numerical stability by subtracting max (doesn't affect softmax)
        logits_stable = logits - logits.max(dim=-1, keepdim=True)[0]
        
        # Create distribution from stabilized logits
        dist = Categorical(logits=logits_stable)
        
        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item(), value.item()
    
    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor, 
                         action_masks: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO update
        
        Args:
            states: Batch of states
            actions: Batch of actions taken
            action_masks: Batch of action masks
        
        Returns:
            log_probs: Log probabilities of actions
            values: State value estimates
            entropy: Policy entropy
        """
        logits, values = self.forward(states, action_masks)
        
        # Add numerical stability
        logits_stable = logits - logits.max(dim=-1, keepdim=True)[0]
        
        # Clamp extreme values to prevent NaN
        logits_stable = torch.clamp(logits_stable, min=-100, max=0)
        
        dist = Categorical(logits=logits_stable)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        # Handle any NaN values that might have slipped through
        log_probs = torch.where(torch.isnan(log_probs), torch.zeros_like(log_probs), log_probs)
        entropy = torch.where(torch.isnan(entropy), torch.zeros_like(entropy), entropy)
        
        return log_probs, values, entropy


class RolloutBuffer:
    """
    Buffer for storing trajectories collected during rollout
    """
    
    def __init__(self, buffer_size: int, state_dim: int, action_dim: int):
        self.buffer_size = buffer_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.reset()
    
    def reset(self):
        """Clear the buffer"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.action_masks = []
        
        self.advantages = None
        self.returns = None
        self.ptr = 0
    
    def add(self, state: np.ndarray, action: int, reward: float, value: float,
            log_prob: float, done: bool, action_mask: np.ndarray):
        """Add a transition to the buffer"""
        self.states.append(state.copy())
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        self.action_masks.append(action_mask.copy())
        self.ptr += 1
    
    def is_full(self) -> bool:
        """Check if buffer is full"""
        return self.ptr >= self.buffer_size
    
    def compute_returns_and_advantages(self, last_value: float, gamma: float = None, 
                                        gae_lambda: float = None):
        """
        Compute returns and advantages using GAE
        
        Args:
            last_value: Value estimate for the state after the last collected state
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
        """
        if gamma is None:
            gamma = config.GAMMA
        if gae_lambda is None:
            gae_lambda = config.GAE_LAMBDA
        
        n = len(self.rewards)
        self.advantages = np.zeros(n, dtype=np.float32)
        self.returns = np.zeros(n, dtype=np.float32)
        
        # GAE computation
        gae = 0.0
        for t in reversed(range(n)):
            if t == n - 1:
                next_value = last_value
                next_non_terminal = 1.0 - float(self.dones[t])
            else:
                next_value = self.values[t + 1]
                next_non_terminal = 1.0 - float(self.dones[t])
            
            # TD error
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            
            # GAE
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            self.advantages[t] = gae
        
        # Returns = advantages + values
        self.returns = self.advantages + np.array(self.values, dtype=np.float32)
    
    def get_tensors(self, device: torch.device) -> Dict[str, torch.Tensor]:
        """Convert buffer to tensors"""
        return {
            'states': torch.FloatTensor(np.array(self.states)).to(device),
            'actions': torch.LongTensor(self.actions).to(device),
            'log_probs': torch.FloatTensor(self.log_probs).to(device),
            'advantages': torch.FloatTensor(self.advantages).to(device),
            'returns': torch.FloatTensor(self.returns).to(device),
            'action_masks': torch.FloatTensor(np.array(self.action_masks)).to(device)
        }
    
    def __len__(self):
        return self.ptr


class PPOAgent:
    """
    Proximal Policy Optimization Agent
    
    Implements:
    - Trajectory collection
    - Generalized Advantage Estimation (GAE)
    - Clipped surrogate objective
    - Action masking for invalid actions
    """
    
    def __init__(self, state_dim: int = None, action_dim: int = None, 
                 device: str = 'auto', seed: int = None):
        """
        Initialize PPO Agent
        
        Args:
            state_dim: Observation dimension
            action_dim: Action space size
            device: Device to use ('cpu', 'cuda', or 'auto')
            seed: Random seed
        """
        # Dimensions from config if not provided
        if state_dim is None:
            state_dim = config.STATE_DIM
        if action_dim is None:
            action_dim = config.ACTION_DIM
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Set seed for reproducibility
        if seed is None:
            seed = config.RANDOM_SEED
        self.seed = seed
        self._set_seed(seed)
        
        # Create actor-critic network
        self.policy = ActorCritic(state_dim, action_dim).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), 
                                    lr=config.LEARNING_RATE_ACTOR)
        
        # PPO hyperparameters
        self.gamma = config.GAMMA
        self.gae_lambda = config.GAE_LAMBDA
        self.clip_epsilon = config.CLIP_EPSILON
        self.entropy_coef = config.ENTROPY_COEF
        self.value_coef = config.VALUE_LOSS_COEF
        self.max_grad_norm = config.MAX_GRAD_NORM
        
        # Training parameters
        self.ppo_epochs = config.PPO_EPOCHS
        self.batch_size = config.BATCH_SIZE
        self.buffer_size = config.BUFFER_SIZE
        
        # Rollout buffer
        self.buffer = RolloutBuffer(self.buffer_size, state_dim, action_dim)
        
        # Training statistics
        self.training_step = 0
        self.episode_count = 0
        
        print(f"PPO Agent initialized on {self.device}")
        print(f"  State dim: {state_dim}, Action dim: {action_dim}")
        print(f"  Buffer size: {self.buffer_size}, Batch size: {self.batch_size}")
    
    def _set_seed(self, seed: int):
        """Set random seeds for reproducibility"""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    
    def select_action(self, state: np.ndarray, action_mask: np.ndarray = None,
                      deterministic: bool = False) -> Tuple[int, float, float]:
        """
        Select action given state
        
        Args:
            state: Current state
            action_mask: Mask for valid actions (1=valid, 0=invalid)
            deterministic: If True, select best action (no exploration)
        
        Returns:
            action: Selected action
            log_prob: Log probability of action
            value: State value estimate
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        if action_mask is not None:
            mask_tensor = torch.FloatTensor(action_mask).unsqueeze(0).to(self.device)
        else:
            mask_tensor = None
        
        with torch.no_grad():
            action, log_prob, value = self.policy.get_action(
                state_tensor, mask_tensor, deterministic
            )
        
        return action, log_prob, value
    
    def store_transition(self, state: np.ndarray, action: int, reward: float,
                         value: float, log_prob: float, done: bool,
                         action_mask: np.ndarray):
        """Store a transition in the buffer"""
        self.buffer.add(state, action, reward, value, log_prob, done, action_mask)
    
    def collect_rollouts(self, env, num_steps: int = None) -> Dict[str, float]:
        """
        Collect rollout data from environment
        
        Args:
            env: Environment to collect from
            num_steps: Number of steps to collect (default: buffer_size)
        
        Returns:
            Dictionary with rollout statistics
        """
        if num_steps is None:
            num_steps = self.buffer_size
        
        self.buffer.reset()
        self.policy.eval()
        
        # Get or reset environment
        state = env.reset()
        episode_rewards = []
        current_episode_reward = 0.0
        episodes_completed = 0
        
        for step in range(num_steps):
            # Get action mask from environment
            action_mask = env.get_action_mask()
            
            # Select action
            action, log_prob, value = self.select_action(state, action_mask)
            
            # Take step in environment
            next_state, reward, done, info = env.step(action)
            
            # Store transition
            self.store_transition(state, action, reward, value, log_prob, done, action_mask)
            
            current_episode_reward += reward
            state = next_state
            
            if done:
                episodes_completed += 1
                episode_rewards.append(current_episode_reward)
                current_episode_reward = 0.0
                state = env.reset()
                self.episode_count += 1
        
        # Get value estimate for last state (for GAE)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            _, last_value = self.policy(state_tensor)
            last_value = last_value.item()
        
        # Compute returns and advantages
        self.buffer.compute_returns_and_advantages(last_value, self.gamma, self.gae_lambda)
        
        # Rollout statistics
        stats = {
            'episodes_completed': episodes_completed,
            'mean_reward': np.mean(episode_rewards) if episode_rewards else 0.0,
            'std_reward': np.std(episode_rewards) if len(episode_rewards) > 1 else 0.0,
            'steps_collected': len(self.buffer)
        }
        
        return stats
    
    def update(self) -> Dict[str, float]:
        """
        Perform PPO update using collected trajectories
        
        Returns:
            Dictionary with training statistics
        """
        self.policy.train()
        
        # Get data from buffer
        data = self.buffer.get_tensors(self.device)
        
        # Normalize advantages
        advantages = data['advantages']
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Training statistics
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_loss = 0.0
        num_updates = 0
        
        # Multiple epochs of PPO updates
        buffer_size = len(self.buffer)
        indices = np.arange(buffer_size)
        
        for epoch in range(self.ppo_epochs):
            np.random.shuffle(indices)
            
            # Mini-batch updates
            for start in range(0, buffer_size, self.batch_size):
                end = min(start + self.batch_size, buffer_size)
                batch_indices = indices[start:end]
                
                # Get batch data
                batch_states = data['states'][batch_indices]
                batch_actions = data['actions'][batch_indices]
                batch_old_log_probs = data['log_probs'][batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = data['returns'][batch_indices]
                batch_masks = data['action_masks'][batch_indices]
                
                # Evaluate actions with current policy
                new_log_probs, values, entropy = self.policy.evaluate_actions(
                    batch_states, batch_actions, batch_masks
                )
                
                # Compute ratio for PPO
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # Clipped surrogate objective
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss (MSE)
                value_loss = nn.functional.mse_loss(values, batch_returns)
                
                # Entropy bonus
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = (policy_loss + 
                        self.value_coef * value_loss + 
                        self.entropy_coef * entropy_loss)
                
                # Gradient update
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Accumulate statistics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                total_loss += loss.item()
                num_updates += 1
        
        self.training_step += 1
        
        # Average statistics
        stats = {
            'policy_loss': total_policy_loss / num_updates,
            'value_loss': total_value_loss / num_updates,
            'entropy': total_entropy / num_updates,
            'total_loss': total_loss / num_updates,
            'training_step': self.training_step
        }
        
        return stats
    
    def train_step(self, env) -> Dict[str, float]:
        """
        Complete training step: collect rollouts and update
        
        Args:
            env: Environment to train on
        
        Returns:
            Combined statistics from rollout and update
        """
        # Collect rollouts
        rollout_stats = self.collect_rollouts(env)
        
        # Update policy
        update_stats = self.update()
        
        # Combine statistics
        stats = {**rollout_stats, **update_stats}
        return stats
    
    def train(self, env, num_epochs: int = None, log_interval: int = None,
              eval_env = None, eval_interval: int = None, 
              callback = None) -> List[Dict[str, float]]:
        """
        Full training loop
        
        Args:
            env: Training environment
            num_epochs: Number of training epochs
            log_interval: How often to log progress
            eval_env: Optional separate evaluation environment
            eval_interval: How often to evaluate
            callback: Optional callback function(agent, epoch, stats)
        
        Returns:
            List of training statistics per epoch
        """
        if num_epochs is None:
            num_epochs = config.NUM_EPOCHS
        if log_interval is None:
            log_interval = config.LOG_INTERVAL
        if eval_interval is None:
            eval_interval = config.EVAL_INTERVAL
        
        training_history = []
        
        print(f"\nStarting PPO training for {num_epochs} epochs...")
        print(f"  Buffer size: {self.buffer_size}, Batch size: {self.batch_size}")
        print(f"  PPO epochs per update: {self.ppo_epochs}")
        print("-" * 60)
        
        for epoch in range(num_epochs):
            # Training step
            stats = self.train_step(env)
            stats['epoch'] = epoch
            training_history.append(stats)
            
            # Logging
            if (epoch + 1) % log_interval == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}")
                print(f"  Mean Reward: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
                print(f"  Policy Loss: {stats['policy_loss']:.4f}, Value Loss: {stats['value_loss']:.4f}")
                print(f"  Entropy: {stats['entropy']:.4f}")
            
            # Evaluation
            if eval_env is not None and (epoch + 1) % eval_interval == 0:
                eval_stats = self.evaluate(eval_env)
                print(f"  [Eval] Mean Reward: {eval_stats['mean_reward']:.2f}")
                stats['eval_mean_reward'] = eval_stats['mean_reward']
            
            # Callback
            if callback is not None:
                callback(self, epoch, stats)
        
        print("-" * 60)
        print("Training completed!")
        
        return training_history
    
    def evaluate(self, env, num_episodes: int = None) -> Dict[str, float]:
        """
        Evaluate agent on environment
        
        Args:
            env: Environment to evaluate on
            num_episodes: Number of evaluation episodes
        
        Returns:
            Evaluation statistics
        """
        if num_episodes is None:
            num_episodes = config.NUM_EVAL_EPISODES
        
        self.policy.eval()
        
        episode_rewards = []
        episode_lengths = []
        
        for _ in range(num_episodes):
            state = env.reset()
            done = False
            episode_reward = 0.0
            episode_length = 0
            
            while not done:
                action_mask = env.get_action_mask()
                action, _, _ = self.select_action(state, action_mask, deterministic=True)
                state, reward, done, _ = env.step(action)
                episode_reward += reward
                episode_length += 1
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        
        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards)
        }
    
    def save(self, path: str):
        """Save agent to file"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_step': self.training_step,
            'episode_count': self.episode_count,
            'config': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'gamma': self.gamma,
                'gae_lambda': self.gae_lambda,
                'clip_epsilon': self.clip_epsilon
            }
        }, path)
        print(f"Agent saved to {path}")
    
    def load(self, path: str):
        """Load agent from file"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_step = checkpoint.get('training_step', 0)
        self.episode_count = checkpoint.get('episode_count', 0)
        print(f"Agent loaded from {path}")


if __name__ == "__main__":
    print("Testing PPO Agent...")
    
    # Import environment
    from envs.cpu_scheduler_env import CPUSchedulerEnv
    
    # Create environment and agent
    env = CPUSchedulerEnv(seed=42)
    agent = PPOAgent(seed=42)
    
    # Test action selection
    print("\n1. Testing action selection:")
    state = env.reset()
    action_mask = env.get_action_mask()
    action, log_prob, value = agent.select_action(state, action_mask)
    print(f"  State shape: {state.shape}")
    print(f"  Action: {action}, Log prob: {log_prob:.4f}, Value: {value:.4f}")
    
    # Test single training step
    print("\n2. Testing single training step:")
    stats = agent.train_step(env)
    print(f"  Episodes: {stats['episodes_completed']}")
    print(f"  Mean Reward: {stats['mean_reward']:.2f}")
    print(f"  Policy Loss: {stats['policy_loss']:.4f}")
    
    # Test evaluation
    print("\n3. Testing evaluation:")
    eval_stats = agent.evaluate(env, num_episodes=3)
    print(f"  Mean Reward: {eval_stats['mean_reward']:.2f}")
    print(f"  Mean Length: {eval_stats['mean_length']:.1f}")
    
    # Test save/load
    print("\n4. Testing save/load:")
    agent.save("./models/test_ppo.pt")
    agent.load("./models/test_ppo.pt")
    
    print("\nPPO Agent test completed!")
