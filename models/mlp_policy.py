"""
MLP Policy Networks for Reinforcement Learning
Modular neural network architectures for actor-critic methods
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from typing import Tuple, List, Optional

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class MLP(nn.Module):
    """
    Generic Multi-Layer Perceptron (MLP) module
    
    Can be used as a building block for various RL network architectures.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_sizes: List[int] = None,
        activation: str = 'relu',
        output_activation: str = None,
        use_layer_norm: bool = False,
        dropout: float = 0.0
    ):
        """
        Initialize MLP
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            hidden_sizes: List of hidden layer sizes
            activation: Activation function ('relu', 'tanh', 'elu', 'gelu')
            output_activation: Activation for output layer (None for linear)
            use_layer_norm: Whether to use layer normalization
            dropout: Dropout probability (0.0 = no dropout)
        """
        super(MLP, self).__init__()
        
        if hidden_sizes is None:
            hidden_sizes = config.HIDDEN_SIZES
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_sizes = hidden_sizes
        
        # Build layers
        layers = []
        prev_size = input_dim
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_size))
            
            layers.append(self._get_activation(activation))
            
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_dim))
        
        if output_activation is not None:
            layers.append(self._get_activation(output_activation))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name"""
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'elu': nn.ELU(),
            'gelu': nn.GELU(),
            'leaky_relu': nn.LeakyReLU(0.1),
            'sigmoid': nn.Sigmoid(),
            'softmax': nn.Softmax(dim=-1)
        }
        return activations.get(name.lower(), nn.ReLU())
    
    def _init_weights(self):
        """Initialize network weights using orthogonal initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network"""
        return self.network(x)


class ActorNetwork(nn.Module):
    """
    Actor network for policy gradient methods
    
    Outputs action logits (before softmax) for discrete action spaces.
    Supports action masking for invalid actions.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: List[int] = None,
        activation: str = None
    ):
        """
        Initialize Actor Network
        
        Args:
            state_dim: State/observation dimension
            action_dim: Number of discrete actions
            hidden_sizes: Hidden layer sizes
            activation: Activation function
        """
        super(ActorNetwork, self).__init__()
        
        if hidden_sizes is None:
            hidden_sizes = config.HIDDEN_SIZES
        if activation is None:
            activation = config.ACTIVATION
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.network = MLP(
            input_dim=state_dim,
            output_dim=action_dim,
            hidden_sizes=hidden_sizes,
            activation=activation,
            output_activation=None  # Linear output for logits
        )
        
        # Smaller initialization for output layer (more stable training)
        nn.init.orthogonal_(self.network.network[-1].weight, gain=0.01)
    
    def forward(
        self,
        state: torch.Tensor,
        action_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass returning action logits
        
        Args:
            state: State tensor of shape (batch, state_dim)
            action_mask: Optional mask (1=valid, 0=invalid) of shape (batch, action_dim)
        
        Returns:
            Action logits (masked if mask provided)
        """
        logits = self.network(state)
        
        # Apply action mask if provided
        if action_mask is not None:
            mask_bool = action_mask > 0.5
            if mask_bool.any():
                logits = logits.masked_fill(~mask_bool, -1e8)
        
        return logits
    
    def get_action(
        self,
        state: torch.Tensor,
        action_mask: torch.Tensor = None,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action from policy
        
        Args:
            state: State tensor
            action_mask: Action validity mask
            deterministic: If True, return most likely action
        
        Returns:
            action: Selected action
            log_prob: Log probability of action
        """
        logits = self.forward(state, action_mask)
        
        # Numerical stability
        logits_stable = logits - logits.max(dim=-1, keepdim=True)[0]
        
        dist = Categorical(logits=logits_stable)
        
        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        
        return action, log_prob
    
    def evaluate_actions(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        action_masks: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate log probability and entropy for given actions
        
        Args:
            states: Batch of states
            actions: Batch of actions
            action_masks: Batch of action masks
        
        Returns:
            log_probs: Log probabilities
            entropy: Policy entropy
        """
        logits = self.forward(states, action_masks)
        
        # Numerical stability
        logits_stable = logits - logits.max(dim=-1, keepdim=True)[0]
        logits_stable = torch.clamp(logits_stable, min=-100, max=0)
        
        dist = Categorical(logits=logits_stable)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        # Handle NaN
        log_probs = torch.where(torch.isnan(log_probs), torch.zeros_like(log_probs), log_probs)
        entropy = torch.where(torch.isnan(entropy), torch.zeros_like(entropy), entropy)
        
        return log_probs, entropy


class CriticNetwork(nn.Module):
    """
    Critic network for value estimation
    
    Outputs a single scalar value estimate for each state.
    """
    
    def __init__(
        self,
        state_dim: int,
        hidden_sizes: List[int] = None,
        activation: str = None
    ):
        """
        Initialize Critic Network
        
        Args:
            state_dim: State/observation dimension
            hidden_sizes: Hidden layer sizes
            activation: Activation function
        """
        super(CriticNetwork, self).__init__()
        
        if hidden_sizes is None:
            hidden_sizes = config.HIDDEN_SIZES
        if activation is None:
            activation = config.ACTIVATION
        
        self.state_dim = state_dim
        
        self.network = MLP(
            input_dim=state_dim,
            output_dim=1,  # Single value output
            hidden_sizes=hidden_sizes,
            activation=activation,
            output_activation=None  # Linear output
        )
        
        # Standard initialization for value head
        nn.init.orthogonal_(self.network.network[-1].weight, gain=1.0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning state value
        
        Args:
            state: State tensor of shape (batch, state_dim)
        
        Returns:
            Value estimate of shape (batch,)
        """
        value = self.network(state)
        return value.squeeze(-1)


class ActorCriticPolicy(nn.Module):
    """
    Combined Actor-Critic Policy Network
    
    Shares feature extraction between actor and critic for efficiency.
    Suitable for PPO, A2C, and similar algorithms.
    """
    
    def __init__(
        self,
        state_dim: int = None,
        action_dim: int = None,
        hidden_sizes: List[int] = None,
        activation: str = None,
        shared_layers: int = None
    ):
        """
        Initialize Actor-Critic Policy
        
        Args:
            state_dim: State dimension (default: from config)
            action_dim: Action dimension (default: from config)
            hidden_sizes: Hidden layer sizes
            activation: Activation function
            shared_layers: Number of shared layers (default: all but last)
        """
        super(ActorCriticPolicy, self).__init__()
        
        # Defaults from config
        if state_dim is None:
            state_dim = config.STATE_DIM
        if action_dim is None:
            action_dim = config.ACTION_DIM
        if hidden_sizes is None:
            hidden_sizes = config.HIDDEN_SIZES
        if activation is None:
            activation = config.ACTIVATION
        if shared_layers is None:
            shared_layers = len(hidden_sizes)
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_sizes = hidden_sizes
        
        # Build shared feature extractor
        shared_hidden = hidden_sizes[:shared_layers]
        layers = []
        input_size = state_dim
        
        for hidden_size in shared_hidden:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(self._get_activation(activation))
            input_size = hidden_size
        
        self.shared_net = nn.Sequential(*layers)
        
        # Actor head
        self.actor_head = nn.Linear(input_size, action_dim)
        
        # Critic head
        self.critic_head = nn.Linear(input_size, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name"""
        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'elu': nn.ELU(),
            'gelu': nn.GELU(),
        }
        return activations.get(name.lower(), nn.ReLU())
    
    def _init_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
        
        # Smaller initialization for output layers
        nn.init.orthogonal_(self.actor_head.weight, gain=0.01)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)
    
    def forward(
        self,
        state: torch.Tensor,
        action_mask: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            state: State tensor
            action_mask: Optional action validity mask
        
        Returns:
            logits: Action logits
            value: State value
        """
        # Shared features
        features = self.shared_net(state)
        
        # Actor output
        logits = self.actor_head(features)
        
        # Apply action mask
        if action_mask is not None:
            mask_bool = action_mask > 0.5
            if mask_bool.any():
                logits = logits.masked_fill(~mask_bool, -1e8)
        
        # Critic output
        value = self.critic_head(features).squeeze(-1)
        
        return logits, value
    
    def get_action(
        self,
        state: torch.Tensor,
        action_mask: torch.Tensor = None,
        deterministic: bool = False
    ) -> Tuple[int, float, float]:
        """
        Sample action from policy
        
        Returns:
            action, log_prob, value
        """
        logits, value = self.forward(state, action_mask)
        
        # Numerical stability
        logits_stable = logits - logits.max(dim=-1, keepdim=True)[0]
        
        dist = Categorical(logits=logits_stable)
        
        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item(), value.item()
    
    def evaluate_actions(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        action_masks: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO update
        
        Returns:
            log_probs, values, entropy
        """
        logits, values = self.forward(states, action_masks)
        
        # Numerical stability
        logits_stable = logits - logits.max(dim=-1, keepdim=True)[0]
        logits_stable = torch.clamp(logits_stable, min=-100, max=0)
        
        dist = Categorical(logits=logits_stable)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        # Handle NaN
        log_probs = torch.where(torch.isnan(log_probs), torch.zeros_like(log_probs), log_probs)
        entropy = torch.where(torch.isnan(entropy), torch.zeros_like(entropy), entropy)
        
        return log_probs, values, entropy


if __name__ == "__main__":
    print("Testing MLP Policy Networks...")
    
    # Test dimensions
    state_dim = config.STATE_DIM
    action_dim = config.ACTION_DIM
    batch_size = 32
    
    # Create test input
    states = torch.randn(batch_size, state_dim)
    actions = torch.randint(0, action_dim, (batch_size,))
    action_masks = torch.ones(batch_size, action_dim)
    action_masks[:, -2:] = 0  # Mask last 2 actions
    
    print(f"\nState dim: {state_dim}, Action dim: {action_dim}")
    print(f"Batch size: {batch_size}")
    
    # Test MLP
    print("\n1. Testing MLP:")
    mlp = MLP(state_dim, 64, hidden_sizes=[128, 64])
    output = mlp(states)
    print(f"  Input shape: {states.shape}")
    print(f"  Output shape: {output.shape}")
    
    # Test Actor Network
    print("\n2. Testing Actor Network:")
    actor = ActorNetwork(state_dim, action_dim)
    logits = actor(states, action_masks)
    action, log_prob = actor.get_action(states[:1], action_masks[:1])
    print(f"  Logits shape: {logits.shape}")
    print(f"  Sampled action: {action.item()}, Log prob: {log_prob.item():.4f}")
    
    # Test Critic Network
    print("\n3. Testing Critic Network:")
    critic = CriticNetwork(state_dim)
    values = critic(states)
    print(f"  Values shape: {values.shape}")
    print(f"  Value range: [{values.min().item():.4f}, {values.max().item():.4f}]")
    
    # Test Actor-Critic Policy
    print("\n4. Testing Actor-Critic Policy:")
    policy = ActorCriticPolicy(state_dim, action_dim)
    logits, values = policy(states, action_masks)
    print(f"  Logits shape: {logits.shape}")
    print(f"  Values shape: {values.shape}")
    
    # Test get_action
    action, log_prob, value = policy.get_action(states[:1], action_masks[:1])
    print(f"  Single action: {action}, log_prob: {log_prob:.4f}, value: {value:.4f}")
    
    # Test evaluate_actions
    log_probs, values, entropy = policy.evaluate_actions(states, actions, action_masks)
    print(f"  Evaluated log_probs shape: {log_probs.shape}")
    print(f"  Mean entropy: {entropy.mean().item():.4f}")
    
    print("\n✓ All MLP policy tests passed!")
