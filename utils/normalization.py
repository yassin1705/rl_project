"""
Normalization Utilities for State Features
Implements running normalization for stable training
"""

import numpy as np
from typing import Optional, Union
import config


class RunningNormalizer:
    """
    Running mean and standard deviation normalizer
    Uses exponential moving average for online updates
    """

    def __init__(self,
                 shape: Union[int, tuple],
                 update_rate: float = None,
                 epsilon: float = None):
        """
        Initialize running normalizer

        Args:
            shape: Shape of the data to normalize (can be int or tuple)
            update_rate: Rate at which running statistics are updated (alpha)
            epsilon: Small constant for numerical stability
        """
        if isinstance(shape, int):
            shape = (shape,)

        self.shape = shape
        self.update_rate = update_rate if update_rate else config.NORM_UPDATE_RATE
        self.epsilon = epsilon if epsilon else config.NORM_EPSILON

        # Initialize running statistics
        self.running_mean = np.zeros(shape, dtype=np.float32)
        self.running_std = np.ones(shape, dtype=np.float32)
        self.count = 0

    def update(self, data: np.ndarray):
        """
        Update running statistics with new data

        Args:
            data: New data point(s) to update statistics with
        """
        data = np.array(data, dtype=np.float32)

        if data.shape[-len(self.shape):] != self.shape:
            raise ValueError(f"Data shape {data.shape} doesn't match normalizer shape {self.shape}")

        # Handle batch dimension
        if len(data.shape) > len(self.shape):
            # Batch of data
            batch_mean = np.mean(data, axis=0)
            batch_std = np.std(data, axis=0)
        else:
            # Single data point
            batch_mean = data
            batch_std = np.zeros_like(data)

        # Update running statistics using exponential moving average
        if self.count == 0:
            self.running_mean = batch_mean
            self.running_std = np.maximum(batch_std, self.epsilon)
        else:
            self.running_mean = (1 - self.update_rate) * self.running_mean + \
                                self.update_rate * batch_mean
            self.running_std = (1 - self.update_rate) * self.running_std + \
                               self.update_rate * np.maximum(batch_std, self.epsilon)

        self.count += 1

    def normalize(self, data: np.ndarray, update: bool = False) -> np.ndarray:
        """
        Normalize data using running statistics

        Args:
            data: Data to normalize
            update: Whether to update running statistics with this data

        Returns:
            Normalized data
        """
        data = np.array(data, dtype=np.float32)

        if update:
            self.update(data)

        # Normalize: (x - mean) / (std + epsilon)
        normalized = (data - self.running_mean) / (self.running_std + self.epsilon)

        return normalized

    def denormalize(self, normalized_data: np.ndarray) -> np.ndarray:
        """
        Convert normalized data back to original scale

        Args:
            normalized_data: Normalized data

        Returns:
            Denormalized data
        """
        denormalized = normalized_data * (self.running_std + self.epsilon) + self.running_mean
        return denormalized

    def reset(self):
        """Reset running statistics"""
        self.running_mean = np.zeros(self.shape, dtype=np.float32)
        self.running_std = np.ones(self.shape, dtype=np.float32)
        self.count = 0

    def get_stats(self) -> dict:
        """Get current normalization statistics"""
        return {
            'mean': self.running_mean.copy(),
            'std': self.running_std.copy(),
            'count': self.count
        }

    def set_stats(self, mean: np.ndarray, std: np.ndarray):
        """
        Manually set normalization statistics

        Args:
            mean: Mean values
            std: Standard deviation values
        """
        self.running_mean = np.array(mean, dtype=np.float32)
        self.running_std = np.array(std, dtype=np.float32)

    def __repr__(self):
        return f"RunningNormalizer(shape={self.shape}, count={self.count})"


class MinMaxNormalizer:
    """
    Min-Max normalizer that scales data to [0, 1] range
    """

    def __init__(self,
                 shape: Union[int, tuple],
                 feature_mins: Optional[np.ndarray] = None,
                 feature_maxs: Optional[np.ndarray] = None):
        """
        Initialize min-max normalizer

        Args:
            shape: Shape of the data to normalize
            feature_mins: Known minimum values for each feature
            feature_maxs: Known maximum values for each feature
        """
        if isinstance(shape, int):
            shape = (shape,)

        self.shape = shape

        if feature_mins is not None:
            self.min_vals = np.array(feature_mins, dtype=np.float32)
        else:
            self.min_vals = np.full(shape, np.inf, dtype=np.float32)

        if feature_maxs is not None:
            self.max_vals = np.array(feature_maxs, dtype=np.float32)
        else:
            self.max_vals = np.full(shape, -np.inf, dtype=np.float32)

    def update(self, data: np.ndarray):
        """
        Update min and max values with new data

        Args:
            data: New data to update statistics with
        """
        data = np.array(data, dtype=np.float32)

        if len(data.shape) > len(self.shape):
            # Batch of data
            batch_min = np.min(data, axis=0)
            batch_max = np.max(data, axis=0)
        else:
            # Single data point
            batch_min = data
            batch_max = data

        self.min_vals = np.minimum(self.min_vals, batch_min)
        self.max_vals = np.maximum(self.max_vals, batch_max)

    def normalize(self, data: np.ndarray, update: bool = False) -> np.ndarray:
        """
        Normalize data to [0, 1] range

        Args:
            data: Data to normalize
            update: Whether to update min/max with this data

        Returns:
            Normalized data in [0, 1] range
        """
        data = np.array(data, dtype=np.float32)

        if update:
            self.update(data)

        # Avoid division by zero
        range_vals = self.max_vals - self.min_vals
        range_vals = np.where(range_vals < 1e-8, 1.0, range_vals)

        # Normalize: (x - min) / (max - min)
        normalized = (data - self.min_vals) / range_vals

        # Clip to [0, 1] to handle outliers
        normalized = np.clip(normalized, 0.0, 1.0)

        return normalized

    def denormalize(self, normalized_data: np.ndarray) -> np.ndarray:
        """
        Convert normalized data back to original scale

        Args:
            normalized_data: Normalized data in [0, 1] range

        Returns:
            Denormalized data
        """
        range_vals = self.max_vals - self.min_vals
        denormalized = normalized_data * range_vals + self.min_vals
        return denormalized

    def reset(self):
        """Reset min and max values"""
        self.min_vals = np.full(self.shape, np.inf, dtype=np.float32)
        self.max_vals = np.full(self.shape, -np.inf, dtype=np.float32)


class StateNormalizer:
    """
    Specialized normalizer for the state vector in CPU scheduling
    Combines different normalization strategies for different features
    """

    def __init__(self, state_dim: int = None):
        """
        Initialize state normalizer

        Args:
            state_dim: Dimension of state vector
        """
        self.state_dim = state_dim if state_dim else config.STATE_DIM

        # Use running normalizer for the entire state
        self.normalizer = RunningNormalizer(shape=self.state_dim)

        # Track if normalization is enabled
        self.enabled = config.NORMALIZE_STATE

    def normalize_state(self, state: np.ndarray, update: bool = True) -> np.ndarray:
        """
        Normalize state vector

        Args:
            state: Raw state vector
            update: Whether to update running statistics

        Returns:
            Normalized state vector
        """
        if not self.enabled:
            return state

        return self.normalizer.normalize(state, update=update)

    def denormalize_state(self, normalized_state: np.ndarray) -> np.ndarray:
        """
        Denormalize state vector

        Args:
            normalized_state: Normalized state

        Returns:
            Original scale state
        """
        if not self.enabled:
            return normalized_state

        return self.normalizer.denormalize(normalized_state)

    def reset(self):
        """Reset normalization statistics"""
        self.normalizer.reset()

    def get_stats(self) -> dict:
        """Get normalization statistics"""
        return self.normalizer.get_stats()

    def set_stats(self, mean: np.ndarray, std: np.ndarray):
        """Set normalization statistics"""
        self.normalizer.set_stats(mean, std)


if __name__ == "__main__":
    print("Testing Normalization Utilities...")

    # Test RunningNormalizer
    print("\n1. Testing RunningNormalizer:")
    normalizer = RunningNormalizer(shape=3)

    # Generate some data
    data_points = [
        np.array([1.0, 2.0, 3.0]),
        np.array([2.0, 4.0, 6.0]),
        np.array([3.0, 6.0, 9.0]),
        np.array([4.0, 8.0, 12.0])
    ]

    print("\nNormalizing data points:")
    for i, data in enumerate(data_points):
        normalized = normalizer.normalize(data, update=True)
        print(f"Point {i + 1}: {data} -> {normalized}")

    print(f"\nRunning statistics:")
    stats = normalizer.get_stats()
    print(f"Mean: {stats['mean']}")
    print(f"Std:  {stats['std']}")

    # Test batch normalization
    print("\n2. Testing Batch Normalization:")
    batch = np.array([
        [1.0, 2.0, 3.0],
        [2.0, 4.0, 6.0],
        [3.0, 6.0, 9.0]
    ])

    batch_normalized = normalizer.normalize(batch, update=False)
    print(f"Batch shape: {batch.shape}")
    print(f"Normalized batch:\n{batch_normalized}")

    # Test MinMaxNormalizer
    print("\n3. Testing MinMaxNormalizer:")
    minmax_norm = MinMaxNormalizer(
        shape=3,
        feature_mins=[0.0, 0.0, 0.0],
        feature_maxs=[10.0, 20.0, 30.0]
    )

    test_data = np.array([5.0, 10.0, 15.0])
    normalized = minmax_norm.normalize(test_data)
    denormalized = minmax_norm.denormalize(normalized)

    print(f"Original:     {test_data}")
    print(f"Normalized:   {normalized}")
    print(f"Denormalized: {denormalized}")

    # Test StateNormalizer
    print("\n4. Testing StateNormalizer:")
    state_norm = StateNormalizer(state_dim=9)

    # Simulate state vectors
    states = [
        np.array([3, 10, 2, 15, 1, 20, 2, 5.0, 8.0]),
        np.array([2, 5, 3, 10, 0, 0, 3, 3.0, 6.0]),
        np.array([4, 20, 1, 8, 2, 25, 1, 7.0, 10.0])
    ]

    print("\nNormalizing state vectors:")
    for i, state in enumerate(states):
        norm_state = state_norm.normalize_state(state, update=True)
        print(f"State {i + 1}:")
        print(f"  Raw:        {state}")
        print(f"  Normalized: {norm_state}")

    print(f"\nState normalizer statistics:")
    stats = state_norm.get_stats()
    print(f"Mean: {stats['mean']}")
    print(f"Std:  {stats['std']}")