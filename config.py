"""
Configuration file for CPU Scheduling with DVFS RL Project
Contains all hyperparameters, system parameters, and training settings
"""

# ============================================================================
# CPU System Configuration
# ============================================================================

# Number of CPU cores in the system
NUM_CORES = 6

# Available DVFS frequency levels (in GHz)
# Discrete frequency choices available for scheduling
FREQUENCY_LEVELS = [1.5, 2.0, 2.5, 3.1]

# Number of discrete frequency levels
NUM_FREQUENCIES = len(FREQUENCY_LEVELS)

# ============================================================================
# Task Workload Configuration
# ============================================================================

# Task categories
TASK_CATEGORIES = ['Low', 'Medium', 'Heavy']
NUM_TASK_CATEGORIES = len(TASK_CATEGORIES)

# Workload sizes for each task category (in arbitrary units)
WORKLOAD_SIZES = {
    'Low': 3,
    'Medium': 6,
    'Heavy': 10
}

MAX_QUEUE_LEN = 20
# ============================================================================
# Task Generation Configuration
# ============================================================================

# Task arrival rate (Poisson process parameter - tasks per time step)
TASK_ARRIVAL_RATE = 1.0

# Probability distribution for task categories
# Must sum to 1.0
TASK_CATEGORY_PROBS = {
    'Low': 0.3,
    'Medium': 0.5,
    'Heavy': 0.2
}

# ============================================================================
# Episode Configuration
# ============================================================================

# Maximum number of scheduling decisions per episode
MAX_SCHEDULING_DECISIONS = 200

# Maximum allowed waiting time before starvation penalty
# Episode terminates if any task exceeds this age
MAX_WAITING_TIME = 20

# Large negative penalty for starvation constraint violation
STARVATION_PENALTY = -1000.0

# ============================================================================
# Reward Function Configuration
# ============================================================================

# Anti-starvation weights for different task categories
# Higher weight = more reward for serving older tasks of this type
ANTI_STARVATION_WEIGHTS = {
    'Low': 1.0,
    'Medium': 2.0,
    'Heavy': 3.0
}

# Energy-latency trade-off parameter (beta)
# Higher beta = more penalty for energy consumption
# Lower beta = more focus on latency reduction
BETA = 0.1

# ============================================================================
# State Space Configuration
# ============================================================================

# State vector size
# [Q_Low_len, Q_Low_age, Q_Med_len, Q_Med_age, Q_Heavy_len, Q_Heavy_age,
#  FreeCores, AvgRemain, MaxRemain]
STATE_DIM = 9

# ============================================================================
# Action Space Configuration
# ============================================================================

# Total action space size: (Task Categories × Frequency Levels)
ACTION_DIM = NUM_TASK_CATEGORIES * NUM_FREQUENCIES

# ============================================================================
# PPO Training Configuration
# ============================================================================

# Learning rates
LEARNING_RATE_ACTOR = 3e-4
LEARNING_RATE_CRITIC = 1e-3

# PPO-specific hyperparameters
GAMMA = 0.99  # Discount factor
GAE_LAMBDA = 0.95  # GAE parameter for advantage estimation
CLIP_EPSILON = 0.2  # PPO clipping parameter
ENTROPY_COEF = 0.01  # Entropy bonus for exploration
VALUE_LOSS_COEF = 0.5  # Value loss coefficient

# Training parameters
NUM_EPOCHS = 1000  # Total training epochs
PPO_EPOCHS = 10  # Number of PPO update epochs per batch
BATCH_SIZE = 64  # Minibatch size for PPO updates
BUFFER_SIZE = 2048  # Experience buffer size before update

# Gradient clipping
MAX_GRAD_NORM = 0.5

# ============================================================================
# Neural Network Architecture
# ============================================================================

# Hidden layer sizes for actor and critic networks
HIDDEN_SIZES = [128, 64]

# Activation function ('relu', 'tanh', 'elu')
ACTIVATION = 'relu'

# ============================================================================
# Normalization Configuration
# ============================================================================

# Enable/disable state normalization
NORMALIZE_STATE = True

# Running mean/std update rate for normalization
NORM_UPDATE_RATE = 0.01

# Small epsilon for numerical stability in normalization
NORM_EPSILON = 1e-8

# ============================================================================
# Logging and Evaluation Configuration
# ============================================================================

# Logging frequency (in episodes)
LOG_INTERVAL = 10

# Evaluation frequency (in episodes)
EVAL_INTERVAL = 50

# Number of evaluation episodes
NUM_EVAL_EPISODES = 10

# Save model checkpoint frequency (in episodes)
SAVE_INTERVAL = 100

# Paths
LOG_DIR = './logs'
MODEL_DIR = './models'
RESULTS_DIR = './results'

# ============================================================================
# Baseline Configurations
# ============================================================================

# Baseline policies for comparison
BASELINE_POLICIES = [
    'fifo_fixed_freq',  # FIFO with fixed middle frequency
    'fifo_max_freq',    # FIFO with maximum frequency
    'random_freq'       # Random frequency selection
]

# Fixed frequency for FIFO baseline (index in FREQUENCY_LEVELS)
FIFO_FIXED_FREQ_INDEX = 2  # 2.0 GHz

# ============================================================================
# Reproducibility
# ============================================================================

# Random seed for reproducibility
RANDOM_SEED = 42

# ============================================================================
# Verbose and Debug Settings
# ============================================================================

# Verbosity levels: 0 (silent), 1 (minimal), 2 (detailed)
VERBOSE = 1

# Enable detailed logging of each scheduling decision
DEBUG_MODE = False

# ============================================================================
# Helper Functions
# ============================================================================

def get_task_category_index(category):
    """Convert task category name to index"""
    return TASK_CATEGORIES.index(category)

def get_action_from_indices(task_idx, freq_idx):
    """Convert task category index and frequency index to action"""
    return task_idx * NUM_FREQUENCIES + freq_idx

def get_indices_from_action(action):
    """Convert action to task category index and frequency index"""
    task_idx = action // NUM_FREQUENCIES
    freq_idx = action % NUM_FREQUENCIES
    return task_idx, freq_idx

def validate_config():
    """Validate configuration parameters"""
    assert sum(TASK_CATEGORY_PROBS.values()) == 1.0, "Task probabilities must sum to 1.0"
    assert len(TASK_CATEGORIES) == len(WORKLOAD_SIZES), "Mismatch in task categories"
    assert len(TASK_CATEGORIES) == len(ANTI_STARVATION_WEIGHTS), "Mismatch in weights"
    assert BETA >= 0, "Beta must be non-negative"
    assert MAX_WAITING_TIME > 0, "Max waiting time must be positive"
    print("Configuration validated successfully!")

if __name__ == "__main__":
    validate_config()
    print("\n=== CPU Scheduling DVFS RL Configuration ===")
    print(f"Cores: {NUM_CORES}")
    print(f"Frequencies: {FREQUENCY_LEVELS}")
    print(f"Task Categories: {TASK_CATEGORIES}")
    print(f"State Dimension: {STATE_DIM}")
    print(f"Action Dimension: {ACTION_DIM}")
    print(f"Max Scheduling Decisions: {MAX_SCHEDULING_DECISIONS}")
    print("=" * 45)