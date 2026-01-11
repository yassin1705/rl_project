"""
Baseline Agents for CPU Scheduling with DVFS
"""

from .fifo_fixed_freq import (
    FIFOFixedFreqAgent,
    FIFOMaxFreqAgent,
    FIFOMinFreqAgent
)

__all__ = [
    'FIFOFixedFreqAgent',
    'FIFOMaxFreqAgent', 
    'FIFOMinFreqAgent'
]
