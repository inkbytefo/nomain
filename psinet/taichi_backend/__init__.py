## Developer: inkbytefo
## Modified: 2025-11-08

"""
Taichi-based SNN backend for PSINet.
High-performance GPU-accelerated spiking neural network simulation.
"""

from .lif_sparse import LIFSparseKernel
from .stdp_sparse import STDPSparseKernel
from .converter import Brian2ToTaichiConverter

__all__ = [
    'LIFSparseKernel',
    'STDPSparseKernel', 
    'Brian2ToTaichiConverter'
]