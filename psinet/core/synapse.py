## Developer: inkbytefo
## Modified: 2025-11-08

import taichi as ti
import numpy as np
from typing import Optional, Dict, Any, Tuple
from ..taichi_backend.stdp_sparse import STDPSparseKernel
from .neuron import BionicNeuron


class BionicSynapse:
    """
    PSINet synapse implementation using Taichi backend.
    Replaces Brian2-based synapse.py with GPU-accelerated sparse STDP synapses.
    Features dopamine modulation and 90% sparsity for 4090 optimization.
    """
    
    def __init__(self, pre_neurons: BionicNeuron, post_neurons: BionicNeuron,
                 tau_pre: float = 0.02, tau_post: float = 0.02, w_max: float = 0.01,
                 a_pre: float = 0.01, a_post: float = -0.0105, initial_weight_max: float = 0.01,
                 dt: float = 0.001, sparsity: float = 0.9):
        """
        Initialize Taichi-based BionicSynapse population.
        
        Args:
            pre_neurons: Pre-synaptic BionicNeuron population
            post_neurons: Post-synaptic BionicNeuron population
            tau_pre: Pre-synaptic STDP trace time constant in seconds
            tau_post: Post-synaptic STDP trace time constant in seconds
            w_max: Maximum synaptic weight
            a_pre: LTP amplitude (positive weight change)
            a_post: LTD amplitude (negative weight change)
            initial_weight_max: Maximum initial weight
            dt: Simulation timestep in seconds
            sparsity: Connection sparsity ratio (0.9 = 90% sparse)
        """
        self.pre_neurons = pre_neurons
        self.post_neurons = post_neurons
        self.tau_pre = tau_pre
        self.tau_post = tau_post
        self.w_max = w_max
        self.a_pre = a_pre
        self.a_post = a_post
        self.initial_weight_max = initial_weight_max
        self.dt = dt
        self.sparsity = sparsity
        
        # Initialize Taichi STDP kernel
        self.kernel = STDPSparseKernel(
            pre_neurons=pre_neurons.num_neurons,
            post_neurons=post_neurons.num_neurons,
            dt=dt,
            sparsity=sparsity
        )
        
        # Set STDP parameters
        self.kernel.set_parameters(
            tau_pre=tau_pre,
            tau_post=tau_post,
            a_plus=a_pre,
            a_minus=a_post,
            w_max=w_max
        )
        
        # Postsynaptic current buffer
        self.postsynaptic_current = np.zeros(post_neurons.num_neurons, dtype=np.float32)
        
        # Weight history for monitoring
        self.weight_history = []
        self.max_history_length = 100
        
        # Dopamine level (global neuromodulation)
        self.dopamine_level = 1.0
        
        # Performance metrics
        self.total_weight_change = 0.0
        self.simulation_time = 0.0
        
    def update(self, pre_spikes: Optional[np.ndarray] = None, 
               post_spikes: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Update synapse states for one timestep.
        
        Args:
            pre_spikes: Optional pre-synaptic spike array. If None, gets from pre_neurons.
            post_spikes: Optional post-synaptic spike array. If None, gets from post_neurons.
            
        Returns:
            Postsynaptic current array
        """
        # Get spikes from neuron populations if not provided
        if pre_spikes is None:
            pre_spikes = self.pre_neurons.get_spikes()
        if post_spikes is None:
            post_spikes = self.post_neurons.get_spikes()
        
        # Update kernel and get postsynaptic current
        self.postsynaptic_current = self.kernel.update(pre_spikes, post_spikes)
        
        # Update metrics
        self.simulation_time += self.dt
        
        # Store weight history periodically
        if int(self.simulation_time / self.dt) % 100 == 0:  # Every 100 timesteps
            self._record_weight_state()
        
        return self.postsynaptic_current
    
    def _record_weight_state(self):
        """Record current weight state for history tracking."""
        weights = self.kernel.get_weights()
        non_zero_weights = weights[weights > 0]
        
        if len(non_zero_weights) > 0:
            self.weight_history.append({
                'time': self.simulation_time,
                'mean_weight': float(np.mean(non_zero_weights)),
                'max_weight': float(np.max(weights)),
                'min_weight': float(np.min(weights)),
                'sparsity': 1.0 - (len(non_zero_weights) / (self.pre_neurons.num_neurons * self.post_neurons.num_neurons)),
                'total_connections': len(non_zero_weights)
            })
            
            # Limit history length
            if len(self.weight_history) > self.max_history_length:
                self.weight_history.pop(0)
    
    def get_weights(self) -> np.ndarray:
        """Get current weight matrix."""
        return self.kernel.get_weights()
    
    def get_sparse_weights(self):
        """Get sparse weight matrix."""
        return self.kernel.get_sparse_weights()
    
    def get_postsynaptic_current(self) -> np.ndarray:
        """Get current postsynaptic current."""
        return self.postsynaptic_current.copy()
    
    def set_dopamine(self, dopamine_level: float):
        """
        Set global dopamine level for neuromodulation.
        
        Args:
            dopamine_level: Dopamine level (0.0 to 2.0, where 1.0 is baseline)
                           Higher dopamine reduces LTD (depression)
        """
        self.dopamine_level = np.clip(dopamine_level, 0.0, 2.0)
        self.kernel.set_dopamine(self.dopamine_level)
    
    def set_learning_params(self, tau_pre: Optional[float] = None,
                           tau_post: Optional[float] = None,
                           w_max: Optional[float] = None,
                           a_pre: Optional[float] = None,
                           a_post: Optional[float] = None):
        """Update STDP learning parameters."""
        params = {}
        if tau_pre is not None:
            params['tau_pre'] = tau_pre
            self.tau_pre = tau_pre
        if tau_post is not None:
            params['tau_post'] = tau_post
            self.tau_post = tau_post
        if w_max is not None:
            params['w_max'] = w_max
            self.w_max = w_max
        if a_pre is not None:
            params['a_plus'] = a_pre
            self.a_pre = a_pre
        if a_post is not None:
            params['a_minus'] = a_post
            self.a_post = a_post
        
        if params:
            self.kernel.set_parameters(**params)
    
    def reset(self):
        """Reset synapse states and history."""
        self.kernel.reset()
        self.postsynaptic_current.fill(0)
        self.weight_history.clear()
        self.total_weight_change = 0.0
        self.simulation_time = 0.0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get synapse population statistics."""
        weight_stats = self.kernel.get_weight_stats()
        trace_stats = self.kernel.get_trace_stats()
        
        return {
            'pre_neurons': self.pre_neurons.num_neurons,
            'post_neurons': self.post_neurons.num_neurons,
            'total_connections': weight_stats['total_connections'],
            'sparsity': weight_stats['sparsity'],
            'mean_weight': weight_stats['mean_weight'],
            'max_weight': weight_stats['max_weight'],
            'min_weight': weight_stats['min_weight'],
            'dopamine_level': self.dopamine_level,
            'pre_trace_mean': trace_stats['pre_trace_mean'],
            'post_trace_mean': trace_stats['post_trace_mean'],
            'simulation_time': self.simulation_time,
            'postsynaptic_current_mean': float(np.mean(self.postsynaptic_current)),
            'postsynaptic_current_max': float(np.max(self.postsynaptic_current))
        }
    
    def get_weight_distribution(self) -> Dict[str, Any]:
        """Get detailed weight distribution statistics."""
        weights = self.get_weights()
        non_zero_weights = weights[weights > 0]
        
        if len(non_zero_weights) == 0:
            return {
                'count': 0,
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'median': 0.0,
                'percentiles': {}
            }
        
        return {
            'count': len(non_zero_weights),
            'mean': float(np.mean(non_zero_weights)),
            'std': float(np.std(non_zero_weights)),
            'min': float(np.min(non_zero_weights)),
            'max': float(np.max(non_zero_weights)),
            'median': float(np.median(non_zero_weights)),
            'percentiles': {
                '25': float(np.percentile(non_zero_weights, 25)),
                '50': float(np.percentile(non_zero_weights, 50)),
                '75': float(np.percentile(non_zero_weights, 75)),
                '90': float(np.percentile(non_zero_weights, 90)),
                '95': float(np.percentile(non_zero_weights, 95)),
                '99': float(np.percentile(non_zero_weights, 99))
            }
        }
    
    def get_strongest_connections(self, n: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the strongest n connections.
        
        Args:
            n: Number of strongest connections to return
            
        Returns:
            Tuple of (pre_indices, post_indices, weights)
        """
        weights = self.get_weights()
        
        # Get indices of top n weights
        flat_indices = np.argpartition(weights.flatten(), -n)[-n:]
        pre_indices, post_indices = np.unravel_index(flat_indices, weights.shape)
        top_weights = weights[pre_indices, post_indices]
        
        # Sort by weight strength
        sort_order = np.argsort(top_weights)[::-1]
        return pre_indices[sort_order], post_indices[sort_order], top_weights[sort_order]
    
    def apply_reward_signal(self, reward: float):
        """
        Apply reward signal as dopamine modulation.
        
        Args:
            reward: Reward signal (positive increases dopamine, negative decreases)
        """
        # Convert reward to dopamine level
        # Baseline is 1.0, reward modulates it
        dopamine_change = reward * 0.5  # Scale factor for dopamine modulation
        new_dopamine = self.dopamine_level + dopamine_change
        self.set_dopamine(new_dopamine)
    
    def normalize_weights(self, target_sum: Optional[float] = None):
        """
        Normalize weights to maintain homeostasis.
        
        Args:
            target_sum: Target sum of all weights. If None, maintains current sum.
        """
        weights = self.get_weights()
        non_zero_weights = weights[weights > 0]
        
        if len(non_zero_weights) == 0:
            return
        
        current_sum = np.sum(non_zero_weights)
        
        if target_sum is None:
            target_sum = current_sum
        
        if current_sum > 0:
            scale_factor = target_sum / current_sum
            # Apply scaling to Taichi kernel
            scaled_weights = weights * scale_factor
            self.kernel.weights.from_numpy(scaled_weights)
    
    def __repr__(self):
        return f"BionicSynapse(pre={self.pre_neurons.num_neurons}, post={self.post_neurons.num_neurons}, backend=Taichi, sparsity={self.sparsity})"