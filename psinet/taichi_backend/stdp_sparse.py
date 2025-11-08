## Developer: inkbytefo
## Modified: 2025-11-08

import taichi as ti
import numpy as np
from typing import Tuple, Optional
from scipy.sparse import csr_matrix

# Initialize Taichi with T4 GPU support
ti.init(device_memory_GB=15)


@ti.data_oriented
class STDPSparseKernel:
    """
    High-performance Taichi kernel for Spike-Timing-Dependent Plasticity with dopamine modulation.
    Supports 90% sparsity and GPU acceleration for T4.
    """
    
    def __init__(self, pre_neurons: int, post_neurons: int, dt: float = 0.001, sparsity: float = 0.9):
        """
        Initialize STDP kernel with sparse connectivity.
        
        Args:
            pre_neurons: Number of pre-synaptic neurons
            post_neurons: Number of post-synaptic neurons
            dt: Simulation timestep in seconds
            sparsity: Connection sparsity ratio (0.9 = 90% sparse)
        """
        self.pre_neurons = pre_neurons
        self.post_neurons = post_neurons
        self.dt = dt
        self.sparsity = sparsity
        
        # Weight matrix (sparse)
        self.weights = self._create_sparse_weights()
        
        # STDP traces
        self.apre = ti.field(dtype=ti.f32, shape=pre_neurons)  # Pre-synaptic trace
        self.apost = ti.field(dtype=ti.f32, shape=post_neurons)  # Post-synaptic trace
        
        # Spike inputs
        self.pre_spikes = ti.field(dtype=ti.i32, shape=pre_neurons)
        self.post_spikes = ti.field(dtype=ti.i32, shape=post_neurons)
        
        # STDP parameters
        self.tau_pre = ti.field(dtype=ti.f32, shape=())  # Pre trace time constant
        self.tau_post = ti.field(dtype=ti.f32, shape=())  # Post trace time constant
        self.a_plus = ti.field(dtype=ti.f32, shape=())  # LTP amplitude
        self.a_minus = ti.field(dtype=ti.f32, shape=())  # LTD amplitude
        self.w_max = ti.field(dtype=ti.f32, shape=())  # Maximum weight
        self.dopamine = ti.field(dtype=ti.f32, shape=())  # Global dopamine level
        
        # Initialize parameters
        self._init_parameters()
        
    def _create_sparse_weights(self) -> ti.field:
        """Create sparse weight matrix with 90% sparsity."""
        weights = ti.field(dtype=ti.f32, shape=(self.pre_neurons, self.post_neurons))
        
        @ti.kernel
        def init_weights():
            for i, j in ti.ndrange(self.pre_neurons, self.post_neurons):
                if ti.random() < self.sparsity:
                    weights[i, j] = 0.0
                else:
                    weights[i, j] = ti.random() * 0.01  # Small random initial weights
        
        init_weights()
        return weights
    
    def _init_parameters(self):
        """Initialize default STDP parameters."""
        self.tau_pre[None] = 0.02  # 20ms pre trace time constant
        self.tau_post[None] = 0.02  # 20ms post trace time constant
        self.a_plus[None] = 0.01  # LTP amplitude
        self.a_minus[None] = -0.0105  # LTD amplitude (slightly stronger)
        self.w_max[None] = 0.01  # Maximum weight
        self.dopamine[None] = 1.0  # Baseline dopamine level
        
        # Initialize traces
        self._reset_traces()
    
    @ti.kernel
    def _reset_traces(self):
        """Reset STDP traces to zero."""
        for i in range(self.pre_neurons):
            self.apre[i] = 0.0
            self.pre_spikes[i] = 0
        for j in range(self.post_neurons):
            self.apost[j] = 0.0
            self.post_spikes[j] = 0
    
    @ti.kernel
    def update_traces(self):
        """Update STDP traces with exponential decay."""
        # Decay pre-synaptic traces
        for i in range(self.pre_neurons):
            self.apre[i] *= ti.exp(-self.dt / self.tau_pre[None])
            if self.pre_spikes[i] == 1:
                self.apre[i] += 1.0
        
        # Decay post-synaptic traces
        for j in range(self.post_neurons):
            self.apost[j] *= ti.exp(-self.dt / self.tau_post[None])
            if self.post_spikes[j] == 1:
                self.apost[j] += 1.0
    
    @ti.kernel
    def update_weights(self):
        """
        Update synaptic weights based on STDP rules with dopamine modulation.
        Dopamine modulates LTD (a_minus) to reduce negative weight changes.
        """
        for i, j in ti.ndrange(self.pre_neurons, self.post_neurons):
            if self.weights[i, j] > 0:  # Only update existing connections
                # Pre-before-post (LTP)
                if self.pre_spikes[i] == 1 and self.apost[j] > 0:
                    dw = self.a_plus[None] * self.apost[j]
                    self.weights[i, j] += dw
                
                # Post-before-pre (LTD) - modulated by dopamine
                if self.post_spikes[j] == 1 and self.apre[i] > 0:
                    # Dopamine reduces LTD (higher dopamine = less depression)
                    effective_a_minus = self.a_minus[None] * (2.0 - self.dopamine[None])
                    dw = effective_a_minus * self.apre[i]
                    self.weights[i, j] += dw
                
                # Enforce weight bounds
                if self.weights[i, j] < 0:
                    self.weights[i, j] = 0
                elif self.weights[i, j] > self.w_max[None]:
                    self.weights[i, j] = self.w_max[None]
    
    @ti.kernel
    def compute_postsynaptic_current(self, pre_spikes: ti.types.ndarray(ndim=1, dtype=ti.i32)) -> ti.types.ndarray(ndim=1, dtype=ti.f32):
        """
        Compute postsynaptic current from presynaptic spikes and weights.
        
        Args:
            pre_spikes: Presynaptic spike array
            
        Returns:
            Postsynaptic current array
        """
        postsynaptic_current = ti.field(dtype=ti.f32, shape=self.post_neurons)
        
        # Initialize to zero
        for j in range(self.post_neurons):
            postsynaptic_current[j] = 0.0
        
        # Compute weighted sum
        for i, j in ti.ndrange(self.pre_neurons, self.post_neurons):
            if self.weights[i, j] > 0 and pre_spikes[i] == 1:
                postsynaptic_current[j] += self.weights[i, j]
        
        return postsynaptic_current
    
    @ti.kernel
    def step_kernel(self, pre_spikes: ti.types.ndarray(ndim=1, dtype=ti.i32), 
                   post_spikes: ti.types.ndarray(ndim=1, dtype=ti.i32)) -> ti.types.ndarray(ndim=1, dtype=ti.f32):
        """
        Perform one STDP update step in kernel.
        
        Args:
            pre_spikes: Presynaptic spike array
            post_spikes: Postsynaptic spike array
            
        Returns:
            Postsynaptic current array
        """
        # Update spike fields
        for i in range(self.pre_neurons):
            self.pre_spikes[i] = pre_spikes[i]
        for j in range(self.post_neurons):
            self.post_spikes[j] = post_spikes[j]
        
        # Update traces
        for i in range(self.pre_neurons):
            self.apre[i] *= ti.exp(-self.dt / self.tau_pre[None])
            if self.pre_spikes[i] == 1:
                self.apre[i] += 1.0
        
        for j in range(self.post_neurons):
            self.apost[j] *= ti.exp(-self.dt / self.tau_post[None])
            if self.post_spikes[j] == 1:
                self.apost[j] += 1.0
        
        # Update weights
        for i, j in ti.ndrange(self.pre_neurons, self.post_neurons):
            if self.weights[i, j] > 0:  # Only update existing connections
                # Pre-before-post (LTP)
                if self.pre_spikes[i] == 1 and self.apost[j] > 0:
                    dw = self.a_plus[None] * self.apost[j]
                    self.weights[i, j] += dw
                
                # Post-before-pre (LTD) - modulated by dopamine
                if self.post_spikes[j] == 1 and self.apre[i] > 0:
                    # Dopamine reduces LTD (higher dopamine = less depression)
                    effective_a_minus = self.a_minus[None] * (2.0 - self.dopamine[None])
                    dw = effective_a_minus * self.apre[i]
                    self.weights[i, j] += dw
                
                # Enforce weight bounds
                if self.weights[i, j] < 0:
                    self.weights[i, j] = 0
                elif self.weights[i, j] > self.w_max[None]:
                    self.weights[i, j] = self.w_max[None]
        
        # Compute postsynaptic current
        postsynaptic_current = ti.field(dtype=ti.f32, shape=self.post_neurons)
        for j in range(self.post_neurons):
            postsynaptic_current[j] = 0.0
        
        for i, j in ti.ndrange(self.pre_neurons, self.post_neurons):
            if self.weights[i, j] > 0 and pre_spikes[i] == 1:
                postsynaptic_current[j] += self.weights[i, j]
        
        return postsynaptic_current
    
    def step(self, pre_spikes: np.ndarray, post_spikes: np.ndarray) -> np.ndarray:
        """
        Perform one STDP update step.
        
        Args:
            pre_spikes: Presynaptic spike array
            post_spikes: Postsynaptic spike array
            
        Returns:
            Postsynaptic current array
        """
        # Convert to proper dtype
        pre_spikes_typed = pre_spikes.astype(np.int32)
        post_spikes_typed = post_spikes.astype(np.int32)
        
        # Update spike fields
        self.pre_spikes.from_numpy(pre_spikes_typed)
        self.post_spikes.from_numpy(post_spikes_typed)
        
        # Update traces
        self.update_traces()
        
        # Update weights
        self.update_weights()
        
        # Compute postsynaptic current
        postsynaptic_current = self.compute_postsynaptic_current(pre_spikes_typed)
        
        return postsynaptic_current.to_numpy()
    
    def get_weights(self) -> np.ndarray:
        """Get weight matrix as numpy array."""
        return self.weights.to_numpy()
    
    def get_sparse_weights(self) -> csr_matrix:
        """Get weight matrix as scipy sparse CSR matrix."""
        dense_weights = self.get_weights()
        return csr_matrix(dense_weights)
    
    def set_dopamine(self, dopamine_level: float):
        """
        Set global dopamine level.
        
        Args:
            dopamine_level: Dopamine level (0.0 to 2.0, where 1.0 is baseline)
        """
        self.dopamine[None] = np.clip(dopamine_level, 0.0, 2.0)
    
    def set_parameters(self, tau_pre: Optional[float] = None,
                      tau_post: Optional[float] = None,
                      a_plus: Optional[float] = None,
                      a_minus: Optional[float] = None,
                      w_max: Optional[float] = None):
        """Update STDP parameters."""
        if tau_pre is not None:
            self.tau_pre[None] = tau_pre
        if tau_post is not None:
            self.tau_post[None] = tau_post
        if a_plus is not None:
            self.a_plus[None] = a_plus
        if a_minus is not None:
            self.a_minus[None] = a_minus
        if w_max is not None:
            self.w_max[None] = w_max
    
    def reset(self):
        """Reset STDP traces and spikes."""
        self._reset_traces()
    
    def get_weight_stats(self) -> dict:
        """Get weight statistics."""
        weights = self.get_weights()
        non_zero_weights = weights[weights > 0]
        
        return {
            'mean_weight': np.mean(non_zero_weights) if len(non_zero_weights) > 0 else 0.0,
            'max_weight': np.max(weights),
            'min_weight': np.min(weights),
            'sparsity': 1.0 - (len(non_zero_weights) / (self.pre_neurons * self.post_neurons)),
            'total_connections': len(non_zero_weights)
        }
    
    def get_trace_stats(self) -> dict:
        """Get trace statistics."""
        apre_np = self.apre.to_numpy()
        apost_np = self.apost.to_numpy()
        
        return {
            'pre_trace_mean': np.mean(apre_np),
            'pre_trace_max': np.max(apre_np),
            'post_trace_mean': np.mean(apost_np),
            'post_trace_max': np.max(apost_np),
            'dopamine_level': self.dopamine[None]
        }