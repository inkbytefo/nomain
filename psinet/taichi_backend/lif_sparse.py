## Developer: inkbytefo
## Modified: 2025-11-08

import taichi as ti
import numpy as np
from typing import Tuple, Optional


@ti.data_oriented
class LIFSparseKernel:
    """
    High-performance Taichi kernel for Leaky Integrate-and-Fire neurons with adaptive threshold.
    Supports 90% sparsity and GPU acceleration for 4090.
    """
    
    def __init__(self, num_neurons: int, dt: float = 0.001, sparsity: float = 0.9):
        """
        Initialize LIF kernel with sparse connectivity.
        
        Args:
            num_neurons: Number of neurons in the population
            dt: Simulation timestep in seconds
            sparsity: Connection sparsity ratio (0.9 = 90% sparse)
        """
        self.num_neurons = num_neurons
        self.dt = dt
        self.sparsity = sparsity
        
        # Initialize Taichi fields
        self.v = ti.field(dtype=ti.f32, shape=num_neurons)  # Membrane potential
        self.theta = ti.field(dtype=ti.f32, shape=num_neurons)  # Adaptive threshold
        self.refractory = ti.field(dtype=ti.i32, shape=num_neurons)  # Refractory counter
        self.spikes = ti.field(dtype=ti.i32, shape=num_neurons)  # Spike output
        
        # Parameters
        self.tau_mem = ti.field(dtype=ti.f32, shape=())  # Membrane time constant
        self.tau_theta = ti.field(dtype=ti.f32, shape=())  # Adaptation time constant
        self.threshold_base = ti.field(dtype=ti.f32, shape=())  # Base threshold
        self.reset_potential = ti.field(dtype=ti.f32, shape=())  # Reset potential
        self.delta_theta = ti.field(dtype=ti.f32, shape=())  # Threshold increment
        
        # Sparse connectivity mask
        self.connectivity_mask = self._create_sparse_mask()
        
        # Initialize parameters
        self._init_parameters()
        
    def _create_sparse_mask(self) -> ti.field:
        """Create sparse connectivity mask for 90% sparsity."""
        mask = ti.field(dtype=ti.i32, shape=(self.num_neurons, self.num_neurons))
        
        @ti.kernel
        def init_mask():
            for i, j in ti.ndrange(self.num_neurons, self.num_neurons):
                if ti.random() < self.sparsity:
                    mask[i, j] = 0
                else:
                    mask[i, j] = 1
        
        init_mask()
        return mask
    
    def _init_parameters(self):
        """Initialize default LIF parameters."""
        self.tau_mem[None] = 0.02  # 20ms membrane time constant
        self.tau_theta[None] = 0.1  # 100ms adaptation time constant
        self.threshold_base[None] = 1.0  # Base threshold
        self.reset_potential[None] = 0.0  # Reset to 0
        self.delta_theta[None] = 0.1  # Threshold increment per spike
        
        # Initialize state
        self._reset_state()
    
    @ti.kernel
    def _reset_state(self):
        """Reset all neuron states to initial conditions."""
        for i in range(self.num_neurons):
            self.v[i] = 0.0
            self.theta[i] = 0.0
            self.refractory[i] = 0
            self.spikes[i] = 0
    
    @ti.kernel
    def update(self, input_current: ti.types.ndarray()):
        """
        Update neuron states for one timestep.
        
        Args:
            input_current: Input current array (numpy array converted to Taichi)
        """
        # Clear spikes from previous timestep
        for i in range(self.num_neurons):
            self.spikes[i] = 0
        
        # Update each neuron
        for i in range(self.num_neurons):
            if self.refractory[i] > 0:
                # Neuron is in refractory period
                self.refractory[i] -= 1
                self.v[i] = self.reset_potential[None]
            else:
                # Update membrane potential (leaky integration)
                dv_dt = (input_current[i] - self.v[i]) / self.tau_mem[None]
                self.v[i] += dv_dt * self.dt
                
                # Update adaptive threshold
                dtheta_dt = -self.theta[i] / self.tau_theta[None]
                self.theta[i] += dtheta_dt * self.dt
                
                # Check for spike
                effective_threshold = self.threshold_base[None] + self.theta[i]
                if self.v[i] > effective_threshold:
                    # Generate spike
                    self.spikes[i] = 1
                    self.v[i] = self.reset_potential[None]
                    self.theta[i] += self.delta_theta[None]
                    self.refractory[i] = 5  # 5 timesteps refractory
    
    def get_spikes(self) -> np.ndarray:
        """Get spike outputs as numpy array."""
        return self.spikes.to_numpy()
    
    def get_membrane_potentials(self) -> np.ndarray:
        """Get membrane potentials as numpy array."""
        return self.v.to_numpy()
    
    def get_thresholds(self) -> np.ndarray:
        """Get adaptive thresholds as numpy array."""
        return self.theta.to_numpy()
    
    def set_parameters(self, tau_mem: Optional[float] = None, 
                      tau_theta: Optional[float] = None,
                      threshold_base: Optional[float] = None,
                      reset_potential: Optional[float] = None,
                      delta_theta: Optional[float] = None):
        """Update LIF parameters."""
        if tau_mem is not None:
            self.tau_mem[None] = tau_mem
        if tau_theta is not None:
            self.tau_theta[None] = tau_theta
        if threshold_base is not None:
            self.threshold_base[None] = threshold_base
        if reset_potential is not None:
            self.reset_potential[None] = reset_potential
        if delta_theta is not None:
            self.delta_theta[None] = delta_theta
    
    def reset(self):
        """Reset all neuron states."""
        self._reset_state()
    
    def get_spike_count(self) -> int:
        """Get total spike count in current timestep."""
        return int(np.sum(self.get_spikes()))
    
    def get_firing_rate(self, window_size: int = 100) -> float:
        """
        Calculate average firing rate over a window.
        
        Args:
            window_size: Number of timesteps to average over
            
        Returns:
            Average firing rate in Hz
        """
        # This would need spike history storage for accurate calculation
        # Simplified version using current timestep
        current_spikes = self.get_spike_count()
        return (current_spikes / self.num_neurons) / self.dt