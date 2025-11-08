## Developer: inkbytefo
## Modified: 2025-11-08

import taichi as ti
import numpy as np
from typing import Tuple, Dict, Any, Optional
from brian2 import NeuronGroup, Synapses
from .lif_sparse import LIFSparseKernel
from .stdp_sparse import STDPSparseKernel


class Brian2ToTaichiConverter:
    """
    Converts Brian2 NeuronGroup and Synapses to Taichi tensor representations.
    Enables migration from CPU-only Brian2 to GPU-accelerated Taichi backend.
    """
    
    def __init__(self, dt: float = 0.001, sparsity: float = 0.9):
        """
        Initialize converter.
        
        Args:
            dt: Simulation timestep in seconds
            sparsity: Connection sparsity ratio (0.9 = 90% sparse)
        """
        self.dt = dt
        self.sparsity = sparsity
        self.converted_neurons = {}
        self.converted_synapses = {}
        
    def from_brian2_neuron_group(self, group: NeuronGroup, name: str = None) -> LIFSparseKernel:
        """
        Convert Brian2 NeuronGroup to Taichi LIF kernel.
        
        Args:
            group: Brian2 NeuronGroup object
            name: Optional name for the converted neuron group
            
        Returns:
            LIFSparseKernel instance
        """
        if name is None:
            name = f"neurons_{group.N}"
        
        # Extract parameters from Brian2 group
        num_neurons = group.N
        
        # Create Taichi LIF kernel
        lif_kernel = LIFSparseKernel(
            num_neurons=num_neurons,
            dt=self.dt,
            sparsity=self.sparsity
        )
        
        # Extract and convert parameters
        self._extract_neuron_parameters(group, lif_kernel)
        
        # Store converted neuron
        self.converted_neurons[name] = lif_kernel
        
        return lif_kernel
    
    def from_brian2_synapses(self, synapses: Synapses, pre_kernel: LIFSparseKernel, 
                           post_kernel: LIFSparseKernel, name: str = None) -> STDPSparseKernel:
        """
        Convert Brian2 Synapses to Taichi STDP kernel.
        
        Args:
            synapses: Brian2 Synapses object
            pre_kernel: Pre-synaptic Taichi LIF kernel
            post_kernel: Post-synaptic Taichi LIF kernel
            name: Optional name for the converted synapses
            
        Returns:
            STDPSparseKernel instance
        """
        if name is None:
            name = f"synapses_{synapses.source.N}_{synapses.target.N}"
        
        # Get dimensions
        pre_neurons = pre_kernel.num_neurons
        post_neurons = post_kernel.num_neurons
        
        # Create Taichi STDP kernel
        stdp_kernel = STDPSparseKernel(
            pre_neurons=pre_neurons,
            post_neurons=post_neurons,
            dt=self.dt,
            sparsity=self.sparsity
        )
        
        # Extract and convert parameters
        self._extract_synapse_parameters(synapses, stdp_kernel)
        
        # Convert weight matrix
        self._convert_weight_matrix(synapses, stdp_kernel)
        
        # Store converted synapses
        self.converted_synapses[name] = stdp_kernel
        
        return stdp_kernel
    
    def _extract_neuron_parameters(self, group: NeuronGroup, lif_kernel: LIFSparseKernel):
        """Extract parameters from Brian2 NeuronGroup and apply to Taichi kernel."""
        # Extract membrane time constant
        if hasattr(group, 'tau'):
            tau_val = float(group.tau[0] if hasattr(group.tau, '__getitem__') else group.tau)
            lif_kernel.set_parameters(tau_mem=tau_val)
        
        # Extract threshold parameters
        if hasattr(group, 'theta'):
            # Adaptive threshold parameters
            if hasattr(group, 'tau_theta'):
                tau_theta_val = float(group.tau_theta[0] if hasattr(group.tau_theta, '__getitem__') else group.tau_theta)
                lif_kernel.set_parameters(tau_theta=tau_theta_val)
        
        # Extract reset potential
        if hasattr(group, 'v_reset'):
            reset_val = float(group.v_reset[0] if hasattr(group.v_reset, '__getitem__') else group.v_reset)
            lif_kernel.set_parameters(reset_potential=reset_val)
        
        # Extract threshold increment
        if hasattr(group, 'delta_theta'):
            delta_val = float(group.delta_theta[0] if hasattr(group.delta_theta, '__getitem__') else group.delta_theta)
            lif_kernel.set_parameters(delta_theta=delta_val)
    
    def _extract_synapse_parameters(self, synapses: Synapses, stdp_kernel: STDPSparseKernel):
        """Extract STDP parameters from Brian2 Synapses and apply to Taichi kernel."""
        # Extract STDP time constants
        if hasattr(synapses, 'taupre'):
            tau_pre_val = float(synapses.taupre[0] if hasattr(synapses.taupre, '__getitem__') else synapses.taupre)
            stdp_kernel.set_parameters(tau_pre=tau_pre_val)
        
        if hasattr(synapses, 'taupost'):
            tau_post_val = float(synapses.taupost[0] if hasattr(synapses.taupost, '__getitem__') else synapses.taupost)
            stdp_kernel.set_parameters(tau_post=tau_post_val)
        
        # Extract STDP amplitudes
        if hasattr(synapses, 'Apre'):
            a_plus_val = float(synapses.Apre[0] if hasattr(synapses.Apre, '__getitem__') else synapses.Apre)
            stdp_kernel.set_parameters(a_plus=a_plus_val)
        
        if hasattr(synapses, 'Apost'):
            a_minus_val = float(synapses.Apost[0] if hasattr(synapses.Apost, '__getitem__') else synapses.Apost)
            stdp_kernel.set_parameters(a_minus=a_minus_val)
        
        # Extract maximum weight
        if hasattr(synapses, 'wmax'):
            w_max_val = float(synapses.wmax[0] if hasattr(synapses.wmax, '__getitem__') else synapses.wmax)
            stdp_kernel.set_parameters(w_max=w_max_val)
    
    def _convert_weight_matrix(self, synapses: Synapses, stdp_kernel: STDPSparseKernel):
        """Convert Brian2 weight matrix to Taichi sparse format."""
        if hasattr(synapses, 'w'):
            # Get weight values from Brian2
            weights_brian2 = synapses.w[:]
            
            # Convert to numpy array
            weights_np = np.array(weights_brian2, dtype=np.float32)
            
            # Apply to Taichi kernel (maintaining sparsity)
            weights_taichi = stdp_kernel.get_weights()
            
            # Get connection indices
            if hasattr(synapses, 'i') and hasattr(synapses, 'j'):
                pre_indices = np.array(synapses.i[:], dtype=np.int32)
                post_indices = np.array(synapses.j[:], dtype=np.int32)
                
                # Set weights in Taichi format
                for idx, (i, j) in enumerate(zip(pre_indices, post_indices)):
                    if idx < len(weights_np):
                        weights_taichi[i, j] = weights_np[idx]
                
                # Update Taichi kernel weights
                stdp_kernel.weights.from_numpy(weights_taichi)
    
    def create_network_from_brian2(self, brian2_network, name_mapping: Dict[str, str] = None) -> Dict[str, Any]:
        """
        Convert entire Brian2 network to Taichi representation.
        
        Args:
            brian2_network: Brian2 Network object
            name_mapping: Optional mapping from Brian2 object names to Taichi names
            
        Returns:
            Dictionary containing converted Taichi components
        """
        if name_mapping is None:
            name_mapping = {}
        
        taichi_components = {}
        
        # Convert neuron groups
        for obj in brian2_network.objects:
            if isinstance(obj, NeuronGroup):
                name = name_mapping.get(obj.name, obj.name)
                lif_kernel = self.from_brian2_neuron_group(obj, name)
                taichi_components[f"neurons_{name}"] = lif_kernel
        
        # Convert synapses (after neurons are converted)
        for obj in brian2_network.objects:
            if isinstance(obj, Synapses):
                name = name_mapping.get(obj.name, obj.name)
                
                # Find corresponding pre/post neuron kernels
                pre_name = f"neurons_{obj.source.name}"
                post_name = f"neurons_{obj.target.name}"
                
                if pre_name in taichi_components and post_name in taichi_components:
                    pre_kernel = taichi_components[pre_name]
                    post_kernel = taichi_components[post_name]
                    
                    stdp_kernel = self.from_brian2_synapses(obj, pre_kernel, post_kernel, name)
                    taichi_components[f"synapses_{name}"] = stdp_kernel
        
        return taichi_components
    
    def get_conversion_stats(self) -> Dict[str, Any]:
        """Get statistics about the conversion."""
        stats = {
            'converted_neurons': len(self.converted_neurons),
            'converted_synapses': len(self.converted_synapses),
            'total_neurons': sum(kernel.num_neurons for kernel in self.converted_neurons.values()),
            'sparsity': self.sparsity,
            'dt': self.dt
        }
        
        # Add synapse statistics
        for name, kernel in self.converted_synapses.items():
            weight_stats = kernel.get_weight_stats()
            stats[f'{name}_connections'] = weight_stats['total_connections']
            stats[f'{name}_sparsity'] = weight_stats['sparsity']
        
        return stats
    
    def reset_all(self):
        """Reset all converted components."""
        for kernel in self.converted_neurons.values():
            kernel.reset()
        
        for kernel in self.converted_synapses.values():
            kernel.reset()
    
    def set_global_dopamine(self, dopamine_level: float):
        """Set dopamine level for all converted synapses."""
        for kernel in self.converted_synapses.values():
            kernel.set_dopamine(dopamine_level)