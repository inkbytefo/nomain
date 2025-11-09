## Developer: inkbytefo
## Modified: 2025-11-08

"""
Brian2 to Taichi converter for PSINet.
Converts Brian2 neuron groups and synapses to Taichi-based implementations.
"""

import numpy as np
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.neuron import BionicNeuron
    from ..core.synapse import BionicSynapse


class Brian2ToTaichiConverter:
    """
    Converter from Brian2 objects to Taichi-based PSINet objects.
    Enables migration from Brian2 to high-performance Taichi backend.
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
        
        # Conversion statistics
        self.conversion_stats = {
            'converted_neurons': 0,
            'converted_synapses': 0,
            'total_neurons': 0,
            'total_synapses': 0
        }
    
    def from_brian2_neuron_group(self, neuron_group: Any, name: str = "converted") -> 'BionicNeuron':
        """
        Convert Brian2 NeuronGroup to BionicNeuron.
        
        Args:
            neuron_group: Brian2 NeuronGroup object
            name: Name for the converted neuron population
            
        Returns:
            BionicNeuron instance
        """
        # Extract parameters from Brian2 NeuronGroup
        num_neurons = getattr(neuron_group, 'N', 100)
        
        # Get neuron parameters with defaults
        tau = self._extract_parameter(neuron_group, 'tau', 0.02)
        threshold_initial = self._extract_parameter(neuron_group, 'theta', 1.0)
        reset_v = self._extract_parameter(neuron_group, 'v_reset', 0.0)
        refractory = self._extract_parameter(neuron_group, 'refractory', 0.005)
        tau_theta = self._extract_parameter(neuron_group, 'tau_theta', 0.1)
        delta_theta = self._extract_parameter(neuron_group, 'delta_theta', 0.05)
        
        # Lazy import to avoid circular dependency
        from ..core.neuron import BionicNeuron
        
        # Create BionicNeuron
        taichi_neuron = BionicNeuron(
            num_neurons=num_neurons,
            tau=tau,
            threshold_initial=threshold_initial,
            reset_v=reset_v,
            refractory=refractory,
            tau_theta=tau_theta,
            delta_theta=delta_theta,
            dt=self.dt,
            sparsity=self.sparsity
        )
        
        # Update statistics
        self.conversion_stats['converted_neurons'] += 1
        self.conversion_stats['total_neurons'] += num_neurons
        
        return taichi_neuron
    
    def from_brian2_synapses(self, synapses: Any, pre_neurons: 'BionicNeuron',
                           post_neurons: 'BionicNeuron', name: str = "converted") -> 'BionicSynapse':
        """
        Convert Brian2 Synapses to BionicSynapse.
        
        Args:
            synapses: Brian2 Synapses object
            pre_neurons: Pre-synaptic BionicNeuron population
            post_neurons: Post-synaptic BionicNeuron population
            name: Name for the converted synapse population
            
        Returns:
            BionicSynapse instance
        """
        # Extract STDP parameters from Brian2 Synapses
        tau_pre = self._extract_parameter(synapses, 'taupre', 0.02)
        tau_post = self._extract_parameter(synapses, 'taupost', 0.02)
        a_pre = self._extract_parameter(synapses, 'Apre', 0.01)
        a_post = self._extract_parameter(synapses, 'Apost', -0.0105)
        w_max = self._extract_parameter(synapses, 'wmax', 0.01)
        
        # Get existing weights if available
        initial_weight_max = w_max
        if hasattr(synapses, 'w') and hasattr(synapses, 'i') and hasattr(synapses, 'j'):
            # Use existing weight distribution
            weights = np.array(synapses.w)
            if len(weights) > 0:
                initial_weight_max = np.max(weights)
        
        # Lazy import to avoid circular dependency
        from ..core.synapse import BionicSynapse
        
        # Create BionicSynapse
        taichi_synapse = BionicSynapse(
            pre_neurons=pre_neurons,
            post_neurons=post_neurons,
            tau_pre=tau_pre,
            tau_post=tau_post,
            w_max=w_max,
            a_pre=a_pre,
            a_post=a_post,
            initial_weight_max=initial_weight_max,
            dt=self.dt,
            sparsity=self.sparsity
        )
        
        # Set existing weights if available
        if hasattr(synapses, 'w') and hasattr(synapses, 'i') and hasattr(synapses, 'j'):
            self._set_synapse_weights(taichi_synapse, synapses)
        
        # Update statistics
        self.conversion_stats['converted_synapses'] += 1
        self.conversion_stats['total_synapses'] += len(getattr(synapses, 'w', []))
        
        return taichi_synapse
    
    def _extract_parameter(self, obj: Any, param_name: str, default_value: float) -> float:
        """
        Extract parameter from Brian2 object with fallback to default.
        
        Args:
            obj: Brian2 object
            param_name: Parameter name to extract
            default_value: Default value if parameter not found
            
        Returns:
            Parameter value
        """
        if hasattr(obj, param_name):
            param_value = getattr(obj, param_name)
            if isinstance(param_value, np.ndarray):
                return float(param_value[0]) if len(param_value) > 0 else default_value
            else:
                return float(param_value)
        return default_value
    
    def _set_synapse_weights(self, taichi_synapse: 'BionicSynapse', brian2_synapses: Any):
        """
        Set weights in Taichi synapse from Brian2 synapses.
        
        Args:
            taichi_synapse: Target BionicSynapse
            brian2_synapses: Source Brian2 Synapses
        """
        try:
            # Get connection information
            weights = np.array(brian2_synapses.w)
            pre_indices = np.array(brian2_synapses.i)
            post_indices = np.array(brian2_synapses.j)
            
            # Create sparse weight matrix
            weight_matrix = np.zeros((
                taichi_synapse.pre_neurons.num_neurons,
                taichi_synapse.post_neurons.num_neurons
            ), dtype=np.float32)
            
            # Set weights for existing connections
            weight_matrix[pre_indices, post_indices] = weights
            
            # Apply to Taichi kernel
            taichi_synapse.kernel.weights.from_numpy(weight_matrix)
            
        except Exception as e:
            # If weight setting fails, continue with random initialization
            print(f"Warning: Could not set synapse weights: {e}")
    
    def get_conversion_stats(self) -> Dict[str, Any]:
        """
        Get conversion statistics.
        
        Returns:
            Dictionary with conversion statistics
        """
        return self.conversion_stats.copy()
    
    def reset_stats(self):
        """Reset conversion statistics."""
        self.conversion_stats = {
            'converted_neurons': 0,
            'converted_synapses': 0,
            'total_neurons': 0,
            'total_synapses': 0
        }
    
    def convert_network(self, brian2_network: Any) -> Dict[str, Any]:
        """
        Convert entire Brian2 network to Taichi-based network.
        
        Args:
            brian2_network: Brian2 Network object
            
        Returns:
            Dictionary containing converted neurons and synapses
        """
        converted_objects = {
            'neurons': {},
            'synapses': {}
        }
        
        # Reset statistics
        self.reset_stats()
        
        # Convert neuron groups
        if hasattr(brian2_network, 'objects'):
            for obj in brian2_network.objects:
                if hasattr(obj, '__class__') and 'NeuronGroup' in obj.__class__.__name__:
                    name = getattr(obj, 'name', f'neuron_{len(converted_objects["neurons"])}')
                    converted_neurons = self.from_brian2_neuron_group(obj, name)
                    converted_objects['neurons'][name] = converted_neurons
                
                elif hasattr(obj, '__class__') and 'Synapses' in obj.__class__.__name__:
                    name = getattr(obj, 'name', f'synapse_{len(converted_objects["synapses"])}')
                    
                    # Find pre and post synaptic neurons (simplified approach)
                    # In a real implementation, you'd need to track connections properly
                    pre_neurons = list(converted_objects['neurons'].values())[0] if converted_objects['neurons'] else None
                    post_neurons = pre_neurons  # Default to same population
                    
                    if pre_neurons is not None:
                        converted_synapses = self.from_brian2_synapses(
                            obj, pre_neurons, post_neurons, name
                        )
                        converted_objects['synapses'][name] = converted_synapses
        
        return converted_objects
    
    def __repr__(self):
        return f"Brian2ToTaichiConverter(dt={self.dt}, sparsity={self.sparsity})"