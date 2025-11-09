## Developer: inkbytefo
## Modified: 2025-11-09

import yaml
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from psinet.core.neuron import BionicNeuron
from psinet.core.synapse import BionicSynapse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ChimeraSimulator:
    """
    Chimera AGI Simulator - Grounded Cognition Architecture
    
    Orchestrates the complete Chimera AGI system with 5 core modules:
    1. Core PSINet Taichi Engine (SNN computation)
    2. Sensory Cortex (vision/audio processing)
    3. World Model & Hippocampus (concept formation)
    4. Language Module (text-concept interface)
    5. Motivation System (curiosity & goals)
    """
    
    def __init__(self, config_path: str):
        """
        Initialize Chimera simulator with configuration.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = Path(config_path)
        self._load_config()
        self._setup_output_directory()
        
        # Neural network components
        self.neurons = {}
        self.synapses = {}
        
        # Module instances
        self.sensory_cortex = None
        self.world_model = None
        self.language_module = None
        self.motivation_system = None
        
        # Simulation state
        self.simulation_time = 0.0
        self.running = False
        
        # Build the network
        self._build_network()
        
    def _load_config(self):
        """Load configuration from YAML file."""
        logger.info(f"Loading Chimera configuration from: {self.config_path}")
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def _setup_output_directory(self):
        """Setup output directory for logging and results."""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_name = self.config.get('run_id', 'chimera_run')
        base_dir = Path(self.config.get('output_params', {}).get('base_output_dir', 'outputs/chimera'))
        self.output_dir = base_dir / f"{run_name}_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output will be saved to: {self.output_dir}")
    
    def _build_network(self):
        """Build the complete neural network from configuration."""
        logger.info("Building Chimera neural network...")
        
        net_params = self.config['network_params']
        
        # Build neuron layers
        self._build_neurons(net_params['neurons'])
        
        # Build synapse connections
        self._build_synapses(net_params['synapses'])
        
        # Initialize modules
        self._initialize_modules()
        
        logger.info("Chimera network build complete.")
    
    def _build_neurons(self, neuron_configs: Dict[str, Any]):
        """Build all neuron layers from configuration."""
        logger.info("Building neuron layers...")
        
        for name, params in neuron_configs.items():
            logger.info(f"Creating neuron layer: {name} ({params['count']} neurons)")
            
            self.neurons[name] = BionicNeuron(
                num_neurons=params['count'],
                tau=params.get('tau', 0.01),
                threshold_initial=params.get('threshold_initial', 1.0),
                reset_v=params.get('reset_v', 0.0),
                refractory=params.get('refractory', 0.005),
                tau_theta=params.get('tau_theta', 0.1),
                delta_theta=params.get('delta_theta', 0.1),
                dt=params.get('dt', 0.001),
                sparsity=params.get('sparsity', 0.9)
            )
    
    def _build_synapses(self, synapse_configs: Dict[str, Any]):
        """Build all synapse connections from configuration."""
        logger.info("Building synapse connections...")
        
        for name, params in synapse_configs.items():
            pre_layer_name = params['from']
            post_layer_name = params['to']
            
            if pre_layer_name not in self.neurons or post_layer_name not in self.neurons:
                raise ValueError(f"Invalid synapse connection: {pre_layer_name} -> {post_layer_name}")
            
            pre_layer = self.neurons[pre_layer_name]
            post_layer = self.neurons[post_layer_name]
            
            logger.info(f"Creating synapse: {name} ({pre_layer_name} -> {post_layer_name})")
            
            self.synapses[name] = BionicSynapse(
                pre_neurons=pre_layer,
                post_neurons=post_layer,
                tau_pre=params.get('tau_pre', 0.02),
                tau_post=params.get('tau_post', 0.02),
                w_max=params.get('w_max', 0.01),
                a_pre=params.get('a_pre', 0.01),
                a_post=params.get('a_post', -0.0105),
                initial_weight_max=params.get('initial_weight_max', 0.01),
                dt=params.get('dt', 0.001),
                sparsity=params.get('sparsity', 0.9)
            )
    
    def _initialize_modules(self):
        """Initialize all Chimera modules."""
        logger.info("Initializing Chimera modules...")
        
        # Module 2: Sensory Cortex
        if self.config.get('sensory_params', {}).get('vision', {}).get('enable', False) or \
           self.config.get('sensory_params', {}).get('audio', {}).get('enable', False):
            from ..modules.sensory_cortex import SensoryCortex
            self.sensory_cortex = SensoryCortex(self.config.get('sensory_params', {}))
            logger.info("Sensory Cortex initialized")
        
        # Module 3: World Model & Hippocampus
        if 'world_model_params' in self.config:
            from ..modules.world_model import WorldModel
            self.world_model = WorldModel(
                associative_cortex=self.neurons.get('associative_cortex'),
                hippocampus=self.neurons.get('hippocampus'),
                config=self.config.get('world_model_params', {})
            )
            logger.info("World Model & Hippocampus initialized")
        
        # Module 4: Language Module
        if self.config.get('language_params', {}).get('enable', False):
            from ..modules.language_module import LanguageModule
            self.language_module = LanguageModule(
                concept_neurons=self.neurons.get('language_concepts'),
                associative_cortex=self.neurons.get('associative_cortex'),
                config=self.config.get('language_params', {})
            )
            logger.info("Language Module initialized")
        
        # Module 5: Motivation System
        if 'motivation_params' in self.config:
            from ..modules.motivation_system import MotivationSystem
            self.motivation_system = MotivationSystem(
                config=self.config.get('motivation_params', {})
            )
            logger.info("Motivation System initialized")
    
    def step(self, inputs: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, Any]:
        """
        Execute one simulation step.
        
        Args:
            inputs: Optional input dictionary with sensory data
            
        Returns:
            Dictionary with current network state
        """
        dt = self.config['run_params']['chunk_duration_ms'] / 1000.0
        
        # Process sensory inputs if available
        if inputs and self.sensory_cortex:
            sensory_spikes = self.sensory_cortex.process_inputs(inputs)
            for layer_name, spike_data in sensory_spikes.items():
                if layer_name in self.neurons:
                    self.neurons[layer_name].apply_poisson_input(spike_data)
        
        # Process language inputs if available
        if inputs and 'language' in inputs and self.language_module:
            language_spikes = self.language_module.process_text_input(inputs['language'])
            if 'language_concepts' in self.neurons:
                self.neurons['language_concepts'].apply_poisson_input(language_spikes)
        
        # Collect all pre-synaptic spikes
        all_spikes = {}
        for name, neuron in self.neurons.items():
            all_spikes[name] = neuron.get_spikes()
        
        # Update synapses and collect postsynaptic currents
        postsynaptic_currents = {}
        for synapse_name, synapse in self.synapses.items():
            pre_name = self.config['network_params']['synapses'][synapse_name]['from']
            post_name = self.config['network_params']['synapses'][synapse_name]['to']
            
            pre_spikes = all_spikes.get(pre_name, np.zeros(self.neurons[pre_name].num_neurons))
            post_spikes = all_spikes.get(post_name, np.zeros(self.neurons[post_name].num_neurons))
            
            psc = synapse.update(pre_spikes, post_spikes)
            
            if post_name not in postsynaptic_currents:
                postsynaptic_currents[post_name] = np.zeros(self.neurons[post_name].num_neurons)
            postsynaptic_currents[post_name] += psc
        
        # Update neurons with postsynaptic currents
        for name, neuron in self.neurons.items():
            if name in postsynaptic_currents:
                neuron.update(postsynaptic_currents[name])
            else:
                neuron.update()
        
        # Update world model
        if self.world_model:
            self.world_model.update(self.neurons, self.simulation_time)
        
        # Update motivation system
        if self.motivation_system:
            motivation_signals = self.motivation_system.update(
                self.neurons, self.world_model, self.simulation_time
            )
            
            # Apply motivation signals as dopamine modulation
            for synapse_name, dopamine_level in motivation_signals.items():
                if synapse_name in self.synapses:
                    self.synapses[synapse_name].set_dopamine(dopamine_level)
        
        # Generate language output
        language_output = None
        if self.language_module:
            language_output = self.language_module.generate_response(
                self.neurons, self.simulation_time
            )
        
        # Update simulation time
        self.simulation_time += dt
        
        # Return current state
        return {
            'simulation_time': self.simulation_time,
            'spikes': all_spikes,
            'membrane_potentials': {name: neuron.get_membrane_potentials() for name, neuron in self.neurons.items()},
            'language_output': language_output,
            'motivation_state': self.motivation_system.get_state() if self.motivation_system else None,
            'world_model_state': self.world_model.get_state() if self.world_model else None
        }
    
    def run(self, duration: Optional[float] = None):
        """
        Run the complete simulation.
        
        Args:
            duration: Simulation duration in seconds. If None, uses config value.
        """
        if duration is None:
            duration = self.config['run_params']['simulation_duration_sec']
        
        logger.info(f"Starting Chimera simulation for {duration} seconds...")
        self.running = True
        
        chunk_duration = self.config['run_params']['chunk_duration_ms'] / 1000.0
        total_steps = int(duration / chunk_duration)
        
        try:
            for step in range(total_steps):
                if not self.running:
                    break
                
                # Execute simulation step
                state = self.step()
                
                # Log progress periodically
                if step % 100 == 0:
                    logger.info(f"Step {step}/{total_steps}, Time: {state['simulation_time']:.2f}s")
                    
                    # Log language output if available
                    if state['language_output']:
                        logger.info(f"AGI: {state['language_output']}")
                
                # Save state periodically
                if step % 500 == 0:
                    self._save_state(state)
        
        except KeyboardInterrupt:
            logger.info("Simulation interrupted by user")
        finally:
            self.running = False
            logger.info("Chimera simulation completed")
    
    def stop(self):
        """Stop the simulation."""
        self.running = False
    
    def _save_state(self, state: Dict[str, Any]):
        """Save current simulation state."""
        # Implementation for periodic state saving
        pass
    
    def get_network_statistics(self) -> Dict[str, Any]:
        """Get comprehensive network statistics."""
        stats = {
            'simulation_time': self.simulation_time,
            'neurons': {},
            'synapses': {}
        }
        
        for name, neuron in self.neurons.items():
            stats['neurons'][name] = neuron.get_statistics()
        
        for name, synapse in self.synapses.items():
            stats['synapses'][name] = synapse.get_statistics()
        
        return stats
    
    def reset(self):
        """Reset all network components to initial state."""
        logger.info("Resetting Chimera network...")
        
        for neuron in self.neurons.values():
            neuron.reset()
        
        for synapse in self.synapses.values():
            synapse.reset()
        
        if self.world_model:
            self.world_model.reset()
        
        if self.motivation_system:
            self.motivation_system.reset()
        
        self.simulation_time = 0.0
        logger.info("Chimera network reset complete")
