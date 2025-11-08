## Developer: inkbytefo
## Modified: 2025-11-08

#!/usr/bin/env python3
"""
Project Chimera - Milestone 11: The Grand Unification - The "Oracle" Loop
Multi-threaded AGI application that combines sensory processing, neural simulation, and dialogue.
"""

import threading
import time
import logging
import queue
import sys
import signal
from pathlib import Path
from typing import Optional, Dict, Any
import argparse

# Add psinet to path
sys.path.insert(0, str(Path(__file__).parent))

from psinet.simulation.simulator import Simulator
from psinet.io.realtime_encoders import RealtimeEncoder
from psinet.language.input import TextToConcepts
from psinet.language.output import ConceptsToText
# TODO: Import Taichi backend when ready
# from psinet.taichi_backend import LIFSparseKernel, STDPSparseKernel, Brian2ToTaichiConverter
# from psinet.core.taichi_neuron import BionicNeuron as TaichiNeuron
# from psinet.core.taichi_synapse import BionicSynapse as TaichiSynapse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SharedState:
    """Shared state container for inter-thread communication."""
    
    def __init__(self):
        self.running = True
        self.sensory_data_queue = queue.Queue(maxsize=10)
        self.user_input_queue = queue.Queue(maxsize=10)
        self.agi_output_queue = queue.Queue(maxsize=10)
        self.neural_rates = None
        self.lock = threading.Lock()
        
        # Language processing
        self.language_input = None
        self.language_output = None
        
        # Statistics
        self.stats = {
            'sensory_updates': 0,
            'simulation_steps': 0,
            'user_inputs': 0,
            'agi_outputs': 0,
            'start_time': time.time()
        }


class PSINetCoreThread(threading.Thread):
    """Thread 1: PSINet Core - Runs the continuous neural simulation."""
    
    def __init__(self, shared_state: SharedState, config_path: str):
        super().__init__(name="PSINet-Core")
        self.shared_state = shared_state
        self.config_path = config_path
        self.simulator = None
        self.hierarchy = None
        
    def run(self):
        """Main PSINet simulation loop."""
        logger.info("Starting PSINet Core thread...")
        
        try:
            # Initialize simulator
            self.simulator = Simulator(self.config_path)
            self.simulator.build()
            
            # TODO: Initialize Taichi backend converter
            # converter = Brian2ToTaichiConverter()
            # taichi_net = converter.create_network_from_brian2(self.simulator.brian2_network)
            # self.taichi_network = taichi_net
            
            # Get references to network components
            self.hierarchy = self.simulator.network_objects['hierarchy']
            self.shared_state.language_input = self.simulator.network_objects.get('language_input')
            self.shared_state.language_output = self.simulator.network_objects.get('language_output')
            
            # Configure for continuous operation
            chunk_duration_ms = 10  # 10ms chunks
            chunk_time = chunk_duration_ms * self.simulator.brian2_network.timestep
            
            logger.info(f"PSINet Core running with {chunk_duration_ms}ms chunks")
            
            # Main simulation loop
            while self.shared_state.running:
                # Update input rates from sensory data
                with self.shared_state.lock:
                    if self.shared_state.neural_rates is not None:
                        self.hierarchy.input_layer.rates = self.shared_state.neural_rates
                
                # Run simulation chunk
                self.simulator.brian2_network.run(chunk_time)
                
                # Process language output periodically
                if self.shared_state.language_output and time.time() % 1.0 < 0.01:  # Every second
                    self._process_language_output()
                
                # Update statistics
                self.shared_state.stats['simulation_steps'] += 1
                
                # Small delay to prevent CPU overload
                time.sleep(0.001)
                
        except Exception as e:
            logger.error(f"PSINet Core thread error: {e}")
        finally:
            logger.info("PSINet Core thread stopped")
    
    def _process_language_output(self):
        """Process language output based on current neural activity."""
        try:
            if not self.shared_state.language_output or not self.hierarchy:
                return
            
            # Get concept activations from highest layer (L3 or L2 or L1)
            if len(self.hierarchy.layers_in_order) >= 3:
                layer_name = self.hierarchy.layers_in_order[2]  # L3
            elif len(self.hierarchy.layers_in_order) >= 2:
                layer_name = self.hierarchy.layers_in_order[1]  # L2
            else:
                layer_name = self.hierarchy.layers_in_order[0]  # L1
            
            # Get spike monitor for the layer
            monitor_key = f'spikes_{layer_name}'
            if monitor_key in self.simulator.monitors:
                monitor = self.simulator.monitors[monitor_key]
                
                # Calculate concept activations
                concept_activations = self._calculate_concept_activations(monitor)
                
                # Calculate error signal (simplified)
                error_signal = self._calculate_error_signal()
                
                # Generate language response
                response = self.shared_state.language_output.generate_response(
                    concept_activations,
                    error_signal,
                    time.time()
                )
                
                if response:
                    self.shared_state.agi_output_queue.put(response)
                    self.shared_state.stats['agi_outputs'] += 1
                    logger.info(f"PSINet: {response}")
                    
        except Exception as e:
            logger.error(f"Error processing language output: {e}")
    
    def _calculate_concept_activations(self, spike_monitor):
        """Calculate concept activations from spike monitor."""
        try:
            if not self.shared_state.language_input:
                return {}
            
            # Get recent spikes (last 1 second)
            current_time = time.time()
            recent_time = current_time - 1.0
            
            # Filter recent spikes
            recent_mask = spike_monitor.t >= recent_time
            if not hasattr(spike_monitor, 't') or len(spike_monitor.t) == 0:
                return {}
            
            if not any(recent_mask):
                return {}
            
            recent_spikes = spike_monitor.i[recent_mask]
            recent_times = spike_monitor.t[recent_mask]
            
            return self.shared_state.language_input.get_active_concepts(
                recent_spikes, recent_times, 1.0
            )
            
        except Exception as e:
            logger.error(f"Error calculating concept activations: {e}")
            return {}
    
    def _calculate_error_signal(self):
        """Calculate prediction error signal (simplified)."""
        # Simplified error calculation based on novelty
        # In full implementation, this would use predictive coding error neurons
        import random
        return random.uniform(0.1, 0.4)


class SensoryThread(threading.Thread):
    """Thread 2: Sensory Processing - Handles video and audio input."""
    
    def __init__(self, shared_state: SharedState, sensory_config: Dict[str, Any]):
        super().__init__(name="Sensory")
        self.shared_state = shared_state
        self.sensory_config = sensory_config
        self.realtime_encoder = None
        
    def run(self):
        """Main sensory processing loop."""
        logger.info("Starting Sensory thread...")
        
        try:
            # Initialize realtime encoder
            self.realtime_encoder = RealtimeEncoder(self.sensory_config)
            
            if not self.realtime_encoder.initialize():
                logger.error("Failed to initialize sensory encoders")
                return
            
            logger.info("Sensory processing initialized")
            
            # Main sensory loop
            while self.shared_state.running:
                # Get current spike rates from sensory inputs
                spike_rates = self.realtime_encoder.get_spike_rates()
                
                # Update shared neural rates
                with self.shared_state.lock:
                    self.shared_state.neural_rates = spike_rates
                
                # Update statistics
                self.shared_state.stats['sensory_updates'] += 1
                
                # Small delay to match sensory processing rate
                time.sleep(0.033)  # ~30 FPS
                
        except Exception as e:
            logger.error(f"Sensory thread error: {e}")
        finally:
            logger.info("Sensory thread stopped")
            if self.realtime_encoder:
                self.realtime_encoder.release()


class DialogueThread(threading.Thread):
    """Thread 3: Dialogue Interface - Handles user interaction."""
    
    def __init__(self, shared_state: SharedState):
        super().__init__(name="Dialogue")
        self.shared_state = shared_state
        
    def run(self):
        """Main dialogue loop."""
        logger.info("Starting Dialogue thread...")
        
        try:
            self._print_welcome()
            
            while self.shared_state.running:
                try:
                    # Get user input
                    user_input = self._get_user_input()
                    
                    if user_input.lower() in ['quit', 'exit', 'q']:
                        logger.info("User requested shutdown")
                        self.shared_state.running = False
                        break
                    
                    # Process user input through language system
                    self._process_user_input(user_input)
                    
                    # Check for AGI output
                    self._display_agi_output()
                    
                except KeyboardInterrupt:
                    logger.info("Dialogue thread interrupted")
                    self.shared_state.running = False
                    break
                except Exception as e:
                    logger.error(f"Dialogue thread error: {e}")
                    
        finally:
            logger.info("Dialogue thread stopped")
    
    def _print_welcome(self):
        """Print welcome message and instructions."""
        print("\n" + "="*60)
        print("PSINet AGI - Oracle Loop")
        print("="*60)
        print("Welcome to the PSINet Artificial General Intelligence!")
        print()
        print("Commands:")
        print("  * Type any text to communicate with PSINet")
        print("  * 'stats' - Show system statistics")
        print("  * 'help' - Show this help message")
        print("  * 'quit' or 'exit' - Shutdown the system")
        print()
        print("PSINet is learning from sensory input and language...")
        print("="*60 + "\n")
    
    def _get_user_input(self) -> str:
        """Get input from user."""
        try:
            user_input = input("You: ").strip()
            self.shared_state.stats['user_inputs'] += 1
            return user_input
        except (EOFError, KeyboardInterrupt):
            return "quit"
    
    def _process_user_input(self, user_input: str):
        """Process user input through language system."""
        if user_input.lower() in ['stats', 'status']:
            self._show_stats()
            return
        
        if user_input.lower() in ['help', '?']:
            self._print_welcome()
            return
        
        # Process through language input system
        if self.shared_state.language_input:
            spike_gen = self.shared_state.language_input.process_text(
                user_input, 
                current_time=time.time()
            )
            
            if spike_gen:
                logger.info(f"Processed user input: '{user_input}'")
            else:
                logger.warning(f"Could not process input: '{user_input}'")
    
    def _display_agi_output(self):
        """Display AGI output to user."""
        try:
            while not self.shared_state.agi_output_queue.empty():
                output = self.shared_state.agi_output_queue.get_nowait()
                print(f"\nPSINet: {output}")
                print()
        except queue.Empty:
            pass
    
    def _show_stats(self):
        """Display system statistics."""
        stats = self.shared_state.stats
        runtime = time.time() - stats['start_time']
        
        print(f"\nSystem Statistics:")
        print(f"   Runtime: {runtime:.1f}s")
        print(f"   Sensory updates: {stats['sensory_updates']}")
        print(f"   Simulation steps: {stats['simulation_steps']}")
        print(f"   User inputs: {stats['user_inputs']}")
        print(f"   AGI outputs: {stats['agi_outputs']}")
        
        if self.shared_state.language_input:
            vocab_stats = self.shared_state.language_input.get_vocabulary_stats()
            print(f"   Vocabulary: {vocab_stats['words_learned']}/{vocab_stats['vocab_size']} words")
        
        print()


def signal_handler(signum, frame):
    """Handle shutdown signals."""
    logger.info("Received shutdown signal")
    sys.exit(0)


def main():
    """Main entry point for the AGI application."""
    parser = argparse.ArgumentParser(description="PSINet AGI - Oracle Loop")
    parser.add_argument(
        '--config', 
        default='configs/language_enabled_realtime.yaml',
        help='Configuration file path'
    )
    parser.add_argument(
        '--no-camera',
        action='store_true',
        help='Disable camera input'
    )
    parser.add_argument(
        '--no-audio',
        action='store_true', 
        help='Disable audio input'
    )
    
    args = parser.parse_args()
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info("Starting PSINet AGI - Oracle Loop")
    
    # Initialize shared state
    shared_state = SharedState()
    
    # Load configuration
    import yaml
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Configure sensory processing
    sensory_config = config['input_params']['realtime_config']
    if args.no_camera:
        sensory_config['enable_video'] = False
    if args.no_audio:
        sensory_config['enable_audio'] = False
    
    # Create and start threads
    threads = []
    
    try:
        # Thread 1: PSINet Core
        psinet_thread = PSINetCoreThread(shared_state, args.config)
        threads.append(psinet_thread)
        psinet_thread.start()
        
        # Thread 2: Sensory Processing
        sensory_thread = SensoryThread(shared_state, sensory_config)
        threads.append(sensory_thread)
        sensory_thread.start()
        
        # Thread 3: Dialogue Interface
        dialogue_thread = DialogueThread(shared_state)
        threads.append(dialogue_thread)
        dialogue_thread.start()
        
        # Wait for threads to complete
        for thread in threads:
            thread.join()
            
    except Exception as e:
        logger.error(f"Error in main: {e}")
        shared_state.running = False
    finally:
        logger.info("PSINet AGI shutdown complete")


if __name__ == "__main__":
    main()