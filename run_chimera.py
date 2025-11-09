#!/usr/bin/env python3
"""
Project Chimera - Grounded Cognition AGI
Main entry point for the Chimera AGI system.

Chimera implements a biologically-inspired architecture with 5 core modules:
1. Core PSINet Taichi Engine - GPU-accelerated SNN computation
2. Sensory Cortex - Vision and audio processing with feature extraction
3. World Model & Hippocampus - Concept formation and memory
4. Language Module - Text-concept interface with hybrid LLM approach
5. Motivation System - Curiosity-driven learning and goal orientation

This replaces the previous Brian2-based system with a pure Taichi implementation
focused on grounded cognition rather than statistical language modeling.
"""
## Developer: inkbytefo
## Modified: 2025-11-09

import taichi as ti
ti.init(arch=ti.cuda, device_memory_GB=15, debug=False, log_level='error')

import argparse
import logging
import sys
import time
import threading
import queue
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from psinet.simulation.simulator import ChimeraSimulator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ChimeraInterface:
    """
    Interactive interface for Chimera AGI system.
    Handles user input and displays AGI responses.
    """
    
    def __init__(self, simulator: ChimeraSimulator):
        self.simulator = simulator
        self.running = True
        self.input_queue = queue.Queue()
        
    def start_interaction(self):
        """Start interactive session with Chimera."""
        print("\n" + "="*60)
        print("PROJECT CHIMERA - Grounded Cognition AGI")
        print("="*60)
        print("\nChimera is now active. You can:")
        print("  • Type text to communicate with the AGI")
        print("  • Type 'quit' or press Ctrl+C to exit")
        print("  • Type 'stats' to see network statistics")
        print("  • Type 'reset' to reset the network")
        print("\nThe AGI learns from sensory input and conversation.")
        print("Initial responses may be limited as concepts form.\n")
        
        # Start simulation in background thread
        sim_thread = threading.Thread(target=self._run_simulation, daemon=True)
        sim_thread.start()
        
        # Handle user input in main thread
        try:
            while self.running:
                try:
                    user_input = input("You: ").strip()
                    
                    if user_input.lower() in ['quit', 'exit', 'q']:
                        break
                    elif user_input.lower() == 'stats':
                        self._show_statistics()
                    elif user_input.lower() == 'reset':
                        self._reset_network()
                    elif user_input:
                        # Send language input to simulation
                        self.input_queue.put({'language': user_input})
                        
                except (EOFError, KeyboardInterrupt):
                    break
                    
        except Exception as e:
            logger.error(f"Interface error: {e}")
        finally:
            self.running = False
            self.simulator.stop()
            print("\nChimera shutting down...")
    
    def _run_simulation(self):
        """Run simulation in background thread."""
        try:
            # Run simulation with continuous processing
            self.simulator.run(duration=3600)  # Run for up to 1 hour
        except Exception as e:
            logger.error(f"Simulation error: {e}")
    
    def _show_statistics(self):
        """Display current network statistics."""
        stats = self.simulator.get_network_statistics()
        
        print("\n" + "-"*50)
        print("CHIMERA NETWORK STATISTICS")
        print("-"*50)
        print(f"Simulation Time: {stats['simulation_time']:.2f}s")
        
        # Neuron statistics
        print("\nNeuron Layers:")
        for name, neuron_stats in stats['neurons'].items():
            print(f"  {name}:")
            print(f"    Active: {neuron_stats['current_spikes']}/{neuron_stats['num_neurons']}")
            print(f"    Rate: {neuron_stats['spike_rate_hz']:.2f} Hz")
            print(f"    Mean V: {neuron_stats['mean_potential']:.3f}")
        
        # Synapse statistics
        print("\nSynapse Connections:")
        for name, synapse_stats in stats['synapses'].items():
            print(f"  {name}:")
            print(f"    Connections: {synapse_stats['total_connections']}")
            print(f"    Sparsity: {synapse_stats['sparsity']:.3f}")
            print(f"    Mean W: {synapse_stats['mean_weight']:.4f}")
            print(f"    Dopamine: {synapse_stats['dopamine_level']:.2f}")
        
        print("-"*50 + "\n")
    
    def _reset_network(self):
        """Reset the network to initial state."""
        print("Resetting Chimera network...")
        self.simulator.reset()
        print("Network reset complete.")


def main():
    """Main entry point for Chimera AGI."""
    parser = argparse.ArgumentParser(description='Project Chimera - Grounded Cognition AGI')
    parser.add_argument(
        '--config', 
        type=str, 
        default='configs/chimera_default.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--duration',
        type=float,
        help='Simulation duration in seconds (overrides config)'
    )
    parser.add_argument(
        '--no-interactive',
        action='store_true',
        help='Run without interactive interface'
    )
    
    args = parser.parse_args()
    
    # Validate config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)
    
    try:
        # Initialize Chimera simulator
        logger.info("Initializing Chimera AGI...")
        simulator = ChimeraSimulator(str(config_path))
        
        if args.no_interactive:
            # Run non-interactively
            duration = args.duration or simulator.config['run_params']['simulation_duration_sec']
            logger.info(f"Running Chimera non-interactively for {duration} seconds...")
            simulator.run(duration=duration)
        else:
            # Run interactive interface
            interface = ChimeraInterface(simulator)
            interface.start_interaction()
        
        # Final statistics
        logger.info("Simulation completed. Final statistics:")
        final_stats = simulator.get_network_statistics()
        logger.info(f"Total simulation time: {final_stats['simulation_time']:.2f}s")
        
        # Count total spikes across all neurons
        total_spikes = sum(
            neuron_stats['total_spikes'] 
            for neuron_stats in final_stats['neurons'].values()
        )
        logger.info(f"Total spikes generated: {total_spikes}")
        
    except KeyboardInterrupt:
        logger.info("Chimera interrupted by user")
    except Exception as e:
        logger.error(f"Chimera failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()