#!/usr/bin/env python3
"""
Integration test for Project Chimera AGI architecture.

Tests the complete system with all 5 modules working together.
"""
## Developer: inkbytefo
## Modified: 2025-11-09

import sys
import numpy as np
import logging
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from psinet.simulation.simulator import ChimeraSimulator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_chimera_basic_functionality():
    """Test basic Chimera functionality without external dependencies."""
    logger.info("Starting Chimera integration test...")
    
    try:
        # Initialize simulator with default config
        config_path = "configs/chimera_default.yaml"
        simulator = ChimeraSimulator(config_path)
        
        logger.info("✅ Chimera simulator initialized successfully")
        
        # Test network structure
        assert len(simulator.neurons) > 0, "No neuron layers created"
        assert len(simulator.synapses) > 0, "No synapse connections created"
        
        logger.info(f"✅ Network created with {len(simulator.neurons)} neuron layers and {len(simulator.synapses)} synapse connections")
        
        # Test single simulation step
        test_input = {
            'vision': np.random.randint(0, 255, (28, 28, 3), dtype=np.uint8),
            'audio': np.random.randn(1024).astype(np.float32),
            'language': 'test input'
        }
        
        state = simulator.step(test_input)
        
        # Verify state structure
        assert 'simulation_time' in state, "Missing simulation time in state"
        assert 'spikes' in state, "Missing spikes in state"
        assert 'membrane_potentials' in state, "Missing membrane potentials in state"
        
        logger.info("✅ Single simulation step completed successfully")
        
        # Test multiple steps
        for i in range(10):
            state = simulator.step(test_input)
            assert state['simulation_time'] > 0, "Simulation time not advancing"
        
        logger.info("✅ Multiple simulation steps completed successfully")
        
        # Test network statistics
        stats = simulator.get_network_statistics()
        assert 'neurons' in stats, "Missing neuron statistics"
        assert 'synapses' in stats, "Missing synapse statistics"
        
        logger.info("✅ Network statistics retrieved successfully")
        
        # Test reset functionality
        simulator.reset()
        assert simulator.simulation_time == 0.0, "Simulation time not reset"
        
        logger.info("✅ Network reset successfully")
        
        logger.info("🎉 All Chimera integration tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Chimera integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_chimera_modules():
    """Test individual Chimera modules."""
    logger.info("Testing individual Chimera modules...")
    
    try:
        # Test sensory cortex
        from psinet.modules.sensory_cortex import SensoryCortex
        
        sensory_config = {
            'vision': {
                'enable': True,
                'target_size': [28, 28],
                'max_rate': 150.0
            },
            'audio': {
                'enable': True,
                'freq_bins': 32,
                'max_rate': 150.0
            }
        }
        
        sensory_cortex = SensoryCortex(sensory_config)
        
        # Test vision processing
        test_frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        vision_spikes = sensory_cortex.vision_processor.process_frame(test_frame)
        assert len(vision_spikes) > 0, "No vision spikes generated"
        
        logger.info("✅ Sensory cortex vision processing works")
        
        # Test audio processing
        test_audio = np.random.randn(2048).astype(np.float32)
        audio_spikes = sensory_cortex.audio_processor.process_audio_chunk(test_audio)
        assert len(audio_spikes) > 0, "No audio spikes generated"
        
        logger.info("✅ Sensory cortex audio processing works")
        
        # Test world model
        from psinet.modules.world_model import WorldModel
        
        # Create mock neuron populations
        class MockNeuron:
            def __init__(self, num_neurons):
                self.num_neurons = num_neurons
            
            def get_membrane_potentials(self):
                return np.random.randn(self.num_neurons) * 0.1
            
            def get_spikes(self):
                return np.random.randint(0, 2, self.num_neurons)
        
        associative_cortex = MockNeuron(100)
        hippocampus = MockNeuron(50)
        
        world_model_config = {
            'concept_formation_threshold': 0.5,
            'pattern_stability_time': 1.0
        }
        
        world_model = WorldModel(associative_cortex, hippocampus, world_model_config)
        
        # Test world model update
        neurons = {
            'associative_cortex': associative_cortex,
            'hippocampus': hippocampus
        }
        
        world_model.update(neurons, simulation_time=1.0)
        
        state = world_model.get_state()
        assert 'concept_count' in state, "Missing concept count in world model state"
        
        logger.info("✅ World model works")
        
        # Test language module
        from psinet.modules.language_module import LanguageModule
        
        concept_neurons = MockNeuron(100)
        language_config = {
            'enable': True,
            'vocab_size': 100,
            'concept_neurons': 100,
            'response_generation': {
                'template_based': True,
                'llm_fallback': False
            }
        }
        
        language_module = LanguageModule(concept_neurons, associative_cortex, language_config)
        
        # Test text processing
        test_text = "hello world"
        concept_activation = language_module.process_text_input(test_text)
        assert len(concept_activation) > 0, "No concept activation generated"
        
        logger.info("✅ Language module works")
        
        # Test motivation system
        from psinet.modules.motivation_system import MotivationSystem
        
        motivation_config = {
            'curiosity': {
                'enable': True,
                'prediction_error_scale': 1.0
            },
            'goal_orientation': {
                'enable': True,
                'goal_persistence': 5.0
            }
        }
        
        motivation_system = MotivationSystem(motivation_config)
        
        # Test motivation update
        modulation_signals = motivation_system.update(neurons, world_model, simulation_time=1.0)
        assert len(modulation_signals) > 0, "No modulation signals generated"
        
        logger.info("✅ Motivation system works")
        
        logger.info("🎉 All individual module tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Module test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all integration tests."""
    logger.info("=" * 60)
    logger.info("PROJECT CHIMERA - INTEGRATION TESTS")
    logger.info("=" * 60)
    
    # Test individual modules first
    module_test_passed = test_chimera_modules()
    
    # Test full integration
    integration_test_passed = test_chimera_basic_functionality()
    
    # Summary
    logger.info("=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Module Tests: {'✅ PASSED' if module_test_passed else '❌ FAILED'}")
    logger.info(f"Integration Tests: {'✅ PASSED' if integration_test_passed else '❌ FAILED'}")
    
    if module_test_passed and integration_test_passed:
        logger.info("🎉 ALL TESTS PASSED - Chimera AGI is ready!")
        return 0
    else:
        logger.error("❌ SOME TESTS FAILED - Check the logs above")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)