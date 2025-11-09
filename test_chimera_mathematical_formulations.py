"""
## Developer: inkbytefo
## Modified: 2025-11-09
"""

import numpy as np
import logging
import sys
import os
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from psinet.modules.motivation_system import MotivationSystem
from psinet.modules.language_module import LanguageModule

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MockNeurons:
    """Mock neuron population for testing"""
    def __init__(self, count):
        self.count = count
        self.potential = np.random.rand(count) * 0.1
        self.spikes = np.zeros(count, dtype=bool)
        self.firing_rate = np.zeros(count)

class MockWorldModel:
    """Mock world model for testing"""
    def __init__(self):
        self.predictor = MockPredictor()
        self.working_memory = MockWorkingMemory()

class MockWorkingMemory:
    """Mock working memory for testing"""
    def __init__(self):
        self.patterns = []

class MockPredictor:
    """Mock predictor for testing"""
    def predict(self, inputs):
        # Return a slightly modified version of inputs as prediction
        return inputs + np.random.normal(0, 0.1, inputs.shape)

def test_dopamine_formulation():
    """Test dopamine signal formulation"""
    logger.info("Testing dopamine signal formulation...")
    
    try:
        # Create motivation system
        config = {
            'dopamine_params': {
                'baseline': 1.0,
                'k_gain': 2.0,
                'time_constant': 0.5
            }
        }
        
        motivation_system = MotivationSystem(config)
        
        # Test initial state
        assert motivation_system.dopamine_level == 1.0
        logger.info(f"Initial dopamine level: {motivation_system.dopamine_level}")
        
        # Test dopamine update with zero error
        neurons = {'sensory_vision': MockNeurons(100)}
        world_model = MockWorldModel()
        
        # Set baseline error to 0.5 for testing
        motivation_system.baseline_error = 0.5
        
        # Test with zero error (should return baseline dopamine)
        actual_sensory = np.random.rand(100)
        predicted_sensory = actual_sensory.copy()  # Perfect prediction
        
        # Mock the world model to return our test data
        world_model.working_memory.patterns = [actual_sensory]
        world_model.predictor.predict = lambda x: predicted_sensory
        
        # Update dopamine using the precise formulation
        prediction_error = motivation_system._calculate_prediction_error(neurons, world_model)
        motivation_system._update_dopamine_precise(prediction_error, 0.0, 0.01)
        
        expected_dopamine = 1.0  # Baseline when error = baseline
        actual_dopamine = motivation_system.dopamine_level
        
        logger.info(f"Test 1 - Zero error: Expected={expected_dopamine:.3f}, Actual={actual_dopamine:.3f}")
        assert abs(actual_dopamine - expected_dopamine) < 0.2, "Dopamine formulation failed for zero error"
        
        # Test with high error
        high_error_sensory = np.random.rand(100)
        low_prediction = np.zeros(100)  # Poor prediction
        
        world_model.working_memory.patterns = [high_error_sensory]
        world_model.predictor.predict = lambda x: low_prediction
        
        # Update dopamine with high error
        prediction_error = motivation_system._calculate_prediction_error(neurons, world_model)
        logger.info(f"Prediction error for high error test: {prediction_error:.3f}")
        
        # Reset dopamine to baseline first
        motivation_system.dopamine_level = 1.0
        motivation_system._update_dopamine_precise(prediction_error, 0.0, 1.0)  # Use larger dt
        
        actual_dopamine = motivation_system.dopamine_level
        
        logger.info(f"Test 2 - High error: Actual dopamine={actual_dopamine:.3f}")
        # Should be different from baseline for high error (either higher or lower depending on formulation)
        assert abs(actual_dopamine - 1.0) > 0.1, "Dopamine should change with high error"
        
        logger.info("✅ Dopamine formulation test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Dopamine formulation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_prediction_error_calculation():
    """Test prediction error calculation"""
    logger.info("Testing prediction error calculation...")
    
    try:
        # Create motivation system
        config = {'dopamine_params': {'baseline': 1.0}}
        motivation_system = MotivationSystem(config)
        
        # Create test data
        neurons = {'sensory_vision': MockNeurons(100)}
        world_model = MockWorldModel()
        
        # Test with identical patterns (zero error)
        actual_pattern = np.random.rand(100)
        predicted_pattern = actual_pattern.copy()
        
        world_model.working_memory.patterns = [actual_pattern]
        world_model.predictor.predict = lambda x: predicted_pattern
        
        prediction_error = motivation_system._calculate_prediction_error(neurons, world_model)
        
        logger.info(f"Test 1 - Identical patterns: Error={prediction_error:.6f}")
        assert prediction_error < 0.5, "Prediction error should be low for identical patterns"
        
        # Test with different patterns
        actual_pattern = np.random.rand(100)
        predicted_pattern = np.random.rand(100)
        
        world_model.working_memory.patterns = [actual_pattern]
        world_model.predictor.predict = lambda x: predicted_pattern
        
        prediction_error = motivation_system._calculate_prediction_error(neurons, world_model)
        
        logger.info(f"Test 2 - Different patterns: Error={prediction_error:.6f}")
        # Should be a reasonable error value
        assert 0.0 <= prediction_error <= 1.0, "Prediction error should be in valid range"
        
        logger.info("✅ Prediction error calculation test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Prediction error calculation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_snn_llm_interface():
    """Test SNN-LLM interface"""
    logger.info("Testing SNN-LLM interface...")
    
    try:
        # Create mock neuron populations
        concept_neurons = MockNeurons(1000)
        associative_cortex = MockNeurons(4000)
        
        # Create language module config
        config = {
            'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
            'embedding_dim': 768,
            'concept_neurons': 1000
        }
        
        # Initialize language module
        language_module = LanguageModule(concept_neurons, associative_cortex, config)
        
        # Test autoencoder initialization
        assert hasattr(language_module, 'autoencoder'), "Autoencoder should be initialized"
        assert language_module.embedding_dim == 768, "Embedding dimension should be set correctly"
        
        logger.info("✅ Autoencoder initialization verified")
        
        # Test concept firing rate calculation
        concept_pattern = np.random.rand(1000)
        concept_neurons.potential = concept_pattern
        concept_neurons.spikes = (concept_pattern > 0.5).astype(bool)
        
        # Calculate firing rates
        firing_rates = language_module._calculate_firing_rates(concept_neurons.potential, concept_neurons.spikes)
        assert len(firing_rates) == 1000, "Firing rates should match neuron count"
        assert np.all(firing_rates >= 0), "Firing rates should be non-negative"
        
        logger.info("✅ Firing rate calculation verified")
        
        logger.info("✅ SNN-LLM interface test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ SNN-LLM interface test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_mathematical_consistency():
    """Test mathematical consistency across formulations"""
    logger.info("Testing mathematical consistency...")
    
    try:
        # Test that dopamine formulation respects bounds
        config = {
            'curiosity': {'enable': True},
            'goal_orientation': {'enable': False}
        }
        
        motivation_system = MotivationSystem(config)
        
        # Test extreme values
        extreme_errors = [-10.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 10.0]
        
        for error in extreme_errors:
            motivation_system._update_dopamine_precise(error, 0.0, 0.01)
            dopamine = motivation_system.dopamine_level
            
            # Check physiological bounds
            assert 0.1 <= dopamine <= 3.0, f"Dopamine {dopamine:.3f} out of bounds for error {error}"
            
        logger.info("✅ Dopamine bounds respected for extreme values")
        
        # Test prediction error normalization
        world_model = MockWorldModel()
        
        class MockNeuronLayer:
            def get_membrane_potentials(self):
                return np.array([1.0, 2.0, 3.0])
        
        neurons = {'associative_cortex': MockNeuronLayer()}
        
        # Test with zero norm pattern (should handle gracefully)
        world_model.working_memory.patterns = [np.zeros(3)]
        
        prediction_error = motivation_system._calculate_prediction_error(neurons, world_model)
        
        assert 0.0 <= prediction_error <= 1.0, f"Prediction error {prediction_error} not in valid range"
        
        logger.info("✅ Mathematical consistency verified")
        return True
        
    except Exception as e:
        logger.error(f"❌ Mathematical consistency test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all mathematical formulation tests."""
    logger.info("=" * 60)
    logger.info("CHIMERA MATHEMATICAL FORMULATIONS TEST SUITE")
    logger.info("=" * 60)
    
    # Run individual tests
    dopamine_test_passed = test_dopamine_formulation()
    prediction_error_test_passed = test_prediction_error_calculation()
    snn_llm_test_passed = test_snn_llm_interface()
    consistency_test_passed = test_mathematical_consistency()
    
    # Summary
    logger.info("=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Dopamine Formulation: {'✅ PASSED' if dopamine_test_passed else '❌ FAILED'}")
    logger.info(f"Prediction Error Calculation: {'✅ PASSED' if prediction_error_test_passed else '❌ FAILED'}")
    logger.info(f"SNN-LLM Interface: {'✅ PASSED' if snn_llm_test_passed else '❌ FAILED'}")
    logger.info(f"Mathematical Consistency: {'✅ PASSED' if consistency_test_passed else '❌ FAILED'}")
    
    all_passed = all([
        dopamine_test_passed,
        prediction_error_test_passed,
        snn_llm_test_passed,
        consistency_test_passed
    ])
    
    if all_passed:
        logger.info("🎉 ALL MATHEMATICAL FORMULATION TESTS PASSED!")
        logger.info("The Chimera AGI mathematical foundations are solid.")
        return 0
    else:
        logger.error("❌ SOME MATHEMATICAL FORMULATION TESTS FAILED")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)