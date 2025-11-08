## Developer: inkbytefo
## Modified: 2025-11-08

#!/usr/bin/env python3
"""
Test script for Project Chimera - Milestone 10: The Language Engine
Tests the language processing modules and integration with the sensory cortex.
"""

import sys
import logging
import time
from pathlib import Path

# Add psinet to path
sys.path.insert(0, str(Path(__file__).parent))

from psinet.simulation.simulator import Simulator
from psinet.language.input import TextToConcepts
from psinet.language.output import ConceptsToText

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_language_modules():
    """Test the language modules independently."""
    logger.info("Testing language modules...")
    
    try:
        # Test TextToConcepts
        text_to_concepts = TextToConcepts(vocab_size=50, concept_neurons=25)
        
        # Test basic vocabulary
        assert text_to_concepts.word_exists("cup"), "Basic vocabulary should contain 'cup'"
        assert text_to_concepts.word_exists("white"), "Basic vocabulary should contain 'white'"
        
        # Test text processing
        spike_gen = text_to_concepts.process_text("I see a white cup", current_time=0.0)
        assert spike_gen is not None, "Should generate spikes for known words"
        
        # Test learning new words
        success = text_to_concepts.learn_new_word("robot")
        assert success, "Should be able to learn new word"
        assert text_to_concepts.word_exists("robot"), "New word should exist in vocabulary"
        
        # Test ConceptsToText
        concepts_to_text = ConceptsToText(text_to_concepts)
        
        # Test template generation
        response = concepts_to_text.force_response("observation", object="cup")
        assert "cup" in response, f"Response should contain object: {response}"
        
        # Test curiosity trigger
        response = concepts_to_text.force_response("curiosity")
        assert "?" in response, f"Curiosity response should be a question: {response}"
        
        logger.info("Language modules test passed!")
        return True
        
    except Exception as e:
        logger.error(f"Language modules test failed: {e}")
        return False


def test_language_integration():
    """Test language integration with the full simulation."""
    logger.info("Testing language integration...")
    
    try:
        # Initialize simulator with language-enabled config
        config_path = "configs/language_enabled_realtime.yaml"
        simulator = Simulator(config_path)
        
        # Build the network
        logger.info("Building network with language support...")
        simulator.build()
        
        # Check that language modules are initialized
        assert simulator.network_objects.get('language_enabled', False), "Language should be enabled"
        assert simulator.language_input is not None, "Language input should be initialized"
        assert simulator.language_output is not None, "Language output should be initialized"
        
        # Test vocabulary stats
        vocab_stats = simulator.language_input.get_vocabulary_stats()
        logger.info(f"Vocabulary stats: {vocab_stats}")
        assert vocab_stats['words_learned'] > 0, "Should have learned some words"
        
        # Test text processing through simulator
        spike_gen = simulator.language_input.process_text("I see a red ball", current_time=0.0)
        assert spike_gen is not None, "Should process text through simulator"
        
        logger.info("Language integration test passed!")
        return True
        
    except Exception as e:
        logger.error(f"Language integration test failed: {e}")
        return False


def test_realtime_language_simulation():
    """Test realtime simulation with language processing (short duration)."""
    logger.info("Testing realtime language simulation...")
    
    try:
        # Initialize simulator with language-enabled config
        config_path = "configs/language_enabled_realtime.yaml"
        simulator = Simulator(config_path)
        
        # Build the network
        logger.info("Building network for realtime language test...")
        simulator.build()
        
        # Modify config for shorter test
        simulator.config['simulation_params']['max_duration_seconds'] = 5
        simulator.config['simulation_params']['chunk_duration_ms'] = 50
        
        # Run a short realtime simulation
        logger.info("Running short realtime simulation with language processing...")
        simulator.run()
        
        # Check language output statistics
        if simulator.language_output:
            output_stats = simulator.language_output.get_output_stats()
            logger.info(f"Language output stats: {output_stats}")
            
            recent_outputs = simulator.language_output.get_recent_outputs(3)
            logger.info(f"Recent outputs: {recent_outputs}")
        
        logger.info("Realtime language simulation test completed!")
        return True
        
    except Exception as e:
        logger.error(f"Realtime language simulation test failed: {e}")
        return False


def test_concept_activation():
    """Test concept activation from spike patterns."""
    logger.info("Testing concept activation...")
    
    try:
        text_to_concepts = TextToConcepts(vocab_size=20, concept_neurons=10)
        
        # Process some text to generate spikes
        spike_gen = text_to_concepts.process_text("big red cup", current_time=0.0)
        
        # Simulate spike data
        spike_indices = [10, 15, 20, 25, 30]  # Example spike indices
        spike_times = [0.1, 0.2, 0.3, 0.4, 0.5]  # Example spike times
        
        # Get concept activations
        activations = text_to_concepts.get_active_concepts(
            spike_indices, spike_times, time_window=1.0
        )
        
        logger.info(f"Concept activations: {activations}")
        
        # Should have some activations for the words we processed
        assert len(activations) > 0, "Should have some concept activations"
        
        logger.info("Concept activation test passed!")
        return True
        
    except Exception as e:
        logger.error(f"Concept activation test failed: {e}")
        return False


def test_curiosity_trigger():
    """Test the curiosity trigger mechanism."""
    logger.info("Testing curiosity trigger...")
    
    try:
        text_to_concepts = TextToConcepts(vocab_size=10, concept_neurons=5)
        concepts_to_text = ConceptsToText(
            text_to_concepts, 
            curiosity_threshold=0.5, 
            curiosity_duration=0.5
        )
        
        # Test with low error (should not trigger curiosity)
        response = concepts_to_text.generate_response(
            {"cup": 0.3}, 
            error_signal=0.3, 
            current_time=1.0
        )
        
        # Test with high error sustained over time
        concepts_to_text.generate_response(
            {"cup": 0.3}, 
            error_signal=0.8, 
            current_time=1.0
        )
        
        time.sleep(0.6)  # Wait for curiosity duration
        
        response = concepts_to_text.generate_response(
            {"cup": 0.3}, 
            error_signal=0.8, 
            current_time=1.6
        )
        
        # Should eventually trigger curiosity
        if response:
            logger.info(f"Curiosity response: {response}")
            assert "?" in response or "what" in response.lower(), "Should generate curiosity question"
        
        logger.info("Curiosity trigger test passed!")
        return True
        
    except Exception as e:
        logger.error(f"Curiosity trigger test failed: {e}")
        return False


if __name__ == "__main__":
    logger.info("=== Project Chimera - Milestone 10 Language Engine Test Suite ===")
    
    # Run all tests
    tests = [
        ("Language Modules", test_language_modules),
        ("Language Integration", test_language_integration),
        ("Concept Activation", test_concept_activation),
        ("Curiosity Trigger", test_curiosity_trigger),
        ("Realtime Language Simulation", test_realtime_language_simulation),
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n--- Running {test_name} Test ---")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n=== Test Results Summary ===")
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All language engine tests passed!")
        sys.exit(0)
    else:
        logger.error("❌ Some tests failed!")
        sys.exit(1)