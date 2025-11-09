## Developer: inkbytefo
## Modified: 2025-11-08

#!/usr/bin/env python3
"""
Test script for Project Chimera - Milestone 9: The Sensory Cortex
Tests the realtime sensory input functionality.
"""

import sys
import logging
from pathlib import Path

# Add psinet to path
sys.path.insert(0, str(Path(__file__).parent))

from psinet.simulation.simulator import Simulator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_realtime_simulation():
    """Test the realtime sensory cortex simulation."""
    logger.info("Starting realtime sensory cortex test...")
    
    try:
        # Initialize simulator with realtime config
        config_path = "configs/realtime_sensory_cortex.yaml"
        simulator = Simulator(config_path)
        
        # Build the network
        logger.info("Building network...")
        simulator.build()
        
        # Run realtime simulation (short duration for testing)
        logger.info("Running realtime simulation...")
        simulator.run()
        
        # Save results
        logger.info("Saving results...")
        simulator.save_results()
        
        logger.info("Realtime test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Realtime test failed: {e}")
        return False


def test_batch_simulation():
    """Test that batch mode still works (regression test)."""
    logger.info("Starting batch mode regression test...")
    
    try:
        # Initialize simulator with MNIST config
        config_path = "configs/mnist_deep_hierarchy.yaml"
        simulator = Simulator(config_path)
        
        # Build the network
        logger.info("Building network...")
        simulator.build()
        
        # Run batch simulation
        logger.info("Running batch simulation...")
        simulator.run()
        
        # Save results
        logger.info("Saving results...")
        simulator.save_results()
        
        logger.info("Batch test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Batch test failed: {e}")
        return False


if __name__ == "__main__":
    logger.info("=== Project Chimera - Milestone 9 Test Suite ===")
    
    # Test batch mode first (regression test)
    batch_success = test_batch_simulation()
    
    # Test realtime mode
    realtime_success = test_realtime_simulation()
    
    # Summary
    logger.info("\n=== Test Results ===")
    logger.info(f"Batch mode: {'PASS' if batch_success else 'FAIL'}")
    logger.info(f"Realtime mode: {'PASS' if realtime_success else 'FAIL'}")
    
    if batch_success and realtime_success:
        logger.info("All tests passed!")
        sys.exit(0)
    else:
        logger.error("Some tests failed!")
        sys.exit(1)