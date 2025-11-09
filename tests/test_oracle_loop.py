## Developer: inkbytefo
## Modified: 2025-11-08

#!/usr/bin/env python3
"""
Test script for the complete Oracle Loop implementation.
Tests the multi-threaded AGI system with sensory processing, neural simulation, and dialogue.
"""

import unittest
import threading
import time
import queue
import sys
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add psinet to path
sys.path.insert(0, str(Path(__file__).parent))

from run_agi import SharedState, PSINetCoreThread, SensoryThread, DialogueThread


class TestSharedState(unittest.TestCase):
    """Test the SharedState class for inter-thread communication."""
    
    def setUp(self):
        self.shared_state = SharedState()
    
    def test_initialization(self):
        """Test SharedState initialization."""
        self.assertTrue(self.shared_state.running)
        self.assertIsInstance(self.shared_state.sensory_data_queue, queue.Queue)
        self.assertIsInstance(self.shared_state.user_input_queue, queue.Queue)
        self.assertIsInstance(self.shared_state.agi_output_queue, queue.Queue)
        self.assertIsNone(self.shared_state.neural_rates)
        self.assertEqual(type(self.shared_state.lock).__name__, 'lock')
        self.assertIsInstance(self.shared_state.stats, dict)
    
    def test_stats_initialization(self):
        """Test statistics initialization."""
        stats = self.shared_state.stats
        self.assertEqual(stats['sensory_updates'], 0)
        self.assertEqual(stats['simulation_steps'], 0)
        self.assertEqual(stats['user_inputs'], 0)
        self.assertEqual(stats['agi_outputs'], 0)
        self.assertIsInstance(stats['start_time'], float)


class TestPSINetCoreThread(unittest.TestCase):
    """Test the PSINet Core thread."""
    
    def setUp(self):
        self.shared_state = SharedState()
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a minimal config file
        self.config_path = Path(self.temp_dir) / "test_config.yaml"
        config_content = """
simulation_params:
  brian2_device: runtime
  chunk_duration_ms: 10
  max_duration_seconds: 5

input_params:
  source: realtime
  realtime_config:
    enable_video: false
    enable_audio: false
    video_device_id: 0
    audio_device_id: 0
    video_resolution: [32, 32]
    audio_sample_rate: 16000
    audio_chunk_size: 1024
    dvs_threshold: 30
    max_rate_hz: 100

network_params:
  layers:
    - name: L1
      num_excitatory: 50
      num_inhibitory: 12
      enable_lateral_inhibition: true
      lateral_strength: 0.2

connections_params:
  inp_l1:
    w_max: 0.3
    a_plus: 0.01
    a_minus: -0.01

monitor_params:
  record_l1_spikes: true
  record_input_spikes: false
  record_weight_subset_size: 50

output_params:
  base_output_dir: test_outputs
  save_raw_data: true
  save_plots: false

language_params:
  enable_language: true
  vocab_size: 50
  concept_neurons: 25
  curiosity_threshold: 0.7
  curiosity_duration: 2.0
"""
        with open(self.config_path, 'w') as f:
            f.write(config_content)
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    @patch('run_agi.Simulator')
    def test_thread_initialization(self, mock_simulator):
        """Test PSINetCoreThread initialization."""
        thread = PSINetCoreThread(self.shared_state, str(self.config_path))
        
        self.assertEqual(thread.shared_state, self.shared_state)
        self.assertEqual(thread.config_path, str(self.config_path))
        self.assertIsNone(thread.simulator)
        self.assertIsNone(thread.hierarchy)
        self.assertEqual(thread.name, "PSINet-Core")
    
    @patch('run_agi.Simulator')
    def test_thread_run_with_mock(self, mock_simulator_class):
        """Test PSINetCoreThread run method with mocked simulator."""
        # Setup mock simulator
        mock_simulator = Mock()
        mock_simulator_class.return_value = mock_simulator
        
        # Mock network components
        mock_hierarchy = Mock()
        mock_hierarchy.input_layer = Mock()
        mock_hierarchy.layers_in_order = ['L1']
        
        mock_network_objects = {
            'hierarchy': mock_hierarchy,
            'language_input': Mock(),
            'language_output': Mock()
        }
        
        mock_monitors = {
            'spikes_L1': Mock()
        }
        
        mock_brian2_network = Mock()
        mock_brian2_network.timestep = 0.1
        
        mock_simulator.network_objects = mock_network_objects
        mock_simulator.monitors = mock_monitors
        mock_simulator.brian2_network = mock_brian2_network
        
        # Create and start thread
        thread = PSINetCoreThread(self.shared_state, str(self.config_path))
        
        # Start thread and let it run briefly
        thread.start()
        time.sleep(0.1)
        
        # Stop thread
        self.shared_state.running = False
        thread.join(timeout=2.0)
        
        # Verify simulator was built
        mock_simulator.build.assert_called_once()
        
        # Verify thread stopped
        self.assertFalse(thread.is_alive())


class TestSensoryThread(unittest.TestCase):
    """Test the Sensory processing thread."""
    
    def setUp(self):
        self.shared_state = SharedState()
        self.sensory_config = {
            'enable_video': False,
            'enable_audio': False,
            'video_device_id': 0,
            'audio_device_id': 0,
            'video_resolution': [32, 32],
            'audio_sample_rate': 16000,
            'audio_chunk_size': 1024,
            'dvs_threshold': 30,
            'max_rate_hz': 100
        }
    
    def test_thread_initialization(self):
        """Test SensoryThread initialization."""
        thread = SensoryThread(self.shared_state, self.sensory_config)
        
        self.assertEqual(thread.shared_state, self.shared_state)
        self.assertEqual(thread.sensory_config, self.sensory_config)
        self.assertIsNone(thread.realtime_encoder)
        self.assertEqual(thread.name, "Sensory")
    
    @patch('run_agi.RealtimeEncoder')
    def test_thread_run_with_mock(self, mock_encoder_class):
        """Test SensoryThread run method with mocked encoder."""
        # Setup mock encoder
        mock_encoder = Mock()
        mock_encoder.initialize.return_value = True
        mock_encoder.get_spike_rates.return_value = [0.1, 0.2, 0.3] * 100
        mock_encoder_class.return_value = mock_encoder
        
        # Create and start thread
        thread = SensoryThread(self.shared_state, self.sensory_config)
        
        # Start thread and let it run briefly
        thread.start()
        time.sleep(0.1)
        
        # Stop thread
        self.shared_state.running = False
        thread.join(timeout=2.0)
        
        # Verify encoder was initialized
        mock_encoder.initialize.assert_called_once()
        
        # Verify thread stopped
        self.assertFalse(thread.is_alive())


class TestDialogueThread(unittest.TestCase):
    """Test the Dialogue interface thread."""
    
    def setUp(self):
        self.shared_state = SharedState()
    
    def test_thread_initialization(self):
        """Test DialogueThread initialization."""
        thread = DialogueThread(self.shared_state)
        
        self.assertEqual(thread.shared_state, self.shared_state)
        self.assertEqual(thread.name, "Dialogue")
    
    @patch('builtins.input', side_effect=['hello', 'stats', 'quit'])
    @patch('builtins.print')
    def test_thread_run_with_mock_input(self, mock_print, mock_input):
        """Test DialogueThread run method with mocked input."""
        # Setup language modules
        self.shared_state.language_input = Mock()
        self.shared_state.language_output = Mock()
        
        # Create and start thread
        thread = DialogueThread(self.shared_state)
        
        # Start thread and wait for completion
        thread.start()
        thread.join(timeout=5.0)
        
        # Verify thread stopped
        self.assertFalse(thread.is_alive())
        
        # Verify shared state was updated
        self.assertFalse(self.shared_state.running)
        self.assertEqual(self.shared_state.stats['user_inputs'], 3)  # hello, stats, quit


class TestOracleLoopIntegration(unittest.TestCase):
    """Integration tests for the complete Oracle Loop."""
    
    def setUp(self):
        self.shared_state = SharedState()
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a minimal config file
        self.config_path = Path(self.temp_dir) / "test_config.yaml"
        config_content = """
simulation_params:
  brian2_device: runtime
  chunk_duration_ms: 10
  max_duration_seconds: 2

input_params:
  source: realtime
  realtime_config:
    enable_video: false
    enable_audio: false
    video_device_id: 0
    audio_device_id: 0
    video_resolution: [32, 32]
    audio_sample_rate: 16000
    audio_chunk_size: 1024
    dvs_threshold: 30
    max_rate_hz: 100

network_params:
  layers:
    - name: L1
      num_excitatory: 20
      num_inhibitory: 5
      enable_lateral_inhibition: true
      lateral_strength: 0.2

connections_params:
  inp_l1:
    w_max: 0.3
    a_plus: 0.01
    a_minus: -0.01

monitor_params:
  record_l1_spikes: true
  record_input_spikes: false
  record_weight_subset_size: 20

output_params:
  base_output_dir: test_outputs
  save_raw_data: true
  save_plots: false

language_params:
  enable_language: true
  vocab_size: 20
  concept_neurons: 10
  curiosity_threshold: 0.7
  curiosity_duration: 2.0
"""
        with open(self.config_path, 'w') as f:
            f.write(config_content)
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    @patch('run_agi.Simulator')
    @patch('run_agi.RealtimeEncoder')
    @patch('builtins.input', side_effect=['hello', 'quit'])
    @patch('builtins.print')
    def test_full_oracle_loop_integration(self, mock_print, mock_input, 
                                         mock_encoder_class, mock_simulator_class):
        """Test the complete Oracle Loop integration."""
        # Setup mock simulator
        mock_simulator = Mock()
        mock_simulator_class.return_value = mock_simulator
        
        # Mock network components
        mock_hierarchy = Mock()
        mock_hierarchy.input_layer = Mock()
        mock_hierarchy.layers_in_order = ['L1']
        
        mock_network_objects = {
            'hierarchy': mock_hierarchy,
            'language_input': Mock(),
            'language_output': Mock()
        }
        
        mock_monitors = {
            'spikes_L1': Mock()
        }
        
        mock_brian2_network = Mock()
        mock_brian2_network.timestep = 0.1
        
        mock_simulator.network_objects = mock_network_objects
        mock_simulator.monitors = mock_monitors
        mock_simulator.brian2_network = mock_brian2_network
        
        # Setup mock encoder
        mock_encoder = Mock()
        mock_encoder.initialize.return_value = True
        mock_encoder.get_spike_rates.return_value = [0.1, 0.2, 0.3] * 20
        mock_encoder_class.return_value = mock_encoder
        
        # Create threads
        psinet_thread = PSINetCoreThread(self.shared_state, str(self.config_path))
        sensory_thread = SensoryThread(self.shared_state, {})
        dialogue_thread = DialogueThread(self.shared_state)
        
        # Start all threads
        psinet_thread.start()
        sensory_thread.start()
        dialogue_thread.start()
        
        # Let threads run briefly
        time.sleep(0.5)
        
        # Wait for dialogue thread to complete (should finish after 'quit')
        dialogue_thread.join(timeout=3.0)
        
        # Stop other threads
        self.shared_state.running = False
        psinet_thread.join(timeout=2.0)
        sensory_thread.join(timeout=2.0)
        
        # Verify all threads stopped
        self.assertFalse(psinet_thread.is_alive())
        self.assertFalse(sensory_thread.is_alive())
        self.assertFalse(dialogue_thread.is_alive())
        
        # Verify statistics were updated
        stats = self.shared_state.stats
        self.assertGreater(stats['user_inputs'], 0)
        self.assertGreater(stats['simulation_steps'], 0)
        self.assertGreater(stats['sensory_updates'], 0)


class TestOracleLoopPerformance(unittest.TestCase):
    """Performance tests for the Oracle Loop."""
    
    def setUp(self):
        self.shared_state = SharedState()
    
    def test_queue_performance(self):
        """Test queue performance under load."""
        # Test sensory data queue
        for i in range(1000):
            try:
                self.shared_state.sensory_data_queue.put_nowait(f"data_{i}")
            except queue.Full:
                break
        
        # Test AGI output queue
        for i in range(100):
            try:
                self.shared_state.agi_output_queue.put_nowait(f"output_{i}")
            except queue.Full:
                break
        
        # Verify queues are not empty
        self.assertFalse(self.shared_state.sensory_data_queue.empty())
        self.assertFalse(self.shared_state.agi_output_queue.empty())
    
    def test_lock_performance(self):
        """Test lock performance under contention."""
        def worker():
            for _ in range(100):
                with self.shared_state.lock:
                    self.shared_state.neural_rates = [0.1] * 100
                    time.sleep(0.001)
        
        # Create multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify final state
        self.assertIsNotNone(self.shared_state.neural_rates)


if __name__ == '__main__':
    # Configure logging for tests
    import logging
    logging.basicConfig(level=logging.WARNING)
    
    # Run tests
    unittest.main(verbosity=2)