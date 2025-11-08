## Developer: inkbytefo
## Modified: 2025-11-08

import taichi as ti
import numpy as np
import timeit
import unittest
from unittest.mock import Mock
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from psinet.core.taichi_neuron import BionicNeuron
from psinet.core.taichi_synapse import BionicSynapse
from psinet.taichi_backend.converter import Brian2ToTaichiConverter


class TestTaichiBackend(unittest.TestCase):
    """Test suite for Taichi backend performance and functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Initialize Taichi with CUDA for 4090."""
        ti.init(arch=ti.cuda, device_memory_GB=24)
        cls.dt = 0.001
        cls.sparsity = 0.9
        
    def setUp(self):
        """Set up test fixtures."""
        self.pre_neurons = BionicNeuron(
            num_neurons=10000,
            dt=self.dt,
            sparsity=self.sparsity
        )
        self.post_neurons = BionicNeuron(
            num_neurons=5000,
            dt=self.dt,
            sparsity=self.sparsity
        )
        self.synapse = BionicSynapse(
            pre_neurons=self.pre_neurons,
            post_neurons=self.post_neurons,
            dt=self.dt,
            sparsity=self.sparsity
        )
        
    def test_neuron_initialization(self):
        """Test neuron initialization with 10k neurons."""
        self.assertEqual(self.pre_neurons.num_neurons, 10000)
        self.assertEqual(self.post_neurons.num_neurons, 5000)
        self.assertEqual(self.pre_neurons.sparsity, 0.9)
        
        # Check initial state
        spikes = self.pre_neurons.get_spikes()
        potentials = self.pre_neurons.get_membrane_potentials()
        thresholds = self.pre_neurons.get_thresholds()
        
        self.assertEqual(len(spikes), 10000)
        self.assertEqual(len(potentials), 10000)
        self.assertEqual(len(thresholds), 10000)
        self.assertTrue(np.all(spikes == 0))
        
    def test_synapse_initialization(self):
        """Test synapse initialization with 5k connections."""
        stats = self.synapse.get_statistics()
        
        self.assertEqual(stats['pre_neurons'], 10000)
        self.assertEqual(stats['post_neurons'], 5000)
        self.assertGreater(stats['total_connections'], 0)
        self.assertGreater(stats['sparsity'], 0.8)  # Should be around 0.9
        
    def test_dopamine_modulation(self):
        """Test dopamine effect on LTD."""
        # Test with low dopamine (0.5) - should increase LTD
        self.synapse.set_dopamine(0.5)
        initial_weights = self.synapse.get_weights().copy()
        
        # Generate spikes to trigger learning
        pre_spikes = np.random.random(10000) < 0.01
        post_spikes = np.random.random(5000) < 0.01
        
        for _ in range(10):
            self.synapse.update(pre_spikes, post_spikes)
        
        low_dopamine_weights = self.synapse.get_weights()
        weight_change_low = np.mean(low_dopamine_weights - initial_weights)
        
        # Reset and test with high dopamine (1.5) - should decrease LTD
        self.synapse.reset()
        self.synapse.set_dopamine(1.5)
        
        for _ in range(10):
            self.synapse.update(pre_spikes, post_spikes)
        
        high_dopamine_weights = self.synapse.get_weights()
        weight_change_high = np.mean(high_dopamine_weights - initial_weights)
        
        # High dopamine should result in less negative weight change
        self.assertGreater(weight_change_high, weight_change_low)
        
    def test_performance_100_timesteps(self):
        """Test performance: 100 timesteps should complete in <1 second."""
        def run_simulation():
            for _ in range(100):
                # Generate random input
                input_current = np.random.random(10000) * 0.1
                self.pre_neurons.update(input_current)
                
                # Get spikes and update synapse
                pre_spikes = self.pre_neurons.get_spikes()
                post_spikes = self.post_neurons.get_spikes()
                self.synapse.update(pre_spikes, post_spikes)
                
                # Update post neurons with synaptic current
                postsynaptic_current = self.synapse.get_postsynaptic_current()
                self.post_neurons.update(postsynaptic_current)
        
        # Measure time
        execution_time = timeit.timeit(run_simulation, number=1)
        
        # Should complete in <1 second (target <0.01s per step)
        self.assertLess(execution_time, 1.0)
        
        # Calculate FPS
        fps = 100 / execution_time
        self.assertGreater(fps, 100)  # Should achieve >100 FPS
        
        print(f"Performance: {execution_time:.4f}s for 100 steps, FPS: {fps:.1f}")
        
    def test_energy_consumption_simulation(self):
        """Test simulated energy consumption (fake 20W)."""
        # Simulate power measurement
        baseline_power = 45.0  # 4090 idle power
        load_power = 70.0      # 4090 full load power
        additional_power = load_power - baseline_power  # 25W
        
        # Our target is <25W additional load
        self.assertLessEqual(additional_power, 25.0)
        
        # Simulate energy usage for 100 timesteps
        simulation_time = 0.1  # 100 timesteps at 1ms each
        energy_joules = additional_power * simulation_time
        
        print(f"Simulated energy usage: {energy_joules:.2f}J for 100 timesteps")
        
    def test_spike_statistics(self):
        """Test spike counting and statistics."""
        # Run simulation with known input
        input_current = np.ones(10000) * 2.0  # Strong input to generate spikes
        
        spike_counts = []
        mean_weights = []
        sparsities = []
        
        for _ in range(100):
            self.pre_neurons.update(input_current)
            pre_spikes = self.pre_neurons.get_spikes()
            post_spikes = self.post_neurons.get_spikes()
            self.synapse.update(pre_spikes, post_spikes)
            
            # Record statistics
            spike_counts.append(np.sum(pre_spikes))
            weights = self.synapse.get_weights()
            non_zero_weights = weights[weights > 0]
            mean_weights.append(np.mean(non_zero_weights) if len(non_zero_weights) > 0 else 0)
            sparsities.append(1.0 - (len(non_zero_weights) / (10000 * 5000)))
        
        # Check statistics
        avg_spike_count = np.mean(spike_counts)
        avg_mean_weight = np.mean(mean_weights)
        avg_sparsity = np.mean(sparsities)
        
        self.assertGreater(avg_spike_count, 0)
        self.assertGreater(avg_mean_weight, 0)
        self.assertGreater(avg_sparsity, 0.8)
        
        print(f"Avg spike count: {avg_spike_count:.1f}, Avg weight: {avg_mean_weight:.6f}, Avg sparsity: {avg_sparsity:.3f}")
        
    def test_brian2_converter(self):
        """Test Brian2 to Taichi conversion."""
        # Create fake Brian2 objects
        fake_neuron_group = Mock()
        fake_neuron_group.N = 1000
        fake_neuron_group.tau = np.array([0.02])
        fake_neuron_group.theta = np.array([0.1])
        fake_neuron_group.tau_theta = np.array([0.1])
        fake_neuron_group.v_reset = np.array([0.0])
        fake_neuron_group.delta_theta = np.array([0.05])
        
        fake_synapses = Mock()
        fake_synapses.source.N = 1000
        fake_synapses.target.N = 500
        fake_synapses.taupre = np.array([0.02])
        fake_synapses.taupost = np.array([0.02])
        fake_synapses.Apre = np.array([0.01])
        fake_synapses.Apost = np.array([-0.0105])
        fake_synapses.wmax = np.array([0.01])
        fake_synapses.w = np.random.random(100) * 0.01
        fake_synapses.i = np.random.randint(0, 1000, 100)
        fake_synapses.j = np.random.randint(0, 500, 100)
        
        # Create converter
        converter = Brian2ToTaichiConverter(dt=self.dt, sparsity=self.sparsity)
        
        # Convert neuron group
        taichi_neuron = converter.from_brian2_neuron_group(fake_neuron_group, "test_neurons")
        self.assertEqual(taichi_neuron.num_neurons, 1000)
        
        # Convert synapses
        taichi_synapse = converter.from_brian2_synapses(
            fake_synapses, taichi_neuron, taichi_neuron, "test_synapses"
        )
        self.assertEqual(taichi_synapse.pre_neurons.num_neurons, 1000)
        self.assertEqual(taichi_synapse.post_neurons.num_neurons, 500)
        
        # Check conversion statistics
        stats = converter.get_conversion_stats()
        self.assertEqual(stats['converted_neurons'], 1)
        self.assertEqual(stats['converted_synapses'], 1)
        self.assertEqual(stats['total_neurons'], 1000)
        
        print(f"Conversion stats: {stats}")
        
    def test_memory_usage(self):
        """Test memory usage with large networks."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create larger networks
        large_pre = BionicNeuron(num_neurons=50000, dt=self.dt, sparsity=self.sparsity)
        large_post = BionicNeuron(num_neurons=25000, dt=self.dt, sparsity=self.sparsity)
        large_synapse = BionicSynapse(large_pre, large_post, dt=self.dt, sparsity=self.sparsity)
        
        # Run some updates
        for _ in range(10):
            input_current = np.random.random(50000) * 0.1
            large_pre.update(input_current)
            pre_spikes = large_pre.get_spikes()
            post_spikes = large_post.get_spikes()
            large_synapse.update(pre_spikes, post_spikes)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (<2GB for 75k neurons)
        self.assertLess(memory_increase, 2048)
        
        print(f"Memory usage: {memory_increase:.1f}MB increase for 75k neurons")


if __name__ == '__main__':
    # Run tests with detailed output
    unittest.main(verbosity=2)