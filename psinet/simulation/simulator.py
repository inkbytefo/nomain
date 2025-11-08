import yaml
import numpy as np
import brian2 as b2
from pathlib import Path
import datetime
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from psinet.io.encoders import image_to_poisson_rates, create_input_layer
from psinet.io.loaders import load_mnist
from psinet.io.realtime_encoders import RealtimeEncoder
from psinet.network.hierarchy import Hierarchy
from psinet.language.input import TextToConcepts
from psinet.language.output import ConceptsToText

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Simulator:
    """
    Orchestrates a PSINet simulation from a configuration file.
    Handles network building, running the simulation, monitoring, and saving results.
    """
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self._load_config()
        self._setup_output_directory()
        self.network_objects = {}
        self.monitors = {}
        self.brian2_network = None
        # self.taichi_sim = None  # TODO: Implement Taichi backend migration
        # TODO: Use Taichi network in run() method instead of brian2_network
        # self.taichi_sim = taichi_net  # Replace brian2_network with Taichi backend
        self.language_input = None
        self.language_output = None

    def _load_config(self):
        logging.info(f"Loading configuration from: {self.config_path}")
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)

    def _setup_output_directory(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        run_name = self.config.get('run_id', 'unnamed_run')
        base_dir = Path(self.config['output_params']['base_output_dir'])
        self.output_dir = base_dir / f"{run_name}_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Output will be saved to: {self.output_dir}")

    # ------------------------------- BUILD ---------------------------------
    def build(self):
        logging.info("Building network components...")

        sim_p = self.config['simulation_params']
        inp_p = self.config['input_params']
        net_p = self.config['network_params']
        mon_p = self.config['monitor_params']
        learn_p = self.config.get('learning_params', {})

        # Device selection
        device = sim_p.get('brian2_device', 'runtime')
        if device == 'cpp_standalone':
            b2.set_device('cpp_standalone', directory=str(self.output_dir / 'brian2_build'), build_on_run=False)
        else:
            b2.prefs.codegen.target = 'numpy'
        # Ensure if user supplies tau constants in YAML we keep brian2 units compatible

        # 1) Input preparation: support mnist, synthetic, and realtime
        dataset = inp_p.get('source', 'mnist')  # Changed from 'dataset' to 'source' for clarity

        def create_digit_zero():
            img = np.zeros((28, 28), dtype=np.uint8)
            for i in range(28):
                for j in range(28):
                    dist = np.sqrt((i - 14)**2 + (j - 14)**2)
                    if 8 <= dist <= 12:
                        img[i, j] = 255
            return img

        def create_digit_one():
            img = np.zeros((28, 28), dtype=np.uint8)
            img[:, 14:17] = 255
            img[4:8, 12:20] = 255
            return img

        # Handle realtime mode
        if dataset == 'realtime':
            logging.info("Initializing realtime sensory encoders...")
            
            # Initialize realtime encoder
            realtime_config = inp_p.get('realtime_config', {})
            realtime_encoder = RealtimeEncoder(realtime_config)
            
            if not realtime_encoder.initialize():
                raise RuntimeError("Failed to initialize realtime encoders")
            
            # Get initial spike rates to determine input size
            initial_rates = realtime_encoder.get_spike_rates()
            input_size = len(initial_rates)
            
            # Create input layer with zero rates initially
            input_layer = create_input_layer(initial_rates * 0)
            
            self.network_objects['input_layer'] = input_layer
            self.network_objects['realtime_encoder'] = realtime_encoder
            self.network_objects['is_realtime'] = True
            
            logging.info(f"Realtime mode initialized with input size: {input_size}")
            
        else:
            # Traditional MNIST or synthetic mode
            rates_map = {}
            digits_list = inp_p.get('patterns_to_learn', [0, 1])
            max_rate = inp_p.get('max_rate_hz', 150)

            if dataset == 'mnist':
                try:
                    images, labels = load_mnist(inp_p.get('data_dir', '~/.psinet_data'))
                    # Allow per-digit index; default to 0 for each if not provided
                    indices_cfg = inp_p.get('image_indices', None)
                    for idx_d, d in enumerate(digits_list):
                        if indices_cfg is None:
                            idx = 0
                        elif isinstance(indices_cfg, list):
                            idx = indices_cfg[min(idx_d, len(indices_cfg)-1)]
                        else:
                            idx = int(indices_cfg)
                        where = np.where(labels == d)[0]
                        if len(where) == 0:
                            raise RuntimeError(f"No images found for digit {d}")
                        sel = where[min(idx, len(where)-1)]
                        img = images[sel]
                        rates_map[d] = image_to_poisson_rates(img, max_rate=max_rate*b2.Hz, invert=False)
                except Exception as e:
                    logging.warning(f"MNIST yüklenemedi (hata: {e}). Sentetik 0/1 ile devam ediliyor. Hata: {e}")
                    img0 = create_digit_zero()
                    img1 = create_digit_one()
                    rates_map = {0: image_to_poisson_rates(img0, max_rate=max_rate*b2.Hz, invert=False),
                                 1: image_to_poisson_rates(img1, max_rate=max_rate*b2.Hz, invert=False)}
                    digits_list = [0, 1]
            else:
                # synthetic fallback
                img0 = create_digit_zero()
                img1 = create_digit_one()
                rates_map = {0: image_to_poisson_rates(img0, max_rate=max_rate*b2.Hz, invert=False),
                             1: image_to_poisson_rates(img1, max_rate=max_rate*b2.Hz, invert=False)}
                digits_list = [0, 1]

            # dynamic input layer initialized to silence
            any_rates = next(iter(rates_map.values()))
            input_layer = create_input_layer(any_rates*0)
            self.network_objects['input_layer'] = input_layer
            self.network_objects['rates_map'] = rates_map
            self.network_objects['digits_list'] = digits_list
            self.network_objects['is_realtime'] = False

        # Initialize language modules if enabled
        language_config = self.config.get('language_params', {})
        if language_config.get('enable_language', False):
            logging.info("Initializing language modules...")
            
            # Initialize text-to-concepts
            vocab_size = language_config.get('vocab_size', 100)
            concept_neurons = language_config.get('concept_neurons', 50)
            self.language_input = TextToConcepts(vocab_size, concept_neurons)
            
            # Initialize concepts-to-text
            curiosity_threshold = language_config.get('curiosity_threshold', 0.7)
            curiosity_duration = language_config.get('curiosity_duration', 2.0)
            self.language_output = ConceptsToText(
                self.language_input, curiosity_threshold, curiosity_duration
            )
            
            self.network_objects['language_input'] = self.language_input
            self.network_objects['language_output'] = self.language_output
            self.network_objects['language_enabled'] = True
            
            logging.info(f"Language modules initialized: {self.language_input.get_vocabulary_stats()}")
        else:
            self.network_objects['language_enabled'] = False

        # 2) Build hierarchy (multi-layer aware)
        connections_params = self.config.get('connections_params', None)
        if 'layers' in net_p:
            layers_config = net_p['layers']
        else:
            # legacy single-layer config mapping
            layers_config = [{
                'name': 'L1',
                'num_excitatory': net_p.get('num_excitatory_l1', 100),
                'num_inhibitory': net_p.get('num_inhibitory_l1', 25),
                'enable_lateral_inhibition': net_p.get('enable_lateral_inhibition', True),
                'lateral_strength': net_p.get('lateral_strength', 0.2),
            }]
            if connections_params is None:
                # derive from legacy learning_params
                connections_params = {
                    'inp_l1': {
                        'w_max': learn_p.get('w_max_inp_l1', 0.3),
                        'a_plus': learn_p.get('a_plus_inp_l1', 0.01),
                        'a_minus': learn_p.get('a_minus_inp_l1', -0.01),
                    }
                }

        hierarchy = Hierarchy(
            input_layer=input_layer,
            layers_config=layers_config,
            connections_params=connections_params
        )
        self.network_objects['hierarchy'] = hierarchy

        # 3) Monitors
        components = []
        # spikes: L1 always
        if mon_p.get('record_l1_spikes', True):
            l1_name = hierarchy.layers_in_order[0]
            self.monitors['spikes_L1'] = b2.SpikeMonitor(hierarchy.layers_by_name[l1_name].excitatory_neurons.group)
            components.append(self.monitors['spikes_L1'])
        if mon_p.get('record_input_spikes', False):
            self.monitors['input_spikes'] = b2.SpikeMonitor(input_layer)
            components.append(self.monitors['input_spikes'])
        # spikes: L2 if present
        if len(hierarchy.layers_in_order) >= 2:
            l2_name = hierarchy.layers_in_order[1]
            self.monitors['spikes_L2'] = b2.SpikeMonitor(hierarchy.layers_by_name[l2_name].excitatory_neurons.group)
            components.append(self.monitors['spikes_L2'])
            # Optional: add a population rate monitor for L2 (could aid performance debugging)
            # self.monitors['rate_L2'] = b2.PopulationRateMonitor(hierarchy.layers_by_name[l2_name].excitatory_neurons.group)
            # components.append(self.monitors['rate_L2'])
        
        # spikes: L3 (Association Cortex) if present
        if len(hierarchy.layers_in_order) >= 3:
            l3_name = hierarchy.layers_in_order[2]
            if mon_p.get('record_l3_spikes', True):
                self.monitors['spikes_L3'] = b2.SpikeMonitor(hierarchy.layers_by_name[l3_name].excitatory_neurons.group)
                components.append(self.monitors['spikes_L3'])

        # weight subset monitors
        subset_size = int(mon_p.get('record_weight_subset_size', 100))
        # Input->L1
        first_name = hierarchy.layers_in_order[0]
        n_exc_l1 = hierarchy.layers_by_name[first_name].excitatory_neurons.group.N
        total_candidates = (28*28) * n_exc_l1
        subset = np.random.choice(np.arange(total_candidates), min(subset_size, total_candidates), replace=False)
        self.monitors['weights_inp_l1'] = b2.StateMonitor(hierarchy.input_to_first_syn.synapses, 'w', record=subset)
        components.append(self.monitors['weights_inp_l1'])
        # L1->L2 if exists
        if len(hierarchy.layers_in_order) >= 2:
            second_name = hierarchy.layers_in_order[1]
            key = f"{first_name.lower()}_{second_name.lower()}"
            if key in hierarchy.connections:
                n_exc_l2 = hierarchy.layers_by_name[second_name].excitatory_neurons.group.N
                total_cand_l1l2 = n_exc_l1 * n_exc_l2
                subset2 = np.random.choice(np.arange(total_cand_l1l2), min(subset_size, total_cand_l1l2), replace=False)
                self.monitors['weights_l1_l2'] = b2.StateMonitor(hierarchy.connections[key].synapses, 'w', record=subset2)
                components.append(self.monitors['weights_l1_l2'])

        # 4) Assemble Network
        net = hierarchy.build_network(*components)
        self.brian2_network = net

        # In cpp_standalone, multiple run calls require build_on_run=False and explicit build
        sim_p = self.config['simulation_params']
        device = sim_p.get('brian2_device', 'runtime')
        # In cpp_standalone we delay final build until after all run() calls
        logging.info("Network build complete.")

    # ------------------------------- RUN -----------------------------------
    def run(self):
        if self.brian2_network is None:
            logging.error("Network has not been built. Call .build() first.")
            return

        sim_p = self.config['simulation_params']
        hierarchy = self.network_objects['hierarchy']
        is_realtime = self.network_objects.get('is_realtime', False)

        if is_realtime:
            self._run_realtime(sim_p, hierarchy)
        else:
            self._run_batch(sim_p, hierarchy)

        logging.info("Simulation finished.")

    def _run_realtime(self, sim_p, hierarchy):
        """Run simulation in realtime mode with continuous sensory input."""
        realtime_encoder = self.network_objects['realtime_encoder']
        language_enabled = self.network_objects.get('language_enabled', False)
        
        # Realtime simulation parameters
        chunk_duration_ms = sim_p.get('chunk_duration_ms', 10)
        max_duration_seconds = sim_p.get('max_duration_seconds', 60)
        
        chunk_time = chunk_duration_ms * b2.ms
        max_time = max_duration_seconds * b2.second
        
        # Language processing variables
        language_output = self.network_objects.get('language_output')
        last_language_check = 0 * b2.second
        language_check_interval = 1.0 * b2.second  # Check for language output every second
        
        logging.info(f"Starting realtime simulation for {max_duration_seconds}s with {chunk_duration_ms}ms chunks")
        if language_enabled:
            logging.info("Language processing enabled - PSINet will generate text responses")
        
        try:
            current_time = 0 * b2.second
            iteration = 0
            
            while current_time < max_time:
                # Get current spike rates from sensory inputs
                spike_rates = realtime_encoder.get_spike_rates()
                
                # Update input layer with new rates
                hierarchy.input_layer.rates = spike_rates * b2.Hz
                
                # Run simulation for this chunk
                self.brian2_network.run(chunk_time, report='text')
                
                current_time += chunk_time
                iteration += 1
                
                # Language processing (if enabled)
                if language_enabled and language_output and (current_time - last_language_check) >= language_check_interval:
                    self._process_language_output(hierarchy, current_time, language_output)
                    last_language_check = current_time
                
                # Log progress every 100 iterations
                if iteration % 100 == 0:
                    elapsed = float(current_time / b2.second)
                    logging.info(f"Realtime simulation progress: {elapsed:.1f}s / {max_duration_seconds}s")
                
                # Optional: break on keyboard interrupt
                if iteration % 10 == 0:
                    # Check if we should continue (could add external control here)
                    pass
                    
        except KeyboardInterrupt:
            logging.info("Realtime simulation interrupted by user")
        except Exception as e:
            logging.error(f"Error in realtime simulation: {e}")
        finally:
            # Clean up realtime resources
            realtime_encoder.release()

    def _run_batch(self, sim_p, hierarchy):
        """Run simulation in traditional batch mode with MNIST/synthetic patterns."""
        show = sim_p['duration_per_pattern_ms'] * b2.ms
        rest = sim_p['silence_period_ms'] * b2.ms
        cycles = int(sim_p['cycles'])

        rates_map = self.network_objects['rates_map']
        digits_list = self.network_objects['digits_list']

        # Windows for per-digit analysis
        self.windows_per_digit = {d: [] for d in digits_list}

        # Sequence construction: if present_all_digits flag, iterate all digits per cycle
        present_all = bool(sim_p.get('present_all_digits', False))
        if present_all:
            # In cpp_standalone, we need to build once before multiple run() calls
            device = self.config['simulation_params'].get('brian2_device', 'runtime')
            if device == 'cpp_standalone':
                b2.device.build(directory=str(self.output_dir / 'brian2_build'), compile=True, run=False, debug=False)

            logging.info(f"Starting multi-digit loop: {cycles} cycles over digits {digits_list} with show={show}, rest={rest}")
        else:
            logging.info(f"Starting 2-digit loop: {cycles}x over {digits_list} with show={show}, rest={rest}")

        current_t = 0 * b2.second
        import random
        for _ in range(cycles):
            seq = digits_list.copy()
            if present_all:
                random.shuffle(seq)
            for d in seq:
                rates = rates_map[d]
                hierarchy.input_layer.rates = rates
                self.brian2_network.run(show, report='text')
                self.windows_per_digit[d].append((current_t, current_t + show))
                current_t += show
                hierarchy.input_layer.rates = 0 * b2.Hz
                self.brian2_network.run(rest)
                current_t += rest
    def _process_language_output(self, hierarchy, current_time, language_output):
        """
        Process language output based on current network activity.
        
        Args:
            hierarchy: Network hierarchy
            current_time: Current simulation time
            language_output: ConceptsToText instance
        """
        try:
            # Get concept activations from the highest layer (Association Cortex)
            if len(hierarchy.layers_in_order) >= 3:
                # Use L3 (Association Cortex) for language processing
                association_layer_name = hierarchy.layers_in_order[2]
            elif len(hierarchy.layers_in_order) >= 2:
                # Fallback to L2 if L3 not available
                association_layer_name = hierarchy.layers_in_order[1]
            else:
                # Fallback to L1 if only one layer
                association_layer_name = hierarchy.layers_in_order[0]
            
            association_layer = hierarchy.layers_by_name[association_layer_name]
            
            # Get recent spikes from association layer
            if f'spikes_{association_layer_name}' in self.monitors:
                monitor = self.monitors[f'spikes_{association_layer_name}']
                
                # Calculate concept activations from recent spikes
                concept_activations = self._calculate_concept_activations(monitor, current_time)
                
                # Calculate error signal (simplified - in full implementation would use predictive coding)
                error_signal = self._calculate_error_signal(association_layer, current_time)
                
                # Generate language response
                response = language_output.generate_response(
                    concept_activations, 
                    error_signal, 
                    float(current_time / b2.second)
                )
                
                if response:
                    logging.info(f"PSINet says: '{response}'")
                    
        except Exception as e:
            logging.error(f"Error in language processing: {e}")
    
    def _calculate_concept_activations(self, spike_monitor, current_time, time_window=1.0):
        """
        Calculate concept activations from recent spikes.
        
        Args:
            spike_monitor: Brian2 SpikeMonitor
            current_time: Current simulation time
            time_window: Time window to consider (seconds)
            
        Returns:
            Dictionary of concept activations
        """
        # Get recent spikes within time window
        recent_time = current_time - time_window * b2.second
        recent_mask = spike_monitor.t >= recent_time
        
        if not np.any(recent_mask):
            return {}
        
        recent_spikes = spike_monitor.i[recent_mask]
        recent_times = spike_monitor.t[recent_mask]
        
        # Use language input module to get concept activations
        if self.language_input:
            return self.language_input.get_active_concepts(recent_spikes, recent_times, time_window)
        
        return {}
    
    def _calculate_error_signal(self, layer, current_time, time_window=0.5):
        """
        Calculate prediction error signal (simplified implementation).
        In full predictive coding implementation, this would compare predictions with actual inputs.
        
        Args:
            layer: Network layer
            current_time: Current simulation time
            time_window: Time window for error calculation
            
        Returns:
            Error signal value (0-1)
        """
        # Simplified error calculation based on activity variance
        # High variance indicates unexpected/novel stimuli -> higher error
        if hasattr(layer, 'excitatory_neurons'):
            # This is a simplified placeholder
            # In full implementation, would use actual prediction error neurons
            return np.random.uniform(0.1, 0.3)  # Placeholder error signal
        
        return 0.0


    # ------------------------------- SAVE ----------------------------------
    def save_results(self):
        logging.info("Saving results...")
        out = self.output_dir
        out.mkdir(parents=True, exist_ok=True)

        is_realtime = self.network_objects.get('is_realtime', False)

        # raw data
        if self.config['output_params'].get('save_raw_data', True):
            npz_payload = {}
            # spikes L1
            if 'spikes_L1' in self.monitors:
                npz_payload['L1_t_ms'] = (self.monitors['spikes_L1'].t / b2.ms).astype(float)
                npz_payload['L1_i'] = np.array(self.monitors['spikes_L1'].i)
            # spikes L2
            if 'spikes_L2' in self.monitors:
                npz_payload['L2_t_ms'] = (self.monitors['spikes_L2'].t / b2.ms).astype(float)
                npz_payload['L2_i'] = np.array(self.monitors['spikes_L2'].i)
            # input
            if 'input_spikes' in self.monitors:
                npz_payload['inp_t_ms'] = (self.monitors['input_spikes'].t / b2.ms).astype(float)
                npz_payload['inp_i'] = np.array(self.monitors['input_spikes'].i)
            # weights
            if 'weights_inp_l1' in self.monitors:
                npz_payload['w_inp_l1_t_ms'] = (self.monitors['weights_inp_l1'].t / b2.ms).astype(float)
                npz_payload['w_inp_l1'] = self.monitors['weights_inp_l1'].w.T
            if 'weights_l1_l2' in self.monitors:
                npz_payload['w_l1_l2_t_ms'] = (self.monitors['weights_l1_l2'].t / b2.ms).astype(float)
                npz_payload['w_l1_l2'] = self.monitors['weights_l1_l2'].w.T
            
            # windows per digit (only for batch mode)
            if not is_realtime and hasattr(self, 'windows_per_digit'):
                win = {int(k): np.array([[w[0]/b2.ms, w[1]/b2.ms] for w in v]) for k, v in self.windows_per_digit.items()}
                npz_payload['windows_per_digit'] = win
            else:
                # For realtime mode, add metadata
                npz_payload['realtime_mode'] = True
                npz_payload['simulation_duration_s'] = float(self.config['simulation_params'].get('max_duration_seconds', 60))
            
            np.savez_compressed(out / 'raw_data.npz', **npz_payload)

        # plots
        if self.config['output_params'].get('save_plots', True) and 'spikes_L1' in self.monitors:
            l1_name = self.network_objects['hierarchy'].layers_in_order[0]
            N1 = self.network_objects['hierarchy'].layers_by_name[l1_name].excitatory_neurons.group.N
            t1 = self.monitors['spikes_L1'].t
            i1 = self.monitors['spikes_L1'].i

            # Prepare figure with optional L2
            has_l2 = 'spikes_L2' in self.monitors
            fig_rows = 3 if has_l2 else 2
            fig, axes = plt.subplots(fig_rows, 3 if has_l2 else 2, figsize=(18 if has_l2 else 14, 6 + 3*fig_rows))

            # L1 raster
            axes[0, 0].plot(t1 / b2.ms, i1, '.k', markersize=1)
            axes[0, 0].set_title('L1 Spikes')
            axes[0, 0].set_xlabel('Time (ms)')
            axes[0, 0].set_ylabel('Neuron index')
            axes[0, 0].grid(True, alpha=0.3)

            # Different analysis for realtime vs batch mode
            if is_realtime:
                # Realtime mode: show overall activity distribution
                total_counts = np.bincount(i1, minlength=N1)
                axes[0, 1].bar(np.arange(N1), total_counts, color='tab:gray', width=0.9)
                axes[0, 1].set_title('L1 Total Spike Counts (Realtime)')
                axes[0, 1].set_xlabel('Neuron index')
                axes[0, 1].set_ylabel('Spike count')
            else:
                # Batch mode: per-digit counts for L1
                counts_per_digit = {d: np.zeros(N1, dtype=int) for d in self.windows_per_digit.keys()}
                for d, windows in self.windows_per_digit.items():
                    for t0, t1w in windows:
                        mask = (t1 >= t0) & (t1 < t1w)
                        if np.any(mask):
                            idx, cnt = np.unique(i1[mask], return_counts=True)
                            counts_per_digit[d][idx] += cnt

                # Simple preference plot for two digits if present, else sum top/bottom
                if len(counts_per_digit) == 2:
                    digits = sorted(counts_per_digit.keys())
                    pref = counts_per_digit[digits[0]] - counts_per_digit[digits[1]]
                    colors = np.where(pref >= 0, 'tab:red', 'tab:blue')
                    axes[0, 1].bar(np.arange(N1), pref, color=colors, width=0.9)
                    axes[0, 1].set_title(f'Preference (pos: {digits[0]}, neg: {digits[1]})')
                    axes[0, 1].axhline(0, color='k', linewidth=0.8)
                else:
                    total_counts = np.sum(np.stack(list(counts_per_digit.values()), axis=0), axis=0)
                    axes[0, 1].bar(np.arange(N1), total_counts, color='tab:gray', width=0.9)
                    axes[0, 1].set_title('L1 total spike counts (all digits)')

            # Weight dynamics (Input->L1)
            if 'weights_inp_l1' in self.monitors:
                axes[1, 0].plot(self.monitors['weights_inp_l1'].t / b2.ms, self.monitors['weights_inp_l1'].w.T, alpha=0.7, linewidth=0.8)
                axes[1, 0].set_title('Input→L1 weight dynamics (sampled)')
                axes[1, 0].set_xlabel('Time (ms)')
                axes[1, 0].set_ylabel('w')
                axes[1, 0].grid(True, alpha=0.3)

            # Summary text / L2 or L1->L2 weights
            if has_l2:
                t2 = self.monitors['spikes_L2'].t
                i2 = self.monitors['spikes_L2'].i
                axes[1, 1].plot(t2 / b2.ms, i2, '.k', markersize=1)
                axes[1, 1].set_title('L2 Spikes')
                axes[1, 1].set_xlabel('Time (ms)')
                axes[1, 1].set_ylabel('Neuron index')
                axes[1, 1].grid(True, alpha=0.3)

                # L1->L2 weight dynamics if present
                if 'weights_l1_l2' in self.monitors and fig_rows >= 3:
                    axes[2, 0].plot(self.monitors['weights_l1_l2'].t / b2.ms, self.monitors['weights_l1_l2'].w.T, alpha=0.7, linewidth=0.8)
                    axes[2, 0].set_title('L1→L2 weight dynamics (sampled)')
                    axes[2, 0].set_xlabel('Time (ms)')
                    axes[2, 0].set_ylabel('w')

                # Enhanced L2 analysis (different for realtime vs batch mode)
                try:
                    l2_name = self.network_objects['hierarchy'].layers_in_order[1]
                    N2 = self.network_objects['hierarchy'].layers_by_name[l2_name].excitatory_neurons.group.N
                except Exception:
                    N2 = None
                    
                if N2 is not None:
                    ax_sel = axes[1, 2] if fig_rows == 3 else axes[1, 1]
                    
                    if is_realtime:
                        # Realtime mode: show overall L2 activity
                        total_counts_l2 = np.bincount(i2, minlength=N2)
                        ax_sel.bar(np.arange(N2), total_counts_l2, color='tab:blue', width=0.9)
                        ax_sel.set_title('L2 Total Spike Counts (Realtime)', fontsize=10, fontweight='bold')
                        ax_sel.set_xlabel('L2 Neuron Index', fontsize=9)
                        ax_sel.set_ylabel('Spike Count', fontsize=9)
                        ax_sel.grid(True, alpha=0.3, axis='y')
                        
                        # Add statistics text for realtime
                        total_spikes_l2 = len(i2)
                        active_l2 = len(np.unique(i2))
                        stats_text = f"Total L2 spikes: {total_spikes_l2}\n"
                        stats_text += f"Active L2 neurons: {active_l2}/{N2}\n"
                        stats_text += f"Mean spikes/neuron: {total_spikes_l2/N2:.1f}"
                        
                        ax_sel.text(0.02, 0.98, stats_text, transform=ax_sel.transAxes,
                                   fontsize=8, verticalalignment='top',
                                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
                    else:
                        # Batch mode: per-digit selectivity analysis
                        counts_l2 = {d: np.zeros(N2, dtype=int) for d in self.windows_per_digit.keys()}
                        for d, windows in self.windows_per_digit.items():
                            for t0, t1w in windows:
                                m = (t2 >= t0) & (t2 < t1w)
                                if np.any(m):
                                    idx, cnt = np.unique(i2[m], return_counts=True)
                                    # clip indices to N2 in case of any monitor drift
                                    mask_valid = idx < N2
                                    counts_l2[d][idx[mask_valid]] += cnt[mask_valid]
                        
                        # Find preferred digit for each L2 neuron
                        digits_sorted = sorted(self.windows_per_digit.keys())
                        mat = np.stack([counts_l2[d] for d in digits_sorted], axis=1)  # shape (N2, num_digits)
                        pref_digit_idx = np.argmax(mat, axis=1)
                        pref_digit = np.array(digits_sorted)[pref_digit_idx]
                        pref_count = mat[np.arange(N2), pref_digit_idx]
                        
                        # Enhanced L2 selectivity bar chart
                        cmap = plt.get_cmap('tab10')
                        colors = [cmap(int(d) % 10) for d in pref_digit]
                        
                        # Create bar chart with enhanced visualization
                        bars = ax_sel.bar(np.arange(N2), pref_count, color=colors, width=0.9)
                        ax_sel.set_title('L2 Neuron Selectivity Analysis\n(Bar color = Preferred Digit, Height = Spike Count)',
                                       fontsize=10, fontweight='bold')
                        ax_sel.set_xlabel('L2 Neuron Index', fontsize=9)
                        ax_sel.set_ylabel('Spike Count for Preferred Digit', fontsize=9)
                        ax_sel.grid(True, alpha=0.3, axis='y')
                        
                        # Add colorbar legend mapping digits to colors
                        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=9))
                        sm.set_array([])
                        cbar = plt.colorbar(sm, ax=ax_sel, fraction=0.046, pad=0.04)
                        cbar.set_ticks(np.arange(10))
                        cbar.set_ticklabels([str(d) for d in range(10)])
                        cbar.set_label('Digit', fontsize=9)
                        
                        # Add statistics text
                        total_neurons = N2
                        digit_specialists = {d: np.sum(pref_digit == d) for d in digits_sorted}
                        max_specialists = max(digit_specialists.values())
                        most_specialized_digit = [d for d, count in digit_specialists.items() if count == max_specialists][0]
                        
                        stats_text = f"Total L2 neurons: {total_neurons}\n"
                        stats_text += f"Most specialized digit: {most_specialized_digit} ({max_specialists} neurons)\n"
                        stats_text += f"Digit specialists: " + ", ".join([f"{d}:{count}" for d, count in digit_specialists.items()])
                        
                        ax_sel.text(0.02, 0.98, stats_text, transform=ax_sel.transAxes,
                                   fontsize=8, verticalalignment='top',
                                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

                    axes[2, 0].grid(True, alpha=0.3)

                # Summary text (different for realtime vs batch mode)
                total_spikes_l1 = len(t1)
                active_l1 = len(np.unique(i1))
                total_spikes_l2 = len(t2)
                active_l2 = len(np.unique(i2))
                
                if is_realtime:
                    duration_s = self.config['simulation_params'].get('max_duration_seconds', 60)
                    text = f"REALTIME MODE\nDuration: {duration_s}s\n"
                    text += f"L1: spikes={total_spikes_l1}, active={active_l1}/{N1}\n"
                    text += f"L2: spikes={total_spikes_l2}, active={active_l2}\n"
                    text += f"L1 rate: {total_spikes_l1/duration_s:.1f} spikes/s\n"
                    text += f"L2 rate: {total_spikes_l2/duration_s:.1f} spikes/s"
                else:
                    text = f"BATCH MODE\n"
                    text += f"L1: spikes={total_spikes_l1}, active={active_l1}/{N1}\n"
                    text += f"L2: spikes={total_spikes_l2}, active={active_l2}"
                
                axes[2 if fig_rows==3 else 1, 1].axis('off')
                axes[2 if fig_rows==3 else 1, 1].text(0.05, 0.9, text, fontsize=12, va='top')
            else:
                # No L2: put summary in bottom-right
                total_spikes_l1 = len(t1)
                active_l1 = len(np.unique(i1))
                
                if is_realtime:
                    duration_s = self.config['simulation_params'].get('max_duration_seconds', 60)
                    text = f"REALTIME MODE\nDuration: {duration_s}s\n"
                    text += f"L1: spikes={total_spikes_l1}, active={active_l1}/{N1}\n"
                    text += f"L1 rate: {total_spikes_l1/duration_s:.1f} spikes/s"
                else:
                    text = f"BATCH MODE\n"
                    text += f"L1: spikes={total_spikes_l1}, active={active_l1}/{N1}"
                
                axes[1, 1].axis('off')
                axes[1, 1].text(0.05, 0.9, text, fontsize=12, va='top')

            plt.tight_layout()
            plt.savefig(out / 'final_plot.png', dpi=150, bbox_inches='tight')

        logging.info("Results saved successfully.")
