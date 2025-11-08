import yaml
import numpy as np
import brian2 as b2
from pathlib import Path
import datetime
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from psinet.io.encoders import image_to_poisson_rates, create_input_layer
from psinet.network.hierarchy import SimpleHierarchy

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
        b2.prefs.codegen.target = 'numpy'

        sim_p = self.config['simulation_params']
        inp_p = self.config['input_params']
        net_p = self.config['network_params']
        mon_p = self.config['monitor_params']
        learn_p = self.config['learning_params']

        # 1) Input preparation: support mnist for this milestone
        dataset = inp_p.get('dataset', 'mnist')
        if dataset == 'mnist':
            try:
                import mnist
                images = mnist.train_images()  # shape: (60000, 28, 28)
            except Exception as e:
                raise RuntimeError(f"MNIST could not be loaded. Install 'mnist' package. Error: {e}")

            digits = inp_p.get('patterns_to_learn', [0, 1])
            indices = inp_p.get('image_indices', [0, 1])
            max_rate = inp_p.get('max_rate_hz', 150)

            # Take one image per digit according to provided indices
            def get_digit_image(d, idx):
                # pick the idx-th image with label d
                labels = mnist.train_labels()
                where = np.where(labels == d)[0]
                if len(where) == 0:
                    raise RuntimeError(f"No images found for digit {d}")
                sel = where[min(idx, len(where)-1)]
                return images[sel]

            img0 = get_digit_image(digits[0], indices[0])
            img1 = get_digit_image(digits[1], indices[1])
            rates0 = image_to_poisson_rates(img0, max_rate=max_rate*b2.Hz, invert=False)
            rates1 = image_to_poisson_rates(img1, max_rate=max_rate*b2.Hz, invert=False)
        else:
            raise NotImplementedError(f"Dataset '{dataset}' not supported in this milestone.")

        # dynamic input layer initialized to silence
        input_layer = create_input_layer(rates0*0)
        self.network_objects['input_layer'] = input_layer

        # 2) Build hierarchy with learning params
        hierarchy = SimpleHierarchy(
            input_layer,
            num_excitatory=net_p.get('num_excitatory_l1', 100),
            num_inhibitory=net_p.get('num_inhibitory_l1', 25),
            enable_learning=True
        )
        # Adjust learning parameters on the created synapse
        syn = hierarchy.input_to_l1_synapse.synapses
        syn.wmax = learn_p.get('w_max_inp_l1', 0.3)
        syn.Apre = learn_p.get('a_plus_inp_l1', 0.01)
        syn.Apost = learn_p.get('a_minus_inp_l1', -0.01)
        self.network_objects['hierarchy'] = hierarchy
        self.network_objects['rates0'] = rates0
        self.network_objects['rates1'] = rates1

        # 3) Monitors
        components = []
        if mon_p.get('record_l1_spikes', True):
            self.monitors['l1_spikes'] = b2.SpikeMonitor(hierarchy.layer1.excitatory_neurons.group)
            components.append(self.monitors['l1_spikes'])
        if mon_p.get('record_input_spikes', False):
            self.monitors['input_spikes'] = b2.SpikeMonitor(input_layer)
            components.append(self.monitors['input_spikes'])

        # weight subset monitor
        subset_size = int(mon_p.get('record_weight_subset_size', 100))
        total_candidates = 28*28 * net_p.get('num_excitatory_l1', 100)
        subset = np.random.choice(np.arange(total_candidates), min(subset_size, total_candidates), replace=False)
        self.monitors['weights'] = b2.StateMonitor(hierarchy.input_to_l1_synapse.synapses, 'w', record=subset)
        components.append(self.monitors['weights'])

        # 4) Assemble Network
        net = hierarchy.build_network(*components)
        self.brian2_network = net
        logging.info("Network build complete.")

    # ------------------------------- RUN -----------------------------------
    def run(self):
        if self.brian2_network is None:
            logging.error("Network has not been built. Call .build() first.")
            return

        sim_p = self.config['simulation_params']
        show = sim_p['duration_per_pattern_ms'] * b2.ms
        rest = sim_p['silence_period_ms'] * b2.ms
        cycles = int(sim_p['cycles'])

        hierarchy = self.network_objects['hierarchy']
        rates0 = self.network_objects['rates0']
        rates1 = self.network_objects['rates1']

        self.zero_windows = []
        self.one_windows = []

        logging.info(f"Starting dynamic loop: {cycles}x [ '0' ({show}), rest ({rest}), '1' ({show}), rest ({rest}) ]")
        current_t = 0 * b2.second
        for _ in range(cycles):
            hierarchy.input_layer.rates = rates0
            self.brian2_network.run(show, report='text')
            self.zero_windows.append((current_t, current_t + show))
            current_t += show

            hierarchy.input_layer.rates = 0 * b2.Hz
            self.brian2_network.run(rest)
            current_t += rest

            hierarchy.input_layer.rates = rates1
            self.brian2_network.run(show, report='text')
            self.one_windows.append((current_t, current_t + show))
            current_t += show

            hierarchy.input_layer.rates = 0 * b2.Hz
            self.brian2_network.run(rest)
            current_t += rest

        logging.info("Simulation finished.")

    # ------------------------------- SAVE ----------------------------------
    def save_results(self):
        logging.info("Saving results...")
        out = self.output_dir
        out.mkdir(parents=True, exist_ok=True)

        # raw data
        if self.config['output_params'].get('save_raw_data', True):
            npz_payload = {}
            if 'l1_spikes' in self.monitors:
                npz_payload['l1_t_ms'] = (self.monitors['l1_spikes'].t / b2.ms).astype(float)
                npz_payload['l1_i'] = np.array(self.monitors['l1_spikes'].i)
            if 'input_spikes' in self.monitors:
                npz_payload['inp_t_ms'] = (self.monitors['input_spikes'].t / b2.ms).astype(float)
                npz_payload['inp_i'] = np.array(self.monitors['input_spikes'].i)
            if 'weights' in self.monitors:
                npz_payload['w_t_ms'] = (self.monitors['weights'].t / b2.ms).astype(float)
                npz_payload['w'] = self.monitors['weights'].w.T
            # windows
            z0 = np.array([[w[0]/b2.ms, w[1]/b2.ms] for w in self.zero_windows])
            o1 = np.array([[w[0]/b2.ms, w[1]/b2.ms] for w in self.one_windows])
            npz_payload['zero_windows_ms'] = z0
            npz_payload['one_windows_ms'] = o1
            np.savez_compressed(out / 'raw_data.npz', **npz_payload)

        # plots
        if self.config['output_params'].get('save_plots', True) and 'l1_spikes' in self.monitors:
            N = self.network_objects['hierarchy'].layer1.excitatory_neurons.group.N
            spike_t = self.monitors['l1_spikes'].t
            spike_i = self.monitors['l1_spikes'].i

            counts_zero = np.zeros(N, dtype=int)
            counts_one = np.zeros(N, dtype=int)
            for t0, t1 in self.zero_windows:
                mask = (spike_t >= t0) & (spike_t < t1)
                if np.any(mask):
                    idx, cnt = np.unique(spike_i[mask], return_counts=True)
                    counts_zero[idx] += cnt
            for t0, t1 in self.one_windows:
                mask = (spike_t >= t0) & (spike_t < t1)
                if np.any(mask):
                    idx, cnt = np.unique(spike_i[mask], return_counts=True)
                    counts_one[idx] += cnt

            preference = counts_zero - counts_one

            fig, axes = plt.subplots(2, 2, figsize=(14, 12))
            axes[0, 0].plot(spike_t / b2.ms, spike_i, '.k', markersize=1)
            axes[0, 0].set_title('L1 Spikes')
            axes[0, 0].set_xlabel('Time (ms)')
            axes[0, 0].set_ylabel('Neuron index')
            axes[0, 0].grid(True, alpha=0.3)

            colors = np.where(preference >= 0, 'tab:red', 'tab:blue')
            axes[0, 1].bar(np.arange(N), preference, color=colors, width=0.9)
            axes[0, 1].set_title('Preference (pos: 0, neg: 1)')
            axes[0, 1].axhline(0, color='k', linewidth=0.8)

            # weight evolution if present
            if 'weights' in self.monitors:
                axes[1, 0].plot(self.monitors['weights'].t / b2.ms, self.monitors['weights'].w.T, alpha=0.7, linewidth=0.8)
                axes[1, 0].set_title('Sampled weight dynamics')
                axes[1, 0].set_xlabel('Time (ms)')
                axes[1, 0].set_ylabel('w')
                axes[1, 0].grid(True, alpha=0.3)

            # simple summary text
            total_spikes = len(spike_t)
            active_neurons = len(np.unique(spike_i))
            text = f"Total spikes: {total_spikes}\nActive L1 neurons: {active_neurons}/{N}"
            axes[1, 1].axis('off')
            axes[1, 1].text(0.05, 0.9, text, fontsize=12, va='top')

            plt.tight_layout()
            plt.savefig(out / 'final_plot.png', dpi=150, bbox_inches='tight')

        logging.info("Results saved successfully.")
