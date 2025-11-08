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
from psinet.io.loaders import load_mnist
from psinet.network.hierarchy import Hierarchy

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

        sim_p = self.config['simulation_params']
        inp_p = self.config['input_params']
        net_p = self.config['network_params']
        mon_p = self.config['monitor_params']
        learn_p = self.config.get('learning_params', {})

        # Device selection
        device = sim_p.get('brian2_device', 'runtime')
        if device == 'cpp_standalone':
            b2.set_device('cpp_standalone', directory=str(self.output_dir / 'brian2_build'))
        b2.prefs.codegen.target = 'numpy'

        # 1) Input preparation: support mnist with fallback to synthetic
        dataset = inp_p.get('dataset', 'mnist')

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
        rates_map = self.network_objects['rates_map']
        digits_list = self.network_objects['digits_list']

        # Windows for per-digit analysis
        self.windows_per_digit = {d: [] for d in digits_list}

        # Sequence construction: if present_all_digits flag, iterate all digits per cycle
        present_all = bool(sim_p.get('present_all_digits', False))
        if present_all:
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

        logging.info("Simulation finished.")

    # ------------------------------- SAVE ----------------------------------
    def save_results(self):
        logging.info("Saving results...")
        out = self.output_dir
        out.mkdir(parents=True, exist_ok=True)

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
            # windows per digit
            win = {int(k): np.array([[w[0]/b2.ms, w[1]/b2.ms] for w in v]) for k, v in self.windows_per_digit.items()}
            np.savez_compressed(out / 'raw_data.npz', **npz_payload, windows_per_digit=win)

        # plots
        if self.config['output_params'].get('save_plots', True) and 'spikes_L1' in self.monitors:
            l1_name = self.network_objects['hierarchy'].layers_in_order[0]
            N1 = self.network_objects['hierarchy'].layers_by_name[l1_name].excitatory_neurons.group.N
            t1 = self.monitors['spikes_L1'].t
            i1 = self.monitors['spikes_L1'].i

            # Per-digit counts for L1 (if classic two-digit, still works)
            counts_per_digit = {d: np.zeros(N1, dtype=int) for d in self.windows_per_digit.keys()}
            for d, windows in self.windows_per_digit.items():
                for t0, t1w in windows:
                    mask = (t1 >= t0) & (t1 < t1w)
                    if np.any(mask):
                        idx, cnt = np.unique(i1[mask], return_counts=True)
                        counts_per_digit[d][idx] += cnt

            # Prepare figure with optional L2
            has_l2 = 'spikes_L2' in self.monitors
            fig_rows = 3 if has_l2 else 2
            fig, axes = plt.subplots(fig_rows, 2, figsize=(14, 6 + 3*fig_rows))

            # L1 raster
            axes[0, 0].plot(t1 / b2.ms, i1, '.k', markersize=1)
            axes[0, 0].set_title('L1 Spikes')
            axes[0, 0].set_xlabel('Time (ms)')
            axes[0, 0].set_ylabel('Neuron index')
            axes[0, 0].grid(True, alpha=0.3)

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
                    axes[2, 0].grid(True, alpha=0.3)

                # Summary text
                total_spikes_l1 = len(t1)
                active_l1 = len(np.unique(i1))
                total_spikes_l2 = len(t2)
                active_l2 = len(np.unique(i2))
                text = f"L1: spikes={total_spikes_l1}, active={active_l1}/{N1}\nL2: spikes={total_spikes_l2}, active={active_l2}"
                axes[2 if fig_rows==3 else 1, 1].axis('off')
                axes[2 if fig_rows==3 else 1, 1].text(0.05, 0.9, text, fontsize=12, va='top')
            else:
                # No L2: put summary in bottom-right
                total_spikes_l1 = len(t1)
                active_l1 = len(np.unique(i1))
                text = f"L1: spikes={total_spikes_l1}, active={active_l1}/{N1}"
                axes[1, 1].axis('off')
                axes[1, 1].text(0.05, 0.9, text, fontsize=12, va='top')

            plt.tight_layout()
            plt.savefig(out / 'final_plot.png', dpi=150, bbox_inches='tight')

        logging.info("Results saved successfully.")
